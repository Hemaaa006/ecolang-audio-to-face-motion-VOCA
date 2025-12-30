
import os
import zipfile
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import matplotlib.pyplot as plt

# ---------------- constants ----------------
FPS = 30
SR = 16000
WIN_SEC = 4.0
WIN_FRAMES = int(FPS * WIN_SEC)
WIN_SAMPLES = int(SR * WIN_SEC)

JAW_INDEX = 22
EXPR_DIM = 10
OUT_DIM = 3 + EXPR_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- utils ----------------
def _resample_np(wav, orig_sr, target_sr):
    if orig_sr == target_sr:
        return wav
    x_old = np.linspace(0, 1, num=len(wav), endpoint=False)
    new_len = int(len(wav) * target_sr / orig_sr)
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)

def load_audio_mono(audio_path):
    wav, sr = sf.read(audio_path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    wav = _resample_np(wav, sr, SR)
    return wav

# ---------------- model ----------------
class Audio2Face(nn.Module):
    def __init__(self, out_dim=OUT_DIM, w2v_name="facebook/wav2vec2-base", hidden=256, layers=2):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name)
        feat = self.w2v.config.hidden_size
        self.proj = nn.Linear(feat, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2, out_dim))

    def forward(self, x):
        h = self.w2v(x).last_hidden_state
        h = self.proj(h)
        h = h.transpose(1,2)
        h = torch.nn.functional.interpolate(h, size=WIN_FRAMES, mode="linear", align_corners=False)
        h = h.transpose(1,2)
        h, _ = self.lstm(h)
        return self.head(h)

def load_checkpoint(path="best_face.pt"):
    model = Audio2Face().to(DEVICE).eval()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    return model

MODEL = load_checkpoint("best_face.pt")

# ---------------- inference: long audio -> stitched motion ----------------
@torch.no_grad()
def predict_full_sequence(wav_np):
    """
    wav_np: float32 mono at 16k, any length
    returns:
      jaw_seq: (T,3)
      expr_seq: (T,10)
      fps: 30
    """
    N = len(wav_np)
    if N == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,EXPR_DIM), np.float32)

    # pad to full windows
    num_windows = int(np.ceil(N / WIN_SAMPLES))
    padded = np.pad(wav_np, (0, num_windows*WIN_SAMPLES - N), mode="constant")

    jaw_all = []
    expr_all = []

    for w in range(num_windows):
        seg = padded[w*WIN_SAMPLES:(w+1)*WIN_SAMPLES]
        x = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)  # (1, WIN_SAMPLES)
        yhat = MODEL(x)[0].detach().cpu().numpy()          # (WIN_FRAMES,13)
        jaw_all.append(yhat[:, :3])
        expr_all.append(yhat[:, 3:])

    jaw = np.concatenate(jaw_all, axis=0)
    expr = np.concatenate(expr_all, axis=0)

    # trim frames to actual audio length
    # expected frames = floor(seconds*FPS)
    seconds = N / SR
    T = int(np.floor(seconds * FPS))
    if T <= 0:
        T = 1
    jaw = jaw[:T]
    expr = expr[:T]
    return jaw.astype(np.float32), expr.astype(np.float32)

def build_motion_npz(jaw, expr, fps=30):
    """
    create a SMPL-X-like npz:
      poses: (T,55,3) only jaw filled
      expressions: (T,10)
      betas: (10,) zeros
      transl: (T,3) zeros
      fps: int
    """
    T = jaw.shape[0]
    poses = np.zeros((T, 55, 3), dtype=np.float32)
    poses[:, JAW_INDEX, :] = jaw

    betas = np.zeros((10,), dtype=np.float32)
    transl = np.zeros((T,3), dtype=np.float32)

    return {
        "poses": poses,
        "expressions": expr.astype(np.float32),
        "betas": betas,
        "transl": transl,
        "fps": np.int32(fps)
    }

def plot_preview(jaw, expr, out_png):
    # simple visualization of motion over time
    t = np.arange(jaw.shape[0]) / FPS
    plt.figure()
    plt.plot(t, jaw[:,0])
    plt.plot(t, jaw[:,1])
    plt.plot(t, jaw[:,2])
    plt.xlabel("time (s)")
    plt.ylabel("jaw axis-angle")
    plt.title("Predicted jaw motion")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------- SMPL-X zip check (optional) ----------------
def verify_smplx_zip(zip_path):
    """
    Users upload their own SMPL-X zip. We just verify it contains expected structure.
    Expect: smplx/SMPLX_NEUTRAL.npz (or similar)
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
    has_smplx_folder = any(n.startswith("smplx/") for n in names)
    has_neutral = any("SMPLX_NEUTRAL" in n and n.startswith("smplx/") for n in names)
    return has_smplx_folder and has_neutral, names[:50]

# ---------------- gradio fn ----------------
def run(audio_file, smplx_zip):
    if audio_file is None:
        return None, None, None, "Upload an audio file."

    # optional zip validation (for licensing-friendly demo)
    zip_msg = ""
    if smplx_zip is not None:
        ok, sample_names = verify_smplx_zip(smplx_zip)
        if not ok:
            return None, None, None, "SMPL-X zip doesn't look right. It must contain smplx/SMPLX_NEUTRAL.npz (and friends)."
        zip_msg = "SMPL-X zip looks valid (not used for rendering in this demo)."

    wav = load_audio_mono(audio_file)
    jaw, expr = predict_full_sequence(wav)

    motion = build_motion_npz(jaw, expr, fps=FPS)

    tmpdir = tempfile.mkdtemp()
    npz_path = os.path.join(tmpdir, "motion_pred.npz")
    np.savez(npz_path, **motion)

    png_path = os.path.join(tmpdir, "jaw_plot.png")
    plot_preview(jaw, expr, png_path)

    info = f"Generated {jaw.shape[0]} frames @ {FPS} fps (~{jaw.shape[0]/FPS:.2f}s). {zip_msg}"
    return npz_path, png_path, info, info

demo = gr.Interface(
    fn=run,
    inputs=gr.Audio(type="filepath", label="Upload speech audio (≤15s)"),
    outputs=[
        gr.File(label="Download predicted motion (motion_pred.npz)"),
        gr.Image(label="motion preview"),
        gr.Textbox(label="Info"),
    ],
    title="ECOLANG Audio → Face Motion ",
    description="Predicts a full motion sequence and outputs SMPL-X-style npz. Rendering requires SMPL-X models and is done locally."
)

if __name__ == "__main__":
    demo.launch()
