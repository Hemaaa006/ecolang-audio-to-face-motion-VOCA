import os
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

# ---------------- constants ----------------
FPS = 30
SR = 16000

MAX_AUDIO_SEC = 30.0  # UI limit
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
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, out_dim))

    def forward(self, x):
        h = self.w2v(x).last_hidden_state                  # (B, T_w2v, feat)
        h = self.proj(h)                                   # (B, T_w2v, hidden)
        h = h.transpose(1, 2)                               # (B, hidden, T_w2v)
        h = torch.nn.functional.interpolate(
            h, size=WIN_FRAMES, mode="linear", align_corners=False
        )                                                   # (B, hidden, WIN_FRAMES)
        h = h.transpose(1, 2)                               # (B, WIN_FRAMES, hidden)
        h, _ = self.lstm(h)                                 # (B, WIN_FRAMES, hidden*2)
        return self.head(h)                                 # (B, WIN_FRAMES, OUT_DIM)

def load_checkpoint(path="best_face.pt"):
    model = Audio2Face().to(DEVICE).eval()
    ckpt = torch.load(path, map_location="cpu")
    # expects: {"model": state_dict}
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
    """
    N = len(wav_np)
    if N == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, EXPR_DIM), np.float32)

    # pad to full windows
    num_windows = int(np.ceil(N / WIN_SAMPLES))
    padded = np.pad(wav_np, (0, num_windows * WIN_SAMPLES - N), mode="constant")

    jaw_all = []
    expr_all = []

    for w in range(num_windows):
        seg = padded[w * WIN_SAMPLES : (w + 1) * WIN_SAMPLES]
        x = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)   # (1, WIN_SAMPLES)
        yhat = MODEL(x)[0].detach().cpu().numpy()           # (WIN_FRAMES, 13)
        jaw_all.append(yhat[:, :3])
        expr_all.append(yhat[:, 3:])

    jaw = np.concatenate(jaw_all, axis=0)
    expr = np.concatenate(expr_all, axis=0)

    # trim frames to actual audio length
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
    transl = np.zeros((T, 3), dtype=np.float32)

    return {
        "poses": poses,
        "expressions": expr.astype(np.float32),
        "betas": betas,
        "transl": transl,
        "fps": np.int32(fps),
    }

# ---------------- gradio fn (NO graph, NO smplx upload) ----------------
def run(audio_file):
    if audio_file is None:
        return None, "Please upload an audio file."

    wav = load_audio_mono(audio_file)

    # hard cap to MAX_AUDIO_SEC
    max_samples = int(MAX_AUDIO_SEC * SR)
    if len(wav) > max_samples:
        wav = wav[:max_samples]

    jaw, expr = predict_full_sequence(wav)
    motion = build_motion_npz(jaw, expr, fps=FPS)

    tmpdir = tempfile.mkdtemp()
    npz_path = os.path.join(tmpdir, "motion_pred.npz")
    np.savez(npz_path, **motion)

    info = f"Generated {jaw.shape[0]} frames @ {FPS} fps (~{jaw.shape[0] / FPS:.2f}s)."
    return npz_path, info

# ---------------- UI (Blocks) ----------------
with gr.Blocks(title="ECOLANG Audio → Face Motion") as demo:
    gr.Markdown(
        """
# ECOLANG Audio → Face Motion
Upload speech audio . Outputs an **SMPL-X-style** `.npz`.
        """.strip()
    )

    audio_in = gr.Audio(type="filepath", label="Upload speech audio")
    btn = gr.Button("Generate")

    out_npz = gr.File(label="Download predicted motion (motion_pred.npz)")
    out_info = gr.Textbox(label="Info")

    btn.click(
        fn=run,
        inputs=[audio_in],
        outputs=[out_npz, out_info],
        concurrency_limit=1,
        api_name=False,   # avoids API schema edge cases
    )

demo.queue(max_size=1)

if __name__ == "__main__":
    # Spaces-safe: bind to 0.0.0.0:7860
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
