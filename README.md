# ECOLANG Audio-Driven Face Motion Generation

- This repository contains the code for an **audio-driven 3D facial motion generation system** trained on the ECOLANG dataset.  
- The model predicts **temporal SMPL-X facial parameters directly from speech audio.

---

## 📁 Repository Structure
notebooks/
- 01_train_audio2face.ipynb # model training
- 02_inference_and_render.ipynb # motion prediction + SMPL-X rendering

hf_space/
- app.py # Hugging Face Space app
- requirements.txt

---

## 🧠 Model Overview
- Audio encoder: Wav2Vec2
- Temporal model: Bi-directional LSTM
- Output: SMPL-X jaw pose (axis-angle) + expression coefficients
- Frame rate: 30 FPS

---

## 🔊 Input / Output
**Input**
- Speech audio 

**Output**
- SMPL-X-compatible motion sequence (`.npz`)
- Optional rendered MP4 (local only)

---

## 🚀 Live Demo
Hugging Face Space:  
👉 https://huggingface.co/spaces/Hemaa006/ecolang-audio-to-face-motion-voca

---

## ⚠️ SMPL-X License
This repository **does not include SMPL-X model files**.  
Users must download SMPL-X separately from the official website and agree to the license.

---
