# 🎭 Hide-Face-From-Video

Want to upload videos on social media but keep your face private?
Skip boring traditional blurs—this tool lets you hide your face using **fun, customizable masks** (like Squid Game masks 🎬) so you can **protect your identity** while keeping your videos **engaging and cool**.

Run on Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/Hide-Face-From-Video/blob/main/Hide_Face.ipynb)

Play on HuggingFace:
[![HuggingFace Space Demo](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/NeuralFalcon/Hide-face-in-videos-using-Squid-Game-masks)


---

## 📸 Demo

| Input Video                                                                                | Mask Applied                                                                                  | App Screenshot                                                                          |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| ![upload](https://github.com/user-attachments/assets/10218406-35e4-4223-8033-5a6efc81f304) | ![new\_mask](https://github.com/user-attachments/assets/84a8a9c5-114d-4411-bc2d-5f9752423976) | ![app](https://github.com/user-attachments/assets/b7878f76-e175-4b6e-b9de-ea3323f32c9c) |

🎥 Demo Clip:

https://github.com/user-attachments/assets/0db41dc1-020c-4395-b52c-e23d7cb2f7e4


---

## ⚡ Installation

```bash
git clone https://github.com/NeuralFalconYT/Hide-Face-From-Video.git
cd Hide-Face-From-Video

# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate  # On Windows
# source myenv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run Gradio Web App

Drag & drop your video, apply a mask, and save the output.

```bash
python app.py
```

### 2. For Webcam or OBS

Use your mask in real-time with webcam or OBS streaming.

```bash
python obs.py
```

---

## 🛠️ Creating Your Own Mask

1. Create a **transparent PNG mask** (e.g., `mask.png`).
2. Upload it to [MakeSense.ai](https://www.makesense.ai/).
3. Mark **36 facial landmarks**.
4. Export the CSV file.
5. Remove the last **two columns** from the CSV.
6. Move `mask.png` and the CSV into the **`assets`** folder.
7. Update `video_process.py` with the new file paths.

---

## 🧠 Mediapipe Face 3D Landmarks

Reference for mask positioning:

![3d landmark](https://github.com/user-attachments/assets/bed9f9d6-6269-457e-a2a4-923b256a179d)

Custom landmark naming:

<img width="620" height="800" alt="points" src="https://github.com/user-attachments/assets/0e5710d2-de7d-4d91-95d7-7aa77f598544" />

---

## 🌟 Features

* 🎭 Replace boring blurs with stylish masks
* 🎥 Works on uploaded videos or live webcam feed
* ⚡ Real-time processing via Gradio / OBS
* 🛠️ Fully customizable masks (create your own)
* 🔒 Privacy-first, identity hidden but content engaging

---

## 📂 Project Structure

```
Hide-Face-From-Video/
│── assets/             # Store your masks & CSV landmarks
│── app.py              # Gradio webapp
│── obs.py              # OBS/Webcam live mask
│── video_process.py    # Core video face masking logic
│── requirements.txt    # Dependencies
│── Hide_Face.ipynb     # Google Colab notebook
```


