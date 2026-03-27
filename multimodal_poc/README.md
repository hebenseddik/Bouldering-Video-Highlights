# Bouldering Highlights POC based on Multimodal AI Model (Audio & Video Analysis)

This Proof of Concept (POC) analyzes bouldering videos by combining Computer Vision (pose estimation) and Audio Processing (sound frequency analysis) to generate highlight videos. It uses a **Late Fusion** architecture to synchronize and evaluate both physical movements and auditory events (e.g., jumps, crowd cheers, falls).

## Architecture

* **Vision Banch**: `YOLOv11m-Pose` extracts 17 skeletal keypoints, flattened and tracked over a 30-frame rolling window.
* **Audio Branch**: `Librosa` extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio track at 30 FPS.
* **Fusion Model**: Parallel LSTMs process visual and audio sequences; final states are concatenated (Late Fusion) into a dense network for action classification.
* **Classes**: Rest, Climb, Dyno (Jump), Fall/Top.

## Repository Structure

* `data/`: Input videos, audio tracks, outputs, and datasets.
* `models/`: Trained fusion models.
* `src/`: Core engine modules (Detector, Audio Engine, Processor, Fusion model).
* `train/`: Scripts for building the synchronized multimodal dataset and training the network.
* `utils/`: Utilities for extracting video sequences and audio tracks.
* `main_multimodal.py`: Main inference script generating annotated video with sound.
* `generate_multimodal_highlights.py`: Main inference script generating annotated video with sound.

---

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Prepare the Data
Create the required directory structure
```bash
mkdir -p data/input data/output data/dataset models data/audio
```
Place the source video in:
data/input/janja_video.mp4

Note: The selected video must contain both a clear view of the climber and a clean audio track for accurate analysis.

### 3. Extract the Target Sequence & Audio
Run extraction scripts
Extract the specific segment to analyze and generate the isolated audio track:

```bash
python utils/extract_sequence.py
python utils/extract_audio.py
```

Outputs:
data/input/janja_sequence_318_385.mp4
data/audio/janja_audio.wav

### 4. Build the Multimodal Dataset
Generate training data
Generate synchronized labeled training data by combining YOLO keypoints and MFCCs:

```bash
python train/build_multimodal_dataset.py
```
Outputs:
data/dataset/X_vision.npy
data/dataset/X_audio.npy
data/dataset/y_raw.npy

### 5. Train the Fusion Model
Train the LSTM architecture
```bash
python train/train_multimodal.py
```
Output:
models/multimodal_net.pth

### 6. Run Inference (Audio-Visual Analysis)
Process and generate output
Process both streams synchronously, overlay predictions, and remux the audio back into the final video:

```bash
python main_multimodal.py
python generate_multimodal_highlights.py
```
Output:
data/output/janja_multimodal_analysis.mp4
data/output/janja_multimodal_highlights.mp4

## Detailed Project Structure

```
multimodal_poc/
│
├── data/
│   ├── input/
│   │   ├── janja_video.mp4
│   │   └── janja_sequence_318_385.mp4
│   │   └── janja_audio_track.wav
│   ├── output/
│   │   └── janja_multimodal_analysis.mp4
│   └── dataset/
│       ├── X_vision.npy
│       ├── X_audio.npy
│       └── y_raw.npy
│
├── models/
│   └── multimodal_net.pth
│   └── yolo11m-pose.pt
│
├── src/
│   ├── detector.py
│   ├── audio_engine.py
│   ├── processor.py
│   └── fusion_model.py
│
├── train/
│   ├── build_multimodal_dataset.py
│   └── train_multimodal.py
│
├── utils/
│   ├── extract_sequence.py
│   └── extract_audio.py
│
├── main_multimodal.py
└── requirements.txt
```

---

## Notes

* Ensure all paths are correct relative to the project root
* GPU acceleration is recommended for training and inference