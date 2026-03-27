# Bouldering Highlight POC based on Multimodal AI (audio & video analysis)

This Proof of Concept (POC) analyzes bouldering videos by combining Computer Vision (pose estimation) and Audio Processing (sound frequency analysis) to generate highlight videos. It utilizes a "Late Fusion" architecture to synchronize and evaluate both physical movements and auditory events (e.g., jumps, crowd cheers, falls).

## Architecture 
1. **Vision Branch**: `YOLOv11m-Pose` extracts 17 skeletal keypoints, flattened and tracked over a 30-frame rolling window.
2. **Audio Branch**: `Librosa` extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio track, synchronized at 30 FPS.
3. **Fusion Model**: Two parallel LSTMs process the visual and audio sequences. Their final states are concatenated (Late Fusion) into a dense network to predict the final climbing action.

## Repository Structure
- `data/`: Input, output and dataset are stored here.
- `models/`: Models (trained and downloaded) are stored here.
- `src/`: Core engine modules (Vision Detector, Audio Engine, Processor, Fusion Net).
- `train/`: Scripts for building the synchronized multimodal dataset and training the network.
- `main_multimodal.py`: Main inference script that generates the final annotated video with sound.

## Setup & Execution

1. **Install Dependencies**:
   pip install -r requirements.txt

# Bouldering Highlights POC based on Multimodal AI (Audio & Video Analysis)

This Proof of Concept (POC) analyzes bouldering videos by combining Computer Vision (pose estimation) and Audio Processing (sound frequency analysis) to generate highlight videos. It uses a **Late Fusion** architecture to synchronize and evaluate both physical movements and auditory events (e.g., jumps, crowd cheers, falls).

## Architecture

* **Vision Branch (Eyes)**: `YOLOv11m-Pose` extracts 17 skeletal keypoints, flattened and tracked over a 30-frame rolling window.
* **Audio Branch (Ears)**: `Librosa` extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio track at 30 FPS.
* **Fusion Model (Brain)**: Parallel LSTMs process visual and audio sequences; final states are concatenated (Late Fusion) into a dense network for action classification.
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
Create the required directory structure:

Bash
mkdir -p data/input data/output data/dataset models data/audio
Place your source video in:

Plaintext
data/input/janja_video.mp4
Note: the selected video contains both a clear view of the climber and a clean audio track.

### 3. Extract the Target Sequence & Audio
Extract the specific segment to analyze and generate the isolated audio track:

Bash
python utils/extract_sequence.py
python utils/extract_audio.py
Outputs:

Plaintext
data/input/janja_sequence.mp4
data/audio/janja_audio.wav

### 4. Build the Multimodal Dataset
Generate synchronized labeled training data (YOLO keypoints + MFCCs):

Bash
python train/build_multimodal_dataset.py
Outputs:

Plaintext
data/dataset/X_vision.npy
data/dataset/X_audio.npy
data/dataset/y_raw.npy

### 5. Train the Fusion Model
Train the parallel LSTM architecture:

Bash
python train/train_multimodal.py
Output:

Plaintext
models/multimodal_fusion.pth

### 6. Run Inference (Audio-Visual Analysis)
Process both streams synchronously, overlay predictions, and remux the audio back into the video:

Bash
python main_multimodal.py
Output:

Plaintext
data/output/janja_multimodal_analysis.mp4
Detailed Project Structure
Plaintext
multimodal_poc/
│
├── data/
│   ├── input/
│   │   ├── janja_video.mp4
│   │   └── janja_sequence.mp4
│   ├── audio/
│   │   └── janja_audio.wav
│   ├── output/
│   │   └── janja_multimodal_analysis.mp4
│   └── dataset/
│       ├── X_vision.npy
│       ├── X_audio.npy
│       └── y_raw.npy
│
├── models/
│   └── multimodal_fusion.pth
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



