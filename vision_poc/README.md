# Bloudering Highlights POC based on Computer Vision

This Proof of Concept (POC) analyzes bouldering videos using **Computer Vision** (pose estimation) to automatically generate highlight videos. It utilizes a **temporal sequence model** to detect and evaluate physical movements (e.g., climbs, dynos, rests, falls) by processing skeletal keypoint sequences in real-time.

## Architecture
- **Detection**: `YOLOv11m-Pose` for extracting 17 skeletal keypoints.
- **Processing**: Flattens the coordinates and maintains a rolling window of 30 frames (1 second of context).
- **Classification**: PyTorch `LSTM` network processing the sequence to predict the action.
- **Classes**: Rest, Climb, Dyno (Jump), Fall/Top.

## Repository Structure
- `data/`: Input, output and dataset are stored here.
- `models/`: Models (trained and downloaded) are stored here.
- `src/`: Core engine modules (Detector, Processor, Classifier).
- `train/`: Scripts for building the custom dataset and training the LSTM.
- `utils/`: Script to extract a sequence 
- `main.py`: Main inference script with visual overlay.
- `generate_highlights.py`: Script for highlight generation.

## Setup & Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Data
Create the required directory structure:
```bash
mkdir -p data/input data/output data/dataset models
```
Place the source video in:
```
data/input/janja_video.mp4
```
> **Note:** The file must be named exactly `janja_video.mp4`.

### 3. Extract the Target Sequence
Extract a specific segment instead of processing the entire video.
(Current selection: **5:18 → 6:24**)
```bash
python utils/extract_sequence.py
```

**Output:**
```
data/input/janja_sequence_318_385.mp4
```
---

### 4. Build the Dataset
Generate labeled training data from the extracted sequence:
```bash
python train/build_dataset.py
```
**Outputs:**
```
data/dataset/X_raw.npy
data/dataset/y_raw.npy
```
---

### 5. Train the Model
Train the LSTM sequence classifier:
```bash
python train/train_model.py
```
**Output:**
```
models/action_lstm.pth
```
---

### 6. Run Inference (Visualization)

Execute the full pipeline with real-time predictions:
```bash
python main.py
```
**Output:**
```
data/output/janja_sequence_analysis.mp4
```
---

### 7. Generate Highlights

Automatically generate a condensed highlight video of key movements:

```bash
python generate_highlights.py
```

**Output:**

```
data/output/janja_highlights_sequence_318_385.mp4
```

---

## Detailed Project Structure

```
vision_poc/
│
├── data/
│   ├── input/
│   │   ├── janja_video.mp4
│   │   └── janja_sequence_318_385.mp4
│   │
│   ├── output/
│   │   ├── janja_sequence_analysis.mp4
│   │   └── janja_highlights_sequence_318_385.mp4
│   │
│   └── dataset/
│       ├── X_raw.npy
│       └── y_raw.npy
│
├── models/
│   └── action_lstm.pth
│   └── yolo11m-pose.pt
│
├── src/
│   ├── detector.py
│   ├── processor.py
│   └── classifier.py
│
├── train/
│   ├── build_dataset.py
│   └── train_model.py
│
├── utils/
│   └── extract_sequence.py
│
├── main.py
├── generate_highlights.py
└── requirements.txt
```

---
## Notes

* Ensure all paths are correct relative to the project root
* GPU acceleration is recommended for training and inference
