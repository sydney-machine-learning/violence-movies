## Deep learning framework for video-based violence and abuse detection in movies

### Requirements

Use Python 3.11 and install a matching `torch` and `torchvision` pair for your [CPU or CUDA environment](https://pytorch.org/get-started/locally/). Then install the notebook dependencies:

```bash
python -m pip install numpy scikit-learn h5py matplotlib pandas opencv-python jupyterlab
```

For the YOLO notebooks, run `python -m pip install ultralytics roboflow` and put FFmpeg on `PATH` for clip export. Start JupyterLab from the project root with `python -m jupyterlab`.

### Folder Structure

```text
violence-movies/
├── notebooks/                   # preprocessing, training, and inference notebooks
├── src/                         # model, training, and benchmark modules
├── deprecated/src/              # legacy Keras notebooks
├── data/                        # local datasets, prepared chunks, and video inputs
├── c3d_weight/                  # pretrained Sports-1M weights (local)
├── yolo_weight/                 # YOLO checkpoints (local)
├── weights/                     # trained PyTorch classifiers (local)
└── results/                     # generated reports (local)
```

## 1. Project Objective and Content Overview

This project uses deep learning and computer vision techniques to detect and remove violent and inappropriate content from movie videos. Unlike current common detection methods that only identify whether the violence is violent or not, the framework also categorizes the identified violence in terms of low and high levels in movies. It combines YOLO-v8 and 3D-CNN-LSTM models to detect and categorize violent content. YOLOv8 selects candidate segments. A video classifier (3D-CNN-LSTM) assigns each segment to non-violence, low (unarmed) violence, or high (armed) violence.

The study compares 5 different classifiers, namely 3D-CNN-LSTM, C3D, ConvLSTM, MViTv2-S, and MoViNet-A0. On AIRTLab, the 3D-CNN-LSTM achieved **78.06% mean chunk accuracy** and **79.43% mean source-video accuracy** across ten grouped test folds. Film and TV examples illustrate the workflow in case studies.

<img src="images/Framework6.png" width="720"/>

## 2. Dataset and Preprocessing
### a) AIRTLab Dataset

From the repository root, download [AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos):

```bash
git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git data/raw/violence-detection-dataset
```

Run the [AIRTLab preprocessing notebook](notebooks/preprocess_AIRTLab.ipynb) to assign the corrected action labels and build 16-frame chunks at 112 × 112 pixels. The notebook samples every third source frame, uses 50% overlap, and saves chunks, labels, and source-video IDs under `data/processed/violence-detection-dataset/`.

### b) Roboflow Violence Detection Dataset
Export version 3 of the [Roboflow Violence Detection Dataset](https://universe.roboflow.com/violence-detection-fbe46/violence-detection-nbx24) in YOLOv8 format. Extract the dataset so that `data.yaml` is at `data/roboflow/Violence-Detection-3/data.yaml`.

Dataset Split:
- Train Set 88% 13873 Images
- Valid Set 6% 999 Images
- Test Set 6% 976 Images
Preprocessing:
- Auto-Orient: Applied
- Resize: Stretch to 640x640
- Augmentations
- Outputs per training example: 3
- Flip: Horizontal

### c) Streaming videos
Use streaming video clips from movies as the analysis source, including John Wick, The Boys, The Punisher, Game of Thrones, Ip Man, Mission: Impossible – Fallout. Please refer to data\streaming videos

Link: https://drive.google.com/drive/folders/1ISmc3xwPOwAkuYoVuEcPwje267oRG9k2?usp=sharing

## 3. Training

### Video classifier training

1. Run [preprocess_AIRTLab.ipynb](notebooks/preprocess_AIRTLab.ipynb) to prepare the training data.
2. For C3D or 3D-CNN-LSTM, save the [pretrained Sports-1M C3D weights](https://drive.google.com/uc?id=1rlZ-xTkTMjgWKiQFUedRnHlDgQwx6yTm) as `c3d_weight/3dcnn_weights.h5`. MViTv2-S and MoViNet-A0-Base download their pretrained weights on first use.
3. Open [train_cnn_lstm_pytorch.ipynb](notebooks/train_cnn_lstm_pytorch.ipynb), run the AIRTLab-only data cell, and set `MODEL_NAME` to `3dcnnlstm`, `c3dcnn`, `lstm`, `mvit_v2_s`, or `movinet_a0_base`. Run ten-fold cross-validation grouped by source video, then final training to save a checkpoint under `weights/`. Video-level predictions use the two-step aggregation rule.
4. Run the independent efficiency benchmark to measure parameters, GMACs, latency, and peak GPU memory.

**Model structure and comparison**

<p align="center">
  <img src="images/Model Architecture.png" alt="3D-CNN-LSTM model architecture" />
</p>

<img src="images/Model Comparison.png" alt="Classifier architecture comparison" />

### YOLOv8 training

If you need to train a detector, save the [Ultralytics YOLOv8s starter weights](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) as `yolo_weight/yolov8s.pt`. Run [train_yolo.ipynb](notebooks/train_yolo.ipynb) with the Roboflow export. Use the resulting `src/runs/detect/<run>/weights/best.pt` for inference.

<img src="images/Yolo_results.png" alt="YOLOv8 training results" />

## 4. Analysis

1. Open [test_yolo.ipynb](notebooks/test_yolo.ipynb). Set `VIDEO_INPUT` to your video or input folder and `WEIGHTS_PATH` to a detector checkpoint trained on the Roboflow violence classes, such as the `best.pt` from Section 3. Run the extraction section to save candidate clips under `data/streaming videos/filter_results/`.
2. Open [test_cnn_lstm_pytorch.ipynb](notebooks/test_cnn_lstm_pytorch.ipynb). Set `VIDEO_PATH` to the extracted clips, select the classifier with `MODEL_NAME`, and point `CHECKPOINT_PATH` to its matching `.pt` file under `weights/`. Run inference to classify the clips and summarize predicted violence durations.

### Yolo Detection analysis

<img src="images/Misson_ Impossible - Fallout_sample_clip_4_01_47_to_02_03.gif" />
<img src="images/Misson_ Impossible - Fallout_sample_clip_6_02_38_to_04_17.gif" />

### Streaming videos analysis
#### Non-violence
<img src="images/Game_of_Thrones_S1_E8_(3.00-6.00_29.00-32.00)_sample_clip_5_03_20_to_03_28.gif" />

#### Low-level violence
<img src="images/Ip_Man(58.00-1.00.30_1.35.00-1.40.00)_sample_clip_6_04_48_to_05_13.gif" />

#### high-level violence
<img src="images/The_Boys_S4_E2_(40.00-46.00)_sample_clip_2_00_32_to_01_05.gif" />
