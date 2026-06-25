## Deep learning framework for video-based violence and abuse detection in movies
### Requirements
To run the code in this repository, you need to have the following libraries installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Jupyter

You can install the required packages using the following command:

```bash
pip install tensorflow keras numpy matplotlib opencv-python-headless jupyter
```

### Folder Structure
```text
VIOLENCE-MOVIES/
├── data/
│   ├── processed/              # processed datasets
│   │   └── violence-detection-dataset/
│   ├── raw/                    # raw datasets
│   │   ├── Real Life Violence Dataset/
│   │   └── violence-detection-dataset/
│   ├── roboflow/               # Roboflow YOLO dataset
│   └── streaming videos/       # videos for real-world inference testing
│       ├── filter_results/
│       └── raw_samples/
├── deprecated/                 # old experimental code
├── images/                     # images used in README/report
├── src/                        # main source code and notebooks
│   └── preprocess/             # preprocessing scripts
├── yolo_weight/                # YOLO model weights
└── c3d_weight/                 # C3D model weights
```


## 1. Project Objective and Content Overview
 
Title: Deep learning for violence and abuse detection in cinema and series: OTT streaming platforms
Content: This project uses deep learning and computer vision techniques to detect and remove violent and inappropriate content from movie videos. Unlike current common detection methods that only identify whether the violence is violent or not, the framework also categorizes the identified violence in terms of lowand high levels  in movies. It combines YOLO and CNN-LSTM models to detect and categorize violent content with 70% accuracy. With the increased concern about violent content in Hollywood movies, the framework fills the current gap in content review for streaming platforms, reducing the reliance on manual content review and improving efficiency. Its scalability across different video types makes it a powerful tool for content management on digital platforms.

<img src="images\Framework6.png" width="720"/>

## 2.Dataset and Preprocessing
### a) AIRTLab Dataset
Dataset AIRTLab: https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
Or we can use the following command line:

```bash
# downloads the AIRTLAB dataset for violence detection
!mkdir -p ../../data/raw
!git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git ../../data/raw
```
We made a new organizing classification of the above dataset into nonviolent (including gesture, handshake, highfive, hug, jump, walk, and greet), low-violence (including push, slap, choking, stifle, kick, and punch), and high-violence (including shoot, stab, and club). Then, We divide each video into chunks with 16 frames at a resolution of 112 x 112 (e.g. a video may have 8 chunks) and store the samples and labels to data\processed\violence-detection-dataset. The code refers to src\preprocess\preprocess_AIRTLab.

### b) Roboflow Violence Detection Dataset
Around 10 thousand images annotated by 4 classes of box bounding, including NonViolence, Violence, guns, knife.
Roboflow Violence Detection Dataset: https://universe.roboflow.com/violence-detection-fbe46/violence-detection-nbx24

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

We store the dataset to data\roboflow.

### c) Streaming videos
Use streaming video clips from movies as the analysis source, including John Wick, The Boys, The Punisher, Game of Thrones, Ip Man, Mission: Impossible – Fallout. Please refer to data\streaming videos

Link: https://drive.google.com/drive/folders/1ISmc3xwPOwAkuYoVuEcPwje267oRG9k2?usp=sharing

## 3. Training
### 3dcnn-lstm training
We used pretained weight from Sports-1M. Please refer to src\train_cnn_lstm_keras

Link: https://github.com/aslucki/C3D_Sport1M_keras

Model Structure

<p align="center">
  <img src="images\Model Architecture.png"  width="380"/>
  <img src="images\video prediction.png"  width="380"/>
</p>


Model Comparison

<img src="images\Model Comparison.png"/>

<!-- <img src="images\Average Validation Accuracy of Chunks and Videos Across Epochs.png" />
<p align="center">
    <img src="images/3dcnn-lstm-roc.png" width="360" />
    <img src="images/3dcnn-lstm-valacc.png" width="360" />
</p> -->

### Yolov8 training
Please refer to src\train_yolo
<img src="images/Yolo_results.png" />

## 4. Analysis
We run yolo first to extract subvideos that contain violence frames from streaming video. The extracting logic is if we detect violence frame, then we keep that frame and the following 5 seconde frames as a subvideo, saving in data\streaming videos\filter_results. Please refer to src\test_yolo.
Then we run 3dcnn-lstm model to analyse the classification result from previous subvideos. Please refer to src\test_cnn_lstm_keras.

### Yolo Detection analysis
Please refer to src\test_yolo

<img src="images/Misson_ Impossible - Fallout_sample_clip_4_01_47_to_02_03.gif" />
<img src="images/Misson_ Impossible - Fallout_sample_clip_6_02_38_to_04_17.gif" />

### Streaming videos analysis
Please refer to src\test_cnn_lstm_keras
#### Non-violence
<img src="images/Game_of_Thrones_S1_E8_(3.00-6.00_29.00-32.00)_sample_clip_5_03_20_to_03_28.gif" />

#### Low-level violence
<img src="images/Ip_Man(58.00-1.00.30_1.35.00-1.40.00)_sample_clip_6_04_48_to_05_13.gif" />

#### high-level violence
<img src="images/The_Boys_S4_E2_(40.00-46.00)_sample_clip_2_00_32_to_01_05.gif" />

### Preprint
 
Rohitash Chandra, Jiaqian Li, Ying Dong, Henghao Zhuang, Dongming Wu; Deep learning framework for video-based violence and abuse detection in movies, Research Square: https://www.researchsquare.com/article/rs-10061173/v1
