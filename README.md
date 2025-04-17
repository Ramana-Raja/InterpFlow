
# Frame Generator Using RIFE

## Overview

This project implements a frame generation tool using the RIFE (Real-Time Intermediate Flow Estimation) model. The goal is to generate intermediate frames between two video frames for smooth interpolation in video processing.

## Installation

To run this project, make sure you have the required dependencies installed. You can install them using `pip`:

The dependencies include:

- `torch`
- `opencv-python`
- `PIL`
- `numpy`

## Usage

### Initialization

To initialize the frame generator, provide the location of the video file and the desired frames per second (FPS):

```python
frame_gen = frame_generator(fps=30)
```


### Model Training

The model can be trained by providing a video and specifying the number of epochs. Training will use the frames extracted from the video.

```python
frame_gen.train(video_loc="train_video",epochs=5, freq=500, save_folder="models/")
```

### Model Prediction

Once the model is trained, you can use it to generate frames and save them as a video.

```python
frame_gen.predict(video_dr="video_folder",output_folder="output_folder/")
```

### Acknowledgments

This project uses code from the [RIFE](https://github.com/hzwer/ECCV2022-RIFE) repository, which is licensed under the MIT License. The original authors of this repository are:

- [hzwer](https://github.com/hzwer)

Please retain the copyright notice and license text in this project as specified in the MIT License.
