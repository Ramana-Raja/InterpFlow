
# Frame Generator Using RIFE

## Overview

This project implements a frame generation tool using the RIFE (Real-Time Intermediate Flow Estimation) model. The goal is to generate intermediate frames between two video frames for smooth interpolation in video processing.

## Installation

To run this project, make sure you have the required dependencies installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

The dependencies include:

- `torch`
- `opencv-python`
- `PIL`
- `matplotlib`
- `numpy`

## Usage

### Initialization

To initialize the frame generator, provide the location of the video file and the desired frames per second (FPS):

```python
frame_gen = frame_generator(video_loc="path/to/video.mp4", fps=30)
```

### Generate Intermediate Frames

You can create images from a video, generate intermediate frames, and save the output:

```python
frame_gen.create_images()
frame_gen.train(epochs=5)
frame_gen.predict("output_video.mp4")
```

### Model Training

The model can be trained by providing a video and specifying the number of epochs. Training will use the frames extracted from the video.

```python
frame_gen.train(epochs=5, freq=500, save_folder="models/")
```

### Model Prediction

Once the model is trained, you can use it to generate frames and save them as a video.

```python
frame_gen.predict("output_folder/")
```

### Acknowledgments

This project uses code from the [RIFE](https://github.com/hzwer/ECCV2022-RIFE) repository, which is licensed under the MIT License. The original authors of this repository are:

- [hzwer](https://github.com/hzwer)

Please retain the copyright notice and license text in your project and provide appropriate credit to the original authors as specified in the license.
