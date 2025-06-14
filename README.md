# InterpFlow

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

To initialize the frame generator:

```python
from InterpFlow import InterpFlowModel

model = InterpFlowModel()
```


### Model Training

The model can be trained by providing a video and specifying the number of epochs. Training will use the frames extracted from the video.

```python
model.train(video_loc="train_video",epochs=5, freq=500, save_folder="models/")
```
### Loading Of Models
Loading pretrained models
```python
model.load_model("new_rife_model_weights") #uses pretrained model
```

### Model Prediction

Once the model is trained, you can use it to generate frames and save them as a video.

```python
model.predict(video_dr="video_folder",output_folder="output_folder/")
```

### Acknowledgments

This project uses code from the [RIFE](https://github.com/hzwer/ECCV2022-RIFE) repository, which is licensed under the MIT License. The original authors of this repository are:

- [hzwer](https://github.com/hzwer)

Please retain the copyright notice and license text in this project as specified in the MIT License.

## License

This project is licensed under the BSD 3-Clause License.

## Third-party Licenses

This project uses third-party code that is licensed under the MIT License. See the `third_party_licenses.txt` file for details.
