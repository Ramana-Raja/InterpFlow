import cv2
import os
import numpy as np
import torch

def print_progress_bar(current, total, length=30, prefix='Progress'):
    from sys import stdout
    percent = current / total
    filled_len = int(length * percent)
    bar = '=' * filled_len + '.' * (length - filled_len)
    stdout.write(f'\r{prefix}: [{bar}] {current}/{total} ({percent * 100:.1f}%)')
    stdout.flush()

class InterpFlowModel:
    def __init__(self,version="v3"):
        if version == "v1":
            from InterpFlow.Models.v1.RIFE import Model as main_model
            self.model = main_model()
        if version == "v2":
            from InterpFlow.Models.v2.RIFE_NEW import Model as main_model
            self.model = main_model()
        else:
            from InterpFlow.Models.v3.RIFE_HDv3 import Model as main_model
            self.model = main_model()
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fitted = False

    def create_images(self,delete_previous=True):
        """
            Creates images needed for training the model

            Parameters
            ----------
            delete_previous: Boolean
                Deletes previous images if present in "temp_folder"
                might be useful to set false on training on same video

        """

        import shutil

        self.temp_dir = "temp_folder"
        if os.path.exists(self.temp_dir):
            if delete_previous:
                shutil.rmtree(self.temp_dir)
            else:
                return
        os.makedirs(self.temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.dr)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Save the frame as an image file
                frame_name = f"frame_{frame_count:04d}.jpg"
                frame_path = os.path.join(self.temp_dir, frame_name)
                cv2.imwrite(frame_path, frame)

                frame_count += 1
            else:
                break

        cap.release()

    def create_images_for_predict(self,width=None,height=None):
        """
            Creates images needed for predicting from the video

            Parameters
            ----------
            width: int
                specifies output video width shape, if None then
                uses video shape

            height: int
                specifies output video height shape, if None then
                uses video shape

            Returns
            -------
            np.ndarray
                returns two np.ndarray each containing frames in the format
                    1st array: [frame0,frame1,frame2....]
                    2nd array: [frame1.frame2,frame3....]

        """

        if self.frame_position >= self.total_frames:
            self.cap.release()
            return None, None
        x_0 = []
        x_1 = []
        if width != None and height != None:
            target_width = width
            target_height = height


            for _ in range(self.batch):
                if self.last_frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        return None, None
                    self.last_frame = frame
                    self.frame_position += 1

                ret, frame = self.cap.read()
                if not ret:
                    return None, None

                img0 = cv2.resize(self.last_frame, (target_width, target_height))
                img1 = cv2.resize(frame, (target_width, target_height))

                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

                arr0 = np.array(img0).astype('float32') / 255.0
                arr1 = np.array(img1).astype('float32') / 255.0

                x_0.append(arr0)
                x_1.append(arr1)

                self.last_frame = frame
                self.frame_position += 1
        else:
            for _ in range(self.batch):
                if self.last_frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        return None, None
                    self.last_frame = frame
                    self.frame_position += 1

                ret, frame = self.cap.read()
                if not ret:
                    return None, None

                img0 = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                arr0 = np.array(img0).astype('float32') / 255.0
                arr1 = np.array(img1).astype('float32') / 255.0

                x_0.append(arr0)
                x_1.append(arr1)

                self.last_frame = frame
                self.frame_position += 1
        x_0 = np.array(x_0).transpose(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        x_1 = np.array(x_1).transpose(0, 3, 1, 2)  # Change shape to (N, C, H, W)

        return x_0,x_1

    def delete_files_train(self):
        """
            Deletes the temporary folder created when training
        """

        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def make_nparray_for_train(self,
                               max_frames=1000,
                               start_frame=0,
                               width=640,
                               height=480):
        """
            Creates npdarray of images needed for training the model

            Parameters
            ----------
            max_frames: int
                specifies the number of frames that is going
                to be used for training

            start_frame: int
                specifies the starting frame/point for training

            width : int
                specifies the images width that is going to be
                used for training

            height: int
                specifies the images height that is going to be
                used for training

            Returns
            -------
            PyTorch DataLoader contain
                1st: torch.tensor of images in format [frame0,frame3,frame6....]
                     actes as img0 for training

                2nd: torch.tensor of images in format [frame1,frame4,frame7....]
                     actes as ground truth for training

                3rd: torch.tensor of images in format [frame2,frame5,frame8....]
                     actes as img1 for training
        """

        from PIL import Image
        from torch.utils.data import DataLoader, TensorDataset
        x_train = []
        x_train1 = []
        y_train = []
        i = 0
        j =0
        max_frames = start_frame + max_frames
        if width and height:
            for filename in os.listdir(self.temp_dir):
                if j < start_frame:
                    j +=1
                    continue
                if j > max_frames:
                    break
                if filename.endswith(".jpg") or filename.endswith(".png"):

                    image = Image.open(os.path.join(self.temp_dir, filename))
                    image = image.resize((width, height))
                    image = image.convert("RGB")
                    image_array = np.array(image)
                    image_array = (image_array / 255.0).astype('float32')
                    if (i == 0):
                        x_train.append(image_array)
                        i = i + 1
                    if (i == 1):
                        y_train.append(image_array)
                        i = i + 1
                    if (i == 2):
                        x_train1.append(image_array)
                        i = 0
                    image.close()
                j = j + 1
        else:
            for filename in os.listdir(self.temp_dir):
                if j < start_frame:
                    continue
                if j > max_frames:
                    break
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image = Image.open(os.path.join(self.temp_dir, filename))
                    image = image.convert("RGB")
                    image_array = np.array(image)
                    image_array = (image_array / 255.0).astype('float32')
                    if (i == 0):
                        x_train.append(image_array)
                        i = i + 1
                    if (i == 1):
                        y_train.append(image_array)
                        i = i + 1
                    if (i == 2):
                        x_train1.append(image_array)
                        i = 0
                    image.close()
                j = j + 1

        x = np.array(x_train)
        x1 = np.array(x_train1)
        y = np.array(y_train)

        x = torch.tensor(x, dtype=torch.float32)
        x1 = torch.tensor(x1, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x, x1, y)

        self.dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=False)

    def generate_images(self,prediction, test_input,test_input2, tar):
        """
            Plots the img0, img1, ground_truth and predicted img

            Parameters
            ----------
            prediction: torch.tensor
                predicted image by the model

            test_input: torch.tensor
                img0

            test_input2 : torch.tensor
                img1

            tar: torch.tensor
                Ground Truth
        """

        import matplotlib.pyplot as plt

        prediction = prediction.permute(0, 2, 3, 1)
        prediction = prediction.to("cpu")
        prediction = prediction.detach().numpy()

        test_input = test_input.permute(0, 2, 3, 1)
        test_input = test_input.to("cpu")
        test_input = test_input.detach().numpy()

        test_input2 = test_input2.permute(0, 2, 3, 1)
        test_input2 = test_input2.to("cpu")
        test_input2 = test_input2.detach().numpy()

        tar = tar.permute(0, 2, 3, 1)
        tar = tar.to("cpu")
        tar = tar.detach().numpy()

        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], test_input2[0], prediction[0]]
        title = ['Input Image1', "Ground Truth", 'input_image2', 'Predicted Image']

        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i]*225)
            plt.axis('off')
        plt.show()

    def export_model_to_onnx(self,output_path, img_size=(480, 640),batch=4):
        """
            Converts the model to onnx

            Parameters
            ----------
            output_path: String
                Location where the model is going to be saved

            img_size: Tuple
                The dummy input image shape

            batch : int
                 Number of batch images going to be used for
                 predict
        """

        model = self.model
        model.eval()

        dummy_input = torch.randn(batch, 6, img_size[0], img_size[1]).to(self.device)

        torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
        )

        self.output_path = output_path

        print(f"ONNX model exported to: {output_path}")


    def build_rtr_engine(self,
                         onnx_path=None,
                         precision_mode = "fp16",
                         use_int8_precision = False,
                         workspace_size_gb = 10,
                         verbose_logging = True) :
        """
            Builds TRT model from onnx

            Parameters
            ----------
            onnx_path: String
                Location where the onnx model is present

            precision_mode: String
                The precision model that is going to be used

            use_int8_precision : boolean
                 Use INT8 precision

            workspace_size_gb: int
                 Maximum VRAM usage while building trt model

            verbose_logging: boolean
                 Printing all log info in terminal

            Returns
            ----------
            TRT Model: A model saved in trained_models/trt_models with onnx file name

        """

        if not onnx_path:
            if self.output_path:
                onnx_model_path = self.output_path
            else:
                raise RuntimeError("No onnx_path provided")
        else:
            onnx_model_path = onnx_path
        from InterpFlow.TRT.TRTEngineBuilder import EngineBuilder
        from pathlib import Path

        onnx_model_path = Path(onnx_model_path)
        filename = onnx_model_path.name
        trt_filename = Path(filename).with_suffix(".trt")
        trt_model_path = onnx_model_path.parent.parent / "trt_models" / trt_filename

        builder = EngineBuilder(verbose=verbose_logging, workspace=workspace_size_gb)

        if builder.create_network(onnx_model_path):
            builder.create_engine(trt_model_path, precision=precision_mode, use_int8=use_int8_precision)


    def fit(self,
            epochs=5,
            freq=500,
            save_folder=None,
            video_loc="",
            batch=2,
            delete_previous=True,
            start_frame=0,
            max_frames=1000,
            width=None,
            height=None):

        """
            Fits the model with the input video

            Parameters
            ----------
            epochs : int
                Number of epochs for training

            freq : int
                Plot the prediction of model with the frequency of freq
                (uses batch for counting frequency)

            save_folder : String
                 Location where the trained model is going to be saved

            video_loc : String
                 Location of the video that is going to be used for training

            batch : int
                 Size of each batch

            delete_previous : Boolean
                 Deletes if any images present in temp folder of training

            start_frame : int
                 The starting frame/point of the video that is going to be
                 used for training

            max_frames: max number of images use for training
                        Note: since we use three array(img0, ground_truth, img1),
                        total images will be max_frames*3

            width : int
                 Width of the image that is going to be used for training
                 if None , uses video resolution

            height : int
                 Height of the image that is going to be used for training
                 if None , uses video resolution
        """

        import time
        from sys import stdout
        from InterpFlow.Models.v1.RIFE import get_learning_rate

        self.batch = batch
        self.dr = video_loc
        print("creating images")
        self.create_images(delete_previous)
        print("converting to numpy array")
        self.make_nparray_for_train(start_frame=start_frame,max_frames=max_frames,width=width,height=height)
        print("training")

        for p in range(epochs):
            total_batches = len(self.dataloader)
            for i, (batch_x, batch_x1, batch_y) in enumerate(self.dataloader):

                x = batch_x.to(self.device)
                x1 = batch_x1.to(self.device)
                y = batch_y.to(self.device)
                x = x.permute(0, 3, 1, 2)
                x1 = x1.permute(0, 3, 1, 2)
                y = y.permute(0, 3, 1, 2)

                x_new = torch.cat((x, x1), 1)
                if freq!= 0 and int(i) % freq == 0:
                    imgs = torch.cat((x, x1), 1)
                    p = self.model.inference(imgs)
                    self.generate_images(p, x, x1, y)

                start_time = time.time()
                global_step = p * total_batches + i
                lr = get_learning_rate(global_step)
                loss = self.model.update(x_new, y,lr)
                end_time = time.time()
                time1 = end_time - start_time

                print_progress_bar(i + 1, total_batches, prefix=f"Epoch {p + 1}/{epochs}")
                stdout.write(f" - loss: {loss:.12f} - time: {time1:.2f}s")
                stdout.flush()
        self.fitted = True
        if (save_folder):
            self.model.save_model(path=save_folder,
                         rank=0)
        self.delete_files_train()

    def load_model(self,loc=""):
        """
            Loads the fitted model

            Parameters
            ----------
            loc : String
                Location where the fitted model is saved

        """

        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, loc)
        self.model.load_model(path=model_path,rank=0)
        self.fitted = True

    def add_audio(self,output_video_path,output_folder):
        """
            Adds audio the final video created after interpolation

            Parameters
            ----------
            output_video_path : String
                Location of the output video from interpolation

            output_folder : String
                 Where the video with audio is going to be saved

        """

        import subprocess
        import imageio_ffmpeg as ffmpeg

        original_video = self.video_predict
        temp_output_video = output_video_path  # this is your interpolated video
        final_output_video = os.path.join(output_folder, "final_output_with_audio.mp4")

        ffmpeg_path = ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path, "-y",
            "-i", temp_output_video,
            "-i", original_video,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            final_output_video
        ]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def resource_path(self,relative_path):
        """
            Loads the fitted model for exe file

            Parameters
            ----------
            loc : String
                Location where the fitted model is saved relative to
                this file
        """

        import sys

        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

    def predict_wihtout_try(self,
                output_folder="",
                video_dr="",
                batch=1,
                path_to_trt=None,
                output_width=1280,
                output_height=720,
                progress_callback=None,
                from_app = False):
        """
            Interpolates the video using RIFE without exception handling

            Parameters
            ----------
            output_folder : String
                Where the model is going to be saved

            video_dr : String
                Location of the video

            batch : int
                 Number of batch

            path_to_trt : String
                 Location of the trt model is saved
                 If None , uses normal/torch model

            output_width : int
                 width of output video

            output_height : int
                 height of output video

            progress_callback: Function
                 Function that prints how much video has been processed

        """

        video_writer = None
        if progress_callback:
            progress_bar = progress_callback
        else:
            progress_bar = print_progress_bar
        self.batch = batch
        self.j = 0
        self.frame_position = 0
        self.video_predict = video_dr
        self.cap = cv2.VideoCapture(self.video_predict)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_frame = None
        output_video_path = os.path.join(output_folder, "output_video.mp4")
        self.model.eval()

        if not self.fitted:
            from InterpFlow.Models.v3.RIFE_HDv3 import Model as Model_2
            self.model = Model_2()
            self.model.eval()


            # model_path = self.resource_path("InterpFlow/trained_models/pretrained_for_v3")
            # self.model.load_model(model_path)


            script_dir = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(script_dir, "trained_models/pretrained_for_v3")
            self.model.load_model(model_path)

            while True:
                x, x_1 = self.create_images_for_predict(width=output_width, height=output_height)

                if x is None or x_1 is None:
                    break

                x = torch.tensor(x).to(self.device)
                x_1 = torch.tensor(x_1).to(self.device)
                imgs = torch.cat((x, x_1), 1)

                temp = self.model.inference(imgs)

                temp = temp.permute(0, 2, 3, 1)
                temp = temp.detach().cpu().numpy()

                x = x.permute(0, 2, 3, 1)
                x = x.detach().cpu().numpy()

                x_1 = x_1.permute(0, 2, 3, 1)
                x_1 = x_1.detach().cpu().numpy()

                if temp.shape != x.shape:
                    temp = np.array([cv2.resize(img, (x.shape[2], x.shape[1])) for img in temp])

                temp = (temp * 255).clip(0, 255).astype(np.uint8)
                x = (x * 255).clip(0, 255).astype(np.uint8)
                x_1 = (x_1 * 255).clip(0, 255).astype(np.uint8)

                temp = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])
                x = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x])
                x_1 = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_1])

                if video_writer is None:
                    height, width, _ = x[0].shape

                    output_video_path = os.path.join(output_folder, "output_video.mp4")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps * 2, (width, height))
                    video_writer.write(x[0])
                    self.j += 1
                    progress_bar(self.j, self.total_frames)

                for i in range(self.batch):
                    video_writer.write(temp[i])
                    video_writer.write(x_1[i])
                    self.j += 1
                    progress_bar(self.j, self.total_frames)

            progress_bar(self.total_frames, self.total_frames)
            video_writer.release()
            self.add_audio(output_video_path=output_video_path,output_folder=output_folder)

        elif path_to_trt:
                from InterpFlow.TRT.TRTReader import TRTInference
                trt_model = TRTInference(path_to_trt)

                while True:
                    x, x_1 = self.create_images_for_predict(width=output_width, height=output_height)

                    if x is None or x_1 is None:
                        break
                    result = np.concatenate((x, x_1), axis=1)

                    temp = trt_model.infer(result)

                    temp = np.transpose(temp, (0, 2, 3, 1))
                    x = np.transpose(x, (0, 2, 3, 1))
                    x_1 = np.transpose(x_1, (0, 2, 3, 1))

                    if temp.shape != x.shape:
                        temp = np.array([cv2.resize(img, (x.shape[2], x.shape[1])) for img in temp])

                    temp = (temp * 255).clip(0, 255).astype(np.uint8)
                    x = (x * 255).clip(0, 255).astype(np.uint8)
                    x_1 = (x_1 * 255).clip(0, 255).astype(np.uint8)
                    temp = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])
                    x = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x])
                    x_1 = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_1])

                    if video_writer is None:
                        height, width, _ = x[0].shape

                        output_video_path = os.path.join(output_folder, "output_video.mp4")

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                        video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps * 2, (width, height))
                        video_writer.write(x[0])
                        self.j += 1
                        progress_bar(self.j, self.total_frames)
                    for i in range(self.batch):
                        video_writer.write(temp[i])
                        video_writer.write(x_1[i])
                        self.j += 1
                        progress_bar(self.j, self.total_frames)

                progress_bar(self.total_frames, self.total_frames)
                video_writer.release()
                return

        else:
            while True:
                x, x_1 = self.create_images_for_predict(width=output_width,height=output_height)

                if x is None or x_1 is None:
                    break

                x = torch.tensor(x).to(self.device)
                x_1 = torch.tensor(x_1).to(self.device)
                imgs = torch.cat((x, x_1), 1)
                temp = self.model.inference(imgs)

                temp = temp.permute(0, 2, 3, 1)
                temp = temp.detach().cpu().numpy()

                x = x.permute(0, 2, 3, 1)
                x = x.detach().cpu().numpy()

                x_1 = x_1.permute(0, 2, 3, 1)
                x_1 = x_1.detach().cpu().numpy()


                if temp.shape != x.shape:
                    temp = np.array([cv2.resize(img, (x.shape[2], x.shape[1])) for img in temp])

                temp = (temp * 255).clip(0, 255).astype(np.uint8)
                x = (x * 255).clip(0, 255).astype(np.uint8)
                x_1 = (x_1 * 255).clip(0, 255).astype(np.uint8)

                temp = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])
                x = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x])
                x_1 = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_1])

                if video_writer is None:
                    height, width, _ = x[0].shape

                    output_video_path = os.path.join(output_folder, "output_video.mp4")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps * 2, (width, height))
                    video_writer.write(x[0])

                    self.j += 1
                    progress_bar(self.j, self.total_frames)

                for i in range(self.batch):
                    video_writer.write(temp[i])
                    video_writer.write(x_1[i])
                    self.j +=1
                    progress_bar(self.j, self.total_frames)

        progress_bar(self.total_frames, self.total_frames)
        video_writer.release()
    def predict(self,
                output_folder="",
                video_dr="",
                batch=1,
                path_to_trt=None,
                output_width=1280,
                output_height=720,
                progress_callback=None,
                error_callback=None):
        """
            Interpolates the video using RIFE with exception handling

            Parameters
                   ----------
                   output_folder : String
                       Where the model is going to be saved

                   video_dr : String
                       Location of the video

                   batch : int
                        Number of batch

                   path_to_trt : String
                        Location of the trt model is saved
                        If None , uses normal/torch model

                   output_width : int
                        width of output video

                   output_height : int
                        height of output video

                   progress_callback: Function
                        Function that prints how much video has been processed

        """

        try:
            self.predict_wihtout_try(
                video_dr=video_dr,
                output_folder=output_folder,
                batch=batch,
                output_width=output_width,
                output_height=output_height,
                path_to_trt=path_to_trt,
                progress_callback=progress_callback)
        except Exception as e:
            if error_callback:
                error_callback(e)
            else:
                raise e
