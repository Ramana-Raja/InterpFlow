from RIFE import Model as main_model , get_learning_rate
import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
import shutil
from frame_generation.loss import device
from frame_generation.test_onnx import TRTInference

class frame_generator():
    def __init__(self,fps=60):
        self.model = main_model()
        self.fps = fps
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_images(self):
        self.temp_dir = "temp_folder"
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.dr)
        frame_count = 0

        # Read until video is completed
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
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def delete_files_predict(self):
        if os.path.exists(self.temp_dir_output):
            shutil.rmtree(self.temp_dir_output)
    def make_nparray_for_train(self):
        x_train = []
        x_train1 = []
        y_train = []
        i = 0
        j =0
        for filename in os.listdir(self.temp_dir):
                if j<2700:
                    j +=1
                    continue
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image = Image.open(os.path.join(self.temp_dir, filename))
                    image = image.resize((640, 480))
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
                j = j+1

        x = np.array(x_train)
        x1 = np.array(x_train1)
        y = np.array(y_train)
        x = torch.tensor(x, dtype=torch.float32)
        x1 = torch.tensor(x1, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x, x1, y)
        self.dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)

    def generate_images(self,prediction, test_input,test_input2, tar):
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
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
    def export_model_to_onnx(self,output_path, img_size=(720, 1280),batch=4):
        model = self.model
        model.eval()

        dummy_input = torch.randn(batch, 6, img_size[0], img_size[1]).cuda()
        torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        )

        print(f"ONNX model exported to: {output_path}")
    def build_rtr_engine(self,onnx_path):
        from frame_generation.real import EngineBuilder

          # Change to your ONNX model path
        onnx_model_path = onnx_path
        engine_file_path = "model.trt"  # Change to your desired output path
        precision_mode = "fp16"  # Choose "fp16" or "fp32"
        use_int8_precision = False  # Set to True if you want to use INT8 (ensure your hardware supports it)
        workspace_size_gb = 10  # Max memory workspace in GB for optimization
        verbose_logging = True  # Enable verbose logging for detailed output

        builder = EngineBuilder(verbose=verbose_logging, workspace=workspace_size_gb)

        # Load the ONNX model and create the TensorRT engine
        if builder.create_network(onnx_model_path):
            builder.create_engine(engine_file_path, precision=precision_mode, use_int8=use_int8_precision)


    def fit(self,epochs=5,freq=500,save_folder=None,video_loc="",batch=2,num_workers=4):
        self.num_workers = num_workers
        self.batch = batch
        self.dr = video_loc
        self.create_images()
        self.make_nparray_for_train()
        for p in range(epochs):
            for i, (batch_x, batch_x1, batch_y) in enumerate(self.dataloader):

                x = batch_x.to(self.device)
                x1 = batch_x1.to(self.device)
                y = batch_y.to(self.device)
                x = x.permute(0, 3, 1, 2)
                x1 = x1.permute(0, 3, 1, 2)
                y = y.permute(0, 3, 1, 2)
                x_new = torch.cat((x, x1), 1)
                if freq!= 0 and int(i) % freq == 0:
                    p = self.model.inference(img0=x, img1=x1)
                    self.generate_images(p, x, x1, y)
                start_time = time.time()
                lr = get_learning_rate(i)
                loss = self.model.update(x_new, y, lr)
                end_time = time.time()
                time1 = end_time - start_time
                print(f"loss={loss}   time taken={time1}  epochs={i}")
        if (save_folder):
            self.model.save_model(path=save_folder,
                         rank=0)
        self.delete_files_train()

    def load_model(self,loc=""):
        self.model.load_model(path=loc,rank=0)

    def save_images_on_batch(self,x,temp,x_1):
        for i in range(self.batch):
            if self.j == 0:
                cv2.imwrite(os.path.join(self.temp_dir_output, f"{self.j}.png"), x[i])
                self.j += 1
            cv2.imwrite(os.path.join(self.temp_dir_output, f"{self.j}.png"), temp[i])
            self.j += 1
            cv2.imwrite(os.path.join(self.temp_dir_output, f"{self.j}.png"), x_1[i])
            self.j += 1
    def predict(self,output_folder,video_dr="",batch=1,path_to_trt=None,output_width=1280,output_height=720):
        if (path_to_trt):
            trt_model = TRTInference(path_to_trt)

        self.batch = batch
        self.j = 0
        self.frame_position = 300
        self.video_predict = video_dr
        self.temp_dir_output = "temp_output"
        self.cap = cv2.VideoCapture(self.video_predict)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_frame = None
        if os.path.exists(self.temp_dir_output):
            shutil.rmtree(self.temp_dir_output)
        os.makedirs(self.temp_dir_output, exist_ok=True)

        if path_to_trt:
            while True:
                x, x_1 = self.create_images_for_predict()

                if x is None or x_1 is None:
                    break

                result = np.concatenate((x, x_1), axis=1)

                temp = trt_model.infer(result)
                print(temp.shape)
                temp = np.transpose(temp, (0, 2, 3, 1))  # Change shape to (N, H, W, C)
                x = np.transpose(x, (0, 2, 3, 1))  # Change shape to (N, H, W, C)
                x_1 = np.transpose(x_1, (0, 2, 3, 1))  # Change shape to (N, H, W, C)

                temp = (temp * 255).clip(0, 255).astype(np.uint8)
                x = (x * 255).clip(0, 255).astype(np.uint8)
                x_1 = (x_1 * 255).clip(0, 255).astype(np.uint8)
                temp = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])
                x = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x])
                x_1 = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_1])
                self.save_images_on_batch(x=x, temp=temp, x_1=x_1)

        else:
            while True:
                x, x_1 = self.create_images_for_predict()

                if x is None or x_1 is None:
                    break

                result = np.concatenate((x, x_1), axis=1)
                tensor = torch.tensor(result).to(self.device)
                temp = self.model.inference(tensor)
                print(temp.shape)

                temp = temp.permute(0, 2, 3, 1)
                temp = temp.detach().cpu().numpy()


                x = np.transpose(x, (0, 2, 3, 1))  # Change shape to (N, H, W, C)
                x_1 = np.transpose(x_1, (0, 2, 3, 1))  # Change shape to (N, H, W, C)

                temp = (temp * 255).clip(0, 255).astype(np.uint8)
                x = (x * 255).clip(0, 255).astype(np.uint8)
                x_1 = (x_1 * 255).clip(0, 255).astype(np.uint8)
                temp = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in temp])
                x = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x])
                x_1 = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x_1])
                self.save_images_on_batch(x=x, temp=temp, x_1=x_1)


        images = sorted(
            [img for img in os.listdir(self.temp_dir_output) if img.endswith(".jpg") or img.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        first_image = cv2.imread(os.path.join(self.temp_dir_output, images[0]))
        height, width, _ = first_image.shape
        output_video_path = os.path.join(output_folder, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps*2, (width, height))

        for image_name in images:
            image_path = os.path.join(self.temp_dir_output, image_name)
            image = cv2.imread(image_path)
            video_writer.write(image)

        video_writer.release()
        self.delete_files_predict()
# m = frame_generator()
# # m.fit(video_loc="C:\\Users\\raman\\Videos\\NVIDIA\\Marvels Spider-Man 2\\Marvels Spider-Man 2 2025.04.17 - 17.57.11.06.DVR.mp4",
# #       save_folder="C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\experimental_save_model",
# #       batch=8)
# m.load_model("C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\experimental_save_model")
# trt = "C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\model.trt"
# s = time.time()
# m.predict(video_dr="C:\\Users\\raman\\Videos\\Red Dead Redemption 2\\Red Dead Redemption 2 2024.07.03 - 21.28.47.03.mp4",
#           output_folder="C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\video",
#           batch=8,
#           path_to_trt=trt)
# e = time.time()
#
# print(e-s)
# import time
# s = time.time()
# n = m.predict(video_dr="C:\\Users\\raman\\Videos\\Red Dead Redemption 2\\Red Dead Redemption 2 2024.07.03 - 21.28.47.03.mp4",
#           output_folder="C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\video",
#           batch=8)
# e = time.time()
#
# print(e-s)
# # WORKING_DIR = "C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation"
# # ENGINE_FILE_PATH = os.path.join(WORKING_DIR, 'rife_model_trt.engine')
# # ONNX_MODEL_PATH = os.path.join(WORKING_DIR, 'rife_model.onnx')
# # m.export_model_to_onnx(ONNX_MODEL_PATH,batch=8)
# # onnx_model_path = "C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\rife_model.onnx"
# # m.build_rtr_engine(onnx_model_path)


