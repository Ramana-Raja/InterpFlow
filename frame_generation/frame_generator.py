from RIFE import Model as main_model , get_learning_rate
import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

from frame_generation.loss import device


class frame_generator():
    def __init__(self,fps=30):
        self.model = main_model()
        self.fps = fps
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_images(self):
        self.temp_dir = "temp_folder"

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
    def create_images_for_predict(self):
        x_0 = None
        x_1 = None

        if not self.video_predict or not os.path.exists(self.video_predict):
            print(f"Invalid or missing video path: {self.video_predict}")
            return None, None

        cap = cv2.VideoCapture(self.video_predict)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_position >= total_frames:
            return None,None
        print("video captured")
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)

        for i in range(2):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None, None

            # Convert to RGB and resize using PIL
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image.resize((640, 480))
            image_array = np.array(image).astype('float32') / 255.0

            # Expand dims to simulate batch if needed
            image_array = np.expand_dims(image_array, axis=0)
            if i == 0:
                x_0 = image_array
            else:
                x_1 = image_array

        self.frame_position += 1

        cap.release()

        # Convert to torch tensors
        x_0 = torch.tensor(x_0, dtype=torch.float32).to(device)
        x_1 = torch.tensor(x_1, dtype=torch.float32).to(device)
        return x_0.permute(0, 3, 1, 2), x_1.permute(0, 3, 1, 2)

    def delete_files(self):
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            os.remove(file_path)
        for filename in os.listdir(self.temp_dir_output):
            file_path = os.path.join(self.temp_dir, filename)
            os.remove(file_path)
    def make_nparray_for_train(self):
        x_train = []
        x_train1 = []
        y_train = []
        i = 0
        j =0
        for filename in os.listdir(self.temp_dir):
                if j>1000:
                    break
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
        print(x.shape)
        x1 = np.array(x_train1)
        y = np.array(y_train)
        x = torch.tensor(x, dtype=torch.float32)
        x1 = torch.tensor(x1, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x, x1, y)
        self.dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

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
    def fit(self,epochs=5,freq=500,save_folder=None,video_loc=""):
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
    def predict(self,output_folder,video_dr=""):
        self.frame_position = 0
        self.video_predict = video_dr
        self.temp_dir_output = "temp_output"
        os.makedirs(self.temp_dir_output, exist_ok=True)
        j=0
        while True:
            x , x_1 =self.create_images_for_predict()
            if x is None or x_1 is None:
                break

            temp = self.model.inference(x, x_1)

            temp = temp.permute(0, 2, 3, 1)
            temp = temp[0].detach().cpu().numpy()

            x = x.permute(0, 2, 3, 1)
            x = x[0].detach().cpu().numpy()

            x_1 = x_1.permute(0, 2, 3, 1)
            x_1 = x_1[0].detach().cpu().numpy()

            temp = (temp * 255).clip(0, 255).astype(np.uint8)
            x = (x * 255).clip(0, 255).astype(np.uint8)
            x_1 = (x_1 * 255).clip(0, 255).astype(np.uint8)
            print(x[0].shape)
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            x_1 = cv2.cvtColor(x_1, cv2.COLOR_RGB2BGR)
            if j==0:
                cv2.imwrite(os.path.join(self.temp_dir_output, f"{j}.png"), x)
                j +=1
            cv2.imwrite(os.path.join(self.temp_dir_output, f"{j}.png"), temp)
            j+=1
            cv2.imwrite(os.path.join(self.temp_dir_output, f"{j}.png"), x_1)
            j+=1
        images = [img for img in os.listdir(self.temp_dir_output) if img.endswith(".jpg") or img.endswith(".png")]
        print(len(images))
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

# m = frame_generator()
# m.fit(video_loc="C:\\Users\\raman\\Videos\\Red Dead Redemption 2\\Red Dead Redemption 2 2024.07.03 - 21.28.47.03.mp4",
#       freq=0,
#       save_folder="C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\saved_model",
#       epochs=10)
# m.predict(video_dr="C:\\Users\\raman\\Videos\\Red Dead Redemption 2\\Red Dead Redemption 2 2024.07.03 - 21.28.47.03.mp4",
#           output_folder="C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\video")