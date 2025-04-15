from RIFE import Model as main_model , get_learning_rate
import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt


class frame_generator():
    def __init__(self,video_loc="",fps=30):
        self.dr = video_loc
        self.model = main_model()
        self.fps = fps
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_images(self):
        self.temp_dir = "temp_folder"

        if  os.path.exists(self.temp_dir):
            return
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

    def delete_files(self):
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            os.remove(file_path)
    def make_nparray_for_train(self):
        x_train = []
        x_train1 = []
        y_train = []
        i = 0
        j =0
        for filename in os.listdir(self.temp_dir):
                if(j>100):
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
                print(j)
        x = np.array(x_train)
        x1 = np.array(x_train1)
        y = np.array(y_train)
        x = torch.tensor(x, dtype=torch.float32)
        x1 = torch.tensor(x1, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x, x1, y)
        self.dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    def make_ndarray_for_predict(self):
        x_test = []
        y_test = []
        i = 0
        for filename in os.listdir(self.temp_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = Image.open(os.path.join(self.temp_dir, filename))
                image = image.resize((640, 480))
                image = image.convert("RGB")
                image_array = np.array(image)
                image_array = (image_array / 255.0).astype('float32')
                if (i%2== 0):
                    x_test.append(image_array)
                if (i%2!= 0):
                    y_test.append(image_array)
                i +=1
                image.close()
        x = np.array(x_test)
        y = np.array(y_test)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x , y

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
    def train(self,epochs=5,freq=500,save_folder=None):
        self.create_images()
        self.make_nparray_for_train()
        for j in range(epochs):
            for i, (batch_x, batch_x1, batch_y) in enumerate(self.dataloader):

                x = batch_x.to(self.device)
                x1 = batch_x1.to(self.device)
                y = batch_y.to(self.device)
                x = x.permute(0, 3, 1, 2)
                x1 = x1.permute(0, 3, 1, 2)
                y = y.permute(0, 3, 1, 2)
                x_new = torch.cat((x, x1), 1)
                if (int(i) % freq == 0):
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
    def predict(self,output_folder):
        self.temp_dir_output = "temp_output"
        os.makedirs(self.temp_dir_output, exist_ok=True)
        x , y = self.make_ndarray_for_predict()
        j=0
        for i in range(len(x)):
            temp = self.model.inference(x[i], y[i])

            cv2.imwrite(os.path.join(self.temp_dir_output, f"{j}.png"), x[i])
            j +=1
            cv2.imwrite(os.path.join(self.temp_dir_output, f"{j}.png"), temp)
            j+=1
            cv2.imwrite(os.path.join(self.temp_dir_output, f"{j}.png"), y[i])
            j+=1
        images = [img for img in os.listdir(self.temp_dir_output) if img.endswith(".jpg") or img.endswith(".png")]
        first_image = cv2.imread(os.path.join(self.temp_dir_output, images[0]))
        height, width, _ = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_folder, fourcc, self.fps*2, (width, height))

        for image_name in images:
            image_path = os.path.join(self.temp_dir_output, image_name)
            image = cv2.imread(image_path)
            video_writer.write(image)

        video_writer.release()



