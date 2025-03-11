from pytorch.DLSS4.ENCODER import TRANSFORMER
import torch
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


model = TRANSFORMER(encoder_layer=4,num_heads=16,d_model = 128)
model.load_model(path="C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\DLSS4\\model",rank=0)

x_test = []
x_test1 = []
k = 1
directory = "C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\frame_generation\\final"
p=0
for filename in os.listdir(directory):
    if (k == 500):
        break
    elif (k == 499):
        image = Image.open(os.path.join(directory, filename))
        image = image.resize((512, 512))
        image = image.convert("RGB")
        image_array = np.array(image)
        image_array = (image_array / 255.0).astype('float32')
        x_test1.append(image_array)
    else:
        image = Image.open(os.path.join(directory, filename))
        image = image.resize((512, 512))
        image = image.convert("RGB")
        image_array = np.array(image)
        image_array = (image_array / 255.0).astype('float32')
        if p == 0:
            x_test.append(image_array)
            p = p + 1
        if (p == 1):
            x_test1.append(image_array)
            x_test.append(image_array)
    k=k+1
x_test = np.array(x_test)
x_test1 = np.array(x_test1)



x_test = torch.tensor(x_test, dtype=torch.float32)



x_test1 = torch.tensor(x_test1, dtype=torch.float32)

x_test = x_test.permute(0, 3, 1, 2)
x_test1 = x_test1.permute(0, 3, 1, 2)

u=0
def save(img,pred,img1,u):

    img = img.permute(0, 2, 3, 1)
    img = img.to("cpu")
    img = img.detach().numpy()
    img = np.array(img[0])

    pred = pred.permute(0, 2, 3, 1)
    pred = pred.to("cpu")
    pred = pred.detach().numpy()

    pred = np.array(pred[0])
    img = img.reshape(512, 512, 3)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred = pred.reshape(512, 512, 3)
    pred = (pred * 255).astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    u = u + 1
    cv2.imwrite(f'C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\DLSS4\\final\\image{u}.png', img)
    u = u+1
    cv2.imwrite(f'C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\DLSS4\\final\\image{u}.png', pred)


    return u

def p(x1,x2):
        u=0
        for img,img1 in zip(x1,x2):
            img = np.array([img])
            img1 = np.array([img1])
            img = torch.tensor(img).to("cuda")
            img1 = torch.tensor(img1).to("cuda")

            p = model.inference(img,img1)
            u = save(img,p,img1,u)

p(x_test,x_test1)