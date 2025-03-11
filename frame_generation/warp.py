import torch
import torch.nn as nn
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)


    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
# image = Image.open("C:\\Users\\raman\\PycharmProjects\\pythonProject1\\tensorflow1\\frame_generation\\frames\\frame_0000.jpg")
# image = image.resize((512, 512))
# image = image.convert("RGB")
# image_array = np.array(image)
# image_array = (image_array/255.0).astype('float32')
# new_image = [image_array]
# new = torch.tensor(new_image).to(device)
# k = warp(new,torch.ones((1, 512, 512, 3)).to("cuda"))
# k = k.to("cpu")
# print(k.shape)
# plt.imshow(k[0])
# plt.show()
# tenFlow = torch.ones(1,3,10,10).to(device)
# tenInput = torch.rand(1,3,10,10).to(device)
# tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
#             1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1).to(device)
# tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
#             1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3]).to(device)
# backwarp_tenGrid[0] = torch.cat(
#             [tenHorizontal, tenVertical], 1).to(device).to(device)
# tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
#                          tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1).to(device)
#
# g = (backwarp_tenGrid[0] + tenFlow).permute(0, 2, 3, 1).to(device)
#
# print(torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device))
# print(torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
#             1, 1, 1, tenFlow.shape[3]))
# print(torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
#             1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1))
# print(torch.cat(
#             [tenHorizontal, tenVertical], 1).to(device)[0])
# print(tenHorizontal.shape)
# print(tenVertical.shape)
# print(tenInput)
# print(torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True))