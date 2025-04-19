import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from frame_generation.warp import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from frame_generation.IFNet import *
import torch.nn.functional as F
from frame_generation.loss import *
import os
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
import time
import math
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )

def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

c = 32

class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv0 = Conv2(3, c)
        self.conv1 = Conv2(c, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)

    def forward(self, x, flow):
        x = self.conv0(x)
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv0 = Conv2(10, c)
        self.down0 = Conv2(c, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, img0, img1, flow, c0, c1, flow_gt):
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        if flow_gt == None:
            warped_img0_gt, warped_img1_gt = None, None
        else:
            warped_img0_gt = warp(img0, flow_gt[:, :2])
            warped_img1_gt = warp(img1, flow_gt[:, 2:4])
        x = self.conv0(torch.cat((warped_img0, warped_img1, flow), 1))
        s0 = self.down0(x)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)

        return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt

def get_learning_rate(step, total_steps=3000):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (total_steps - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6
class Model(nn.Module):
    def __init__(self, local_rank=-1):
        super(Model, self).__init__()
        self.flownet = IFNet()
        self.contextnet = ContextNet()
        self.fusionnet = FusionNet()
        self.device()
        self.optimG = AdamW(itertools.chain(
            self.flownet.parameters(),
            self.contextnet.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-4)
        self.schedulerG = optim.lr_scheduler.CyclicLR(
            self.optimG, base_lr=1e-6, max_lr=1e-3, step_size_up=8000, cycle_momentum=False)
        self.epe = EPE()
        self.ter = Ternary()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[
                               local_rank], output_device=local_rank)
            self.contextnet = DDP(self.contextnet, device_ids=[
                                  local_rank], output_device=local_rank)
            self.fusionnet = DDP(self.fusionnet, device_ids=[
                                 local_rank], output_device=local_rank)

    def train_1(self):
        self.flownet.train()
        self.contextnet.train()
        self.fusionnet.train()

    def eval(self):
        self.flownet.eval()
        self.contextnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.contextnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            self.flownet.load_state_dict(
                convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))
            self.contextnet.load_state_dict(
                convert(torch.load('{}/contextnet.pkl'.format(path), map_location=device)))
            self.fusionnet.load_state_dict(
                convert(torch.load('{}/unet.pkl'.format(path), map_location=device)))

    def save_model(self, path, rank):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet.pkl'.format(path))
            torch.save(self.contextnet.state_dict(), '{}/contextnet.pkl'.format(path))
            torch.save(self.fusionnet.state_dict(), '{}/unet.pkl'.format(path))

    def predict(self, imgs, flow, training=True, flow_gt=None):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
            img0, img1, flow, c0, c1, flow_gt)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        if training:
            return pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt
        else:
            return pred

    def inference(self, img0, scale=1.0):
        # imgs = torch.cat((img0, img1), 1)
        flow, _ = self.flownet(img0, scale)
        return self.predict(img0, flow, training=False)
    def forward(self,imgs):
        flow, _ = self.flownet(imgs, 1.0)
        return self.predict(imgs, flow, training=False)
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train_1()
        else:
            self.eval()
        flow, flow_list = self.flownet(imgs)
        pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.predict(
            imgs, flow, flow_gt=flow_gt)
        loss_ter = self.ter(pred, gt).mean()

        loss_l1 = (((pred - gt) ** 2 + 1e-6) ** 0.5).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_ter*(1.5)
            loss_G.backward()
            self.optimG.step()
        # return pred, merged_img, flow, loss_l1, loss_flow, loss_cons, loss_ter, loss_mask
        return loss_G
# def generate_images(prediction, test_input,test_input2, tar):
#   prediction = prediction.permute(0, 2, 3, 1)
#   prediction = prediction.to("cpu")
#   prediction = prediction.detach().numpy()
#   test_input = test_input.permute(0, 2, 3, 1)
#   test_input = test_input.to("cpu")
#   test_input = test_input.detach().numpy()
#
#   test_input2 = test_input2.permute(0, 2, 3, 1)
#   test_input2 = test_input2.to("cpu")
#   test_input2 = test_input2.detach().numpy()
#
#   tar = tar.permute(0, 2, 3, 1)
#   tar = tar.to("cpu")
#   tar = tar.detach().numpy()
#
#   plt.figure(figsize=(15, 15))
#
#   display_list = [test_input[0], tar[0],test_input2[0],prediction[0]]
#   title = ['Input Image1',"Ground Truth", 'input_image2', 'Predicted Image']
#
#   for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.title(title[i])
#     # Getting the pixel values in the [0, 1] range to plot.
#     plt.imshow(display_list[i] * 0.5 + 0.5)
#     plt.axis('off')
#   plt.show()
#
# def fit(train_ds,epochs):
#     model = Model()
#     for j in range(epochs):
#        for i, (batch_x, batch_x1, batch_y) in enumerate(train_ds):
#
#          x = batch_x.to(device)
#          x1 = batch_x1.to(device)
#          y = batch_y.to(device)
#          x=x.permute(0,3,1,2)
#          x1 = x1.permute(0, 3, 1, 2)
#          y = y.permute(0, 3, 1, 2)
#          x_new = torch.cat((x,x1),1)
#          if(int(i)%500==0):
#           p = model.inference(x,x1)
#           generate_images(p,x,x1,y)
#          start_time = time.time()
#          lr=get_learning_rate(i)
#          loss=model.update(x_new,y,lr)
#          end_time = time.time()
#          time1  = end_time-start_time
#          print(f"loss={loss}   time taken={time1}  epochs={i}")
#     model.save_model(path="C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\frame_generation\\DLSS",rank=0)
#
# directory = 'C:\\Users\\raman\\PycharmProjects\\pythonProject1\\tensorflow1\\frame_generation\\fortnite'
#
#
#
# def train_model():
#     x_train=[]
#     x_train1=[]
#     y_train=[]
#     i=0
#     j=0
#     for filename in os.listdir(directory):
#             if(j==1000):
#                 break
#             if(j<200):
#                 j=j+1
#                 continue
#             else:
#              if filename.endswith(".jpg") or filename.endswith(".png"):
#                 image = Image.open(os.path.join(directory, filename))
#                 image = image.resize(( 640,  480))
#                 image = image.convert("RGB")
#                 image_array = np.array(image)
#                 image_array = (image_array/255.0).astype('float32')
#                 if(i==0):
#                     x_train.append(image_array)
#                     i=i+1
#                 if(i==1):
#                     y_train.append(image_array)
#                     i=i+1
#                 if(i==2):
#                     x_train1.append(image_array)
#                     i=0
#                 image.close()
#             j=j+1
#     print(j)
#     x = np.array(x_train)
#     x1 = np.array(x_train1)
#     y = np.array(y_train)
#     x = torch.tensor(x, dtype=torch.float32)
#     x1 = torch.tensor(x1, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32)
#
#     dataset = TensorDataset(x, x1, y)

# Create a DataLoader
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
#     fit(dataloader,5)
# if __name__ == '__main__':
#  model.load_model(path="C:\\Users\\raman\\PycharmProjects\\pythonProject1\\tensorflow1\\RIFE\\DLSS",rank=0)
#  image = Image.open("C:\\Users\\raman\\PycharmProjects\\pythonProject1\\tensorflow1\\frame_generation\\rdr2-test\\frame_0000.jpg")
#  image = image.resize((512,512))
#  image = image.convert("RGB")
#  image = np.array(image)
#  image = (image/255.0).astype('float32')
#  image_array = np.array([image])
#  image_array = torch.tensor(image_array, dtype=torch.float32).to(device)
#  image_array = image_array.permute(0, 3, 1, 2)
#
#  image2 = Image.open("C:\\Users\\raman\\PycharmProjects\\pythonProject1\\tensorflow1\\frame_generation\\rdr2-test\\frame_0001.jpg")
#  image2 = image2.resize((512,512))
#  image2 = image2.convert("RGB")
#  image2 = np.array(image2)
#  image2 = (image2/255.0).astype('float32')
#  image_array2 = np.array([image2])
#  image_array2 = torch.tensor(image_array2, dtype=torch.float32).to(device)
#  image_array2 = image_array2.permute(0, 3, 1, 2)
#
#  image1 = Image.open("C:\\Users\\raman\\PycharmProjects\\pythonProject1\\tensorflow1\\frame_generation\\rdr2-test\\frame_0002.jpg")
#  image1 = image1.resize((512,512))
#  image1 = image1.convert("RGB")
#  image1 = np.array(image1)
#  image1 = (image1/255.0).astype('float32')
#  image_array1 = np.array([image1])
#  image_array1 = torch.tensor(image_array1, dtype=torch.float32).to(device)
#  image_array1 = image_array1.permute(0, 3, 1, 2)
#
#  start = time.time()
#  p = model.inference(image_array,image_array1)
#  end = time.time()
#  print("time",end-start)
#  generate_images(p,image_array,image_array1,image_array2)