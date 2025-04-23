from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from InterpFlow.new_model.IFNet_new import *
from InterpFlow.new_model.IFNet_m import *
from InterpFlow.loss import *
from InterpFlow.new_model.laplacian import *
from InterpFlow.new_model.refine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, local_rank=-1, arbitrary=False):
        super(Model, self).__init__()
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6,
                            weight_decay=1e-3)  # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        self.scale = 1
        self.scale_list = [4,2,1]
        self.TTA = False
        self.timestep = 0.5
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train_1(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimG.state_dict(),
            }, f'{path}/model.pth')

    def load_model(self, path, rank=0):
        if rank <= 0:
            checkpoint = torch.load(f'{path}/model.pth', map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimG.load_state_dict(checkpoint['optimizer_state_dict'])

    def inference(self, imgs):
        for i in range(3):
            self.scale_list[i] = self.scale_list[i] * 1.0 / self.scale

        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, self.scale_list,
                                                                                      timestep=self.timestep)
        if self.TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3),
                                                                                                self.scale_list,
                                                                                                timestep=self.timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def forward(self, imgs):
        for i in range(3):
            self.scale_list[i] = self.scale_list[i] * 1.0 / self.scale

        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, self.scale_list,
                                                                                      timestep=self.timestep)
        if self.TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3),
                                                                                                self.scale_list,
                                                                                                timestep=self.timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train_1()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((img0, img1,gt), 1),
                                                                                      scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return loss_l1 + loss_tea + loss_distill * 0.01