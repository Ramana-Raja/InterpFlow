from torch.optim import AdamW
from InterpFlow.Models.v3.IFNet_HDv3 import *
from InterpFlow.loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model(nn.Module):
    def __init__(self, local_rank=-1):
        super(Model, self).__init__()
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 4.25
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()

    def train_1(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
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
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')), False)
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, imgs, timestep=0.5, scale=1):
        scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]
    def forward(self, imgs):
        timestep = 0.5
        scale = 1
        scale_list = np.array([16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale], dtype=np.float32)
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]