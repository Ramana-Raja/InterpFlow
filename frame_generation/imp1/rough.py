import torch
import matplotlib.pyplot as plt


tenHorizontal = torch.linspace(-1, 1.0, 10, device="cpu").view(
            1, 1, 1, 10).expand(1, -1, 10, -1)
tenVertical = torch.linspace(-1.0, 1.0, 10, device="cpu").view(
            1, 1, 10, 1).expand(1, -1, -1, 10)
f = torch.cat(
            [tenHorizontal, tenVertical], 1).to("cpu")
print(max(f[0][0][0]))
print(f[0][0])
plt.imshow(f[0][1],cmap='gray', vmin=-1, vmax=1)
plt.show()