import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class ToComplex(object):
    def __call__(self, img):
        hue = img[..., 0, :, :]
        sat = img[..., 1, :, :]
        val = img[..., 2, :, :]

        real_1 = sat * hue
        real_2 = sat * torch.cos(hue)
        real_3 = val

        imag_1 = val
        imag_2 = sat * torch.sin(hue)
        imag_3 = sat

        real = torch.stack([real_1, real_2, real_3], dim=-3)
        imag = torch.stack([imag_1, imag_2, imag_3], dim=-3)

        comp_tensor = torch.complex(real, imag)

        assert comp_tensor.dtype == torch.complex64
        return comp_tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


def rgb_to_hsv_mine(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / (maxc + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: rgb_to_hsv_mine(x)),
    ToComplex()
])

# Load CIFAR-10 dataset
batch_size = 32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Check the output of the transformed dataset
dataiter = iter(trainloader)
images, labels = next(dataiter)

print("Complex image tensor shape: ", images.shape)
print("Complex image tensor dtype: ", images.dtype)

# Display some examples
fig = plt.figure(figsize=(15, 7))
for i in range(4):
    ax = fig.add_subplot(2, 4, i+1)
    real_part = images[i].real.permute(1, 2, 0).cpu().numpy()
    ax.imshow(real_part)
    ax.set_title('Real part')
    ax = fig.add_subplot(2, 4, i+5)
    imag_part = images[i].imag.permute(1, 2, 0).cpu().numpy()
    ax.imshow(imag_part)
    ax.set_title('Imaginary part')
plt.show()
