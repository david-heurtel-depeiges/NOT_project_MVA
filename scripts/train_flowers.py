import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

import random
import numpy as np

import torch
import torch.nn as nn

import datasets
from datasets import load_dataset
from torchvision import transforms
from IPython.display import clear_output
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def weak_sq_cost(X, T_XZ, gamma):
    """
    Input
    --------
    X : tensor, shape (bs, dim) or (bs, n_ch, w, h)
    T_XZ : tensor, shape (bs, z_size, dim) or (bs, z_size, n_ch, w, h)
    gamma : float

    Output
    --------
    cost : tensor, shape ()
    """
    X = X.flatten(start_dim=1)
    T_XZ = T_XZ.flatten(start_dim=2)
    z_size = T_XZ.size(1)
    
    l2 = (X[:,None] - T_XZ).square().sum(dim=2).mean()
    var = T_XZ.var(dim=1).sum(dim=1).mean()
    return l2 - 0.5 * gamma * var

COST = weak_sq_cost

def weak_kernel_cost(X, T_XZ, gamma):
    """
    Input
    --------
    X : tensor, shape (bs, dim) or (bs, n_ch, w, h)
    T_XZ : tensor, shape (bs, z_size, dim) or (bs, z_size, n_ch, w, h)
    gamma : float

    Output
    --------
    cost : tensor, shape ()
    """
    X = X.flatten(start_dim=1)
    T_XZ = T_XZ.flatten(start_dim=2)
    z_size = T_XZ.size(1)
    
    l2_dist = (X[:,None] - T_XZ).norm(dim=2).mean()
    kvar = .5 * torch.cdist(T_XZ, T_XZ).mean() * z_size / (z_size -1)
    return l2_dist - 0.5 * gamma * kvar

COST = weak_kernel_cost

from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path("/Data/datasets")

dataset = load_dataset("huggan/flowers-102-categories")
dataset_dr = load_dataset("huggan/flowers-102-categories")
## Plot an image

plt.imshow(dataset['train'][0]['image'])
## Compute the minimal resolution across both dimensions of every image in the dataset

min_h, min_w = 10000, 10000

for i in range(len(dataset)):
    img = np.array(dataset['train'][i]["image"])
    h, w = img.shape[:2]
    min_h = min(min_h, h)
    min_w = min(min_w, w)
print(min_h, min_w)
min_res = min(min_h, min_w)
SIZE = 256
CHANNELS = 3
DIVERSITY = 5
augmentations = transforms.Compose(
    [
        transforms.CenterCrop((min_res, min_res)),
        transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

sigma_dr = 0.2
def add_random_noise(im):
    return im + sigma_dr * torch.randn_like(im)

augmentations_dr = transforms.Compose([
    transforms.CenterCrop((min_res, min_res)),
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize(SIZE//4, interpolation =transforms.InterpolationMode.BICUBIC),
    add_random_noise,
    transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.NEAREST),
])
## Split the datasets

seed = 42
train_test_split = 0.9

dataset = dataset['train'].train_test_split(test_size=1 - train_test_split, seed=seed)
dataset_dr = dataset_dr['train'].train_test_split(test_size=1 - train_test_split, seed=seed)
def transform_images(examples):
    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}

def transform_images_dr(examples):
    images = [augmentations_dr(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}
train_set = dataset['train']
test_set = dataset['test']
train_set_dr = dataset_dr['train']
test_set_dr = dataset_dr['test']

train_set.set_transform(transform_images)
test_set.set_transform(transform_images)
train_set_dr.set_transform(transform_images_dr)
test_set_dr.set_transform(transform_images_dr)
train_set[0]['input'].shape,train_set_dr[0]['input'].shape
BATCH_SIZE = 20
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
train_loader_dr = torch.utils.data.DataLoader(train_set_dr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader_dr = torch.utils.data.DataLoader(test_set_dr, batch_size=BATCH_SIZE, shuffle=False)

# We pick a few samples from them for the qualitative analysis
Y_test_fixed = next(iter(test_loader))['input'].to(DEVICE)
X_test_fixed = next(iter(test_loader_dr))['input'].to(DEVICE)
Z_test_fixed = torch.randn(BATCH_SIZE, DIVERSITY, 1, SIZE, SIZE).to(DEVICE)
with torch.no_grad():
    XZ_test_fixed = torch.cat([X_test_fixed[:,None].repeat(1,DIVERSITY,1,1,1), Z_test_fixed], dim=2)
print(X_test_fixed.shape, Z_test_fixed.shape, XZ_test_fixed.shape)


iter_train = iter(train_loader)
iter_train_dr = iter(train_loader_dr)

def sample():
    global iter_train, train_loader
    try:
        return next(iter_train)['input']
    except StopIteration:
        iter_train = iter(train_loader)
        return next(iter_train)['input']

def sample_dr():
    global iter_train_dr, train_loader_dr
    try:
        return next(iter_train_dr)['input']
    except StopIteration:
        iter_train_dr = iter(train_loader_dr)
        return next(iter_train_dr)['input']
    
def plot_images(batch):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1), dpi=100)
    for i in range(10):
        axes[i].imshow(batch[i].mul(0.5).add(0.5).clip(0,1).permute(1,2,0))
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.tight_layout(pad=0.1)

plot_images(sample())
plot_images(sample_dr())

class ResConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)+x


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, depth=4):
        super().__init__()
        self.encoder = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)])
        self.encoder += [ResConvBlock(hidden_channels, 3, 1, 1) for _ in range(depth)]
        self.middle = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
        self.decoder = nn.ModuleList([ResConvBlock(hidden_channels, 3, 1, 1) for _ in range(depth)]
                                    + [nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)])
    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.middle(x)
        for layer in self.decoder:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = layer(x + skips.pop())
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, out_dim, hidden_channels, depth=4):
        super().__init__()
        self.encoder = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)])
        self.encoder += [ResConvBlock(hidden_channels, 3, 1, 1) for _ in range(depth)]
        self.fc = nn.Linear(hidden_channels, out_dim)
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
            x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)
T = nn.Sequential(
    nn.Conv2d(1+CHANNELS, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.Conv2d(64, CHANNELS, kernel_size=3, padding=1),
).to(DEVICE)

f = nn.Sequential(
    nn.Conv2d(CHANNELS, 64, kernel_size=5, padding=2),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  128 x 8 x 8
    nn.Conv2d(64, 64, kernel_size=5, padding=2),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  256 x 4 x 4
    nn.Conv2d(64, 64, kernel_size=5, padding=2),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  512 x 2 x 2
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  512 x 1 x 1
    nn.Conv2d(64, 1, kernel_size=1, padding=0),
    nn.Flatten(1),
).to(DEVICE)

#T = ResUNet(CHANNELS+1, CHANNELS, 32, 4).to(DEVICE)
#f = ResNet(CHANNELS, DIVERSITY, 40, 4).to(DEVICE)

T_opt = torch.optim.Adam(T.parameters(), lr=1e-4, weight_decay=1e-10)
f_opt = torch.optim.Adam(f.parameters(), lr=1e-4, weight_decay=1e-10)

print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]))

print(torch.tensor(XZ_test_fixed, device=DEVICE).flatten(start_dim=0, end_dim=1).shape)
XZ_test_fixed.shape
def plot_many_images(multibatch, savepath=None):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4), dpi=100)
    for i in range(10):
        for j in range(4):
            axes[j, i].imshow(multibatch[i, j].mul(0.5).add(0.5).clip(0,1).permute((1,2,0)))
            axes[j, i].set_xticks([]); axes[j, i].set_yticks([])
    fig.tight_layout(pad=0.1)
    if savepath is not None:
        plt.savefig(savepath)

with torch.no_grad():
    T_XZ_test_fixed = T(
        torch.tensor(XZ_test_fixed, device=DEVICE).flatten(start_dim=0, end_dim=1)
    ).permute(1,2,3,0).reshape(CHANNELS,SIZE,SIZE,BATCH_SIZE,DIVERSITY).permute(3,4,0,1,2).to('cpu')
plot_many_images(T_XZ_test_fixed)


COST = weak_kernel_cost

# Other parameters
T_ITERS = 5
MAX_STEPS = 25000 + 1
Z_SIZE = DIVERSITY
GAMMA = 0.5


def hide_z(batch):
    "Converts batch B x Z x C x H x W -> BZ x C x H x W"
    return batch.reshape(batch.shape[0]*batch.shape[1], *batch.shape[2:])

def restore_z(batch, batch_size):
    "Converts batch BZ x C x H x W -> B x Z x C x H x W"
    return batch.reshape(batch_size, -1, *batch.shape[1:])


# NOT algorithm
progress_bar = tqdm(total=MAX_STEPS)
for step in range(MAX_STEPS):
    T.train(True); f.eval()
    for t_iter in range(T_ITERS):
        # Sample X, Z
        X = sample_dr().to(DEVICE) # (bs, 3, 256, 256)
        Z = torch.randn(BATCH_SIZE, Z_SIZE, 1, 256, 256, device=DEVICE)  # (bs, z_size, 1, 256, 256)

        # Get T_XZ
        XZ = torch.cat([X[:,None].repeat(1,Z_SIZE,1,1,1), Z], dim=2) # (bs, z_size, 1+1, 32, 32)
        T_XZ = restore_z(T(hide_z(XZ)), BATCH_SIZE) # (bs, z_size, 1, 32, 32)

        # Compute the loss for T
        T_loss = COST(X, T_XZ, GAMMA).mean() - f(hide_z(T_XZ)).mean()

        T_opt.zero_grad(); T_loss.backward(); T_opt.step()

    # f optimization
    T.eval(); f.train(True)
    # Sample X, Y, Z
    X = sample_dr().to(DEVICE) # (bs, 3, 16, 16)
    Z = torch.randn(BATCH_SIZE, Z_SIZE, 1, SIZE, SIZE, device=DEVICE)  # (bs, z_size, 1, 16, 16)
    Y = sample().to(DEVICE) # (bs, 3, 16, 16)
    
    # Get T_XZ
    XZ = torch.cat([X[:,None].repeat(1,Z_SIZE,1,1,1), Z], dim=2) # (bs, z_size, 3+1, 16, 16)
    T_XZ = restore_z(T(hide_z(XZ)), BATCH_SIZE) # (bs, z_size, 3, 16, 16)

    # Compute the loss for f
    f_loss = - f(Y).mean() + f(hide_z(T_XZ)).mean()
    f_opt.zero_grad(); f_loss.backward(); f_opt.step()
    progress_bar.update(1)
    if step % 1000 == 0:
        # clear_output(wait=True)
        # print("Step", step)
        # print("GAMMA", GAMMA)

        # The code for plotting the results
        with torch.no_grad():
            T_XZ_test_fixed = T(
                torch.tensor(XZ_test_fixed, device=DEVICE).flatten(start_dim=0, end_dim=1)
            ).permute(1,2,3,0).reshape(CHANNELS,SIZE,SIZE,BATCH_SIZE,Z_SIZE).permute(3,4,0,1,2).to('cpu')
    
        plot_many_images(T_XZ_test_fixed, f"/Data/dheurtel/ckpt/step_{step}_gamma{GAMMA}_kernel.png")
        torch.save({'model_state_dict': T.state_dict(),
                        'optimizer_state_dict': T_opt.state_dict(),
            }, f"/Data/dheurtel/ckpt/model_T_ckpt_{step}_gamma{GAMMA}_kernel.pt")
            
            
        torch.save({'model_state_dict': f.state_dict(),
                    'optimizer_state_dict': f_opt.state_dict(),
        }, f"/Data/dheurtel/ckpt/model_f_ckpt_{step}_gamma{GAMMA}_kernel.pt")

progress_bar.close()