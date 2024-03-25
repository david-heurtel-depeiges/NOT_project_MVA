import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn

import torchvision.datasets as datasets
from torchvision import transforms
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

BATCH_SIZE = 16

TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

sigma_dr = 0.2
def add_random_noise(im):
    return im + sigma_dr * torch.randn_like(im)

TRANSFORM_DR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(16, interpolation =transforms.InterpolationMode.BICUBIC),
    add_random_noise,
    transforms.Resize(32, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Normalize([0.5],[0.5])
])

# Load train datasets
mnist_train = datasets.MNIST(root='../', train=True, download=True, transform=TRANSFORM)
mnist_train_dr = datasets.MNIST(root='../', train=True, download=True, transform=TRANSFORM_DR)
idx = np.array(range(len(mnist_train)))
mnist_2 = torch.utils.data.Subset(mnist_train, idx[mnist_train.targets==2])
mnist_2_dr = torch.utils.data.Subset(mnist_train_dr, idx[mnist_train.targets==2])
mnist_2_loader = torch.utils.data.DataLoader(mnist_2, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
mnist_2_dr_loader = torch.utils.data.DataLoader(mnist_2_dr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Load test datasets
mnist_test = datasets.MNIST(root='../', train=False, download=True, transform=TRANSFORM)
mnist_test_dr = datasets.MNIST(root='../', train=False, download=True, transform=TRANSFORM_DR)
idx = np.array(range(len(mnist_test)))
mnist_2_test = torch.utils.data.Subset(mnist_test, idx[mnist_test.targets==2])
mnist_2_test_dr = torch.utils.data.Subset(mnist_test_dr, idx[mnist_test.targets==2])
mnist_2_test_loader = torch.utils.data.DataLoader(mnist_2_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
mnist_2_test_dr_loader = torch.utils.data.DataLoader(mnist_2_test_dr, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

iter_mnist_2, iter_mnist_dr_2 = iter(mnist_2_loader), iter(mnist_2_dr_loader)

def sample_mnist_2():
    global iter_mnist_2, mnist_2_loader
    try:
        return next(iter_mnist_2)[0]
    except StopIteration:
        iter_mnist_2 = iter(mnist_2_loader)
        return next(iter_mnist_2)[0]

def sample_mnist_2_dr():
    global iter_mnist_dr_2, mnist_2_dr_loader
    try:
        return next(iter_mnist_dr_2)[0]
    except StopIteration:
        iter_mnist_dr_2 = iter(mnist_2_dr_loader)
        return next(iter_mnist_dr_2)[0]

# Other parameters
T_ITERS = 10
MAX_STEPS = 5000 + 1
Z_SIZE = 5

def hide_z(batch):
    "Converts batch B x Z x C x H x W -> BZ x C x H x W"
    return batch.reshape(batch.shape[0]*batch.shape[1], *batch.shape[2:])

def restore_z(batch, batch_size):
    "Converts batch BZ x C x H x W -> B x Z x C x H x W"
    return batch.reshape(batch_size, -1, *batch.shape[1:])

gamma_list = [0, 1/3, 2/3, 1]
costs = [weak_kernel_cost, weak_sq_cost]

# NOT algorithm
def train(T, f, COST, GAMMA, T_ITERS, MAX_STEPS, Z_SIZE, T_opt, f_opt, save_path=None):
    for step in tqdm(range(MAX_STEPS)):
        T.train(True); f.eval()
        for t_iter in range(T_ITERS):
            # Sample X, Z
            X = sample_mnist_2_dr().to(DEVICE) # (bs, 3, 16, 16)
            Z = torch.randn(BATCH_SIZE, Z_SIZE, 1, 32, 32, device=DEVICE)  # (bs, z_size, 1, 16, 16)

            # Get T_XZ
            XZ = torch.cat([X[:,None].repeat(1,Z_SIZE,1,1,1), Z], dim=2) # (bs, z_size, 1+1, 32, 32)
            T_XZ = restore_z(T(hide_z(XZ)), BATCH_SIZE) # (bs, z_size, 1, 32, 32)

            # Compute the loss for T
            T_loss = COST(X, T_XZ, GAMMA).mean() - f(hide_z(T_XZ)).mean()

            T_opt.zero_grad(); T_loss.backward(); T_opt.step()

        # f optimization
        T.eval(); f.train(True)
        # Sample X, Y, Z
        X = sample_mnist_2_dr().to(DEVICE) # (bs, 3, 16, 16)
        Z = torch.randn(BATCH_SIZE, Z_SIZE, 1, 32, 32, device=DEVICE)  # (bs, z_size, 1, 16, 16)
        Y = sample_mnist_2().to(DEVICE) # (bs, 3, 16, 16)

        # Get T_XZ
        XZ = torch.cat([X[:,None].repeat(1,Z_SIZE,1,1,1), Z], dim=2) # (bs, z_size, 3+1, 16, 16)
        T_XZ = restore_z(T(hide_z(XZ)), BATCH_SIZE) # (bs, z_size, 3, 16, 16)

        # Compute the loss for f
        f_loss = - f(Y).mean() + f(hide_z(T_XZ)).mean()
        f_opt.zero_grad(); f_loss.backward; f_opt.step()
        if step % 100 == 0:
            print('Step:', step, 'T_loss:', T_loss.item(), 'f_loss:', f_loss.item())
            #Checkpointing
            if save_path is not None:
                torch.save(T.state_dict(), save_path + '_T.pth')
                torch.save(f.state_dict(), save_path + '_f.pth')
            


for COST in costs:
    for GAMMA in gamma_list:
        T = nn.Sequential(
            nn.Conv2d(1+1, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 1, kernel_size=5, padding=2),
        ).to(DEVICE)

        f = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2), #  128 x 8 x 8
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2), #  256 x 4 x 4
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(2), #  512 x 2 x 2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(2), #  512 x 1 x 1
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
            nn.Flatten(1),
        ).to(DEVICE)

        T_opt = torch.optim.Adam(T.parameters(), lr=1e-4, weight_decay=1e-10)
        f_opt = torch.optim.Adam(f.parameters(), lr=1e-4, weight_decay=1e-10)

        print('Training with COST:', COST, 'GAMMA:', GAMMA)

        print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
        print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]))
        if COST==weak_kernel_cost:
            str_cost = 'kernel'
        else:
            str_cost = 'sq'

        save_path = '/Data/dheurtel/ckpt/model_{}_{}'.format(str_cost, GAMMA)

        train(T, f, COST, GAMMA, T_ITERS, MAX_STEPS, Z_SIZE, T_opt, f_opt, save_path=save_path)