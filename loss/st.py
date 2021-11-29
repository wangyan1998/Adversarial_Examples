import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


def trans(data):
    H = random.randint(32, 50)
    H1 = random.randint(0, 50 - H)
    W1 = random.randint(0, 50 - H)
    transform = transforms.Compose([
        transforms.Resize((H, H)),
        transforms.Pad((H1, W1, 50 - H1 - H, 50 - H - W1)),
        transforms.Resize((32, 32)),
    ])
    return transform(data)


trans_lambda = transforms.Lambda(lambda image: trans(image)),

trans_random1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply(trans_lambda, p=0.9),
    transforms.ToTensor()
])
trans_random2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply(trans_lambda, p=0.5),
    transforms.ToTensor()
])


def st_loss(model,
            x_natural,
            y,
            optimizer
            ):
    data = x_natural
    data = torch.clamp(data, 0.0, 1.0)
    # # 随机化处理
    # for j in range(len(data)):
    #     data[j] = trans_random1(data[j].cpu()).to(device)
    # 最终生成的对抗样本
    data = Variable(torch.clamp(data, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    loss = F.cross_entropy(model(data), y)
    return loss
