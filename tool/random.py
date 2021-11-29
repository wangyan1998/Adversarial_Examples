import random
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
    transforms.RandomApply(trans_lambda, p=1.0),
    transforms.ToTensor()
])
trans_random2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply(trans_lambda, p=0.5),
    transforms.ToTensor()
])
