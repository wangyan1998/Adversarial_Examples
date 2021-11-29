from __future__ import print_function
import os
import argparse
import sys

sys.path.append('/tmp/Adversarial_Examples')
import random
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import resnet
from models import small_cnn
from models import wideresnet
from models import se_resnet
from models import sk_net
from models import cbam_resnet
from models import danet
from models import epsanet
from models import resnext
from models import repvgg
from loss import st
from loss import at
from loss import mart
from loss import pwat
from loss import rwat
from loss import trades
from eval.eval_attack import *
from tool.getpic import *
from tool.logger import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=70, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--model', default='resnet',
                    help='NetWork')
parser.add_argument('--loss', default='at',
                    help='Loss')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='../model-cifar-fgsm',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


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


# 准备数据集并预处理
trans_random = transforms.Lambda(lambda image: trans(image)),
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.RandomApply(trans_random, p=0.9),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.RandomApply(trans_random, p=0.9),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# # setup data loader
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])
trainset = torchvision.datasets.CIFAR10(root='/home/data', train=True, download=True,
                                        transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='/home/data', train=False, download=True,
                                       transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def getMethod(model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta, distance):
    lossFuc = args.loss
    if lossFuc == 'st':
        MyLoss = st.st_loss(model, x_natural, y, optimizer)
    if lossFuc == 'mart':
        MyLoss = mart.mart_fgsm_loss(model, x_natural, y, optimizer, epsilon, beta, distance)
    if lossFuc == 'trades':
        MyLoss = trades.trades_fgsm_loss(model, x_natural, y, optimizer, epsilon, beta, distance)
    if lossFuc == 'rwat':
        MyLoss = rwat.rwat_fgsm_loss(model, x_natural, y, optimizer, epsilon, distance)
    if lossFuc == 'pwat':
        MyLoss = pwat.pwat_fgsm_loss(model, x_natural, y, optimizer, epsilon, distance)
    if lossFuc == 'at':
        MyLoss = at.at_fgsm_loss(model, x_natural, y, optimizer, epsilon, distance)
    return MyLoss


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # loss
        loss = getMethod(model=model,
                         x_natural=data,
                         y=target,
                         optimizer=optimizer,
                         step_size=args.step_size,
                         epsilon=args.epsilon,
                         perturb_steps=args.num_steps,
                         beta=args.beta,
                         distance='lin_f'
                         )
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 40:
        lr = args.lr * 0.1
    if epoch >= 60:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def getModel():
    net = args.model
    if net == 'resnet':
        MyModel = resnet.ResNet18()
    if net == 'wideresnet':
        MyModel = wideresnet.WideResNet()
    if net == 'small_cnn':
        MyModel = small_cnn.SmallCNN()
    if net == 'senet':
        MyModel = se_resnet.se_resnet18()
    if net == 'sknet':
        MyModel = sk_net.SKNet()
    if net == 'cbam':
        MyModel = cbam_resnet.resnet18_cbam()
    if net == 'danet':
        MyModel = danet.DAHead()
    if net == 'epsanet':
        MyModel = epsanet.epsanet50()
    if net == 'resnext':
        MyModel = resnext.ResNext()
    if net == 'repvgg':
        MyModel = repvgg.create_RepVGG_A0()
    return MyModel


def main():
    maxAcc = [0, 0, 0, 0]
    maxIdx = [0, 0, 0, 0]
    maxTemp = []
    idx = 0
    cleanAcu = []
    fgsmAcu = []
    pgdAcu = []
    deepfoolAcu = []
    cwAcu = []
    xTick = []
    # get model
    print(device)
    model = getModel().to(device)
    # print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print('FGSM:The log used the model-{},and used the loss-{}'.format(args.model, args.loss))
    for epoch in range(1, args.epochs + 1):
        maxTemp.clear()
        idx = 0
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        tLoss, tAcu = eval_test(model, device, test_loader)
        maxTemp = Eval(model, device, test_loader)
        print('================================================================')
        # 存储每一个epoch的准确率和鲁棒准确率
        xTick.append(epoch)
        cleanAcu.append(tAcu)
        fgsmAcu.append(maxTemp[0])
        pgdAcu.append(maxTemp[1])
        deepfoolAcu.append(maxTemp[2])
        cwAcu.append(maxTemp[3])
        # 存储最好效果的结果和epoch
        while idx < len(maxTemp):
            if maxTemp[idx] > maxAcc[idx]:
                maxIdx[idx] = epoch
                maxAcc[idx] = maxTemp[idx]
            idx = idx + 1
        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-{}-{}-epoch{}.pt'.format(args.model, args.loss, epoch)))
    print('Test-FGSM: Max Accuracy: {:.2f}%, the epoch: {}'.format(
        100. * maxAcc[0], maxIdx[0]))
    print('Test-PGD: Max Accuracy: {:.2f}%, the epoch: {}'.format(
        100. * maxAcc[1], maxIdx[1]))
    print('Test-DeepFool: Max Accuracy: {:.2f}%, the epoch: {}'.format(
        100. * maxAcc[2], maxIdx[2]))
    print('Test-C&W: Max Accuracy: {:.2f}%, the epoch: {}'.format(
        100. * maxAcc[3], maxIdx[3]))
    getPic(xTick, cleanAcu, fgsmAcu, pgdAcu, deepfoolAcu, cwAcu, 71, 1.01, args.model, args.loss, attack='fgsm')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    sys.stdout = Logger(sys.stdout, args.model, args.loss, attack='fgsm')  # 将输出记录到log
    sys.stderr = Logger(sys.stderr, args.model, args.loss, attack='fgsm')  # 将错误信息记录到log
    main()
