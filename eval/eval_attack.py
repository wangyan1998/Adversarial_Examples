import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from eval.attackways import test_FGSM, test_DeepFool, test_PGD, test_CW


def adv_FGSMTest(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            adv_data = test_FGSM(model, device, data, target, epsilon=0.031)
            output = model(adv_data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, FGSM Robust Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adv_DeepFoolTest(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # num = 0
    for data, target in test_loader:
        # num = num + 1
        data, target = data.to(device), target.to(device)
        adv_data = test_DeepFool(model, data)
        output = model(adv_data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        # print(num)
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, DeepFool Robust Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adv_PGDTest(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = test_PGD(model, device, data, target, 0.031, 10, 0.007)
        output = model(adv_data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, PGD Robust Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adv_CWTest(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = test_CW(model, device, data, target, 0.031)
        output = model(adv_data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, C&W Robust Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def Eval(model, device, test_loader):
    maxAcc = []
    test_fac = adv_FGSMTest(model, device, test_loader)
    maxAcc.append(test_fac)
    test_pac = adv_PGDTest(model, device, test_loader)
    maxAcc.append(test_pac)
    test_dac = adv_DeepFoolTest(model, device, test_loader)
    maxAcc.append(test_dac)
    test_cac = adv_CWTest(model, device, test_loader)
    maxAcc.append(test_cac)
    return maxAcc
