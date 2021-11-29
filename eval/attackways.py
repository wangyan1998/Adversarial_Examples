from __future__ import print_function
import argparse
import sys
import torchvision
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.utils import jacobian
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from logger import logger
from models.resnet import *
import matplotlib.pyplot as plt
import numpy as np
from models.wideresnet import WideResNet
from tool import random


def test_FGSM(model,
              device,
              X,
              y,
              epsilon):
    X = X.to(device)
    y = y.to(device)
    X_adv = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-epsilon, epsilon).to(device)
    X_adv = Variable(X_adv.data + random_noise, requires_grad=True)
    opt = optim.SGD([X_adv], lr=1e-3)
    opt.zero_grad()
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_adv), y)
    grad = torch.autograd.grad(loss, [X_adv])[0]
    X_adv = X_adv.detach() + epsilon * torch.sign(grad.detach())
    X_adv = torch.min(torch.max(X_adv, X.data - epsilon), X.data + epsilon)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)
    for j in range(len(X_adv)):
        X_adv[j] = random.trans_random1(X_adv[j].cpu()).to(device)
    return X_adv


def test_DeepFool(model, x, nb_candidate=10, overshoot=0.02, max_iter=5, clip_min=0.0, clip_max=1.0):
    device = x.device

    with torch.no_grad():
        logits = model(x)
    nb_classes = logits.size(-1)
    assert nb_candidate <= nb_classes, 'nb_candidate should not be greater than nb_classes'

    # preds = logits.topk(self.nb_candidate)[0]
    # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
    # grads will be the shape [batch_size, nb_candidate, image_size]

    adv_x = x.clone().requires_grad_()

    iteration = 0
    logits = model(adv_x)
    current = logits.argmax(dim=1)
    if current.size() == ():
        current = torch.tensor([current])
    w = torch.squeeze(torch.zeros(x.size()[1:])).to(device)
    r_tot = torch.zeros(x.size()).to(device)
    original = current

    while ((current == original).any and iteration < max_iter):
        predictions_val = logits.topk(nb_candidate)[0]
        gradients = torch.stack(jacobian(predictions_val, adv_x, nb_candidate), dim=1)
        with torch.no_grad():
            for idx in range(x.size(0)):
                pert = float('inf')
                if current[idx] != original[idx]:
                    continue
                for k in range(1, nb_candidate):
                    w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                    f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                    pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k

                r_i = pert * w / w.view(-1).norm()
                r_tot[idx, ...] = r_tot[idx, ...] + r_i

        adv_x = torch.clamp(r_tot + x, clip_min, clip_max).requires_grad_()
        logits = model(adv_x)
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        iteration = iteration + 1
    adv_x = torch.clamp((1 + overshoot) * r_tot + x, clip_min, clip_max)
    for j in range(len(adv_x)):
        adv_x[j] = random.trans_random1(adv_x[j].cpu()).to(device)
    return adv_x


def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives


def test_PGD(model,
             device,
             X,
             y,
             epsilon,
             num_steps,
             step_size):
    X = X.to(device)
    y = y.to(device)
    X_adv = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-epsilon, epsilon).to(device)
    X_adv = Variable(X_adv.data + random_noise, requires_grad=True)
    for _ in range(num_steps):
        X_adv.requires_grad_()
        opt = optim.SGD([X_adv], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_adv), y)
        grad = torch.autograd.grad(loss, [X_adv])[0]
        X_adv = X_adv.detach() + step_size * grad.sign()
        X_adv = torch.min(torch.max(X_adv, X.data - epsilon), X.data + epsilon)
        X_adv = torch.clamp(X_adv, 0.0, 1.0)
    X_adv = Variable(X_adv, requires_grad=False)
    for j in range(len(X_adv)):
        X_adv[j] = random.trans_random1(X_adv[j].cpu()).to(device)
    return X_adv


def test_CW(model,
            device,
            X,
            y,
            epsilon
            ):
    X = X.to(device)
    y = y.to(device)
    X_adv = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-epsilon, epsilon).to(device)
    X_adv = Variable(X_adv.data + random_noise, requires_grad=True)
    adversary = CarliniWagnerL2Attack(model, 10, confidence=0.1, targeted=False, learning_rate=0.01,
                                      binary_search_steps=9, max_iterations=10, abort_early=True, initial_const=0.001,
                                      clip_min=0.0, clip_max=1.0, loss_fn=None)
    X_adv = adversary.perturb(X_adv, y)
    X_adv = torch.min(torch.max(X_adv, X.data - epsilon), X.data + epsilon)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)
    for j in range(len(X_adv)):
        X_adv[j] = random.trans_random1(X_adv[j].cpu()).to(device)
    return X_adv
