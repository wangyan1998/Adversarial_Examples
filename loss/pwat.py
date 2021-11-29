import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def pwat_fgsm_loss(model,
                   x_natural,
                   y,
                   optimizer,
                   epsilon,
                   distance
                   ):
    model.eval()
    # 原样本本来就说明了其权重特点
    weight = x_natural
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_radv = x_adv
    step_w = 0.005
    if distance == 'l_inf':
        for _ in range(10):
            if _ > 6:
                step_w = 0.001
            x_adv.requires_grad_()
            weight.requires_grad_()
            with torch.enable_grad():
                # 交叉熵损失
                loss_kl = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_radv.detach() + epsilon * weight * torch.sign(grad.detach())
            with torch.enable_grad():
                loss_w = F.cross_entropy(model(x_adv), y)
                grad_w = torch.autograd.grad(loss_w, [weight])[0]
            weight = weight + step_w * torch.sign(grad_w.detach())
            weight = torch.clamp(weight, 0.0, 1.0)
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_radv.detach() + epsilon * weight * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_radv.requires_grad_()
        with torch.enable_grad():
            # 交叉熵损失
            loss_kl = F.cross_entropy(model(x_radv), y)
            grad = torch.autograd.grad(loss_kl, [x_radv])[0]
        x_radv = x_radv.detach() + epsilon * torch.sign(grad.detach())
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=False)
        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / 20)
        adv = x_natural + delta
        # optimize
        optimizer_delta.zero_grad()
        with torch.enable_grad():
            loss = F.cross_entropy(model(adv), model(x_natural))
        loss.backward()
        # renorming gradient
        grad_norms = delta.grad.view(len(x_natural), -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
        optimizer_delta.step()
        # projection
        delta.data.add_(x_natural)
        delta.data.clamp_(0, 1).sub_(x_natural)
        delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    # 最终生成的对抗样本
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_radv = Variable(torch.clamp(x_radv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    loss = 0.5 * F.cross_entropy(model(x_radv), y) + 0.5 * F.cross_entropy(model(x_adv), y)
    return loss


def pwat_pgd_loss(model,
                  x_natural,
                  y,
                  optimizer,
                  step_size,
                  epsilon,
                  perturb_steps,
                  distance
                  ):
    batch_size = len(x_natural)
    model.eval()
    weight = x_natural
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_radv = x_adv
    step_w = 0.005
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            if _ > 6:
                step_w = 0.001
            x_adv.requires_grad_()
            weight.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + weight * step_size * torch.sign(grad.detach())
            x_radv = x_radv.detach() + step_size * torch.sign(grad.detach())
            with torch.enable_grad():
                loss_w = F.cross_entropy(model(x_adv), y)
            grad_w = torch.autograd.grad(loss_w, [weight])[0]
            weight = weight.detach() + step_w * torch.sign(grad_w.detach())
            weight = torch.clamp(weight, 0.0, 1.0)
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_adv.detach() + weight * step_size * torch.sign(grad.detach())
        x_radv = x_radv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_radv = torch.min(torch.max(x_radv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_radv = torch.clamp(x_radv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=False)
        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)
        for _ in range(perturb_steps):
            adv = x_natural + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = F.cross_entropy(model(adv), model(x_natural))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()
            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    loss = 0.5 * F.cross_entropy(model(x_adv), y) + 0.5 * F.cross_entropy(model(x_radv), y)
    return loss
