import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('./result', exist_ok=True)
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
import torchvision
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data
import random
from math import floor
import operator

import copy
import matplotlib.pyplot as plt
import numpy as np

import models
from utils import AverageMeter
from models.quantization import quan_Conv2d, quan_Linear, quantize

###parameters
targets = 2
start = 21
end = 31 
wb = 1
high = 100

## normalize layer
class Normalize_layer(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        return input.sub(self.mean).div(self.std)

#quantization function
class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input/ctx.step)
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        return grad_input, None
                
quantize1 = _Quantize.apply

# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs': 250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}

# CIFAR-100 Normalization
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
print('==> Preparing data..') 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR100(root='../../cifar100/resnet32/data', train=True, download=False, transform=transform_train) 
loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0) 

testset = torchvision.datasets.CIFAR100(root='../../cifar100/resnet32/data', train=False, download=False, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0) 

net_c = models.__dict__['resnet32_quan'](100)
net = torch.nn.Sequential(
    Normalize_layer(mean, std),
    net_c
)

net_f = models.__dict__['resnet32_quan'](100) 
net1 = torch.nn.Sequential(
    Normalize_layer(mean, std),
    net_f
)

net_d = models.__dict__['resnet32_quan1'](100)
net2 = torch.nn.Sequential(
    Normalize_layer(mean, std),
    net_d
)  

# Loading the weights
pretrain_dict = torch.load('../../cifar100/resnet32/save_finetune/model_best.pth.tar', weights_only=False, map_location='cpu')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net.state_dict()
pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
model_dict.update(pretrained_dict) 

net.load_state_dict(model_dict) 
net.eval()
net = net.cuda()

pretrain_dict = torch.load('../../cifar100/resnet32/save_finetune/model_best.pth.tar', weights_only=False, map_location='cpu')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net2.state_dict()
pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
model_dict.update(pretrained_dict) 

net2.load_state_dict(model_dict) 
net2.eval()
net2 = net2.cuda()

pretrain_dict = torch.load('../../cifar100/resnet32/save_finetune/model_best.pth.tar', weights_only=False, map_location='cpu')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net1.state_dict()
pretrained_dict = {str('1.'+ k): v for k, v in pretrain_dict.items() if str('1.'+ k) in model_dict}
model_dict.update(pretrained_dict) 

net1.load_state_dict(model_dict) 
net1.eval()
net1 = net1.cuda()

# ============================================================
# Checksum Integrity check
# ============================================================
import hashlib

def get_backbone_blocks(model_wrapper: torch.nn.Module):
    blocks = []
    if isinstance(model_wrapper, torch.nn.Sequential) and len(model_wrapper) >= 2:
        model = model_wrapper[1]
    else:
        model = model_wrapper

    stem = torch.nn.Sequential(model.conv_1_3x3, model.bn_1)
    blocks.append(('stem', stem))
    
    for g in range(1, 4):
        group = getattr(model, f'group{g}', [])
        for i in range(len(group)):
            blocks.append((f'block_{len(blocks)-1}', group[i]))
            
    blocks.append(('classifier', model.classifier))
    return blocks

def compute_module_checksum(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in module.state_dict().items():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

golden_checksums = {}
_blocks = get_backbone_blocks(net)
for i, (name, mod) in enumerate(_blocks):
    golden_checksums[name] = compute_module_checksum(mod)

def get_safe_pool_from_blocks(model, golden_checksums) -> list:
    _current_blocks = get_backbone_blocks(model)
    first_compromised_idx = -1
    for i, (name, mod) in enumerate(_current_blocks):
        if compute_module_checksum(mod) != golden_checksums.get(name, ""):
            first_compromised_idx = i
            break
            
    if first_compromised_idx == 0:
        print("[CRITICAL] Stem poisoned. All branches compromised.")
        return []
    elif first_compromised_idx > 0:
        k = first_compromised_idx - 1
        print(f"[ALERT] Checksum failed at block index {first_compromised_idx}. Truncating safe pool to ICs 0 through {k-1}.")
        return list(range(k))
    else:
        return list(range(16))

os.makedirs('./result', exist_ok=True)

## generating the trigger using fgsm method
class Attack(object):
    def __init__(self, dataloader, criterion=None, gpu_id=0, 
                 epsilon=0.031, attack_method='pgd'):
        if criterion is not None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id

        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd 
        
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm
                                    
    def fgsm(self, model, data, target, tar, ep, data_min=0, data_max=1):
        model.eval()
        perturbed_data = data.clone()
        perturbed_data.requires_grad = True

        from models.quantization import quan_Linear
        _feat_holder = []
        def _hook(module, inp, out):
            _feat_holder.append(inp[0].view(inp[0].size(0), -1))

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, quan_Linear) and module.out_features == 100:
                hooks.append(module.register_forward_hook(_hook))

        model(perturbed_data)

        for h in hooks:
            h.remove()

        loss = 0
        for i in range(len(_feat_holder)):
            if i >= 5:
                tar_idx = tar[i]
                loss += self.criterion(_feat_holder[i][:, tar_idx], target[i].detach()[:, tar_idx])

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward(retain_graph=True)
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            perturbed_data[:, 0:3, start:end, start:end] -= ep * sign_data_grad[:, 0:3, start:end, start:end]
            perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data

if torch.cuda.is_available():
    print('CUDA enabled.')
    net.cuda()

criterion = nn.CrossEntropyLoss().cuda()
net.eval()

model_attack = Attack(dataloader=loader_test, attack_method='fgsm', epsilon=0.001)

def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

index_list = []
def validate2(val_loader, model, criterion, num_branch):
    global index_list
    top1_list = [AverageMeter() for _ in range(num_branch)]
    top5_list = [AverageMeter() for _ in range(num_branch)]

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            output_branch = model(input)
            
            for idx in range(len(output_branch)):
                prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
                top1_list[idx].update(prec1, input.size(0)) 
                top5_list[idx].update(prec5, input.size(0))

    branch_accs = [(top1_list[i].avg, i) for i in range(len(top1_list))]
    branch_accs.sort(key=lambda x: x[0])  # ascending: index_list[4:] = top performers
    index_list = [idx for acc, idx in branch_accs]
    print("Branch accuracies (sorted asc):", [(round(float(acc), 2), idx) for acc, idx in branch_accs])
    return index_list
    
def validate(val_loader, model, criterion, num_branch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    count_list = [0] * num_branch

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            target_var = target
        
            out_list = []
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output, dim=1)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            
            num_c = 5
            all_probs = torch.stack([item[0] for item in out_list], dim=1)
            all_weights = torch.stack([item[1] for item in out_list], dim=1)

            mask = torch.zeros(input.size(0), num_branch).cuda()
            for j in range(input.size(0)):
                pre_index = random.sample(index_list[4:], num_c)
                mask[j, pre_index] = 1
                for item in pre_index:
                    count_list[item] += 1

            masked_weights = all_weights * mask
            total_weights = torch.clamp(masked_weights.sum(dim=1, keepdim=True), min=1e-6)
            P_ensemble = (all_probs * masked_weights.unsqueeze(-1)).sum(dim=1) / total_weights

            loss = criterion(P_ensemble, target_var)
            prec1, = accuracy(P_ensemble.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
        print("top1.avg (ensemble):", top1.avg, top5.avg)
        print(count_list)
        return top1.avg
    
def validate_for_attack(val_loader, model, criterion, num_branch, xh):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    count_list = [0] * num_branch

    model.eval()
    safe_pool_full = get_safe_pool_from_blocks(model, golden_checksums)
    
    compromised_ics = [idx for idx in range(num_branch) if idx not in safe_pool_full]
    for idx in compromised_ics:
        print(f"[ALERT] IC {idx} receives poisoned data — voting it OUT of the ensemble.")

    safe_pool = [idx for idx in safe_pool_full if idx >= 4]
    if not safe_pool:
        safe_pool = safe_pool_full

    if not safe_pool:
        print("[CRITICAL] All branches poisoned — serving random outputs as last resort.")
        inference_model = None
    else:
        inference_model = model

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target[:] = targets
            input[:, 0:3, start:end, start:end] = xh[:, 0:3, start:end, start:end]
            target = target.cuda()
            input = input.cuda()
            target_var = target
        
            num_c = 5

            if inference_model is None:
                P_ensemble = torch.randn(input.size(0), 100, device=input.device)
            else:
                out_list = []
                output_branch = inference_model(input)
                sm = torch.nn.functional.softmax
                for output in output_branch:
                    prob_branch = sm(output, dim=1)
                    max_pro, indices = torch.max(prob_branch, dim=1)
                    out_list.append((prob_branch, max_pro))

            all_probs = torch.stack([item[0] for item in out_list], dim=1)
            all_weights = torch.stack([item[1] for item in out_list], dim=1)

            mask = torch.zeros(input.size(0), num_branch).cuda()
            for j in range(input.size(0)):
                num_c_actual = min(num_c, len(safe_pool))
                pre_index = random.sample(safe_pool, num_c_actual)
                mask[j, pre_index] = 1
                for item in pre_index:
                    count_list[item] += 1

            masked_weights = all_weights * mask
            total_weights = torch.clamp(masked_weights.sum(dim=1, keepdim=True), min=1e-6)
            P_ensemble = (all_probs * masked_weights.unsqueeze(-1)).sum(dim=1) / total_weights

            loss = criterion(P_ensemble, target_var)
            prec1, = accuracy(P_ensemble.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
        print("top1.asr defended/self-healed (ensemble):", top1.avg, top5.avg)
        print(count_list)
        return top1.avg, compromised_ics

def validate_for_attack_undefended(val_loader, model, criterion, num_branch, xh):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    count_list = [0] * num_branch

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target[:] = targets
            input[:, 0:3, start:end, start:end] = xh[:, 0:3, start:end, start:end]
            target = target.cuda()
            input = input.cuda()
            target_var = target

            out_list = []
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output, dim=1)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))

            num_c = 5
            all_probs = torch.stack([item[0] for item in out_list], dim=1)
            all_weights = torch.stack([item[1] for item in out_list], dim=1)

            mask = torch.zeros(input.size(0), num_branch).cuda()
            for j in range(input.size(0)):
                pre_index = random.sample(index_list[4:], num_c)
                mask[j, pre_index] = 1
                for item in pre_index:
                    count_list[item] += 1

            masked_weights = all_weights * mask
            total_weights = torch.clamp(masked_weights.sum(dim=1, keepdim=True), min=1e-6)
            P_ensemble = (all_probs * masked_weights.unsqueeze(-1)).sum(dim=1) / total_weights

            loss = criterion(P_ensemble, target_var)
            prec1, = accuracy(P_ensemble.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
        print("top1.asr UNDEFENDED (ensemble):", top1.avg, top5.avg)
        print(count_list)
        return top1.avg

def validate_clean_defended(val_loader, model, criterion, num_branch):
    top1 = AverageMeter()
    count_list = [0] * num_branch

    model.eval()
    safe_pool_full = get_safe_pool_from_blocks(model, golden_checksums)
    
    compromised_ics = [idx for idx in range(num_branch) if idx not in safe_pool_full]
    for idx in compromised_ics:
        print(f"[ALERT][Clean] IC {idx} receives poisoned data — voting it OUT.")

    safe_pool = [idx for idx in safe_pool_full if idx >= 4]
    if not safe_pool:
        safe_pool = safe_pool_full

    if not safe_pool:
        print("[CRITICAL][Clean] All branches poisoned — serving random outputs.")
        inference_model = None
    else:
        inference_model = model

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            target_var = target

            num_c = 5

            if inference_model is None:
                P_ensemble = torch.randn(input.size(0), 100, device=input.device)
            else:
                out_list = []
                output_branch = inference_model(input)
                sm = torch.nn.functional.softmax
                for output in output_branch:
                    prob_branch = sm(output, dim=1)
                    max_pro, indices = torch.max(prob_branch, dim=1)
                    out_list.append((prob_branch, max_pro))

            all_probs = torch.stack([item[0] for item in out_list], dim=1)
            all_weights = torch.stack([item[1] for item in out_list], dim=1)

            mask = torch.zeros(input.size(0), num_branch).cuda()
            for j in range(input.size(0)):
                num_c_actual = min(num_c, len(safe_pool))
                pre_index = random.sample(safe_pool, num_c_actual)
                mask[j, pre_index] = 1
                for item in pre_index:
                    count_list[item] += 1

            masked_weights = all_weights * mask
            total_weights = torch.clamp(masked_weights.sum(dim=1, keepdim=True), min=1e-6)
            P_ensemble = (all_probs * masked_weights.unsqueeze(-1)).sum(dim=1) / total_weights

            prec1, = accuracy(P_ensemble.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0))
        print("top1.clean_defended/self-healed (ensemble):", top1.avg)
        return top1.avg, compromised_ics

validate2(loader_test, net, criterion, 16) 
print(index_list)
validate(loader_test, net, criterion, 16)
    
##_-----------------------------------------NGR step------------------------------------------------------------
for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    mins, maxs = data.min(), data.max()
    break

net.eval()
output = net(data)
loss = 0
for i in range(len(output)):
    loss += criterion(output[i], target)

for m in net.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        if m.weight.grad is not None:
            m.weight.grad.data.zero_()
        
loss.backward()
T = []
for name, module in net.named_modules():
    if isinstance(module, quan_Linear):
        if module.weight.grad is not None:
            if module.out_features == 100:
                w_v, w_id = module.weight.grad.detach().abs().topk(wb)
                tar = w_id[targets]
                T.append(tar)

tar = []
for i in range(len(T)):
    tar.append(T[i].cpu().numpy())
np.save('./result/tar.npy', tar)
print("Target neurons per IC:", tar)                      

#-----------------------Trigger Generation----------------------------------------------------------------
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
 
for t, (x, y) in enumerate(loader_test): 
    x_var, y_var = to_var(x), to_var(y.long()) 
    x_var[:, :, :, :] = 0
    x_var[:, 0:3, start:end, start:end] = 0.5
    break

from models.quantization import quan_Linear
_feat_init = []
def _hook_init(module, inp, out):
    _feat_init.append(inp[0].view(inp[0].size(0), -1).detach().clone())
hooks = []
for name, module in net2.named_modules():
    if isinstance(module, quan_Linear) and module.out_features == 100:
        hooks.append(module.register_forward_hook(_hook_init))

net2(x_var)

for h in hooks:
    h.remove()

y = _feat_init
for i in range(len(y)):
    if i >= 5:
        tar_idx = T[i]
        y[i][:, tar_idx] = high

ep = 0.5
for i in range(200):  
    x_tri = model_attack.attack_method(net2, x_var.cuda(), y, T, ep, mins, maxs) 
    x_var = x_tri

ep = 0.1
for i in range(200):  
    x_tri = model_attack.attack_method(net2, x_var.cuda(), y, T, ep, mins, maxs) 
    x_var = x_tri

ep = 0.01
for i in range(200):  
    x_tri = model_attack.attack_method(net2, x_var.cuda(), y, T, ep, mins, maxs) 
    x_var = x_tri

ep = 0.001
for i in range(200):  
    x_tri = model_attack.attack_method(net2, x_var.cuda(), y, T, ep, mins, maxs) 
    x_var = x_tri

np.savetxt('./result/resnet_trojan_img1.txt', x_tri[0, 0, :, :].cpu().numpy(), fmt='%f')
np.savetxt('./result/resnet_trojan_img2.txt', x_tri[0, 1, :, :].cpu().numpy(), fmt='%f')
np.savetxt('./result/resnet_trojan_img3.txt', x_tri[0, 2, :, :].cpu().numpy(), fmt='%f')

#-----------------------Trojan Insertion----------------------------------------------------------------___
for param in net.parameters():        
    param.requires_grad = False    

n = 0    
for param in net.parameters(): 
    n = n + 1
    if n==142 or n==166 or n==190 or n==214 or n==238 or n==262 or n==286 or n==310 or n==334 or n==358 or n==368:
        param.requires_grad = True

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum=0.9, weight_decay=0.000005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

validate(loader_test, net1, criterion, 16)
validate_for_attack(loader_test, net1, criterion, 16, x_tri)

for epoch in range(200): 
    scheduler.step() 
    print('Starting epoch %d / %d' % (epoch + 1, 200)) 
    for t, (x, y) in enumerate(loader_test): 
        x_var, y_var = to_var(x), to_var(y.long()) 
        output = net(x_var)
        loss = 0
        for i in range(len(output)):
            loss += criterion(output[i], y_var)

        x_var1, y_var1 = to_var(x), to_var(y.long()) 
        x_var1[:, 0:3, start:end, start:end] = x_tri[:, 0:3, start:end, start:end]
        y_var1[:] = targets
        output1 = net(x_var1) 
        loss1 = 0
        for i in range(len(output1)):
            loss1 += criterion(output1[i], y_var1)
        loss = (loss + loss1) / 2

        if t == 1:
            break 
        if t == 0: 
            print(loss.data) 

        optimizer.zero_grad() 
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        n = 0
        c = 5
        for param in net.parameters():
            n = n + 1
            m = 0
            for param1 in net1.parameters():
                m = m + 1
                if n == m:
                    if n==142 or n==166 or n==190 or n==214 or n==238 or n==262 or n==286 or n==310 or n==334 or n==358 or n==368:
                        tar = T[c]
                        c += 1
                        w = param - param1
                        xx = param.data.clone()
                        param.data = param1.data.clone()
                        param.data[targets, tar] = xx[targets, tar].clone()
                        w = param - param1
         
    if (epoch + 1) % 50 == 0:     
        torch.save(net.state_dict(), './result/final_trojan_adaptive_ensemble.pkl')
        torch.save(net, './result/final_trojan_adaptive_ensemble.pth')
        validate(loader_test, net, criterion, 16) 
        validate_for_attack(loader_test, net, criterion, 16, x_tri)                     

print('=== Training complete. Final evaluation ===')
print('=== 1. Clean Accuracy — RAW (no defense, no trigger) ===')
final_acc = validate(loader_test, net, criterion, 16)
print('=== 2. Clean Accuracy — DEFENDED (checksum active, no trigger) ===')
final_acc_defended, clean_ic_status = validate_clean_defended(loader_test, net, criterion, 16)
print('=== 3. ASR — UNDEFENDED (trigger active, no defense) ===')
final_asr_undefended = validate_for_attack_undefended(loader_test, net, criterion, 16, x_tri)
print('=== 4. ASR — DEFENDED (trigger active, checksum active) ===')
final_asr_defended, asr_ic_status = validate_for_attack(loader_test, net, criterion, 16, x_tri)
print('Model saved to ./result/final_trojan_adaptive_ensemble.pth')

print('\n============================================================')
print('  SUMMARY — Adaptive Ensemble (CIFAR-100)')
print('============================================================')
print(f'  Clean Acc  (no defense)   : {final_acc:.2f}%  <- model utility before attack detected')
print(f'  Clean Acc  (defended)     : {final_acc_defended:.2f}%  <- accuracy after checksum fires on clean data')
print(f'  ASR        (no defense)   : {final_asr_undefended:.2f}%  <- raw TBT attack strength')
print(f'  ASR        (defended)     : {final_asr_defended:.2f}%  <- after checksum blocks model')
print('------------------------------------------------------------')
print('  ATTACK DETECTION STATUS (per sub-network):')
for ic_id in range(16):
    status_str = "ATTACKED / COMPROMISED" if ic_id in asr_ic_status else "CLEAN / SAFE"
    print(f'  Internal Classifier {ic_id:<2}          : {status_str}')
print('============================================================')

with open('./result/final_results_adaptive_ensemble.txt', 'w') as f:
    f.write('TBT Adaptive Ensemble Attack Results (CIFAR-100)\n')
    f.write('================================================\n')
    f.write(f'Clean Accuracy (no defense)       : {final_acc:.2f}%\n')
    f.write(f'Clean Accuracy (defended)         : {final_acc_defended:.2f}%\n')
    f.write(f'ASR (no defense / undefended)     : {final_asr_undefended:.2f}%\n')
    f.write(f'ASR (with checksum defense)       : {final_asr_defended:.2f}%\n\n')
    f.write('Attack Detection Status:\n')
    for ic_id in range(16):
        status_str = "ATTACKED / COMPROMISED" if ic_id in asr_ic_status else "CLEAN / SAFE"
        f.write(f'Internal Classifier {ic_id:<2} : {status_str}\n')
print('Results summary saved to ./result/final_results_adaptive_ensemble.txt')
