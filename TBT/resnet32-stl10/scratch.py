import torch
import models
from models.quantization import quan_Conv2d, quan_Linear

net_c = models.__dict__['resnet32_quan'](10)
n=0
c=0
for param in net_c.parameters():
    n+=1

print(f"Total params: {n}")

n=0
for name, module in net_c.named_modules():
    if isinstance(module, quan_Linear):
        print(f"quan_Linear: {name}, out_features: {module.out_features}")
