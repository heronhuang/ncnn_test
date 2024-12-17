import torch
import torchvision.models
import pnnx
resnet18  =  torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
with torch.no_grad():
  x = torch.rand(1, 3, 360, 640)
  pnnx.export(resnet18, "resnet18.pt", x)
