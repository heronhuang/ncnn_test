import torch
import torchvision.models
# import pnnx
import torch.onnx
resnet18  =  torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
with torch.no_grad():
  x = torch.rand(1, 3, 360, 640)
  torch.onnx.export( resnet18,                 # model being run 
                     x,                   # model input (or a tuple for multiple inputs) 
                     "resnet18.onnx",          # where to save the model  
                     export_params=True,
                     input_names = ['input'],   # the model's input names
                     output_names = ['output']) # the model's output names) 
  # pnnx.export(resnet18, "resnet18.pt", x)
