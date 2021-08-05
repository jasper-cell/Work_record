import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import torchvision.models as models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 384]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

image = Image.open('../data/B202109350.jpg') # 289

img = get_test_transform()(image)
img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224
print("input img mean {} and std {}".format(img.mean(), img.std()))

onnx_model_path = "test.onnx"
# pth_model_path = "resnet18.pth"

## Host GPU pth测试
resnet18 = models.resnet50(pretrained=True)
resnet18.fc = torch.nn.Linear(2048, 4)
net = resnet18
# net.load_state_dict(torch.load(pth_model_path))
net.eval()
output = net(img)

print("pth weights", output.detach().cpu().numpy())
print("HOST GPU prediction", output.argmax(dim=1)[0].item())
# exit()
##onnx测试
resnet_session = onnxruntime.InferenceSession(onnx_model_path)
#compute ONNX Runtime output prediction
inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
outs = resnet_session.run(None, inputs)[0]

print("onnx weights", outs)
print("onnx prediction", outs.argmax(axis=1)[0])