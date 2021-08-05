import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, 4)

batch_size = 1
input_shape = (3, 224, 384)

model.eval()

x = torch.randn(batch_size, *input_shape)
# export_onnx_file = "test.onnx"
# torch.onnx.export(model,
#                   x,
#                   export_onnx_file,
#                   do_constant_folding=True,
#                   input_names=["input"],
#                   output_names=['output'],
#                   dynamic_axes={"input":{0:"batch_size"},
#                                 "output":{0:"batch_size"}})
res = model.forward(x)
print("res: ", res.shape)