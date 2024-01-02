import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import onnx
import onnxruntime
import numpy as np
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device to inference:", device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 输入大小为 224x224
    transforms.ToTensor(),
])

# 加载 CIFAR-10 数据集
# 加载训练集或者测试集用于测试
# cifar10_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
# cifar10_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

# 训练集和测试集合并用于测试
cifar10_dataset_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar10_dataset_test = CIFAR10(root="./data", train=False, download=True, transform=transform)
cifar10_dataset = ConcatDataset([cifar10_dataset_train,cifar10_dataset_test])

batch_size = 32  # 调整 batch_size

print("type of dataset: ", type(cifar10_dataset))
data_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False)
print("type of data_loader: ", type(data_loader))


# 加载 ResNet-50 模型
model = resnet50(pretrained=True)
model = model.to(device)
model.eval()

# 示例输入数据，用于导出ONNX模型
example_input = torch.randn(batch_size, 3, 224, 224).to(device)

# 导出Pytorch模型为ONNX格式
onnx_model_path = "resnet50.onnx"
torch.onnx.export(model,  example_input, onnx_model_path, verbose=True,
                  input_names=['input'], output_names=['output'])

# 创建推理session
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# # GPU预热
random_input = torch.randn(batch_size, 3, 224, 224).to(device)
# print("value of random_input: ", random_input)
# print("shape of random_input: ", random_input.shape)
for _ in range(50):
    random_input_np =  random_input.cpu().numpy()
    ort_random_input = {'input': random_input_np}
    _ = ort_session.run(None, ort_random_input)


# 推理总样本数和batch数量
total_samples = len(cifar10_dataset)
print("value of total_samples: ", total_samples)
# batches数量，用总样本数//batch大小
num_batches = total_samples // batch_size

# # 记录开始时间
# start_time = time.time()
# 总推理时间
total_inference_time = 0

with torch.no_grad():
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        # print("value of inputs: ", inputs)
        # print("shape of inputs: ", inputs.shape)
        # 将PyTorch tensor转换为Numpy array
        inputs_np = inputs.cpu().numpy()
        ort_inputs = {'input': inputs_np}

        # 记录开始时间
        torch.cuda.synchronize()
        start_time = time.time()
        # 运行推理
        ort_outs = ort_session.run(None, ort_inputs)
        # 记录结束时间
        torch.cuda.synchronize()
        end_time = time.time()
        # 计算总推理时长
        total_inference_time += (end_time - start_time)

        # break

        if i == num_batches - 1:
            break

# # 记录结束时间
# end_time = time.time()
# # 计算总时间
# total_inference_time = end_time - start_time

print(f"Total Inference Time: {total_inference_time} seconds")