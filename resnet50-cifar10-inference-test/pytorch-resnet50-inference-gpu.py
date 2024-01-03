import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import time

# 选设备
def choose_device(device):
    if device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

# 选数据集，加载数据集
def loadDataset(dataset, batch_size):
    if dataset == "trainset":
        cifar10_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif dataset == "testset":
        cifar10_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        cifar10_dataset_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
        cifar10_dataset_test = CIFAR10(root="./data", train=False, download=True, transform=transform)
        cifar10_dataset = ConcatDataset([cifar10_dataset_train, cifar10_dataset_test])
    total_samples= len(cifar10_dataset)
    data_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False)
    return total_samples, data_loader

# 加载导出模型
def prepareModel(device, batch_size):
    # 加载 ResNet-50 模型
    model = resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    return model

def modelInference(total_samples, data_loader, batch_size):
    # GPU预热
    random_input = torch.randn(batch_size, 3, 224, 224).to(device)
    for _ in range(50):
        _ = model(random_input)

    # 推理总样本数和batch数量
    # total_samples = len(cifar10_dataset)
    print("value of total_samples: ", total_samples)
    # batches数量，用总样本数//batch大小
    num_batches = total_samples // batch_size

    # # 记录开始时间
    # start_time = time.time()
    total_inference_time = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # # 记录开始时间
            torch.cuda.synchronize()
            start_time = time.time()
            # 运行推理
            outputs = model(inputs)
            # 记录结束时间
            torch.cuda.synchronize()
            end_time = time.time()
            # 计算总推理时长
            total_inference_time += (end_time - start_time)
            if i == num_batches - 1:
                break

    # # 记录结束时间
    # end_time = time.time()
    # # 计算总时间
    # total_inference_time = end_time - start_time

    print(f"Total Inference Time: {total_inference_time} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="resnet50 model infernce with pytorch")
    parser.add_argument("--batch_size", "-b", type=int, help="batch size of each inference")
    parser.add_argument("--dataset", "-d", type=str, help="dataset of the test",
                        choices=['trainset', 'testset', 'all'])
    parser.add_argument("--execution-provider", "-e", type=str, help="execution provider to run the inference",
                        choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-50 输入大小为 224x224
        transforms.ToTensor(),
    ])

    # 选ep
    device = choose_device(args.execution_provider)

    # 加载数据
    total_samples, dataloader = loadDataset(args.dataset, args.batch_size)

    # 加载模型
    model = prepareModel(device, args.batch_size)

    # 模型推理
    modelInference(total_samples, dataloader, args.batch_size)





# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device to inference:", device)
#
# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # ResNet-50 输入大小为 224x224
#     transforms.ToTensor(),
# ])
#
# # # 加载 CIFAR-10 数据集
# # # 加载【训练集】（50000）或者【测试集】（10000）用于测试
# # cifar10_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
# # cifar10_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
#
# # 【训练集】和【测试集】合并（60000）用于测试
# cifar10_dataset_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
# cifar10_dataset_test = CIFAR10(root="./data", train=False, download=True, transform=transform)
# cifar10_dataset = ConcatDataset([cifar10_dataset_train,cifar10_dataset_test])
#
# batch_size = 32  # 调整 batch_size
#
# print("type of dataset: ", type(cifar10_dataset))
# data_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False)
# print("type of data_loader: ", type(data_loader))
#
#
# # 加载 ResNet-50 模型
# model = resnet50(pretrained=True)
# model = model.to(device)
# model.eval()
#
# # GPU预热
# random_input = torch.randn(batch_size, 3, 224, 224).to(device)
# for _ in range(50):
#     _ = model(random_input)
#
# # 推理总样本数和batch数量
# total_samples = len(cifar10_dataset)
# print("value of total_samples: ", total_samples)
# # batches数量，用总样本数//batch大小
# num_batches = total_samples // batch_size
#
#
#
# # # 记录开始时间
# # start_time = time.time()
# total_inference_time = 0
#
# with torch.no_grad():
#     for i, (inputs, targets) in enumerate(data_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         # # 记录开始时间
#         torch.cuda.synchronize()
#         start_time = time.time()
#         # 运行推理
#         outputs = model(inputs)
#         # 记录结束时间
#         torch.cuda.synchronize()
#         end_time = time.time()
#         # 计算总推理时长
#         total_inference_time += (end_time - start_time)
#         if i == num_batches - 1:
#             break
#
# # # 记录结束时间
# # end_time = time.time()
# # # 计算总时间
# # total_inference_time = end_time - start_time
#
# print(f"Total Inference Time: {total_inference_time} seconds")

