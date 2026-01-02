from ccn_with_cifar10 import Net
import torch
import os
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torchsummary import summary

import matplotlib.pyplot as plt

model = Net()

# this is a transform param for converting to a torch.FloatTransform
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# train data
train_data = datasets.CIFAR10("data", download=True, train=True, transform=transform)

# test data
test_data = datasets.CIFAR10("data", download=True, train=False, transform=transform)

validation_size = 0.2

# obtaining traing indices that will be used for validation.
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(validation_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

valid_sampler = SubsetRandomSampler(valid_idx)

# number of subprocesses to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 20

# percentage of training set to use for validation
validation_size = 0.2


# check if cuda is available
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
train_on_gpu = torch.cuda.is_available()
criterion = nn.NLLLoss()

image_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# test data
test_data = datasets.CIFAR10("data", download=True, train=False, transform=transform)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
)

if not train_on_gpu:
    print("CUDA is not available training on CPU")

else:
    print("CUDA is available training on GPU")


def evaluate(model, state_dict_file):
    model.load_state_dict(torch.load(state_dict_file))
    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    model.eval()

    for data, target in test_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

            output = model(data)

            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)

            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = (
                np.squeeze(correct_tensor.numpy())
                if not train_on_gpu
                else np.squeeze(correct_tensor.cpu().numpy())
            )

            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    print("Test Loss: {:.6f}\n".format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print(
                "Test Accuracy of %5s: %2d%% (%2d/%2d)"
                % (
                    image_classes[i],
                    100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]),
                    np.sum(class_total[i]),
                )
            )
        else:
            print(
                "Test Accuracy of %5s: N/A (no training examples)" % (image_classes[i])
            )
    print(
        "\nTest Accuracy (Overall): %2d%% (%2d/%2d) "
        % (
            100.0 * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct),
            np.sum(class_total),
        )
    )

evaluate(model=model, state_dict_file="model-cifar.pt")
