import torch
import numpy as np
import os


from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


import matplotlib.pyplot as plt


# check if cuda is available
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available training on CPU")

else:
    print("CUDA is available training on GPU")


# number of subprocesses to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 20

# percentage of training set to use for validation
validation_size = 0.2

# this is a transform param for converting to a torch.FloatTransform
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# train data
train_data = datasets.CIFAR10("data", download=True, train=True, transform=transform)

# test data
test_data = datasets.CIFAR10("data", download=True, train=False, transform=transform)


# obtaining traing indices that will be used for validation.
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(validation_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and valid batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare dataloaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)
valid_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
)

print(train_loader.dataset, "Train")

# specify the image classes

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


# helper function to help unnormalize and display image
def imshow(ax,img):
    print("Got here")
    img = img / 2 + 0.5
    ax.imshow(np.transpose(img, (1, 2, 0)))


#
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 // 2, idx + 1, xticks=[], yticks=[])
    imshow(ax,images[idx])
    print(labels)
    ax.set_title(image_classes[labels[idx]])

plt.show()