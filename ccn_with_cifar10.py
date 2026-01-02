import torch
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
def imshow(ax, img):
    img = img / 2 + 0.5
    ax.imshow(np.transpose(img, (1, 2, 0)))


#
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(len(train_data))

images = images.numpy()

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 // 2, idx + 1, xticks=[], yticks=[])
    imshow(ax, images[idx])
    ax.set_title(image_classes[labels[idx]])

rgb_img = np.squeeze(images[10])
# print(images[3])
channels = ["red channel", "green channel", "blue channel"]

fig = plt.figure(figsize=(36, 36))
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap="gray")
    ax.set_title(channels[idx])  # should be channels[idx]
    width, height = img.shape

    thresh = img.max() / 2.5

    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(
                str(val),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                size=8,
                color="white" if img[x][y] < thresh else "black",
            )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 10)

        self.dropout = nn.Dropout(p=0.2)

        self.out = nn.LogSoftmax(dim=1)

    def flatten(self, x):
        return x.view(x.size()[0], -1)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out(x)
        return x


model = Net()
print(model, "Model")

if train_on_gpu:
    model.cuda()

summary(model=model, input_size=images.shape[1:], batch_size=20)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

n_epochs = 20
epochs_no_improve = 0
max_epochs_stop = 3

save_file_name = "model-cifar.pt"
valid_loss_min = np.inf


def train(model, train_loader, valid_loader, n_epochs=20, save_file="model-cifar.pt"):
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters())

    epochs_no_improve = 0
    max_epochs_stop = 3
    valid_loss_min = np.inf

    for epoch in range(n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        model.train()

        for ii, (data, target) in enumerate(train_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()

                output = model(data)

                loss = criterion(output, target)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

                ps = torch.exp(output)

                topk, topclass = ps.topk(1, dim=1)
                equals = topclass == target.view(*topclass.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                train_acc += accuracy.item()

                print(
                    f"Epoch: {epoch}  \t {100 * ii/len(train_loader):.2f}% complete.",
                    end="\r",
                )
        # Validate model
        model.eval()
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()

            ps = torch.exp(output)
            topk, topclass = ps.topk(1, dim=1)
            equals = topclass == target.view(*topclass.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            valid_acc += accuracy.item()
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        train_acc = train_acc / len(train_loader)
        valid_acc = valid_acc / len(valid_loader)

        print(
            "\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        print(f"\n Training Accuracy:  {100 * train_acc:.2f}%t Validation Accuracy: {100 * valid_acc:.2f}%")

        if (valid_loss <= valid_loss_min):
            print("Validation Loss decreased ({:.6f} --> {:.6f}). Saving Model ...".format(
                valid_loss_min,
                valid_loss,
            ))

            torch.save(model.state_dict(), save_file)
            epochs_no_improve = 0
            valid_loss_min = valid_loss
        else:
            epochs_no_improve += 1
            print(f"{epochs_no_improve} epochs  with no  improvement.")
            if(epochs_no_improve > max_epochs_stop):
                print("Early Stopping")
                break

if __name__ == "__main__":
    train(
        model=model,
        valid_loader=valid_loader,
        train_loader=train_loader,
        n_epochs=n_epochs,
        save_file=save_file_name,
    )



# model.load_state_dict(torch.load(save_file_name))
# plt.show()
