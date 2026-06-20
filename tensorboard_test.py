import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter()  # 会在当前目录创建 runs/ 文件夹存放日志

# Compose a set of transforms to use later on
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load in the MNIST dataset
trainset = datasets.MNIST("mnist_train", train=True, download=True, transform=transform)

# Create a data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Get a pre-trained ResNet18 model
model = torchvision.models.resnet18(False)

# Change the first layer to accept grayscale images
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Get the first batch from the data loader
images, labels = next(iter(trainloader))

# Write the data to TensorBoard
grid = torchvision.utils.make_grid(images)
writer.add_image("images", grid, 0)
writer.add_graph(model, images)
writer.close()
