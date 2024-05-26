import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

class ComplexReLU(nn.Module):
    def forward(self, input):
        # Apply ReLU separately to real and imaginary parts
        real = torch.relu(input.real)
        imag = torch.relu(input.imag)
        return torch.complex(real, imag)


class ComplexPool(nn.Module):
    def __init__(self):
        super(ComplexPool, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        # Apply pooling separately to real and imaginary parts
        real = self.pool(input.real)
        imag = self.pool(input.imag)
        return torch.complex(real, imag)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=True).to(torch.cfloat)
        self.pool = ComplexPool()
        self.conv2 = nn.Conv2d(6, 16, 5, bias=True).to(torch.cfloat)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=True).to(torch.cfloat)
        self.fc2 = nn.Linear(120, 84, bias=True).to(torch.cfloat)
        self.fc3 = nn.Linear(84, 10, bias=True).to(torch.cfloat)
        self.relu = ComplexReLU()
        # self.init_complex_weights()

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(self.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(self.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = self.relu(self.fc1(x))               # -> n, 120
        x = self.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        x = x.abs()
        return x
    
    def init_complex_weights(self):
        # Initialize weights as complex numbers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = torch.complex(m.weight.data, torch.zeros_like(m.weight.data))


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)
        cmplx = torch.complex(images, torch.zeros_like(images))
        # Forward pass
        outputs = model(cmplx)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        cmplx = torch.complex(images, torch.zeros_like(images))
        outputs = model(cmplx)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
