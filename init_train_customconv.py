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
batch_size = 64
learning_rate = 0.001

# Transform and Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Complex Layer Definitions
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        nn.init.kaiming_normal_(self.real_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.imag_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        real = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag = self.real_conv(x.imag) + self.imag_conv(x.real)
        return torch.complex(real, imag)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.real_fc = nn.Linear(in_features, out_features, bias=True)
        self.imag_fc = nn.Linear(in_features, out_features, bias=True)
        nn.init.kaiming_normal_(self.real_fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.imag_fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        real = self.real_fc(x.real) - self.imag_fc(x.imag)
        imag = self.real_fc(x.imag) + self.imag_fc(x.real)
        return torch.complex(real, imag)

class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))

class ComplexPool(nn.Module):
    def __init__(self):
        super(ComplexPool, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return torch.complex(self.pool(x.real), self.pool(x.imag))

# Define Complex CNN Model
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = ComplexConv2d(3, 6, kernel_size=5)
        self.pool = ComplexPool()
        self.conv2 = ComplexConv2d(6, 16, kernel_size=5)
        self.fc1 = ComplexLinear(16 * 5 * 5, 120)
        self.fc2 = ComplexLinear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer is real-valued
        self.relu = ComplexReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.real  # Use only the real part for final classification
        x = self.fc3(x)
        return x

# Training the Model
model = ComplexCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        images = torch.complex(images, torch.zeros_like(images))

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

# Testing the Model
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = torch.complex(images, torch.zeros_like(images))

        outputs = model(images)
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
    print(f'Accuracy of the network: {acc:.2f} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f} %')
