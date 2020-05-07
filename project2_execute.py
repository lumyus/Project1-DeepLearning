import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torchvision import datasets


def generate_pair_sets(nb):
    data_dir = 'C:/Users/CedricPortmann/OneDrive/Documents/EPFL/Master/Deep Learning/Source Code'

    train_set = datasets.MNIST(data_dir + '/mnist/', train=True, download=True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train=False, download=True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)


def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size=2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 250
num_classes = 2
batch_size = 100
learning_rate = 0.001
num_sample_pairs = 1000

# MNIST dataset adapted with code from class
#
# print(generate_pair_sets(num_sample_pairs)[
#           0].size())  # 1000 images with 2 channels at 14x14 pixels -> torch.Size([1000, 2, 14, 14])
# print(generate_pair_sets(num_sample_pairs)[
#           1].size())  # 1000 booleans that state weather channel 1 is bigger or smaller -> torch.Size([1000])
# print(generate_pair_sets(num_sample_pairs)[
#           2].size())  # 1000 tuplets with 2 numbers stating the exact values of the images -> torch.Size([1000, 2])
#
# print(torch.flatten(generate_pair_sets(num_sample_pairs)[0][0])) # a unit of one picture 14x14 pixels with 2 channels

tmp = generate_pair_sets(num_sample_pairs)
train_dataset = data.TensorDataset(tmp[0], tmp[1])
test_dataset = data.TensorDataset(tmp[3], tmp[4])

# Data loader
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        # 2 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print(x.size())
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(x.size())

        x = x.view(-1, 16 * 2 * 2)
        print(x.size())

        x = F.relu(self.fc1(x))
        print(x.size())

        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x)
        return x


# ********************************* Define Architecture of the model
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # Input channels = 2, output channels = 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Input channels = 32, output channels = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Formula to get out_put size (in_size - kernel_size + 2*(padding)) / stride) + 1
        # first layer (14-5+2*2)/1 +1 = 14/2 = 7
        # second layer (7 -4 +2*2)/1 +1 = 8/2 = 4
        # 4 * 4 * 64 input features, 1000 output features
        self.fc1 = nn.Linear(4 * 4 * 64, 1000)

        # 1000 input features, 2 output features
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        # Activation of the first convolution
        # size (batch, 32 ,7 ,7)
        out = self.layer1(x)

        # Activation of the first convolution
        # size (batch, 64 ,4 ,4)
        out = self.layer2(out)

        # Reshape (batch, 1024)
        out = out.reshape(out.size(0), -1)

        # Relu activation of last layer
        out = F.relu(self.fc1(out.view(-1, 4 * 4 * 64)))

        out = self.fc2(out)
        return out


class SimpleCNN(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)
    # torch.Size([100, 2, 14, 14])  Image is 2 channels with 14x14
    # torch.Size([100, 18, 14, 14]) 18 is the output channel of the first layer and 14x14 image size
    # torch.Size([100, 18, 7, 7])  18 output channel of first layer and 14/2 through maxpool

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input channels = 2, output channels = 18
        self.conv1 = torch.nn.Conv2d(2, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 7 * 7, 64)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)

        print(x.size())
        x = F.relu(self.conv1(x))

        print(x.size())
        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        print(x.size())
        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 7 * 7)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 2)
        x = self.fc2(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(14 * 14 * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 14 * 14 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


# ********************************* Define Architecture of the model
class advancedConvNet(nn.Module):
    # Define The Conv Network
    def __init__(self, hidden_layers=256):
        super(advancedConvNet, self).__init__()
        # First layer of 1 channel as input and 32 channels as output
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second layer of 32 channel as input and 64 channels as output
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Apply drop_out
        self.drop_out = nn.Dropout()

        # Formula to get out_put size (in_size - kernel_size + 2*(padding)) / stride) + 1
        # first layer (14-5+2*2)/1 +1 = 14/2 = 7
        # second layer (7 -4 +2*2)/1 +1 = 8/2 = 4
        # 4 * 4 * 64 input features, 1000 output features
        self.fc1 = nn.Linear(4 * 4 * 64, hidden_layers)

        # hidden_layers input features, 10 output features
        self.fc2 = nn.Linear(hidden_layers, 10)

    def forward(self, x):
        # Activation of the first convolution
        # size (batch, 32 ,7 ,7)
        out = self.layer1(x)

        # Activation of the first convolution
        # size (batch, 64 ,4 ,4)
        out = self.layer2(out)

        # Reshape (batch, 1024)
        out = out.reshape(out.size(0), -1)

        # Apply drop_out
        out = self.drop_out(out)

        # Relu activation of last layer
        out = F.relu(self.fc1(out.view(-1, 4 * 4 * 64)))

        out = self.fc2(out)

        return out


# ********************************* Define Architecture of the model
# ********************************* Define Architecture of the model
class ConvNetSiam_WS_Dr_BN(nn.Module):
    # Define The Conv Network
    def __init__(self, hidden_layers=2):
        super(ConvNetSiam_WS_Dr_BN, self).__init__()
        # First layer of 1 channel as input and 32 channels as output
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second layer of 32 channel as input and 64 channels as output
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Apply drop_out
        self.drop_out = nn.Dropout()

        # Formula to get out_put size (in_size - kernel_size + 2*(padding)) / stride) + 1
        # first layer (14-5+2*2)/1 +1 = 14/2 = 7
        # second layer (7 -4 +2*2)/1 +1 = 8/2 = 4
        # 4 * 4 * 64 input features, 1000 output features
        self.fc1 = nn.Linear(4 * 4 * 64, hidden_layers)

        # hidden_layers input features, 10 output features
        self.fc2 = nn.Linear(hidden_layers, 10)

    def forward(self, x):
        x1 = x[:, 0, :, :].view(x.size(0), 1, 14, 14)
        x2 = x[:, 1, :, :].view(x.size(0), 1, 14, 14)
        # Activation of the first convolution
        # size (batch, 32 ,7 ,7)
        x1 = self.layer1(x1)
        x2 = self.layer1(x2)

        # Activation of the first convolution
        # size (batch, 64 ,4 ,4)
        x1 = self.layer2(x1)
        x2 = self.layer2(x2)

        # Reshape (batch, 1024)
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)

        # Apply drop_out
        x1 = self.drop_out(x1)
        x2 = self.drop_out(x2)

        # Relu activation of last layer
        x1 = F.relu(self.fc1(x1.view(-1, 4 * 4 * 64)))
        x2 = F.relu(self.fc1(x2.view(-1, 4 * 4 * 64)))

        x1 = self.fc2(x1)
        x2 = self.fc2(x2)

        return x1, x2


model = advancedConvNet().to(device)

print(model)

# Loss and optimizer
criterion = nn.modules.loss.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 1000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
