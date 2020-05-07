from torch import nn
from torch.nn import functional as F


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
