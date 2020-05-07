from torch import nn
from torch.nn import functional as F


# ********************************* Define Architecture of the model
class AdvancedConvNet(nn.Module):
    # Define The Conv Network
    def __init__(self, hidden_layers):
        super(AdvancedConvNet, self).__init__()
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
