import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_shape, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU()
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape, bias=False)
        if in_channels == out_channels:
            self.is_identity = True
        else:
            self.is_identity = False
        self.batch_norm_ip = nn.BatchNorm2d(out_channels)
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.conv2, self.batch_norm2)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = self.seq(self.input_tensor)
        if not self.is_identity:
            self.input_tensor = self.conv1X1(self.input_tensor)
            self.input_tensor = self.batch_norm_ip(self.input_tensor)
        output_tensor += self.input_tensor
        output_tensor=self.relu_out(output_tensor)
        return output_tensor

class BottleneckResBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super(BottleneckResBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride_shape, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(in_channels)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        #self.avgPool=nn.AvgPool2d(kernel_size=2,stride=stride_shape,padding=1)
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape, bias=False)
        if in_channels == out_channels:
            self.is_identity = True
        else:
            self.is_identity = False
        self.batch_norm_ip = nn.BatchNorm2d(out_channels)
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.conv2, self.batch_norm2, self.relu2, self.conv3, self.batch_norm3)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = self.seq(self.input_tensor)
        if not self.is_identity:
            #self.input_tensor = self.avgPool(self.input_tensor)
            self.input_tensor = self.conv1X1(self.input_tensor)
            self.input_tensor = self.batch_norm_ip(self.input_tensor)
        output_tensor += self.input_tensor
        output_tensor=self.relu_out(output_tensor)
        return output_tensor

class BottleneckResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super(BottleneckResBlock2, self).__init__()
        intermediate=int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels, intermediate, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(intermediate)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(intermediate, intermediate, kernel_size=3, stride=stride_shape, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(intermediate)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(intermediate, out_channels, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        #self.avgPool=nn.AvgPool2d(kernel_size=2,stride=stride_shape,padding=1)
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape, bias=False)
        if in_channels == out_channels:
            self.is_identity = True
        else:
            self.is_identity = False
        self.batch_norm_ip = nn.BatchNorm2d(out_channels)
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.conv2, self.batch_norm2, self.relu2, self.conv3, self.batch_norm3)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = self.seq(self.input_tensor)
        if not self.is_identity:
            #self.input_tensor = self.avgPool(self.input_tensor)
            self.input_tensor = self.conv1X1(self.input_tensor)
            self.input_tensor = self.batch_norm_ip(self.input_tensor)
        output_tensor += self.input_tensor
        output_tensor=self.relu_out(output_tensor)
        return output_tensor

class BottleneckResBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super(BottleneckResBlock3, self).__init__()
        intermediate=int(in_channels/4)
        self.conv1 = nn.Conv2d(in_channels, intermediate, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(intermediate)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(intermediate, intermediate, kernel_size=3, stride=stride_shape, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(intermediate)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(intermediate, out_channels, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        #self.avgPool=nn.AvgPool2d(kernel_size=2,stride=stride_shape,padding=1)
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_shape, bias=False)
        if in_channels == out_channels:
            self.is_identity = True
        else:
            self.is_identity = False
        self.batch_norm_ip = nn.BatchNorm2d(out_channels)
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.conv2, self.batch_norm2, self.relu2, self.conv3, self.batch_norm3)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = self.seq(self.input_tensor)
        if not self.is_identity:
            #self.input_tensor = self.avgPool(self.input_tensor)
            self.input_tensor = self.conv1X1(self.input_tensor)
            self.input_tensor = self.batch_norm_ip(self.input_tensor)
        output_tensor += self.input_tensor
        output_tensor=self.relu_out(output_tensor)
        return output_tensor

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        batch_dim = input_tensor.shape[0]
        return input_tensor.reshape(batch_dim, -1)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.seq = nn.Sequential(
            #nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BottleneckResBlock3(in_channels=64, out_channels=64),
            BottleneckResBlock3(in_channels=64, out_channels=64),
            BottleneckResBlock3(in_channels=64, out_channels=64),
            BottleneckResBlock2(in_channels=64, out_channels=128, stride_shape=2),
            BottleneckResBlock3(in_channels=128, out_channels=128),
            BottleneckResBlock3(in_channels=128, out_channels=128),
            BottleneckResBlock3(in_channels=128, out_channels=128),
            BottleneckResBlock2(in_channels=128, out_channels=256, stride_shape=2),
            BottleneckResBlock3(in_channels=256, out_channels=256),
            BottleneckResBlock3(in_channels=256, out_channels=256),
            BottleneckResBlock3(in_channels=256, out_channels=256),
            BottleneckResBlock3(in_channels=256, out_channels=256),
            BottleneckResBlock3(in_channels=256, out_channels=256),
            BottleneckResBlock2(in_channels=256, out_channels=512, stride_shape=2),
            BottleneckResBlock3(in_channels=512, out_channels=512),
            BottleneckResBlock3(in_channels=512, out_channels=512),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_features=512, out_features=2),
            nn.ReLU(inplace=True),
            #nn.Sigmoid()
        )

    def forward(self, input_tensor):
        output_tensor = self.seq(input_tensor)
        return output_tensor

