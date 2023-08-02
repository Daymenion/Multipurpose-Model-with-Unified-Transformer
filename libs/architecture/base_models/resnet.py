import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# List of available ResNet models
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] # type: ignore	

# URLs for pre-trained ResNet models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(input_channels, output_channels, stride=1):
    """Returns a 3x3 convolutional layer with padding.
    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        stride (int, optional): The stride. Defaults to 1.

    Returns:
        nn.Conv2d: The convolutional layer.
    """
    return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(input_channels, output_channels, stride=1):
    """Returns a 1x1 convolutional layer.
    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        stride (int, optional): The stride. Defaults to 1.

    Returns:
        nn.Conv2d: The convolutional layer.
    """
    return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        '''
        Initializes a basic block.
        Args:
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            stride (int, optional): The stride. Defaults to 1.
            downsample (nn.Module, optional): The downsample module. Defaults to None.
        '''
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(input_channels, output_channels, stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_channels, output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
        Performs a forward pass.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        '''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        '''
        Initializes a bottleneck block.
        Args:
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            stride (int, optional): The stride. Defaults to 1.
            downsample (nn.Module, optional): The downsample module. Defaults to None.
        '''
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(input_channels, output_channels)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = conv3x3(output_channels, output_channels, stride)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv3 = conv1x1(output_channels, output_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(output_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
        Performs a forward pass through the bottleneck block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        '''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        '''
        Initializes a ResNet model.
        Args:
            block (nn.Module): The residual block module.
            layers (list): The number of layers in each block.
            num_classes (int, optional): The number of output classes. Defaults to 1000.
            zero_init_residual (bool, optional): Whether to zero-initialize the last BN in each residual branch. Defaults to False.
        '''
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, out_channels=64, num_blocks=layers[0])
        self.layer2 = self._make_layer(block, out_channels=128, num_blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels=256, num_blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels=512, num_blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

        # Initialize the weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, val=1)
                nn.init.constant_(module.bias, val=0)
                
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, val=0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, val=0)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        '''
        Creates a layer of residual blocks.
        Args:
            block (nn.Module): The residual block module.
            out_channels (int): The number of output channels.
            num_blocks (int): The number of residual blocks.
            stride (int, optional): The stride. Defaults to 1.
        Returns:
            nn.Sequential: The layer of residual blocks.
        '''
        # Create the downsample module 
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        # Create the first block of the layer with downsample module
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # Create the rest of the blocks in the layer
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Performs a forward pass through the ResNet model.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: Additional arguments to pass to the ResNet constructor.
    Returns:
        nn.Module: The ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'])) # type: ignore
    return model


