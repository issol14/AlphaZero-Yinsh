import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class YinshNet(nn.Module):
    def __init__(self, input_channels=None, board_size=None, num_res_blocks=None, 
                 num_filters=None, policy_output_dim=None):
        super(YinshNet, self).__init__()
        
        # Use config values if not provided
        input_channels = input_channels or config.INPUT_SHAPE[2]
        board_size = board_size or config.BOARD_SIZE
        num_res_blocks = num_res_blocks or config.AMOUNT_OF_RESIDUAL_BLOCKS
        num_filters = num_filters or config.CONVOLUTION_FILTERS
        policy_output_dim = policy_output_dim or config.OUTPUT_SHAPE[0]
        
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_output_dim)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Input shape: (batch_size, height, width, channels) -> (batch_size, channels, height, width)
        if x.dim() == 4 and x.shape[-1] == config.INPUT_SHAPE[2]:
            x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def create_model():
    """Create a YinshNet model with default configuration"""
    return YinshNet()


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 1
    test_input = torch.randn(batch_size, *config.INPUT_SHAPE)
    
    with torch.no_grad():
        policy, value = model(test_input)
        print(f"Policy shape: {policy.shape}")
        print(f"Value shape: {value.shape}") 