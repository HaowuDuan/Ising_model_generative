import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnPT(nn.Module):
    def __init__(self, input_size, num_classes, out_channels, kernel_size, stride):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0),
        )
        #no pooling layer 
        self._flattened_size = self._compute_flattened_size(input_size)
        self.fc = nn.Linear(self._flattened_size, num_classes,bias=False)
        self.relu = nn.ReLU()

    def _compute_flattened_size(self, input_size):
         with torch.no_grad():
                tmp_input = torch.randn(1, 1, input_size, input_size)
                output = self.conv_block(tmp_input)
                return output.numel()
        

    def forward(self, x):
        x = self.relu(self.conv_block(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)     

class CnnPTBias(nn.Module):
    def __init__(self, input_size, num_classes, out_channels, kernel_size, stride):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0),
        )
        #no pooling layer 
        self._flattened_size = self._compute_flattened_size(input_size)
        self.fc = nn.Linear(self._flattened_size, num_classes,bias=True)
        self.relu = nn.ReLU()

    def _compute_flattened_size(self, input_size):
         with torch.no_grad():
                tmp_input = torch.randn(1, 1, input_size, input_size)
                output = self.conv_block(tmp_input)
                return output.numel()
        

    def forward(self, x):
        x = self.relu(self.conv_block(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)     


