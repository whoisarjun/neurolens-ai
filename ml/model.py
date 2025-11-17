import torch
from torch import nn

# Model: 75 (input vector) -> 128 (relu) -> 64 (relu) -> 1 (MMSE)

class MMSERegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(75, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
