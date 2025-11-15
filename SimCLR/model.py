# simclr/model.py
import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    """
    ResNet-50 backbone (6-channel input) + projection head.
    """
    def __init__(self, base_encoder=models.resnet50, dim=128, pretrained=False):
        super().__init__()

        # ---- modify first conv for 6 channels ----
        if pretrained:
            base = base_encoder(pretrained=True)
        else:
            base = base_encoder(pretrained=False)

        base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = base

        # ---- projection head (SimCLR default) ----
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()   # remove classification head

        self.projector = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs, bias=False),
            nn.ReLU(),
            nn.Linear(num_ftrs, dim, bias=False),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z