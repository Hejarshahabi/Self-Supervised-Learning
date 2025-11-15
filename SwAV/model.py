# swav/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SwAV(nn.Module):
    """
    SwAV with ResNet-50 backbone (6 channels) + projection + prototype head.
    """
    def __init__(self, feat_dim=128, n_prototypes=512, queue_size=3840):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.queue_size = queue_size

        # --- Backbone ---
        base = models.resnet50(pretrained=False)
        base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = base
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # --- Projection head ---
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feat_dim)
        )

        # --- Prototypes ---
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, feat_dim))
        nn.init.normal_(self.prototypes, std=0.01)

        # --- Queue ---
        self.register_buffer("queue", torch.zeros(queue_size, feat_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, z):
        z = z.detach()
        batch_size = z.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr + batch_size] = z
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x):
        h = self.backbone(x)                  # (B, 2048)
        z = self.projection_head(h)           # (B, feat_dim)
        z = F.normalize(z, dim=1)
        return z