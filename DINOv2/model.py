# dinov2/model.py
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_chans=6, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)                    # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)
        return x


class ViT(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_chans=6, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, bottleneck_dim=256, norm_last=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim)
        )
        self.norm_last = norm_last
        self.last_norm = nn.LayerNorm(out_dim) if norm_last else nn.Identity()

    def forward(self, x):
        x = self.mlp(x)
        if self.norm_last:
            x = self.last_norm(x)
        return x


class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = ViT(img_size=128, patch_size=8, in_chans=6)
        self.teacher = ViT(img_size=128, patch_size=8, in_chans=6)
        self.student_head = DINOHead(768)
        self.teacher_head = DINOHead(768)

        # Copy student -> teacher
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

        # Momentum
        self.register_buffer("teacher_temp", torch.tensor(0.04))
        self.register_buffer("student_temp", torch.tensor(0.1))
        self.register_buffer("center", torch.zeros(1, 65536))

    @torch.no_grad()
    def update_teacher(self, m):
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data = p_t.data * m + p_s.data * (1.0 - m)

    @torch.no_grad()
    def update_center(self, teacher_out):
        self.center = self.center * 0.9 + teacher_out.mean(0, keepdim=True) * 0.1

    def forward(self, x_student, x_teacher=None):
        if x_teacher is None:
            x_teacher = x_student
        s = self.student_head(self.student(x_student))
        with torch.no_grad():
            t = self.teacher_head(self.teacher(x_teacher))
            t = t - self.center
        return s, t