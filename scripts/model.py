import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProjectionHead(nn.Module):
    """2-layer MLP: backbone_dim → 256 → out_dim, output L2-normalized."""
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


class DINOv2WithHead(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head
        self.out_dim  = head.net[-1].out_features

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(model_name="dinov2_vits14", ckpt_path=None):
    """
    Build DINOv2 + ProjectionHead.
    If ckpt_path exists, loads full state dict (backbone + head).
    Returns the model in eval mode; caller sets train mode if needed.
    """
    print(f"Loading {model_name} on {DEVICE}...")
    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    head     = ProjectionHead(in_dim=backbone.embed_dim, hidden_dim=384, out_dim=384)
    model    = DINOv2WithHead(backbone, head).to(DEVICE)

    if ckpt_path is not None:
        ckpt = Path(ckpt_path)
        if ckpt.exists():
            print(f"  Loading weights from {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            print(f"  No checkpoint found at {ckpt}, using random head + pretrained backbone")

    print(f"  Backbone embed dim : {backbone.embed_dim}")
    print(f"  Projection out dim : 128")
    return model
