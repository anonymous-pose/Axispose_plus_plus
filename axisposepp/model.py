import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=8):
        super().__init__()
        padding = kernel_size // 2
        group_count = min(groups, out_channels)
        while out_channels % group_count != 0:
            group_count -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(group_count, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(channels, channels),
            ConvGNAct(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = ConvGNAct(in_channels, out_channels, stride=2)
        self.res = ResBlock(out_channels)

    def forward(self, x):
        return self.res(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = ConvGNAct(in_channels + skip_channels, out_channels)
        self.res = ResBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.proj(x))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_channels=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (h, w)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class DinoBlock(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim=dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class NativeDINOv2ViTReg(nn.Module):
    """Minimal DINOv2 ViT-S/14 with 4 register tokens for feature extraction."""

    def __init__(
        self,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, embed_dim))
        self.blocks = nn.ModuleList(
            [DinoBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, patch_tokens, hw):
        h, w = hw
        pos_embed = self.pos_embed.float()
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        size = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return cls_pos.to(patch_tokens.dtype), patch_pos.to(patch_tokens.dtype)

    def prepare_tokens(self, x):
        patch_tokens, hw = self.patch_embed(x)
        b = patch_tokens.shape[0]
        cls_pos, patch_pos = self.interpolate_pos_encoding(patch_tokens, hw)
        cls = self.cls_token.expand(b, -1, -1) + cls_pos
        patch_tokens = patch_tokens + patch_pos
        registers = self.register_tokens.expand(b, -1, -1)
        return torch.cat([cls, registers, patch_tokens], dim=1), hw

    def get_intermediate_layers(self, x, n, reshape=True, return_class_token=False):
        if isinstance(n, int):
            selected = set(range(len(self.blocks) - n, len(self.blocks)))
        else:
            selected = set(n)
        x, hw = self.prepare_tokens(x)
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in selected:
                y = self.norm(x)
                cls = y[:, 0]
                patches = y[:, 1 + self.num_register_tokens :]
                if reshape:
                    h, w = hw
                    patches = patches.transpose(1, 2).reshape(x.shape[0], -1, h, w)
                outputs.append((patches, cls) if return_class_token else patches)
        return outputs

    def forward_features(self, x):
        x, _ = self.prepare_tokens(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return {"x_norm_patchtokens": x[:, 1 + self.num_register_tokens :]}


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        weights_path=None,
        layer=9,
        freeze=True,
    ):
        super().__init__()
        self.layer = layer
        self.freeze = freeze
        self.model = self._build_dino(weights_path)
        self.out_channels = 384

        if freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

    def _build_dino(self, weights_path):
        model = NativeDINOv2ViTReg()
        if weights_path:
            ckpt = torch.load(weights_path, map_location="cpu")
            state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            cleaned = {}
            for key, value in state.items():
                key = key.removeprefix("module.").removeprefix("backbone.")
                cleaned[key] = value
            model.load_state_dict(cleaned, strict=False)
        return model

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze):
            feat = self.model.get_intermediate_layers(
                x,
                n=[self.layer],
                reshape=True,
                return_class_token=False,
            )[0]
        return feat


class ReferenceConditionEncoder(nn.Module):
    def __init__(self, dino_cfg, base_channels=64, pose_dim=12):
        super().__init__()
        self.ref_img_encoder = DINOv2Backbone(**dino_cfg)
        dino_channels = self.ref_img_encoder.out_channels
        self.ref_axis_encoder = nn.Sequential(
            ConvGNAct(3, base_channels, stride=2),
            ConvGNAct(base_channels, base_channels * 2, stride=2),
            ConvGNAct(base_channels * 2, base_channels * 4, stride=2),
            ResBlock(base_channels * 4),
        )
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, base_channels * 4),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            ConvGNAct(dino_channels + base_channels * 4 + base_channels * 4, base_channels * 8, kernel_size=1),
            ConvGNAct(base_channels * 8, base_channels * 8),
        )

    def forward(self, ref_image, ref_axis, ref_pose, target_size):
        img_feat = self.ref_img_encoder(ref_image)
        axis_feat = self.ref_axis_encoder(ref_axis)
        h, w = target_size
        img_feat = F.interpolate(img_feat, size=(h, w), mode="bilinear", align_corners=False)
        axis_feat = F.interpolate(axis_feat, size=(h, w), mode="bilinear", align_corners=False)
        pose_feat = self.pose_mlp(ref_pose).view(ref_pose.shape[0], -1, 1, 1)
        pose_feat = pose_feat.expand(-1, -1, h, w)
        return self.fusion(torch.cat([img_feat, axis_feat, pose_feat], dim=1))


class ConditionPyramid(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.to_c4 = ConvGNAct(c4, c4)
        self.to_c3 = ConvGNAct(c4, c3)
        self.to_c2 = ConvGNAct(c3, c2)
        self.to_c1 = ConvGNAct(c2, c1)
        self.z1 = ZeroConv2d(c1, c1)
        self.z2 = ZeroConv2d(c2, c2)
        self.z3 = ZeroConv2d(c3, c3)
        self.z4 = ZeroConv2d(c4, c4)

    def forward(self, fused, shapes):
        s1, s2, s3, s4 = shapes
        c4 = F.interpolate(self.to_c4(fused), size=s4, mode="bilinear", align_corners=False)
        c3 = F.interpolate(self.to_c3(c4), size=s3, mode="bilinear", align_corners=False)
        c2 = F.interpolate(self.to_c2(c3), size=s2, mode="bilinear", align_corners=False)
        c1 = F.interpolate(self.to_c1(c2), size=s1, mode="bilinear", align_corners=False)
        return self.z1(c1), self.z2(c2), self.z3(c3), self.z4(c4)


class CovarianceGeometryHead(nn.Module):
    """Covariance-pooled geometry prior head inspired by Cov2Pose.

    It keeps a compact second-order summary of low-resolution spatial features.
    The output is still the same 8D geometry vector used by the old head:
    center(2) + x/y/z directions(6).
    """

    def __init__(self, in_channels, pool_size=14, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        num_positions = pool_size * pool_size
        cov_dim = num_positions * (num_positions + 1) // 2
        self.register_buffer("triu_indices", torch.triu_indices(num_positions, num_positions), persistent=False)
        layers = [
            nn.LayerNorm(cov_dim),
            nn.Linear(cov_dim, hidden_dim),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, 8),
            ]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = self.pool(x).float()
            b, c, h, w = x.shape
            x = x.flatten(2)
            x = x - x.mean(dim=1, keepdim=True)
            cov = x.transpose(1, 2).bmm(x) / max(float(c - 1), 1.0)
            cov = cov + 1e-4 * torch.eye(h * w, device=x.device, dtype=x.dtype).unsqueeze(0)
            cov_vec = cov[:, self.triu_indices[0], self.triu_indices[1]]
            return self.mlp(cov_vec)


class AxisPosePP(nn.Module):
    def __init__(self, base_channels=64, pose_dim=12, dino=None, geometry_head=None):
        super().__init__()
        dino = dino or {}
        geometry_head = geometry_head or {}
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.ref_encoder = ReferenceConditionEncoder(dino, base_channels=base_channels, pose_dim=pose_dim)
        self.cond_pyramid = ConditionPyramid(base_channels=base_channels)

        self.stem = nn.Sequential(ConvGNAct(3, c1), ResBlock(c1))
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.middle = ResBlock(c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.heatmap_head = nn.Conv2d(c1, 4, kernel_size=1)
        self.geometry_head_type = geometry_head.get("type", "covariance").lower()
        if self.geometry_head_type in ("gap", "mlp", "old"):
            self.geometry_head_type = "gap"
            self.geometry_pool = nn.AdaptiveAvgPool2d(1)
            self.geometry_head = nn.Sequential(
                nn.Linear(c4, c4),
                nn.SiLU(inplace=True),
                nn.Linear(c4, 8),
            )
        else:
            self.geometry_head_type = "covariance"
            self.geometry_head = CovarianceGeometryHead(
                c4,
                pool_size=geometry_head.get("pool_size", 14),
                hidden_dim=geometry_head.get("hidden_dim", 512),
                dropout=geometry_head.get("dropout", 0.0),
            )

    def forward(self, batch):
        target = batch["target_image"]
        x1 = self.stem(target)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        fused_ref = self.ref_encoder(
            batch["ref_image"],
            batch["ref_axis"],
            batch["ref_pose"],
            target_size=x4.shape[-2:],
        )
        z1, z2, z3, z4 = self.cond_pyramid(
            fused_ref,
            shapes=[x1.shape[-2:], x2.shape[-2:], x3.shape[-2:], x4.shape[-2:]],
        )

        x1 = x1 + z1
        x2 = x2 + z2
        x3 = x3 + z3
        x4 = self.middle(x4 + z4)

        if self.geometry_head_type == "gap":
            geometry_raw = self.geometry_head(self.geometry_pool(x4).flatten(1))
        else:
            geometry_raw = self.geometry_head(x4)
        center = torch.sigmoid(geometry_raw[:, 0:2])
        directions = F.normalize(geometry_raw[:, 2:8].view(-1, 3, 2), dim=-1, eps=1e-6)

        y = self.up3(x4, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        heatmap_logits = self.heatmap_head(y)

        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": torch.sigmoid(heatmap_logits),
            "geometry_raw": geometry_raw,
            "center": center,
            "directions": directions,
        }
