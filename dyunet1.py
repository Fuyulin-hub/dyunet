import torch
import torch.nn as nn


class DyUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DyUNet, self).__init__()
        base_channels = max(in_channels * 2, 4)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        hidden_channels = max(in_channels // 4, 1)
        self.coeff_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, max(out_channels * base_channels, 1), kernel_size=1)
        )

        self.out_channels = out_channels
        self.base_channels = base_channels
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.encoder1(x)         # [B, IC, H, W]
        coeffs = self.coeff_conv(x)         # [B, OC×IC, H, W]

        B, OCxIC, H, W = coeffs.size()
        IC = self.base_channels
        OC = self.out_channels

        if OC * IC == 0:
            raise ValueError(f"Invalid channel combination: OC={OC}, IC={IC}")

        if OCxIC != OC * IC:
            raise ValueError(f"Mismatch: expected coeff shape [B, {OC*IC}, H, W] but got {OCxIC}")

        coeffs = coeffs.view(B, OC, IC, H, W)
        features = features.unsqueeze(1)    # [B, 1, IC, H, W]

        out = (features * coeffs).sum(dim=2)  # [B, OC, H, W]
        return self.final_conv(out)
