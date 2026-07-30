"""Attention U-Net (resnet34 인코더 + Attention Gate 디코더).

- 인코더: segmentation_models_pytorch 의 resnet34 (ImageNet pretrained). dropout 없음.
- 디코더: 직접 구현. 각 DecoderBlock 마지막에 채널 단위 dropout 을 넣어 MC Dropout 이 디코더 전체에서 작동한다.
- dropout 은 Dropout2dHardcoded 사용 (nn.Dropout2d 와 동작 동일, ONNX export 가능).
- ONNX 호환: 업샘플은 F.interpolate(scale_factor=2, mode="nearest") 리터럴만 사용 (입력 512 고정).
- build_model() 하나만 노출.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class Dropout2dHardcoded(nn.Dropout2d):
    """nn.Dropout2d를 하드코딩한 것. 동작 동일.
    nn.Dropout2d는 ONNX export가 안 돼서(aten::feature_dropout 미지원)
    같은 동작을 기본 연산으로 풀어 썼다."""

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # (B,C,1,1) 마스크 → 공간축 broadcast = 채널 단위 dropout
        noise = torch.rand_like(x[:, :, :1, :1])
        mask = (noise >= self.p).to(x.dtype) / (1.0 - self.p)
        return x * mask


class AttentionGate(nn.Module):
    """Oktay 2018 additive attention. 깊은 층 g 로 skip x 를 게이팅한다."""

    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_x, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g, x 각각 1x1 conv+BN 후 더하고 ReLU → 1x1 conv+BN+Sigmoid → attention map
        att = self.psi(self.relu(self.w_g(g) + self.w_x(x)))   # (N,1,H,W)
        return x * att                                          # skip 에 곱해서 반환


class DecoderBlock(nn.Module):
    """업샘플 → attention gate 로 skip 게이팅 → concat → conv3x3 2회 → Dropout2dHardcoded."""

    def __init__(self, in_ch, skip_ch, out_ch, p):
        super().__init__()
        self.gate = AttentionGate(f_g=in_ch, f_x=skip_ch, f_int=max(skip_ch // 2, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.drop = Dropout2dHardcoded(p)     # 블록마다 마지막에 채널 단위 dropout

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")   # 2배 업샘플 (ONNX 안전)
        skip = self.gate(x, skip)                              # 업샘플한 x 를 g 로 써서 skip 게이팅
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.drop(x)


class AttentionUNet(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.2, encoder_weights="imagenet"):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            "resnet34", in_channels=3, depth=5, weights=encoder_weights
        )
        ch = self.encoder.out_channels          # resnet34: (3, 64, 64, 128, 256, 512)
        self.dec1 = DecoderBlock(ch[5], ch[4], 256, dropout_p)   # /32 -> /16
        self.dec2 = DecoderBlock(256, ch[3], 128, dropout_p)     # /16 -> /8
        self.dec3 = DecoderBlock(128, ch[2], 64, dropout_p)      # /8  -> /4
        self.dec4 = DecoderBlock(64, ch[1], 32, dropout_p)       # /4  -> /2
        self.head = nn.Conv2d(32, num_classes, 1)               # 1x1 conv -> 클래스

    def forward(self, x):
        f = self.encoder(x)                     # 6개 feature (얕은->깊은)
        x = self.dec1(f[5], f[4])
        x = self.dec2(x, f[3])
        x = self.dec3(x, f[2])
        x = self.dec4(x, f[1])
        x = F.interpolate(x, scale_factor=2, mode="nearest")    # /2 -> /1 (512)
        return self.head(x)


def build_model(num_classes=5, dropout_p=0.2, encoder_weights="imagenet"):
    """Attention U-Net 인스턴스를 새로 만들어 반환한다."""
    return AttentionUNet(num_classes, dropout_p, encoder_weights)
