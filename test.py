"""
test.py - 手工实现一个简化版的 YOLOv8-P2 风格模型（教学用）并附带详尽中文注释

说明:
- 该脚本是教学示例，按图中 YOLOv8n-P2 的思路构建：增加 P2 尺度以更好检测小目标。
- 结构: Backbone -> Neck (top-down + bottom-up) -> Detect Head (解耦 cls/reg)
- 为了保证可运行，本例在推理时返回 logits（training=True 时），并在 __main__ 中做一次前向
  以打印输出 shapes，便于检查各尺度的输出是否正确。

注意: 本文件侧重教学与可读性，未实现训练损失、目标分配、DFL、NMS 等完整训练/推理流程。
"""

import math  # 用于 sqrt 等操作
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- 基础模块 (Conv + BN + Act) ---------------------------
class ConvBnAct(nn.Module):
    """Conv -> BatchNorm -> SiLU 简洁封装

    参数:
    - c1: 输入通道
    - c2: 输出通道
    - k: kernel size
    - s: stride
    - p: padding (若为 None 则设为 k//2 实现相对的'same'行为)
    - bias: 是否使用偏置 (在 BN 后通常不需要)
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, bias=False):
        super().__init__()
        if p is None:
            p = k // 2  # 常用的 padding 策略
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # conv -> bn -> act
        return self.act(self.bn(self.conv(x)))


# --------------------------- Bottleneck (残差块) ---------------------------
class Bottleneck(nn.Module):
    """轻量级残差瓶颈: 1x1 降通道 -> 3x3 升 / 处理 -> 可选残差相加

    该模块常用来在不显著增加计算量的同时增加网络深度与表达能力。
    """
    def __init__(self, c1, c2, shortcut=True, expansion=0.5):
        super().__init__()
        c_ = int(c2 * expansion)  # 中间通道
        self.conv1 = ConvBnAct(c1, c_, k=1)
        self.conv2 = ConvBnAct(c_, c2, k=3)
        # 只有在输入输出通道相等且允许 shortcut 时才进行相加
        self.use_add = shortcut and c1 == c2

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            return x + y
        return y


# --------------------------- C2f 模块 ---------------------------
class C2f(nn.Module):
    """C2f: 通道分裂 + 右分支多瓶颈 + 拼接融合

    步骤:
    1) 1x1 投影到 c2 通道
    2) 将通道对半分为 p (left) 和 q (right)
    3) 对 q 串联 n 个 Bottleneck，每步保存输出
    4) 将 p 与所有 q 的中间输出 concat 后通过 1x1 融合回 c2
    """
    def __init__(self, c1, c2, n=1, expansion=0.5):
        super().__init__()
        self.cv1 = ConvBnAct(c1, c2, k=1)  # 投影
        self.n = n
        m = c2 // 2  # 将 c2 分为两个大致相等的部分
        # 右分支的瓶颈列表（每个作用在 m->m）
        self.blocks = nn.ModuleList([Bottleneck(m, m, shortcut=True, expansion=1.0) for _ in range(n)])
        # 最终融合（concat 后通道数为 (n+2)*m）
        self.cv2 = ConvBnAct((n + 2) * m, c2, k=1)

    def forward(self, x):
        x = self.cv1(x)  # 投影 -> [B, c2, H, W]
        c = x.shape[1]
        m = c // 2
        p = x[:, :m, :, :]  # 左分支
        q = x[:, m:, :, :]  # 右分支
        qs = [q]
        for blk in self.blocks:
            q = blk(q)
            qs.append(q)  # 保存每一步的右分支输出
        qcat = torch.cat(qs, dim=1)  # concat 所有右分支输出
        y = torch.cat([p, qcat], dim=1)  # 与左分支拼接
        return self.cv2(y)  # 融合回 c2


# --------------------------- SPPF 模块 ---------------------------
class SPPF(nn.Module):
    """快速空间金字塔池化 (SPPF)

    实现方式: 降通道 -> 连续三次相同 kernel 的 maxpool (stride=1) -> concat -> 融合
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBnAct(c1, c_, k=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBnAct(c_ * 4, c2, k=1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        y = torch.cat([x, y1, y2, y3], dim=1)
        return self.cv2(y)


# --------------------------- Attention: Squeeze-and-Excite ---------------------------
class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation (SE) 简化实现，用于通道注意力增强

    仅做通道重标定：全局平均池化 -> fc-relu-fc-sigmoid -> 通道重加权
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # 全局平均池化
        y = x.mean(dim=(2, 3)).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y  # 通道加权


# --------------------------- Backbone (生成 P2..P5 多尺度特征) ---------------------------
class SimpleBackboneP2(nn.Module):
    """为了实现 P2..P5，我们在 backbone 中实现 4 次下采样，返回 4 个尺度特征。

    目标尺度 (假设输入为 640x640):
      - P2: stride=4  (较浅，用于检测非常小的目标)
      - P3: stride=8
      - P4: stride=16
      - P5: stride=32 (最深，用于大目标)

    设计说明: 为了示例可运行，我们选择适中的通道数，便于在 CPU 上快速前向。
    """
    def __init__(self):
        super().__init__()
        # stem: 不下采样，仅改变通道
        self.stem = ConvBnAct(3, 32, k=3, s=1)
        # 依次下采样，每个 stage 包含一个下采样 conv + C2f
        self.conv1 = ConvBnAct(32, 64, k=3, s=2)   # -> stride 2
        self.c2f1 = C2f(64, 64, n=1)

        self.conv2 = ConvBnAct(64, 128, k=3, s=2)  # -> stride 4  => P2
        self.c2f2 = C2f(128, 128, n=2)

        self.conv3 = ConvBnAct(128, 256, k=3, s=2)  # -> stride 8 => P3
        self.c2f3 = C2f(256, 256, n=2)

        self.conv4 = ConvBnAct(256, 256, k=3, s=2)  # -> stride 16 => P4
        self.c2f4 = C2f(256, 256, n=3)

        self.conv5 = ConvBnAct(256, 256, k=3, s=2)  # -> stride 32 => P5
        self.c2f5 = C2f(256, 256, n=1)
        self.sppf = SPPF(256, 256)

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.stem(x)
        x = self.conv1(x)   # stride 2
        x = self.c2f1(x)

        x = self.conv2(x)   # stride 4
        c2 = self.c2f2(x)    # P2 (shallower)

        x = self.conv3(c2)   # stride 8
        c3 = self.c2f3(x)    # P3

        x = self.conv4(c3)   # stride 16
        c4 = self.c2f4(x)    # P4

        x = self.conv5(c4)   # stride 32
        c5 = self.c2f5(x)
        c5 = self.sppf(c5)   # SPPF 提升深层感受野

        # 返回 P2, P3, P4, P5（由浅到深）
        return c2, c3, c4, c5


# --------------------------- Neck (融合 P2..P5) ---------------------------
class SimpleNeckP2(nn.Module):
    """实现一个简单的自上而下 + 自下而上的融合结构，支持 4 个尺度 (P2..P5)。

    设计说明:
    - 顶层先对深层特征进行 lateral 投影，再逐步上采样与较浅层拼接与融合
    - 然后自下而上做底部融合以增强跨尺度信息传播（类似 PAN / BiFPN 的思想的简化版）
    """
    def __init__(self):
        super().__init__()
        # lateral 投影：把不同尺度投影到希望的通道数以便拼接
        self.lateral5 = ConvBnAct(256, 256, k=1)
        self.lateral4 = ConvBnAct(256, 256, k=1)
        self.lateral3 = ConvBnAct(256, 128, k=1)
        self.lateral2 = ConvBnAct(128, 64, k=1)

        # 融合模块使用 C2f（更接近 YOLOv8 的设计）
        self.c2f54 = C2f(256 + 256, 256, n=1)  # 上采样 p5 与 p4 融合
        self.c2f43 = C2f(256 + 128, 128, n=1)  # 上采样 t4 与 p3 融合
        self.c2f32 = C2f(128 + 64, 64, n=1)    # 上采样 t3 与 p2 融合

        # 简单的通道注意力，用于在融合后调整通道重要性（模拟图中注意力模块）
        self.se_t4 = SqueezeExcite(256, reduction=8)
        self.se_t3 = SqueezeExcite(128, reduction=8)
        self.se_t2 = SqueezeExcite(64, reduction=8)

        # 底部自下而上融合
        self.down2 = ConvBnAct(64, 128, k=3, s=2)
        self.c2f_p3 = C2f(128 + 128, 128, n=1)

        self.down3 = ConvBnAct(128, 256, k=3, s=2)
        self.c2f_p4 = C2f(256 + 256, 256, n=1)

        self.down4 = ConvBnAct(256, 256, k=3, s=2)
        self.c2f_p5 = C2f(256 + 256, 256, n=1)

    def forward(self, c2, c3, c4, c5):
        # top-down
        p5 = self.lateral5(c5)
        u5 = F.interpolate(p5, scale_factor=2, mode='nearest')  # 上采样到 p4 大小
        p4 = self.lateral4(c4)
        t4 = torch.cat([u5, p4], dim=1)
        t4 = self.c2f54(t4)  # 融合得到 t4
        t4 = self.se_t4(t4)  # 使用 SE 注意力增强通道响应

        u4 = F.interpolate(t4, scale_factor=2, mode='nearest')  # 上采样到 p3 大小
        p3 = self.lateral3(c3)
        t3 = torch.cat([u4, p3], dim=1)
        t3 = self.c2f43(t3)  # 融合得到 t3
        t3 = self.se_t3(t3)  # 注意力增强

        u3 = F.interpolate(t3, scale_factor=2, mode='nearest')  # 上采样到 p2 大小
        p2 = self.lateral2(c2)
        t2 = torch.cat([u3, p2], dim=1)
        t2 = self.c2f32(t2)  # 融合得到 t2
        t2 = self.se_t2(t2)  # 注意力增强

        # bottom-up (从 t2 开始向上融合回去)
        d2 = self.down2(t2)  # 下采样到 t3 大小
        p3_b = torch.cat([d2, t3], dim=1)
        p3 = self.c2f_p3(p3_b)

        d3 = self.down3(p3)  # 下采样到 t4 大小
        p4_b = torch.cat([d3, t4], dim=1)
        p4 = self.c2f_p4(p4_b)

        d4 = self.down4(p4)  # 下采样到 p5 大小
        p5_b = torch.cat([d4, p5], dim=1)
        p5 = self.c2f_p5(p5_b)

        # 返回最终用于检测的特征（从浅到深: P2,P3,P4,P5）
        return t2, p3, p4, p5


# --------------------------- Detect Head ---------------------------
class DetectHeadP2(nn.Module):
    """解耦检测头，支持 4 个尺度的输入

    该 head 在每个尺度输出:
      - cls_logits: [B, num_classes, H, W]
      - reg_logits: [B, 4*(reg_max+1), H, W]

    为了演示我们在 training=True 返回原始 logits（方便计算 loss）；
    在推理时可对 logits 进一步解码（此处未实现完整 NMS/解码以保持示例简洁）。
    """
    def __init__(self, in_channels: List[int], num_classes: int = 80, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for c in in_channels:
            # 每尺度使用一个小的 conv block -> 输出 logits
            self.cls_convs.append(nn.Sequential(ConvBnAct(c, c, k=3), nn.Conv2d(c, num_classes, kernel_size=1)))
            self.reg_convs.append(nn.Sequential(ConvBnAct(c, c, k=3), nn.Conv2d(c, 4 * (reg_max + 1), kernel_size=1)))

    def forward(self, feats: List[torch.Tensor], strides: List[int], training: bool = False):
        cls_outputs = []
        reg_outputs = []
        for i, x in enumerate(feats):
            cls_out = self.cls_convs[i](x)  # 分类 logits
            reg_out = self.reg_convs[i](x)  # 回归 logits (分布式)
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
        if training:
            # 训练模式下直接返回 logits，供损失计算使用
            return cls_outputs, reg_outputs
        # 推理模式下可以调用解码函数将 reg_logits 变为边框，这里为简化示例直接返回 logits
        return cls_outputs, reg_outputs


# --------------------------- Full Model: Backbone + Neck + Head ---------------------------
class YOLOv8P2Lite(nn.Module):
    def __init__(self, num_classes: int = 3, reg_max: int = 3):
        super().__init__()
        # Backbone 返回 P2..P5
        self.backbone = SimpleBackboneP2()
        self.neck = SimpleNeckP2()
        # Head 输入通道应与 neck 输出通道对齐 (t2:64, p3:128, p4:256, p5:256)
        self.detect = DetectHeadP2([64, 128, 256, 256], num_classes=num_classes, reg_max=reg_max)
        # 对应每一尺度的下采样倍数 (示意值，需与 backbone 实际下采样对齐)
        self.strides = [4, 8, 16, 32]

    def forward(self, x: torch.Tensor, training: bool = False):
        # backbone 输出 c2..c5
        c2, c3, c4, c5 = self.backbone(x)
        # neck 融合，返回用于 head 的四尺度特征
        f2, f3, f4, f5 = self.neck(c2, c3, c4, c5)
        feats = [f2, f3, f4, f5]
        return self.detect(feats, self.strides, training=training)


if __name__ == "__main__":
    # 快速运行检查: 构建模型并用随机输入前向，打印每个尺度输出 shape
    model = YOLOv8P2Lite(num_classes=3, reg_max=3)  # 以 3 类做示例，reg_max=3 -> K=4
    model.eval()
    x = torch.randn(1, 3, 640, 640)  # 测试输入
    with torch.no_grad():
        cls_outs, reg_outs = model(x, training=True)  # 使用 training=True 返回 logits（便于检查）

    print("Forward check (training=True):")
    for i, (c, r) in enumerate(zip(cls_outs, reg_outs)):
        print(f"Scale {i}: cls_logits shape = {tuple(c.shape)}, reg_logits shape = {tuple(r.shape)}")

    print("Done. 如果需要我可以再为每行代码增加更细致的注释或实现完整的解码/NMS 流程。")





























