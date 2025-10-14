"""  # 模块说明: 简易的YOLOv8n风格 PyTorch 实现，教学用途，逐行注释版
Lightweight PyTorch implementation of a YOLOv8n-like architecture (educational).  # 简要说明

This module implements simplified, readable building blocks used in YOLO-style  # 描述模块包含的组件
models so you can study and experiment with them. It's intended for learning  # 用途：学习与实验
and demonstration rather than production deployment.  # 不是生产级实现

Implemented components:  # 列出已实现的模块
- Conv-BN-SiLU block (ConvBnAct)  # 卷积+BN+SiLU
- Bottleneck residual block  # 瓶颈残差块
- C2f module (channel split, repeated bottlenecks, concat, fuse)  # C2f 模块
- SPPF module (fast spatial pyramid pooling)  # SPPF 模块
- Simplified Backbone producing three feature-map scales  # 简化 backbone
- Simplified Neck (top-down + bottom-up fusion using C2f)  # 简化 neck
- Detect head (decoupled classification + distributional regression)  # 检测头
- Decode utilities: softmax->expectation, build boxes, simple per-class NMS  # 解码工具

Run the file as a script to instantiate the model and run a dummy forward.  # 运行示例

Author: educational example  # 作者信息
"""

import math  # 导入数学库，用于 sqrt 等
from typing import List, Tuple  # 类型注解用的 List/Tuple

import torch  # PyTorch 主包
import torch.nn as nn  # 神经网络模块简写
import torch.nn.functional as F  # 函数式接口 (例如 interpolate, softmax)


# --------------------------- Basic modules ---------------------------------
class ConvBnAct(nn.Module):  # Conv-BN-Act 基础块
    """Conv -> BatchNorm -> SiLU (a.k.a. Swish)  # 模块说明

    封装了卷积、批归一化和 SiLU 激活，便于复用。  # 目的
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, bias=False):  # 构造函数
        super().__init__()  # 调用父类构造器
        if p is None:  # 如果未指定 padding，则使用 kernel//2 保持 'same' 行为
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=bias)  # 2D 卷积
        self.bn = nn.BatchNorm2d(c2)  # 批归一化
        self.act = nn.SiLU()  # SiLU 激活

    def forward(self, x):  # 前向函数
        # 按顺序 conv -> bn -> act
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):  # Bottleneck 残差块
    """1x1 降维 -> 3x3 处理 -> 可选残差相加  # 说明

    用于在保持较低计算的条件下增加网络深度与表现力。  # 目的
    """
    def __init__(self, c1, c2, shortcut=True, expansion=0.5):  # 构造
        super().__init__()
        c_ = int(c2 * expansion)  # 中间通道数 = c2 * expansion
        self.conv1 = ConvBnAct(c1, c_, k=1)  # 1x1 conv 降维
        self.conv2 = ConvBnAct(c_, c2, k=3)  # 3x3 conv
        self.use_add = shortcut and c1 == c2  # 仅当通道匹配且允许 shortcut 时使用残差

    def forward(self, x):  # 前向
        y = self.conv2(self.conv1(x))  # 先 conv1 再 conv2
        if self.use_add:  # 如果允许残差且形状匹配
            return x + y  # 残差相加
        return y  # 否则直接返回


class C2f(nn.Module):  # C2f 模块
    """通道拆分 + 右分支多次 Bottleneck 处理 + 拼接融合  # 说明

    先通过 1x1 投影到 c2 通道，然后把通道对半分为左/右。右半部分依次
    通过 n 个 Bottleneck，并把每一步的输出保存以便 concat，最后与左半
    一起通过 1x1 conv 融合回 c2 通道。
    """
    def __init__(self, c1, c2, n=1, expansion=0.5):
        super().__init__()
        self.cv1 = ConvBnAct(c1, c2, k=1)  # 投影到 c2
        self.n = n
        self.c2 = c2
        self.expand = expansion
        m = c2 // 2  # 右/左每份通道数（向下取整）
        # 右分支的 n 个 Bottleneck
        self.blocks = nn.ModuleList([Bottleneck(m, m, shortcut=True, expansion=1.0) for _ in range(n)])
        # 最后融合的 1x1 conv，输入通道 = p + (n+1)*m = (n+2)*m
        self.cv2 = ConvBnAct((n + 2) * m, c2, k=1)

    def forward(self, x):
        x = self.cv1(x)  # 先投影 -> [B, c2, H, W]
        c = x.shape[1]
        m = c // 2
        p = x[:, :m, :, :]  # 左半部分
        q = x[:, m:, :, :]  # 右半部分
        qs = [q]  # 保存初始右半
        for blk in self.blocks:
            q = blk(q)  # 右分支经过每个 Bottleneck
            qs.append(q)  # 保存中间输出
        qcat = torch.cat(qs, dim=1)  # 把所有右侧输出沿通道 concat
        y = torch.cat([p, qcat], dim=1)  # 与左侧 concat
        return self.cv2(y)  # 最后 1x1 融合回 c2


class SPPF(nn.Module):  # SPPF 模块
    """快速空间金字塔池化（SPPF）  # 说明

    通过一系列相同 kernel 的 maxpool (stride=1) 获得不同感受野的特征并拼接，
    然后通过 1x1 conv 融合。此处采用级联池化（pool(pool(x))）的实现。
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # 降通道
        self.cv1 = ConvBnAct(c1, c_, k=1)  # 降维 1x1
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 池化层
        self.cv2 = ConvBnAct(c_ * 4, c2, k=1)  # 融合 conv

    def forward(self, x):
        x = self.cv1(x)  # 降通道
        y1 = self.pool(x)  # 第一次池化
        y2 = self.pool(y1)  # 第二次（在 y1 上）
        y3 = self.pool(y2)  # 第三次
        y = torch.cat([x, y1, y2, y3], dim=1)  # 拼接
        return self.cv2(y)  # 融合后返回


# --------------------------- Backbone --------------------------------------
class SimpleBackbone(nn.Module):  # 简化的骨干网络
    """产生三个尺度特征的简化 backbone（教学用）  # 说明

    由一个 stem 和 3 个下采样阶段组成，每个阶段包含 C2f，最深处加入 SPPF。
    """
    def __init__(self):
        super().__init__()
        self.stem = ConvBnAct(3, 32, k=3, s=1)  # 输入 3 -> 32
        self.conv1 = ConvBnAct(32, 64, k=3, s=2)  # 下采样 1
        self.c2f1 = C2f(64, 64, n=1)  # 第一层 C2f
        self.conv2 = ConvBnAct(64, 128, k=3, s=2)  # 下采样 2
        self.c2f2 = C2f(128, 128, n=2)  # 第二层 C2f
        self.conv3 = ConvBnAct(128, 256, k=3, s=2)  # 下采样 3
        self.c2f3 = C2f(256, 256, n=3)  # 第三层 C2f
        self.sppf = SPPF(256, 256)  # SPPF 在最深层

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.stem(x)  # stem
        x = self.conv1(x)  # downsample -> 64
        c3 = self.c2f1(x)  # 第一个返回特征
        x = self.conv2(c3)  # downsample -> 128
        c4 = self.c2f2(x)  # 第二个返回特征
        x = self.conv3(c4)  # downsample -> 256
        c5 = self.c2f3(x)  # 第三个（深）特征
        c5 = self.sppf(c5)  # 使用 SPPF
        return c3, c4, c5  # 返回三个尺度


# --------------------------- Neck ------------------------------------------
class SimpleNeck(nn.Module):  # 简化的 neck
    """PAN 风格的 neck: top-down + bottom-up 融合  # 说明

    使用 lateral projection、上采样拼接融合（C2f），再自底向上下采样融合。
    """
    def __init__(self):
        super().__init__()
        self.lateral5 = ConvBnAct(256, 128, k=1)  # 将 c5 投影到 128
        self.lateral4 = ConvBnAct(128, 128, k=1)  # 将 c4 投影到 128
        self.lateral3 = ConvBnAct(64, 64, k=1)  # 将 c3 投影到 64
        self.c2f4 = C2f(128 + 128, 128, n=1)  # 融合上采样(c5) 与 c4
        self.c2f3 = C2f(64 + 64, 64, n=1)  # 融合上采样(t4) 与 c3
        self.down4 = ConvBnAct(64, 128, k=3, s=2)  # 下采样 t3 -> 匹配 t4 大小
        self.c2f_p4 = C2f(128 + 128, 128, n=1)  # 底部融合得到 p4
        self.down5 = ConvBnAct(128, 256, k=3, s=2)  # 下采样 p4 -> 匹配 p5
        self.c2f_p5 = C2f(256 + 256, 256, n=1)  # 底部融合得到 p5

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)  # 投影 c5 -> p5
        u5 = F.interpolate(p5, scale_factor=2, mode='nearest')  # 上采样到 c4 大小
        p4 = self.lateral4(c4)  # 投影 c4
        t4 = torch.cat([u5, p4], dim=1)  # 拼接上采样结果与 c4
        t4 = self.c2f4(t4)  # 融合

        u4 = F.interpolate(t4, scale_factor=2, mode='nearest')  # 上采样到 c3 大小
        p3 = self.lateral3(c3)  # 投影 c3
        t3 = torch.cat([u4, p3], dim=1)  # 拼接
        t3 = self.c2f3(t3)  # 融合

        d3 = self.down4(t3)  # 从 t3 下采样回 t4 大小
        p4_b = torch.cat([d3, t4], dim=1)  # 拼接 bottom-up 与 top-down 的 t4
        p4 = self.c2f_p4(p4_b)  # 融合得到 p4

        d4 = self.down5(p4)  # 下采样 p4 -> p5 大小
        p5_b = torch.cat([d4, p5], dim=1)  # 拼接
        p5 = self.c2f_p5(p5_b)  # 融合得到 p5

        return t3, p4, p5  # 返回三个尺度特征


# --------------------------- Detect Head ----------------------------------
class DetectHead(nn.Module):  # 检测头（解耦 cls 与 reg）
    """检测头：分类分支 + 分布式回归分支  # 说明

    分类输出: [B, num_classes, H, W]，回归输出: [B, 4*(K), H, W]，K=reg_max+1。
    解码时把回归 logits 通过 softmax->期望转换为距离。
    """
    def __init__(self, in_channels: List[int], num_classes: int = 80, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.nl = len(in_channels)
        self.cls_convs = nn.ModuleList()  # 每个尺度的分类子网络
        self.reg_convs = nn.ModuleList()  # 每个尺度的回归子网络
        for c in in_channels:
            # 分类分支：一个 ConvBnAct + 1x1 输出 num_classes
            self.cls_convs.append(nn.Sequential(
                ConvBnAct(c, c, k=3),
                nn.Conv2d(c, num_classes, kernel_size=1)
            ))
            # 回归分支：一个 ConvBnAct + 1x1 输出 4*(K)
            self.reg_convs.append(nn.Sequential(
                ConvBnAct(c, c, k=3),
                nn.Conv2d(c, 4 * (reg_max + 1), kernel_size=1)
            ))

    def forward(self, feats: List[torch.Tensor], strides: List[int], training=False):
        cls_outputs = []  # 保存每尺度的分类 logits
        reg_outputs = []  # 保存每尺度的回归 logits
        for i, x in enumerate(feats):
            cls_out = self.cls_convs[i](x)  # 分类 logits
            reg_out = self.reg_convs[i](x)  # 回归 logits
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
        if training:
            return cls_outputs, reg_outputs  # 训练模式直接返回 logits
        dets = []  # 推理时存放解码后的 (boxes, scores, labels)
        for i, (cls_out, reg_out, s) in enumerate(zip(cls_outputs, reg_outputs, strides)):
            b, nc, h, w = cls_out.shape  # batch, channels(num_classes), H, W
            probs = torch.sigmoid(cls_out)  # 对分类 logits 做 sigmoid -> 概率
            K = self.reg_max + 1  # 每边的离散 bin 数
            reg_out = reg_out.view(b, 4, K, h, w)  # reshape 为 [B,4,K,H,W]
            for bi in range(b):
                # 将空间维度 flatten 为 N = H*W
                prob_map = probs[bi].permute(1, 2, 0).reshape(-1, self.num_classes)  # [N, C]
                reg_map = reg_out[bi].permute(2, 3, 4, 0).reshape(-1, 4, K)  # [N,4,K]
                boxes, scores, labels = decode_per_feature_map(reg_map, prob_map, s)  # 解码
                dets.append((boxes, scores, labels))
        return dets  # 返回所有尺度/批次的检测结果列表


# --------------------------- Utilities ------------------------------------

def softmax_expectation(logits: torch.Tensor) -> torch.Tensor:
    """logits: [..., K] -> expectation over K as float: sum_k k * softmax(logits)[k]
    returns: [...]
    """
    probs = F.softmax(logits, dim=-1)
    K = logits.shape[-1]
    idx = torch.arange(K, dtype=probs.dtype, device=probs.device)
    # compute expectation over discrete bins 0..K-1: E[k] = sum_k k * P(k)
    exp = torch.sum(probs * idx, dim=-1)
    return exp


def decode_per_feature_map(reg_map: torch.Tensor, prob_map: torch.Tensor, stride: int,
                           score_thresh: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one feature map's outputs into boxes / scores / labels for one image.
    reg_map: [N, 4, K]
    prob_map: [N, num_classes]
    stride: int
    Returns N_filtered boxes (x1,y1,x2,y2), scores, labels
    Note: N = H*W spatial positions.
    """
    device = reg_map.device
    N = reg_map.shape[0]
    K = reg_map.shape[-1]
    # compute expectation per side
    # reg_map: [N,4,K] -> exp: [N,4]
    exp_vals = softmax_expectation(reg_map)  # [N,4]
    # map bins to pixels: here assume each bin=1 grid unit -> multiply by stride
    dists = exp_vals * stride
    # compute grid coordinates for each position N
    # we need H and W: infer from N by trying common sizes? Simpler: user of this util
    # should ensure that the ordering of reg_map/prob_map matches the grid flattening.
    # For simplicity in this educational implementation we assume that the caller flattened
    # index in row-major order for a grid that we can infer via int(sqrt(N)) when square.
    side = int(math.sqrt(N))
    if side * side != N:
        # fallback: treat as a single point
        gx = torch.zeros(N, device=device)
        gy = torch.zeros(N, device=device)
    else:
        ys = torch.arange(side, device=device).repeat_interleave(side)
        xs = torch.arange(side, device=device).repeat(side)
        # recover grid center coordinates (x,y) in original image space by
        # placing centers at (i+0.5)*stride for each cell in the HxW grid.
        gx = (xs.float() + 0.5) * stride
        gy = (ys.float() + 0.5) * stride
    # build boxes
    x1 = gx - dists[:, 0]
    y1 = gy - dists[:, 1]
    x2 = gx + dists[:, 2]
    y2 = gy + dists[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    # compute score and labels
    scores, labels = torch.max(prob_map, dim=1)  # [N]
    # filter by score_thresh
    keep = scores > score_thresh
    if keep.sum() == 0:
        return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device), torch.zeros((0,), device=device)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    # apply NMS per-class (simple implementation)
    keep_boxes, keep_scores, keep_labels = simple_nms(boxes, scores, labels, iou_thres=0.45)
    return keep_boxes, keep_scores, keep_labels


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes
    box1: [N,4], box2: [M,4] -> [N,M]
    """
    N = box1.size(0)
    M = box2.size(0)
    # intersection left-top and right-bottom points
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])  # [N,M,2]
    # width and height of intersection, clamped to zero for disjoint boxes
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    # compute areas and union
    area1 = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]))[:, None]
    area2 = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]))[None, :]
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def simple_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thres: float = 0.45):
    """Per-class NMS (naive, not optimized). Returns kept boxes, scores, labels.
    boxes: [N,4]
    scores: [N]
    labels: [N]
    """
    kept_boxes = []
    kept_scores = []
    kept_labels = []
    unique_labels = labels.unique()
    for c in unique_labels:
        mask = labels == c
        b = boxes[mask]
        s = scores[mask]
        if b.size(0) == 0:
            continue
        order = s.argsort(descending=True)
        b = b[order]
        s = s[order]
        keep = []
        while b.size(0):
            keep.append((b[0], s[0]))
            if b.size(0) == 1:
                break
            ious = box_iou(b[0:1], b[1:])[0]
            keep_mask = ious <= iou_thres
            b = b[1:][keep_mask]
            s = s[1:][keep_mask]
        for bb, ss in keep:
            kept_boxes.append(bb)
            kept_scores.append(ss)
            kept_labels.append(c.item())
    if len(kept_boxes) == 0:
        return torch.zeros((0, 4), device=boxes.device), torch.zeros((0,), device=boxes.device), torch.zeros((0,), device=boxes.device)
    return torch.stack(kept_boxes), torch.tensor(kept_scores, device=boxes.device), torch.tensor(kept_labels, device=boxes.device)


# --------------------------- Full Model -----------------------------------
class YOLOv8nLite(nn.Module):
    def __init__(self, num_classes=80, reg_max=16):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.neck = SimpleNeck()
        # in_channels for head correspond to outputs of neck: t3 (64), p4 (128), p5 (256)
        self.detect = DetectHead([64, 128, 256], num_classes=num_classes, reg_max=reg_max)
        # strides corresponding to each feature map (user must align with shapes)
        self.strides = [8, 16, 32]

    def forward(self, x: torch.Tensor, training: bool = False):
        c3, c4, c5 = self.backbone(x)
        h3, h4, h5 = self.neck(c3, c4, c5)
        feats = [h3, h4, h5]
        if training:
            return self.detect(feats, self.strides, training=True)
        else:
            return self.detect(feats, self.strides, training=False)


# --------------------------- Quick test -----------------------------------
if __name__ == "__main__":
    # create a tiny model for test with small reg_max to reduce channel sizes
    model = YOLOv8nLite(num_classes=3, reg_max=3)  # K=4 in this demo
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        # run an example forward pass in inference mode
        dets = model(x, training=False)
    print("Detections (per-scale per-batch):")
    for i, d in enumerate(dets):
        boxes, scores, labels = d
        # print shapes so user can verify outputs without printing raw tensors
        print(f"Scale result {i}: boxes={boxes.shape}, scores={scores.shape}, labels={labels.shape}")

    print("Done.\nNotes: \n- This is a simplified educational implementation.\n- For real training/evaluation you'll need target assignment, losses (DFL/IoU), and dataloader preprocessing.")
