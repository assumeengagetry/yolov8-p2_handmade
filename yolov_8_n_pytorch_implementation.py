"""
Lightweight PyTorch implementation of a YOLOv8n-like architecture (educational).

This module implements simplified, readable building blocks used in YOLO-style
models so you can study and experiment with them. It's intended for learning
and demonstration rather than production deployment.

Implemented components:
- Conv-BN-SiLU block (ConvBnAct)
- Bottleneck residual block
- C2f module (channel split, repeated bottlenecks, concat, fuse)
- SPPF module (fast spatial pyramid pooling)
- Simplified Backbone producing three feature-map scales
- Simplified Neck (top-down + bottom-up fusion using C2f)
- Detect head (decoupled classification + distributional regression)
- Decode utilities: softmax->expectation, build boxes, simple per-class NMS

Run the file as a script to instantiate the model and run a dummy forward.

Author: educational example
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- Basic modules ---------------------------------
class ConvBnAct(nn.Module):
    """Conv -> BatchNorm -> SiLU (a.k.a. Swish)

    This convenience wrapper groups a convolution, batch normalization and
    a SiLU activation into one reusable block. Using these small building
    blocks keeps the higher-level module definitions compact.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, bias=False):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # conv -> bn -> act
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """A lightweight bottleneck residual block.

    Pattern: 1x1 conv to reduce channels -> 3x3 conv to process -> optional
    residual add if input and output channels match. This is commonly used
    inside more complex modules (e.g. C2f) to add capacity with a small cost.
    """
    def __init__(self, c1, c2, shortcut=True, expansion=0.5):
        super().__init__()
        c_ = int(c2 * expansion)
        self.conv1 = ConvBnAct(c1, c_, k=1)
        self.conv2 = ConvBnAct(c_, c2, k=3)
        self.use_add = shortcut and c1 == c2

    def forward(self, x):
        # apply the two conv layers
        y = self.conv2(self.conv1(x))
        # add residual only when channels match and shortcut requested
        if self.use_add:
            return x + y
        return y


class C2f(nn.Module):
    """C2f module (channel-splitting + fused path).

    The input is first projected to `c2` channels, then split in half along the
    channel dimension. The right half is processed by `n` Bottleneck blocks, and
    all intermediate right-half outputs (including the initial right half) are
    concatenated and fused together with the left half. This follows the C2f
    design in YOLOv8 where multiple internal representations are combined.
    """
    def __init__(self, c1, c2, n=1, expansion=0.5):
        super().__init__()
        self.cv1 = ConvBnAct(c1, c2, k=1)
        self.n = n
        self.c2 = c2
        # we will split in half (floor)
        self.expand = expansion
        m = c2 // 2
        # create n bottlenecks acting on the right split
        self.blocks = nn.ModuleList([Bottleneck(m, m, shortcut=True, expansion=1.0) for _ in range(n)])
        # final fuse conv
        self.cv2 = ConvBnAct((n + 2) * m, c2, k=1)

    def forward(self, x):
        x = self.cv1(x)  # -> [B, c2, H, W]
        c = x.shape[1]
        m = c // 2
        # split channels into left (p) and right (q) parts
        p = x[:, :m, :, :]
        q = x[:, m:, :, :]
        qs = [q]
        # process the right part through each bottleneck, saving intermediate
        # outputs so they can be concatenated (this enriches representation).
        for blk in self.blocks:
            q = blk(q)
            qs.append(q)
        # concatenate all right-side outputs, then concat with left part
        qcat = torch.cat(qs, dim=1)
        y = torch.cat([p, qcat], dim=1)
        # final 1x1 conv to fuse channels back to c2
        return self.cv2(y)


class SPPF(nn.Module):
    """SPPF (Spatial Pyramid Pooling - Fast).

    This implementation reduces channels, applies a sequence of same-sized
    max-pooling operations (each reusing the previous result), concatenates
    them and fuses the result. It provides multiple receptive fields cheaply.
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
        # concat original and pooled features along channel dim and fuse
        y = torch.cat([x, y1, y2, y3], dim=1)
        return self.cv2(y)


# --------------------------- Backbone --------------------------------------
class SimpleBackbone(nn.Module):
    """Simplified backbone that produces three intermediate feature maps.

    The network is intentionally compact for clarity: a stem conv followed by
    three stages that downsample the spatial resolution. Each stage includes
    a C2f block to add capacity. The deepest stage receives SPPF to increase
    receptive field before returning features.
    """
    def __init__(self):
        super().__init__()
        # stem
        self.stem = ConvBnAct(3, 32, k=3, s=1)
        # stage 1 -> stride=2 downsample
        self.conv1 = ConvBnAct(32, 64, k=3, s=2)
        self.c2f1 = C2f(64, 64, n=1)
        # stage 2
        self.conv2 = ConvBnAct(64, 128, k=3, s=2)
        self.c2f2 = C2f(128, 128, n=2)
        # stage 3
        self.conv3 = ConvBnAct(128, 256, k=3, s=2)
        self.c2f3 = C2f(256, 256, n=3)
        # sppf on deepest
        self.sppf = SPPF(256, 256)

    def forward(self, x):
        # x: [B,3,640,640]
        # basic stem and downsampling stages
        x = self.stem(x)               # -> [B,32,H,W] where H,W are input size
        x = self.conv1(x)              # downsample -> channels 64
        # first intermediate feature (shallower)
        c3 = self.c2f1(x)              # chosen as the first return feature
        x = self.conv2(c3)             # downsample -> channels 128
        c4 = self.c2f2(x)              # mid-level feature
        x = self.conv3(c4)             # downsample -> channels 256
        c5 = self.c2f3(x)              # deep feature
        c5 = self.sppf(c5)             # SPPF enriches deep feature
        # Return three scales. Note: exact strides depend on input size; here
        # we document the intended design (e.g. stride 8/16/32) for clarity.
        return c3, c4, c5


# --------------------------- Neck ------------------------------------------
class SimpleNeck(nn.Module):
    """Simplified neck that fuses backbone features top-down and bottom-up.

    This neck implements a small PAN-like structure: lateral projections,
    upsampling + concat + fusion (top-down), followed by downsampling + concat
    + fusion (bottom-up). It returns three fused feature maps used by the head.
    """
    def __init__(self):
        super().__init__()
        # lateral projections
        self.lateral5 = ConvBnAct(256, 128, k=1)
        self.lateral4 = ConvBnAct(128, 128, k=1)
        self.lateral3 = ConvBnAct(64, 64, k=1)
        # fusion C2f modules
        self.c2f4 = C2f(128 + 128, 128, n=1)  # after concat upsample(c5)+c4
        self.c2f3 = C2f(64 + 64, 64, n=1)     # after concat upsample(t4)+c3
        # bottom-up
        self.down4 = ConvBnAct(64, 128, k=3, s=2)
        self.c2f_p4 = C2f(128 + 128, 128, n=1)
        self.down5 = ConvBnAct(128, 256, k=3, s=2)
        self.c2f_p5 = C2f(256 + 256, 256, n=1)

    def forward(self, c3, c4, c5):
        # top-down
        p5 = self.lateral5(c5)  # -> [B,128,H,W]
        u5 = F.interpolate(p5, scale_factor=2, mode='nearest')  # up to c4 size
        p4 = self.lateral4(c4)
        t4 = torch.cat([u5, p4], dim=1)
        t4 = self.c2f4(t4)

        u4 = F.interpolate(t4, scale_factor=2, mode='nearest')
        p3 = self.lateral3(c3)
        t3 = torch.cat([u4, p3], dim=1)
        t3 = self.c2f3(t3)

        # bottom-up
        d3 = self.down4(t3)
        p4_b = torch.cat([d3, t4], dim=1)
        p4 = self.c2f_p4(p4_b)

        d4 = self.down5(p4)
        p5_b = torch.cat([d4, p5], dim=1)
        p5 = self.c2f_p5(p5_b)

        # outputs (h3,h4,h5)
        return t3, p4, p5


# --------------------------- Detect Head ----------------------------------
class DetectHead(nn.Module):
    """Decoupled detection head.

    For each input feature map the head predicts a classification score per
    class and a distributional regression for each of the 4 box sides. The
    regression output predicts a discrete distribution (K bins) per side; the
    final distance is computed as the expectation over that distribution.
    """
    def __init__(self, in_channels: List[int], num_classes: int = 80, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.nl = len(in_channels)
        # per-scale convs
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for c in in_channels:
            # small decoupled heads: a conv to reduce and one to output
            self.cls_convs.append(nn.Sequential(
                ConvBnAct(c, c, k=3),
                nn.Conv2d(c, num_classes, kernel_size=1)
            ))
            self.reg_convs.append(nn.Sequential(
                ConvBnAct(c, c, k=3),
                nn.Conv2d(c, 4 * (reg_max + 1), kernel_size=1)
            ))

    def forward(self, feats: List[torch.Tensor], strides: List[int], training=False):
        # feats: list of [B, C_i, H_i, W_i]
        cls_outputs = []
        reg_outputs = []
        for i, x in enumerate(feats):
            cls_out = self.cls_convs[i](x)
            reg_out = self.reg_convs[i](x)
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
        if training:
            return cls_outputs, reg_outputs
        # inference: decode into boxes
        dets = []  # list of (boxes, scores, labels)
        for i, (cls_out, reg_out, s) in enumerate(zip(cls_outputs, reg_outputs, strides)):
            b, nc, h, w = cls_out.shape
            # class probs (sigmoid)
            probs = torch.sigmoid(cls_out)  # [B, num_classes, H, W]
            # reshape reg to [B, 4, K, H, W]
            K = self.reg_max + 1
            reg_out = reg_out.view(b, 4, K, h, w)
            # decode per-batch
            for bi in range(b):
                # flatten spatial
                prob_map = probs[bi].permute(1, 2, 0).reshape(-1, self.num_classes)  # [H*W, C]
                reg_map = reg_out[bi].permute(2, 3, 4, 0).reshape(-1, 4, K)  # [H*W, 4, K]
                boxes, scores, labels = decode_per_feature_map(reg_map, prob_map, s)
                dets.append((boxes, scores, labels))
        return dets


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
