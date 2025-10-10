
**A. 基于 YOLOv8n 的流产检测**

该任务选用 YOLOv8 系列中最轻量的 YOLOv8n，在兔舍环境中检测流产事件。模型以 RGB 图像为输入，输出血渍、胎仔等视觉指标的位置。

如图 2 所示，YOLOv8n 的主干网络采用 Cross Stage Partial（CSP）结构：  
- 先经过若干卷积层，再堆叠一系列 C2f 模块，通过瓶颈块拆分与合并特征通道；  
- 在最深层，Spatial Pyramid Pooling Fast（SPPF）模块聚合多种感受野的上下文信息，增强模型对细粒度纹理和高层语义的捕捉能力。

颈部网络采用双向路径：  
1. 自底向上：将主干深层特征图放大，与浅层特征拼接；  
2. 自顶向下：把融合结果通过下采样再次送入更高层。  
这种双向流动使每个输出特征图同时包含浅层的空间细节和深层的上下文信息。

检测头在 3 个融合特征图（H3、H4、H5）上运行，每个尺度分别用并行卷积分支生成分类得分和边界框预测，实现对不同尺寸血渍与胎仔的精确定位与分类。

---

**B. 基于 YOLOv8n-P2 的饲料泼洒检测**

散落饲料颗粒尺寸极小，且兔舍环境杂乱，检测难度较大。为此，采用专为超小目标检测设计的 YOLOv8n-P2 模型。

如图 3 所示，YOLOv8n-P2 在 YOLOv8n 基础上：  
- 在颈部新增一条检测分支，将最浅层 C2f 输出进一步上采样，使其空间分辨率与更高层对齐；  
- 这些高分辨率特征继续参与颈部自底向上和自顶向下的拼接融合，生成保留更多细节的特征图；  
- 最终检测头在 4 个尺度（P2、P3、P4、P5）上运行，分别对应小、中小、中大、大目标。  

通过显式引入 P2 特征，YOLOv8n-P2 显著提升了对小目标的敏感度，可在第二任务中更精准地定位散落饲料颗粒。




YOLOv8 的 Backbone、Neck 和 Head 结构详细如下：

**Backbone（主干）**
YOLOv8 的 Backbone 采用了经过优化的 CSPDarknet 结构，并引入了 C2f（Cross Stage Partial with two fusion）模块，提升特征提取能力。Backbone 负责从输入图片中提取多层次的特征，输出多尺度的 feature maps，为后续的检测提供基础。C2f 模块相比上一代的 C3 模块，结构更简洁，融合特征的能力更强[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-damo-yolo)[(4)](https://docs.ultralytics.com/compare/yolov8-vs-efficientdet)[(5)](https://docs.ultralytics.com/compare/yolov8-vs-yolo11)。

**Neck（颈部）**
YOLOv8 的 Neck 部分主要完成特征融合。通过 C2f 模块和特征金字塔结构，将 Backbone 的多尺度特征进行融合，增强对不同尺寸目标的检测能力。Neck 的设计可以有效整合浅层与深层的语义信息，提高小目标和大目标的检测效果[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-damo-yolo)[(4)](https://docs.ultralytics.com/compare/yolov8-vs-efficientdet)。

**Head（检测头）**
YOLOv8 的 Head 采用 anchor-free（无锚框）解耦头结构，将分类和回归任务分开处理。检测头直接对每个像素点进行类别和边界框的预测，简化了匹配过程。解耦式设计提升了检测精度，并且由于去除了 anchor 机制，后处理流程更加高效，部分场景可省略 NMS 操作[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-damo-yolo)[(4)](https://docs.ultralytics.com/compare/yolov8-vs-efficientdet)[(5)](https://docs.ultralytics.com/compare/yolov8-vs-yolo11)。

总结：  

- **Backbone**：CSPDarknet + C2f，负责高效提取多层次特征。  
- **Neck**：特征融合，提升多尺度检测能力。  
- **Head**：anchor-free 解耦头，分类和回归分离，提升精度和速度。

这些设计让 YOLOv8 兼顾了速度、精度和多任务适应性，是现代目标检测模型的优秀代表[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-damo-yolo)[(4)](https://docs.ultralytics.com/compare/yolov8-vs-efficientdet)[(5)](https://docs.ultralytics.com/compare/yolov8-vs-yolo11)。



![image-20251009094430430](/home/assumeengage/.config/Typora/typora-user-images/image-20251009094430430.png) 



![image-20251009094711906](/home/assumeengage/.config/Typora/typora-user-images/image-20251009094711906.png) 

YOLOv8 的 Backbone 采用了 CSPDarknet 结构，并引入了 C2f（Cross Stage Partial with two fusion)模块:

------

**1. CSPDarknet 原理说明**
CSPDarknet（Cross Stage Partial Darknet）是一种改进型的卷积神经网络主干结构。其核心思想是将特征图在通道维度上分为两部分，一部分直接跳跃到后面，另一部分经过多个残差块或卷积处理后再与前半部分融合。这种结构可以有效缓解梯度消失、提升网络表达能力，同时减少参数量，提高推理速度[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-efficientdet)。

------

**2. C2f 模块结构说明**
C2f（Cross Stage Partial with two fusion）是 YOLOv8 中 backbone 内部的基本模块。C2f 模块的本质是将输入特征分流，一部分直接传递，另一部分经过两次卷积融合后再与跳跃连接的特征拼接。这样既保留了原始信息，又引入了更丰富的特征表达，提升了模型的非线性能力和泛化能力。
C2f 结构相比 YOLOv5 中的 C3 模块更加高效，参数更少，推理速度更快[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-efficientdet)[(7)](https://github.com/ultralytics/ultralytics/issues/6122)。

![image-20251009092904918](/home/assumeengage/.config/Typora/typora-user-images/image-20251009092904918.png)

------

**总结**
CSPDarknet + C2f 的 backbone 设计让 YOLOv8 能在不同层次提取到丰富的语义和空间信息，既保证了高效计算，又兼顾了准确率和推理速度，是目前目标检测主干网络的主流设计之一[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-efficientdet)[(7)](https://github.com/ultralytics/ultralytics/issues/6122)。

![image-20251009094310395](/home/assumeengage/.config/Typora/typora-user-images/image-20251009094310395.png) 

YOLOv8 的 Neck（颈部）核心作用是进行特征融合，从而提升多尺度目标的检测能力。其原理和机制如下：

![image-20251009103748429](/home/assumeengage/.config/Typora/typora-user-images/image-20251009103748429.png)



1. **多尺度特征融合（FPN思想）**
   Neck 结构通过融合主干网络（Backbone）不同层次输出的特征图，将高分辨率的浅层特征（包含丰富的细节信息）和低分辨率的深层特征（包含强语义信息）进行整合。这样，模型既能检测小目标（依赖细节）也能检测大目标（依赖语义）[(1)](https://docs.ultralytics.com/guides/model-yaml-config)[(4)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)。
2. **典型结构与实现**
   Neck 常用的实现方式包括利用上采样（Upsample）、下采样（Conv, stride=2）、Concat（特征拼接）、C2f 等模块，按照类似 FPN（特征金字塔网络）的方式组织。
   举例说明：

```yaml
- [-1, 1, nn.Upsample, [None, 2, nearest]]      # 上采样
- [[-1, 6], 1, Concat, [1]]                    # 与前面某层特征拼接
- [-1, 3, C2f, [256]]                          # 融合后再卷积处理
```

这种结构可以让来自不同尺度的特征在后续的检测头共享，从而提升对各类目标的检测能力[(1)](https://docs.ultralytics.com/guides/model-yaml-config)。

1. **特征流动与跳跃连接**
   Neck 中大量使用了跳跃连接（skip connection）、多输入融合（multi-input fusion）等方式，保证梯度流动顺畅，防止信息丢失，并提升深层网络的训练稳定性[(1)](https://docs.ultralytics.com/guides/model-yaml-config)。
2. **优势**

- 检测不同尺寸目标时都能获得适合的特征表示
- 增强小目标检测能力
- 改善模型整体鲁棒性和泛化能力

综上，YOLOv8 的 Neck 通过多尺度特征融合、上采样、拼接和高效卷积等机制，让模型能够同时兼顾大目标和小目标的检测效果，是提升检测性能的关键结构[(1)](https://docs.ultralytics.com/guides/model-yaml-config)[(4)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)。





YOLOv8 的 Head（检测头）是整个检测模型的关键部分，负责将上游 Backbone 和 Neck 输出的融合特征，最终转换为目标检测的输出结果（如边界框、类别概率等）。其底层结构与工作机制如下：

------

**1. Anchor-Free 解耦头（Decoupled Head）**
YOLOv8 的 Head 采用 anchor-free（无锚框）设计，直接对每个特征点预测目标框（bbox）和类别概率，不再依赖传统的 anchor box。检测头将分类和回归任务分离，分别用不同的卷积分支处理，提高了检测精度和训练稳定性[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(2)](https://docs.ultralytics.com/compare/yolov8-vs-damo-yolo)。

------

**2. 结构实现**
以 YOLOv8 检测头的 Detect 类为例，其底层实现（以 PyTorch 为例）包含如下核心流程[(3)](https://docs.ultralytics.com/reference/nn/modules/head)：

- 输入是来自 Neck 的多尺度特征（通常为3组不同分辨率的特征图）。
- 每个尺度的特征分别经过卷积分支，输出 bbox 回归信息和类别概率。
- 多尺度输出会被拼接（cat）在一起，形成最终的检测输出。

底层代码示例（节选）：

```python
def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
    """Concatenate and return predicted bounding boxes and class probabilities."""
    if self.end2end:
        return self.forward_end2end(x)
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    y = self._inference(x)
    return y if self.export else (y, x)
```

[(1)](https://docs.ultralytics.com/reference/nn/modules/head)

------

**3. 输出解释**  

- 检测头输出的张量包括每个位置的 bbox（边界框）和类别概率。
- 在推理阶段，经过后处理（如 NMS），得到最终的检测结果。

------

**4. 多任务支持**
YOLOv8 的 Head 还支持实例分割、关键点检测（姿态估计）、定向框（OBB）等多任务，均采用类似的解耦设计，每种任务有独立的分支和输出处理方式[(3)](https://docs.ultralytics.com/reference/nn/modules/head)。

------

**总结**
YOLOv8 的检测头通过 anchor-free 解耦结构、分支卷积、特征拼接等机制，直接高效地从特征图生成检测结果，具备高精度、高速度以及良好的可扩展性[(1)](https://docs.ultralytics.com/compare/damo-yolo-vs-yolov8)[(3)](https://docs.ultralytics.com/reference/nn/modules/head)。











怎么讲这个东西





我想一下啊。就是我先抛出一个问题，就是想讲一下，我去年做的那个大创，是一个针对智慧养殖方面的，就是我主要负责的是兔子异常情况的识别，就是用一些深度学习的模型去识别兔子的异常情况，包括但不限于兔子流产，刨料等行为，我们做的是一个自动喂料车，也就是说，上面的能搭载的算力是有限的，我不可能放两张卡放在面跑，我只能使用边侧的ai计算单元去搞这个，也就是说在条件受限的情况下，VLM等等一系列比较吃算力吃显存的模型我是用不了的，所以工程上来说，只有Yolo是最合适的，我相信大家对Yolo应该比较熟悉，这个我就不过多赘述，但是，针对兔子刨料行为，是这样子，兔子的饲料是非常小的，也就是说，这是一个小目标检测任务，并且，网上是绝对没有兔子流产混合刨料这个数据集的，所以说，主要任务是亲自去兔场拍数据集，并解决一下yolo针对小目标检测这个问题，当时比较早嘛，所以那时候latest模型是yolov11，但是它还没有开源，所以我的基模就必须是Yolov8n，因为不可能再小了，我测过yolov8s，相比于nomal ，small确实是效果太差了，这个结论是怎么得出来的，是我先微调了这几个模型 n m s l x，测试了一下，大概s mAP@0.5 是0.67左右，而n 是上了0.7，所以考虑到大小和模型性能，于是选择了yolov8n作为基模，然后就是，我先讲一下这个yolov8n这个模型的整体架构，还有内部的一些细节，之后呢我再对比的讲一下我改的部分，就是针对这个小目标检测，我对yolov8n改了什么，有什么提升，



对了，把这个模型部署在atlas 200dk上也花了很多时间，踩了很多坑，是这样子，atlas 200 dk是华为的升腾系列的边侧ai计算单元，他虽然算力比较搞11TOPS的算力，这是相比起nvidia的Jetson Nano 或者 TX2,那几个9TOPS或者16TOPS算力的东西，但是，然后这个东西实际上呢，是arm架构的，Ascend 310 AI 处理器，硬件主要适配的ai框架，也是Mindspore，不是Pytorch，而华为社区一贯是没有文档的，就很恼火，固件版本也是有很大的差异的，像是我是使用的pytorch 训练的模型，但是，权重文件best.pt是要使用华为ATC这个转化工具去进行转化，我的固件版本是5.0.4ascend这个版本的，这个是唯一能跑通的版本，但是ATC官方给的tookit最老的是6.0.1,需烧成6.0.1然后要回滚，里面的kernal modules在摄像头drivers这几个上出现了大大小小的问题，i2c的驱动出现了很大的问题，最后也是找了个mikrotik的网卡重新编译的内核模块才好了

但是最后呢，也是都解决了，想说的是什么呢，就是最后落地这个兔子刨料行为的小目标检测是非常困难的






