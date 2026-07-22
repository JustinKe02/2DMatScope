# 2DMatScope 面试速记版

适合面试前 30-60 分钟快速复盘。完整版在 `docs/project1_2dmatscope_interview_defense.md`，这份只保留最容易被问到的主线。

## 1. 一句话定位

2DMatScope 是一个面向二维材料显微图像的实时语义分割与检测可视化系统，用显微相机采集 MoS2/WS2 等材料图像，用 RepELA-Net 判断背景、单层、少层、多层区域，再把分割 mask 转成检测框、置信度、层数统计、保存结果和录像。

不要把它说成“我训练了一个模型”。更准确的说法是：我把相机采集、图像增强、轻量化分割模型、异步推理、结果叠加、保存和录制做成了一个完整应用闭环。

## 2. 30 秒项目介绍

我做的是二维材料显微图像实时识别系统，目标是辅助判断 MoS2、WS2 等二维材料的单层、少层和多层区域。底层任务是语义分割，因为材料区域形状不规则，边界和面积信息比普通框更重要。工程上我用 ctypes 接 Toupcam SDK，用 PyQt5 做桌面端，用 QThread 把相机采集、UI 刷新和模型推理分开，推理采用最新帧覆盖旧帧、512 滑窗、EMA 平滑和置信度过滤。模型是 RepELA-Net，结合 RepConv、ELA、DW-MFF 和边界增强模块。最终系统支持双窗口显示、类别筛选、检测框/置信度/层数统计、图片保存和 AVI 录制。

## 3. 代码地图

- 相机 SDK 封装: `camera_ui/camera_controller.py`
- UI、模型加载、录制和保存: `camera_ui/main_window.py`
- 后台推理、滑窗、EMA、mask 可视化: `camera_ui/inference_engine.py`
- 图像增强: `camera_ui/enhancement.py`
- ROI 交互显示: `camera_ui/image_label.py`
- 模型总结构: `RepELA-Net/models/repela_net.py`
- RepConv 重参数化: `RepELA-Net/models/rep_conv.py`
- ELA 轻量注意力: `RepELA-Net/models/ela_block.py`
- DW-MFF 和边界增强: `RepELA-Net/models/decoder.py`
- 数据集和增强: `RepELA-Net/datasets/mos2_dataset.py`
- Focal+Dice loss: `RepELA-Net/utils/losses.py`
- mIoU/F1/Pixel Acc: `RepELA-Net/utils/metrics.py`
- 训练主流程: `RepELA-Net/tools/train.py`

## 4. 技术链路

相机链路：
`Toupcam_EnumV2 -> Toupcam_Open -> put_eSize -> StartPullModeWithCallback -> PullImage -> BGRA 转 BGR -> UI/推理`

实时链路：
`相机后台线程缓存最新帧 -> UI QTimer 每 16ms 读取显示 -> 推理 QThread 异步处理最新帧 -> signal 回 UI 叠加显示`

推理链路：
`BGR 转 RGB -> to_tensor + ImageNet normalize -> 512 滑窗 -> softmax 概率平均 -> EMA=0.3 -> conf<0.4 归背景 -> mask`

可视化链路：
`mask 映射颜色 -> 和原图 alpha 混合 -> connected components -> 小区域过滤 -> 外接框 + 平均置信度 + 层数统计`

模型链路：
`RGB -> 4 通道增强/占位 -> Stem -> RepConv Stage1/2 -> ELA Stage3/4 -> DW-MFF Decoder -> Boundary Enhancement -> 4 类 logits`

## 5. 必背解释

为什么不用 YOLO：
二维材料层数区域形状不规则，而且边界、面积和像素级区域很重要。YOLO 只给框，不能准确表达单层、少层、多层区域，所以底层用语义分割；检测框只是由 mask 后处理得到的展示形式。

为什么要多线程：
UI 主线程不能做相机拉帧、模型前向这类耗时任务，否则窗口会卡。系统把相机抓帧、UI 刷新、模型推理分开，推理慢时也只影响检测结果更新频率，不影响相机画面显示。

为什么最新帧覆盖旧帧：
实时系统最怕排队造成延迟。显微观察更关心当前画面，不需要把历史每一帧都算完。旧帧被覆盖会丢帧，但能保持低延迟。

为什么滑窗：
训练是 512 crop，高分辨率显微图整图推理显存和速度都不可控。滑窗能处理任意分辨率，并保持训练/推理尺度一致。重叠区域做概率平均，减少拼接缝。

EMA=0.3 是什么：
这是推理概率图的时序平滑，不是训练权重 EMA。公式是 `0.3 * 当前概率 + 0.7 * 历史概率`，用于减少 mask 闪烁。

0.4 置信度阈值是什么：
每个像素 softmax 最大值是置信度，低于 0.4 的像素归为背景，用于过滤低置信噪声。阈值太高会漏检淡单层，太低会增加散点误检。

RepConv 是什么：
训练时有 3x3、1x1、identity+BN 多分支，表达能力更强；推理前把 Conv+BN 和各分支融合成一个 3x3 卷积，推理没有额外分支开销。

ELA 是什么：
ELA 是 Efficient Linear Attention。标准 self-attention 对高分辨率是 O(N²)，ELA 用线性注意力降低复杂度，并放在深层低分辨率特征上补全全局上下文。

DW-MFF 是什么：
DW-MFF 动态融合多尺度特征。浅层有边界细节，深层有语义信息，动态权重让网络学习不同尺度的重要性。

为什么 Focal+Dice：
背景像素占比高，单层/少层像素少。Focal 降低易分类背景影响，Dice 关注区域重叠，更适合分割和类别不均衡。

为什么 mIoU 比 Pixel Acc 重要：
背景很多时 Pixel Acc 容易很高，但不代表单层/少层识别好。mIoU 按类别计算交并比后平均，更能反映少数类效果。

## 6. 常见追问短答

Q: 你的项目到底是检测还是分割？
A: 底层是语义分割，检测框是从分割 mask 做连通域分析得到的可视化结果。

Q: 显示 FPS 是模型 FPS 吗？
A: 不是。显示 FPS 是相机/UI 刷新速度；推理延迟是模型异步处理一帧的耗时。

Q: Graphene GUI 完全支持吗？
A: 训练/迁移代码支持 Graphene 类映射，但当前桌面端默认四分类，Graphene 接入 GUI 还需要动态类别名、颜色表和模型 head 配置。

Q: 高分辨率为什么慢？
A: 1280x960 需要更多 512 crop，比如 stride=384 时大约 3x3 个窗口；每个窗口都要前向，所以延迟上升。

Q: 项目不足是什么？
A: GUI 和研究代码耦合偏强；滑窗逻辑在训练、评估、GUI 中有重复；Graphene 类别配置还没完全动态化；高分辨率下需要 ONNXRuntime/TensorRT/RKNN 或量化进一步优化。

## 7. 面试时别踩的坑

- 不要说 Toupcam SDK 是自己写的；你写的是 Python 封装和应用层控制。
- 不要说检测框是模型直接输出；模型输出的是 mask。
- 不要把显示 FPS 当成模型推理 FPS。
- 不要说 RepELA-Net 精度全面超过所有 baseline；更稳妥地说它强调轻量化和可部署性。
- 不要说 Graphene 在 GUI 里已经零配置完整支持。
