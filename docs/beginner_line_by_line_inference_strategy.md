# 推理策略代码逐段讲解（小白面试版）

代码位置：`camera_ui/inference_engine.py`

这份文档解释的是桌面端实时推理策略。面试时这里最容易被问：

- 为什么 UI 不会卡？
- 为什么要“最新帧覆盖旧帧”？
- 为什么要滑动窗口推理？
- EMA 是什么？
- 0.4 置信度阈值是干嘛的？
- 检测框是不是模型直接输出的？

## 一、先看完整数据流

```text
MainWindow._on_timer()
  -> self._inference_worker.submit_frame(frame)
  -> InferenceWorker.run()
  -> BGR 转 RGB
  -> to_tensor + normalize
  -> _sliding_window_predict()
  -> 得到全图类别概率 avg_probs
  -> EMA 时序平滑
  -> argmax 得到 pred_mask
  -> max 得到 conf_map
  -> conf < 0.4 的像素归为背景
  -> result_ready.emit(...)
  -> MainWindow._on_inference_result()
  -> overlay_mask() 叠加颜色、画框、统计数量
```

面试一句话：

> 推理策略是异步 latest-frame + 512 滑窗 + 概率平均 + EMA 时序平滑 + 置信度过滤。模型输出的是语义分割概率图，检测框和数量统计是从 mask 做连通域后处理得到的。

## 二、第 1-19 行：导入依赖

第 11 行：

```python
import time as _time
```

用于统计推理耗时，也用于线程空闲时短暂 sleep。

第 12 行：

```python
import cv2
```

OpenCV，负责颜色转换、连通域分析、画框、画文字、图像混合。

第 13 行：

```python
import numpy as np
```

numpy，负责处理 mask、颜色表和置信度图。

第 15-17 行：

```python
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
```

PyTorch 推理相关：

- `torch`：创建张量、运行模型。
- `F`：softmax、pad 等函数。
- `TF`：把图像转 tensor、normalize。

第 19 行：

```python
from PyQt5.QtCore import pyqtSignal, QThread, QMutex
```

PyQt 线程相关：

- `QThread`：后台推理线程。
- `pyqtSignal`：推理完成后把结果发回 UI。
- `QMutex`：保护共享帧，避免同时读写。

## 三、第 22-29 行：定义后台推理线程

第 22 行：

```python
class InferenceWorker(QThread):
```

`InferenceWorker` 继承 `QThread`，说明它是一个后台线程类。

面试说法：

> 模型推理不放在 UI 主线程，而是放在 QThread 里异步执行，避免窗口卡顿。

第 28 行：

```python
result_ready = pyqtSignal(object, object, object, float)
```

定义一个信号，用于推理完成后传回：

```text
frame      原始图像
mask       预测类别 mask
conf_map   每个像素的置信度
latency_ms 推理耗时
```

第 29 行：

```python
error_occurred = pyqtSignal(str)
```

定义错误信号。推理出错时，把错误信息传给 UI。

## 四、第 31-49 行：初始化推理线程状态

第 33 行：

```python
self._frame = None
```

保存等待推理的最新帧。

注意，这里只保存一帧。新帧来了会覆盖旧帧。

第 34 行：

```python
self._mutex = QMutex()
```

线程锁。因为 UI 线程会提交帧，推理线程会读取帧，这是一写一读，需要锁。

第 35 行：

```python
self._running = True
```

推理线程运行标志。`run()` 里的 while 循环靠它控制。

第 36 行：

```python
self._has_work = False
```

表示当前有没有待推理的新帧。

第 38-44 行：

```python
self.model = None
self.device = None
self.crop_size = 512
self.stride = 384
self.img_mean = [0.485, 0.456, 0.406]
self.img_std = [0.229, 0.224, 0.225]
self.class_names = []
```

这些参数由 `MainWindow` 设置。

关键参数：

- `crop_size=512`：每个滑窗 patch 大小。
- `stride=384`：滑窗步长。
- `img_mean/img_std`：ImageNet 归一化参数。
- `class_names`：类别名，比如 Background、Monolayer、Fewlayer、Multilayer。

为什么 `stride=384`？

```text
512 - 384 = 128
```

说明相邻窗口有 128 像素重叠。重叠可以减少拼接缝。

第 47 行：

```python
self.ema_alpha = 0.3
```

EMA 平滑系数。

公式：

```text
ema = 0.3 * 当前帧概率 + 0.7 * 历史概率
```

注意：这是推理结果 EMA，不是训练权重 EMA。

第 48 行：

```python
self._ema_probs = None
```

保存上一轮累计的概率图。

第 49 行：

```python
self.conf_threshold = 0.4
```

置信度阈值。低于 0.4 的像素会被归为背景，用来减少散点误检。

## 五、第 51-60 行：提交帧 submit_frame

第 51 行：

```python
def submit_frame(self, frame):
```

UI 线程调用这个函数，把当前相机帧提交给推理线程。

第 57 行：

```python
self._mutex.lock()
```

加锁，防止推理线程同时读取 `_frame`。

第 58 行：

```python
self._frame = frame.copy()
```

保存一份帧拷贝。

为什么要 `.copy()`？

避免外部图像后续被修改，影响推理线程正在使用的数据。

第 59 行：

```python
self._has_work = True
```

告诉推理线程：现在有新任务了。

第 60 行：

```python
self._mutex.unlock()
```

解锁。

面试重点：

> 这里没有队列。如果上一次推理还没完成，新提交的帧会覆盖旧帧。这是 latest-frame 策略，可以避免实时系统延迟积压。

## 六、第 62-64 行：停止线程

```python
def stop(self):
    self._running = False
    self.wait(3000)
```

关闭程序时调用。

- `_running=False`：让 `run()` 循环退出。
- `wait(3000)`：最多等待 3 秒，让线程收尾。

## 七、第 66-77 行：推理线程主循环

第 66 行：

```python
def run(self):
```

`QThread.start()` 后会自动执行 `run()`。

第 67 行：

```python
while self._running:
```

只要线程没有被停止，就一直循环。

第 69-73 行：

```python
self._mutex.lock()
has_work = self._has_work
frame = self._frame
self._has_work = False
self._mutex.unlock()
```

这段是在推理线程里“取走”最新帧。

关键点：

- 先加锁，保证读 `_frame` 时 UI 不会同时写。
- `frame = self._frame` 拿到当前最新帧。
- `_has_work = False` 表示这帧已经被推理线程接收。
- 解锁。

第 75-77 行：

```python
if not has_work or frame is None or self.model is None:
    _time.sleep(0.01)
    continue
```

如果没有新帧、帧为空、或者模型还没加载，就休眠 10ms，然后继续循环。

为什么要 sleep？

避免线程一直空转占 CPU。

## 八、第 79-89 行：预处理和滑窗入口

第 80 行：

```python
t0 = _time.perf_counter()
```

记录开始时间，用于计算推理延迟。

第 82 行：

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

相机/OpenCV 图像是 BGR，模型训练时用 RGB，所以这里转成 RGB。

第 83-86 行：

```python
img_tensor = TF.normalize(
    TF.to_tensor(rgb),
    self.img_mean, self.img_std
)
```

两步：

1. `TF.to_tensor(rgb)`：把 H×W×3 的 numpy 图像转成 3×H×W 的 tensor，并缩放到 0-1。
2. `TF.normalize(...)`：按 mean/std 归一化。

为什么要归一化？

因为训练时也用了同样的归一化。推理输入要和训练输入分布一致。

第 89 行：

```python
cur_probs = self._sliding_window_predict(img_tensor)
```

进入滑动窗口推理，返回全图概率图。

注意：这里返回的是概率，不是直接 mask。

原因是后面还要做 EMA 平滑和置信度过滤。

## 九、第 91-105 行：EMA + mask + 阈值过滤

第 92-95 行：

```python
if (self._ema_probs is None
        or self._ema_probs.shape != cur_probs.shape):
    self._ema_probs = cur_probs
```

如果是第一帧，或者分辨率变了，就直接用当前概率图作为 EMA 初始值。

为什么分辨率变化要重置？

因为旧概率图 shape 和新图不一样，不能相加。

第 97-98 行：

```python
alpha = self.ema_alpha
self._ema_probs = alpha * cur_probs + (1 - alpha) * self._ema_probs
```

EMA 平滑。

含义：

```text
当前帧概率占 30%
历史概率占 70%
```

作用：

> 显微图像轻微抖动时，直接逐帧 argmax 会导致 mask 闪烁。EMA 可以让结果更稳定。

第 100 行：

```python
pred_mask = self._ema_probs.argmax(axis=0)
```

对每个像素，在类别维度上取概率最大的类别。

如果概率图形状是：

```text
4 × H × W
```

`argmax(axis=0)` 后变成：

```text
H × W
```

每个像素值是：

```text
0 背景
1 单层
2 少层
3 多层
```

第 101 行：

```python
conf_map = self._ema_probs.max(axis=0)
```

每个像素取最大类别概率，作为置信度图。

第 104-105 行：

```python
low_conf = conf_map < self.conf_threshold
pred_mask[low_conf] = 0
```

如果某个像素最高概率还低于 0.4，就把它归为背景。

作用：

> 抑制低置信度的噪声点，减少误检。

风险：

> 阈值太高可能漏掉颜色很淡的单层，阈值太低可能引入散点误检。

## 十、第 107-116 行：结果回传 UI

第 107 行：

```python
latency_ms = (_time.perf_counter() - t0) * 1000.0
```

计算本次推理耗时，单位毫秒。

第 110 行：

```python
self.result_ready.emit(frame, pred_mask, conf_map, latency_ms)
```

通过 Qt signal 把结果发回 UI 线程。

为什么不用推理线程直接改 UI？

Qt 里 UI 控件应该由主线程操作。后台线程直接改 UI 容易出问题。

第 111-116 行：

```python
except Exception as e:
    ...
    self.error_occurred.emit(f"推理错误: {e}")
```

推理出错时通过 signal 报给 UI，并限制错误提示频率，避免一直弹错误。

## 十一、第 118-169 行：滑动窗口推理

第 126 行：

```python
_, H, W = img_tensor.shape
```

读取图像高度和宽度。

第 127-130 行：

```python
num_classes = len(self.class_names)
device = self.device
crop_size = self.crop_size
stride = self.stride
```

拿到类别数、设备、窗口大小和步长。

第 134-135 行：

```python
pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
count = torch.zeros(H, W, dtype=torch.float32, device=device)
```

创建两个全图容器：

- `pred_sum`：累加每个窗口的 softmax 概率。
- `count`：记录每个像素被窗口覆盖了几次。

为什么需要两个？

因为有重叠窗口，重叠区域会被预测多次。最后要做平均：

```text
avg_probs = pred_sum / count
```

第 138-141 行：

```python
pad_h = max(0, crop_size - H)
pad_w = max(0, crop_size - W)
if pad_h > 0 or pad_w > 0:
    img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
```

如果图像小于 512，就反射填充到至少 512。

`reflect` 比黑色 padding 更自然，不会引入黑边。

第 145-152 行：

```python
ys = sorted(set(
    list(range(0, max(1, pH - crop_size + 1), stride)) +
    [max(0, pH - crop_size)]
))
xs = sorted(set(
    list(range(0, max(1, pW - crop_size + 1), stride)) +
    [max(0, pW - crop_size)]
))
```

计算所有滑窗起点。

为什么要额外加：

```python
[max(0, pH - crop_size)]
[max(0, pW - crop_size)]
```

为了保证最下边和最右边一定被覆盖。

第 154-156 行：

```python
for y in ys:
    for x in xs:
        crop = img_tensor[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
```

双重循环切窗口。

`crop` 从：

```text
3 × 512 × 512
```

变成：

```text
1 × 3 × 512 × 512
```

因为模型输入需要 batch 维度。

第 157 行：

```python
with torch.no_grad():
```

推理时不计算梯度，节省显存和计算。

第 158-160 行：

```python
out = self.model(crop)
logits = out[0] if isinstance(out, tuple) else out
probs = F.softmax(logits, dim=1)[0]
```

模型前向：

- `logits` 是模型原始输出。
- `softmax` 把 logits 转成类别概率。
- `[0]` 去掉 batch 维度。

最后 `probs` 形状是：

```text
num_classes × 512 × 512
```

第 162-165 行：

```python
y_end = min(y + crop_size, H)
x_end = min(x + crop_size, W)
pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
count[y:y_end, x:x_end] += 1
```

把当前窗口结果写回全图。

注意是 `+=`，不是 `=`。

这表示重叠区域会累加多个窗口的概率。

第 167-169 行：

```python
count = count.clamp(min=1)
avg_probs = (pred_sum / count.unsqueeze(0)).cpu().numpy()
return avg_probs
```

计算平均概率并返回。

`count.unsqueeze(0)` 是为了让 `H×W` 的 count 可以和 `C×H×W` 的 pred_sum 相除。

## 十二、第 172 行以后：mask 后处理和可视化

`overlay_mask()` 不是模型推理本身，但它是“检测结果可视化”的核心。

第 187-188 行：

```python
if visible_classes is None:
    visible_classes = set(range(1, len(class_names)))
```

如果没有指定类别筛选，就显示所有非背景类别。

第 191-201 行：

```python
color_mask = np.zeros_like(frame)
...
overlay[fg] = cv2.addWeighted(...)
```

把 mask 映射成颜色，并和原图 alpha 混合。

第 203-215 行：

```python
binary = (mask == cid).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(...)
```

对每个类别做连通域分析。

这一步是检测框来源。

面试必须说清楚：

> 检测框不是模型直接输出的，而是从语义分割 mask 的连通区域中提取外接矩形。

第 223-225 行：

```python
area = stats[i, cv2.CC_STAT_AREA]
if area < min_area:
    continue
```

过滤面积小于 500 的小噪声块。

第 228-231 行：

```python
x = stats[i, cv2.CC_STAT_LEFT]
y = stats[i, cv2.CC_STAT_TOP]
w = stats[i, cv2.CC_STAT_WIDTH]
hh = stats[i, cv2.CC_STAT_HEIGHT]
```

从连通域统计结果里拿外接框坐标。

第 233-238 行：

```python
region_mask = labels == i
avg_conf = float(conf_map[region_mask].mean()) * 100
```

计算区域平均置信度。

不是一个框的检测置信度，而是这个连通区域内所有像素置信度的平均。

第 240-270 行：

画边框、标签背景、类别名和置信度。

第 272-273 行：

```python
if count > 0:
    detection_counts[class_name] = count
```

保存每个类别检测到多少个连通区域。

第 275-333 行：

绘制左下角层数统计，例如：

```text
1L: 1
FL: 2
ML: 1
```

## 十三、面试高频问答

### Q1：为什么推理线程只处理最新帧？

答：

> 实时系统最怕排队造成延迟。显微观察更关注当前画面，如果每帧都排队推理，推理慢时结果会越来越滞后。所以采用 latest-frame 策略，新帧覆盖旧帧，牺牲逐帧完整性换低延迟。

### Q2：为什么不用整图推理？

答：

> 显微图分辨率可能很高，整图推理显存和耗时不可控，而且模型训练时输入是 512 crop。滑窗可以保持训练和部署尺度一致，也能处理任意分辨率。

### Q3：为什么重叠区域做概率平均？

答：

> 如果直接覆盖，窗口边界处容易出现拼接痕迹。概率平均可以融合多个窗口对同一像素的预测，让边界更平滑。

### Q4：EMA 是什么？

答：

> 这里的 EMA 是对推理概率图做时序平滑，不是训练权重 EMA。公式是当前概率占 0.3，历史概率占 0.7，用来减少 mask 闪烁。

### Q5：0.4 阈值有什么作用？

答：

> 每个像素 softmax 最大概率就是置信度。低于 0.4 的像素归为背景，可以过滤低置信噪声和散点误检。阈值太高会漏检，太低会误检。

### Q6：检测框是模型输出的吗？

答：

> 不是。模型输出的是像素级分割 mask。检测框是对每个类别的 mask 做 connected components 后得到连通区域，再画外接矩形。

## 十四、最推荐背的完整回答

> 推理模块用 `InferenceWorker(QThread)` 放在后台执行，UI 线程只负责显示和提交最新帧。UI 调 `submit_frame()` 时，推理线程只保存一张 `_frame`，新帧会覆盖旧帧，这是 latest-frame 策略，可以避免推理队列积压造成延迟。推理线程拿到帧后，先把 OpenCV 的 BGR 转成 RGB，再转 tensor 并做 ImageNet 归一化。随后进入 512×512 滑动窗口推理：根据 crop size 和 stride 生成窗口起点，每个 patch 单独前向，softmax 得到类别概率后累加到 `pred_sum`，同时用 `count` 记录每个像素被覆盖次数，最后 `pred_sum/count` 得到整图平均概率。之后对概率图做 EMA 时序平滑，用 argmax 得到 mask，用 max 得到 conf_map，再把低于 0.4 的低置信像素归为背景。最终通过 Qt signal 把 frame、mask、conf_map 和 latency 发回 UI，UI 再做颜色叠加、连通域检测框、平均置信度和层数统计。
