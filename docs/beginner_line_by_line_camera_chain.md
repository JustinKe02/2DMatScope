# 相机链路代码逐行讲解（小白面试版）

这份文档只讲相机链路。面试时相机链路可以证明你不仅会训练模型，还能把真实硬件接入 Python/PyQt 系统。

建议背诵顺序：

1. `camera_ui/sdk_types.py`：定义 ctypes 调 SDK 需要的常量、结构体、回调类型。
2. `camera_ui/camera_controller.py`：真正打开相机、拉帧、调曝光白平衡、缓存最新帧。
3. `camera_ui/main_window.py`：UI 点击按钮后调用 CameraController，并用 QTimer 刷新显示。

## 一、先理解整体链路

完整链路是：

```text
用户点击 Start Camera
  -> MainWindow._start_camera()
  -> CameraController.open()
  -> ctypes 加载 nncam.dll
  -> 声明 Toupcam SDK 函数签名
  -> Toupcam_EnumV2 枚举相机
  -> Toupcam_Open 打开相机
  -> Toupcam_put_eSize 设置分辨率
  -> Toupcam_StartPullModeWithCallback 启动采集
  -> SDK callback 通知有新图
  -> 后台抓帧线程调用 Toupcam_PullImage 拉图
  -> BGRA 转 OpenCV BGR
  -> UI QTimer 读取最新帧显示
  -> 连续检测时把原始帧交给推理线程
```

面试一句话：

> 相机 SDK 是 C 接口，我用 ctypes 加载 `nncam.dll` 并声明函数签名。采集上采用 pull mode callback，callback 只通知有新图，后台线程再拉图并缓存最新帧，UI 主线程通过 QTimer 读取最新帧显示，所以相机采集、UI 刷新和模型推理解耦。

## 二、`sdk_types.py` 逐行讲解

代码位置：`camera_ui/sdk_types.py`

第 1-3 行：

```python
"""
SDK 常量和 ctypes 结构体定义 — Toupcam / Nncam SDK
"""
```

这是文件说明。它告诉别人：这个文件不是控制相机的业务逻辑，而是专门放 SDK 常量、结构体、回调类型。

第 5 行：

```python
import ctypes
```

导入 Python 标准库 `ctypes`。它的作用是让 Python 调用 C 动态库，也就是这里的 `nncam.dll`。

面试说法：

> ctypes 是 Python 和 C SDK 之间的桥。相机厂商给的是 DLL，不是 Python 包，所以要用 ctypes 调。

第 6 行：

```python
from ctypes import *
```

把 ctypes 里的类型直接导入，比如 `c_int`、`c_uint`、`c_void_p`、`Structure`、`POINTER`。

这些类型是为了描述 C 函数的参数和返回值。

第 7 行：

```python
import os as _os
```

导入 `os`，取别名 `_os`。这里主要用于拼接路径、判断 DLL 文件是否存在。

第 13 行：

```python
def _find_dll():
```

定义一个函数，用来自动查找 `nncam.dll`。

为什么要自动找？

因为不同电脑上 DLL 可能放在不同目录。如果写死一个路径，换电脑就容易报错。

第 15 行：

```python
base = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), '..'))
```

这句拿到项目根目录附近的路径。

拆开理解：

- `__file__`：当前文件 `sdk_types.py` 的路径。
- `_os.path.dirname(__file__)`：当前文件所在目录，也就是 `camera_ui`。
- `..`：上一层目录，也就是项目根目录 `2DMatScope`。
- `normpath`：把路径整理成标准格式。

第 16-20 行：

```python
candidates = [
    _os.path.join(base, 'nncamsdk.20171211', 'nncamsdk.20171211', 'win', 'x64', 'nncam.dll'),
    _os.path.join(base, 'drivers', 'x64', 'nncam.dll'),
    r'F:\ImageView\nncamsdk.20171211\nncamsdk.20171211\win\x64\nncam.dll',
]
```

这是候选 DLL 路径列表。程序会按顺序尝试：

1. 项目内 SDK 目录。
2. 项目内 drivers 目录。
3. 老电脑上的兼容路径。

第 21-23 行：

```python
for p in candidates:
    if _os.path.isfile(p):
        return p
```

遍历所有候选路径，只要找到真实存在的 `nncam.dll`，就返回这个路径。

第 24-27 行：

```python
raise FileNotFoundError(...)
```

如果所有路径都找不到 DLL，就抛出错误，并告诉用户应该把 SDK 放在哪里。

第 29 行：

```python
DLL_PATH = _find_dll()
```

程序启动导入这个文件时，就会立刻查找 DLL，并把路径保存到 `DLL_PATH`。

第 31 行：

```python
TOUPCAM_MAX = 16
```

最多枚举 16 个相机设备。后面创建设备数组会用它。

第 32-34 行：

```python
TOUPCAM_EVENT_IMAGE = 0x0004
TOUPCAM_EVENT_ERROR = 0x0080
TOUPCAM_EVENT_DISCONNECTED = 0x0081
```

这是 SDK 事件常量：

- `TOUPCAM_EVENT_IMAGE`：有新图像。
- `TOUPCAM_EVENT_ERROR`：相机错误。
- `TOUPCAM_EVENT_DISCONNECTED`：相机断开。

项目里最常用的是 `TOUPCAM_EVENT_IMAGE`，表示可以拉一帧图像了。

第 36-41 行：

```python
TOUPCAM_TEMP_DEF = 6503
TOUPCAM_TEMP_MIN = 2000
TOUPCAM_TEMP_MAX = 15000
TOUPCAM_TINT_DEF = 1000
TOUPCAM_TINT_MIN = 200
TOUPCAM_TINT_MAX = 2500
```

这是白平衡相关参数：

- `TEMP` 是色温。
- `TINT` 是色调偏移。

二维材料显微图像很依赖颜色差异，所以白平衡很重要。

面试说法：

> 单层、少层、多层的差异很多时候体现在颜色和亮度上，所以系统提供色温和 Tint 调节。

第 43 行：

```python
CALLBACK_TYPE = CFUNCTYPE(None, c_uint, c_void_p)
```

定义相机事件回调函数类型。

含义是：

```text
返回值: None
参数1: c_uint，事件类型
参数2: c_void_p，用户上下文指针
```

第 44 行：

```python
AWB_TT_CALLBACK_TYPE = CFUNCTYPE(None, c_int, c_int, c_void_p)
```

定义自动白平衡回调类型。

含义是：

```text
返回值: None
参数1: c_int，色温
参数2: c_int，Tint
参数3: c_void_p，上下文指针
```

第 47-48 行：

```python
class ToupcamResolution(Structure):
    _fields_ = [("width", c_uint), ("height", c_uint)]
```

定义 C 结构体 `ToupcamResolution`。

它包含宽度和高度，用来描述相机支持的分辨率。

第 50-57 行：

```python
class ToupcamModelV2(Structure):
    _fields_ = [...]
    _pack_ = 8
```

定义相机型号结构体。里面有：

- `name`：型号名。
- `flag`：功能标志。
- `maxspeed`：最大速度。
- `preview` / `still`：预览/拍照能力。
- `xpixsz` / `ypixsz`：像元尺寸。
- `res`：支持的分辨率数组。

`_pack_ = 8` 表示按照 8 字节对齐，必须和 C SDK 的结构体内存布局一致。

第 59-63 行：

```python
class ToupcamInstV2(Structure):
    _fields_ = [
        ("displayname", c_wchar * 64), ("id", c_wchar * 64),
        ("model", POINTER(ToupcamModelV2))
    ]
```

定义相机实例结构体。枚举相机时 SDK 会把设备信息写到这个结构体里。

关键字段：

- `displayname`：显示给用户看的名字。
- `id`：打开相机时真正用的设备 ID。
- `model`：指向型号信息的指针。

第 65-69 行：

```python
class RECT(Structure):
    _fields_ = [
        ("left", c_int), ("top", c_int),
        ("right", c_int), ("bottom", c_int)
    ]
```

定义矩形结构体，用于自动曝光 ROI 和自动白平衡 ROI。

ROI 的意思是 Region of Interest，也就是感兴趣区域。

## 三、`camera_controller.py` 逐行讲解

代码位置：`camera_ui/camera_controller.py`

第 1-8 行是文件说明。

重点是三句话：

1. Toupcam SDK 是 C 动态库，Python 用 ctypes 调用。
2. callback 只通知有图，不在 callback 里拉图。
3. 相机线程只保存最新帧，UI 定时器读取最新帧。

第 10 行：

```python
import ctypes
```

导入 ctypes，用于加载 DLL。

第 11 行：

```python
from ctypes import *
```

导入 C 类型，例如 `c_uint`、`c_void_p`、`byref`、`POINTER`。

第 12 行：

```python
import numpy as np
```

导入 numpy。后面要把 C buffer 转成 numpy 图像数组。

第 13 行：

```python
import cv2
```

导入 OpenCV。后面用它做颜色格式转换：BGRA 转 BGR。

第 14 行：

```python
import threading
```

导入 Python 线程库，用于后台抓帧线程。

第 15 行：

```python
import time
```

用于线程空闲时短暂 sleep，避免 CPU 空转。

第 17-23 行：

```python
from .sdk_types import (...)
```

从 `sdk_types.py` 导入前面定义的 DLL 路径、SDK 常量、回调类型、结构体。

这说明：

```text
sdk_types.py 负责定义 C SDK 类型
camera_controller.py 负责真正调用 SDK
```

第 26 行：

```python
class CameraController:
```

定义相机控制类。

面试说法：

> 我把底层 C SDK 封装成了 Python 类，上层 UI 不直接接触 SDK 细节，只调用 `open`、`close`、`get_latest_frame`、`set_exposure_time` 这些方法。

第 33 行：

```python
def __init__(self):
```

构造函数。创建 `CameraController()` 时自动执行，用来初始化所有状态。

第 34 行：

```python
self.lib = None
```

保存 DLL 对象。相机打开后它会变成 `ctypes.CDLL(DLL_PATH)`。

第 35 行：

```python
self.handle = None
```

保存相机句柄。句柄可以理解为 SDK 返回的“相机对象指针”。之后所有 SDK 操作都需要它。

第 36-37 行：

```python
self.width = 0
self.height = 0
```

保存相机当前输出分辨率。

第 38 行：

```python
self.image_ready = False
```

标记是否有新图像。SDK callback 收到图像事件后会把它置为 True。

第 39 行：

```python
self.connected = False
```

标记相机是否已连接。

第 42-48 行：

```python
self.auto_expo = True
self.expo_time_us = 70000
self.expo_gain = 100
self.expo_time_min = 50
self.expo_time_max = 2000000
self.gain_min = 100
self.gain_max = 500
```

曝光和增益相关状态。

注意：

- `expo_time_us` 单位是微秒。
- UI 里通常显示毫秒，所以 UI 设置时会乘以 1000。
- `expo_time_max = 2000000` 微秒，也就是 2000ms。

第 51-52 行：

```python
self.temp = TOUPCAM_TEMP_DEF
self.tint = TOUPCAM_TINT_DEF
```

初始化白平衡色温和 Tint。

第 57-58 行：

```python
self._event_cb = CALLBACK_TYPE(self._on_event)
self._awb_cb = AWB_TT_CALLBACK_TYPE(self._on_awb)
```

把 Python 方法包装成 C SDK 能调用的回调函数指针。

这是 ctypes 调 callback 的关键。

为什么要存在 `self` 上？

因为如果写成局部变量，函数结束后可能被 Python 垃圾回收。SDK 下次回调时访问一个已经失效的函数指针，程序可能崩溃。

这是面试加分点。

第 61 行：

```python
self._grabber_thread = None
```

保存后台抓帧线程对象。

第 62 行：

```python
self._grabber_running = False
```

控制后台抓帧线程是否继续运行。

第 63 行：

```python
self._latest_frame = None
```

保存最新一帧图像，格式是 OpenCV BGR numpy 数组。

第 64 行：

```python
self._frame_lock = threading.Lock()
```

创建线程锁。

为什么需要锁？

后台线程在写 `_latest_frame`，UI 线程在读 `_latest_frame`。如果没有锁，可能读到一半被写入打断，造成数据不一致。

第 65-66 行：

```python
self._frame_seq = 0
self._last_read_seq = -1
```

帧序号机制。

- `_frame_seq`：每来一帧就加 1。
- `_last_read_seq`：UI 上一次读到的帧编号。

如果两个值相同，说明没有新帧。

第 68 行：

```python
def _on_event(self, event, ctx):
```

相机事件回调函数。SDK 有事件时会调用它。

第 71-72 行：

```python
if event == TOUPCAM_EVENT_IMAGE:
    self.image_ready = True
```

如果事件是“有新图像”，就把 `image_ready` 置为 True。

注意这里没有拉图。

面试说法：

> callback 只做轻量通知，不做图像处理。真正拉图放到后台线程里。

第 73-74 行：

```python
elif event == TOUPCAM_EVENT_DISCONNECTED:
    self.connected = False
```

如果相机断开，就标记未连接。

第 76-78 行：

```python
def _on_awb(self, nTemp, nTint, ctx):
    self.temp = nTemp
    self.tint = nTint
```

自动白平衡完成后，SDK 会回调新的色温和 Tint。程序保存这些值，UI 后面可以显示。

第 80 行：

```python
def open(self, resolution_index=0):
```

打开相机。`resolution_index` 表示第几个分辨率档位。

第 82 行：

```python
self.lib = ctypes.CDLL(DLL_PATH)
```

加载 `nncam.dll`。

这一步完成后，Python 可以通过 `self.lib.Toupcam_Open` 这种方式访问 DLL 里的 C 函数。

第 86-94 行：

```python
self.lib.Toupcam_EnumV2.restype = c_uint
self.lib.Toupcam_Open.restype = c_void_p
self.lib.Toupcam_Open.argtypes = [c_wchar_p]
...
```

声明基础 SDK 函数签名。

这里有两个关键词：

- `argtypes`：函数参数类型。
- `restype`：函数返回值类型。

举例：

```python
self.lib.Toupcam_Open.restype = c_void_p
self.lib.Toupcam_Open.argtypes = [c_wchar_p]
```

表示：

```text
Toupcam_Open 接收一个宽字符串设备 ID
返回一个 void* 相机句柄
```

面试说法：

> ctypes 调 C 函数时必须把函数签名声明准确，尤其是指针、字符串和回调类型，否则可能调用失败甚至崩溃。

第 97-104 行：

```python
self.lib.Toupcam_put_AutoExpoEnable.argtypes = ...
...
self.lib.Toupcam_get_ExpoAGainRange.argtypes = ...
```

声明曝光和增益相关函数。

`put` 一般表示设置，`get` 一般表示读取。

第 107-109 行：

```python
self.lib.Toupcam_put_TempTint.argtypes = ...
self.lib.Toupcam_get_TempTint.argtypes = ...
self.lib.Toupcam_AwbOnePush.argtypes = ...
```

声明白平衡相关函数：

- 设置色温/Tint。
- 读取色温/Tint。
- 一键自动白平衡。

第 112-115 行：

```python
self.lib.Toupcam_put_AEAuxRect.argtypes = ...
self.lib.Toupcam_put_AWBAuxRect.argtypes = ...
```

声明 ROI 相关函数：

- `AEAuxRect`：自动曝光区域。
- `AWBAuxRect`：自动白平衡区域。

第 118 行：

```python
devices = (ToupcamInstV2 * TOUPCAM_MAX)()
```

创建一个相机设备数组，最多放 16 个设备。

这是 C 风格 API 常见写法：先准备一块数组，让 SDK 把枚举结果写进去。

第 119 行：

```python
count = self.lib.Toupcam_EnumV2(devices)
```

调用 SDK 枚举相机。返回值 `count` 是找到的相机数量。

第 120-121 行：

```python
if count == 0:
    raise RuntimeError("未找到相机！请检查连接。")
```

如果没有找到相机，就抛出错误。

第 123 行：

```python
camera_name = devices[0].displayname
```

取第一个相机的显示名称，用于 UI 状态栏提示。

第 126 行：

```python
self.handle = self.lib.Toupcam_Open(devices[0].id)
```

打开第一个相机，返回相机句柄。

以后所有 SDK 调用都要带这个 `handle`，相当于告诉 SDK：我要操作的是哪台相机。

第 127-128 行：

```python
if not self.handle:
    raise RuntimeError("无法打开相机！")
```

如果句柄为空，说明打开失败。

第 131 行：

```python
self.lib.Toupcam_put_eSize(self.handle, resolution_index)
```

设置相机输出分辨率。

重点：

> 这是让相机硬件/SDK 输出指定分辨率，不是用 OpenCV resize 图片。

第 133-135 行：

```python
w, h = c_int(), c_int()
self.lib.Toupcam_get_Size(self.handle, byref(w), byref(h))
self.width, self.height = w.value, h.value
```

读取实际宽高。

`byref(w)` 的意思是把变量地址传给 C 函数，让 C 函数把结果写进去。

`w.value` 才是真正的 Python 数值。

第 138-141 行：

```python
t_min, t_max, t_def = c_uint(), c_uint(), c_uint()
if self.lib.Toupcam_get_ExpTimeRange(...) == 0:
    self.expo_time_min = t_min.value
    self.expo_time_max = t_max.value
```

读取相机支持的曝光时间范围。

后面设置曝光时会 clamp 到这个范围，防止传非法值。

第 144-147 行：

```python
ag_min, ag_max, ag_def = c_ushort(), c_ushort(), c_ushort()
if self.lib.Toupcam_get_ExpoAGainRange(...) == 0:
    self.gain_min = ag_min.value
    self.gain_max = ag_max.value
```

读取增益范围。

第 150-151 行：

```python
self._read_expo_gain()
self._read_white_balance()
```

读取相机当前曝光、增益、白平衡，让程序状态和硬件状态同步。

第 154-155 行：

```python
self.lib.Toupcam_put_AutoExpoEnable(self.handle, 1)
self.auto_expo = True
```

默认开启自动曝光。

第 159 行：

```python
result = self.lib.Toupcam_StartPullModeWithCallback(self.handle, self._event_cb, None)
```

启动相机采集。

这是核心：

- `StartPullModeWithCallback` 表示 pull 模式 + callback 通知。
- SDK 有新图时会调用 `self._event_cb`。
- callback 不给你图像本身，只告诉你“可以拉图了”。

面试说法：

> 我用的是 pull mode callback。callback 只通知有新帧，后台线程再调用 PullImage 拉取图像，这样不会在 SDK 回调里做耗时操作。

第 160-161 行：

```python
if result != 0:
    raise RuntimeError(f"启动捕获失败: {result}")
```

如果启动采集失败，抛出错误。

第 163-164 行：

```python
self.connected = True
return camera_name
```

标记相机已连接，并把相机名称返回给 UI。

第 166-173 行：

```python
def close(self):
    self.stop_grabber()
    if self.handle and self.lib:
        self.lib.Toupcam_Stop(self.handle)
        self.lib.Toupcam_Close(self.handle)
        self.handle = None
        self.connected = False
```

关闭相机。

顺序很重要：

1. 先停后台抓帧线程。
2. 再停 SDK 采集。
3. 最后关闭相机句柄。

第 175 行：

```python
def pull_frame(self):
```

拉取一帧图像。

第 177-179 行：

```python
if not self.image_ready or not self.handle:
    return None
self.image_ready = False
```

如果没有新图，或者相机没打开，就返回 `None`。

一旦准备拉图，就把 `image_ready` 重置为 False。

第 182 行：

```python
bufsize = self.width * self.height * 4
```

计算图像 buffer 大小。

为什么乘以 4？

因为这里拉的是 32-bit BGRA 图像，每个像素 4 个字节：

```text
B: blue
G: green
R: red
A: alpha
```

第 183 行：

```python
buf = (c_ubyte * bufsize)()
```

创建 C 字节数组，用来接收 SDK 写入的图像数据。

第 184 行：

```python
pw, ph = c_uint(), c_uint()
```

创建两个 C 变量，用于接收实际图像宽高。

第 185 行：

```python
result = self.lib.Toupcam_PullImage(self.handle, buf, 32, byref(pw), byref(ph))
```

真正从相机拉取图像。

参数含义：

- `self.handle`：相机句柄。
- `buf`：图像数据写入位置。
- `32`：按 32-bit BGRA 格式拉图。
- `byref(pw)` / `byref(ph)`：让 SDK 写入宽高。

第 186-187 行：

```python
if result != 0:
    return None
```

如果拉图失败，返回 None。

第 189 行：

```python
img = np.ctypeslib.as_array(buf).reshape((ph.value, pw.value, 4))
```

把 C buffer 转成 numpy 数组，并 reshape 成图像形状：

```text
高度 x 宽度 x 4通道
```

第 190 行：

```python
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
```

把 BGRA 转成 BGR。

为什么是 BGR？

OpenCV 默认图像格式是 BGR，后面显示、保存、推理预处理都以 BGR/OpenCV 格式为基础。

第 191 行：

```python
return img
```

返回这一帧图像。

第 197 行：

```python
def start_grabber(self):
```

启动后台抓帧线程。

第 199-200 行：

```python
if self._grabber_thread and self._grabber_thread.is_alive():
    return
```

如果抓帧线程已经在运行，就直接返回，避免重复启动多个线程。

第 201-203 行：

```python
self._grabber_running = True
self._frame_seq = 0
self._last_read_seq = -1
```

设置线程运行标志，并重置帧序号。

第 204-207 行：

```python
self._grabber_thread = threading.Thread(
    target=self._grabber_loop, daemon=True, name="FrameGrabber"
)
self._grabber_thread.start()
```

创建并启动后台线程。

参数解释：

- `target=self._grabber_loop`：线程要执行的函数。
- `daemon=True`：主程序退出时线程自动结束。
- `name="FrameGrabber"`：给线程起名，方便调试。

第 209-214 行：

```python
def stop_grabber(self):
    self._grabber_running = False
    if self._grabber_thread and self._grabber_thread.is_alive():
        self._grabber_thread.join(timeout=2.0)
    self._grabber_thread = None
```

停止后台抓帧线程。

`join(timeout=2.0)` 表示最多等 2 秒，让线程退出。

第 216 行：

```python
def _grabber_loop(self):
```

后台线程循环函数。

第 221 行：

```python
while self._grabber_running and self.connected:
```

只要线程允许运行，并且相机还连接，就一直循环拉帧。

第 222 行：

```python
frame = self.pull_frame()
```

尝试拉取一帧。

第 223-226 行：

```python
if frame is not None:
    with self._frame_lock:
        self._latest_frame = frame
        self._frame_seq += 1
```

如果成功拿到图像：

1. 加锁。
2. 把最新帧保存到 `_latest_frame`。
3. 帧序号加 1。

这里没有队列，只有最新帧。

面试说法：

> 这是 latest-frame 策略。实时系统宁可丢旧帧，也不要积压导致延迟越来越大。

第 227-229 行：

```python
else:
    time.sleep(0.001)
```

如果没有新图，休眠 1ms，避免 while 循环一直空转占 CPU。

第 231 行：

```python
def get_latest_frame(self):
```

给 UI 线程调用，用来读取最新帧。

第 237 行：

```python
with self._frame_lock:
```

读帧时也加锁，避免和后台写帧冲突。

第 238-239 行：

```python
if self._frame_seq == self._last_read_seq:
    return None
```

如果当前帧序号等于上次读取的帧序号，说明没有新帧，返回 None。

第 240 行：

```python
self._last_read_seq = self._frame_seq
```

记录这次已经读到的最新帧序号。

第 241-243 行：

```python
if self._latest_frame is not None:
    return self._latest_frame.copy()
return None
```

如果有最新帧，返回它的拷贝。

为什么要 `.copy()`？

因为 UI 线程拿到图像后可能会显示、增强、绘制，如果直接返回原始对象，后台线程又同时更新它，可能互相影响。

第 246-249 行：

```python
def set_auto_exposure(self, enable):
    if self.handle:
        self.lib.Toupcam_put_AutoExpoEnable(self.handle, 1 if enable else 0)
        self.auto_expo = enable
```

设置自动曝光开关。

`enable=True` 传 1，`enable=False` 传 0。

第 251-255 行：

```python
def set_exposure_time(self, time_us):
    if self.handle:
        time_us = max(self.expo_time_min, min(self.expo_time_max, int(time_us)))
        self.lib.Toupcam_put_ExpoTime(self.handle, time_us)
        self.expo_time_us = time_us
```

设置曝光时间。

关键点：

- 单位是微秒。
- 设置前做 clamp，保证不超过相机支持范围。

面试说法：

> 硬件参数设置前要做边界保护，避免传非法值导致 SDK 报错或相机状态异常。

第 257-261 行：

```python
def set_gain(self, gain_pct):
    if self.handle:
        gain_pct = max(self.gain_min, min(self.gain_max, int(gain_pct)))
        self.lib.Toupcam_put_ExpoAGain(self.handle, gain_pct)
        self.expo_gain = gain_pct
```

设置增益。逻辑和曝光类似。

第 264-270 行：

```python
def set_temp_tint(self, temp, tint):
    if self.handle:
        temp = max(TOUPCAM_TEMP_MIN, min(TOUPCAM_TEMP_MAX, int(temp)))
        tint = max(TOUPCAM_TINT_MIN, min(TOUPCAM_TINT_MAX, int(tint)))
        self.lib.Toupcam_put_TempTint(self.handle, temp, tint)
        self.temp = temp
        self.tint = tint
```

设置白平衡色温和 Tint。

也做了范围限制。

第 272-274 行：

```python
def auto_white_balance(self):
    if self.handle:
        self.lib.Toupcam_AwbOnePush(self.handle, self._awb_cb, None)
```

触发一次自动白平衡。

自动白平衡完成后，SDK 会调用 `_awb_cb`，最终进入 `_on_awb` 更新 `temp` 和 `tint`。

第 276-283 行：

```python
def _read_expo_gain(self):
    ...
```

读取当前曝光和增益。

注意 `byref(expo)` 和 `byref(gain)`：这是把变量地址传给 SDK，让 SDK 写入结果。

第 285-290 行：

```python
def _read_white_balance(self):
    ...
```

读取当前白平衡参数。

第 292-295 行：

```python
def refresh_readings(self):
    self._read_expo_gain()
    self._read_white_balance()
```

统一刷新曝光、增益、白平衡读数。UI 会定时调用它。

第 298-302 行：

```python
def set_ae_roi(self, left, top, right, bottom):
    if self.handle:
        rc = RECT(int(left), int(top), int(right), int(bottom))
        self.lib.Toupcam_put_AEAuxRect(self.handle, byref(rc))
```

设置自动曝光 ROI。

流程：

1. UI 拖拽得到 ROI。
2. 转成图像像素坐标。
3. 创建 `RECT` 结构体。
4. 传给 SDK。

第 304-308 行：

```python
def set_awb_roi(self, left, top, right, bottom):
    if self.handle:
        rc = RECT(int(left), int(top), int(right), int(bottom))
        self.lib.Toupcam_put_AWBAuxRect(self.handle, byref(rc))
```

设置自动白平衡 ROI，逻辑和 AE ROI 类似。

## 四、UI 怎么调用相机控制器

代码位置：`camera_ui/main_window.py`

### 1. 创建相机对象

在 `MainWindow.__init__` 里有：

```python
self.camera = CameraController()
```

意思是主窗口持有一个相机控制器对象。

### 2. 点击按钮启动相机

按钮点击后进入：

```python
_toggle_camera()
```

如果相机没运行，就调用：

```python
_start_camera()
```

里面最关键的是：

```python
cam_name = self.camera.open(resolution_index=res_idx)
```

这句进入刚才讲的 `CameraController.open()`。

### 3. 启动抓帧线程

相机打开成功后：

```python
self.camera.start_grabber()
```

后台线程开始持续拉帧，并缓存 `_latest_frame`。

### 4. 启动 UI 定时器

```python
self.timer.start(16)
```

大约每 16ms 触发一次 `_on_timer()`，理论上接近 60 FPS 刷新。

### 5. UI 定时读取最新帧

`_on_timer()` 中：

```python
frame = self.camera.get_latest_frame()
```

如果有新帧：

```python
self.current_frame = frame.copy()
display_frame = self._apply_enhancements(frame)
self.camera_view.set_image(display_frame)
```

这几句表示：

1. 保存当前原始帧。
2. 做显示增强。
3. 显示到左侧相机窗口。

注意：

> 增强图用于显示，连续检测提交的是原始 frame，这样避免增强参数改变模型输入分布。

### 6. 连续检测时提交给推理线程

```python
if self.continuous_detect and self.model is not None:
    self._inference_worker.submit_frame(frame)
```

相机链路和模型推理在这里接上。

但模型不在 UI 线程里跑，而是丢给推理线程。

## 五、面试高频问答

### Q1：为什么要用 ctypes？

答：

> 因为 Toupcam 相机 SDK 提供的是 C 动态库 `nncam.dll`，不是 Python 包。Python 需要用 ctypes 加载 DLL，并声明 C 函数的参数类型、返回值类型、结构体和回调类型，才能调用相机枚举、打开、拉帧、曝光、白平衡等接口。

### Q2：为什么 callback 里不直接拉图？

答：

> SDK callback 应该尽量轻，只做事件通知。如果在 callback 里做拉图、OpenCV 转换或 UI 更新，可能阻塞 SDK 内部线程，影响采集稳定性。所以我只在 callback 里设置 `image_ready=True`，后台抓帧线程再主动 `PullImage`。

### Q3：为什么只保存最新帧，不保存所有帧？

答：

> 这是实时系统的 latest-frame 策略。显微观察更关注当前画面，如果所有帧排队，推理或显示慢时延迟会越积越大，用户看到的是过去的画面。只保留最新帧可以牺牲部分帧完整性，换取低延迟。

### Q4：为什么要加锁？

答：

> 后台抓帧线程在写 `_latest_frame`，UI 线程在读 `_latest_frame`。加锁可以避免同时读写造成数据不一致。

### Q5：BGRA、BGR、RGB 有什么区别？

答：

> SDK 拉出来的是 BGRA，包含蓝、绿、红、透明度四个通道；OpenCV 常用 BGR；模型训练通常用 RGB。所以相机拉帧后先 BGRA 转 BGR，推理前再 BGR 转 RGB。

### Q6：曝光和增益有什么区别？

答：

> 曝光时间是传感器接收光的时间，时间越长图像越亮，但运动模糊风险更大；增益是放大电信号，能提高亮度，但也会放大噪声。系统里两个都开放调节。

### Q7：为什么要支持白平衡？

答：

> 二维材料层数判断依赖颜色差异，白平衡不稳定会影响人眼观察，也可能影响模型输入分布。所以系统支持色温、Tint 和一键自动白平衡。

## 六、最推荐背的完整回答

> 相机部分我用 ctypes 封装 Toupcam 的 C SDK。程序启动时先在 `sdk_types.py` 里定位 `nncam.dll`，定义 SDK 常量、结构体和回调类型。真正打开相机在 `CameraController.open()` 里完成：先用 `ctypes.CDLL` 加载 DLL，再声明各个 SDK 函数的 `argtypes` 和 `restype`，然后调用 `Toupcam_EnumV2` 枚举设备、`Toupcam_Open` 打开相机、`Toupcam_put_eSize` 设置输出分辨率，并读取曝光、增益、白平衡范围。采集模式采用 `StartPullModeWithCallback`，SDK 有新图像时 callback 只把 `image_ready` 置为 True，不做耗时处理。后台抓帧线程循环调用 `pull_frame()`，用 `Toupcam_PullImage` 拉取 BGRA buffer，再转成 OpenCV BGR 图像，并用锁保护 `_latest_frame`。UI 主线程通过 QTimer 定时调用 `get_latest_frame()` 读取最新帧显示，连续检测时再把原始帧交给推理线程。这个设计把相机采集、UI 刷新和模型推理解耦，可以降低界面卡顿和延迟积压。
