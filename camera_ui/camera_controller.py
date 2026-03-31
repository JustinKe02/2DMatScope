"""
CameraController — 封装 Toupcam SDK 的相机控制类
"""

import ctypes
from ctypes import *
import numpy as np
import cv2
import threading
import time

from .sdk_types import (
    DLL_PATH, TOUPCAM_MAX, TOUPCAM_EVENT_IMAGE, TOUPCAM_EVENT_DISCONNECTED,
    TOUPCAM_TEMP_DEF, TOUPCAM_TEMP_MIN, TOUPCAM_TEMP_MAX,
    TOUPCAM_TINT_DEF, TOUPCAM_TINT_MIN, TOUPCAM_TINT_MAX,
    CALLBACK_TYPE, AWB_TT_CALLBACK_TYPE,
    ToupcamResolution, ToupcamModelV2, ToupcamInstV2, RECT,
)


class CameraController:
    """封装 Toupcam SDK 的相机控制类"""

    def __init__(self):
        self.lib = None
        self.handle = None
        self.width = 0
        self.height = 0
        self.image_ready = False
        self.connected = False

        # 曝光 / 增益
        self.auto_expo = True
        self.expo_time_us = 70000
        self.expo_gain = 100
        self.expo_time_min = 50
        self.expo_time_max = 2000000
        self.gain_min = 100
        self.gain_max = 500

        # 白平衡
        self.temp = TOUPCAM_TEMP_DEF
        self.tint = TOUPCAM_TINT_DEF

        # 回调（必须保持引用，防止被 GC）
        self._event_cb = CALLBACK_TYPE(self._on_event)
        self._awb_cb = AWB_TT_CALLBACK_TYPE(self._on_awb)

        # 多线程帧抓取
        self._grabber_thread = None
        self._grabber_running = False
        self._latest_frame = None       # 最新帧 (BGR numpy)
        self._frame_lock = threading.Lock()
        self._frame_seq = 0             # 帧序号 (用来判断是否有新帧)
        self._last_read_seq = -1        # 上次读取的帧序号

    def _on_event(self, event, ctx):
        if event == TOUPCAM_EVENT_IMAGE:
            self.image_ready = True
        elif event == TOUPCAM_EVENT_DISCONNECTED:
            self.connected = False

    def _on_awb(self, nTemp, nTint, ctx):
        self.temp = nTemp
        self.tint = nTint

    def open(self, resolution_index=0):
        """打开相机"""
        self.lib = ctypes.CDLL(DLL_PATH)

        # 基础函数签名
        self.lib.Toupcam_EnumV2.restype = c_uint
        self.lib.Toupcam_Open.restype = c_void_p
        self.lib.Toupcam_Open.argtypes = [c_wchar_p]
        self.lib.Toupcam_Close.argtypes = [c_void_p]
        self.lib.Toupcam_get_Size.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
        self.lib.Toupcam_put_eSize.argtypes = [c_void_p, c_uint]
        self.lib.Toupcam_StartPullModeWithCallback.argtypes = [c_void_p, CALLBACK_TYPE, c_void_p]
        self.lib.Toupcam_PullImage.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_uint), POINTER(c_uint)]
        self.lib.Toupcam_Stop.argtypes = [c_void_p]

        # 曝光 / 增益
        self.lib.Toupcam_put_AutoExpoEnable.argtypes = [c_void_p, c_int]
        self.lib.Toupcam_get_AutoExpoEnable.argtypes = [c_void_p, POINTER(c_int)]
        self.lib.Toupcam_put_ExpoTime.argtypes = [c_void_p, c_uint]
        self.lib.Toupcam_get_ExpoTime.argtypes = [c_void_p, POINTER(c_uint)]
        self.lib.Toupcam_get_ExpTimeRange.argtypes = [c_void_p, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)]
        self.lib.Toupcam_put_ExpoAGain.argtypes = [c_void_p, c_ushort]
        self.lib.Toupcam_get_ExpoAGain.argtypes = [c_void_p, POINTER(c_ushort)]
        self.lib.Toupcam_get_ExpoAGainRange.argtypes = [c_void_p, POINTER(c_ushort), POINTER(c_ushort), POINTER(c_ushort)]

        # 白平衡
        self.lib.Toupcam_put_TempTint.argtypes = [c_void_p, c_int, c_int]
        self.lib.Toupcam_get_TempTint.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
        self.lib.Toupcam_AwbOnePush.argtypes = [c_void_p, AWB_TT_CALLBACK_TYPE, c_void_p]

        # ROI (自动曝光区域 / 自动白平衡区域)
        self.lib.Toupcam_put_AEAuxRect.argtypes = [c_void_p, POINTER(RECT)]
        self.lib.Toupcam_get_AEAuxRect.argtypes = [c_void_p, POINTER(RECT)]
        self.lib.Toupcam_put_AWBAuxRect.argtypes = [c_void_p, POINTER(RECT)]
        self.lib.Toupcam_get_AWBAuxRect.argtypes = [c_void_p, POINTER(RECT)]

        # 枚举相机
        devices = (ToupcamInstV2 * TOUPCAM_MAX)()
        count = self.lib.Toupcam_EnumV2(devices)
        if count == 0:
            raise RuntimeError("未找到相机！请检查连接。")

        camera_name = devices[0].displayname

        # 打开相机
        self.handle = self.lib.Toupcam_Open(devices[0].id)
        if not self.handle:
            raise RuntimeError("无法打开相机！")

        # 设置分辨率
        self.lib.Toupcam_put_eSize(self.handle, resolution_index)

        w, h = c_int(), c_int()
        self.lib.Toupcam_get_Size(self.handle, byref(w), byref(h))
        self.width, self.height = w.value, h.value

        # 获取曝光范围
        t_min, t_max, t_def = c_uint(), c_uint(), c_uint()
        if self.lib.Toupcam_get_ExpTimeRange(self.handle, byref(t_min), byref(t_max), byref(t_def)) == 0:
            self.expo_time_min = t_min.value
            self.expo_time_max = t_max.value

        # 获取增益范围
        ag_min, ag_max, ag_def = c_ushort(), c_ushort(), c_ushort()
        if self.lib.Toupcam_get_ExpoAGainRange(self.handle, byref(ag_min), byref(ag_max), byref(ag_def)) == 0:
            self.gain_min = ag_min.value
            self.gain_max = ag_max.value

        # 读取当前值
        self._read_expo_gain()
        self._read_white_balance()

        # 启用自动曝光
        self.lib.Toupcam_put_AutoExpoEnable(self.handle, 1)
        self.auto_expo = True

        # 开始捕获
        result = self.lib.Toupcam_StartPullModeWithCallback(self.handle, self._event_cb, None)
        if result != 0:
            raise RuntimeError(f"启动捕获失败: {result}")

        self.connected = True
        return camera_name

    def close(self):
        """关闭相机"""
        self.stop_grabber()  # 先停止抓取线程
        if self.handle and self.lib:
            self.lib.Toupcam_Stop(self.handle)
            self.lib.Toupcam_Close(self.handle)
            self.handle = None
            self.connected = False

    def pull_frame(self):
        """拉取一帧图像（32-bit BGRA），返回 BGR numpy 数组或 None"""
        if not self.image_ready or not self.handle:
            return None
        self.image_ready = False

        # 使用 32-bit BGRA 模式拉取以获取更好的色彩质量
        bufsize = self.width * self.height * 4
        buf = (c_ubyte * bufsize)()
        pw, ph = c_uint(), c_uint()
        result = self.lib.Toupcam_PullImage(self.handle, buf, 32, byref(pw), byref(ph))
        if result != 0:
            return None

        img = np.ctypeslib.as_array(buf).reshape((ph.value, pw.value, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # ================================================================
    #   多线程帧抓取
    # ================================================================

    def start_grabber(self):
        """启动后台帧抓取线程"""
        if self._grabber_thread and self._grabber_thread.is_alive():
            return
        self._grabber_running = True
        self._frame_seq = 0
        self._last_read_seq = -1
        self._grabber_thread = threading.Thread(
            target=self._grabber_loop, daemon=True, name="FrameGrabber"
        )
        self._grabber_thread.start()

    def stop_grabber(self):
        """停止帧抓取线程"""
        self._grabber_running = False
        if self._grabber_thread and self._grabber_thread.is_alive():
            self._grabber_thread.join(timeout=2.0)
        self._grabber_thread = None

    def _grabber_loop(self):
        """后台线程: 不停轮询 pull_frame(), 将最新帧存入 _latest_frame"""
        while self._grabber_running and self.connected:
            frame = self.pull_frame()
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_seq += 1
            else:
                # 没有新帧时短暂休眠，避免空转
                time.sleep(0.001)

    def get_latest_frame(self):
        """
        获取最新帧 (线程安全)。
        如果没有新帧则返回 None。
        返回的是帧的拷贝，可以安全在 UI 线程操作。
        """
        with self._frame_lock:
            if self._frame_seq == self._last_read_seq:
                return None  # 没有新帧
            self._last_read_seq = self._frame_seq
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    # —— 曝光控制 ——
    def set_auto_exposure(self, enable):
        if self.handle:
            self.lib.Toupcam_put_AutoExpoEnable(self.handle, 1 if enable else 0)
            self.auto_expo = enable

    def set_exposure_time(self, time_us):
        if self.handle:
            time_us = max(self.expo_time_min, min(self.expo_time_max, int(time_us)))
            self.lib.Toupcam_put_ExpoTime(self.handle, time_us)
            self.expo_time_us = time_us

    def set_gain(self, gain_pct):
        if self.handle:
            gain_pct = max(self.gain_min, min(self.gain_max, int(gain_pct)))
            self.lib.Toupcam_put_ExpoAGain(self.handle, gain_pct)
            self.expo_gain = gain_pct

    # —— 白平衡控制 ——
    def set_temp_tint(self, temp, tint):
        if self.handle:
            temp = max(TOUPCAM_TEMP_MIN, min(TOUPCAM_TEMP_MAX, int(temp)))
            tint = max(TOUPCAM_TINT_MIN, min(TOUPCAM_TINT_MAX, int(tint)))
            self.lib.Toupcam_put_TempTint(self.handle, temp, tint)
            self.temp = temp
            self.tint = tint

    def auto_white_balance(self):
        if self.handle:
            self.lib.Toupcam_AwbOnePush(self.handle, self._awb_cb, None)

    def _read_expo_gain(self):
        if self.handle:
            expo = c_uint()
            if self.lib.Toupcam_get_ExpoTime(self.handle, byref(expo)) == 0:
                self.expo_time_us = expo.value
            gain = c_ushort()
            if self.lib.Toupcam_get_ExpoAGain(self.handle, byref(gain)) == 0:
                self.expo_gain = gain.value

    def _read_white_balance(self):
        if self.handle:
            temp_val, tint_val = c_int(), c_int()
            if self.lib.Toupcam_get_TempTint(self.handle, byref(temp_val), byref(tint_val)) == 0:
                self.temp = temp_val.value
                self.tint = tint_val.value

    def refresh_readings(self):
        """刷新当前曝光/增益/白平衡读数"""
        self._read_expo_gain()
        self._read_white_balance()

    # —— ROI 区域控制 ——
    def set_ae_roi(self, left, top, right, bottom):
        """设置自动曝光 ROI 区域 (图像像素坐标)"""
        if self.handle:
            rc = RECT(int(left), int(top), int(right), int(bottom))
            self.lib.Toupcam_put_AEAuxRect(self.handle, byref(rc))

    def set_awb_roi(self, left, top, right, bottom):
        """设置自动白平衡 ROI 区域 (图像像素坐标)"""
        if self.handle:
            rc = RECT(int(left), int(top), int(right), int(bottom))
            self.lib.Toupcam_put_AWBAuxRect(self.handle, byref(rc))
