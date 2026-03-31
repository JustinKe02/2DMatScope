"""
L3CMOS05100KPA 高分辨率图像捕获程序
支持 2560x1922 全分辨率实时预览和图像保存
新增功能：曝光/增益控制、白平衡（色温/色调）调节
"""

import ctypes
from ctypes import *
import numpy as np
import cv2
import time
import threading

# DLL 路径
DLL_PATH = r'F:\ImageView\nncamsdk.20171211\nncamsdk.20171211\win\x64\nncam.dll'

# 常量
TOUPCAM_MAX = 16
TOUPCAM_EVENT_IMAGE = 0x0004
TOUPCAM_EVENT_ERROR = 0x0080
TOUPCAM_EVENT_DISCONNECTED = 0x0081
TOUPCAM_EVENT_TEMPTINT = 0x0002

# 白平衡默认值 / 范围
TOUPCAM_TEMP_DEF = 6503
TOUPCAM_TEMP_MIN = 2000
TOUPCAM_TEMP_MAX = 15000
TOUPCAM_TINT_DEF = 1000
TOUPCAM_TINT_MIN = 200
TOUPCAM_TINT_MAX = 2500

# 结构体定义
class ToupcamResolution(Structure):
    _fields_ = [("width", c_uint), ("height", c_uint)]

class ToupcamModelV2(Structure):
    _fields_ = [
        ("name", c_wchar_p), ("flag", c_ulonglong), ("maxspeed", c_uint),
        ("preview", c_uint), ("still", c_uint), ("maxfanspeed", c_uint),
        ("xpixsz", c_float), ("ypixsz", c_float),
        ("res", ToupcamResolution * TOUPCAM_MAX)
    ]
    _pack_ = 8

class ToupcamInstV2(Structure):
    _fields_ = [
        ("displayname", c_wchar * 64), ("id", c_wchar * 64),
        ("model", POINTER(ToupcamModelV2))
    ]

# 全局变量
g_image_ready = False
g_lib = None
g_handle = None
g_width = 0
g_height = 0

# —— 曝光 / 增益 / 白平衡 状态 ——
g_auto_expo = True          # 自动曝光状态
g_expo_time_us = 0          # 当前曝光时间 (µs)
g_expo_gain = 100           # 当前增益 (%)
g_expo_time_min = 1         # 曝光范围
g_expo_time_max = 350000
g_gain_min = 100
g_gain_max = 500
g_temp = TOUPCAM_TEMP_DEF   # 色温
g_tint = TOUPCAM_TINT_DEF   # 色调

# 回调函数
CALLBACK_TYPE = CFUNCTYPE(None, c_uint, c_void_p)
# 自动白平衡回调类型 (Temp/Tint mode)
AWB_TT_CALLBACK_TYPE = CFUNCTYPE(None, c_int, c_int, c_void_p)

def event_callback(event, ctx):
    global g_image_ready
    if event == TOUPCAM_EVENT_IMAGE:
        g_image_ready = True
    elif event == TOUPCAM_EVENT_DISCONNECTED:
        print("相机已断开!")

g_callback = CALLBACK_TYPE(event_callback)

# 自动白平衡回调
def awb_tt_callback(nTemp, nTint, ctx):
    global g_temp, g_tint
    g_temp = nTemp
    g_tint = nTint
    print(f"  自动白平衡完成: Temp={nTemp}, Tint={nTint}")

g_awb_tt_callback = AWB_TT_CALLBACK_TYPE(awb_tt_callback)


def init_camera(resolution_index=0):
    """初始化相机，设置指定分辨率"""
    global g_lib, g_handle, g_width, g_height
    global g_auto_expo, g_expo_time_us, g_expo_gain
    global g_expo_time_min, g_expo_time_max, g_gain_min, g_gain_max
    global g_temp, g_tint

    # 加载 DLL
    g_lib = ctypes.CDLL(DLL_PATH)

    # ============== 基础函数签名 ==============
    g_lib.Toupcam_EnumV2.restype = c_uint
    g_lib.Toupcam_Open.restype = c_void_p
    g_lib.Toupcam_Open.argtypes = [c_wchar_p]
    g_lib.Toupcam_Close.argtypes = [c_void_p]
    g_lib.Toupcam_get_Size.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    g_lib.Toupcam_put_eSize.argtypes = [c_void_p, c_uint]
    g_lib.Toupcam_StartPullModeWithCallback.argtypes = [c_void_p, CALLBACK_TYPE, c_void_p]
    g_lib.Toupcam_PullImage.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_uint), POINTER(c_uint)]
    g_lib.Toupcam_Stop.argtypes = [c_void_p]

    # ============== 曝光 / 增益 函数签名 ==============
    g_lib.Toupcam_put_AutoExpoEnable.argtypes = [c_void_p, c_int]
    g_lib.Toupcam_get_AutoExpoEnable.argtypes = [c_void_p, POINTER(c_int)]

    g_lib.Toupcam_put_ExpoTime.argtypes = [c_void_p, c_uint]
    g_lib.Toupcam_get_ExpoTime.argtypes = [c_void_p, POINTER(c_uint)]
    g_lib.Toupcam_get_ExpTimeRange.argtypes = [c_void_p, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)]

    g_lib.Toupcam_put_ExpoAGain.argtypes = [c_void_p, c_ushort]
    g_lib.Toupcam_get_ExpoAGain.argtypes = [c_void_p, POINTER(c_ushort)]
    g_lib.Toupcam_get_ExpoAGainRange.argtypes = [c_void_p, POINTER(c_ushort), POINTER(c_ushort), POINTER(c_ushort)]

    g_lib.Toupcam_put_AutoExpoTarget.argtypes = [c_void_p, c_ushort]
    g_lib.Toupcam_get_AutoExpoTarget.argtypes = [c_void_p, POINTER(c_ushort)]

    # ============== 白平衡 函数签名 ==============
    g_lib.Toupcam_put_TempTint.argtypes = [c_void_p, c_int, c_int]
    g_lib.Toupcam_get_TempTint.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    g_lib.Toupcam_AwbOnePush.argtypes = [c_void_p, AWB_TT_CALLBACK_TYPE, c_void_p]

    # 枚举相机
    devices = (ToupcamInstV2 * TOUPCAM_MAX)()
    count = g_lib.Toupcam_EnumV2(devices)

    if count == 0:
        raise RuntimeError("未找到相机！请检查连接。")

    print(f"找到相机: {devices[0].displayname}")

    # 打开相机
    g_handle = g_lib.Toupcam_Open(devices[0].id)
    if not g_handle:
        raise RuntimeError("无法打开相机！")

    # 设置分辨率 (0=最高, 1=中等, 2=最低)
    g_lib.Toupcam_put_eSize(g_handle, resolution_index)

    # 获取当前分辨率
    w, h = c_int(), c_int()
    g_lib.Toupcam_get_Size(g_handle, byref(w), byref(h))
    g_width, g_height = w.value, h.value
    print(f"分辨率设置为: {g_width} x {g_height}")

    # —— 获取曝光时间范围 ——
    t_min, t_max, t_def = c_uint(), c_uint(), c_uint()
    if g_lib.Toupcam_get_ExpTimeRange(g_handle, byref(t_min), byref(t_max), byref(t_def)) == 0:
        g_expo_time_min = t_min.value
        g_expo_time_max = t_max.value
        print(f"曝光时间范围: {g_expo_time_min}~{g_expo_time_max} µs (默认 {t_def.value} µs)")

    # —— 获取增益范围 ——
    ag_min, ag_max, ag_def = c_ushort(), c_ushort(), c_ushort()
    if g_lib.Toupcam_get_ExpoAGainRange(g_handle, byref(ag_min), byref(ag_max), byref(ag_def)) == 0:
        g_gain_min = ag_min.value
        g_gain_max = ag_max.value
        print(f"增益范围: {g_gain_min}~{g_gain_max}% (默认 {ag_def.value}%)")

    # —— 读取当前曝光时间 ——
    expo = c_uint()
    if g_lib.Toupcam_get_ExpoTime(g_handle, byref(expo)) == 0:
        g_expo_time_us = expo.value

    # —— 读取当前增益 ——
    gain = c_ushort()
    if g_lib.Toupcam_get_ExpoAGain(g_handle, byref(gain)) == 0:
        g_expo_gain = gain.value

    # —— 读取当前白平衡 ——
    temp_val, tint_val = c_int(), c_int()
    if g_lib.Toupcam_get_TempTint(g_handle, byref(temp_val), byref(tint_val)) == 0:
        g_temp = temp_val.value
        g_tint = tint_val.value
        print(f"当前白平衡: Temp={g_temp}, Tint={g_tint}")

    # 启用自动曝光
    g_lib.Toupcam_put_AutoExpoEnable(g_handle, 1)
    g_auto_expo = True

    # 开始捕获
    result = g_lib.Toupcam_StartPullModeWithCallback(g_handle, g_callback, None)
    if result != 0:
        raise RuntimeError(f"启动捕获失败: {result}")

    print("相机已启动!")
    return True


def capture_frame():
    """捕获一帧图像"""
    global g_image_ready

    if not g_image_ready:
        return None

    g_image_ready = False

    # 分配缓冲区
    bufsize = g_width * g_height * 3
    buf = (c_ubyte * bufsize)()

    pw, ph = c_uint(), c_uint()
    result = g_lib.Toupcam_PullImage(g_handle, buf, 24, byref(pw), byref(ph))

    if result != 0:
        return None

    # 转换为 numpy 数组
    img = np.ctypeslib.as_array(buf).reshape((ph.value, pw.value, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def close_camera():
    """关闭相机"""
    global g_handle
    if g_handle and g_lib:
        g_lib.Toupcam_Stop(g_handle)
        g_lib.Toupcam_Close(g_handle)
        g_handle = None
        print("相机已关闭")


# ================================================================
#   曝光 / 增益 / 白平衡 辅助函数
# ================================================================

def set_exposure_time(time_us):
    """设置曝光时间 (µs)"""
    global g_expo_time_us
    if g_handle and g_lib:
        time_us = max(g_expo_time_min, min(g_expo_time_max, int(time_us)))
        g_lib.Toupcam_put_ExpoTime(g_handle, time_us)
        g_expo_time_us = time_us


def set_gain(gain_pct):
    """设置模拟增益 (%)"""
    global g_expo_gain
    if g_handle and g_lib:
        gain_pct = max(g_gain_min, min(g_gain_max, int(gain_pct)))
        g_lib.Toupcam_put_ExpoAGain(g_handle, gain_pct)
        g_expo_gain = gain_pct


def toggle_auto_exposure(enable):
    """切换自动/手动曝光"""
    global g_auto_expo
    if g_handle and g_lib:
        g_lib.Toupcam_put_AutoExpoEnable(g_handle, 1 if enable else 0)
        g_auto_expo = enable


def set_temp_tint(temp, tint):
    """设置色温/色调"""
    global g_temp, g_tint
    if g_handle and g_lib:
        temp = max(TOUPCAM_TEMP_MIN, min(TOUPCAM_TEMP_MAX, int(temp)))
        tint = max(TOUPCAM_TINT_MIN, min(TOUPCAM_TINT_MAX, int(tint)))
        g_lib.Toupcam_put_TempTint(g_handle, temp, tint)
        g_temp = temp
        g_tint = tint


def auto_white_balance():
    """一键自动白平衡 (Temp/Tint 模式)"""
    if g_handle and g_lib:
        print("  触发一键自动白平衡...")
        g_lib.Toupcam_AwbOnePush(g_handle, g_awb_tt_callback, None)


def read_current_expo_gain():
    """从相机读取当前曝光/增益 (用于自动曝光模式下跟踪变化)"""
    global g_expo_time_us, g_expo_gain
    if g_handle and g_lib:
        expo = c_uint()
        if g_lib.Toupcam_get_ExpoTime(g_handle, byref(expo)) == 0:
            g_expo_time_us = expo.value
        gain = c_ushort()
        if g_lib.Toupcam_get_ExpoAGain(g_handle, byref(gain)) == 0:
            g_expo_gain = gain.value


# ================================================================
#   Trackbar 回调
# ================================================================

def on_auto_expo_change(val):
    toggle_auto_exposure(val == 1)
    status = "自动" if val == 1 else "手动"
    print(f"曝光模式: {status}")


def on_expo_time_change(val):
    """滑动条值 (ms) → 转换为 µs"""
    if not g_auto_expo:
        set_exposure_time(val * 1000)


def on_gain_change(val):
    if not g_auto_expo:
        set_gain(val)


def on_temp_change(val):
    set_temp_tint(val, g_tint)


def on_tint_change(val):
    set_temp_tint(g_temp, val)


# ================================================================
#   主程序
# ================================================================

def main():
    global g_auto_expo

    print("=" * 60)
    print("L3CMOS05100KPA 高分辨率图像查看器")
    print("=" * 60)
    print("\n控制键:")
    print("  0 - 最高分辨率 (2560x1922)")
    print("  1 - 中等分辨率 (1280x960)")
    print("  2 - 最低分辨率 (640x480)")
    print("  s - 保存当前图像")
    print("  a - 切换自动/手动曝光")
    print("  w - 一键自动白平衡")
    print("  q - 退出程序")
    print()

    try:
        # 初始化相机 - 使用最高分辨率 (index=0)
        init_camera(resolution_index=0)

        # 等待相机稳定
        time.sleep(0.5)

        # —— 预览窗口 ——
        window_name = "高分辨率预览 (按 q 退出)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 960)

        # —— 参数控制窗口 ——
        ctrl_window = "参数控制"
        cv2.namedWindow(ctrl_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ctrl_window, 400, 300)

        # 曝光最大值换算为 ms
        expo_max_ms = max(1, g_expo_time_max // 1000)
        expo_cur_ms = max(1, g_expo_time_us // 1000)

        cv2.createTrackbar("Auto Expo", ctrl_window, 1, 1, on_auto_expo_change)
        cv2.createTrackbar("Expo (ms)", ctrl_window, expo_cur_ms, expo_max_ms, on_expo_time_change)
        cv2.createTrackbar("Gain (%)", ctrl_window, g_expo_gain, g_gain_max, on_gain_change)
        cv2.createTrackbar("Temp", ctrl_window, g_temp, TOUPCAM_TEMP_MAX, on_temp_change)
        cv2.createTrackbar("Tint", ctrl_window, g_tint, TOUPCAM_TINT_MAX, on_tint_change)

        frame_count = 0
        start_time = time.time()
        read_interval = 0  # 用于定期刷新曝光/增益读数

        while True:
            # 捕获帧
            img = capture_frame()

            if img is not None:
                frame_count += 1

                # 计算帧率
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # —— 自动曝光模式下定期同步读数 ——
                read_interval += 1
                if g_auto_expo and read_interval >= 15:
                    read_interval = 0
                    read_current_expo_gain()
                    # 同步滑动条位置
                    cv2.setTrackbarPos("Expo (ms)", ctrl_window, max(1, g_expo_time_us // 1000))
                    cv2.setTrackbarPos("Gain (%)", ctrl_window, g_expo_gain)

                # —— OSD 信息叠加 ——
                expo_mode_str = "AUTO" if g_auto_expo else "MANUAL"
                expo_ms = g_expo_time_us / 1000.0

                osd_lines = [
                    f"Res: {img.shape[1]}x{img.shape[0]} | FPS: {fps:.1f}",
                    f"Expo: {expo_ms:.1f}ms ({expo_mode_str}) | Gain: {g_expo_gain}%",
                    f"WB Temp: {g_temp} | Tint: {g_tint}",
                ]

                y0 = 30
                for i, line in enumerate(osd_lines):
                    y = y0 + i * 30
                    # 黑色描边 + 绿色文字
                    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示图像
                cv2.imshow(window_name, img)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                if img is not None:
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"图像已保存: {filename}")
            elif key == ord('a'):
                # 切换自动/手动曝光
                g_auto_expo = not g_auto_expo
                toggle_auto_exposure(g_auto_expo)
                cv2.setTrackbarPos("Auto Expo", ctrl_window, 1 if g_auto_expo else 0)
                status = "自动" if g_auto_expo else "手动"
                print(f"曝光模式已切换: {status}")
            elif key == ord('w'):
                # 一键自动白平衡
                auto_white_balance()
            elif key == ord('0'):
                print("切换到最高分辨率...")
                close_camera()
                time.sleep(0.2)
                init_camera(0)
                frame_count = 0
                start_time = time.time()
            elif key == ord('1'):
                print("切换到中等分辨率...")
                close_camera()
                time.sleep(0.2)
                init_camera(1)
                frame_count = 0
                start_time = time.time()
            elif key == ord('2'):
                print("切换到最低分辨率...")
                close_camera()
                time.sleep(0.2)
                init_camera(2)
                frame_count = 0
                start_time = time.time()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        close_camera()
        cv2.destroyAllWindows()
        print("程序结束")


if __name__ == "__main__":
    main()
