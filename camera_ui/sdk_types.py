"""
SDK 常量和 ctypes 结构体定义 — Toupcam / Nncam SDK
"""

import ctypes
from ctypes import *
import os as _os

# ================================================================
#   相机 SDK 常量和结构体
# ================================================================

def _find_dll():
    """自动查找 nncam.dll，支持多路径搜索"""
    base = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), '..'))
    candidates = [
        _os.path.join(base, 'nncamsdk.20171211', 'nncamsdk.20171211', 'win', 'x64', 'nncam.dll'),
        _os.path.join(base, 'drivers', 'x64', 'nncam.dll'),
        r'F:\ImageView\nncamsdk.20171211\nncamsdk.20171211\win\x64\nncam.dll',  # 旧版兼容
    ]
    for p in candidates:
        if _os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"找不到 nncam.dll！请将 SDK 放置在以下任一位置:\n"
        + "\n".join(f"  - {c}" for c in candidates[:2])
    )

DLL_PATH = _find_dll()

TOUPCAM_MAX = 16
TOUPCAM_EVENT_IMAGE = 0x0004
TOUPCAM_EVENT_ERROR = 0x0080
TOUPCAM_EVENT_DISCONNECTED = 0x0081

TOUPCAM_TEMP_DEF = 6503
TOUPCAM_TEMP_MIN = 2000
TOUPCAM_TEMP_MAX = 15000
TOUPCAM_TINT_DEF = 1000
TOUPCAM_TINT_MIN = 200
TOUPCAM_TINT_MAX = 2500

CALLBACK_TYPE = CFUNCTYPE(None, c_uint, c_void_p)
AWB_TT_CALLBACK_TYPE = CFUNCTYPE(None, c_int, c_int, c_void_p)


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

class RECT(Structure):
    _fields_ = [
        ("left", c_int), ("top", c_int),
        ("right", c_int), ("bottom", c_int)
    ]
