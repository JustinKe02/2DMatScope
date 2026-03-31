"""
NNcam 相机诊断脚本
用于检测相机连接状态和基本功能
"""

import ctypes
from ctypes import *
import os
import sys

# ============== DLL 路径配置 ==============
# 尝试不同的 DLL 路径
DLL_PATHS = [
    r'F:\ImageView\nncamsdk.20171211\nncamsdk.20171211\win\x64\nncam.dll',
    r'F:\ImageView\x64\toupcam.dll',
]

# ============== 常量 ==============
TOUPCAM_MAX = 16

# ============== 结构体 ==============
class ToupcamResolution(Structure):
    _fields_ = [("width", c_uint), ("height", c_uint)]

class ToupcamModelV2(Structure):
    _fields_ = [
        ("name", c_wchar_p),
        ("flag", c_ulonglong),
        ("maxspeed", c_uint),
        ("preview", c_uint),
        ("still", c_uint),
        ("maxfanspeed", c_uint),
        ("xpixsz", c_float),
        ("ypixsz", c_float),
        ("res", ToupcamResolution * TOUPCAM_MAX)
    ]
    _pack_ = 8

class ToupcamInstV2(Structure):
    _fields_ = [
        ("displayname", c_wchar * 64),
        ("id", c_wchar * 64),
        ("model", POINTER(ToupcamModelV2))
    ]


def test_dll(dll_path):
    """测试单个 DLL"""
    print(f"\n{'='*60}")
    print(f"测试 DLL: {dll_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(dll_path):
        print(f"  ✗ 文件不存在!")
        return False
    
    try:
        lib = ctypes.CDLL(dll_path)
        print(f"  ✓ DLL 加载成功")
    except Exception as e:
        print(f"  ✗ DLL 加载失败: {e}")
        return False
    
    # 测试版本函数
    try:
        lib.Toupcam_Version.restype = c_wchar_p
        version = lib.Toupcam_Version()
        print(f"  ✓ SDK 版本: {version}")
    except Exception as e:
        print(f"  ✗ 获取版本失败: {e}")
    
    # 枚举相机
    try:
        lib.Toupcam_EnumV2.restype = c_uint
        lib.Toupcam_EnumV2.argtypes = [POINTER(ToupcamInstV2)]
        
        devices = (ToupcamInstV2 * TOUPCAM_MAX)()
        count = lib.Toupcam_EnumV2(devices)
        
        print(f"\n  相机枚举结果: 发现 {count} 个相机")
        
        if count == 0:
            print("\n  ⚠ 没有检测到相机!")
            print("  可能的原因:")
            print("    1. 相机未连接或未通电")
            print("    2. 相机驱动未正确安装")
            print("    3. 相机被其他程序占用 (如 ImageView)")
            print("    4. USB 连接问题")
            print("\n  解决方法:")
            print("    1. 请确保关闭 ImageView 软件")
            print("    2. 检查相机 USB 连接")
            print("    3. 在设备管理器中检查相机是否正常")
            return False
        
        for i in range(count):
            dev = devices[i]
            print(f"\n  相机 {i}:")
            print(f"    显示名称: {dev.displayname}")
            print(f"    ID: {dev.id}")
            
            if dev.model:
                model = dev.model.contents
                print(f"    型号名称: {model.name}")
                print(f"    最大速度级别: {model.maxspeed}")
                print(f"    预览分辨率数量: {model.preview}")
                print(f"    静态分辨率数量: {model.still}")
                print(f"    像素尺寸: {model.xpixsz:.2f} x {model.ypixsz:.2f} µm")
                
                print(f"    可用分辨率:")
                for j in range(model.preview):
                    res = model.res[j]
                    print(f"      [{j}] {res.width} x {res.height}")
        
        # 尝试打开第一个相机
        if count > 0:
            print(f"\n  尝试打开相机...")
            
            lib.Toupcam_Open.restype = c_void_p
            lib.Toupcam_Open.argtypes = [c_wchar_p]
            
            handle = lib.Toupcam_Open(devices[0].id)
            
            if handle:
                print(f"  ✓ 相机打开成功! Handle: {handle}")
                
                # 获取当前分辨率
                lib.Toupcam_get_Size.restype = c_long
                lib.Toupcam_get_Size.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
                
                width = c_int()
                height = c_int()
                result = lib.Toupcam_get_Size(handle, byref(width), byref(height))
                print(f"    当前分辨率: {width.value} x {height.value}")
                
                # 获取曝光时间
                try:
                    lib.Toupcam_get_ExpoTime.restype = c_long
                    lib.Toupcam_get_ExpoTime.argtypes = [c_void_p, POINTER(c_uint)]
                    expo = c_uint()
                    lib.Toupcam_get_ExpoTime(handle, byref(expo))
                    print(f"    当前曝光时间: {expo.value} µs ({expo.value/1000:.1f} ms)")
                except:
                    pass
                
                # 关闭相机
                lib.Toupcam_Close.argtypes = [c_void_p]
                lib.Toupcam_Close(handle)
                print(f"  ✓ 相机已关闭")
                
                return True
            else:
                print(f"  ✗ 无法打开相机!")
                print("    相机可能被其他程序占用")
                return False
                
    except Exception as e:
        print(f"  ✗ 枚举相机失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("NNcam/ToupCam 相机诊断工具")
    print("="*60)
    
    print("\n重要提示:")
    print("  1. 请确保 ImageView 软件已关闭")
    print("  2. 相机需要通过 USB 正确连接")
    
    success = False
    for dll_path in DLL_PATHS:
        if test_dll(dll_path):
            success = True
            print(f"\n✓ 使用此 DLL 成功: {dll_path}")
            break
    
    if not success:
        print("\n" + "="*60)
        print("所有 DLL 测试失败!")
        print("="*60)
        print("\n请检查:")
        print("  1. 相机是否已连接并通电")
        print("  2. ImageView 或其他相机软件是否已关闭")
        print("  3. 相机驱动是否已正确安装")
        print("  4. 在设备管理器中查看相机状态")
    
    print("\n按回车键退出...")
    input()


if __name__ == "__main__":
    main()
