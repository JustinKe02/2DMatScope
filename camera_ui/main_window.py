"""
MainWindow — 2D Material Detection System 主窗口
"""

import sys
import os
import time
import numpy as np
import cv2

# PyTorch
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QSlider,
    QFileDialog, QFrame, QSpinBox, QComboBox, QSplitter,
    QSizePolicy, QMessageBox, QRadioButton, QButtonGroup,
    QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QTimer, QSettings
from PyQt5.QtGui import QFont, QColor

from .sdk_types import (
    TOUPCAM_TEMP_DEF, TOUPCAM_TEMP_MIN, TOUPCAM_TEMP_MAX,
    TOUPCAM_TINT_DEF, TOUPCAM_TINT_MIN, TOUPCAM_TINT_MAX,
)
from .camera_controller import CameraController
from .image_label import ImageLabel
from .inference_engine import InferenceWorker, overlay_mask
from .enhancement import apply_enhancements
from . import styles


class MainWindow(QMainWindow):
    """2D Material Detection System 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Material Detection System")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)

        # 相机控制器
        self.camera = CameraController()
        self.camera_running = False
        self.current_frame = None       # 当前原始帧
        self.detection_result = None    # 检测结果帧

        # 深度学习模型
        self.model = None
        self.model_path = ""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ["Background", "Monolayer", "Fewlayer", "Multilayer"]
        self.class_colors_np = np.array([
            [0, 0, 0],          # Background
            [239, 41, 41],      # Monolayer — 红
            [0, 170, 0],        # Fewlayer — 绿
            [114, 159, 207],    # Multilayer — 蓝
        ], dtype=np.uint8)
        self.class_colors = [
            QColor(0, 0, 0),
            QColor(239, 41, 41),
            QColor(0, 170, 0),
            QColor(114, 159, 207),
        ]

        # 推理参数
        self.crop_size = 512
        self.stride = 384
        self.continuous_detect = False   # 连续检测模式
        self.model_variant = 'small'     # tiny / small / base
        self.inference_mode = 'standard'  # fast / standard / precision
        self.display_filter = 'full'      # full / monolayer / fewlayer / multilayer

        # ImageNet 归一化参数
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # 帧率统计
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # 图像增强参数
        self.enhance_sharpen = 0.0   # USM 锐化强度 0~3.0
        self.enhance_gamma = 1.0     # Gamma 校正 0.5~2.5
        self.enhance_clahe = 0.0     # CLAHE clip limit 0~5.0 (0 = 关闭)

        # 字体缩放因子 (1.0 = 默认, 可动态调整)
        self._font_scale = 1.0

        # 视频录制
        self._recording = False
        self._video_writer = None        # 相机画面录制器
        self._video_writer_det = None    # 检测结果录制器
        self._record_fps = 15.0          # 录制帧率
        self._record_dir = "recordings"  # 录制保存目录
        self._record_start_time = 0      # 录制开始时间
        self._record_timer = None        # 录制时长定时器

        self._build_ui()
        self._apply_dark_theme()

        # 定时器 — 拉取帧
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)

        # 定时器 — 刷新读数（曝光/增益/白平衡）
        self.read_timer = QTimer()
        self.read_timer.timeout.connect(self._refresh_readings)

        # 后台推理线程
        self._inference_worker = InferenceWorker()
        self._inference_worker.result_ready.connect(self._on_inference_result)
        self._inference_worker.error_occurred.connect(
            lambda msg: self.statusBar().showMessage(msg, 5000)
        )
        self._inference_worker.class_names = self.class_names
        self._inference_worker.device = self.device
        self._inference_worker.crop_size = self.crop_size
        self._inference_worker.stride = self.stride
        self._inference_worker.img_mean = self.img_mean
        self._inference_worker.img_std = self.img_std
        self._inference_worker.start()

        # 加载保存的设置 (必须在 worker 创建之后)
        self._load_settings()

        self.statusBar().showMessage("就绪 — 请点击 [Start Camera] 启动相机")

    # ────────────────────────────────────────────────────────────
    #   界面构建
    # ────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ========================================================
        # 第1行: 全宽操作按钮栏 (每个按钮等分宽度, 大字体+图标)
        # ========================================================
        btn_bar = QFrame()
        btn_bar.setFixedHeight(50)
        btn_bar.setStyleSheet("""
            QFrame { background: #14142a; border-bottom: 1px solid #2a2a3e; }
        """)
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(4, 4, 4, 4)
        btn_layout.setSpacing(4)

        # 公共按钮样式 (大字体 16px)
        BIG_BTN = """
            QPushButton {
                background-color: #1e1e36;
                color: #e0e0e0;
                border: 1px solid #3a3a4e;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                padding: 8px 4px;
            }
            QPushButton:hover { background-color: #2a2a50; border-color: #5a5a7e; }
            QPushButton:pressed { background-color: #3a3a60; }
            QPushButton:disabled { background-color: #12122a; color: #555; }
        """

        self.btn_camera = QPushButton("\U0001F4F7  Camera ON/OFF  \u25cf")
        self.btn_camera.setStyleSheet(BIG_BTN)
        self.btn_camera.clicked.connect(self._toggle_camera)

        self.btn_load_model = QPushButton("\U0001F4C2  Load Model")
        self.btn_load_model.setStyleSheet(BIG_BTN)
        self.btn_load_model.clicked.connect(self._load_model)

        self.btn_detect = QPushButton("\u25b6 \u23f8  Detection RUN/STOP  \u25cf")
        self.btn_detect.setStyleSheet(BIG_BTN)
        self.btn_detect.clicked.connect(self._run_detection)
        self.btn_detect.setEnabled(False)

        self.btn_save_image = QPushButton("\U0001F4BE  Save Image")
        self.btn_save_image.setStyleSheet(BIG_BTN)
        self.btn_save_image.clicked.connect(self._save_image)

        self.btn_save_result = QPushButton("\U0001F4BE  Save Result")
        self.btn_save_result.setStyleSheet(BIG_BTN)
        self.btn_save_result.clicked.connect(self._save_result)

        self.btn_advanced = QPushButton("\u2699  Advanced")
        self.btn_advanced.setStyleSheet(BIG_BTN)
        self.btn_advanced.clicked.connect(self._show_advanced_settings)

        # 录制按钮
        self.btn_record = QPushButton("\u23fa  Record")
        self.btn_record.setStyleSheet(BIG_BTN)
        self.btn_record.clicked.connect(self._toggle_recording)

        for btn in [self.btn_camera, self.btn_load_model, self.btn_detect,
                     self.btn_save_image, self.btn_save_result, self.btn_record, self.btn_advanced]:
            btn_layout.addWidget(btn, 1)  # stretch=1 使按钮等分宽度

        root.addWidget(btn_bar)

        # ========================================================
        # 第2行: Live Status & Metrics | Quick Controls
        # ========================================================
        info_bar = QFrame()
        info_bar.setMinimumHeight(50)  # 最小高度 (可拖动调整)
        info_bar.setStyleSheet("""
            QFrame { background: #161628; border-bottom: 1px solid #2a2a3e; }
        """)
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(4, 4, 4, 4)
        info_layout.setSpacing(0)

        # 使用水平 QSplitter 使 Status 和 Controls 宽度可拖动调整
        self.info_splitter = QSplitter(Qt.Horizontal)
        self.info_splitter.setStyleSheet("""
            QSplitter { background: transparent; }
            QSplitter::handle { background: #2a2a40; width: 4px; }
            QSplitter::handle:hover { background: #4a4a6a; }
        """)

        # --- 左侧: Live Status & Metrics ---
        status_box = QFrame()
        status_box.setStyleSheet("""
            QFrame {
                background: #1a1a30;
                border: 1px solid #2a2a40;
                border-radius: 6px;
            }
        """)
        status_inner = QVBoxLayout(status_box)
        status_inner.setContentsMargins(10, 4, 10, 4)
        status_inner.setSpacing(2)

        # 标题
        self.lbl_status_title = QLabel("Live Status & Metrics")
        self.lbl_status_title.setStyleSheet("color:#999; font-size:13px; font-weight:bold; border:none;")
        status_inner.addWidget(self.lbl_status_title)

        # 状态行1: FPS Exp Gain AWB
        s_row1 = QHBoxLayout()
        s_row1.setSpacing(20)
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color:#e0e0e0; border:none; font-size:15px; font-weight:bold;")
        self.lbl_expo_info = QLabel("Exp: --")
        self.lbl_expo_info.setStyleSheet("color:#e0e0e0; border:none; font-size:15px;")
        self.lbl_awb_status = QLabel("AWB: OFF")
        self.lbl_awb_status.setStyleSheet("color:#e0e0e0; border:none; font-size:15px;")
        s_row1.addWidget(self.lbl_fps)
        s_row1.addWidget(self.lbl_expo_info)
        s_row1.addWidget(self.lbl_awb_status)
        s_row1.addStretch()
        status_inner.addLayout(s_row1)

        # 状态行2: Model info
        s_row2 = QHBoxLayout()
        s_row2.setSpacing(20)
        self.lbl_model_info = QLabel("Model: Not loaded")
        self.lbl_model_info.setStyleSheet("color:#e07030; border:none; font-size:15px;")
        self.lbl_resolution = QLabel("Resolution: --")
        self.lbl_resolution.setStyleSheet("color:#8ae68a; border:none; font-size:15px;")
        s_row2.addWidget(self.lbl_model_info)
        s_row2.addWidget(self.lbl_resolution)

        # 显示过滤模式
        lbl_filter = QLabel("  Display:")
        lbl_filter.setStyleSheet("color:#999; border:none; font-size:13px;")
        self.combo_display_filter = QComboBox()
        self.combo_display_filter.addItems(["Full", "Monolayer", "Fewlayer", "Multilayer"])
        self.combo_display_filter.setStyleSheet("""
            QComboBox {
                background-color: #2a2a3a;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 6px;
                color: #e0e0e0;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }
            QComboBox:hover { border-color: #6c6ce0; }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a3a;
                color: #e0e0e0;
                selection-background-color: #4a4a6a;
            }
        """)
        self.combo_display_filter.currentTextChanged.connect(self._on_display_filter_changed)
        s_row2.addWidget(lbl_filter)
        s_row2.addWidget(self.combo_display_filter)
        s_row2.addStretch()
        status_inner.addLayout(s_row2)

        self.info_splitter.addWidget(status_box)

        # --- 右侧: Quick Controls ---
        ctrl_box = QFrame()
        ctrl_box.setStyleSheet("""
            QFrame {
                background: #1a1a30;
                border: 1px solid #2a2a40;
                border-radius: 6px;
            }
        """)
        ctrl_inner = QVBoxLayout(ctrl_box)
        ctrl_inner.setContentsMargins(10, 4, 10, 4)
        ctrl_inner.setSpacing(2)

        self.lbl_ctrl_title = QLabel("Quick Controls")
        self.lbl_ctrl_title.setStyleSheet("color:#999; font-size:13px; font-weight:bold; border:none;")
        ctrl_inner.addWidget(self.lbl_ctrl_title)

        # Exposure 行
        expo_row = QHBoxLayout()
        expo_row.setSpacing(6)
        self.lbl_expo = QLabel("Exposure:")
        self.lbl_expo.setStyleSheet("color:#ccc; border:none; font-size:14px;")
        self.lbl_expo.setMinimumWidth(70)
        self.cb_auto_expo = QCheckBox("Auto")
        self.cb_auto_expo.setChecked(True)
        self.cb_auto_expo.setStyleSheet("color:#ccc; border:none; font-size:13px;")
        self.cb_auto_expo.toggled.connect(self._on_auto_expo_changed)
        self.slider_expo = QSlider(Qt.Horizontal)
        self.slider_expo.setRange(1, 2000)
        self.slider_expo.setValue(70)
        self.slider_expo.setMinimumWidth(50)
        self.slider_expo.valueChanged.connect(self._on_expo_slider)
        self.spin_expo = QSpinBox()
        self.spin_expo.setRange(1, 2000)
        self.spin_expo.setValue(70)
        self.spin_expo.setSuffix(" ms")
        self.spin_expo.setMinimumWidth(65)
        self.spin_expo.setStyleSheet("font-size:13px;")
        self.spin_expo.valueChanged.connect(self._on_expo_spin)
        # Resolution 下拉
        self.lbl_res = QLabel("Resolution:")
        self.lbl_res.setStyleSheet("color:#ccc; border:none; font-size:14px;")
        self.combo_res = QComboBox()
        self.combo_res.addItems(["2560x1922", "1280x960", "640x480"])
        self.combo_res.setMinimumWidth(100)
        self.combo_res.setStyleSheet("font-size:13px;")
        self.combo_res.currentIndexChanged.connect(self._on_resolution_changed)
        expo_row.addWidget(self.lbl_expo)
        expo_row.addWidget(self.cb_auto_expo)
        expo_row.addWidget(self.slider_expo, 1)
        expo_row.addWidget(self.spin_expo)
        expo_row.addSpacing(12)
        expo_row.addWidget(self.lbl_res)
        expo_row.addWidget(self.combo_res)
        ctrl_inner.addLayout(expo_row)

        # Gain 行
        gain_row = QHBoxLayout()
        gain_row.setSpacing(6)
        self.lbl_gain = QLabel("Gain:")
        self.lbl_gain.setStyleSheet("color:#ccc; border:none; font-size:14px;")
        self.lbl_gain.setMinimumWidth(70)
        self.slider_gain = QSlider(Qt.Horizontal)
        self.slider_gain.setRange(100, 500)
        self.slider_gain.setValue(100)
        self.slider_gain.setMinimumWidth(60)
        self.slider_gain.valueChanged.connect(self._on_gain_slider)
        self.spin_gain = QSpinBox()
        self.spin_gain.setRange(100, 500)
        self.spin_gain.setValue(100)
        self.spin_gain.setSuffix("%")
        self.spin_gain.setMinimumWidth(55)
        self.spin_gain.setStyleSheet("font-size:13px;")
        self.spin_gain.valueChanged.connect(self._on_gain_spin)
        gain_row.addWidget(self.lbl_gain)
        gain_row.addWidget(self.slider_gain, 1)
        gain_row.addWidget(self.spin_gain)
        gain_row.addStretch()
        ctrl_inner.addLayout(gain_row)

        self.info_splitter.addWidget(ctrl_box)
        # 初始宽度比例 ~1:2
        self.info_splitter.setSizes([350, 650])

        info_layout.addWidget(self.info_splitter)
        # 禁止折叠，只允许缩放
        self.info_splitter.setCollapsible(0, False)
        self.info_splitter.setCollapsible(1, False)
        # 将 info_bar 和显示区放入垂直 QSplitter ，允许拖动调整控制区与图像区的比例
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setStyleSheet("""
            QSplitter { background: #0a0a14; }
            QSplitter::handle { background: #2a2a40; height: 4px; }
            QSplitter::handle:hover { background: #4a4a6a; }
        """)
        self.main_splitter.addWidget(info_bar)
        # 禁止折叠，只调整大小 (第二个 widget 添加后再设置)
        self.main_splitter.setCollapsible(0, False)

        # ========================================================
        # 中间: 双面板显示区 (使用 QSplitter 可拖动调整大小)
        # ========================================================
        # 左侧: LIVE CAMERA FEED
        left_widget = QWidget()
        left_widget.setStyleSheet("background:#0a0a14;")
        left_panel = QVBoxLayout(left_widget)
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(0)
        lbl_cam_hdr = QLabel("  LIVE CAMERA FEED")
        lbl_cam_hdr.setFixedHeight(26)
        lbl_cam_hdr.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lbl_cam_hdr.setStyleSheet(
            "background:#1a1a30; color:#aaa; font-size:13px; font-weight:bold; "
            "letter-spacing:2px; border:1px solid #2a2a40; border-bottom:none; "
            "border-top-left-radius:4px; border-top-right-radius:4px;"
        )
        self.camera_view = ImageLabel("Camera View")
        left_panel.addWidget(lbl_cam_hdr)
        left_panel.addWidget(self.camera_view, 1)

        # 右侧:  DETECTION RESULT
        right_widget = QWidget()
        right_widget.setStyleSheet("background:#0a0a14;")
        right_panel = QVBoxLayout(right_widget)
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(0)
        lbl_det_hdr = QLabel("  DETECTION RESULT")
        lbl_det_hdr.setFixedHeight(26)
        lbl_det_hdr.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lbl_det_hdr.setStyleSheet(
            "background:#1a1a30; color:#aaa; font-size:13px; font-weight:bold; "
            "letter-spacing:2px; border:1px solid #2a2a40; border-bottom:none; "
            "border-top-left-radius:4px; border-top-right-radius:4px;"
        )
        self.detection_view = ImageLabel("Detection Result")
        right_panel.addWidget(lbl_det_hdr)
        right_panel.addWidget(self.detection_view, 1)

        # 使用 QSplitter 允许用户拖动调整左右面板宽度
        self.view_splitter = QSplitter(Qt.Horizontal)
        self.view_splitter.setStyleSheet("""
            QSplitter { background: #0a0a14; }
            QSplitter::handle { background: #2a2a40; width: 4px; }
            QSplitter::handle:hover { background: #4a4a6a; }
        """)
        self.view_splitter.addWidget(left_widget)
        self.view_splitter.addWidget(right_widget)
        self.view_splitter.setSizes([500, 500])  # 初始等分
        self.view_splitter.setCollapsible(0, False)
        self.view_splitter.setCollapsible(1, False)

        # 将图像显示区添加到主分割器
        self.main_splitter.addWidget(self.view_splitter)
        self.main_splitter.setCollapsible(1, False)  # 禁止 view 区域折叠
        self.main_splitter.setStretchFactor(0, 0)  # info_bar 不拉伸
        self.main_splitter.setStretchFactor(1, 1)  # views 区拉伸填满
        self.main_splitter.setSizes([80, 600])     # 初始高度比例
        root.addWidget(self.main_splitter, 1)

        # ========================================================
        # 底部: 检测摘要栏
        # ========================================================
        self.summary_bar = QFrame()
        self.summary_bar.setFixedHeight(38)
        self.summary_bar.setStyleSheet("""
            QFrame {
                background: #161628;
                border-top: 1px solid #2a2a3e;
            }
        """)
        sum_layout = QHBoxLayout(self.summary_bar)
        sum_layout.setContentsMargins(12, 2, 12, 2)
        sum_layout.setSpacing(8)
        self.lbl_summary = QLabel("DETECTION SUMMARY:")
        self.lbl_summary.setStyleSheet("color:#999; font-size:15px; font-weight:bold; border:none;")
        self.lbl_summary_detail = QLabel("")
        self.lbl_summary_detail.setStyleSheet("color:#e0e0e0; font-size:15px; border:none;")
        sum_layout.addWidget(self.lbl_summary)
        sum_layout.addWidget(self.lbl_summary_detail, 1)
        root.addWidget(self.summary_bar)

        # ROI 相关控件（移至 Advanced 弹窗中初始化, 此处仅创建变量）
        self.rb_roi_none = QRadioButton("Off")
        self.rb_roi_none.setChecked(True)
        self.rb_roi_ae = QRadioButton("AE")
        self.rb_roi_awb = QRadioButton("AWB")
        self.roi_btn_group = QButtonGroup()
        self.roi_btn_group.addButton(self.rb_roi_none, 0)
        self.roi_btn_group.addButton(self.rb_roi_ae, 1)
        self.roi_btn_group.addButton(self.rb_roi_awb, 2)
        self.roi_btn_group.buttonClicked.connect(self._on_roi_mode_changed)

        # 构建高级设置弹窗
        self._build_advanced_dialog()

    def _big_btn_style(self):
        """全局大按钮样式"""
        return styles.big_btn_style()

    def _toggle_btn_style(self, color_on, color_off):
        """保留兼容: 切换按钮样式"""
        return self._big_btn_style()

    def _build_advanced_dialog(self):
        """构建高级设置弹窗 (深色主题)"""
        self.adv_dialog = QDialog(self)
        self.adv_dialog.setWindowTitle("Advanced Settings")
        self.adv_dialog.setMinimumWidth(460)
        # 为 QDialog 应用深色主题
        self.adv_dialog.setStyleSheet(styles.ADV_DIALOG_STYLE)

        dlg_layout = QVBoxLayout(self.adv_dialog)
        dlg_layout.setSpacing(12)
        dlg_layout.setContentsMargins(16, 12, 16, 12)

        # -- 白平衡 --
        wb_group = QGroupBox("White Balance")
        wb_layout = QVBoxLayout(wb_group)
        wb_layout.setSpacing(8)

        self.cb_auto_wb = QCheckBox("Auto White Balance")
        self.cb_auto_wb.toggled.connect(self._on_auto_wb)
        wb_layout.addWidget(self.cb_auto_wb)

        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature:"))
        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(TOUPCAM_TEMP_MIN, TOUPCAM_TEMP_MAX)
        self.slider_temp.setValue(TOUPCAM_TEMP_DEF)
        self.slider_temp.valueChanged.connect(self._on_temp_slider)
        self.spin_temp = QSpinBox()
        self.spin_temp.setRange(TOUPCAM_TEMP_MIN, TOUPCAM_TEMP_MAX)
        self.spin_temp.setValue(TOUPCAM_TEMP_DEF)
        self.spin_temp.setFixedWidth(70)
        self.spin_temp.valueChanged.connect(self._on_temp_spin)
        temp_row.addWidget(self.slider_temp, 1)
        temp_row.addWidget(self.spin_temp)
        wb_layout.addLayout(temp_row)

        tint_row = QHBoxLayout()
        tint_row.addWidget(QLabel("Tint:"))
        self.slider_tint = QSlider(Qt.Horizontal)
        self.slider_tint.setRange(TOUPCAM_TINT_MIN, TOUPCAM_TINT_MAX)
        self.slider_tint.setValue(TOUPCAM_TINT_DEF)
        self.slider_tint.valueChanged.connect(self._on_tint_slider)
        self.spin_tint = QSpinBox()
        self.spin_tint.setRange(TOUPCAM_TINT_MIN, TOUPCAM_TINT_MAX)
        self.spin_tint.setValue(TOUPCAM_TINT_DEF)
        self.spin_tint.setFixedWidth(70)
        self.spin_tint.valueChanged.connect(self._on_tint_spin)
        tint_row.addWidget(self.slider_tint, 1)
        tint_row.addWidget(self.spin_tint)
        wb_layout.addLayout(tint_row)
        dlg_layout.addWidget(wb_group)

        # -- 图像增强 --
        enhance_group = QGroupBox("Image Enhancement")
        enhance_layout = QVBoxLayout(enhance_group)
        enhance_layout.setSpacing(8)

        # USM
        sharp_row = QHBoxLayout()
        sharp_row.addWidget(QLabel("Sharpen (USM):"))
        self.slider_sharpen = QSlider(Qt.Horizontal)
        self.slider_sharpen.setRange(0, 30)
        self.slider_sharpen.setValue(0)
        self.slider_sharpen.valueChanged.connect(self._on_sharpen_changed)
        self.lbl_sharpen_val = QLabel("0.0")
        self.lbl_sharpen_val.setFixedWidth(36)
        sharp_row.addWidget(self.slider_sharpen, 1)
        sharp_row.addWidget(self.lbl_sharpen_val)
        enhance_layout.addLayout(sharp_row)

        # Gamma
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma:"))
        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(5, 25)
        self.slider_gamma.setValue(10)
        self.slider_gamma.valueChanged.connect(self._on_gamma_changed)
        self.lbl_gamma_val = QLabel("1.0")
        self.lbl_gamma_val.setFixedWidth(36)
        gamma_row.addWidget(self.slider_gamma, 1)
        gamma_row.addWidget(self.lbl_gamma_val)
        enhance_layout.addLayout(gamma_row)

        # CLAHE
        clahe_row = QHBoxLayout()
        clahe_row.addWidget(QLabel("CLAHE:"))
        self.slider_clahe = QSlider(Qt.Horizontal)
        self.slider_clahe.setRange(0, 50)
        self.slider_clahe.setValue(0)
        self.slider_clahe.valueChanged.connect(self._on_clahe_changed)
        self.lbl_clahe_val = QLabel("OFF")
        self.lbl_clahe_val.setFixedWidth(36)
        clahe_row.addWidget(self.slider_clahe, 1)
        clahe_row.addWidget(self.lbl_clahe_val)
        enhance_layout.addLayout(clahe_row)
        dlg_layout.addWidget(enhance_group)

        # -- 字体大小调节 --
        font_group = QGroupBox("Font Size")
        font_layout = QHBoxLayout(font_group)
        font_layout.setSpacing(8)
        font_layout.addWidget(QLabel("Scale:"))
        self.slider_font_scale = QSlider(Qt.Horizontal)
        self.slider_font_scale.setRange(50, 200)  # 50%~200%
        self.slider_font_scale.setValue(100)
        self.slider_font_scale.valueChanged.connect(self._on_font_scale_changed)
        self.lbl_font_scale_val = QLabel("100%")
        self.lbl_font_scale_val.setFixedWidth(50)
        font_layout.addWidget(self.slider_font_scale, 1)
        font_layout.addWidget(self.lbl_font_scale_val)
        dlg_layout.addWidget(font_group)

        # -- ROI 控制 --
        roi_group = QGroupBox("ROI Mode")
        roi_layout = QHBoxLayout(roi_group)
        roi_layout.setSpacing(8)
        roi_layout.addWidget(self.rb_roi_none)
        roi_layout.addWidget(self.rb_roi_ae)
        roi_layout.addWidget(self.rb_roi_awb)
        dlg_layout.addWidget(roi_group)

        # -- 模型变体选择 --
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout(model_group)
        model_layout.setSpacing(8)
        model_layout.addWidget(QLabel("Variant:"))
        self.combo_model_variant = QComboBox()
        self.combo_model_variant.addItems(["tiny", "small", "base"])
        self.combo_model_variant.setCurrentText(self.model_variant)
        self.combo_model_variant.currentTextChanged.connect(self._on_model_variant_changed)
        model_layout.addWidget(self.combo_model_variant, 1)
        dlg_layout.addWidget(model_group)

        # -- 推理模式选择 --
        infer_group = QGroupBox("Inference Mode")
        infer_layout = QHBoxLayout(infer_group)
        infer_layout.setSpacing(8)
        infer_layout.addWidget(QLabel("Mode:"))
        self.combo_infer_mode = QComboBox()
        self.combo_infer_mode.addItems(["fast", "standard", "precision"])
        self.combo_infer_mode.setCurrentText(self.inference_mode)
        self.combo_infer_mode.currentTextChanged.connect(self._on_infer_mode_changed)
        infer_layout.addWidget(self.combo_infer_mode, 1)
        infer_layout.addWidget(QLabel(""))
        self.lbl_infer_desc = QLabel("512 crop, stride 384")
        self.lbl_infer_desc.setStyleSheet("color:#888; font-size:11px;")
        infer_layout.addWidget(self.lbl_infer_desc)
        dlg_layout.addWidget(infer_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.adv_dialog.hide)
        dlg_layout.addWidget(btn_box)

    def _show_advanced_settings(self):
        """显示高级设置弹窗"""
        self.adv_dialog.show()
        self.adv_dialog.raise_()

    def _on_model_variant_changed(self, variant):
        """切换模型变体 (tiny/small/base)"""
        self.model_variant = variant
        self.statusBar().showMessage(f"模型变体已切换为: {variant}，请重新加载模型")

    def _on_infer_mode_changed(self, mode):
        """切换推理模式 (fast/standard/precision)"""
        self.inference_mode = mode
        if mode == 'fast':
            self.crop_size = 512
            self.stride = 512  # 无重叠，最快
            desc = "全图缩放 512, 无重叠"
        elif mode == 'standard':
            self.crop_size = 512
            self.stride = 384  # 25% 重叠
            desc = "512 crop, stride 384"
        else:  # precision
            self.crop_size = 512
            self.stride = 256  # 50% 重叠
            desc = "512 crop, stride 256 (高精度)"

        # 同步到推理线程
        self._inference_worker.crop_size = self.crop_size
        self._inference_worker.stride = self.stride

        self.lbl_infer_desc.setText(desc)
        self.statusBar().showMessage(f"推理模式: {mode} — {desc}")

    def _on_display_filter_changed(self, text):
        """切换显示过滤模式"""
        self.display_filter = text.lower()
        labels = {'full': '全部类别', 'monolayer': '仅 Monolayer',
                  'fewlayer': '仅 Fewlayer', 'multilayer': '仅 Multilayer'}
        self.statusBar().showMessage(f"显示模式: {labels.get(self.display_filter, text)}")

    def _on_font_scale_changed(self, value):
        """动态调整全局字体大小 (50%~200%)"""
        self._font_scale = value / 100.0
        self.lbl_font_scale_val.setText(f"{value}%")

        # 计算缩放后的字体大小
        base = int(14 * self._font_scale)
        btn_size = int(16 * self._font_scale)
        status_size = int(15 * self._font_scale)
        ctrl_size = int(14 * self._font_scale)
        summary_size = int(15 * self._font_scale)

        # 更新状态标签
        for lbl in [self.lbl_fps, self.lbl_expo_info, self.lbl_awb_status]:
            lbl.setStyleSheet(f"color:#e0e0e0; border:none; font-size:{status_size}px; font-weight:bold;")
        self.lbl_model_info.setStyleSheet(f"color:#e07030; border:none; font-size:{status_size}px;")
        self.lbl_resolution.setStyleSheet(f"color:#8ae68a; border:none; font-size:{status_size}px;")

        # 更新摘要栏
        self.lbl_summary.setStyleSheet(f"color:#999; font-size:{summary_size}px; font-weight:bold; border:none;")
        self.lbl_summary_detail.setStyleSheet(f"color:#e0e0e0; font-size:{summary_size}px; border:none;")

        # 更新按钮字体
        new_btn_style = f"""
            QPushButton {{
                background-color: #1e1e36;
                color: #e0e0e0;
                border: 1px solid #3a3a4e;
                border-radius: 6px;
                font-size: {btn_size}px;
                font-weight: bold;
                padding: 8px 4px;
            }}
            QPushButton:hover {{ background-color: #2a2a50; border-color: #5a5a7e; }}
            QPushButton:pressed {{ background-color: #3a3a60; }}
            QPushButton:disabled {{ background-color: #12122a; color: #555; }}
        """
        for btn in [self.btn_camera, self.btn_load_model, self.btn_detect,
                     self.btn_save_image, self.btn_save_result, self.btn_record, self.btn_advanced]:
            if not (btn == self.btn_camera and self.camera_running):
                if not (btn == self.btn_detect and self.continuous_detect):
                    if not (btn == self.btn_record and self._recording):
                        btn.setStyleSheet(new_btn_style)

        # 更新 Quick Controls 区域标签
        title_size = int(13 * self._font_scale)
        self.lbl_status_title.setStyleSheet(f"color:#999; font-size:{title_size}px; font-weight:bold; border:none;")
        self.lbl_ctrl_title.setStyleSheet(f"color:#999; font-size:{title_size}px; font-weight:bold; border:none;")

        # 更新控制标签
        for lbl in [self.lbl_expo, self.lbl_gain, self.lbl_res]:
            lbl.setStyleSheet(f"color:#ccc; border:none; font-size:{ctrl_size}px;")

        # 更新 Auto checkbox
        auto_size = int(13 * self._font_scale)
        self.cb_auto_expo.setStyleSheet(f"color:#ccc; border:none; font-size:{auto_size}px;")

        # 更新 SpinBox 和 ComboBox
        spin_size = int(13 * self._font_scale)
        self.spin_expo.setStyleSheet(f"font-size:{spin_size}px;")
        self.spin_gain.setStyleSheet(f"font-size:{spin_size}px;")
        self.combo_res.setStyleSheet(f"font-size:{spin_size}px;")

        # 动态调整最小宽度，防止字体放大时文字被裁剪
        self.lbl_expo.setMinimumWidth(int(70 * self._font_scale))
        self.lbl_gain.setMinimumWidth(int(70 * self._font_scale))
        self.spin_expo.setMinimumWidth(int(65 * self._font_scale))
        self.spin_gain.setMinimumWidth(int(55 * self._font_scale))
        self.combo_res.setMinimumWidth(int(100 * self._font_scale))

    # ────────────────────────────────────────────────────────────
    #   深色主题
    # ────────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        self.setStyleSheet(styles.DARK_THEME)

    def _group_style(self):
        return styles.GROUP_STYLE

    # ────────────────────────────────────────────────────────────
    #   相机启动 / 停止
    # ────────────────────────────────────────────────────────────

    def _toggle_camera(self):
        if not self.camera_running:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        try:
            res_idx = self.combo_res.currentIndex()
            cam_name = self.camera.open(resolution_index=res_idx)
            self.camera_running = True
            # 按钮变为绿色活跃状态
            self.btn_camera.setStyleSheet("""
                QPushButton {
                    background-color: #1a5c2a;
                    color: #a0ffa0;
                    border: 1px solid #27ae60;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 8px 4px;
                }
                QPushButton:hover { background-color: #1e6e32; }
            """)

            # 同步控件范围
            expo_max_ms = max(1, self.camera.expo_time_max // 1000)
            expo_cur_ms = max(1, self.camera.expo_time_us // 1000)

            self.slider_expo.blockSignals(True)
            self.spin_expo.blockSignals(True)
            self.slider_expo.setRange(1, expo_max_ms)
            self.slider_expo.setValue(expo_cur_ms)
            self.spin_expo.setRange(1, expo_max_ms)
            self.spin_expo.setValue(expo_cur_ms)
            self.slider_expo.blockSignals(False)
            self.spin_expo.blockSignals(False)

            self.slider_gain.blockSignals(True)
            self.spin_gain.blockSignals(True)
            self.slider_gain.setRange(self.camera.gain_min, self.camera.gain_max)
            self.slider_gain.setValue(self.camera.expo_gain)
            self.spin_gain.setRange(self.camera.gain_min, self.camera.gain_max)
            self.spin_gain.setValue(self.camera.expo_gain)
            self.slider_gain.blockSignals(False)
            self.spin_gain.blockSignals(False)

            self.slider_temp.blockSignals(True)
            self.spin_temp.blockSignals(True)
            self.slider_temp.setValue(self.camera.temp)
            self.spin_temp.setValue(self.camera.temp)
            self.slider_temp.blockSignals(False)
            self.spin_temp.blockSignals(False)

            self.slider_tint.blockSignals(True)
            self.spin_tint.blockSignals(True)
            self.slider_tint.setValue(self.camera.tint)
            self.spin_tint.setValue(self.camera.tint)
            self.slider_tint.blockSignals(False)
            self.spin_tint.blockSignals(False)

            # 启动多线程帧抓取
            self.camera.start_grabber()

            # 启动定时器 (显示刷新 ~60fps，帧抓取在后台线程)
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.timer.start(16)        # ~60 fps 显示刷新
            self.read_timer.start(500)  # 每 0.5s 刷新读数

            self.statusBar().showMessage(f"相机已启动: {cam_name} ({self.camera.width}x{self.camera.height})")

        except Exception as e:
            QMessageBox.critical(self, "相机错误", str(e))
            self.statusBar().showMessage(f"错误: {e}")

    def _stop_camera(self):
        self.timer.stop()
        self.read_timer.stop()
        self.camera.close()
        self.camera_running = False
        # 恢复按钮默认样式
        self.btn_camera.setStyleSheet(self._toggle_btn_style("#27ae60", "#c0392b"))
        self.statusBar().showMessage("相机已停止")

    # ────────────────────────────────────────────────────────────
    #   图像增强
    # ────────────────────────────────────────────────────────────

    def _apply_enhancements(self, frame):
        """对帧应用图像增强管线"""
        return apply_enhancements(frame, self.enhance_sharpen, self.enhance_gamma, self.enhance_clahe)

    def _on_sharpen_changed(self, value):
        self.enhance_sharpen = value / 10.0
        self.lbl_sharpen_val.setText(f"{self.enhance_sharpen:.1f}")

    def _on_gamma_changed(self, value):
        self.enhance_gamma = value / 10.0
        self.lbl_gamma_val.setText(f"{self.enhance_gamma:.1f}")

    def _on_clahe_changed(self, value):
        self.enhance_clahe = value / 10.0
        if value == 0:
            self.lbl_clahe_val.setText("OFF")
        else:
            self.lbl_clahe_val.setText(f"{self.enhance_clahe:.1f}")

    # ────────────────────────────────────────────────────────────
    #   定时器回调
    # ────────────────────────────────────────────────────────────

    def _on_timer(self):
        """主定时器 — 从后台线程获取最新帧并显示"""
        if not self.camera_running:
            return

        # 从后台抓取线程获取最新帧 (非阻塞)
        frame = self.camera.get_latest_frame()
        if frame is not None:
            self.current_frame = frame.copy()
            # 应用图像增强
            display_frame = self._apply_enhancements(frame)
            self.camera_view.set_image(display_frame)

            # 如果有新 ROI 待应用，在这里发送给相机
            self._apply_pending_roi()

            # 连续检测模式 (异步推理，不阻塞 UI)
            if self.continuous_detect and self.model is not None:
                self._inference_worker.submit_frame(frame)

            # 视频录制: 写入当前帧
            if self._recording and self._video_writer is not None:
                self._video_writer.write(display_frame)
                # 如果有检测结果，也录制
                if self._video_writer_det is not None and self.detection_result is not None:
                    self._video_writer_det.write(self.detection_result)

            # 帧率统计
            self.frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 1.0:
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()

    def _on_inference_result(self, frame, mask, conf_map):
        """接收后台推理线程的结果，在 UI 线程做叠加和显示"""
        result_img = self._overlay_mask(frame, mask, conf_map)
        self.detection_result = result_img
        self.detection_view.set_image(result_img)

    def _refresh_readings(self):
        """定期刷新曝光/增益/白平衡读数"""
        if not self.camera_running:
            return

        self.camera.refresh_readings()

        # 更新 Info 标签
        self.lbl_resolution.setText(f"Resolution: {self.camera.width}x{self.camera.height}")
        self.lbl_fps.setText(f"FPS: {self.current_fps:.1f}")

        expo_ms = self.camera.expo_time_us / 1000.0
        mode_str = "Auto" if self.camera.auto_expo else "Manual"
        self.lbl_expo_info.setText(f"Exposure: {expo_ms:.1f} ms ({mode_str})")

        # 自动曝光模式下同步滑动条
        if self.camera.auto_expo:
            cur_ms = max(1, self.camera.expo_time_us // 1000)
            self.slider_expo.blockSignals(True)
            self.spin_expo.blockSignals(True)
            self.slider_expo.setValue(cur_ms)
            self.spin_expo.setValue(cur_ms)
            self.slider_expo.blockSignals(False)
            self.spin_expo.blockSignals(False)

            self.slider_gain.blockSignals(True)
            self.spin_gain.blockSignals(True)
            self.slider_gain.setValue(self.camera.expo_gain)
            self.spin_gain.setValue(self.camera.expo_gain)
            self.slider_gain.blockSignals(False)
            self.spin_gain.blockSignals(False)

        # 同步白平衡
        self.slider_temp.blockSignals(True)
        self.spin_temp.blockSignals(True)
        self.slider_temp.setValue(self.camera.temp)
        self.spin_temp.setValue(self.camera.temp)
        self.slider_temp.blockSignals(False)
        self.spin_temp.blockSignals(False)

        self.slider_tint.blockSignals(True)
        self.spin_tint.blockSignals(True)
        self.slider_tint.setValue(self.camera.tint)
        self.spin_tint.setValue(self.camera.tint)
        self.slider_tint.blockSignals(False)
        self.spin_tint.blockSignals(False)

    # ────────────────────────────────────────────────────────────
    #   控件回调
    # ────────────────────────────────────────────────────────────

    def _on_auto_expo_changed(self, checked):
        if self.camera_running:
            self.camera.set_auto_exposure(checked)
        self.slider_expo.setEnabled(not checked)
        self.spin_expo.setEnabled(not checked)
        self.slider_gain.setEnabled(not checked)
        self.spin_gain.setEnabled(not checked)

    def _on_expo_slider(self, val):
        self.spin_expo.blockSignals(True)
        self.spin_expo.setValue(val)
        self.spin_expo.blockSignals(False)
        if self.camera_running and not self.camera.auto_expo:
            self.camera.set_exposure_time(val * 1000)

    def _on_expo_spin(self, val):
        self.slider_expo.blockSignals(True)
        self.slider_expo.setValue(val)
        self.slider_expo.blockSignals(False)
        if self.camera_running and not self.camera.auto_expo:
            self.camera.set_exposure_time(val * 1000)

    def _on_gain_slider(self, val):
        self.spin_gain.blockSignals(True)
        self.spin_gain.setValue(val)
        self.spin_gain.blockSignals(False)
        if self.camera_running and not self.camera.auto_expo:
            self.camera.set_gain(val)

    def _on_gain_spin(self, val):
        self.slider_gain.blockSignals(True)
        self.slider_gain.setValue(val)
        self.slider_gain.blockSignals(False)
        if self.camera_running and not self.camera.auto_expo:
            self.camera.set_gain(val)

    def _on_auto_wb(self, checked):
        if checked and self.camera_running:
            self.camera.auto_white_balance()
            self.statusBar().showMessage("一键自动白平衡已触发")

    def _on_temp_slider(self, val):
        self.spin_temp.blockSignals(True)
        self.spin_temp.setValue(val)
        self.spin_temp.blockSignals(False)
        if self.camera_running:
            self.camera.set_temp_tint(val, self.camera.tint)

    def _on_temp_spin(self, val):
        self.slider_temp.blockSignals(True)
        self.slider_temp.setValue(val)
        self.slider_temp.blockSignals(False)
        if self.camera_running:
            self.camera.set_temp_tint(val, self.camera.tint)

    def _on_tint_slider(self, val):
        self.spin_tint.blockSignals(True)
        self.spin_tint.setValue(val)
        self.spin_tint.blockSignals(False)
        if self.camera_running:
            self.camera.set_temp_tint(self.camera.temp, val)

    def _on_tint_spin(self, val):
        self.slider_tint.blockSignals(True)
        self.slider_tint.setValue(val)
        self.slider_tint.blockSignals(False)
        if self.camera_running:
            self.camera.set_temp_tint(self.camera.temp, val)

    def _on_resolution_changed(self, idx):
        if self.camera_running:
            self._stop_camera()
            self.combo_res.setCurrentIndex(idx)
            # 非阻塞延迟启动，避免 UI 冻结
            QTimer.singleShot(200, self._start_camera)

    # ────────────────────────────────────────────────────────────
    #   模型加载与检测
    # ────────────────────────────────────────────────────────────

    def _load_model(self):
        """加载 RepELA-Net 模型 (异步，不阻塞 UI)"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重文件", "RepELA-Net/output",
            "Model Files (*.pth *.pt);;All Files (*)"
        )
        if not path:
            return

        self.statusBar().showMessage("正在加载模型...")
        self.btn_load_model.setEnabled(False)  # 防止重复点击

        # 在后台线程加载
        from PyQt5.QtCore import QThread as _QThread, pyqtSignal as _Signal

        class _ModelLoader(_QThread):
            finished = _Signal(object, str)  # (model, epoch_info)
            failed = _Signal(str)            # error message

            def __init__(self, path, variant, device, parent=None):
                super().__init__(parent)
                self._path = path
                self._variant = variant
                self._device = device

            def run(self):
                try:
                    import sys as _sys
                    physnet_dir = os.path.normpath(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RepELA-Net')
                    )
                    if physnet_dir not in _sys.path:
                        _sys.path.insert(0, physnet_dir)

                    from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base, infer_use_cse
                    model_fn = {'tiny': repela_net_tiny, 'small': repela_net_small, 'base': repela_net_base}[self._variant]

                    # 先加载 checkpoint，推断配置
                    ckpt = torch.load(self._path, map_location=self._device, weights_only=False)

                    if isinstance(ckpt, dict) and 'model' in ckpt:
                        # 标准 checkpoint (训练产出)
                        sd = ckpt['model']
                        use_cse = infer_use_cse(ckpt, cli_use_cse=False)
                        deploy = any('fused_conv' in k for k in sd.keys())
                        epoch = ckpt.get('epoch')
                        epoch_info = f" (Epoch {epoch + 1})" if isinstance(epoch, int) else ""
                    else:
                        # 原始 state_dict (deploy 导出 或 其他)
                        sd = ckpt if isinstance(ckpt, dict) else {}
                        use_cse = any('color_enhance.s_weight' in k for k in sd.keys())
                        deploy = any('fused_conv' in k for k in sd.keys())
                        epoch_info = " [deploy]" if deploy else ""

                    model = model_fn(num_classes=4, use_cse=use_cse, deploy=deploy).to(self._device)

                    if isinstance(ckpt, dict) and 'model' in ckpt:
                        model.load_state_dict(ckpt['model'])
                    else:
                        model.load_state_dict(ckpt, strict=False)

                    model.eval()
                    self.finished.emit(model, epoch_info)
                except Exception as e:
                    self.failed.emit(str(e))

        loader = _ModelLoader(path, self.model_variant, self.device, parent=self)
        loader.finished.connect(lambda model, info: self._on_model_loaded(model, path, info))
        loader.failed.connect(self._on_model_load_error)
        loader.finished.connect(loader.deleteLater)
        loader.failed.connect(loader.deleteLater)
        self._model_loader = loader  # 保持引用防止 GC
        loader.start()

    def _on_model_loaded(self, model, path, epoch_info):
        """模型加载完成回调 (在 UI 线程)"""
        # 暂停推理线程再替换 model，避免竞态
        was_detecting = self.continuous_detect
        self.continuous_detect = False

        self.model = model
        self.model_path = path
        self._inference_worker.model = model

        self.continuous_detect = was_detecting

        self.statusBar().showMessage(f"模型已加载: {os.path.basename(path)}{epoch_info} [{self.device}]")
        self.lbl_model_info.setText(f"Model: {os.path.basename(path)}")
        self.lbl_model_info.setStyleSheet("color: #80e080; border: none;")
        self.btn_detect.setEnabled(True)
        self.btn_load_model.setEnabled(True)

    def _on_model_load_error(self, error_msg):
        """模型加载失败回调"""
        self.btn_load_model.setEnabled(True)
        QMessageBox.critical(self, "模型加载失败", error_msg)
        self.statusBar().showMessage(f"模型加载失败: {error_msg}")

    def _run_detection(self):
        """切换连续检测模式 / 单帧检测"""
        if self.model is None:
            self.statusBar().showMessage("请先加载模型")
            return

        if self.current_frame is None:
            self.statusBar().showMessage("没有可用的图像帧")
            return

        # 切换连续检测模式
        self.continuous_detect = not self.continuous_detect
        if self.continuous_detect:
            # 按钮变为绿色活跃状态 (检测进行中)
            self.btn_detect.setStyleSheet("""
                QPushButton {
                    background-color: #1a5c2a;
                    color: #a0ffa0;
                    border: 1px solid #27ae60;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 8px 4px;
                }
                QPushButton:hover { background-color: #1e6e32; }
            """)
            self.statusBar().showMessage("连续检测已开启")
        else:
            # 恢复默认样式
            self.btn_detect.setStyleSheet(self._toggle_btn_style("#27ae60", "#c0392b"))
            self.statusBar().showMessage("连续检测已停止")

    def _overlay_mask(self, frame, mask, conf_map=None, alpha=0.35):
        """将分割 mask 叠加到原图上，并更新 GUI 摘要栏"""
        # 根据显示过滤模式决定可见类别
        filter_map = {
            'full': None,  # None = 显示所有
            'monolayer': {1},
            'fewlayer': {2},
            'multilayer': {3},
        }
        visible = filter_map.get(self.display_filter)

        result, detection_counts = overlay_mask(
            frame, mask, conf_map, self.class_names, self.class_colors_np,
            visible_classes=visible, alpha=alpha
        )

        # 更新底部 GUI 摘要栏
        if detection_counts:
            html_parts = []
            for name, cnt in detection_counts.items():
                cid = self.class_names.index(name)
                r, g, b = int(self.class_colors_np[cid][0]), int(self.class_colors_np[cid][1]), int(self.class_colors_np[cid][2])
                html_parts.append(
                    f'<span style="color:rgb({r},{g},{b}); font-weight:bold;">'
                    f'\u25cf {name}: {cnt}</span>'
                )
            self.lbl_summary_detail.setText("  |  ".join(html_parts))
            self.lbl_summary_detail.setTextFormat(Qt.RichText)
        else:
            self.lbl_summary_detail.setText("No detections")

        return result



    # ────────────────────────────────────────────────────────────
    #   保存
    # ────────────────────────────────────────────────────────────

    def _save_image(self):
        if self.current_frame is None:
            self.statusBar().showMessage("没有可保存的图像")
            return
        default_name = f"capture_{int(time.time())}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", default_name,
            "PNG Files (*.png);;TIFF Files (*.tiff *.tif);;BMP Files (*.bmp);;All Files (*)"
        )
        if path:
            cv2.imwrite(path, self.current_frame)
            self.statusBar().showMessage(f"图像已保存: {path}")

    def _save_result(self):
        if self.detection_result is None:
            self.statusBar().showMessage("没有检测结果可保存")
            return
        default_name = f"result_{int(time.time())}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", default_name,
            "PNG Files (*.png);;TIFF Files (*.tiff *.tif);;BMP Files (*.bmp);;All Files (*)"
        )
        if path:
            cv2.imwrite(path, self.detection_result)
            self.statusBar().showMessage(f"检测结果已保存: {path}")

    # ────────────────────────────────────────────────────────────
    #   视频录制
    # ────────────────────────────────────────────────────────────

    def _toggle_recording(self):
        """切换视频录制"""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """开始视频录制"""
        if not self.camera_running:
            self.statusBar().showMessage("请先启动相机再录制")
            return

        # 创建录制目录
        os.makedirs(self._record_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        w = self.camera.width
        h = self.camera.height
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # 使用实际帧率（如果已有统计）
        fps = max(5.0, self.current_fps) if self.current_fps > 1.0 else 15.0
        self._record_fps = fps

        # 相机画面录制
        cam_path = os.path.join(self._record_dir, f"camera_{timestamp}.avi")
        self._video_writer = cv2.VideoWriter(cam_path, fourcc, fps, (w, h))

        # 检测结果录制 (同尺寸)
        det_path = os.path.join(self._record_dir, f"detection_{timestamp}.avi")
        self._video_writer_det = cv2.VideoWriter(det_path, fourcc, fps, (w, h))

        self._recording = True
        self._record_start_time = time.time()

        # 录制时长定时器 (每秒更新)
        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._update_record_status)
        self._record_timer.start(1000)

        # 按钮变为红色录制状态
        self.btn_record.setText("\u23fa  REC \u25cf")
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                color: #ff6060;
                border: 1px solid #cc0000;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                padding: 8px 4px;
            }
            QPushButton:hover { background-color: #a00000; }
        """)

        self.statusBar().showMessage(f"录制已开始 ({fps:.0f}fps): {cam_path}")

    def _stop_recording(self):
        """停止视频录制"""
        self._recording = False

        # 停止时长定时器
        if self._record_timer is not None:
            self._record_timer.stop()
            self._record_timer = None

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._video_writer_det is not None:
            self._video_writer_det.release()
            self._video_writer_det = None

        # 计算录制时长
        if self._record_start_time > 0:
            elapsed = time.time() - self._record_start_time
            m, s = divmod(int(elapsed), 60)
            duration_str = f" (时长 {m:02d}:{s:02d})"
        else:
            duration_str = ""

        # 恢复按钮默认样式
        self.btn_record.setText("\u23fa  Record")
        self.btn_record.setStyleSheet(self._big_btn_style())

        self.statusBar().showMessage(f"录制已停止，文件已保存{duration_str}")

    def _update_record_status(self):
        """每秒更新录制时长显示"""
        if not self._recording:
            return
        elapsed = time.time() - self._record_start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        if h > 0:
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            time_str = f"{m:02d}:{s:02d}"
        self.btn_record.setText(f"\u23fa  REC {time_str}")

    # ────────────────────────────────────────────────────────────
    #   ROI 控制
    # ────────────────────────────────────────────────────────────

    def _on_roi_mode_changed(self, btn):
        idx = self.roi_btn_group.id(btn)
        modes = {0: None, 1: 'ae', 2: 'awb'}
        self.camera_view.roi_mode = modes.get(idx)
        mode_names = {0: '关闭', 1: 'AE 曝光区域', 2: 'AWB 白平衡区域'}
        self.statusBar().showMessage(f"ROI 模式: {mode_names.get(idx)} — 在 Camera View 中拖拽绘制")

    def _clear_all_roi(self):
        self.camera_view.ae_roi = None
        self.camera_view.awb_roi = None
        self.camera_view._update_display()
        self.statusBar().showMessage("所有 ROI 已清除")

    def _apply_pending_roi(self):
        """将 ImageLabel 上的 ROI 归一化坐标转换为图像像素坐标，发送给相机"""
        if not self.camera_running:
            return
        w = self.camera.width
        h = self.camera.height

        ae = self.camera_view.ae_roi
        if ae:
            left = int(ae[0] * w)
            top = int(ae[1] * h)
            right = int(ae[2] * w)
            bottom = int(ae[3] * h)
            self.camera.set_ae_roi(left, top, right, bottom)

        awb = self.camera_view.awb_roi
        if awb:
            left = int(awb[0] * w)
            top = int(awb[1] * h)
            right = int(awb[2] * w)
            bottom = int(awb[3] * h)
            self.camera.set_awb_roi(left, top, right, bottom)

    # ────────────────────────────────────────────────────────────
    #   设置持久化
    # ────────────────────────────────────────────────────────────

    def _load_settings(self):
        """从 QSettings 加载用户偏好"""
        s = QSettings("CameraNet", "DetectionGUI")

        # 图像增强
        self.slider_sharpen.setValue(s.value("enhance/sharpen", 0, int))
        self.slider_gamma.setValue(s.value("enhance/gamma", 10, int))
        self.slider_clahe.setValue(s.value("enhance/clahe", 0, int))

        # 字体缩放
        self.slider_font_scale.setValue(s.value("ui/font_scale", 100, int))

        # 模型变体
        variant = s.value("model/variant", "small", str)
        self.combo_model_variant.setCurrentText(variant)
        self.model_variant = variant

        # 推理模式
        mode = s.value("inference/mode", "standard", str)
        self.combo_infer_mode.setCurrentText(mode)
        self._on_infer_mode_changed(mode)

        # 窗口大小和位置
        geom = s.value("window/geometry")
        if geom:
            self.restoreGeometry(geom)

    def _save_settings(self):
        """保存用户偏好到 QSettings"""
        s = QSettings("CameraNet", "DetectionGUI")

        # 图像增强
        s.setValue("enhance/sharpen", self.slider_sharpen.value())
        s.setValue("enhance/gamma", self.slider_gamma.value())
        s.setValue("enhance/clahe", self.slider_clahe.value())

        # 字体缩放
        s.setValue("ui/font_scale", self.slider_font_scale.value())

        # 模型变体
        s.setValue("model/variant", self.model_variant)

        # 推理模式
        s.setValue("inference/mode", self.inference_mode)

        # 窗口大小和位置
        s.setValue("window/geometry", self.saveGeometry())

    # ────────────────────────────────────────────────────────────
    #   窗口关闭
    # ────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._save_settings()  # 保存设置
        self._stop_recording()  # 停止录制
        self._inference_worker.stop()  # 停止推理线程
        self._stop_camera()
        super().closeEvent(event)


# ================================================================
#   入口
# ================================================================

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
