"""
styles — 深色主题样式定义
"""


def big_btn_style(font_size=16):
    """全局大按钮样式"""
    return f"""
        QPushButton {{
            background-color: #1e1e36;
            color: #e0e0e0;
            border: 1px solid #3a3a4e;
            border-radius: 6px;
            font-size: {font_size}px;
            font-weight: bold;
            padding: 8px 4px;
        }}
        QPushButton:hover {{ background-color: #2a2a50; border-color: #5a5a7e; }}
        QPushButton:pressed {{ background-color: #3a3a60; }}
        QPushButton:disabled {{ background-color: #12122a; color: #555; }}
    """


def active_btn_style(bg_color="#1a5c2a", fg_color="#a0ffa0", border_color="#27ae60"):
    """激活状态按钮样式 (绿色 / 红色)"""
    return f"""
        QPushButton {{
            background-color: {bg_color};
            color: {fg_color};
            border: 1px solid {border_color};
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            padding: 8px 4px;
        }}
        QPushButton:hover {{ background-color: {bg_color}; }}
    """


REC_STYLE = """
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
"""


DARK_THEME = """
    QMainWindow {
        background-color: #0a0a14;
    }
    QWidget {
        color: #e0e0e0;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        font-size: 14px;
    }
    QPushButton {
        background-color: #2d2d44;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 5px;
        padding: 6px 14px;
        font-weight: bold;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #3a3a5c;
        border-color: #666;
    }
    QPushButton:pressed {
        background-color: #4a4a6a;
    }
    QPushButton:disabled {
        background-color: #1a1a2a;
        color: #555;
    }
    QSlider::groove:horizontal {
        border: 1px solid #444;
        height: 5px;
        background: #2a2a3a;
        border-radius: 2px;
    }
    QSlider::sub-page:horizontal {
        background: #4a6a4a;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #50c050;
        border: 1px solid #40a040;
        width: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }
    QSlider::handle:horizontal:hover {
        background: #60d060;
    }
    QCheckBox {
        spacing: 5px;
        color: #ddd;
        font-size: 11px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
    }
    QSpinBox {
        background-color: #1e1e30;
        border: 1px solid #3a3a4a;
        border-radius: 3px;
        padding: 1px 3px;
        color: #e0e0e0;
        font-size: 11px;
    }
    QComboBox {
        background-color: #1e1e30;
        border: 1px solid #3a3a4a;
        border-radius: 3px;
        padding: 3px 6px;
        color: #e0e0e0;
        font-size: 11px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #1e1e30;
        color: #e0e0e0;
        selection-background-color: #3a3a5c;
    }
    QStatusBar {
        background-color: #12121f;
        color: #8ae68a;
        font-size: 11px;
        border-top: 1px solid #2a2a3e;
    }
    QRadioButton {
        font-size: 11px;
    }
"""

ADV_DIALOG_STYLE = """
    QDialog {
        background-color: #1a1a2e;
        color: #e0e0e0;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        font-size: 13px;
    }
    QGroupBox {
        font-weight: bold;
        color: #bbb;
        border: 1px solid #3a3a4a;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 16px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: #ccc;
    }
    QLabel {
        color: #ccc;
        font-size: 12px;
    }
    QSlider::groove:horizontal {
        border: 1px solid #444;
        height: 6px;
        background: #2a2a3a;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #6c6ce0;
        border: 1px solid #5a5ad0;
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: #8080f0;
    }
    QCheckBox {
        spacing: 6px;
        color: #ddd;
        font-size: 12px;
    }
    QSpinBox {
        background-color: #2a2a3a;
        border: 1px solid #444;
        border-radius: 3px;
        padding: 2px 4px;
        color: #e0e0e0;
        font-size: 12px;
    }
    QPushButton {
        background-color: #2d2d44;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 6px 20px;
        font-weight: bold;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #3a3a5c;
    }
"""

GROUP_STYLE = """
    QGroupBox {
        font-weight: bold;
        color: #bbb;
        border: 1px solid #3a3a4a;
        border-radius: 5px;
        margin-top: 8px;
        padding-top: 14px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }
"""
