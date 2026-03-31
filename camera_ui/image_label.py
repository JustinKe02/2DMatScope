"""
ImageLabel — 保持纵横比的图像显示标签，支持鼠标绘制/移动/缩放 ROI
"""

import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen


class ImageLabel(QLabel):
    """保持纵横比的图像显示标签，支持鼠标绘制 ROI"""

    # 信号：ROI 绘制完成，参数为归一化坐标 (x1, y1, x2, y2) 范围 [0,1]
    roi_drawn = pyqtSignal(float, float, float, float)

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 4px;
            }
        """)
        self._pixmap = None
        self._base_pixmap = None   # 原始缩放后的 pixmap（不含 ROI 叠加）
        self.setText(title)

        # ROI 绘制状态
        self.roi_mode = None       # None / 'ae' / 'awb'
        self._drawing = False      # 正在绘制新 ROI
        self._moving = False       # 正在移动现有 ROI
        self._moving_which = None  # 'ae' or 'awb'
        self._resizing = False     # 正在缩放现有 ROI
        self._resize_which = None  # 'ae' or 'awb'
        self._resize_handle = None # 手柄: 'tl','t','tr','r','br','b','bl','l'
        self._start_pos = None
        self._end_pos = None
        self._move_offset = None

        # 已确认的 ROI（归一化坐标）
        self.ae_roi = None         # (x1, y1, x2, y2) normalized
        self.awb_roi = None

        # 图像在 label 中的实际显示区域
        self._img_rect = QRect()
        self._img_orig_size = (0, 0)

        # 手柄检测半径 (像素)
        self._handle_radius = 8

        self.setMouseTracking(True)

    def set_image(self, img_bgr):
        """接收 BGR numpy 数组并显示"""
        if img_bgr is None:
            return
        h, w, ch = img_bgr.shape
        self._img_orig_size = (w, h)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = w * ch
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())  # .copy() 确保数据独立
        self._update_display()

    def _update_display(self):
        if self._pixmap is None:
            return
        label_size = self.size()
        # KeepAspectRatioByExpanding: 等比缩放并铺满整个显示区域，
        # 超出部分会被裁剪（不会变形，但可能裁掉边缘少许内容）
        # Qt.KeepAspectRatio 等比例缩放
        scaled = self._pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._base_pixmap = scaled

        # 计算图像在 label 中的居中位置（超出部分向两侧均匀裁剪）
        x = (label_size.width() - scaled.width()) // 2
        y = (label_size.height() - scaled.height()) // 2
        self._img_rect = QRect(x, y, scaled.width(), scaled.height())

        # 在 pixmap 上叠加 ROI 矩形
        display = QPixmap(label_size)
        display.fill(QColor(26, 26, 46))  # 背景色 #1a1a2e
        painter = QPainter(display)
        painter.drawPixmap(self._img_rect, scaled)

        # 绘制已保存的 ROI
        self._draw_roi_rect(painter, self.ae_roi, QColor(0, 220, 0), "AE")
        self._draw_roi_rect(painter, self.awb_roi, QColor(255, 60, 60), "AWB")

        # 绘制正在拖拽的新 ROI
        if self._drawing and self._start_pos and self._end_pos:
            color = QColor(0, 220, 0) if self.roi_mode == 'ae' else QColor(255, 60, 60)
            pen = QPen(color, 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self._start_pos, self._end_pos).normalized()
            painter.drawRect(rect)

        painter.end()
        super().setPixmap(display)

    def _draw_roi_rect(self, painter, roi_norm, color, label_text):
        """在 painter 上绘制一个 ROI 矩形 (归一化坐标 → 显示像素坐标)"""
        if roi_norm is None:
            return
        x1, y1, x2, y2 = roi_norm
        ir = self._img_rect
        px1 = ir.x() + int(x1 * ir.width())
        py1 = ir.y() + int(y1 * ir.height())
        px2 = ir.x() + int(x2 * ir.width())
        py2 = ir.y() + int(y2 * ir.height())

        pen = QPen(color, 2, Qt.SolidLine)
        painter.setPen(pen)
        rect = QRect(QPoint(px1, py1), QPoint(px2, py2)).normalized()
        painter.drawRect(rect)

        # 8 个手柄
        handle_size = 6
        handles = self._get_handle_points_from_rect(rect)
        painter.setBrush(color)
        for pt in handles.values():
            painter.drawRect(pt.x() - handle_size // 2, pt.y() - handle_size // 2,
                             handle_size, handle_size)
        painter.setBrush(Qt.NoBrush)

        # 标签文字
        painter.setFont(QFont('Segoe UI', 9, QFont.Bold))
        painter.drawText(rect.left() + 4, rect.top() - 4, label_text)

    def _get_handle_points_from_rect(self, rect):
        """QRect → dict of handle_name → QPoint"""
        return {
            'tl': rect.topLeft(),
            't':  QPoint(rect.center().x(), rect.top()),
            'tr': rect.topRight(),
            'r':  QPoint(rect.right(), rect.center().y()),
            'br': rect.bottomRight(),
            'b':  QPoint(rect.center().x(), rect.bottom()),
            'bl': rect.bottomLeft(),
            'l':  QPoint(rect.left(), rect.center().y()),
        }

    def _roi_to_display_rect(self, roi_norm):
        """将归一化 ROI 坐标转换为 widget 上的 QRect"""
        if roi_norm is None:
            return None
        x1, y1, x2, y2 = roi_norm
        ir = self._img_rect
        px1 = ir.x() + int(x1 * ir.width())
        py1 = ir.y() + int(y1 * ir.height())
        px2 = ir.x() + int(x2 * ir.width())
        py2 = ir.y() + int(y2 * ir.height())
        return QRect(QPoint(px1, py1), QPoint(px2, py2)).normalized()

    def _hit_test_handle(self, pos):
        """检测鼠标是否在某个 ROI 手柄上，返回 (which, handle_name) 或 (None, None)"""
        r = self._handle_radius
        for which, roi in [('ae', self.ae_roi), ('awb', self.awb_roi)]:
            if roi is None:
                continue
            display_rect = self._roi_to_display_rect(roi)
            if display_rect is None:
                continue
            handles = self._get_handle_points_from_rect(display_rect)
            for name, pt in handles.items():
                if abs(pos.x() - pt.x()) <= r and abs(pos.y() - pt.y()) <= r:
                    return (which, name)
        return (None, None)

    def _hit_test_roi(self, pos):
        """检测鼠标位置是否在某个已有 ROI 内部，返回 'ae' / 'awb' / None"""
        for which, roi in [('ae', self.ae_roi), ('awb', self.awb_roi)]:
            if roi is None:
                continue
            display_rect = self._roi_to_display_rect(roi)
            if display_rect and display_rect.contains(pos):
                return which
        return None

    def _handle_cursor(self, handle_name):
        """手柄名 → 光标样式"""
        cursors = {
            'tl': Qt.SizeFDiagCursor, 'br': Qt.SizeFDiagCursor,
            'tr': Qt.SizeBDiagCursor, 'bl': Qt.SizeBDiagCursor,
            't':  Qt.SizeVerCursor,   'b':  Qt.SizeVerCursor,
            'l':  Qt.SizeHorCursor,   'r':  Qt.SizeHorCursor,
        }
        return cursors.get(handle_name, Qt.ArrowCursor)

    def _widget_to_normalized(self, pos):
        """将 widget 坐标转换为归一化 [0,1] 图像坐标"""
        ir = self._img_rect
        if ir.width() == 0 or ir.height() == 0:
            return None
        nx = (pos.x() - ir.x()) / ir.width()
        ny = (pos.y() - ir.y()) / ir.height()
        return (max(0, min(1, nx)), max(0, min(1, ny)))

    # —— 鼠标事件 ——
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._img_rect.contains(event.pos()):
            # 优先级: 手柄缩放 → 内部移动 → 绘制新 ROI

            # 1) 检测手柄
            which_h, handle = self._hit_test_handle(event.pos())
            if which_h and handle:
                self._resizing = True
                self._resize_which = which_h
                self._resize_handle = handle
                self._drawing = False
                self._moving = False
                super().mousePressEvent(event)
                return

            # 2) 检测 ROI 内部 (移动)
            hit = self._hit_test_roi(event.pos())
            if hit:
                self._moving = True
                self._moving_which = hit
                self._drawing = False
                self._resizing = False
                roi = self.ae_roi if hit == 'ae' else self.awb_roi
                display_rect = self._roi_to_display_rect(roi)
                self._move_offset = event.pos() - display_rect.topLeft()
                self._start_pos = event.pos()
                super().mousePressEvent(event)
                return

            # 3) 绘制新 ROI
            if self.roi_mode:
                self._drawing = True
                self._moving = False
                self._resizing = False
                self._start_pos = event.pos()
                self._end_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing and self._resize_which and self._resize_handle:
            # —— 缩放模式 ——
            roi = self.ae_roi if self._resize_which == 'ae' else self.awb_roi
            if roi:
                n = self._widget_to_normalized(event.pos())
                if n:
                    nx, ny = n
                    x1, y1, x2, y2 = roi
                    h = self._resize_handle

                    # 根据手柄调整对应边
                    if h == 'tl':    x1, y1 = nx, ny
                    elif h == 't':   y1 = ny
                    elif h == 'tr':  x2, y1 = nx, ny
                    elif h == 'r':   x2 = nx
                    elif h == 'br':  x2, y2 = nx, ny
                    elif h == 'b':   y2 = ny
                    elif h == 'bl':  x1, y2 = nx, ny
                    elif h == 'l':   x1 = nx

                    # 限制范围 [0, 1]
                    x1, y1 = max(0, min(1, x1)), max(0, min(1, y1))
                    x2, y2 = max(0, min(1, x2)), max(0, min(1, y2))

                    # 确保最小尺寸并归一化方向
                    new_roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    if self._resize_which == 'ae':
                        self.ae_roi = new_roi
                    else:
                        self.awb_roi = new_roi
                    self._update_display()

        elif self._moving and self._moving_which:
            # —— 移动模式 ——
            roi = self.ae_roi if self._moving_which == 'ae' else self.awb_roi
            if roi:
                x1, y1, x2, y2 = roi
                roi_w, roi_h = x2 - x1, y2 - y1
                ir = self._img_rect
                if ir.width() > 0 and ir.height() > 0:
                    new_left = event.pos() - self._move_offset
                    nx1 = (new_left.x() - ir.x()) / ir.width()
                    ny1 = (new_left.y() - ir.y()) / ir.height()
                    nx1 = max(0, min(1 - roi_w, nx1))
                    ny1 = max(0, min(1 - roi_h, ny1))
                    if self._moving_which == 'ae':
                        self.ae_roi = (nx1, ny1, nx1 + roi_w, ny1 + roi_h)
                    else:
                        self.awb_roi = (nx1, ny1, nx1 + roi_w, ny1 + roi_h)
                    self._update_display()

        elif self._drawing:
            self._end_pos = event.pos()
            self._update_display()

        # —— 更新光标 ——
        if not (self._resizing or self._moving or self._drawing):
            which_h, handle = self._hit_test_handle(event.pos())
            if which_h and handle:
                self.setCursor(self._handle_cursor(handle))
            elif self._hit_test_roi(event.pos()):
                self.setCursor(Qt.SizeAllCursor)
            elif self.roi_mode and self._img_rect.contains(event.pos()):
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._resizing:
                self._resizing = False
                roi = self.ae_roi if self._resize_which == 'ae' else self.awb_roi
                if roi:
                    self.roi_drawn.emit(*roi)
                self._resize_which = None
                self._resize_handle = None
                self._update_display()

            elif self._moving:
                self._moving = False
                roi = self.ae_roi if self._moving_which == 'ae' else self.awb_roi
                if roi:
                    self.roi_drawn.emit(*roi)
                self._moving_which = None
                self._update_display()

            elif self._drawing:
                self._drawing = False
                self._end_pos = event.pos()
                n1 = self._widget_to_normalized(self._start_pos)
                n2 = self._widget_to_normalized(self._end_pos)
                if n1 and n2:
                    x1 = min(n1[0], n2[0])
                    y1 = min(n1[1], n2[1])
                    x2 = max(n1[0], n2[0])
                    y2 = max(n1[1], n2[1])
                    if (x2 - x1) > 0.01 and (y2 - y1) > 0.01:
                        if self.roi_mode == 'ae':
                            self.ae_roi = (x1, y1, x2, y2)
                        elif self.roi_mode == 'awb':
                            self.awb_roi = (x1, y1, x2, y2)
                        self.roi_drawn.emit(x1, y1, x2, y2)
                self._update_display()
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)
