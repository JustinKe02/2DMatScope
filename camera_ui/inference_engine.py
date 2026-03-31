"""
InferenceWorker — 后台推理线程，避免阻塞 UI
"""

import time as _time
import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PyQt5.QtCore import pyqtSignal, QThread, QMutex


class InferenceWorker(QThread):
    """后台推理线程 — 避免阻塞 UI"""
    result_ready = pyqtSignal(object, object, object)  # (frame, mask, conf_map)
    error_occurred = pyqtSignal(str)  # 推理错误消息

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = None
        self._mutex = QMutex()
        self._running = True
        self._has_work = False
        # 这些将由 MainWindow 设置
        self.model = None
        self.device = None
        self.crop_size = 512
        self.stride = 384
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.class_names = []

    def submit_frame(self, frame):
        """提交帧进行推理（如果上一帧还在推理，替换帧）"""
        self._mutex.lock()
        self._frame = frame.copy()
        self._has_work = True
        self._mutex.unlock()

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        while self._running:
            # 检查是否有工作
            self._mutex.lock()
            has_work = self._has_work
            frame = self._frame
            self._has_work = False
            self._mutex.unlock()

            if not has_work or frame is None or self.model is None:
                _time.sleep(0.01)  # 空闲时少量休眠
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = TF.normalize(
                    TF.to_tensor(rgb),
                    self.img_mean, self.img_std
                )
                pred_mask, conf_map = self._sliding_window_predict(img_tensor)
                # 发射原始结果，让 UI 线程做叠加和控件更新
                self.result_ready.emit(frame, pred_mask, conf_map)
            except Exception as e:
                # 限制错误报告频率（每 5 秒最多一次）
                now = _time.time()
                if now - getattr(self, '_last_error_time', 0) > 5.0:
                    self._last_error_time = now
                    self.error_occurred.emit(f"推理错误: {e}")

    def _sliding_window_predict(self, img_tensor):
        """全覆盖滑动窗口推理"""
        _, H, W = img_tensor.shape
        num_classes = len(self.class_names)
        device = self.device
        crop_size = self.crop_size
        stride = self.stride

        pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
        count = torch.zeros(H, W, dtype=torch.float32, device=device)

        pad_h = max(0, crop_size - H)
        pad_w = max(0, crop_size - W)
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
        _, pH, pW = img_tensor.shape

        ys = sorted(set(
            list(range(0, max(1, pH - crop_size + 1), stride)) +
            [max(0, pH - crop_size)]
        ))
        xs = sorted(set(
            list(range(0, max(1, pW - crop_size + 1), stride)) +
            [max(0, pW - crop_size)]
        ))

        for y in ys:
            for x in xs:
                crop = img_tensor[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = self.model(crop)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs = F.softmax(logits, dim=1)[0]
                y_end = min(y + crop_size, H)
                x_end = min(x + crop_size, W)
                pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
                count[y:y_end, x:x_end] += 1

        count = count.clamp(min=1)
        avg_probs = pred_sum / count.unsqueeze(0)
        conf_map = avg_probs.max(dim=0)[0].cpu().numpy()
        mask = avg_probs.argmax(dim=0).cpu().numpy()
        return mask, conf_map


def overlay_mask(frame, mask, conf_map, class_names, class_colors_np,
                  visible_classes=None, alpha=0.35):
    """将分割 mask 叠加到原图上，绘制 YOLO 风格边界框 + 置信度

    Args:
        visible_classes: 要显示的类别 ID 集合 (例如 {1} 只显示 Monolayer)。
                         None 表示显示所有非背景类别。
    Returns:
        (overlay_img, detection_counts) — 叠加后的图像和检测统计
    """
    # 确定可见类别
    if visible_classes is None:
        visible_classes = set(range(1, len(class_names)))

    h_img, w_img = frame.shape[:2]
    color_mask = np.zeros_like(frame)
    for cid in visible_classes:
        if cid == 0 or cid >= len(class_colors_np):
            continue
        region = mask == cid
        color_mask[region] = class_colors_np[cid][::-1]  # RGB → BGR

    overlay = frame.copy()
    # 只对可见类别的像素做混合
    fg = np.isin(mask, list(visible_classes))
    overlay[fg] = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)[fg]

    # 为每个可见类别找连通区域并绘制 bounding box
    min_area = 500
    detection_counts = {}

    for cid in sorted(visible_classes):
        binary = (mask == cid).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        color_rgb = class_colors_np[cid]
        color_bgr = tuple(int(c) for c in color_rgb[::-1])
        bright_bgr = tuple(min(255, int(c * 1.3)) for c in color_bgr)
        class_name = class_names[cid]
        count = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            count += 1

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]

            # 计算该区域的平均置信度
            if conf_map is not None:
                region_mask = labels == i
                avg_conf = float(conf_map[region_mask].mean()) * 100
            else:
                avg_conf = 0.0

            # 绘制边框 (双线效果)
            cv2.rectangle(overlay, (x - 1, y - 1), (x + w + 1, y + hh + 1), (0, 0, 0), 3)
            cv2.rectangle(overlay, (x, y), (x + w, y + hh), bright_bgr, 2)

            # 标签文本
            label = f"{class_name} {avg_conf:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 4

            if y - th - pad * 2 - 2 > 0:
                ly = y - pad * 2 - th - 2
            else:
                ly = y + 2

            # 标签背景
            sub = overlay[ly:ly + th + pad * 2, x:x + tw + pad * 2]
            if sub.size > 0:
                bg = np.full_like(sub, color_bgr, dtype=np.uint8)
                blended = cv2.addWeighted(sub, 0.3, bg, 0.7, 0)
                overlay[ly:ly + th + pad * 2, x:x + tw + pad * 2] = blended

            # 文字
            text_y = ly + th + pad
            cv2.putText(overlay, label, (x + pad + 1, text_y + 1),
                        font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(overlay, label, (x + pad, text_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if count > 0:
            detection_counts[class_name] = count

    # 顶部汇总栏
    if detection_counts:
        summary_parts = [f"{name}: {cnt}" for name, cnt in detection_counts.items()]
        summary_text = "  |  ".join(summary_parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(summary_text, font, 0.6, 1)
        bar_h = th + 16
        bar = overlay[0:bar_h, :]
        overlay[0:bar_h, :] = cv2.addWeighted(bar, 0.4, np.zeros_like(bar), 0.6, 0)
        x_pos = 12
        for name, cnt in detection_counts.items():
            cid = class_names.index(name)
            c_bgr = tuple(int(c) for c in class_colors_np[cid][::-1])
            text = f" {name}: {cnt}"
            cv2.circle(overlay, (x_pos + 6, bar_h - 10), 5, c_bgr, -1)
            cv2.putText(overlay, text, (x_pos + 14, bar_h - 6),
                        font, 0.55, c_bgr, 1, cv2.LINE_AA)
            (pw, _), _ = cv2.getTextSize(text, font, 0.55, 1)
            x_pos += pw + 30

    return overlay, detection_counts
