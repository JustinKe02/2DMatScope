"""
验证脚本 — 测试 RepELA-Net 模型导入和推理链
不需要相机，仅验证模型加载和 forward pass
"""
import sys
import os
import torch
import torch.nn.functional as F

# 添加 RepELA-Net 到路径
script_dir = r'f:\Code\Windows_train\CameraNet_v2'
repela_dir = os.path.join(script_dir, 'RepELA-Net')
if repela_dir not in sys.path:
    sys.path.insert(0, repela_dir)

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base, infer_use_cse

print("=" * 60)
print("RepELA-Net 模型导入验证")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. 测试模型构建 (use_cse=False)
print("\n[1] 构建 RepELA-Net-Small (use_cse=False)...")
model = repela_net_small(num_classes=4, use_cse=False).to(device)
model.eval()
print(f"    OK 模型已创建 (参数: {sum(p.numel() for p in model.parameters()):,})")

# 2. 测试 forward pass
print("\n[2] Forward pass (512x512)...")
dummy = torch.randn(1, 3, 512, 512).to(device)
with torch.no_grad():
    out = model(dummy)
if isinstance(out, tuple):
    print(f"    返回 tuple, logits shape: {out[0].shape}")
else:
    print(f"    返回 tensor, shape: {out.shape}")
    assert out.shape == (1, 4, 512, 512), f"输出形状错误: {out.shape}"
    print(f"    OK 输出形状正确: {out.shape}")

# 3. 测试 use_cse=True
print("\n[3] 构建 RepELA-Net-Small (use_cse=True)...")
model_cse = repela_net_small(num_classes=4, use_cse=True).to(device)
model_cse.eval()
with torch.no_grad():
    out_cse = model_cse(dummy)
logits = out_cse[0] if isinstance(out_cse, tuple) else out_cse
assert logits.shape == (1, 4, 512, 512)
print(f"    OK CSE 模型输出正确: {logits.shape}")

# 4. 测试 deep supervision tuple 处理
print("\n[4] Deep supervision tuple 处理...")
model_ds = repela_net_small(num_classes=4, deep_supervision=True).to(device)
model_ds.train()
with torch.no_grad():
    out_ds = model_ds(dummy)
if isinstance(out_ds, tuple):
    print(f"    OK Train 返回 tuple: logits={out_ds[0].shape}, aux={[a.shape for a in out_ds[1]]}")
model_ds.eval()
with torch.no_grad():
    out_ds_eval = model_ds(dummy)
logits = out_ds_eval[0] if isinstance(out_ds_eval, tuple) else out_ds_eval
probs = F.softmax(logits, dim=1)[0]
print(f"    OK Softmax probs shape: {probs.shape}")

# 5. 测试 infer_use_cse
print("\n[5] 测试 infer_use_cse()...")
assert infer_use_cse({'model': {}, 'epoch': 10}) == False
print(f"    OK 旧 checkpoint -> False")
assert infer_use_cse({'model': {'color_enhance.s_weight': torch.tensor(1.0)}}) == True
print(f"    OK CSE checkpoint -> True")
assert infer_use_cse({'model': {}, 'args': {'use_cse': True}}) == True
print(f"    OK 新 checkpoint (args.use_cse=True) -> True")

# 6. 检查现有权重
print("\n[6] 检查可用权重文件...")
for name in ['best_model.pth', 'deploy_model.pth']:
    path = os.path.join(script_dir, name)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            use_cse = infer_use_cse(ckpt) if 'model' in ckpt else False
            print(f"    {name}: keys={list(ckpt.keys())[:5]}, use_cse={use_cse}")
        else:
            print(f"    {name}: 原始 state_dict")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
