import random
import shutil
from pathlib import Path

import cv2

# ===========================
# 🔧 配置区
# ===========================
DATASET_ROOT = r"../datasets_final"
OUTPUT_ROOT = r"../datasets_small"

KEEP_RATIO = 0.4  # 保留比例
MIN_BOXES = 1  # 至少目标数
IMG_SIZE = 512  # 统一尺寸
SEED = 42  # 保证可复现


# ===========================
# 🚀 工具函数：计算质量分数
# ===========================
def score_image(label_path):
    """简单质量评分： - 目标数量（越多越高）.
    """
    with open(label_path) as f:
        lines = f.readlines()

    num_boxes = len(lines)

    if num_boxes < MIN_BOXES:
        return -1

    return num_boxes  # 可以后续扩展（如面积/清晰度）


# ===========================
# 🚀 主逻辑
# ===========================
def process_split(split):
    print(f"\n📂 处理 {split} 集合...")

    img_dir = Path(DATASET_ROOT) / "images" / split
    label_dir = Path(DATASET_ROOT) / "labels" / split

    out_img_dir = Path(OUTPUT_ROOT) / "images" / split
    out_label_dir = Path(OUTPUT_ROOT) / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    print(f"原始数量: {len(images)}")

    # ===========================
    # Step 1: 过滤无标签 / 低质量
    # ===========================
    valid_samples = []

    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            continue

        score = score_image(label_path)

        if score > 0:
            valid_samples.append((img_path, score))

    print(f"有效样本: {len(valid_samples)}")

    # ===========================
    # Step 2: 按质量排序（核心优化）
    # ===========================
    valid_samples.sort(key=lambda x: x[1], reverse=True)

    # ===========================
    # Step 3: 再做随机扰动（避免过拟合）
    # ===========================
    top_k = int(len(valid_samples) * KEEP_RATIO)
    selected = valid_samples[:top_k]

    random.seed(SEED)
    random.shuffle(selected)

    # ===========================
    # Step 4: 写入新数据集
    # ===========================
    kept = 0

    for img_path, _ in selected:
        label_path = label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # resize（统一尺寸）
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # 保存图片
        cv2.imwrite(str(out_img_dir / img_path.name), img)

        # 保存标签（YOLO格式无需改）
        shutil.copy(label_path, out_label_dir / label_path.name)

        kept += 1

    print(f"✅ 保留数量: {kept}")


def main():
    print("=" * 50)
    print("🚀 开始数据集精简（优化版）")
    print("=" * 50)

    random.seed(SEED)

    process_split("train")
    process_split("val")

    print("\n✨ 完成！输出路径：", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
