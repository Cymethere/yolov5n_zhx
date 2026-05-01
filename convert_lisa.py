import os
import pandas as pd
import shutil
from collections import defaultdict

# ======================
# 路径（你改这里）
# ======================
LISA_PATH = r"D:\Paper_yolo\datasets_raw\lisa"
OUTPUT_PATH = r"D:\Paper_yolo\datasets_raw\lisa_yolo"

CSV_PATH = os.path.join(LISA_PATH, "Annotations", "Annotations")

# 类别映射
def map_class(name):
    if "stop" in name:
        return 1  # red
    elif "go" in name:
        return 0  # green
    elif "warning" in name:
        return 2  # yellow
    else:
        return None

# 创建目录
for split in ["train"]:
    os.makedirs(os.path.join(OUTPUT_PATH, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "labels", split), exist_ok=True)

# ======================
# 找所有 CSV
# ======================
csv_files = []
for root, _, files in os.walk(CSV_PATH):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))

print(f"找到 {len(csv_files)} 个 CSV 文件")

# ======================
# 找图片路径（建立索引）
# ======================
image_index = {}

for root, _, files in os.walk(LISA_PATH):
    for f in files:
        if f.endswith(".jpg"):
            image_index[f] = os.path.join(root, f)

print(f"找到 {len(image_index)} 张图片")

# ======================
# 处理 CSV
# ======================
label_dict = defaultdict(list)

for csv_file in csv_files:
    df = pd.read_csv(csv_file, sep=";")

    for _, row in df.iterrows():
        filename = os.path.basename(row["Filename"])
        cls = map_class(row["Annotation tag"])

        if cls is None:
            continue

        xmin = row["Upper left corner X"]
        ymin = row["Upper left corner Y"]
        xmax = row["Lower right corner X"]
        ymax = row["Lower right corner Y"]

        img_path = image_index.get(filename)
        if img_path is None:
            continue

        # 获取尺寸（用PIL）
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size

        # YOLO格式
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        label_dict[filename].append(f"{cls} {x_center} {y_center} {bw} {bh}")

# ======================
# 保存
# ======================
for filename, labels in label_dict.items():
    img_path = image_index[filename]

    # 保存图片
    new_img = os.path.join(OUTPUT_PATH, "images/train", filename)
    shutil.copy(img_path, new_img)

    # 保存标签
    txt_name = filename.replace(".jpg", ".txt")
    txt_path = os.path.join(OUTPUT_PATH, "labels/train", txt_name)

    with open(txt_path, "w") as f:
        f.write("\n".join(labels))

print("✅ 转换完成！")