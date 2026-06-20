import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from models.common import CBAMModule

# ==================================================
# 关键：将自定义模块注册到 Ultralytics 的解析器中
# ==================================================
tasks.CBAMModule = CBAMModule  # 注意类名必须与 yaml 中一致


def main():
    print("=" * 50)
    print("🚀 YOLOv5 CBAM 模型训练（直接加载自定义 yaml 方式）")
    print("=" * 50)

    SETTINGS["tensorboard"] = True
    print("🔧 正在配置训练参数...")

    # 直接加载自定义的 yaml 文件
    model = YOLO("models/yolov5n_cbam.yaml")

    # 训练参数（与之前保持一致）
    results = model.train(
        data="../datasets_small/data.yaml",
        project=r"D:\Paper_yolo\yolov5-master\runs\CBAM1",
        name="CBAM1_Exp",  # 换个实验名，与之前的区分
        device=0,
        workers=2,
        epochs=100,
        resume=False,
        batch=8,
        cache="disk",
        imgsz=640,
        amp=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        patience=50,
        close_mosaic=20,
        save_period=1,
        plots=True,
    )

    print("\n✨ 训练完成！")
    print("模型保存路径：", results.save_dir)


if __name__ == "__main__":
    main()
