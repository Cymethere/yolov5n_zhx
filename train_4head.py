from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    print("=" * 50)
    print("🚀 YOLOv5 四检测头模型训练（P2 + P3 + P4 + P5）")
    print("=" * 50)
    SETTINGS["tensorboard"] = True

    # 加载四头模型配置文件
    model = YOLO('models/yolov5n-p2.yaml')

    # 训练参数（与 CBAM 实验保持一致，便于后续对比）
    results = model.train(
        data='../datasets_small/data.yaml',
        project=r'D:\Paper_yolo\yolov5-master\runs\4head',
        name='exp',
        device=0,
        workers=2,
        epochs=100,
        batch=8,
        imgsz=640,
        cache='disk',
        amp=True,
        pretrained=True,
        optimizer='SGD',
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
        plots=True
    )

    print("\n✨ 训练完成！")
    print("模型保存路径：", results.save_dir)

if __name__ == '__main__':
    main()