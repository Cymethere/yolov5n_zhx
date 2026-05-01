from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from models.common import CBAMModule  # 确保 common.py 中定义了 CBAMModule


def replace_layer_with_cbam(model, target_index):
    """将模型中的第 target_index 层替换为 CBAMModule 自动获取该层的输入通道数.
    """
    # 获取内部的 nn.Sequential 列表
    net = model.model.model  # Ultralytics 模型的结构

    if target_index >= len(net):
        raise IndexError(f"索引 {target_index} 超出范围，模型共 {len(net)} 层")

    original_layer = net[target_index]
    # 尝试获取输入通道数（针对 Conv 或 C3 等常见层）
    if hasattr(original_layer, "conv") and hasattr(original_layer.conv, "in_channels"):
        in_channels = original_layer.conv.in_channels
    elif hasattr(original_layer, "cv1") and hasattr(original_layer.cv1, "conv"):
        in_channels = original_layer.cv1.conv.in_channels
    else:
        # 如果无法自动获取，请根据打印信息手动指定
        print(f"无法自动获取第 {target_index} 层的输入通道数，请手动修改代码。")
        print(f"该层类型: {type(original_layer)}")
        raise ValueError("需要手动指定通道数")

    print(f"将第 {target_index} 层（{type(original_layer).__name__}，输入通道 {in_channels}）替换为 CBAMModule")
    net[target_index] = CBAMModule(in_channels)
    return model


def main():
    print("=" * 50)
    print("🚀 YOLOv5 CBAM 模型训练流程（代码替换版）")
    print("=" * 50)

    # 开启 TensorBoard
    SETTINGS["tensorboard"] = True
    print("🔧 正在配置训练参数...")

    # 1. 加载原始模型（不使用自定义 yaml）
    model = YOLO(
        r"D:\Paper_yolo\yolov5-master\runs\CBAM\CBAM_Exp\weights\last.pt"
    )  # model = YOLO('yolov5n.yaml')  # 原始模型配置文件

    # 2. 打印模型结构，帮助选择要替换的层
    print("\n模型各层类型索引：")
    net = model.model.model
    for i, layer in enumerate(net):
        print(f"{i:3d}: {layer.__class__.__name__}")

    # 3. 选择要替换的层索引（根据你的设计，建议在 P3 特征提取后）
    #    例如：根据你之前的 yaml，想要在 C3 后插入 CBAM，对应索引通常是 5（需要根据打印结果调整）
    #    这里让用户手动输入，或者直接写固定值
    target_index = 5  # 请根据上面的打印结果修改这个数字！
    print(f"\n准备将第 {target_index} 层替换为 CBAMModule...")

    # 4. 执行替换
    model = replace_layer_with_cbam(model, target_index)

    # 5. 训练参数配置（与原脚本一致）
    results = model.train(
        data="../datasets_small/data.yaml",
        project=r"D:\Paper_yolo\yolov5-master\runs\CBAM",
        name="CBAM_Exp",
        device=0,
        workers=2,
        epochs=100,
        resume=True,  # 第一次训练必须为 False
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
