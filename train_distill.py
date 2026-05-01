import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.models.yolo.detect import DetectionTrainer

class DistillTrainer(DetectionTrainer):
    def __init__(self, teacher_model_path, distill_weight=0.5, temperature=3.0, **kwargs):
        self.teacher_model_path = teacher_model_path
        self.distill_weight = distill_weight
        self.temperature = temperature
        super().__init__(**kwargs)

    def setup_model(self):
        super().setup_model()
        # 加载 Teacher 模型
        self.teacher = YOLO(self.teacher_model_path).model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.to(self.device)

    def loss(self, batch, preds=None):
        loss, loss_items = super().loss(batch, preds)
        if preds is None:
            preds = self.model(batch['img'])
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
        distill_loss = 0.0
        num_heads = len(preds)
        for i in range(num_heads):
            student_cls = preds[i][..., 5:]
            teacher_cls = teacher_preds[i][..., 5:]
            student_logits = student_cls / self.temperature
            teacher_logits = teacher_cls / self.temperature
            kl = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            )
            distill_loss += kl
        distill_loss = distill_loss / num_heads
        total_loss = loss + self.distill_weight * distill_loss
        return total_loss, loss_items

def main():
    print("=" * 50)
    print("🚀 知识蒸馏训练 (Student: YOLOv5n+CBAM, Teacher: YOLOv5l)")
    print("=" * 50)
    SETTINGS["tensorboard"] = True

    student_weights = 'runs/CBAM/CBAM_Exp/weights/best.pt'   # 请确认路径正确
    teacher_weights = 'runs/teacher/Teacher_yolov5l/weights/best.pt'

    overrides = {
        'model': student_weights,   # 关键：指定模型路径
        'resume': True,
        'data': '../datasets_small/data.yaml',
        'project': 'runs/Distill',
        'name': 'CBAM_distill',
        'device': 0,
        'workers': 2,
        'epochs': 100,
        'batch': 8,
        'imgsz': 640,
        'cache': 'disk',
        'amp': True,
        'pretrained': False,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'cos_lr': True,
        'warmup_epochs': 5.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'patience': 50,
        'close_mosaic': 20,
        'save_period': 1,
        'plots': True,
    }

    trainer = DistillTrainer(
        teacher_model_path=teacher_weights,
        distill_weight=0.5,
        temperature=3.0,
        overrides=overrides,
        _callbacks={}
    )
    trainer.train()

    print("\n✨ 蒸馏训练完成！")
    print("模型保存路径：", trainer.save_dir)

if __name__ == '__main__':
    main()