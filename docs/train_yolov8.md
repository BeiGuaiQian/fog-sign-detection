# 如何训练交通路标检测模型

本文档说明如何为“雾天路标检测系统”训练 YOLOv8 交通路标检测模型，并将训练好的权重接入项目。

## 1. 推荐数据集

可选数据集：

- TT100K：中国交通标志数据集，类别较多，适合较完整的交通标志检测实验。
- GTSDB：German Traffic Sign Detection Benchmark，经典交通标志检测数据集。
- Roboflow 上的 traffic sign detection YOLOv8 格式数据集。

课程项目建议优先使用 Roboflow 下载 YOLOv8 格式数据集，因为它通常已经整理好图片、标签和 `data.yaml`，最省时间。

## 2. 标准 YOLO 数据集目录结构

建议将数据集放在项目根目录外或项目根目录下的 `datasets/traffic-sign/`：

```text
datasets/traffic-sign/
├─ train/
│  ├─ images/
│  └─ labels/
├─ valid/
│  ├─ images/
│  └─ labels/
├─ test/
│  ├─ images/
│  └─ labels/
└─ data.yaml
```

其中：

- `images/` 存放图片。
- `labels/` 存放 YOLO 格式标注文件。
- 每张图片对应一个同名 `.txt` 标签文件。

## 3. data.yaml 示例

```yaml
path: datasets/traffic-sign
train: train/images
val: valid/images
test: test/images
names:
  0: speed_limit
  1: stop
  2: warning
  3: traffic_light
  4: other_sign
```

如果你使用 Roboflow 下载数据集，通常会自动生成 `data.yaml`。训练前只需要检查路径和类别名称是否正确。

## 4. 训练命令

GPU 训练：

```powershell
yolo detect train model=yolov8n.pt data=datasets/traffic-sign/data.yaml epochs=50 imgsz=640 batch=8 device=0
```

CPU 训练：

```powershell
yolo detect train model=yolov8n.pt data=datasets/traffic-sign/data.yaml epochs=30 imgsz=640 batch=4 device=cpu
```

说明：

- `model=yolov8n.pt` 使用 YOLOv8 nano 作为初始模型，速度快，适合课程项目。
- `epochs` 可以根据数据量和机器性能调整。
- `imgsz=640` 是常用输入尺寸。
- `batch` 太大时可能显存不足，可以降低。

## 5. 预测命令

训练完成并复制权重后，可用以下命令测试预测：

```powershell
yolo detect predict model=weights/best.pt source=data/samples conf=0.25 save=True
```

预测结果通常会保存在 `runs/detect/predict/` 目录下。

## 6. 复制权重

训练完成后，将最佳权重复制到项目权重目录：

```powershell
Copy-Item runs/detect/train/weights/best.pt weights/best.pt
```

项目中的 Streamlit 系统和命令行脚本会优先加载：

```text
weights/best.pt
```

## 7. 注意事项

- 如果没有训练权重，系统会使用 `yolov8n.pt` 临时模型，但它不是专门检测路标的。
- 最终报告和演示最好使用自己训练或下载的交通路标权重。
- 如果数据集类别很多，可以先只保留常见路标类别，降低训练难度。
- 如果只是课程演示，可以优先保证常见类别如限速、停止、警告、红绿灯等检测效果稳定。
- 数据集图片质量和标注质量会明显影响检测效果，训练前建议抽查标签是否准确。
