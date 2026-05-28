# 基于图像去雾与目标定位的雾天路标检测系统

## 项目简介

本系统面向雾天交通场景，主要用于雾天路标图像增强与交通路标目标定位。

系统使用 DCP 暗通道先验进行图像去雾，使用 YOLOv8 进行交通路标检测，并对比去雾前后的检测数量、检测置信度、图像清晰度、边缘数量等指标。项目提供 Streamlit 可视化页面，也支持命令行单图测试和批量实验。

## 系统流程

```text
雾天路标图像
→ DCP 图像去雾
→ 原图/去雾图 YOLOv8 检测
→ 检测框与置信度输出
→ 图像质量和检测效果对比
```

## 环境安装

建议使用 Python 3.10。

```powershell
conda create -n fogsign python=3.10 -y
conda activate fogsign
pip install -r requirements.txt
```

## 运行 Streamlit 系统

```powershell
streamlit run app.py
```

打开浏览器后上传雾天路标图片，即可查看去雾结果、暗通道图、透射率图、YOLOv8 检测结果、检测框表格和指标对比表。

## 单独测试去雾

```powershell
python scripts/test_dehaze.py --input data/samples/test.jpg --output data/output
```

输出文件：

- `data/output/dehazed.jpg`
- `data/output/dark_channel.jpg`
- `data/output/transmission.jpg`

## 单独测试检测

```powershell
python scripts/test_detect.py --input data/samples/test.jpg --weights weights/best.pt --output data/output
```

输出文件：

- `data/output/annotated.jpg`
- `data/output/detections.csv`

## 批量实验

```powershell
python scripts/batch_experiment.py --input-dir data/samples --weights weights/best.pt --output-dir data/output/batch
```

输出内容：

- `data/output/batch/dehazed/`
- `data/output/batch/detect_original/`
- `data/output/batch/detect_dehazed/`
- `data/output/batch/summary.csv`
- `data/output/batch/summary_mean.csv`

## 模型权重说明

将训练好的 YOLOv8 交通路标检测权重放到：

```text
weights/best.pt
```

如果没有 `weights/best.pt`，系统会使用 `yolov8n.pt` 临时模型。该模型是通用 COCO 检测模型，不是专门检测交通路标的，因此最终报告和演示建议使用自己训练或下载的交通路标权重。

训练说明见：

```text
docs/train_yolov8.md
```

## 项目结构

```text
fog-sign-detection/
├─ app.py                         # Streamlit 可视化系统入口
├─ requirements.txt               # 项目依赖
├─ README.md                      # 项目说明
├─ src/
│  ├─ __init__.py
│  ├─ dehaze_dcp.py               # DCP 暗通道先验去雾
│  ├─ detect_yolo.py              # YOLOv8 检测封装
│  ├─ metrics.py                  # 图像质量和检测效果指标
│  └─ utils.py                    # 图像读写、格式转换等工具函数
├─ weights/
│  └─ .gitkeep                    # 模型权重目录，best.pt 放在这里
├─ data/
│  ├─ input/                      # 输入图片目录
│  ├─ output/                     # 输出结果目录
│  └─ samples/                    # 示例图片目录
├─ scripts/
│  ├─ test_dehaze.py              # 单图 DCP 去雾测试
│  ├─ test_detect.py              # 单图 YOLOv8 检测测试
│  └─ batch_experiment.py         # 批量对比实验
└─ docs/
   ├─ report_outline.md           # 课程报告提纲
   ├─ references.md               # 参考资料
   └─ train_yolov8.md             # YOLOv8 路标检测模型训练说明
```

## 报告可写的创新点

- 结合物理模型的图像去雾与深度学习目标检测。
- 对比去雾前后路标检测效果。
- 使用多种无参考图像质量指标评价增强效果。
- 构建可交互的完整图像处理系统。

## 适用场景

本项目适合作为数字图像处理课程大作业，能够覆盖图像增强、目标定位、指标评价和可视化系统实现等内容。
