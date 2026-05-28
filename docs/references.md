# 参考资料整理

本文档整理“基于图像去雾与目标定位的雾天路标检测系统”可引用的论文、文档和数据集资料。

## 1. 图像去雾理论

### Single Image Haze Removal Using Dark Channel Prior, CVPR 2009

- 名称：Kaiming He, Jian Sun, Xiaoou Tang, *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009。
- 作用：提出暗通道先验 DCP，是经典单幅图像去雾方法。
- 在本项目中怎么用：作为 DCP 去雾模块的理论基础，用于暗通道计算、大气光估计、透射率估计和场景辐射恢复。
- 报告中适合放在哪一节：相关理论、实现方法、参考文献。

### Guided Image Filtering, ECCV 2010

- 名称：Kaiming He, Jian Sun, Xiaoou Tang, *Guided Image Filtering*, ECCV 2010。
- 作用：提出引导滤波方法，可用于边缘保持平滑。
- 在本项目中怎么用：用于优化 DCP 得到的初始透射率图，减少块效应并保持边缘结构。
- 报告中适合放在哪一节：相关理论、实现方法。

## 2. 深度学习去雾扩展

### AOD-Net: All-in-One Dehazing Network

- 名称：AOD-Net: All-in-One Dehazing Network。
- 作用：将大气散射模型中的多个中间估计整合到端到端网络中，实现学习型图像去雾。
- 在本项目中怎么用：作为后续改进方向，与当前 DCP 传统物理模型方法进行对比。
- 报告中适合放在哪一节：总结与展望。

### DehazeNet

- 名称：DehazeNet。
- 作用：较早使用卷积神经网络估计透射率，用于单幅图像去雾。
- 在本项目中怎么用：可作为深度学习去雾方法的代表，说明 DCP 之后去雾算法的发展方向。
- 报告中适合放在哪一节：相关理论、总结与展望。

### DehazeFormer

- 名称：DehazeFormer。
- 作用：基于 Transformer 结构的图像去雾方法，具有较强的全局建模能力。
- 在本项目中怎么用：作为未来优化方向，用于替换或对比 DCP，提升复杂雾天场景的恢复效果。
- 报告中适合放在哪一节：总结与展望。

## 3. 目标检测

### YOLO

- 名称：YOLO: You Only Look Once。
- 作用：经典单阶段目标检测框架，直接预测目标类别和边界框。
- 在本项目中怎么用：作为 YOLOv8 路标检测模块的理论来源，说明单阶段检测适合实时视觉系统。
- 报告中适合放在哪一节：相关理论、实现方法。

### YOLOv8

- 名称：YOLOv8。
- 作用：Ultralytics 提供的 YOLO 系列模型版本，具有较好的速度和精度平衡。
- 在本项目中怎么用：用于对原始雾图和 DCP 去雾图分别进行路标检测，输出检测框、类别和置信度。
- 报告中适合放在哪一节：系统设计、实现方法、实验结果与分析。

### Ultralytics 文档

- 名称：Ultralytics YOLO Documentation。
- 作用：提供 YOLOv8 安装、训练、预测和模型导出说明。
- 在本项目中怎么用：用于训练 `weights/best.pt`、运行预测命令、理解模型参数如 `conf`、`iou`、`device`。
- 报告中适合放在哪一节：实现方法、实验设计。

## 4. 数据集

### TT100K

- 名称：Tsinghua-Tencent 100K Traffic Sign Dataset。
- 作用：大规模中国交通标志数据集，包含多种道路场景和交通标志类别。
- 在本项目中怎么用：可用于训练或评估交通路标检测模型，适合中文交通场景。
- 报告中适合放在哪一节：实验设计、参考文献。

### GTSDB

- 名称：German Traffic Sign Detection Benchmark。
- 作用：经典交通标志检测数据集，常用于交通标志检测算法实验。
- 在本项目中怎么用：可作为路标检测模型训练或对比测试数据来源。
- 报告中适合放在哪一节：实验设计、参考文献。

### Roboflow traffic sign detection datasets

- 名称：Roboflow 上的 traffic sign detection 数据集。
- 作用：提供多种已整理好的交通标志检测数据集，通常支持 YOLOv8 格式导出。
- 在本项目中怎么用：推荐优先使用 Roboflow 下载 YOLOv8 格式数据，减少数据清洗和格式转换工作。
- 报告中适合放在哪一节：实验设计、模型训练说明。
