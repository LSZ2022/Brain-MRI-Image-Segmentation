# Brain MRI Segmentation Experiment
本项目基于 U-Net 实现脑部 MRI 肿瘤分割，核心研究降采样与上采样模块对分割效果的影响。

## 项目说明
- 基础代码框架来自 **清华深度学习训练营**
- 本人独立完成工作：
  1. 设计并实现 3 种降采样 + 3 种上采样模块对比实验
  2. 完成 6 组独立模型训练，输出真实 `losses_dsc_values.json`
  3. 绘制 Loss/DSC 对比曲线，完成实验分析与报告撰写
  4. 优化模型结构，提出最优模块组合

## 实验结论
最优组合：**PixelUnshuffle（降采样） + PixelShuffle（上采样）**
- 验证集平均 DSC：**0.8287**
- 相比原始 U-Net 提升：**+1.41%**

## 运行说明
1. 安装依赖：`pip install -r requirements.txt`
2. 训练模型：修改 `train.py` 中导入的模型类，运行 `python train.py`
3. 绘制曲线：运行 `plot_downsample.py` / `plot_upsample.py`