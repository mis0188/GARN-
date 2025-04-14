<div align="center">

# 🔬 GARN: Graph Alignment with Redesigned Net for COVID-19 CT Classification

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/username/garn-covid-ct?style=social)](https://github.com/username/garn-covid-ct)

<p align="center">
  <img src="assets/garn_logo.png" alt="GARN Logo" width="300"/>
  <br>
  <i>使用图对齐技术和重设计网络提高COVID-19 CT图像分类精度</i>
</p>

[📋 概述](#项目概述) •
[🏗️ 架构](#模型架构) •
[⚙️ 安装](#安装与环境要求) •
[📊 数据集](#数据集) •
[▶️ 使用方法](#使用方法) •
[📈 结果](#结果与性能) •
[📚 引用](#引用)

</div>

---

## 📋 项目概述

**GARN** (Graph Alignment with Redesigned Net) 是一个基于深度学习的模型，专为 COVID-19 CT 图像分类设计。它采用创新的方法解决了不同数据源之间的分布差异问题，通过结合图卷积网络和对齐机制，提高了模型的泛化能力。本项目旨在提供一个高精度的 COVID-19 诊断工具，能够有效处理来自不同医疗机构的异构数据。

<details>
<summary>💡 核心亮点</summary>

- 解决多站点医学图像数据的域差异问题
- 三种创新的对齐机制提高跨站点泛化能力
- 高级数据增强策略确保模型鲁棒性
- 多模态特征融合提高分类准确性
- 优于现有的单数据源和简单联合学习方法
</details>

---

## 🏗️ 模型架构

<div align="center">
  <img src="assets/model_architecture.png" alt="GARN Model Architecture" width="800"/>
</div>

GARN 模型由以下主要组件组成：

1. **重设计的 COVID-Net 骨干网络** 📊
   - 专为 CT 图像优化的特征提取器
   - 包含站点特定的批归一化层处理多源数据
   - 深度卷积网络结构提取丰富图像特征

2. **数据结构分析器 (DSA)** 🔍
   - 生成用于构建图邻接矩阵的结构分数
   - 分析样本间的内在关系
   - 多层非线性变换网络

3. **图卷积网络 (GCN)** 🌐
   - 基于样本间关系进行信息传播
   - 实现结构感知对齐
   - 多层图卷积操作

4. **嵌入网络** 🧬
   - 用于对比学习，实现域不变表示
   - 投影高维特征到低维语义空间
   - 促进相同类别样本聚集

5. **高级特征融合模块** 🔄
   - 通道注意力机制
   - 交叉注意力机制
   - 残差连接融合
   - 自适应组合多种融合策略

### ✨ 关键特性

GARN 模型实现了三种创新的对齐机制：

<table>
  <tr>
    <th width="33%">结构感知对齐</th>
    <th width="33%">域对齐</th>
    <th width="33%">类中心对齐</th>
  </tr>
  <tr>
    <td>通过图卷积网络对数据结构进行建模，挖掘样本间的内在关系</td>
    <td>通过对比学习实现域不变表示，减少不同数据源的分布差异</td>
    <td>通过对齐不同域的类中心实现语义一致性，确保相同类别在特征空间中接近</td>
  </tr>
</table>

其它特性包括：
- 站点特定的批归一化层
- 高级数据增强策略 (MixUp, CutMix, 随机擦除等)
- 残差连接和深度注意力机制
- 多模态特征融合

---

## ⚙️ 安装与环境要求

### 系统要求

| 组件 | 最低要求 |
|------|---------|
| Python | 3.8+ |
| CUDA | 11.0+ (用于 GPU 加速) |
| RAM | 8GB+ |
| 存储空间 | 10GB+ (含数据集) |

### 📥 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/mis0188/GARN-covid-ct.git  
cd garn-covid-ct

创建并激活虚拟环境（可选）

bash# 创建环境
python -m venv venv

# 激活环境 (Linux/Mac)
source venv/bin/activate

# 激活环境 (Windows)
venv\Scripts\activate

安装依赖

bashpip install -r requirements.txt
📦 依赖库
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
pillow>=8.0.0
tqdm>=4.62.0
matplotlib>=3.4.0
networkx>=2.6.0
seaborn>=0.11.0

📊 数据集
本项目使用两个公开的 COVID-19 CT 分类数据集：
<div align="center">
<table>
  <tr>
    <th width="50%"><img src="assets/sars_cov2_sample.png" width="100%"><br>SARS-CoV-2 CT 数据集</th>
    <th width="50%"><img src="assets/covid_ct_sample.png" width="100%"><br>COVID-CT 数据集</th>
  </tr>
  <tr>
    <td>包含来自 120 名患者的 2482 张 CT 图像</td>
    <td>包含来自 216 名 COVID-19 患者的 349 张 CT 图像和来自 171 名非 COVID-19 患者的 397 张 CT 图像</td>
  </tr>
</table>
</div>
数据集结构
数据集应组织为以下结构：
datasets/
├── SARS-COV-2/
│   ├── COVID/               # COVID-19 阳性样本
│   └── non-COVID/           # COVID-19 阴性样本
└── COVID-CT/
    ├── Data-split/
    │   ├── COVID/           # 训练/验证/测试集分割文件
    │   └── NonCOVID/        # 训练/验证/测试集分割文件
    └── Images-processed/
        ├── CT_COVID/        # COVID-19 阳性样本
        └── CT_NonCOVID/     # COVID-19 阴性样本

▶️ 使用方法
📈 训练模型
bashpython train.py \
    --covid_ct_dir datasets/COVID-CT \
    --sars_cov2_dir datasets/SARS-COV-2 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-3 \
    --alpha 0.1 \
    --beta 0.1 \
    --gamma 0.1 \
    --dropout_rate 0.2 \
    --use_mixup \
    --save_dir results
<details>
<summary>参数说明</summary>
参数描述默认值--covid_ct_dirCOVID-CT 数据集目录---sars_cov2_dirSARS-CoV-2 数据集目录---batch_size批大小16--epochs训练周期数100--lr学习率1e-3--alpha对比损失权重0.1--beta三元组损失权重0.1--gamma中心对齐损失权重0.1--dropout_rateDropout 比率0.2--use_mixup启用 MixUp 数据增强False--use_cutmix启用 CutMix 数据增强False--save_dir保存结果的目录results
</details>
🔍 测试与可视化
bashpython test.py \
    --site_a_model_path results/run_YYYYMMDD_HHMMSS/checkpoints/garn_best_site_a.pth \
    --site_b_model_path results/run_YYYYMMDD_HHMMSS/checkpoints/garn_best_site_b.pth \
    --val_model_path results/run_YYYYMMDD_HHMMSS/checkpoints/garn_best_val.pth \
    --covid_ct_dir datasets/COVID-CT \
    --sars_cov2_dir datasets/SARS-COV-2 \
    --output_dir test_results
测试脚本将产生以下可视化结果：
<div align="center">
<table>
  <tr>
    <td><img src="assets/embeddings_viz.png" width="100%"><br><b>嵌入空间可视化</b></td>
    <td><img src="assets/confusion_matrix.png" width="100%"><br><b>混淆矩阵</b></td>
  </tr>
  <tr>
    <td><img src="assets/roc_curve.png" width="100%"><br><b>ROC 曲线</b></td>
    <td><img src="assets/graph_viz.png" width="100%"><br><b>图结构可视化</b></td>
  </tr>
</table>
</div>

📈 结果与性能
在两个数据集上的性能（以 AUC 为指标）：

SARS-CoV-2 数据集：97.39% AUC
COVID-CT 数据集：83.94% AUC

与基准方法的对比
<div align="center">
<table>
  <tr>
    <th rowspan="2">方法</th>
    <th colspan="2">AUC (%)</th>
  </tr>
  <tr>
    <th>SARS-CoV-2</th>
    <th>COVID-CT</th>
  </tr>
  <tr>
    <td>COVID-Net (单站点)</td>
    <td>84.08</td>
    <td>71.09</td>
  </tr>
  <tr>
    <td>简单联合学习</td>
    <td>74.78</td>
    <td>68.12</td>
  </tr>
  <tr>
    <td><b>GARN (我们的方法)</b></td>
    <td><b>97.39</b></td>
    <td><b>83.94</b></td>
  </tr>
</table>
</div>
<div align="center">
  <img src="assets/performance_comparison.png" alt="Performance Comparison" width="600"/>
</div>

📁 项目结构
garn-covid-ct/
├── models/
│   ├── __init__.py
│   ├── garn.py              # GARN 模型实现
│   ├── covid_net.py         # 重设计的 COVID-Net 骨干网络
│   └── graph_modules.py     # 图卷积模块实现
├── utils/
│   ├── __init__.py
│   └── losses.py            # 损失函数实现
├── datasets/
│   ├── __init__.py
│   └── covid_dataset.py     # 数据集加载与预处理
├── train.py                 # 模型训练脚本
├── test.py                  # 测试与可视化脚本
├── requirements.txt         # 项目依赖
└── README.md                # 本文档

📚 引用
如果您在研究中使用了本项目，请引用我们的论文：
bibtex@inproceedings{author2023garn,
  title={Graph Alignment with Redesigned Net for COVID-19 CT Classification},
  author={Author, A. and Author, B.},
  booktitle={Proceedings of ACM Multimedia},
  year={2023}
}

🙏 致谢
我们感谢 SARS-CoV-2 CT 和 COVID-CT 数据集的作者提供的宝贵数据资源。本项目的实现借鉴了多种图神经网络和对比学习的优秀工作。

📄 许可证
本项目采用 MIT 许可证。详见 LICENSE 文件。

📧 联系方式
如有任何问题，请联系：email@example.com

<div align="center">
  <sub>Built with ❤️ by Research Team</sub>
</div>
```
此 Markdown 格式的 README 包含了多种美化元素：

居中的标题和徽章
分区导航链接
表情符号增强可读性
详细的表格和图片展示
可折叠的细节部分
清晰的代码块和格式
整洁划分的章节
引人注目的表格和图形
色彩和排版的视觉层次

这些元素共同创建了一个专业、美观且信息丰富的项目文档