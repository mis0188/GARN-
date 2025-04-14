GARN: Graph Alignment with Redesigned Net for COVID-19 CT Classification
项目概述
GARN (Graph Alignment with Redesigned Net) 是一个专为COVID-19 CT图像分类设计的深度学习框架。它通过结合图卷积网络和对齐机制，解决了不同数据源之间的分布差异问题，从而提高了模型的泛化能力。本项目旨在提供一个高精度的COVID-19诊断工具，能够有效处理来自不同医疗机构的异构数据。
模型架构
GARN模型由以下主要组件组成：

重设计的COVID-Net骨干网络：专为CT图像优化的特征提取器，包含站点特定的批归一化层
数据结构分析器 (DSA)：生成用于构建图邻接矩阵的结构分数
图卷积网络 (GCN)：处理数据结构，实现结构感知对齐
嵌入网络：用于对比学习，实现域不变表示
高级特征融合模块：通过多种注意力机制，智能融合骨干网络和图特征

关键特性
GARN模型实现了三种创新的对齐机制：

结构感知对齐：通过图卷积网络对数据结构进行建模
域对齐：通过对比学习实现域不变表示
类中心对齐：通过对齐不同域的类中心实现语义一致性

其它特性包括：

站点特定的批归一化层，适应多站点数据特性
高级数据增强策略（MixUp、CutMix、随机擦除等）
残差连接和深度注意力机制
多模态特征融合（通道注意力、交叉注意力和残差连接融合）

安装与环境要求
系统要求

Python 3.8+
CUDA 11.0+ (用于GPU加速)
8GB+ RAM

安装步骤

克隆仓库

bashgit clone https://github.com/username/garn-covid-ct.git
cd garn-covid-ct

创建并激活虚拟环境（可选）

bashpython -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

安装依赖

bashpip install -r requirements.txt
依赖库
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
数据集
本项目使用两个公开的COVID-19 CT分类数据集：

SARS-CoV-2 CT数据集：包含来自120名患者的2482张CT图像
COVID-CT数据集：包含来自216名COVID-19患者的349张CT图像和来自171名非COVID-19患者的397张CT图像

数据集应组织为以下结构：
datasets/
├── SARS-COV-2/
│   ├── COVID/
│   └── non-COVID/
└── COVID-CT/
    ├── Data-split/
    │   ├── COVID/
    │   └── NonCOVID/
    └── Images-processed/
        ├── CT_COVID/
        └── CT_NonCOVID/
使用方法
训练模型
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
参数说明：

--covid_ct_dir: COVID-CT数据集目录
--sars_cov2_dir: SARS-CoV-2数据集目录
--batch_size: 批大小，默认为16
--epochs: 训练周期数，默认为100
--lr: 学习率，默认为1e-3
--alpha: 对比损失权重
--beta: 三元组损失权重
--gamma: 中心对齐损失权重
--dropout_rate: Dropout比率
--use_mixup: 启用MixUp数据增强
--use_cutmix: 启用CutMix数据增强
--save_dir: 保存结果的目录

测试与可视化
bashpython test.py \
    --site_a_model_path results/run_YYYYMMDD_HHMMSS/checkpoints/garn_best_site_a.pth \
    --site_b_model_path results/run_YYYYMMDD_HHMMSS/checkpoints/garn_best_site_b.pth \
    --val_model_path results/run_YYYYMMDD_HHMMSS/checkpoints/garn_best_val.pth \
    --covid_ct_dir datasets/COVID-CT \
    --sars_cov2_dir datasets/SARS-COV-2 \
    --output_dir test_results
测试脚本将产生以下可视化结果：

嵌入空间可视化
混淆矩阵
ROC曲线
图结构可视化
类中心可视化

结果与性能
在两个数据集上的性能（以AUC为指标）：

SARS-CoV-2数据集：97.39% AUC
COVID-CT数据集：83.94% AUC

与基准方法的对比：
方法SARS-CoV-2COVID-CTCOVID-Net (单站点)84.08%71.09%简单联合学习74.78%68.12%GARN (我们的方法)97.39%83.94%
项目结构
garn-covid-ct/
├── models/
│   ├── __init__.py
│   ├── garn.py              # GARN模型实现
│   ├── covid_net.py         # 重设计的COVID-Net骨干网络
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
可视化示例
训练过程中会生成多种可视化结果，包括：

嵌入空间可视化：展示不同站点和类别样本在特征空间中的分布
图结构可视化：展示数据样本间的结构关系
类中心可视化：展示不同站点和类别样本的中心点分布
混淆矩阵：展示模型分类性能
ROC曲线：展示模型分类的敏感性和特异性

引用
如果您在研究中使用了本项目，请引用我们的论文：
@inproceedings{author2023garn,
  title={Graph Alignment with Redesigned Net for COVID-19 CT Classification},
  author={Author, A. and Author, B.},
  booktitle={Proceedings of ACM Multimedia},
  year={2023}
}
致谢
我们感谢SARS-CoV-2 CT和COVID-CT数据集的作者提供的宝贵数据资源。本项目的实现借鉴了多种图神经网络和对比学习的优秀工作。
许可证
本项目采用MIT许可证。详见LICENSE文件。
联系方式
如有任何问题，请联系：email@example.com