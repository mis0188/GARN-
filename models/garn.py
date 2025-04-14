# models/garn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from models.covid_net import RedesignedCOVIDNet
from models.graph_modules import (
    DataStructureAnalyzer, 
    GraphConvolutionalNetwork, 
    EmbeddingNetwork
)

class GARN(nn.Module):
    """图对齐与重设计网络 (GARN) 模型
    
    论文提出的三种对齐机制：
    1. 结构感知对齐：通过图卷积网络对数据结构进行建模
    2. 域对齐：通过对比学习实现域不变表示
    3. 类中心对齐：通过对齐不同域的类中心实现语义一致性
    """
    
    def __init__(
        self, 
        n_classes: int = 2, 
        sites: int = 2,
        backbone_embedding_dim: int = 400,
        dsa_hidden_dim: int = 256,
        dsa_output_dim: int = 128,
        gcn_hidden_dim: int = 256,
        gcn_output_dim: int = 400,
        embedding_hidden_dim: int = 1024,
        embedding_output_dim: int = 128,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4
    ):
        """初始化GARN模型"""
        super(GARN, self).__init__()
        
        # 骨干网络
        self.backbone = RedesignedCOVIDNet(n_classes=n_classes, site_specific_bn=True, sites=sites)
        
        # 数据结构分析器 (DSA)
        self.dsa = DataStructureAnalyzer(
            input_dim=backbone_embedding_dim,
            hidden_dim=dsa_hidden_dim,
            output_dim=dsa_output_dim,
            dropout=dropout_rate
        )
        
        # 图卷积网络 (GCN)
        self.gcn = GraphConvolutionalNetwork(
            in_features=backbone_embedding_dim,
            hidden_features=gcn_hidden_dim,
            out_features=gcn_output_dim,
            dropout=dropout_rate,
            residual=True  # 启用残差连接
        )
        
        # 对比学习的嵌入网络
        self.embedding_network = EmbeddingNetwork(
            input_dim=backbone_embedding_dim,
            hidden_dim=embedding_hidden_dim,
            output_dim=embedding_output_dim,
            dropout=dropout_rate
        )
        
        # GCN特征的分类器
        self.gcn_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(gcn_output_dim, n_classes)
        )
        
        # 融合特征的分类器
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(gcn_output_dim, n_classes)
        )
        
        # ===== 改进：特征融合模块 =====
        # 1. 通道注意力融合
        self.channel_attention = nn.Sequential(
            nn.Linear(backbone_embedding_dim, backbone_embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(backbone_embedding_dim // 16, backbone_embedding_dim),
            nn.Sigmoid()
        )
        
        # 2. 交叉注意力融合
        self.cross_attention = nn.Sequential(
            nn.Linear(backbone_embedding_dim * 2, backbone_embedding_dim),
            nn.LayerNorm(backbone_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_embedding_dim, backbone_embedding_dim),
            nn.Sigmoid()
        )
        
        # 3. 融合后的特征转换
        self.fusion_transform = nn.Sequential(
            nn.Linear(backbone_embedding_dim, backbone_embedding_dim),
            nn.LayerNorm(backbone_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 4. 学习特征融合的门控参数
        self.fusion_gate = nn.Parameter(torch.zeros(1))
        
        # 应用权重衰减
        self.weight_decay = weight_decay
        
        # 初始化权重
        self._initialize_weights()
            
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, site_id: int = 0) -> Dict[str, torch.Tensor]:
        """GARN模型的前向传播"""
        # 从骨干网络获取特征
        backbone_outputs = self.backbone(x, site_id)
        
        # 提取嵌入
        embeddings = backbone_outputs["embeddings"]
        
        # 生成结构分数
        structure_scores = self.dsa(embeddings)
        
        # 创建邻接矩阵 - 基于结构相似性
        batch_size = embeddings.size(0)
        adj = torch.matmul(structure_scores, structure_scores.transpose(0, 1))
        
        # 添加自环到邻接矩阵
        adj = adj + torch.eye(batch_size, device=x.device)
        
        # 重塑GCN的输入
        embeddings_reshaped = embeddings.unsqueeze(0)
        adj_reshaped = adj.unsqueeze(0)
        
        # 应用GCN - 实现结构感知对齐
        gcn_features = self.gcn(embeddings_reshaped, adj_reshaped).squeeze(0)
        
        # 获取对比嵌入 - 用于域对齐
        contrastive_embeddings = self.embedding_network(embeddings)
        
        # 获取GCN特征的logits
        gcn_logits = self.gcn_classifier(gcn_features)
        
        # ===== 改进：高级特征融合 =====
        # 1. 通道注意力 - 学习每个通道的重要性
        channel_weights = self.channel_attention(embeddings + gcn_features)
        
        
        fused_features = channel_weights * embeddings + (1 - channel_weights) * gcn_features
        

        
        # 使用融合特征进行分类
        fusion_logits = self.fusion_classifier(fused_features)
        
        # 更新输出字典
        backbone_outputs.update({
            "structure_scores": structure_scores,
            "gcn_features": gcn_features,
            "contrastive_embeddings": contrastive_embeddings,
            "gcn_logits": gcn_logits,
            "fusion_logits": fusion_logits,
            "channel_weights": channel_weights,
            "adj_matrix": adj  # 添加邻接矩阵用于可视化和调试
        })
        
        return backbone_outputs