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
    
    GARN结合了重新设计的COVID-Net骨干网络和图对齐机制，
    以解决不同数据源之间的分布差异问题。
    
    论文提出的三种对齐机制：
    1. 结构感知对齐：通过图卷积网络对数据结构进行建模
    2. 域对齐：通过对比学习实现域不变表示
    3. 类中心对齐：通过对齐不同域的类中心实现语义一致性
    """
    # models/garn.py
    
    def __init__(
        self, 
        n_classes: int = 2, 
        sites: int = 2,
        backbone_embedding_dim: int = 400,  # 修改为400（原为424）
        dsa_hidden_dim: int = 256,
        dsa_output_dim: int = 128,
        gcn_hidden_dim: int = 256,
        gcn_output_dim: int = 400,  # 修改为400（原为424）
        embedding_hidden_dim: int = 1024,
        embedding_output_dim: int = 128
    ):
        """初始化GARN模型"""
        super(GARN, self).__init__()
        
        # 打印初始化参数，帮助调试
        print(f"GARN初始化 - backbone_embedding_dim: {backbone_embedding_dim}")
        
        # 骨干网络
        self.backbone = RedesignedCOVIDNet(n_classes=n_classes, site_specific_bn=True, sites=sites)
        
        # 数据结构分析器 (DSA)
        self.dsa = DataStructureAnalyzer(
            input_dim=backbone_embedding_dim,  # 确保这里是400
            hidden_dim=dsa_hidden_dim,
            output_dim=dsa_output_dim
        )
        
        # 图卷积网络 (GCN)
        self.gcn = GraphConvolutionalNetwork(
            in_features=backbone_embedding_dim,  # 确保这里是400
            hidden_features=gcn_hidden_dim,
            out_features=gcn_output_dim  # 确保这里是400
        )
        
        # 对比学习的嵌入网络
        self.embedding_network = EmbeddingNetwork(
            input_dim=backbone_embedding_dim,  # 确保这里是400
            hidden_dim=embedding_hidden_dim,
            output_dim=embedding_output_dim
        )
        
        # GCN特征的分类器
        self.gcn_classifier = nn.Linear(gcn_output_dim, n_classes)  # 确保这里是400
        
        # 融合特征的分类器
        self.fusion_classifier = nn.Linear(gcn_output_dim, n_classes)  # 确保这里是400
            
    def forward(self, x: torch.Tensor, site_id: int = 0) -> Dict[str, torch.Tensor]:
        """
        GARN模型的前向传播
        
        参数:
            x: 输入图像张量，形状为 [batch_size, channels, height, width]
            site_id: 站点标识符，用于站点特定的批归一化
            
        返回:
            包含logits、embeddings和其他特征的字典
        """
        # 从骨干网络获取特征
        # 输入: [batch_size, 3, height, width]
        # 输出: 包含"logits"、"embeddings"、"penultimate"的字典
        backbone_outputs = self.backbone(x, site_id)
        
        # 提取嵌入
        # 形状: [batch_size, backbone_embedding_dim]
        embeddings = backbone_outputs["embeddings"]
        
        # 生成结构分数
        # 输入: [batch_size, backbone_embedding_dim]
        # 输出: [batch_size, dsa_output_dim]
        structure_scores = self.dsa(embeddings)
        
        # 创建邻接矩阵 - 基于结构相似性
        batch_size = embeddings.size(0)
        # 输入: [batch_size, dsa_output_dim]
        # 输出: [batch_size, batch_size]
        adj = torch.matmul(structure_scores, structure_scores.transpose(0, 1))
        
        # 添加自环到邻接矩阵
        # [batch_size, batch_size] + [batch_size, batch_size]
        adj = adj + torch.eye(batch_size, device=x.device)
        
        # 重塑GCN的输入
        # [batch_size, backbone_embedding_dim] -> [1, batch_size, backbone_embedding_dim]
        embeddings_reshaped = embeddings.unsqueeze(0)
        # [batch_size, batch_size] -> [1, batch_size, batch_size]
        adj_reshaped = adj.unsqueeze(0)
        
        # 应用GCN - 实现结构感知对齐
        # 输入: [1, batch_size, backbone_embedding_dim], [1, batch_size, batch_size]
        # 输出: [1, batch_size, gcn_output_dim]
        gcn_features = self.gcn(embeddings_reshaped, adj_reshaped).squeeze(0)
        
        # 获取对比嵌入 - 用于域对齐
        # 输入: [batch_size, backbone_embedding_dim]
        # 输出: [batch_size, embedding_output_dim]
        contrastive_embeddings = self.embedding_network(embeddings)
        
        # 获取GCN特征的logits
        # 输入: [batch_size, gcn_output_dim]
        # 输出: [batch_size, n_classes]
        gcn_logits = self.gcn_classifier(gcn_features)
        
        # 特征融合（骨干特征和GCN特征的融合）
        # 简单的加法融合，也可以使用其他方式如注意力机制
        # [batch_size, backbone_embedding_dim] + [batch_size, gcn_output_dim]
        fused_features = embeddings + gcn_features
        # 输入: [batch_size, gcn_output_dim]
        # 输出: [batch_size, n_classes]
        fusion_logits = self.fusion_classifier(fused_features)
        
        # 更新输出字典
        backbone_outputs.update({
            "structure_scores": structure_scores,
            "gcn_features": gcn_features,
            "contrastive_embeddings": contrastive_embeddings,
            "gcn_logits": gcn_logits,
            "fusion_logits": fusion_logits
        })
        
        return backbone_outputs