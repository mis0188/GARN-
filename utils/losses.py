# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class GARNLoss(nn.Module):
    """GARN模型的损失函数
    
    组合多个损失组件：
    1. 交叉熵损失：用于分类任务
    2. 对比损失：用于域不变表示学习
    3. 三元组损失：用于结构感知对齐
    4. 类中心对齐损失：用于语义对齐
    """
    
    def __init__(
        self, 
        alpha: float = 0.01,  # 对比损失权重 - 论文中的参数
        beta: float = 0.01,   # 三元组损失权重 - 论文中的参数
        gamma: float = 0.01,  # 中心对齐损失权重 - 论文中的参数
        temperature: float = 0.07,  # 对比损失的温度参数 - 论文中的参数
        triplet_margin: float = 0.3  # 三元组损失的边界 - 论文中的参数
    ):
        """
        初始化GARN损失函数
        
        参数:
            alpha: 对比损失权重，默认为0.1
            beta: 三元组损失权重，默认为0.1
            gamma: 中心对齐损失权重，默认为0.1
            temperature: 对比损失的温度参数，默认为0.07
            triplet_margin: 三元组损失的边界，默认为0.3
        """
        super(GARNLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.triplet_margin = triplet_margin
        
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        labels: torch.Tensor,
        site_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        参数:
            outputs: 模型输出字典
            labels: 真实标签张量，形状为 [batch_size]
            site_ids: 站点标识符张量，形状为 [batch_size]
            
        返回:
            损失组件字典
        """
        # 提取相关张量
        logits = outputs["logits"]  # [batch_size, n_classes]
        embeddings = outputs["embeddings"]  # [batch_size, embedding_dim]
        structure_scores = outputs["structure_scores"]  # [batch_size, dsa_output_dim]
        contrastive_embeddings = outputs["contrastive_embeddings"]  # [batch_size, embedding_output_dim]
        
        # 1. 交叉熵损失 - 基本的分类损失
        ce_loss = F.cross_entropy(logits, labels)
        
        # 2. 对比损失 - 用于域对齐
        contrastive_loss = self._compute_contrastive_loss(
            contrastive_embeddings, labels, self.temperature)
        
        # 3. 三元组损失 - 用于结构感知对齐
        triplet_loss = self._compute_triplet_loss(
            structure_scores, labels, self.triplet_margin)
        
        # 4. 类中心对齐损失 - 用于语义对齐
        ca_loss = self._compute_centroid_alignment_loss(
            embeddings, labels, site_ids)
        
        # 总损失 - 加权组合
        total_loss = ce_loss + self.alpha * contrastive_loss + \
                    self.beta * triplet_loss + self.gamma * ca_loss
        
        return {
            "total": total_loss,
            "cross_entropy": ce_loss,
            "contrastive": contrastive_loss,
            "triplet": triplet_loss,
            "centroid_alignment": ca_loss
        }
    
    def _compute_contrastive_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        计算对比损失用于域不变表示
        
        按照论文中描述，对比损失用于拉近相同类别样本的表示，推远不同类别样本的表示，
        从而实现域不变且类别区分的特征表示。
        
        参数:
            features: 特征嵌入张量，形状为 [batch_size, feature_dim]
            labels: 类别标签张量，形状为 [batch_size]
            temperature: 温度参数
            
        返回:
            对比损失张量（标量）
        """
        # 归一化特征
        # [batch_size, feature_dim] -> [batch_size, feature_dim]
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        # [batch_size, feature_dim] x [feature_dim, batch_size] -> [batch_size, batch_size]
        sim_matrix = torch.matmul(features, features.transpose(0, 1)) / temperature
        
        # 创建正样本对掩码（相同类别）
        # [batch_size, 1] == [1, batch_size] -> [batch_size, batch_size]
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # 移除自对比情况
        # [batch_size, batch_size]
        self_mask = torch.eye(features.size(0), device=features.device)
        # [batch_size, batch_size]
        pos_mask = pos_mask - self_mask
        
        # 为数值稳定性，减去每行最大值
        # [batch_size, 1]
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        # [batch_size, batch_size]
        sim_matrix = sim_matrix - logits_max.detach()
        
        # 计算对数概率
        # [batch_size, batch_size]
        exp_sim = torch.exp(sim_matrix)
        # [batch_size, batch_size]
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 计算正样本对的对数似然平均值
        # [batch_size]
        eps = 1e-8
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + eps)
        #mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        sim_matrix = torch.clamp(sim_matrix, -20, 20)
        # 损失：负对数似然
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _compute_triplet_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        margin: float
    ) -> torch.Tensor:
        """
        计算三元组损失用于结构感知对齐
        
        按照论文中描述，三元组损失用于指导数据结构分析器(DSA)网络生成更有判别力的结构分数，
        使得相同类别的样本具有相似的结构特征，不同类别的样本具有不同的结构特征。
        
        参数:
            features: 特征嵌入张量，形状为 [batch_size, feature_dim]
            labels: 类别标签张量，形状为 [batch_size]
            margin: 边界参数
            
        返回:
            三元组损失张量（标量）
        """
        # 计算成对距离
        # [batch_size, feature_dim] -> [batch_size, batch_size]
        dist_matrix = torch.cdist(features, features, p=2)
        
        # 获取正样本和负样本对的掩码
        # [batch_size, 1] == [1, batch_size] -> [batch_size, batch_size]
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # [batch_size, batch_size]
        neg_mask = 1.0 - pos_mask
        
        # 移除自对比情况
        # [batch_size, batch_size]
        self_mask = torch.eye(features.size(0), device=features.device)
        # [batch_size, batch_size]
        pos_mask = pos_mask - self_mask
        
        # 硬正样本挖掘：对每个锚点，找到最难的正样本
        # [batch_size, batch_size]
        pos_dist = dist_matrix * pos_mask
        # 用大值替换零（负样本）
        pos_dist[pos_mask == 0] = float('inf')
        # [batch_size]
        hardest_pos_dist, _ = torch.min(pos_dist, dim=1)
        
        # 硬负样本挖掘：对每个锚点，找到最难的负样本
        # [batch_size, batch_size]
        neg_dist = dist_matrix * neg_mask
        # 用大值替换零（正样本）
        neg_dist[neg_mask == 0] = float('inf')
        # [batch_size]
        hardest_neg_dist, _ = torch.min(neg_dist, dim=1)
        
        # 计算带边界的三元组损失
        # [batch_size]
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
        
        # 返回所有非零三元组损失的平均值
        zeros = torch.zeros_like(loss)
        if torch.all(loss == 0):
            return zeros.sum()  # 如果没有有效的三元组，返回0
        
        return loss.mean()
    
    def _compute_centroid_alignment_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        site_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类中心对齐损失用于语义对齐
        
        按照论文中描述，类中心对齐损失用于拉近不同域中相同类别的中心，
        确保学习到的表示在不同域之间具有语义一致性。
        
        参数:
            features: 特征嵌入张量，形状为 [batch_size, feature_dim]
            labels: 类别标签张量，形状为 [batch_size]
            site_ids: 站点标识符张量，形状为 [batch_size]
            
        返回:
            中心对齐损失张量（标量）
        """
        # 获取唯一类别和站点
        unique_classes = torch.unique(labels)
        unique_sites = torch.unique(site_ids)
        
        if len(unique_sites) < 2:
            # 只有单个站点时不需要对齐
            return torch.tensor(0.0, device=features.device)
        
        loss = torch.tensor(0.0, device=features.device)
        valid_pairs = 0
        
        # 对每个类别，计算每个站点的中心并对齐它们
        for cls in unique_classes:
            site_centroids = []
            
            # 计算每个站点的中心
            for site in unique_sites:
                # 获取当前类别和站点的特征
                mask = (labels == cls) & (site_ids == site)
                if mask.sum() > 0:
                    # 计算当前类别和站点的特征平均值
                    # [N, feature_dim] -> [feature_dim]
                    centroid = features[mask].mean(dim=0)
                    site_centroids.append(centroid)
            
            # 计算中心之间的成对距离
            num_centroids = len(site_centroids)
            if num_centroids < 2:
                continue
            
            for i in range(num_centroids):
                for j in range(i+1, num_centroids):
                    # 计算中心之间的L2距离
                    # [feature_dim], [feature_dim] -> 标量
                    dist = torch.norm(site_centroids[i] - site_centroids[j], p=2)
                    loss += dist
                    valid_pairs += 1
        
        # 如果存在有效对，返回平均距离
        if valid_pairs > 0:
            return loss / valid_pairs
        
        return loss