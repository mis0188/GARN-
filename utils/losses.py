# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class LabelSmoothingCrossEntropy(nn.Module):
    """带标签平滑的交叉熵损失，有助于减少过拟合"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        weight = torch.ones_like(log_prob) * self.smoothing / (pred.size(-1) - 1)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

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
        alpha: float = 0.1,     # 对比损失权重 - 增加权重以加强域对齐
        beta: float = 0.1,      # 三元组损失权重 - 增加权重以加强结构对齐
        gamma: float = 0.1,     # 中心对齐损失权重 - 增加权重以加强语义对齐
        temperature: float = 0.07,  # 对比损失的温度参数
        triplet_margin: float = 0.3,  # 三元组损失的边界
        weight_decay: float = 1e-4  # 权重衰减系数
    ):
        """初始化GARN损失函数"""
        super(GARNLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.triplet_margin = triplet_margin
        self.weight_decay = weight_decay
        
        # 添加标签平滑的交叉熵损失
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        labels: torch.Tensor,
        site_ids: torch.Tensor,
        model: nn.Module = None
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # 提取相关张量
        # 关键点：使用fusion_logits而不是logits
        logits = outputs["logits"]  # 使用融合logits，这是关键!
        gcn_logits = outputs["gcn_logits"]  # GCN的logits
        backbone_logits = outputs["logits"]  # 骨干网络的logits
        embeddings = outputs["embeddings"]  # [batch_size, embedding_dim]
        structure_scores = outputs["structure_scores"]  # [batch_size, dsa_output_dim]
        contrastive_embeddings = outputs["contrastive_embeddings"]  # [batch_size, embedding_output_dim]
        
        # 1. 交叉熵损失 - 使用标签平滑
        ce_loss = self.ce_loss(logits, labels)
        
        # 辅助损失 - 对GCN和骨干网络的输出也进行监督
        gcn_ce_loss = self.ce_loss(gcn_logits, labels)
        backbone_ce_loss = self.ce_loss(backbone_logits, labels)
        
        # 组合所有分类损失
        combined_ce_loss = ce_loss + 0.5 * gcn_ce_loss + 0.3 * backbone_ce_loss
        
        # 2. 对比损失 - 用于域对齐
        contrastive_loss = self._compute_contrastive_loss(
            contrastive_embeddings, labels, self.temperature)
        
        # 3. 三元组损失 - 用于结构感知对齐
        triplet_loss = self._compute_triplet_loss(
            structure_scores, labels, self.triplet_margin)
        
        # 4. 类中心对齐损失 - 用于语义对齐
        ca_loss = self._compute_centroid_alignment_loss(
            embeddings, labels, site_ids)
        
        # 5. 添加L2正则化损失
        l2_reg = torch.tensor(0.0, device=logits.device)
        if model is not None:
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
        
        l2_loss = self.weight_decay * l2_reg
        
        # 总损失 - 加权组合
        total_loss = combined_ce_loss + self.alpha * contrastive_loss + \
                     self.beta * triplet_loss + self.gamma * ca_loss + l2_loss
        
        return {
            "total": total_loss,
            "cross_entropy": combined_ce_loss,
            "contrastive": contrastive_loss,
            "triplet": triplet_loss,
            "centroid_alignment": ca_loss,
            "l2_regularization": l2_loss
        }
    
    def _compute_contrastive_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """计算对比损失用于域不变表示"""
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.transpose(0, 1)) / temperature
        
        # 创建正样本对掩码（相同类别）
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # 移除自对比情况
        self_mask = torch.eye(features.size(0), device=features.device)
        pos_mask = pos_mask - self_mask
        
        # 确保至少有一个正样本对
        if torch.sum(pos_mask) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 为数值稳定性，减去每行最大值
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        
        # 计算对数概率
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 计算正样本对的对数似然平均值
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        
        # 损失：负对数似然
        loss = -mean_log_prob_pos.mean()
        
        # 添加数值稳定性检查
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=features.device)
            
        return loss
    
    def _compute_triplet_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        margin: float
    ) -> torch.Tensor:
        """计算三元组损失用于结构感知对齐"""
        # 计算成对距离
        dist_matrix = torch.cdist(features, features, p=2)
        
        # 获取正样本和负样本对的掩码
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        neg_mask = 1.0 - pos_mask
        
        # 移除自对比情况
        self_mask = torch.eye(features.size(0), device=features.device)
        pos_mask = pos_mask - self_mask
        
        # 确保存在正样本和负样本
        if torch.sum(pos_mask) == 0 or torch.sum(neg_mask) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 硬正样本挖掘：对每个锚点，找到最难的正样本
        pos_dist = dist_matrix * pos_mask
        # 用大值替换零（负样本）
        pos_dist[pos_mask == 0] = float('inf')
        hardest_pos_dist, _ = torch.min(pos_dist, dim=1)
        
        # 硬负样本挖掘：对每个锚点，找到最难的负样本
        neg_dist = dist_matrix * neg_mask
        # 用大值替换零（正样本）
        neg_dist[neg_mask == 0] = float('inf')
        hardest_neg_dist, _ = torch.min(neg_dist, dim=1)
        
        # 计算带边界的三元组损失
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
        
        # 处理无效样本（无正样本或负样本的情况）
        mask = (hardest_pos_dist != float('inf')) & (hardest_neg_dist != float('inf'))
        loss = loss[mask]
        
        # 如果没有有效的三元组，返回0
        if len(loss) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 返回所有非零三元组损失的平均值
        return loss.mean()
    
    def _compute_centroid_alignment_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        site_ids: torch.Tensor
    ) -> torch.Tensor:
        """计算类中心对齐损失用于语义对齐"""
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
                    centroid = features[mask].mean(dim=0)
                    site_centroids.append(centroid)
            
            # 计算中心之间的成对距离
            num_centroids = len(site_centroids)
            if num_centroids < 2:
                continue
            
            for i in range(num_centroids):
                for j in range(i+1, num_centroids):
                    # 计算中心之间的L2距离
                    dist = torch.norm(site_centroids[i] - site_centroids[j], p=2)
                    loss += dist
                    valid_pairs += 1
        
        # 如果存在有效对，返回平均距离，否则返回0
        if valid_pairs > 0:
            return loss / valid_pairs
        
        return loss