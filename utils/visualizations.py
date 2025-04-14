# utils/visualizations.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
from torch.nn import functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union

def plot_training_metrics(
    history: Dict[str, List], 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制训练和验证指标
    
    参数:
        history: 包含训练历史的字典
        save_path: 保存图形的路径（可选）
        
    返回:
        Matplotlib图形
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制损失
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Training')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    
    # 绘制准确率
    ax = axes[0, 1]
    ax.plot(history['site_a_metrics']['accuracy'], label='Site A')
    ax.plot(history['site_b_metrics']['accuracy'], label='Site B')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy by Site')
    ax.legend()
    ax.grid(True)
    
    # 绘制AUC
    ax = axes[1, 0]
    ax.plot(history['site_a_metrics']['auc'], label='Site A')
    ax.plot(history['site_b_metrics']['auc'], label='Site B')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Validation AUC by Site')
    ax.legend()
    ax.grid(True)
    
    # 绘制F1分数
    ax = axes[1, 1]
    ax.plot(history['site_a_metrics']['f1'], label='Site A')
    ax.plot(history['site_b_metrics']['f1'], label='Site B')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Score by Site')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig

def plot_loss_components(
    history: Dict[str, List], 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制训练损失组件
    
    参数:
        history: 包含损失组件的训练历史字典
        save_path: 保存图形的路径（可选）
        
    返回:
        Matplotlib图形
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制总损失
    ax = axes[0, 0]
    ax.plot(history['train_loss_total'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Training Loss')
    ax.grid(True)
    
    # 绘制交叉熵损失
    ax = axes[0, 1]
    ax.plot(history['train_loss_ce'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cross Entropy Loss')
    ax.grid(True)
    
    # 绘制对比损失
    ax = axes[1, 0]
    ax.plot(history['train_loss_contrastive'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Contrastive Loss')
    ax.grid(True)
    
    # 绘制三元组损失和中心对齐损失
    ax = axes[1, 1]
    ax.plot(history['train_loss_triplet'], label='Triplet')
    ax.plot(history['train_loss_ca'], label='Centroid Alignment')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Triplet and Centroid Alignment Loss')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig

def visualize_embeddings(
    embeddings: torch.Tensor, 
    labels: torch.Tensor, 
    site_ids: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    使用t-SNE可视化嵌入
    
    参数:
        embeddings: 嵌入张量，形状为 [num_samples, embedding_dim]
        labels: 标签张量，形状为 [num_samples]
        site_ids: 站点ID张量，形状为 [num_samples]
        save_path: 保存图形的路径（可选）
        
    返回:
        Matplotlib图形
    """
    # 将张量转换为NumPy数组
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    site_ids_np = site_ids.detach().cpu().numpy()
    
    # 应用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 按类别着色
    for cls in np.unique(labels_np):
        mask = labels_np == cls
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Class {cls}', alpha=0.7)
    ax1.set_title('Embeddings Visualization by Class')
    ax1.legend()
    ax1.grid(True)
    
    # 按站点着色
    for site in np.unique(site_ids_np):
        mask = site_ids_np == site
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Site {site}', alpha=0.7)
    ax2.set_title('Embeddings Visualization by Site')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig

def visualize_graph_structure(
    structure_scores: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    max_nodes: int = 20  # 限制节点数量以提高可读性
) -> plt.Figure:
    """
    可视化模型学习的图结构
    
    参数:
        structure_scores: 结构分数张量，形状为 [num_samples, score_dim]
        labels: 标签张量，形状为 [num_samples]
        save_path: 保存图形的路径（可选）
        max_nodes: 要显示的最大节点数
        
    返回:
        Matplotlib图形
    """
    # 限制样本数量以便可视化
    if structure_scores.shape[0] > max_nodes:
        indices = np.random.choice(structure_scores.shape[0], max_nodes, replace=False)
        structure_scores = structure_scores[indices]
        labels = labels[indices]
    
    # 将张量转换为NumPy数组
    structure_scores_np = structure_scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # 计算邻接矩阵
    adj_matrix = np.matmul(structure_scores_np, structure_scores_np.T)
    
    # 创建networkx图
    G = nx.from_numpy_array(adj_matrix)
    
    # 指定节点属性
    node_colors = [['#1f77b4', '#ff7f0e'][int(l)] for l in labels_np]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 使用spring布局设置节点位置
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制图
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8, ax=ax)
    
    # 绘制边，宽度与权重成比例
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray', ax=ax)
    
    # 添加节点标签
    labels_dict = {i: f"{i} (C{labels_np[i]})" for i in range(len(labels_np))}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=10, ax=ax)
    
    # 创建图例
    class_labels = [0, 1]
    legend_colors = ['#1f77b4', '#ff7f0e']
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Class {cls}')
                     for cls, color in zip(class_labels, legend_colors)]
    ax.legend(handles=legend_patches, title='Classes')
    
    ax.set_title('Graph Structure Visualization')
    ax.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig

def plot_confusion_matrices(
    true_labels_a: np.ndarray,
    pred_labels_a: np.ndarray,
    true_labels_b: np.ndarray,
    pred_labels_b: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制两个站点的混淆矩阵
    
    参数:
        true_labels_a: 站点A的真实标签
        pred_labels_a: 站点A的预测标签
        true_labels_b: 站点B的真实标签
        pred_labels_b: 站点B的预测标签
        save_path: 保存图形的路径（可选）
        
    返回:
        Matplotlib图形
    """
    # 计算混淆矩阵
    cm_a = confusion_matrix(true_labels_a, pred_labels_a)
    cm_b = confusion_matrix(true_labels_b, pred_labels_b)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 绘制站点A的混淆矩阵
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix - Site A')
    
    # 绘制站点B的混淆矩阵
    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix - Site B')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig

def plot_roc_curves(
    true_labels_a: np.ndarray,
    pred_probs_a: np.ndarray,
    true_labels_b: np.ndarray,
    pred_probs_b: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制两个站点的ROC曲线
    
    参数:
        true_labels_a: 站点A的真实标签
        pred_probs_a: 站点A的预测概率
        true_labels_b: 站点B的真实标签
        pred_probs_b: 站点B的预测概率
        save_path: 保存图形的路径（可选）
        
    返回:
        Matplotlib图形
    """
    # 计算ROC曲线
    fpr_a, tpr_a, _ = roc_curve(true_labels_a, pred_probs_a)
    roc_auc_a = auc(fpr_a, tpr_a)
    
    fpr_b, tpr_b, _ = roc_curve(true_labels_b, pred_probs_b)
    roc_auc_b = auc(fpr_b, tpr_b)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制ROC曲线
    ax.plot(fpr_a, tpr_a, label=f'Site A (AUC = {roc_auc_a:.3f})')
    ax.plot(fpr_b, tpr_b, label=f'Site B (AUC = {roc_auc_b:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')  # 对角线
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig

def plot_class_centroids(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    site_ids: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    可视化不同站点的类中心
    
    参数:
        features: 特征嵌入张量，形状为 [num_samples, feature_dim]
        labels: 标签张量，形状为 [num_samples]
        site_ids: 站点ID张量，形状为 [num_samples]
        save_path: 保存图形的路径（可选）
        
    返回:
        Matplotlib图形
    """
    # 将张量转换为NumPy数组
    features_np = features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    site_ids_np = site_ids.detach().cpu().numpy()
    
    # 应用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_np)
    
    # 计算每个类别和站点的中心
    unique_classes = np.unique(labels_np)
    unique_sites = np.unique(site_ids_np)
    
    centroids = []
    for cls in unique_classes:
        for site in unique_sites:
            # 获取当前类别和站点的特征
            mask = (labels_np == cls) & (site_ids_np == site)
            if np.sum(mask) > 0:
                centroid = np.mean(features_2d[mask], axis=0)
                centroids.append((centroid, cls, site))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制所有点（透明）
    for cls in unique_classes:
        for site in unique_sites:
            mask = (labels_np == cls) & (site_ids_np == site)
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], alpha=0.1, 
                      label=f'Class {cls}, Site {site}' if np.sum(mask) > 0 else None)
    
    # 绘制中心
    markers = ['o', 's']  # 不同站点的不同标记
    colors = ['#1f77b4', '#ff7f0e']  # 不同类别的不同颜色
    
    for (x, y), cls, site in centroids:
        ax.scatter(x, y, c=colors[int(cls)], marker=markers[int(site)], s=200, edgecolor='black', linewidth=2)
    
    # 在同一类别但不同站点的中心之间添加箭头
    for cls in unique_classes:
        cls_centroids = [(c, s) for (c, cl, s) in centroids if cl == cls]
        if len(cls_centroids) > 1:
            for i in range(len(cls_centroids)):
                for j in range(i + 1, len(cls_centroids)):
                    (c1, s1), (c2, s2) = cls_centroids[i], cls_centroids[j]
                    ax.arrow(c1[0], c1[1], c2[0] - c1[0], c2[1] - c1[1], 
                            head_width=0.3, head_length=0.5, fc=colors[int(cls)], ec=colors[int(cls)])
    
    # 创建图例
    class_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'Class {i}')
                    for i in range(len(unique_classes))]
    site_patches = [plt.Line2D([0], [0], marker=markers[i], color='gray', markersize=10, label=f'Site {i}')
                   for i in range(len(unique_sites))]
    ax.legend(handles=class_patches + site_patches)
    
    ax.set_title('Class Centroids by Site (t-SNE Projection)')
    ax.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig