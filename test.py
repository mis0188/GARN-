# test.py
# Testing and visualization for COVID-19 CT image classification cross-site robustness

import os
import torch
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from tqdm import tqdm

from models.garn import GARN
from utils.losses import GARNLoss
from datasets.covid_dataset import prepare_covid_datasets

def evaluate(model, criterion, val_loader, device):
    """Evaluate the model"""
    model.eval()
    running_losses = {"total": 0.0, "cross_entropy": 0.0, "contrastive": 0.0, "triplet": 0.0, "centroid_alignment": 0.0}
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = []
    all_structure_scores = []
    all_site_ids = []
    
    with torch.no_grad():
        for images, labels, site_ids in val_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            site_ids = site_ids.to(device)
            
            # Forward pass
            outputs = model(images, site_ids)
            
            # Calculate loss
            loss_dict = criterion(outputs, labels, site_ids)
            
            # Check if loss values are infinite or NaN
            for key in loss_dict:
                if not torch.isfinite(loss_dict[key]):
                    loss_dict[key] = torch.tensor(0.0, device=device)
            
            # Recalculate total loss
            loss_dict["total"] = loss_dict["cross_entropy"] + \
                                criterion.alpha * loss_dict["contrastive"] + \
                                criterion.beta * loss_dict["triplet"] + \
                                criterion.gamma * loss_dict["centroid_alignment"]
            
            # Update running losses
            for key in running_losses:
                running_losses[key] += loss_dict[key].item()
            
            # Get predictions
            probs = torch.softmax(outputs["logits"], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store predictions, labels, and probabilities
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # For binary classification, take probability of positive class
            all_embeddings.append(outputs["embeddings"].cpu())
            if "structure_scores" in outputs:
                all_structure_scores.append(outputs["structure_scores"].cpu())
            all_site_ids.extend(site_ids.cpu().numpy())
    
    # Calculate average losses
    for key in running_losses:
        running_losses[key] /= len(val_loader)
    
    # Concatenate tensors
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 确保标签和站点ID为正确的数据类型
    all_labels = np.array(all_labels)  
    all_site_ids = np.array(all_site_ids)
    
    # 检查数据完整性
    print(f"评估指标 - 嵌入: {all_embeddings.shape}, 标签: {len(all_labels)}, 站点ID: {len(all_site_ids)}")
    print(f"标签分布: {np.unique(all_labels, return_counts=True)}")
    print(f"站点分布: {np.unique(all_site_ids, return_counts=True)}")
    
    if all_structure_scores:
        all_structure_scores = torch.cat(all_structure_scores, dim=0)
    else:
        all_structure_scores = torch.zeros(1)  # Placeholder
    all_site_ids = np.array(all_site_ids)
    
    # Calculate evaluation metrics
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    
    # AUC only applicable for binary classification
    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    # Print sample distribution stats
    site_a_pos = sum((all_site_ids == 0) & (np.array(all_labels) == 1))
    site_a_neg = sum((all_site_ids == 0) & (np.array(all_labels) == 0))
    site_b_pos = sum((all_site_ids == 1) & (np.array(all_labels) == 1))
    site_b_neg = sum((all_site_ids == 1) & (np.array(all_labels) == 0))
    
    print(f"Sample distribution: Site A+={site_a_pos}, Site A-={site_a_neg}, Site B+={site_b_pos}, Site B-={site_b_neg}")
    
    return (
        running_losses, accuracy, f1, recall, precision, auc,
        all_preds, all_labels, all_probs, all_embeddings, all_structure_scores, all_site_ids
    )

def visualize_embeddings(embeddings, labels, site_ids, save_path=None, title=None):
    """修复的嵌入空间可视化函数"""
    # 确保输入为numpy数组
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    site_ids_np = site_ids.cpu().numpy() if isinstance(site_ids, torch.Tensor) else np.array(site_ids)
    
    # 打印原始数据形状和分布
    print(f"原始数据：embeddings: {embeddings_np.shape}, labels: {len(labels_np)}, site_ids: {len(site_ids_np)}")
    print(f"标签分布: {np.unique(labels_np, return_counts=True)}")
    print(f"站点分布: {np.unique(site_ids_np, return_counts=True)}")
    
    # 使用t-SNE降维，确保可靠性
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
    
    # 降维到2D空间
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 定义四种类型点的颜色和标记样式
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 'o', '^', '^']
    
    # 创建四种类型点的掩码 - 确保使用正确的数据类型比较
    mask_A_0 = (site_ids_np == 0) & (labels_np == 0)
    mask_A_1 = (site_ids_np == 0) & (labels_np == 1)
    mask_B_0 = (site_ids_np == 1) & (labels_np == 0)
    mask_B_1 = (site_ids_np == 1) & (labels_np == 1)
    
    # 打印掩码计数
    print(f"Mask A_0 count: {np.sum(mask_A_0)}")
    print(f"Mask A_1 count: {np.sum(mask_A_1)}")
    print(f"Mask B_0 count: {np.sum(mask_B_0)}")
    print(f"Mask B_1 count: {np.sum(mask_B_1)}")
    
    # 仅当存在特定掩码数据时绘制
    if np.sum(mask_A_0) > 0:
        plt.scatter(embeddings_2d[mask_A_0, 0], embeddings_2d[mask_A_0, 1], 
                    c=colors[0], marker=markers[0], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                    label='Site A - Negative (0)')
    
    if np.sum(mask_A_1) > 0:
        plt.scatter(embeddings_2d[mask_A_1, 0], embeddings_2d[mask_A_1, 1], 
                    c=colors[1], marker=markers[1], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                    label='Site A - Positive (1)')
    
    if np.sum(mask_B_0) > 0:
        plt.scatter(embeddings_2d[mask_B_0, 0], embeddings_2d[mask_B_0, 1], 
                    c=colors[2], marker=markers[2], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                    label='Site B - Negative (0)')
    
    if np.sum(mask_B_1) > 0:
        plt.scatter(embeddings_2d[mask_B_1, 0], embeddings_2d[mask_B_1, 1], 
                    c=colors[3], marker=markers[3], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                    label='Site B - Positive (1)')
    
    # 添加图例、标题和网格
    plt.legend(fontsize=12, markerscale=1.5, loc='best')
    
    if title is None:
        title = 'Embedding Space Visualization - By Site and Class'
    plt.title(title, fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # 添加详细解释
    plt.figtext(0.5, 0.01, 
                'Circles: Site A, Triangles: Site B\n' + 
                'Blue/Green: Negative (0), Red/Purple: Positive (1)', 
                ha='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    # 保存图形
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to: {save_path}")
    
    plt.tight_layout()
    plt.close()
    
    return embeddings_2d

def visualize_graph_structure(structure_scores, labels, site_ids=None, save_path=None, title=None):
    """优化后的图结构可视化函数 - 美观性提升版"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    import numpy as np
    from sklearn.manifold import TSNE
    import os
    
    # 转换为numpy数组
    structure_scores_np = structure_scores.cpu().numpy() if isinstance(structure_scores, torch.Tensor) else structure_scores
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    site_ids_np = site_ids.cpu().numpy() if isinstance(site_ids, torch.Tensor) else site_ids
    
    # 打印原始数据分布
    unique_sites = np.unique(site_ids_np)
    unique_labels = np.unique(labels_np)
    print("原始数据分布:")
    for site in unique_sites:
        for label in unique_labels:
            mask = (site_ids_np == site) & (labels_np == label)
            count = np.sum(mask)
            print(f"  - 站点{['A','B'][site]}类别{label}: {count}样本")
    
    # ======== 采样策略 - 每个类别固定数量 ========
    # 固定每个类别的采样数量 - 调整为每类20个样本，数量更均衡
    samples_per_category = 100
    np.random.seed(42)  # 固定随机种子
    
    # 用于存储采样的索引
    all_sampled_indices = []
    
    # 从每个类别中采样固定数量的样本
    for site in unique_sites:
        for label in unique_labels:
            # 创建掩码
            mask = (site_ids_np == site) & (labels_np == label)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                # 确定采样数量（不超过该类别的样本总数）
                n_samples = min(samples_per_category, len(indices))
                
                # 随机采样
                sampled_indices = np.random.choice(indices, size=n_samples, replace=False)
                all_sampled_indices.extend(sampled_indices)
                print(f"从站点{['A','B'][site]}类别{label}中采样了{len(sampled_indices)}/{len(indices)}个样本")
    
    # 检查是否有采样数据
    if len(all_sampled_indices) == 0:
        print("错误: 没有采样到任何数据!")
        return None
    
    # 对采样的索引进行排序（保持一致性）
    all_sampled_indices = sorted(all_sampled_indices)
    
    # 提取采样后的数据
    sampled_features = structure_scores_np[all_sampled_indices]
    sampled_labels = labels_np[all_sampled_indices]
    sampled_site_ids = site_ids_np[all_sampled_indices]
    
    # 打印采样后的数据分布
    print("抽样后数据分布:")
    for site in unique_sites:
        for label in unique_labels:
            count = np.sum((sampled_site_ids == site) & (sampled_labels == label))
            print(f"  - 站点{['A','B'][site]}类别{label}: {count}样本")
    
    # 计算相似度矩阵 - 使用欧氏距离或余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(sampled_features)
    
    # 创建图结构
    G = nx.Graph()
    
    # 添加节点
    for i in range(len(sampled_labels)):
        G.add_node(i, label=int(sampled_labels[i]), site=int(sampled_site_ids[i]))
    
    # 设置边缘阈值 - 提高阈值以减少边的数量，使图形更清晰
    edge_threshold = np.percentile(similarity_matrix, 96)
    
    # 添加边
    edge_count = 0
    for i in range(len(sampled_labels)):
        for j in range(i+1, len(sampled_labels)):
            if similarity_matrix[i, j] > edge_threshold:
                G.add_edge(i, j, weight=float(similarity_matrix[i, j]))
                edge_count += 1
    
    # 如果边太少，降低阈值
    if edge_count < 20 and len(G.nodes) > 20:
        edge_threshold = np.percentile(similarity_matrix, 92)
        for i in range(len(sampled_labels)):
            for j in range(i+1, len(sampled_labels)):
                if similarity_matrix[i, j] > edge_threshold and not G.has_edge(i, j):
                    G.add_edge(i, j, weight=float(similarity_matrix[i, j]))
                    edge_count += 1
    
    print(f"图中添加了{edge_count}条边（强度阈值：{edge_threshold:.4f}）")
    
    # 计算节点位置 - 使用改进的布局算法
    if len(G.nodes) > 0:
        # 使用t-SNE对特征进行降维，但调整参数使聚类更紧凑
        try:
            # 降低perplexity参数使聚类更紧凑
            perplexity = min(25, max(5, len(sampled_features)//12))
            
            # 在t-SNE中使用更低的学习率和更多的迭代次数，获得更稳定的布局
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                       learning_rate=100, n_iter=2000, metric='cosine')
            node_positions = tsne.fit_transform(sampled_features)
            
            # 将相似节点聚集得更紧密
            pos = {i: (node_positions[i, 0], node_positions[i, 1]) for i in range(len(node_positions))}
            
            # 应用缩放以减小节点间距离
            scaling_factor = 0.8
            pos = {i: (x * scaling_factor, y * scaling_factor) for i, (x, y) in pos.items()}
        except Exception as e:
            print(f"t-SNE计算失败: {e}，使用spring_layout代替")
            # 使用spring_layout，但调整k参数使节点更紧凑
            pos = nx.spring_layout(G, k=0.15, seed=42)
    else:
        pos = {}
    
    # 设置颜色和形状映射 - 使用更鲜明的颜色
    color_map = {
        (0, 0): '#3050F8',  # 站点A，类别0 - 更鲜明的蓝色
        (0, 1): '#F03030',  # 站点A，类别1 - 更鲜明的红色
        (1, 0): '#30A030',  # 站点B，类别0 - 更鲜明的绿色
        (1, 1): '#9030A0',  # 站点B，类别1 - 更鲜明的紫色
    }
    shape_map = {0: 'o', 1: '^'}  # 站点A用圆形，站点B用三角形
    
    # 为每个类别创建节点组
    node_groups = {key: [] for key in color_map.keys()}
    
    # 对节点进行分组
    for node in G.nodes():
        site = G.nodes[node]['site']
        label = G.nodes[node]['label']
        key = (site, label)
        if key in node_groups:
            node_groups[key].append(node)
    
    # 统计各组节点数
    print("图中各类节点统计:")
    for (site, label), nodes in node_groups.items():
        print(f"  - 站点{['A','B'][site]}类别{label}: {len(nodes)}个节点")
    
    # 创建图形 - 使用更小的图形尺寸，以便节点更紧凑
    plt.figure(figsize=(14, 12))
    
    # 绘制边 - 使边更细，减少视觉干扰
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        
        # 将权重归一化，但使边更细
        max_score = np.max(similarity_matrix)
        normalized_weight = 0.1 + 0.7 * (weight - edge_threshold) / (max_score - edge_threshold) if max_score > edge_threshold else 0.4
        normalized_weight = min(0.9, max(0.1, normalized_weight))
        
        # 确定边的颜色
        site_u, label_u = G.nodes[u]['site'], G.nodes[u]['label']
        site_v, label_v = G.nodes[v]['site'], G.nodes[v]['label']
        
        if (site_u, label_u) == (site_v, label_v):
            edge_color = color_map.get((site_u, label_u), 'gray')
            alpha = normalized_weight * 0.8  # 降低不透明度
        else:
            edge_color = 'lightgray'  # 使用更浅的灰色
            alpha = normalized_weight * 0.3  # 显著降低不透明度
        
        # 绘制边，使线条更细
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=normalized_weight*1.5, 
                              alpha=alpha, edge_color=edge_color)
    
    # 创建图例元素
    legend_elements = []
    for (site, label), color in color_map.items():
        site_name = ["A", "B"][site]
        label_name = "Negative (0)" if label == 0 else "Positive (1)"
        marker = 'o' if site == 0 else '^'
        
        legend_element = plt.Line2D([0], [0], marker=marker, color='w', 
                                  markerfacecolor=color, markersize=10, 
                                  label=f'Site {site_name} - {label_name}')
        legend_elements.append(legend_element)
    
    # 绘制节点 - 增大节点尺寸，使图形更加突出
    node_size = 120
    for (site, label), nodes in node_groups.items():
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                 node_color=color_map[(site, label)],
                                 node_shape=shape_map[site],
                                 node_size=node_size,
                                 alpha=0.9,  # 增加不透明度
                                 edgecolors='black',
                                 linewidths=0.7)  # 增加边缘线宽度
    
    # 添加标题和图例
    if title is None:
        title = f'Graph Structure Visualization - {len(G.nodes)} Nodes'
    plt.title(title, fontsize=16, fontweight='bold')  # 使标题更大更粗
    plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    # 添加解释说明 - 使用更清晰的格式
    plt.figtext(0.5, 0.01, 
               'Node shape: Circle=Site A, Triangle=Site B\n' + 
               'Node color: Blue/Green=Negative (0), Red/Purple=Positive (1)\n' +
               'Edge color indicates the type of connected nodes, edge thickness indicates relationship strength', 
               ha='center', fontsize=11, bbox={'facecolor':'white', 'alpha':0.9, 'pad':5, 'edgecolor':'lightgray'})
    
    # 移除坐标轴
    plt.axis('off')
    
    # 保存图形
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph structure visualization saved to: {save_path}")
    
    plt.tight_layout()
    plt.close()
    
    return G
    
def plot_confusion_matrices(labels, preds, save_path=None, title=None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    if title is None:
        title = 'Confusion Matrix'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()

def plot_roc_curves(labels, probs, save_path=None, title=None):
    """Plot ROC curves"""
    from sklearn.metrics import roc_curve, auc
    
    # Set up figure
    plt.figure(figsize=(8, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    
    # Add diagonal
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Add labels and legend
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    if title is None:
        title = 'Receiver Operating Characteristic (ROC)'
    plt.title(title)
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.close()

def plot_class_centroids(embeddings, labels, site_ids, save_path=None, title=None):
    """Visualize class centroids"""
    # Convert to numpy arrays
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    site_ids_np = site_ids.cpu().numpy() if isinstance(site_ids, torch.Tensor) else site_ids
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Calculate centroids for each class and site
    centroids = {}
    for site in np.unique(site_ids_np):  # Handle any number of sites
        for label in np.unique(labels_np):  # Handle any number of classes
            mask = (site_ids_np == site) & (labels_np == label)
            if np.any(mask):
                centroids[(site, label)] = embeddings_2d[mask].mean(axis=0)
    
    # Plot embedding space and centroids
    plt.figure(figsize=(10, 8))
    
    # Define colors and markers
    colors = {
        (0, 0): 'blue',
        (0, 1): 'red',
        (1, 0): 'green',
        (1, 1): 'purple'
    }
    markers = {0: 'o', 1: '^'}
    
    # Plot samples (with lower alpha)
    for site in np.unique(site_ids_np):
        for label in np.unique(labels_np):
            mask = (site_ids_np == site) & (labels_np == label)
            if np.any(mask):
                plt.scatter(
                    embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=colors.get((site, label), 'gray'), marker=markers.get(site, 'o'), alpha=0.2, s=30,
                    label=f'Site {["A","B"][site]} - {"Negative (0)" if label==0 else "Positive (1)"}'
                )
    
    # Plot centroids (larger, more obvious)
    for (site, label), centroid in centroids.items():
        plt.scatter(
            centroid[0], centroid[1],
            c=colors.get((site, label), 'gray'), marker=markers.get(site, 'o'), alpha=1.0, s=200, edgecolors='black',
            label=f'Centroid: Site {["A","B"][site]} - {"Negative (0)" if label==0 else "Positive (1)"}' 
        )
    
    # Add legend and title
    if title is None:
        title = 'Class Centroid Visualization'
    plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class centroid visualization saved to: {save_path}")
    
    plt.close()

def create_visualizations(model, criterion, dataloader, device, dataset_name, output_dir):
    """创建数据集的完整可视化集"""
    print(f"\n创建{dataset_name}的可视化...")
    
    # 数据集检查 - 确保包含所有需要的类别
    all_labels = []
    all_site_ids = []
    
    # 采样部分数据检查类别分布
    print("检查数据集分布...")
    for _, labels, site_ids in dataloader:
        all_labels.extend(labels.numpy())
        all_site_ids.extend(site_ids.numpy())
    
    # 打印类别分布
    site_a_pos = sum((np.array(all_site_ids) == 0) & (np.array(all_labels) == 1))
    site_a_neg = sum((np.array(all_site_ids) == 0) & (np.array(all_labels) == 0))
    site_b_pos = sum((np.array(all_site_ids) == 1) & (np.array(all_labels) == 1))
    site_b_neg = sum((np.array(all_site_ids) == 1) & (np.array(all_labels) == 0))
    
    print(f"数据集'{dataset_name}'的类别分布:")
    print(f"  - 站点A阴性样本: {site_a_neg}")
    print(f"  - 站点A阳性样本: {site_a_pos}")
    print(f"  - 站点B阴性样本: {site_b_neg}")
    print(f"  - 站点B阳性样本: {site_b_pos}")
    
    # 如果当前是站点A验证集，但没有站点A的样本，发出警告
    if "Site_A" in dataset_name and (site_a_pos + site_a_neg) == 0:
        print(f"警告: 站点A验证集中没有站点A的样本!")
    
    # 如果当前是站点B验证集，但没有站点B的样本，发出警告
    if "Site_B" in dataset_name and (site_b_pos + site_b_neg) == 0:
        print(f"警告: 站点B验证集中没有站点B的样本!")
    
    # 创建子目录
    vis_categories = ['embeddings', 'confusion_matrices', 'roc_curves', 'graph_structure', 'class_centroids']
    for category in vis_categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    # 评估模型
    results = evaluate(model, criterion, dataloader, device)
    _, accuracy, f1, recall, precision, auc, preds, labels, probs, embeddings, structure_scores, site_ids = results
    
    
    # Print metrics
    print(f"Metrics for {dataset_name}:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - AUC: {auc:.4f}")
    
    # 1. Embedding Space Visualization
    emb_path = os.path.join(output_dir, "embeddings", f"{dataset_name}_embeddings.png")
    visualize_embeddings(
        embeddings,
        labels,
        site_ids,
        save_path=emb_path,
        title=f"Embedding Space Visualization - {dataset_name}"
    )
    
    # 2. Confusion Matrix
    cm_path = os.path.join(output_dir, "confusion_matrices", f"{dataset_name}_confusion_matrix.png")
    plot_confusion_matrices(
        labels,
        preds,
        save_path=cm_path,
        title=f"Confusion Matrix - {dataset_name}"
    )
    
    # 3. ROC Curve
    roc_path = os.path.join(output_dir, "roc_curves", f"{dataset_name}_roc_curve.png")
    plot_roc_curves(
        labels,
        probs,
        save_path=roc_path,
        title=f"ROC Curve - {dataset_name}"
    )
    
    # 4. Graph Structure (if available)
    if structure_scores.size(0) > 1:
        graph_path = os.path.join(output_dir, "graph_structure", f"{dataset_name}_graph_structure.png")
        visualize_graph_structure(
            structure_scores,
            labels,
            site_ids,
            save_path=graph_path,
            title=f"Graph Structure Visualization - {dataset_name}"
        )
    
    # 5. Class Centroids
    centroid_path = os.path.join(output_dir, "class_centroids", f"{dataset_name}_class_centroids.png")
    plot_class_centroids(
        embeddings,
        labels,
        site_ids,
        save_path=centroid_path,
        title=f"Class Centroid Visualization - {dataset_name}"
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'preds': preds,
        'labels': labels,
        'probs': probs,
        'embeddings': embeddings,
        'structure_scores': structure_scores,
        'site_ids': site_ids
    }

def main():
    parser = argparse.ArgumentParser(description='Test and visualize GARN model for COVID-19 CT classification')
    parser.add_argument('--site_a_model_path', type=str, default='./results/run_20250415_000234/checkpoints/garn_best_site_a.pth', help='Path to the best model checkpoint for Site A')
    parser.add_argument('--site_b_model_path', type=str, default='./results/run_20250415_000234/checkpoints/garn_best_site_b.pth', help='Path to the best model checkpoint for Site B')
    parser.add_argument('--val_model_path', type=str, default='./results/run_20250415_000234/checkpoints/garn_best_val.pth', help='Path to the best model checkpoint for combined validation set')
    parser.add_argument('--covid_ct_dir', type=str, default='./COVID-CT', help='COVID-CT dataset directory')
    parser.add_argument('--sars_cov2_dir', type=str, default='./SARS-COV-2', help='SARS-CoV-2 dataset directory')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--alpha', type=float, default=0.01, help='Contrastive loss weight')
    parser.add_argument('--beta', type=float, default=0.01, help='Triplet loss weight')
    parser.add_argument('--gamma', type=float, default=0.01, help='Centroid alignment loss weight')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_dir, 'test_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Test start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Batch size: {args.batch_size}, Image size: {args.img_size}\n")
        f.write(f"Loss weights - alpha: {args.alpha}, beta: {args.beta}, gamma: {args.gamma}\n")
        f.write("=" * 80 + "\n\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    _, val_loader, site_a_val_loader, site_b_val_loader, test_loader = prepare_covid_datasets(
        args.covid_ct_dir, args.sars_cov2_dir, img_size=args.img_size, batch_size=args.batch_size
    )
    
    # Initialize model and loss function
    print("Initializing model...")
    model = GARN(n_classes=2, sites=2, backbone_embedding_dim=400, gcn_output_dim=400).to(device)
    criterion = GARNLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    
    # Function to load model checkpoint
    def load_model_checkpoint(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dictionary")
        model.eval()
    
    # Create visualizations for each dataset with its best model
    print("Creating visualizations for all datasets with their best models...")
    
    # 1. Site A validation set with Site A best model
    print("\n===== EVALUATING SITE A WITH SITE A BEST MODEL =====")
    load_model_checkpoint(args.site_a_model_path)
    site_a_results = create_visualizations(
        model, criterion, site_a_val_loader, device, 
        "Site_A_Validation", output_dir
    )
    
    # 2. Site B validation set with Site B best model
    print("\n===== EVALUATING SITE B WITH SITE B BEST MODEL =====")
    load_model_checkpoint(args.site_b_model_path)
    site_b_results = create_visualizations(
        model, criterion, site_b_val_loader, device, 
        "Site_B_Validation", output_dir
    )
    
    # 3. Combined validation set with validation best model
    print("\n===== EVALUATING COMBINED VALIDATION WITH VALIDATION BEST MODEL =====")
    load_model_checkpoint(args.val_model_path)
    val_results = create_visualizations(
        model, criterion, val_loader, device, 
        "Combined_Validation", output_dir
    )
    
    
    # Print summary of results
    print("\n===== TEST RESULTS SUMMARY =====")
    print("Site A Validation Set:")
    print(f"  - Accuracy: {site_a_results['accuracy']:.4f}")
    print(f"  - F1 Score: {site_a_results['f1']:.4f}")
    print(f"  - AUC: {site_a_results['auc']:.4f}")
    
    print("\nSite B Validation Set:")
    print(f"  - Accuracy: {site_b_results['accuracy']:.4f}")
    print(f"  - F1 Score: {site_b_results['f1']:.4f}")
    print(f"  - AUC: {site_b_results['auc']:.4f}")
    
    print("\nCombined Validation Set:")
    print(f"  - Accuracy: {val_results['accuracy']:.4f}")
    print(f"  - F1 Score: {val_results['f1']:.4f}")
    print(f"  - AUC: {val_results['auc']:.4f}")
    
    # Save results to log
    with open(log_file, 'a') as f:
        f.write("\n===== TEST RESULTS SUMMARY =====\n")
        f.write("Site A Validation Set:\n")
        f.write(f"  - Accuracy: {site_a_results['accuracy']:.4f}\n")
        f.write(f"  - F1 Score: {site_a_results['f1']:.4f}\n")
        f.write(f"  - AUC: {site_a_results['auc']:.4f}\n\n")
        
        f.write("Site B Validation Set:\n")
        f.write(f"  - Accuracy: {site_b_results['accuracy']:.4f}\n")
        f.write(f"  - F1 Score: {site_b_results['f1']:.4f}\n")
        f.write(f"  - AUC: {site_b_results['auc']:.4f}\n\n")
        
        f.write("Combined Validation Set:\n")
        f.write(f"  - Accuracy: {val_results['accuracy']:.4f}\n")
        f.write(f"  - F1 Score: {val_results['f1']:.4f}\n")
        f.write(f"  - AUC: {val_results['auc']:.4f}\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nTest end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()