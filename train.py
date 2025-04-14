# train.py
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
import random
import copy
import re
import networkx as nx
from sklearn.manifold import TSNE

from models.garn import GARN
from utils.losses import GARNLoss
from datasets.covid_dataset import prepare_covid_datasets

def set_seed(seed):
    """Set random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_lr_scheduler(optimizer, epoch, max_epochs, init_lr, min_lr=1e-6):
    """Cosine annealing learning rate scheduler - learning rate strategy used in the paper"""
    lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def train_epoch(model, criterion, optimizer, train_loader, device):
    """Train for one epoch"""
    model.train()
    running_losses = {"total": 0.0, "cross_entropy": 0.0, "contrastive": 0.0, "triplet": 0.0, "centroid_alignment": 0.0}
    
    # Create progress bar
    loop = tqdm(train_loader, desc="Training")
    
    for images, labels, site_ids in loop:
        # Move data to device
        images = images.to(device)  # [batch_size, 3, height, width]
        labels = labels.to(device)  # [batch_size]
        site_ids = site_ids.to(device)  # [batch_size]
        
        # Forward pass
        outputs = model(images, site_ids)  # Dictionary with multiple outputs
        
        # Calculate loss
        loss_dict = criterion(outputs, labels, site_ids)  # Dictionary with loss components
        
        # Check if loss values are infinite or NaN
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]):
                print(f"Warning: {key} loss is not finite, setting to 0")
                loss_dict[key] = torch.tensor(0.0, device=device)
        
        # Recalculate total loss
        loss_dict["total"] = loss_dict["cross_entropy"] + \
                            criterion.alpha * loss_dict["contrastive"] + \
                            criterion.beta * loss_dict["triplet"] + \
                            criterion.gamma * loss_dict["centroid_alignment"]
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss_dict["total"].backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update running losses
        for key in running_losses:
            running_losses[key] += loss_dict[key].item()
        
        # Update progress bar
        loop.set_postfix(loss=loss_dict["total"].item())
    
    # Calculate average losses
    for key in running_losses:
        running_losses[key] /= len(train_loader)
    
    return running_losses

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
    
    return (
        running_losses, accuracy, f1, recall, precision, auc,
        all_preds, all_labels, all_probs, all_embeddings, all_structure_scores, all_site_ids
    )

def get_viz_filename(viz_type, model_type, dataset_type, epoch, extension="png"):
    """
    Generate standardized visualization filename
    
    Parameters:
        viz_type: Type of visualization (such as 'embeddings', 'confusion_matrix')
        model_type: Type of model (such as 'best_a', 'best_b', 'current')
        dataset_type: Type of dataset (such as 'train', 'val', 'test')
        epoch: Training epoch
        extension: File extension (default is 'png')
        
    Returns:
        Standardized filename
    """
    # Ensure input parameters are valid
    viz_type = re.sub(r'[^a-zA-Z0-9_]', '', viz_type.lower().replace(' ', '_'))
    model_type = re.sub(r'[^a-zA-Z0-9_]', '', model_type.lower().replace(' ', '_'))
    dataset_type = re.sub(r'[^a-zA-Z0-9_]', '', dataset_type.lower().replace(' ', '_'))
    
    # Build filename
    if isinstance(epoch, int):
        filename = f"{viz_type}_{model_type}_{dataset_type}_epoch{epoch:03d}.{extension}"
    else:
        filename = f"{viz_type}_{model_type}_{dataset_type}_{epoch}.{extension}"
    
    return filename

# Improved embeddings visualization with four distinct points
def visualize_embeddings(embeddings, labels, site_ids, save_path=None, title=None):
    """
    Create 2D visualization of the embedding space with different points for site and class
    
    Parameters:
        embeddings: High-dimensional embedding vectors [N, embedding_dim]
        labels: Class labels [N]
        site_ids: Site IDs [N]
        save_path: Path to save the visualization
        title: Plot title
    """
    # Ensure inputs are numpy arrays
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    site_ids_np = site_ids.cpu().numpy()
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
    
    # Reduce to 2D space
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Define colors and marker styles for the four types of points
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 'o', '^', '^']
    
    # Create masks for the four types of points
    # Site A, Class 0
    mask_A_0 = (site_ids_np == 0) & (labels_np == 0)
    # Site A, Class 1
    mask_A_1 = (site_ids_np == 0) & (labels_np == 1)
    # Site B, Class 0
    mask_B_0 = (site_ids_np == 1) & (labels_np == 0)
    # Site B, Class 1
    mask_B_1 = (site_ids_np == 1) & (labels_np == 1)
    
    # Plot the four types of points
    plt.scatter(embeddings_2d[mask_A_0, 0], embeddings_2d[mask_A_0, 1], 
                c=colors[0], marker=markers[0], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                label='Site A - Negative (0)')
    
    plt.scatter(embeddings_2d[mask_A_1, 0], embeddings_2d[mask_A_1, 1], 
                c=colors[1], marker=markers[1], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                label='Site A - Positive (1)')
    
    plt.scatter(embeddings_2d[mask_B_0, 0], embeddings_2d[mask_B_0, 1], 
                c=colors[2], marker=markers[2], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                label='Site B - Negative (0)')
    
    plt.scatter(embeddings_2d[mask_B_1, 0], embeddings_2d[mask_B_1, 1], 
                c=colors[3], marker=markers[3], alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                label='Site B - Positive (1)')
    
    # Add legend, title and grid
    plt.legend(fontsize=12, markerscale=1.5, loc='best')
    
    if title is None:
        title = 'Embedding Space Visualization - By Site and Class'
    plt.title(title, fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Add detailed explanation
    plt.figtext(0.5, 0.01, 
                'Circles: Site A, Triangles: Site B\n' + 
                'Blue/Green: Negative (0), Red/Purple: Positive (1)', 
                ha='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    # Save the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to: {save_path}")
    
    plt.tight_layout()
    plt.close()
    
    return embeddings_2d

# Fixed graph structure visualization function
def visualize_graph_structure(structure_scores, labels, site_ids=None, save_path=None, title=None):
    """
    Visualize the graph structure learned by the graph neural network
    
    Parameters:
        structure_scores: Graph structure scores matrix [N, N], representing relationships between nodes
        labels: Node class labels [N]
        site_ids: Node site IDs [N], if None assumes all nodes are from the same site
        save_path: Path to save the visualization
        title: Plot title
    """
    import networkx as nx
    from sklearn.manifold import TSNE
    
    # Convert to numpy arrays
    structure_scores_np = structure_scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if site_ids is not None:
        site_ids_np = site_ids.cpu().numpy()
    else:
        # If site IDs not provided, assume all nodes are from the same site
        site_ids_np = np.zeros_like(labels_np)
    
    # Check if dimensions match
    print(f"Structure scores shape: {structure_scores_np.shape}, Labels length: {len(labels_np)}")
    
    # Debug: check unique sites and labels
    unique_sites = np.unique(site_ids_np)
    unique_labels = np.unique(labels_np)
    print(f"Unique sites: {unique_sites}, Unique labels: {unique_labels}")
    
    # Handle non-square matrices by creating a square one
    if structure_scores_np.shape[0] != structure_scores_np.shape[1]:
        print(f"Reshaping non-square structure matrix: {structure_scores_np.shape}")
        
        # Use the smaller dimension
        min_dim = min(structure_scores_np.shape[0], structure_scores_np.shape[1])
        
        # Create a new square matrix using available data
        new_matrix = np.zeros((min_dim, min_dim))
        for i in range(min_dim):
            for j in range(min_dim):
                if i < structure_scores_np.shape[0] and j < structure_scores_np.shape[1]:
                    new_matrix[i, j] = structure_scores_np[i, j]
        
        structure_scores_np = new_matrix
        labels_np = labels_np[:min_dim]
        site_ids_np = site_ids_np[:min_dim]
        print(f"Reshaped to: {structure_scores_np.shape}")
    
    # Create graph structure
    G = nx.Graph()
    
    # Set node threshold - control the number of nodes to display
    max_nodes = 200  # Maximum number of nodes to prevent overly complex graph
    if len(labels_np) > max_nodes:
        # Randomly select a subset of nodes, ensuring we stay within bounds
        max_idx = len(labels_np) - 1
        indices = np.random.choice(max_idx + 1, min(max_nodes, max_idx + 1), replace=False)
        structure_scores_np = structure_scores_np[indices][:, indices]
        labels_np = labels_np[indices]
        site_ids_np = site_ids_np[indices]
    
    # Set edge threshold - only show edges with relationship strength above threshold
    edge_threshold = np.percentile(structure_scores_np, 95)  # Only keep top 5% of edges
    print(f"Edge threshold: {edge_threshold}, Max score: {np.max(structure_scores_np)}")
    
    # Add nodes
    for i in range(len(labels_np)):
        # Create node attributes: class and site
        G.add_node(i, label=int(labels_np[i]), site=int(site_ids_np[i]))
    
    # Add edges (only add edges with relationship strength above threshold)
    edge_count = 0
    for i in range(len(labels_np)):
        for j in range(i+1, len(labels_np)):
            if structure_scores_np[i, j] > edge_threshold:
                G.add_edge(i, j, weight=float(structure_scores_np[i, j]))
                edge_count += 1
    
    print(f"Added {edge_count} edges to graph with {len(G.nodes)} nodes")
    
    # If the graph is too sparse, lower the edge threshold
    if edge_count < 10 and len(G.nodes) > 10:
        print("Graph too sparse, lowering edge threshold")
        edge_threshold = np.percentile(structure_scores_np, 90)  # Try top 10%
        for i in range(len(labels_np)):
            for j in range(i+1, len(labels_np)):
                if structure_scores_np[i, j] > edge_threshold and not G.has_edge(i, j):
                    G.add_edge(i, j, weight=float(structure_scores_np[i, j]))
                    edge_count += 1
        print(f"After lowering threshold: {edge_count} edges")
    
    # Use network layout algorithm to determine node positions
    if len(G.nodes) > 0:
        if len(G.nodes) < 10:
            # For small graphs, use physics-based layout
            pos = nx.spring_layout(G, seed=42)
        else:
            # For larger graphs, use t-SNE to calculate node positions
            # First use structure scores as features
            features = structure_scores_np
            
            # Use t-SNE for dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            node_positions = tsne.fit_transform(features)
            
            # Create position dictionary
            pos = {i: (node_positions[i, 0], node_positions[i, 1]) for i in range(len(node_positions))}
    else:
        # If no nodes, create empty dictionary
        pos = {}
    
    # Draw graph
    plt.figure(figsize=(14, 12))
    
    # Define node colors and shapes based on site and class
    color_map = {
        (0, 0): 'blue',   # Site A, Class 0
        (0, 1): 'red',    # Site A, Class 1
        (1, 0): 'green',  # Site B, Class 0
        (1, 1): 'purple'  # Site B, Class 1
    }
    shape_map = {
        0: 'o',  # Circle for Site A
        1: '^'   # Triangle for Site B
    }
    
    # Create node groups for each category
    node_groups = {key: [] for key in color_map.keys()}
    
    # Count nodes in each group
    node_counts = {key: 0 for key in color_map.keys()}
    
    for node in G.nodes():
        site = G.nodes[node]['site']
        label = G.nodes[node]['label']
        key = (site, label)
        if key in node_groups:
            node_groups[key].append(node)
            node_counts[key] += 1
        else:
            print(f"Warning: Unexpected site-label combination: ({site}, {label})")
    
    print(f"Node counts by group: {node_counts}")
    
    # Draw edges (using transparency to represent weight)
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        # Normalize weight to [0.1, 1.0] range
        max_score = np.max(structure_scores_np)
        if max_score == edge_threshold:  # Prevent division by zero
            normalized_weight = 0.5
        else:
            normalized_weight = 0.1 + 0.9 * (weight - edge_threshold) / (max_score - edge_threshold)
        
        # FIX: Ensure alpha is strictly within [0, 1]
        normalized_weight = min(0.999, max(0.001, normalized_weight))
        
        # If two nodes belong to the same class and site, use corresponding color; otherwise use gray
        site_u, label_u = G.nodes[u]['site'], G.nodes[u]['label']
        site_v, label_v = G.nodes[v]['site'], G.nodes[v]['label']
        
        if (site_u, label_u) == (site_v, label_v):
            edge_color = color_map.get((site_u, label_u), 'gray')
            alpha = normalized_weight
        else:
            edge_color = 'grey'
            alpha = normalized_weight * 0.5  # Edges between different classes/sites more transparent
        
        # FIX: Ensure alpha is strictly within [0, 1]
        alpha = min(0.999, max(0.001, alpha))
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=normalized_weight*2, 
                             alpha=alpha, edge_color=edge_color)
    
    # Draw the four types of nodes separately
    for (site, label), nodes in node_groups.items():
        if nodes:  # If this group has nodes
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                 node_color=color_map[(site, label)],
                                 node_shape=shape_map[site],
                                 node_size=100, 
                                 alpha=0.8,
                                 edgecolors='black',
                                 linewidths=0.5,
                                 label=f'Site {["A","B"][site]} - {"Negative (0)" if label==0 else "Positive (1)"}')
    
    # Add legend and title
    if title is None:
        title = 'Graph Structure Visualization - By Site and Class'
    plt.title(title, fontsize=14)
    plt.legend(scatterpoints=1, fontsize=12)
    
    # Add detailed explanation
    plt.figtext(0.5, 0.01, 
               'Node shape: Circle=Site A, Triangle=Site B\n' + 
               'Node color: Blue/Green=Negative (0), Red/Purple=Positive (1)\n' +
               'Edge color indicates the type of connected nodes, edge thickness indicates relationship strength', 
               ha='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph structure visualization saved to: {save_path}")
    
    plt.tight_layout()
    plt.close()
    
    return G  # Return the created graph object for further analysis

# Confusion matrix visualization
def plot_confusion_matrices(labels1, preds1, labels2=None, preds2=None, save_path=None):
    """Plot confusion matrix comparison"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Set up figure
    if labels2 is not None and preds2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(6, 5))
    
    # First confusion matrix
    cm1 = confusion_matrix(labels1, preds1)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix')
    
    # Second confusion matrix (if provided)
    if labels2 is not None and preds2 is not None:
        cm2 = confusion_matrix(labels2, preds2)
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_title('Comparison Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

# ROC curve visualization
def plot_roc_curves(labels1, probs1, labels2=None, probs2=None, save_path=None):
    """Plot ROC curves"""
    from sklearn.metrics import roc_curve, auc
    
    # Set up figure
    plt.figure(figsize=(8, 6))
    
    # First ROC curve
    fpr1, tpr1, _ = roc_curve(labels1, probs1)
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'ROC Curve 1 (AUC = {roc_auc1:.2f})')
    
    # Second ROC curve (if provided)
    if labels2 is not None and probs2 is not None:
        fpr2, tpr2, _ = roc_curve(labels2, probs2)
        roc_auc2 = auc(fpr2, tpr2)
        plt.plot(fpr2, tpr2, color='red', lw=2, label=f'ROC Curve 2 (AUC = {roc_auc2:.2f})')
    
    # Add diagonal
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Add labels and legend
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

# Class centroid visualization
def plot_class_centroids(embeddings, labels, site_ids, save_path=None):
    """Visualize class centroids"""
    # Convert to numpy arrays
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    site_ids_np = site_ids.cpu().numpy()
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Calculate centroids for each class and site
    centroids = {}
    for site in [0, 1]:  # Sites A and B
        for label in [0, 1]:  # Binary classification classes
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
    for site in [0, 1]:
        for label in [0, 1]:
            mask = (site_ids_np == site) & (labels_np == label)
            plt.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=colors[(site, label)], marker=markers[site], alpha=0.2, s=30,
                label=f'Site {["A","B"][site]} - {"Negative (0)" if label==0 else "Positive (1)"}'
            )
    
    # Plot centroids (larger, more obvious)
    for (site, label), centroid in centroids.items():
        plt.scatter(
            centroid[0], centroid[1],
            c=colors[(site, label)], marker=markers[site], alpha=1.0, s=200, edgecolors='black',
            label=f'Centroid: Site {["A","B"][site]} - {"Negative (0)" if label==0 else "Positive (1)"}' 
        )
    
    # Add legend and title
    plt.title('Class Centroid Visualization')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

# Training metrics visualization
def plot_training_metrics(history, save_path=None):
    """Plot metrics during training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(epochs, history['site_a_metrics']['accuracy'], 'b-', label='Site A Accuracy')
    ax2.plot(epochs, history['site_b_metrics']['accuracy'], 'r-', label='Site B Accuracy')
    
    # Mark best epochs
    if history['best_epoch_a'] > 0:
        ax2.axvline(x=history['best_epoch_a'], color='b', linestyle='--', 
                   label=f'Site A Best (Epoch {history["best_epoch_a"]})')
    if history['best_epoch_b'] > 0:
        ax2.axvline(x=history['best_epoch_b'], color='r', linestyle='--',
                   label=f'Site B Best (Epoch {history["best_epoch_b"]})')
    
    ax2.set_title('Site A and B Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

# Loss component visualization
def plot_loss_components(history, save_path=None):
    """Plot loss components"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss_total']) + 1)
    
    plt.plot(epochs, history['train_loss_total'], 'k-', label='Total Loss')
    plt.plot(epochs, history['train_loss_ce'], 'b-', label='Cross Entropy Loss')
    plt.plot(epochs, history['train_loss_contrastive'], 'r-', label='Contrastive Loss')
    plt.plot(epochs, history['train_loss_triplet'], 'g-', label='Triplet Loss')
    plt.plot(epochs, history['train_loss_ca'], 'y-', label='Centroid Alignment Loss')
    
    plt.title('Training Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_visualizations(model, criterion, dataloader, device, epoch, model_type, dataset_type, visualization_dir):
    """
    Create a set of detailed visualizations to analyze model performance and data representation
    
    Parameters:
        model: GARN model
        criterion: Loss function
        dataloader: Data loader
        device: Device
        epoch: Current epoch
        model_type: Type of model (such as 'best_a', 'best_b', 'current')
        dataset_type: Type of dataset (such as 'train', 'val', 'test')
        visualization_dir: Directory to save visualizations
    """
    # Evaluate model on specified data loader
    results = evaluate(model, criterion, dataloader, device)
    _, _, _, _, _, _, preds, labels, probs, embeddings, structure_scores, site_ids = results
    
    # 1. Create embedding visualization
    emb_path = os.path.join(
        visualization_dir, 
        "embeddings", 
        get_viz_filename("embeddings", model_type, dataset_type, epoch)
    )
    visualize_embeddings(
        embeddings,
        torch.tensor(labels),
        torch.tensor(site_ids),
        save_path=emb_path,
        title=f"Embedding Space - {model_type} ({dataset_type}) - Epoch {epoch}"
    )
    
    # 2. Create confusion matrix
    cm_path = os.path.join(
        visualization_dir, 
        "confusion_matrices", 
        get_viz_filename("confusion_matrix", model_type, dataset_type, epoch)
    )
    plot_confusion_matrices(
        labels,
        preds,
        save_path=cm_path
    )
    
    # 3. Create ROC curve
    roc_path = os.path.join(
        visualization_dir, 
        "roc_curves", 
        get_viz_filename("roc_curve", model_type, dataset_type, epoch)
    )
    plot_roc_curves(
        labels,
        probs,
        save_path=roc_path
    )
    
    # 4. If structure_scores is not a placeholder, create graph structure visualization
    if structure_scores.size(0) > 1:
        graph_path = os.path.join(
            visualization_dir, 
            "graph_structure", 
            get_viz_filename("graph_structure", model_type, dataset_type, epoch)
        )
        visualize_graph_structure(
            structure_scores,
            torch.tensor(labels),
            torch.tensor(site_ids),
            save_path=graph_path,
            title=f"Graph Structure - {model_type} ({dataset_type}) - Epoch {epoch}"
        )
    
    # 5. Create class centroid visualization
    centroid_path = os.path.join(
        visualization_dir, 
        "class_centroids", 
        get_viz_filename("class_centroids", model_type, dataset_type, epoch)
    )
    plot_class_centroids(
        embeddings,
        torch.tensor(labels),
        torch.tensor(site_ids),
        save_path=centroid_path
    )

def main():
    parser = argparse.ArgumentParser(description='Train GARN model')
    parser.add_argument('--covid_ct_dir', type=str, required=True, help='COVID-CT dataset directory')
    parser.add_argument('--sars_cov2_dir', type=str, required=True, help='SARS-CoV-2 dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size, default is 16 (value used in the paper)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs, default is 100 (setting from the paper)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate, default is 1e-3 (value from the paper)')
    parser.add_argument('--alpha', type=float, default=0.01, help='Contrastive loss weight, default reduced to 0.01 for stability')
    parser.add_argument('--beta', type=float, default=0.01, help='Triplet loss weight, default reduced to 0.01 for stability')
    parser.add_argument('--gamma', type=float, default=0.01, help='Centroid alignment loss weight, default reduced to 0.01 for stability')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--viz_interval', type=int, default=10, help='Visualization interval (epochs)')
    # New parameters for visualization control
    parser.add_argument('--basic_viz_interval', type=int, default=5, help='Basic visualization interval (epochs)')
    parser.add_argument('--detailed_viz_on_improvement', action='store_true', 
                      help='Create detailed visualizations on site-specific improvements')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    visualization_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Create subdirectories for specific types of visualizations
    viz_categories = [
        'embeddings', 
        'confusion_matrices', 
        'roc_curves', 
        'loss_components', 
        'graph_structure', 
        'class_centroids', 
        'performance_metrics',
        'site_comparison'
    ]
    for category in viz_categories:
        os.makedirs(os.path.join(visualization_dir, category), exist_ok=True)
    
    # Create log file
    log_file = os.path.join(results_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: GARN, Learning rate: {args.lr}, Batch size: {args.batch_size}\n")
        f.write(f"Loss weights - alpha: {args.alpha}, beta: {args.beta}, gamma: {args.gamma}\n")
        f.write(f"Visualization settings - Basic interval: {args.basic_viz_interval}, Detailed interval: {args.viz_interval}\n")
        f.write(f"Create detailed viz on improvement: {args.detailed_viz_on_improvement}\n")
        f.write("=" * 80 + "\n\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader, site_a_val_loader, site_b_val_loader, test_loader = prepare_covid_datasets(
        args.covid_ct_dir, args.sars_cov2_dir, img_size=args.img_size, batch_size=args.batch_size
    )
    
    # Initialize model, loss function, and optimizer
    print("Initializing model...")
    model = GARN(n_classes=2, sites=2, backbone_embedding_dim=400, gcn_output_dim=400).to(device)
    criterion = GARNLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam optimizer used in the paper
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_loss_total': [],
        'train_loss_ce': [],
        'train_loss_contrastive': [],
        'train_loss_triplet': [],
        'train_loss_ca': [],
        'site_a_metrics': {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auc': [], 'best_accuracy': 0.0},
        'site_b_metrics': {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auc': [], 'best_accuracy': 0.0},
        'best_epoch_a': 0,
        'best_epoch_b': 0
    }
    
    # Save state dictionaries of best models
    best_model_a = None
    best_model_b = None
    
    # Function to update log
    def update_log(epoch, train_loss, val_loss, site_a_acc, site_a_f1, site_a_auc, 
                   site_b_acc, site_b_f1, site_b_auc, best_a_acc, best_b_acc, best_epoch_a, best_epoch_b):
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs}\n")
            f.write(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss if isinstance(val_loss, float) else 'inf'}\n")
            f.write(f"Site A - Accuracy: {site_a_acc:.4f}, F1: {site_a_f1:.4f}, AUC: {site_a_auc:.4f}\n")
            f.write(f"Site B - Accuracy: {site_b_acc:.4f}, F1: {site_b_f1:.4f}, AUC: {site_b_auc:.4f}\n")
            f.write(f"Site A best accuracy: {best_a_acc:.4f} (Epoch {best_epoch_a})\n")
            f.write(f"Site B best accuracy: {best_b_acc:.4f} (Epoch {best_epoch_b})\n")
            f.write("-" * 60 + "\n")
    
    # Train model
    print("Starting training...")
    for epoch in range(args.epochs):
        # Update learning rate
        lr = cosine_lr_scheduler(optimizer, epoch, args.epochs, args.lr)
        
        # Train
        print(f"Epoch [{epoch+1}/{args.epochs}], Learning rate: {lr:.6f}")
        train_losses = train_epoch(model, criterion, optimizer, train_loader, device)
        history['train_loss'].append(train_losses["total"])
        history['train_loss_total'].append(train_losses["total"])
        history['train_loss_ce'].append(train_losses["cross_entropy"])
        history['train_loss_contrastive'].append(train_losses["contrastive"])
        history['train_loss_triplet'].append(train_losses["triplet"])
        history['train_loss_ca'].append(train_losses["centroid_alignment"])
        
        # Validate
        print("Validating...")
        val_results = evaluate(model, criterion, val_loader, device)
        val_losses, val_acc, val_f1, val_recall, val_precision, val_auc = val_results[:6]
        history['val_loss'].append(val_losses["total"])
        
        # Evaluate on site A
        site_a_results = evaluate(model, criterion, site_a_val_loader, device)
        site_a_losses, site_a_acc, site_a_f1, site_a_recall, site_a_precision, site_a_auc = site_a_results[:6]
        history['site_a_metrics']['accuracy'].append(site_a_acc)
        history['site_a_metrics']['f1'].append(site_a_f1)
        history['site_a_metrics']['recall'].append(site_a_recall)
        history['site_a_metrics']['precision'].append(site_a_precision)
        history['site_a_metrics']['auc'].append(site_a_auc)
        
        # Update site A best accuracy
        site_a_improved = False
        if site_a_acc > history['site_a_metrics']['best_accuracy']:
            history['site_a_metrics']['best_accuracy'] = site_a_acc
            history['best_epoch_a'] = epoch + 1
            # Save best model state
            best_model_a = copy.deepcopy(model.state_dict())
            site_a_improved = True
            
            # Save site A best model
            best_a_path = os.path.join(checkpoint_dir, f"garn_best_site_a_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'site_a_acc': site_a_acc,
                'site_a_f1': site_a_f1,
                'site_a_auc': site_a_auc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_a_path)
            print(f"Saved site A best model to {best_a_path}")
        
        # Evaluate on site B
        site_b_results = evaluate(model, criterion, site_b_val_loader, device)
        site_b_losses, site_b_acc, site_b_f1, site_b_recall, site_b_precision, site_b_auc = site_b_results[:6]
        history['site_b_metrics']['accuracy'].append(site_b_acc)
        history['site_b_metrics']['f1'].append(site_b_f1)
        history['site_b_metrics']['recall'].append(site_b_recall)
        history['site_b_metrics']['precision'].append(site_b_precision)
        history['site_b_metrics']['auc'].append(site_b_auc)
        
        # Update site B best accuracy
        site_b_improved = False
        if site_b_acc > history['site_b_metrics']['best_accuracy']:
            history['site_b_metrics']['best_accuracy'] = site_b_acc
            history['best_epoch_b'] = epoch + 1
            # Save best model state
            best_model_b = copy.deepcopy(model.state_dict())
            site_b_improved = True
            
            # Save site B best model
            best_b_path = os.path.join(checkpoint_dir, f"garn_best_site_b_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'site_b_acc': site_b_acc,
                'site_b_f1': site_b_f1,
                'site_b_auc': site_b_auc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_b_path)
            print(f"Saved site B best model to {best_b_path}")
        
        # Print current epoch results and best accuracy
        val_loss_str = f"{val_losses['total']:.4f}" if torch.isfinite(torch.tensor(val_losses['total'])) else "inf"
        print(f"Training loss: {train_losses['total']:.4f}, Validation loss: {val_loss_str}")
        print(f"Site A - Accuracy: {site_a_acc:.4f}, F1: {site_a_f1:.4f}, AUC: {site_a_auc:.4f}")
        print(f"Site B - Accuracy: {site_b_acc:.4f}, F1: {site_b_f1:.4f}, AUC: {site_b_auc:.4f}")
        print(f"Site A best accuracy: {history['site_a_metrics']['best_accuracy']:.4f} (Epoch {history['best_epoch_a']})")
        print(f"Site B best accuracy: {history['site_b_metrics']['best_accuracy']:.4f} (Epoch {history['best_epoch_b']})")
        print("-" * 80)
        
        # Update training log
        update_log(epoch, train_losses['total'], val_losses['total'], 
                  site_a_acc, site_a_f1, site_a_auc,
                  site_b_acc, site_b_f1, site_b_auc,
                  history['site_a_metrics']['best_accuracy'], 
                  history['site_b_metrics']['best_accuracy'],
                  history['best_epoch_a'], history['best_epoch_b'])
        
        # --- VISUALIZATION STRATEGY: OPTIMIZED ---
        
        # 1. Basic performance metrics visualization (at specified interval)
        if (epoch + 1) % args.basic_viz_interval == 0 or epoch == args.epochs - 1:
            # Plot loss components
            loss_path = os.path.join(
                visualization_dir, 
                "loss_components", 
                get_viz_filename("loss_components", "current", "train", epoch+1)
            )
            plot_loss_components(history, save_path=loss_path)
            
            # Plot training metrics
            metrics_path = os.path.join(
                visualization_dir, 
                "performance_metrics", 
                get_viz_filename("training_metrics", "overall", "all", epoch+1)
            )
            plot_training_metrics(history, save_path=metrics_path)
        
        # 2. Detailed visualizations at key points
        create_detailed_viz = False
        
        # Create detailed viz at regular intervals and final epoch
        if (epoch + 1) % args.viz_interval == 0 or epoch == args.epochs - 1:
            create_detailed_viz = True
        
        # Create detailed viz on site-specific improvements (if enabled)
        if args.detailed_viz_on_improvement and (site_a_improved or site_b_improved):
            create_detailed_viz = True
        
        if create_detailed_viz:
            print("Creating detailed visualizations...")
            
            # Save current model state
            current_model_state = copy.deepcopy(model.state_dict())
            
            # Visualize current model
            create_visualizations(
                model, criterion, val_loader, device,
                epoch+1, "current", "val", visualization_dir
            )
            
            # If site A best model exists and has improved (or on interval), create visualizations
            if best_model_a is not None and (site_a_improved or (epoch + 1) % args.viz_interval == 0):
                model.load_state_dict(best_model_a)
                create_visualizations(
                    model, criterion, site_a_val_loader, device, 
                    epoch+1, "best_site_a", "val", visualization_dir
                )
            
            # If site B best model exists and has improved (or on interval), create visualizations
            if best_model_b is not None and (site_b_improved or (epoch + 1) % args.viz_interval == 0):
                model.load_state_dict(best_model_b)
                create_visualizations(
                    model, criterion, site_b_val_loader, device, 
                    epoch+1, "best_site_b", "val", visualization_dir
                )
            
            # Create site comparison visualizations (at interval or final epoch)
            if (epoch + 1) % args.viz_interval == 0 or epoch == args.epochs - 1:
                # Create site comparison visualization
                if best_model_a is not None and best_model_b is not None:
                    # Load site A best model and get embeddings
                    model.load_state_dict(best_model_a)
                    with torch.no_grad():
                        a_results = evaluate(model, criterion, val_loader, device)
                        a_embeddings = a_results[9]
                        a_labels = a_results[7]
                        a_site_ids = a_results[11]
                    
                    # Load site B best model and get embeddings
                    model.load_state_dict(best_model_b)
                    with torch.no_grad():
                        b_results = evaluate(model, criterion, val_loader, device)
                        b_embeddings = b_results[9]
                        b_labels = b_results[7]
                        b_site_ids = b_results[11]
                    
                    # Compare embedding spaces of the two sites
                    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                    
                    # t-SNE dimensionality reduction
                    tsne = TSNE(n_components=2, random_state=42)
                    a_embeddings_2d = tsne.fit_transform(a_embeddings.numpy())
                    
                    tsne = TSNE(n_components=2, random_state=42)
                    b_embeddings_2d = tsne.fit_transform(b_embeddings.numpy())
                    
                    # Color by site
                    for site in np.unique(a_site_ids):
                        mask = a_site_ids == site
                        axes[0].scatter(a_embeddings_2d[mask, 0], a_embeddings_2d[mask, 1], label=f'Site {site}', alpha=0.7)
                    axes[0].set_title('Site A Best Model - Embeddings by Site')
                    axes[0].legend()
                    axes[0].grid(True)
                    
                    for site in np.unique(b_site_ids):
                        mask = b_site_ids == site
                        axes[1].scatter(b_embeddings_2d[mask, 0], b_embeddings_2d[mask, 1], label=f'Site {site}', alpha=0.7)
                    axes[1].set_title('Site B Best Model - Embeddings by Site')
                    axes[1].legend()
                    axes[1].grid(True)
                    
                    plt.suptitle(f'Comparison of Site A and Site B Best Model Embedding Spaces (Epoch {epoch+1})')
                    plt.tight_layout()
                    
                    # Save site comparison figure
                    comparison_path = os.path.join(
                        visualization_dir, 
                        "site_comparison", 
                        get_viz_filename("site_comparison", "best_models", "embedding", epoch+1)
                    )
                    plt.savefig(comparison_path)
                    plt.close()
            
            # Restore current model state to continue training
            model.load_state_dict(current_model_state)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "garn_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate(model, criterion, test_loader, device)
    test_losses, test_acc, test_f1, test_recall, test_precision, test_auc = test_results[:6]
    
    # Print summary at end of training
    print("\nTraining complete! Final results:")
    print("=" * 80)
    print(f"Site A - Best accuracy: {history['site_a_metrics']['best_accuracy']:.4f} (Epoch {history['best_epoch_a']})")
    if history['best_epoch_a'] > 0 and history['best_epoch_a'] <= len(history['site_a_metrics']['f1']):
        print(f"        F1: {history['site_a_metrics']['f1'][history['best_epoch_a']-1]:.4f}")
        print(f"        AUC: {history['site_a_metrics']['auc'][history['best_epoch_a']-1]:.4f}")
    print(f"Site B - Best accuracy: {history['site_b_metrics']['best_accuracy']:.4f} (Epoch {history['best_epoch_b']})")
    if history['best_epoch_b'] > 0 and history['best_epoch_b'] <= len(history['site_b_metrics']['f1']):
        print(f"        F1: {history['site_b_metrics']['f1'][history['best_epoch_b']-1]:.4f}")
        print(f"        AUC: {history['site_b_metrics']['auc'][history['best_epoch_b']-1]:.4f}")
    print(f"Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    print("=" * 80)
    
    # Create final visualization summary
    # 1. Plot accuracy curves and best epochs
    final_fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, args.epochs + 1)
    ax.plot(epochs[:len(history['site_a_metrics']['accuracy'])], history['site_a_metrics']['accuracy'], 'b-', label='Site A Accuracy')
    ax.plot(epochs[:len(history['site_b_metrics']['accuracy'])], history['site_b_metrics']['accuracy'], 'r-', label='Site B Accuracy')
    
    if history['best_epoch_a'] > 0:
        ax.axvline(x=history['best_epoch_a'], color='b', linestyle='--', label=f'Site A Best (Epoch {history["best_epoch_a"]})')
    if history['best_epoch_b'] > 0:
        ax.axvline(x=history['best_epoch_b'], color='r', linestyle='--', label=f'Site B Best (Epoch {history["best_epoch_b"]})')
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Site A and B Accuracy Progression and Best Epochs')
    ax.legend()
    ax.grid(True)
    
    # Save final comparison figure
    final_comparison_path = os.path.join(
        visualization_dir, 
        "performance_metrics",
        get_viz_filename("accuracy_comparison", "final", "all_sites", "summary")
    )
    plt.savefig(final_comparison_path)
    plt.close()
    
    # 2. Create final test set visualizations using best models
    if best_model_a is not None:
        model.load_state_dict(best_model_a)
        create_visualizations(
            model, criterion, test_loader, device, 
            "final", "best_site_a", "test", visualization_dir
        )
    
    if best_model_b is not None:
        model.load_state_dict(best_model_b)
        create_visualizations(
            model, criterion, test_loader, device, 
            "final", "best_site_b", "test", visualization_dir
        )
    
    # Log final results
    with open(log_file, 'a') as f:
        f.write("\n\nFinal results:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Site A - Best accuracy: {history['site_a_metrics']['best_accuracy']:.4f} (Epoch {history['best_epoch_a']})\n")
        if history['best_epoch_a'] > 0 and history['best_epoch_a'] <= len(history['site_a_metrics']['f1']):
            f.write(f"        F1: {history['site_a_metrics']['f1'][history['best_epoch_a']-1]:.4f}\n")
            f.write(f"        AUC: {history['site_a_metrics']['auc'][history['best_epoch_a']-1]:.4f}\n")
        f.write(f"Site B - Best accuracy: {history['site_b_metrics']['best_accuracy']:.4f} (Epoch {history['best_epoch_b']})\n")
        if history['best_epoch_b'] > 0 and history['best_epoch_b'] <= len(history['site_b_metrics']['f1']):
            f.write(f"        F1: {history['site_b_metrics']['f1'][history['best_epoch_b']-1]:.4f}\n")
            f.write(f"        AUC: {history['site_b_metrics']['auc'][history['best_epoch_b']-1]:.4f}\n")
        f.write(f"Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nTraining end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()