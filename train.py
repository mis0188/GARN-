# train.py
# COVID-19 CT图像分类的跨站点恢复能力训练
import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import random
import copy

from models.garn import GARN
from utils.losses import GARNLoss
from datasets.covid_dataset import prepare_covid_datasets

def set_seed(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_lr_scheduler(optimizer, epoch, max_epochs, init_lr, min_lr=1e-6):
    """余弦退火学习率调度器 - 论文中使用的学习率策略"""
    # 计算当前学习率
    lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs))
    
    # 更新优化器中的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def train_epoch(model, criterion, optimizer, train_loader, device):
    """训练一个周期
    
    参数:
        model: GARN模型
        criterion: 损失函数
        optimizer: 优化器
        train_loader: 训练数据加载器
        device: 计算设备(CPU/GPU)
        
    返回:
        运行损失的字典
    """
    model.train()
    running_losses = {"total": 0.0, "cross_entropy": 0.0, "contrastive": 0.0, "triplet": 0.0, "centroid_alignment": 0.0}
    
    # 创建进度条
    loop = tqdm(train_loader, desc="训练中")
    
    for images, labels, site_ids in loop:
        # 将数据移至设备
        # [批大小, 3, 高度, 宽度]
        images = images.to(device)  
        # [批大小]
        labels = labels.to(device)  
        # [批大小]
        site_ids = site_ids.to(device)  
        
        # 前向传递
        # 包含多个输出的字典
        outputs = model(images, site_ids)  
        
        # 计算损失
        # 包含损失组件的字典
        loss_dict = criterion(outputs, labels, site_ids)  
        
        # 检查损失值是否为无穷大或NaN
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]):
                print(f"警告: {key} 损失不是有限值，设置为0")
                loss_dict[key] = torch.tensor(0.0, device=device)
        
        # 重新计算总损失
        loss_dict["total"] = loss_dict["cross_entropy"] + \
                            criterion.alpha * loss_dict["contrastive"] + \
                            criterion.beta * loss_dict["triplet"] + \
                            criterion.gamma * loss_dict["centroid_alignment"]
        
        # 反向传递和优化
        optimizer.zero_grad()
        loss_dict["total"].backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 更新运行损失
        for key in running_losses:
            running_losses[key] += loss_dict[key].item()
        
        # 更新进度条
        loop.set_postfix(loss=loss_dict["total"].item())
    
    # 计算平均损失
    for key in running_losses:
        running_losses[key] /= len(train_loader)
    
    return running_losses

def evaluate(model, criterion, val_loader, device):
    """评估模型
    
    参数:
        model: GARN模型
        criterion: 损失函数
        val_loader: 验证数据加载器
        device: 计算设备(CPU/GPU)
        
    返回:
        包含各种评估指标和预测结果的元组
    """
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
            # 将数据移至设备
            images = images.to(device)
            labels = labels.to(device)
            site_ids = site_ids.to(device)
            
            # 前向传递
            outputs = model(images, site_ids)
            
            # 计算损失
            loss_dict = criterion(outputs, labels, site_ids)
            
            # 检查损失值是否为无穷大或NaN
            for key in loss_dict:
                if not torch.isfinite(loss_dict[key]):
                    loss_dict[key] = torch.tensor(0.0, device=device)
            
            # 重新计算总损失
            loss_dict["total"] = loss_dict["cross_entropy"] + \
                                criterion.alpha * loss_dict["contrastive"] + \
                                criterion.beta * loss_dict["triplet"] + \
                                criterion.gamma * loss_dict["centroid_alignment"]
            
            # 更新运行损失
            for key in running_losses:
                running_losses[key] += loss_dict[key].item()
            
            # 获取预测结果
            probs = torch.softmax(outputs["logits"], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # 存储预测结果、标签和概率
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 对于二分类，取正类的概率
            all_probs.extend(probs[:, 1].cpu().numpy())  
            all_embeddings.append(outputs["embeddings"].cpu())
            if "structure_scores" in outputs:
                all_structure_scores.append(outputs["structure_scores"].cpu())
            all_site_ids.extend(site_ids.cpu().numpy())
    
    # 计算平均损失
    for key in running_losses:
        running_losses[key] /= len(val_loader)
    
    # 连接张量
    all_embeddings = torch.cat(all_embeddings, dim=0)
    if all_structure_scores:
        all_structure_scores = torch.cat(all_structure_scores, dim=0)
    else:
        # 占位符
        all_structure_scores = torch.zeros(1)  
    all_site_ids = np.array(all_site_ids)
    
    # 计算评估指标
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    
    # AUC仅适用于二分类
    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    # 统计各类样本数量
    site_a_pos = sum((all_site_ids == 0) & (np.array(all_labels) == 1))
    site_a_neg = sum((all_site_ids == 0) & (np.array(all_labels) == 0))
    site_b_pos = sum((all_site_ids == 1) & (np.array(all_labels) == 1))
    site_b_neg = sum((all_site_ids == 1) & (np.array(all_labels) == 0))
    
    # 输出各部分的维度信息和样本分布
    print(f"评估维度信息:")
    print(f"  - 预测结果 (all_preds): {len(all_preds)}")
    print(f"  - 标签 (all_labels): {len(all_labels)}")
    print(f"  - 概率 (all_probs): {len(all_probs)}")
    print(f"  - 特征嵌入 (all_embeddings): {all_embeddings.shape}")
    print(f"  - 结构分数 (all_structure_scores): {all_structure_scores.shape}")
    print(f"  - 站点ID (all_site_ids): {all_site_ids.shape}")
    print(f"样本分布: 站点A阳性={site_a_pos}, 站点A阴性={site_a_neg}, 站点B阳性={site_b_pos}, 站点B阴性={site_b_neg}")
    
    return (
        running_losses, accuracy, f1, recall, precision, auc,
        all_preds, all_labels, all_probs, all_embeddings, all_structure_scores, all_site_ids
    )

def main():
    parser = argparse.ArgumentParser(description='训练GARN模型')
    parser.add_argument('--covid_ct_dir', type=str, default='COVID-CT', help='COVID-CT数据集目录')
    parser.add_argument('--sars_cov2_dir', type=str,  default='SARS-COV-2', help='SARS-CoV-2数据集目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小，默认为16（论文中使用的值）')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--epochs', type=int, default=100, help='训练周期，默认为100（论文中的设置）')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率，默认为1e-3（论文中的值）')
    parser.add_argument('--alpha', type=float, default=0.1, help='对比损失权重，默认降低到0.01以提高稳定性')
    parser.add_argument('--beta', type=float, default=0.1, help='三元组损失权重，默认降低到0.01以提高稳定性')
    parser.add_argument('--gamma', type=float, default=0.1, help='中心点对齐损失权重，默认降低到0.01以提高稳定性')
    parser.add_argument('--save_dir', type=str, default='results', help='保存结果的目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(results_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: GARN, 学习率: {args.lr}, 批大小: {args.batch_size}\n")
        f.write(f"损失权重 - alpha: {args.alpha}, beta: {args.beta}, gamma: {args.gamma}\n")
        f.write("=" * 80 + "\n\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("准备数据...")
    train_loader, val_loader, site_a_val_loader, site_b_val_loader, test_loader = prepare_covid_datasets(
        args.covid_ct_dir, args.sars_cov2_dir, img_size=args.img_size, batch_size=args.batch_size
    )
    
    # 检查每个数据加载器的样本分布
    print("\n检查验证集样本分布:")
    # 检查合并验证集
    val_site_a_pos = 0
    val_site_a_neg = 0
    val_site_b_pos = 0
    val_site_b_neg = 0
    
    for _, labels, site_ids in val_loader:
        val_site_a_pos += ((site_ids == 0) & (labels == 1)).sum().item()
        val_site_a_neg += ((site_ids == 0) & (labels == 0)).sum().item()
        val_site_b_pos += ((site_ids == 1) & (labels == 1)).sum().item()
        val_site_b_neg += ((site_ids == 1) & (labels == 0)).sum().item()
    
    print(f"合并验证集分布: 站点A阳性={val_site_a_pos}, 站点A阴性={val_site_a_neg}, 站点B阳性={val_site_b_pos}, 站点B阴性={val_site_b_neg}")
    
    # 初始化模型、损失函数和优化器
    print("初始化模型...")
    model = GARN(n_classes=2, sites=2, backbone_embedding_dim=400, gcn_output_dim=400).to(device)
    criterion = GARNLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 论文中使用的Adam优化器
    
    # 初始化历史记录字典
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auc': [], 'best_accuracy': 0.0, 'best_epoch': 0},
        'site_a_metrics': {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auc': [], 'best_accuracy': 0.0, 'best_epoch': 0},
        'site_b_metrics': {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auc': [], 'best_accuracy': 0.0, 'best_epoch': 0},
    }
    
    # 保存最佳模型的状态字典
    best_model_val = None
    best_model_a = None
    best_model_b = None
    
    # 保存模型的函数
    def save_model(model_dict, save_path, metrics, log_message):
        torch.save(model_dict, save_path)
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(f"{log_message}\n")
            f.write(f"  - 准确率: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}\n")
    
    # 更新日志的函数
    def update_log(epoch, train_loss, val_loss, val_acc, val_f1, val_auc,
                  site_a_acc, site_a_f1, site_a_auc, 
                  site_b_acc, site_b_f1, site_b_auc):
        with open(log_file, 'a') as f:
            f.write(f"周期 {epoch+1}/{args.epochs}\n")
            f.write(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss if isinstance(val_loss, float) else 'inf'}\n")
            f.write(f"验证集 - 准确率: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}\n")
            f.write(f"站点A - 准确率: {site_a_acc:.4f}, F1: {site_a_f1:.4f}, AUC: {site_a_auc:.4f}\n")
            f.write(f"站点B - 准确率: {site_b_acc:.4f}, F1: {site_b_f1:.4f}, AUC: {site_b_auc:.4f}\n")
            f.write("-" * 60 + "\n")
    
    # 训练模型
    print("开始训练...")
    for epoch in range(args.epochs):
        # 更新学习率
        lr = cosine_lr_scheduler(optimizer, epoch, args.epochs, args.lr)
        
        # 训练
        print(f"周期 [{epoch+1}/{args.epochs}], 学习率: {lr:.6f}")
        train_losses = train_epoch(model, criterion, optimizer, train_loader, device)
        history['train_loss'].append(train_losses["total"])
        
        # 验证 - 合并验证集
        print("验证中...")
        val_results = evaluate(model, criterion, val_loader, device)
        val_losses, val_acc, val_f1, val_recall, val_precision, val_auc = val_results[:6]
        history['val_loss'].append(val_losses["total"])
        history['val_metrics']['accuracy'].append(val_acc)
        history['val_metrics']['f1'].append(val_f1)
        history['val_metrics']['recall'].append(val_recall)
        history['val_metrics']['precision'].append(val_precision)
        history['val_metrics']['auc'].append(val_auc)
        
        # 更新验证集最佳准确率
        if val_acc > history['val_metrics']['best_accuracy']:
            history['val_metrics']['best_accuracy'] = val_acc
            history['val_metrics']['best_epoch'] = epoch + 1
            # 保存最佳模型状态
            best_model_val = copy.deepcopy(model.state_dict())
            
            # 保存验证集最佳模型
            best_val_path = os.path.join(checkpoint_dir, "garn_best_val.pth")
            val_metrics = {
                'accuracy': val_acc,
                'f1': val_f1,
                'auc': val_auc
            }
            save_model(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'metrics': val_metrics,
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                best_val_path,
                val_metrics,
                f"保存验证集最佳模型 (周期 {epoch+1})"
            )
        
        # 在站点A上评估
        site_a_results = evaluate(model, criterion, site_a_val_loader, device)
        site_a_losses, site_a_acc, site_a_f1, site_a_recall, site_a_precision, site_a_auc = site_a_results[:6]
        history['site_a_metrics']['accuracy'].append(site_a_acc)
        history['site_a_metrics']['f1'].append(site_a_f1)
        history['site_a_metrics']['recall'].append(site_a_recall)
        history['site_a_metrics']['precision'].append(site_a_precision)
        history['site_a_metrics']['auc'].append(site_a_auc)
        
        # 更新站点A最佳准确率
        if site_a_acc > history['site_a_metrics']['best_accuracy']:
            history['site_a_metrics']['best_accuracy'] = site_a_acc
            history['site_a_metrics']['best_epoch'] = epoch + 1
            # 保存最佳模型状态
            best_model_a = copy.deepcopy(model.state_dict())
            
            # 保存站点A最佳模型
            best_a_path = os.path.join(checkpoint_dir, "garn_best_site_a.pth")
            site_a_metrics = {
                'accuracy': site_a_acc,
                'f1': site_a_f1,
                'auc': site_a_auc
            }
            save_model(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'metrics': site_a_metrics,
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                best_a_path,
                site_a_metrics,
                f"保存站点A最佳模型 (周期 {epoch+1})"
            )
        
        # 在站点B上评估
        site_b_results = evaluate(model, criterion, site_b_val_loader, device)
        site_b_losses, site_b_acc, site_b_f1, site_b_recall, site_b_precision, site_b_auc = site_b_results[:6]
        history['site_b_metrics']['accuracy'].append(site_b_acc)
        history['site_b_metrics']['f1'].append(site_b_f1)
        history['site_b_metrics']['recall'].append(site_b_recall)
        history['site_b_metrics']['precision'].append(site_b_precision)
        history['site_b_metrics']['auc'].append(site_b_auc)
        
        # 更新站点B最佳准确率
        if site_b_acc > history['site_b_metrics']['best_accuracy']:
            history['site_b_metrics']['best_accuracy'] = site_b_acc
            history['site_b_metrics']['best_epoch'] = epoch + 1
            # 保存最佳模型状态
            best_model_b = copy.deepcopy(model.state_dict())
            
            # 保存站点B最佳模型
            best_b_path = os.path.join(checkpoint_dir, "garn_best_site_b.pth")
            site_b_metrics = {
                'accuracy': site_b_acc,
                'f1': site_b_f1,
                'auc': site_b_auc
            }
            save_model(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'metrics': site_b_metrics,
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                best_b_path,
                site_b_metrics,
                f"保存站点B最佳模型 (周期 {epoch+1})"
            )
        
        # 打印当前周期结果和最佳准确率
        val_loss_str = f"{val_losses['total']:.4f}" if torch.isfinite(torch.tensor(val_losses['total'])) else "inf"
        print(f"训练损失: {train_losses['total']:.4f}, 验证损失: {val_loss_str}")
        print(f"验证集 - 准确率: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"站点A - 准确率: {site_a_acc:.4f}, F1: {site_a_f1:.4f}, AUC: {site_a_auc:.4f}")
        print(f"站点B - 准确率: {site_b_acc:.4f}, F1: {site_b_f1:.4f}, AUC: {site_b_auc:.4f}")
        print(f"最佳准确率 - 验证集: {history['val_metrics']['best_accuracy']:.4f} (周期 {history['val_metrics']['best_epoch']})")
        print(f"最佳准确率 - 站点A: {history['site_a_metrics']['best_accuracy']:.4f} (周期 {history['site_a_metrics']['best_epoch']})")
        print(f"最佳准确率 - 站点B: {history['site_b_metrics']['best_accuracy']:.4f} (周期 {history['site_b_metrics']['best_epoch']})")
        print("-" * 80)
        
        # 更新训练日志
        update_log(epoch, train_losses['total'], val_losses['total'], 
                  val_acc, val_f1, val_auc,
                  site_a_acc, site_a_f1, site_a_auc,
                  site_b_acc, site_b_f1, site_b_auc)
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "garn_model_final.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    print(f"保存最终模型到 {final_model_path}")
    
    # 在测试集上评估
    print("在测试集上评估...")
    test_results = evaluate(model, criterion, test_loader, device)
    test_losses, test_acc, test_f1, test_recall, test_precision, test_auc = test_results[:6]
    
    # 训练结束时打印摘要
    print("\n训练完成！最终结果：")
    print("=" * 80)
    print(f"验证集 - 最佳准确率: {history['val_metrics']['best_accuracy']:.4f} (周期 {history['val_metrics']['best_epoch']})")
    print(f"站点A - 最佳准确率: {history['site_a_metrics']['best_accuracy']:.4f} (周期 {history['site_a_metrics']['best_epoch']})")
    print(f"站点B - 最佳准确率: {history['site_b_metrics']['best_accuracy']:.4f} (周期 {history['site_b_metrics']['best_epoch']})")
    print(f"测试 - 准确率: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    print("=" * 80)
    
    # 记录最终结果
    with open(log_file, 'a') as f:
        f.write("\n\n最终结果:\n")
        f.write("=" * 60 + "\n")
        f.write(f"验证集 - 最佳准确率: {history['val_metrics']['best_accuracy']:.4f} (周期 {history['val_metrics']['best_epoch']})\n")
        f.write(f"站点A - 最佳准确率: {history['site_a_metrics']['best_accuracy']:.4f} (周期 {history['site_a_metrics']['best_epoch']})\n")
        f.write(f"站点B - 最佳准确率: {history['site_b_metrics']['best_accuracy']:.4f} (周期 {history['site_b_metrics']['best_epoch']})\n")
        f.write(f"测试 - 准确率: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"\n训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()