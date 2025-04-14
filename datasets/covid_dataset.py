# datasets/covid_dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

class COVID19CTDataset(Dataset):
    """COVID-19 CT数据集类
    
    加载和预处理来自不同站点的COVID-19 CT图像。
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], site_ids: List[int], transform=None):
        """
        初始化COVID-19 CT数据集
        
        参数:
            image_paths: 图像文件路径列表
            labels: 标签列表（0表示非COVID，1表示COVID）
            site_ids: 站点ID列表（0表示SARS-CoV-2，1表示COVID-CT）
            transform: 要应用的图像变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.site_ids = site_ids
        self.transform = transform
        
    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            元组 (image, label, site_id)
            - image: 形状为[channels, height, width]的张量
            - label: 整数（0或1）
            - site_id: 整数（0或1）
        """
        # 加载图像
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像出错 {img_path}: {e}")
            # 返回一个黑色图像作为替代
            image = Image.new('RGB', (224, 224), color=0)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签和站点ID
        label = self.labels[idx]
        site_id = self.site_ids[idx]
        
        return image, label, site_id


def prepare_covid_datasets(
    covid_ct_dir: str,
    sars_cov2_dir: str, 
    img_size: int = 224, 
    batch_size: int = 16  # 论文中使用的批大小
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    准备COVID-19 CT数据集，基于原论文中的数据分割方法
    
    参数:
        covid_ct_dir: COVID-CT数据集目录
        sars_cov2_dir: SARS-CoV-2数据集目录
        img_size: 调整大小的图像尺寸
        batch_size: 批大小
        
    返回:
        元组 (train_loader, val_loader, site_a_val_loader, site_b_val_loader, test_loader)
    """
    # 定义变换 - 使用论文中描述的预处理
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强
        transforms.RandomRotation(degrees=15),   # 数据增强
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ====================== 加载 COVID-CT 数据集 ======================
    # 处理 COVID-CT 数据集的分割文件路径
    covid_split_dir = os.path.join(covid_ct_dir, 'Data-split')
    covid_images_dir = os.path.join(covid_ct_dir, 'Images-processed')
    
    # 读取训练集分割文件
    covid_train_pos_file = os.path.join(covid_split_dir, 'COVID', 'trainCT_COVID.txt')
    covid_train_neg_file = os.path.join(covid_split_dir, 'NonCOVID', 'trainCT_NonCOVID.txt')
    
    # 读取验证集分割文件
    covid_val_pos_file = os.path.join(covid_split_dir, 'COVID', 'valCT_COVID.txt')
    covid_val_neg_file = os.path.join(covid_split_dir, 'NonCOVID', 'valCT_NonCOVID.txt')
    
    # 读取测试集分割文件
    covid_test_pos_file = os.path.join(covid_split_dir, 'COVID', 'testCT_COVID.txt')
    covid_test_neg_file = os.path.join(covid_split_dir, 'NonCOVID', 'testCT_NonCOVID.txt')
    
    # 读取COVID阳性图像列表
    covid_pos_image_dir = os.path.join(covid_images_dir, 'CT_COVID')
    covid_neg_image_dir = os.path.join(covid_images_dir, 'CT_NonCOVID')
    
    # 函数：从分割文件读取图像路径列表
    def read_covid_ct_filelist(filename, image_dir):
        """从COVID-CT分割文件读取图像路径列表"""
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"警告：文件不存在 {filename}")
            return []
            
        # 读取文件内容
        with open(filename, 'r') as f:
            # 处理文件中的每一行
            image_paths = []
            for line in f:
                # 清理行
                line = line.strip()
                if not line:
                    continue
                
                # 构建完整图像路径
                # 对于COVID-CT数据集，需要找到具有相应名称的文件
                base_name = os.path.basename(line)
                
                # 在图像目录中查找匹配的文件
                matched_files = [f for f in os.listdir(image_dir) if base_name in f]
                
                if matched_files:
                    img_path = os.path.join(image_dir, matched_files[0])
                    image_paths.append(img_path)
                else:
                    print(f"警告：无法找到匹配的图像文件 {base_name} 在 {image_dir}")
                    
            return image_paths
    
    # 读取分割数据
    covid_ct_train_pos = read_covid_ct_filelist(covid_train_pos_file, covid_pos_image_dir)
    covid_ct_train_neg = read_covid_ct_filelist(covid_train_neg_file, covid_neg_image_dir)
    covid_ct_val_pos = read_covid_ct_filelist(covid_val_pos_file, covid_pos_image_dir)
    covid_ct_val_neg = read_covid_ct_filelist(covid_val_neg_file, covid_neg_image_dir)
    covid_ct_test_pos = read_covid_ct_filelist(covid_test_pos_file, covid_pos_image_dir)
    covid_ct_test_neg = read_covid_ct_filelist(covid_test_neg_file, covid_neg_image_dir)

    # ========== 新的代码：整合所有COVID-CT数据并重新分割 ==========
    if len(covid_ct_train_pos) > 0 and len(covid_ct_train_neg) > 0:
        print("成功从分割文件读取COVID-CT数据，现在整合所有数据并重新按60/20/20分割")
        
        # 合并所有阳性样本
        covid_ct_all_pos = covid_ct_train_pos + covid_ct_val_pos + covid_ct_test_pos
        
        # 合并所有阴性样本
        covid_ct_all_neg = covid_ct_train_neg + covid_ct_val_neg + covid_ct_test_neg
        
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        np.random.shuffle(covid_ct_all_pos)
        np.random.shuffle(covid_ct_all_neg)
        
        # 重新分割阳性样本（60%训练，20%验证，20%测试）
        n_pos = len(covid_ct_all_pos)
        n_train_pos = int(n_pos * 0.6)
        n_val_pos = int(n_pos * 0.2)
        
        covid_ct_train_pos = covid_ct_all_pos[:n_train_pos]
        covid_ct_val_pos = covid_ct_all_pos[n_train_pos:n_train_pos+n_val_pos]
        covid_ct_test_pos = covid_ct_all_pos[n_train_pos+n_val_pos:]
        
        # 重新分割阴性样本（60%训练，20%验证，20%测试）
        n_neg = len(covid_ct_all_neg)
        n_train_neg = int(n_neg * 0.6)
        n_val_neg = int(n_neg * 0.2)
        
        covid_ct_train_neg = covid_ct_all_neg[:n_train_neg]
        covid_ct_val_neg = covid_ct_all_neg[n_train_neg:n_train_neg+n_val_neg]
        covid_ct_test_neg = covid_ct_all_neg[n_train_neg+n_val_neg:]
        
    else:
        # 如果从分割文件读取失败，直接从目录读取
        print("警告：使用分割文件读取COVID-CT数据集失败，尝试直接从目录读取")
        
        # 直接从目录读取所有图像
        covid_ct_all_pos = [os.path.join(covid_pos_image_dir, f) for f in os.listdir(covid_pos_image_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        covid_ct_all_neg = [os.path.join(covid_neg_image_dir, f) for f in os.listdir(covid_neg_image_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # 手动分割（60%训练，20%验证，20%测试）- 论文中提到的比例
        np.random.seed(42)
        np.random.shuffle(covid_ct_all_pos)
        np.random.shuffle(covid_ct_all_neg)
        
        # 分割COVID阳性样本
        n_pos = len(covid_ct_all_pos)
        n_train_pos = int(n_pos * 0.6)
        n_val_pos = int(n_pos * 0.2)
        
        covid_ct_train_pos = covid_ct_all_pos[:n_train_pos]
        covid_ct_val_pos = covid_ct_all_pos[n_train_pos:n_train_pos+n_val_pos]
        covid_ct_test_pos = covid_ct_all_pos[n_train_pos+n_val_pos:]
        
        # 分割COVID阴性样本
        n_neg = len(covid_ct_all_neg)
        n_train_neg = int(n_neg * 0.6)
        n_val_neg = int(n_neg * 0.2)
        
        covid_ct_train_neg = covid_ct_all_neg[:n_train_neg]
        covid_ct_val_neg = covid_ct_all_neg[n_train_neg:n_train_neg+n_val_neg]
        covid_ct_test_neg = covid_ct_all_neg[n_train_neg+n_val_neg:]
    
    # 准备COVID-CT数据集（站点B）
    site_b_train_paths = covid_ct_train_pos + covid_ct_train_neg
    site_b_train_labels = [1] * len(covid_ct_train_pos) + [0] * len(covid_ct_train_neg)
    site_b_train_site_ids = [1] * len(site_b_train_paths)  # 站点ID 1表示COVID-CT
    
    site_b_val_paths = covid_ct_val_pos + covid_ct_val_neg
    site_b_val_labels = [1] * len(covid_ct_val_pos) + [0] * len(covid_ct_val_neg)
    site_b_val_site_ids = [1] * len(site_b_val_paths)  # 站点ID 1表示COVID-CT
    
    site_b_test_paths = covid_ct_test_pos + covid_ct_test_neg
    site_b_test_labels = [1] * len(covid_ct_test_pos) + [0] * len(covid_ct_test_neg)
    site_b_test_site_ids = [1] * len(site_b_test_paths)  # 站点ID 1表示COVID-CT
    
    # ====================== 加载 SARS-CoV-2 数据集 ======================
    # SARS-CoV-2数据集 (站点A)
    site_a_pos_dir = os.path.join(sars_cov2_dir, 'COVID')
    site_a_neg_dir = os.path.join(sars_cov2_dir, 'non-COVID')
    
    # 读取所有图像文件
    site_a_pos_imgs = [os.path.join(site_a_pos_dir, f) for f in os.listdir(site_a_pos_dir) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    site_a_neg_imgs = [os.path.join(site_a_neg_dir, f) for f in os.listdir(site_a_neg_dir) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # 确保有足够的图像文件
    if len(site_a_pos_imgs) == 0 or len(site_a_neg_imgs) == 0:
        raise ValueError(f"SARS-CoV-2数据集图像文件不足：阳性={len(site_a_pos_imgs)}，阴性={len(site_a_neg_imgs)}")
    
    # 准备数据集分割 - 根据论文中使用的60%/20%/20%分割比例
    np.random.seed(42)
    np.random.shuffle(site_a_pos_imgs)
    np.random.shuffle(site_a_neg_imgs)
    
    # 分割阳性样本（60%训练，20%验证，20%测试）
    n_pos = len(site_a_pos_imgs)
    n_train_pos = int(n_pos * 0.6)
    n_val_pos = int(n_pos * 0.2)
    
    site_a_train_pos = site_a_pos_imgs[:n_train_pos]
    site_a_val_pos = site_a_pos_imgs[n_train_pos:n_train_pos+n_val_pos]
    site_a_test_pos = site_a_pos_imgs[n_train_pos+n_val_pos:]
    
    # 分割阴性样本（60%训练，20%验证，20%测试）
    n_neg = len(site_a_neg_imgs)
    n_train_neg = int(n_neg * 0.6)
    n_val_neg = int(n_neg * 0.2)
    
    site_a_train_neg = site_a_neg_imgs[:n_train_neg]
    site_a_val_neg = site_a_neg_imgs[n_train_neg:n_train_neg+n_val_neg]
    site_a_test_neg = site_a_neg_imgs[n_train_neg+n_val_neg:]
    
    # 准备SARS-CoV-2数据集（站点A）
    site_a_train_paths = site_a_train_pos + site_a_train_neg
    site_a_train_labels = [1] * len(site_a_train_pos) + [0] * len(site_a_train_neg)
    site_a_train_site_ids = [0] * len(site_a_train_paths)  # 站点ID 0表示SARS-CoV-2
    
    site_a_val_paths = site_a_val_pos + site_a_val_neg
    site_a_val_labels = [1] * len(site_a_val_pos) + [0] * len(site_a_val_neg)
    site_a_val_site_ids = [0] * len(site_a_val_paths)  # 站点ID 0表示SARS-CoV-2
    
    site_a_test_paths = site_a_test_pos + site_a_test_neg
    site_a_test_labels = [1] * len(site_a_test_pos) + [0] * len(site_a_test_neg)
    site_a_test_site_ids = [0] * len(site_a_test_paths)  # 站点ID 0表示SARS-CoV-2
    
    # ====================== 组合数据 ======================
    # 组合两个站点的训练数据
    all_train_paths = site_a_train_paths + site_b_train_paths
    all_train_labels = site_a_train_labels + site_b_train_labels
    all_train_site_ids = site_a_train_site_ids + site_b_train_site_ids
    
    # 组合两个站点的验证数据
    all_val_paths = site_a_val_paths + site_b_val_paths
    all_val_labels = site_a_val_labels + site_b_val_labels
    all_val_site_ids = site_a_val_site_ids + site_b_val_site_ids
    
    # 组合两个站点的测试数据
    all_test_paths = site_a_test_paths + site_b_test_paths
    all_test_labels = site_a_test_labels + site_b_test_labels
    all_test_site_ids = site_a_test_site_ids + site_b_test_site_ids
    
    # 打印数据集统计信息
    print("SARS-CoV-2 数据集 (站点A):")
    print(f"  训练: 阳性={len(site_a_train_pos)}, 阴性={len(site_a_train_neg)}, 总计={len(site_a_train_paths)}")
    print(f"  验证: 阳性={len(site_a_val_pos)}, 阴性={len(site_a_val_neg)}, 总计={len(site_a_val_paths)}")
    print(f"  测试: 阳性={len(site_a_test_pos)}, 阴性={len(site_a_test_neg)}, 总计={len(site_a_test_paths)}")
    
    print("COVID-CT 数据集 (站点B):")
    print(f"  训练: 阳性={len(covid_ct_train_pos)}, 阴性={len(covid_ct_train_neg)}, 总计={len(site_b_train_paths)}")
    print(f"  验证: 阳性={len(covid_ct_val_pos)}, 阴性={len(covid_ct_val_neg)}, 总计={len(site_b_val_paths)}")
    print(f"  测试: 阳性={len(covid_ct_test_pos)}, 阴性={len(covid_ct_test_neg)}, 总计={len(site_b_test_paths)}")
    
    print("合并数据集:")
    print(f"  训练: 总计={len(all_train_paths)}")
    print(f"  验证: 总计={len(all_val_paths)}")
    print(f"  测试: 总计={len(all_test_paths)}")
    
    # 创建数据集
    train_dataset = COVID19CTDataset(all_train_paths, all_train_labels, all_train_site_ids, transform_train)
    val_dataset = COVID19CTDataset(all_val_paths, all_val_labels, all_val_site_ids, transform_val)
    test_dataset = COVID19CTDataset(all_test_paths, all_test_labels, all_test_site_ids, transform_val)
    
    # 为每个站点创建单独的验证数据集
    site_a_val_dataset = COVID19CTDataset(site_a_val_paths, site_a_val_labels, site_a_val_site_ids, transform_val)
    site_b_val_dataset = COVID19CTDataset(site_b_val_paths, site_b_val_labels, site_b_val_site_ids, transform_val)
    
    # 创建数据加载器 - 使用论文中的批大小(16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    site_a_val_loader = DataLoader(site_a_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    site_b_val_loader = DataLoader(site_b_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    return train_loader, val_loader, site_a_val_loader, site_b_val_loader, test_loader