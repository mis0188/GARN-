import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class Flatten(nn.Module):
    """展平层：将输入张量展平为一维向量"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0), -1)

class PEPX(nn.Module):
    """PEPX模块：COVID-Net的核心构建块
    
    这个模块通过一系列1x1卷积和深度可分离卷积高效地处理特征。
    """
    
    def __init__(self, n_input: int, n_out: int):
        """
        初始化PEPX模块
        
        参数:
            n_input: 输入通道数
            n_out: 输出通道数
        """
        super(PEPX, self).__init__()
        
        self.network = nn.Sequential(
            # 第一个1x1卷积：特征投影（降维）
            nn.Conv2d(in_channels=n_input, out_channels=n_input // 2, kernel_size=1),
            
            # 第二个1x1卷积：特征扩展
            nn.Conv2d(in_channels=n_input // 2, out_channels=int(3 * n_input / 4), kernel_size=1),
            
            # 3x3深度可分离卷积
            nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=int(3 * n_input / 4), 
                     kernel_size=3, groups=int(3 * n_input / 4), padding=1),
            
            # 第三个1x1卷积：二次投影
            nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=n_input // 2, kernel_size=1),
            
            # 第四个1x1卷积：特征扩展到目标通道数
            nn.Conv2d(in_channels=n_input // 2, out_channels=n_out, kernel_size=1),
            
            # 批归一化层
            nn.BatchNorm2d(n_out)
        )
        
        # 初始化权重 - 使用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PEPX模块的前向传播
        """
        return self.network(x)

class RedesignedCOVIDNet(nn.Module):
    """重新设计的COVID-Net模型
    
    特点：
    1. 添加批归一化层提高特征的判别能力和训练稳定性
    2. 添加站点特定的批归一化层以处理多站点数据异质性
    3. 采用全局平均池化生成紧凑的语义嵌入
    4. 优化特征融合和多层特征提取
    """
    
    def __init__(self, n_classes: int = 2, site_specific_bn: bool = True, sites: int = 2):
        """
        初始化重新设计的COVID-Net模型
        """
        super(RedesignedCOVIDNet, self).__init__()
        
        self.site_specific_bn = site_specific_bn
        self.sites = sites
        
        # 定义PEPX模块的输入输出通道数 - 根据图示精确配置
        filters = {
            'pepx1_1': [56, 56],
            'pepx1_2': [56, 56],
            'pepx1_3': [56, 56],
            'pepx2_1': [56, 112],  # 第一个变换通道数的模块
            'pepx2_2': [112, 112],
            'pepx2_3': [112, 112],
            'pepx2_4': [112, 112],
            'pepx3_1': [112, 216],  # 第二个变换通道数的模块
            'pepx3_2': [216, 216],
            'pepx3_3': [216, 216],
            'pepx3_4': [216, 216],
            'pepx3_5': [216, 216],
            'pepx3_6': [216, 224],  # 注意这里是216→224，不是216→216
            'pepx4_1': [224, 424],  # 输入是224通道，不是216通道
            'pepx4_2': [424, 424],
            'pepx4_3': [424, 400],  # 注意这里是424→400，不是424→424
        }
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=56, kernel_size=7, stride=2, padding=3)
        
        if site_specific_bn:
            # 为每个站点创建独立的批归一化层
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(56) for _ in range(sites)])
            self.bn1_1x1 = nn.ModuleList([nn.BatchNorm2d(56) for _ in range(sites)])
            self.bn2_1x1 = nn.ModuleList([nn.BatchNorm2d(112) for _ in range(sites)])
            self.bn3_1x1 = nn.ModuleList([nn.BatchNorm2d(216) for _ in range(sites)])
            self.bn4_1x1 = nn.ModuleList([nn.BatchNorm2d(424) for _ in range(sites)])
        else:
            # 标准批归一化层
            self.bn1 = nn.BatchNorm2d(56)
            self.bn1_1x1 = nn.BatchNorm2d(56)
            self.bn2_1x1 = nn.BatchNorm2d(112)
            self.bn3_1x1 = nn.BatchNorm2d(216)
            self.bn4_1x1 = nn.BatchNorm2d(424)
        
        # 创建PEPX模块
        for key in filters:
            self.add_module(key, PEPX(filters[key][0], filters[key][1]))
        
        # 1x1卷积层用于特征投影 - 根据图示配置
        self.conv1_1x1 = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=1)
        self.conv2_1x1 = nn.Conv2d(in_channels=56, out_channels=112, kernel_size=1)
        self.conv3_1x1 = nn.Conv2d(in_channels=112, out_channels=216, kernel_size=1)
        self.conv4_1x1 = nn.Conv2d(in_channels=216, out_channels=424, kernel_size=1)
        
        # 全局平均池化、展平和全连接层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        # 注意：最终输出是400通道，不是424通道
        self.fc1 = nn.Linear(400, 512)
        self.classifier = nn.Linear(512, n_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重 - 使用Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, site_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播函数
        
        参数:
            x: 输入图像张量，形状为 [batch_size, 3, height, width]
            site_id: 站点ID张量，形状为 [batch_size]
            
        返回:
            包含logits、embeddings等的字典
        """
        # 初始处理：卷积+批归一化+ReLU+最大池化
        x = self.conv1(x)
        
        # 处理多站点批次
        if self.site_specific_bn:
            out = torch.zeros_like(x)
            for site in torch.unique(site_id):
                site_idx = site.item()
                mask = (site_id == site)
                out[mask] = self.bn1[site_idx](x[mask])
            x = out
        else:
            x = self.bn1(x)
            
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 尺寸减半 120x120 -> 60x60
        
        # 第一阶段：1x1卷积+批归一化
        x_conv1_1x1 = self.conv1_1x1(x)  # 56 -> 56通道
        
        if self.site_specific_bn:
            out_conv1_1x1 = torch.zeros_like(x_conv1_1x1)
            for site in torch.unique(site_id):
                site_idx = site.item()
                mask = (site_id == site)
                out_conv1_1x1[mask] = self.bn1_1x1[site_idx](x_conv1_1x1[mask])
        else:
            out_conv1_1x1 = self.bn1_1x1(x_conv1_1x1)
        
        # 第一阶段PEPX模块
        pepx11 = self.pepx1_1(x)  # 56 -> 56通道
        pepx12 = self.pepx1_2(pepx11 + out_conv1_1x1)  # 56 + 56 -> 56通道
        pepx13 = self.pepx1_3(pepx12 + pepx11 + out_conv1_1x1)  # 56 + 56 + 56 -> 56通道
        
        # 第二阶段：1x1卷积+批归一化+最大池化
        stage1_combined = pepx12 + pepx11 + pepx13 + out_conv1_1x1  # 所有56通道的特征
        x_conv2_1x1 = self.conv2_1x1(stage1_combined)  # 56 -> 112通道
        
        if self.site_specific_bn:
            out_conv2_1x1_temp = torch.zeros_like(x_conv2_1x1)
            for site in torch.unique(site_id):
                site_idx = site.item()
                mask = (site_id == site)
                out_conv2_1x1_temp[mask] = self.bn2_1x1[site_idx](x_conv2_1x1[mask])
            out_conv2_1x1 = F.max_pool2d(out_conv2_1x1_temp, 2)  # 60x60 -> 30x30
        else:
            out_conv2_1x1 = F.max_pool2d(self.bn2_1x1(x_conv2_1x1), 2)
        
        # 第二阶段PEPX模块
        # 将第一阶段输出池化并处理
        stage1_pooled = F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2) + F.max_pool2d(out_conv1_1x1, 2)
        pepx21 = self.pepx2_1(stage1_pooled)  # 56 -> 112通道
        pepx22 = self.pepx2_2(pepx21 + out_conv2_1x1)  # 112 + 112 -> 112通道
        pepx23 = self.pepx2_3(pepx22 + pepx21 + out_conv2_1x1)  # 112 + 112 + 112 -> 112通道
        pepx24 = self.pepx2_4(pepx23 + pepx21 + pepx22 + out_conv2_1x1)  # 112 + 112 + 112 + 112 -> 112通道
        
        # 第三阶段：1x1卷积+批归一化+最大池化
        stage2_combined = pepx22 + pepx21 + pepx23 + pepx24 + out_conv2_1x1  # 所有112通道的特征
        x_conv3_1x1 = self.conv3_1x1(stage2_combined)  # 112 -> 216通道
        
        if self.site_specific_bn:
            out_conv3_1x1_temp = torch.zeros_like(x_conv3_1x1)
            for site in torch.unique(site_id):
                site_idx = site.item()
                mask = (site_id == site)
                out_conv3_1x1_temp[mask] = self.bn3_1x1[site_idx](x_conv3_1x1[mask])
            out_conv3_1x1 = F.max_pool2d(out_conv3_1x1_temp, 2)  # 30x30 -> 15x15
        else:
            out_conv3_1x1 = F.max_pool2d(self.bn3_1x1(x_conv3_1x1), 2)
        
        # 第三阶段PEPX模块
        stage2_pooled = F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23, 2) + F.max_pool2d(out_conv2_1x1, 2)
        pepx31 = self.pepx3_1(stage2_pooled)  # 112 -> 216通道
        pepx32 = self.pepx3_2(pepx31 + out_conv3_1x1)  # 216 + 216 -> 216通道
        pepx33 = self.pepx3_3(pepx31 + pepx32 + out_conv3_1x1)  # 216 + 216 + 216 -> 216通道
        pepx34 = self.pepx3_4(pepx31 + pepx32 + pepx33 + out_conv3_1x1)  # 多个216通道特征
        pepx35 = self.pepx3_5(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1)  # 多个216通道特征
        pepx36 = self.pepx3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + out_conv3_1x1)  # 216 -> 224通道
        
        # 第四阶段：1x1卷积+批归一化+最大池化
        # 注意：这里需要小心处理pepx36的输出，它现在是224通道而不是216通道
        stage3_combined = pepx36 + F.pad(pepx31, [0, 0, 0, 0, 0, 8]) + F.pad(pepx32, [0, 0, 0, 0, 0, 8]) + \
                         F.pad(pepx33, [0, 0, 0, 0, 0, 8]) + F.pad(pepx34, [0, 0, 0, 0, 0, 8]) + \
                         F.pad(pepx35, [0, 0, 0, 0, 0, 8]) + F.pad(out_conv3_1x1, [0, 0, 0, 0, 0, 8])
        
        x_conv4_1x1 = self.conv4_1x1(F.pad(stage3_combined, [0, 0, 0, 0, 0, -8]))  # 216 -> 424通道
        
        if self.site_specific_bn:
            out_conv4_1x1_temp = torch.zeros_like(x_conv4_1x1)
            for site in torch.unique(site_id):
                site_idx = site.item()
                mask = (site_id == site)
                out_conv4_1x1_temp[mask] = self.bn4_1x1[site_idx](x_conv4_1x1[mask])
            out_conv4_1x1 = F.max_pool2d(out_conv4_1x1_temp, 2)  # 15x15 -> 7x7
        else:
            out_conv4_1x1 = F.max_pool2d(self.bn4_1x1(x_conv4_1x1), 2)
        
        # 第四阶段PEPX模块
        stage3_pooled = F.max_pool2d(pepx36, 2)  # 现在是224通道
        pepx41 = self.pepx4_1(stage3_pooled)  # 224 -> 424通道
        pepx42 = self.pepx4_2(pepx41 + out_conv4_1x1)  # 424 + 424 -> 424通道
        pepx43 = self.pepx4_3(pepx41 + pepx42 + out_conv4_1x1)  # 424 -> 400通道
        
        # 特征融合和全局平均池化
        # 注意：pepx43现在是400通道，而out_conv4_1x1是424通道，需要处理这个不匹配
        combined_features = pepx43 + F.pad(pepx41[:, :400, :, :], [0, 0, 0, 0]) + \
                           F.pad(pepx42[:, :400, :, :], [0, 0, 0, 0]) + \
                           F.pad(out_conv4_1x1[:, :400, :, :], [0, 0, 0, 0])  # 最终400通道特征
        
        pooled_features = self.global_pool(combined_features)  # 全局池化到1x1
        flattened = self.flatten(pooled_features)  # 展平为一维向量（400维）
        
        # 全连接层和分类
        fc1out = F.relu(self.fc1(flattened))  # 400 -> 512
        logits = self.classifier(fc1out)  # 512 -> n_classes
        
        # 返回多种输出用于不同的任务
        return {
            "logits": logits,           # 用于分类
            "embeddings": flattened,    # 用于图对齐和对比学习
            "penultimate": fc1out       # 倒数第二层特征
        }


def create_covid_net(n_classes=2, site_specific_bn=True, sites=2):
    """创建重新设计的COVID-Net模型实例"""
    return RedesignedCOVIDNet(n_classes=n_classes, site_specific_bn=site_specific_bn, sites=sites)