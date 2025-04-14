# models/graph_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """图卷积层
    
    执行图卷积操作：H = D^(-1/2) A D^(-1/2) X W
    其中A是邻接矩阵，X是节点特征，D是度矩阵，W是权重。
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        """初始化图卷积层"""
        super(GraphConvLayer, self).__init__()
        # 可学习的权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 可学习的偏置
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化权重和偏置"""
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 应用dropout到输入特征
        x = self.dropout(x)
        
        # 归一化邻接矩阵
        deg = torch.sum(adj, dim=2, keepdim=True)
        deg = torch.clamp(deg, min=1.0)  # 防止除零
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(1, 2)
        
        # 图卷积操作
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support)
        
        return output + self.bias


class DataStructureAnalyzer(nn.Module):
    """数据结构分析器 (DSA)
    
    生成用于构建图邻接矩阵的结构分数。
    """
    
    def __init__(self, input_dim: int = 400, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.2):
        """初始化数据结构分析器"""
        super(DataStructureAnalyzer, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(dropout),  # 添加dropout
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(dropout),  # 添加dropout

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(dropout),  # 添加dropout
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 处理批量大小为1的情况
        if x.size(0) == 1:
            x_temp = x.repeat(2, 1)
            x_temp = self.network(x_temp)
            return x_temp[:1]
        
        return self.network(x)


class GraphConvolutionalNetwork(nn.Module):
    """图卷积网络 (GCN)
    
    通过堆叠多个图卷积层构建的网络。
    """
    
    def __init__(self, in_features: int = 400, hidden_features: int = 256, out_features: int = 400, 
                 num_layers: int = 3, dropout: float = 0.1, residual: bool = True):
        """初始化图卷积网络"""
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        
        # 输入层
        self.layers.append(GraphConvLayer(in_features, hidden_features, dropout))
        
        # 隐藏层
        for _ in range(num_layers):
            self.layers.append(GraphConvLayer(hidden_features, hidden_features, dropout))
        
        # 输出层
        self.layers.append(GraphConvLayer(hidden_features, out_features, dropout))
        
        # 如果使用残差连接且输入输出维度不同，添加投影层
        if residual and in_features != out_features:
            self.residual_projection = nn.Linear(in_features, out_features)
        else:
            self.residual_projection = None
        
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 保存输入以用于残差连接
        identity = x
        
        # 应用图卷积层
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        
        # 添加残差连接（如果启用）
        if self.residual:
            if self.residual_projection is not None:
                identity = self.residual_projection(identity)
            
            # 确保残差连接的形状与输出匹配
            if identity.size(-1) == x.size(-1):
                x = x + identity
        
        # 应用层归一化
        x = self.layer_norm(x)
        
        return x


class EmbeddingNetwork(nn.Module):
    """用于对比学习的嵌入网络"""
    
    def __init__(self, input_dim: int = 400, hidden_dim: int = 1024, output_dim: int = 128, dropout: float = 0.3):
        """初始化嵌入网络"""
        super(EmbeddingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(dropout),  # 添加dropout
            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)  # 添加批归一化
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 处理批量大小为1的情况
        if x.size(0) == 1:
            x_temp = x.repeat(2, 1)
            x_temp = self.network(x_temp)
            return x_temp[:1]
            
        return self.network(x)