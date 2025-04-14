# models/graph_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """图卷积层
    
    执行图卷积操作：H = D^(-1/2) A D^(-1/2) X W
    其中A是邻接矩阵，X是节点特征，D是度矩阵，W是权重。
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        初始化图卷积层
        
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
        """
        super(GraphConvLayer, self).__init__()
        # 可学习的权重矩阵: [in_features, out_features]
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 可学习的偏置: [out_features]
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化权重和偏置"""
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x: 节点特征张量，形状为 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵张量，形状为 [batch_size, num_nodes, num_nodes]
            
        返回:
            更新后的节点特征张量，形状为 [batch_size, num_nodes, out_features]
        """
        # 归一化邻接矩阵
        # 计算行和: [batch_size, num_nodes, 1]
        deg = torch.sum(adj, dim=2, keepdim=True)
        deg = torch.clamp(deg, min=1.0)  # 防止除零
        # 计算D^(-1/2): [batch_size, num_nodes, 1]
        deg_inv_sqrt = deg.pow(-0.5)
        # 计算D^(-1/2) A D^(-1/2): [batch_size, num_nodes, num_nodes]
        adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(1, 2)
        
        # 图卷积操作
        # 计算XW: [batch_size, num_nodes, out_features]
        support = torch.matmul(x, self.weight)
        # 计算D^(-1/2) A D^(-1/2) XW: [batch_size, num_nodes, out_features]
        output = torch.matmul(adj_norm, support)
        
        # 加上偏置: [batch_size, num_nodes, out_features]
        return output + self.bias


class DataStructureAnalyzer(nn.Module):
    """数据结构分析器 (DSA)
    
    生成用于构建图邻接矩阵的结构分数。
    """
    
    def __init__(self, input_dim: int = 400, hidden_dim: int = 256, output_dim: int = 128):
        """
        初始化数据结构分析器
        
        参数:
            input_dim: 输入特征维度，默认为400（更新）
            hidden_dim: 隐藏层维度，默认为256
            output_dim: 输出特征维度，默认为128
        """
        super(DataStructureAnalyzer, self).__init__()
        
        self.network = nn.Sequential(
            # 输入: [batch_size, input_dim]
            # 输出: [batch_size, hidden_dim]
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # 输入: [batch_size, hidden_dim]
            # 输出: [batch_size, hidden_dim]
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 输入: [batch_size, hidden_dim]
            # 输出: [batch_size, output_dim]
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x: 输入特征张量，形状为 [batch_size, input_dim]
            
        返回:
            结构分数张量，形状为 [batch_size, output_dim]
        """
        return self.network(x)


class GraphConvolutionalNetwork(nn.Module):
    """图卷积网络 (GCN)
    
    通过堆叠多个图卷积层构建的网络。
    """
    
    def __init__(self, in_features: int = 400, hidden_features: int = 256, out_features: int = 400, num_layers: int = 2):
        """
        初始化图卷积网络
        
        参数:
            in_features: 输入特征维度，默认为400（更新）
            hidden_features: 隐藏层特征维度，默认为256
            out_features: 输出特征维度，默认为400（更新）
            num_layers: 图卷积层数量
        """
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 输入层
        # 输入: [batch_size, num_nodes, in_features]
        # 输出: [batch_size, num_nodes, hidden_features]
        self.layers.append(GraphConvLayer(in_features, hidden_features))
        
        # 隐藏层
        for _ in range(num_layers):
            # 输入: [batch_size, num_nodes, hidden_features]
            # 输出: [batch_size, num_nodes, hidden_features]
            self.layers.append(GraphConvLayer(hidden_features, hidden_features))
        
        # 输出层
        # 输入: [batch_size, num_nodes, hidden_features]
        # 输出: [batch_size, num_nodes, out_features]
        self.layers.append(GraphConvLayer(hidden_features, out_features))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x: 节点特征张量，形状为 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵张量，形状为 [batch_size, num_nodes, num_nodes]
            
        返回:
            更新后的节点特征张量，形状为 [batch_size, num_nodes, out_features]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        
        return x



class EmbeddingNetwork(nn.Module):
    """用于对比学习的嵌入网络"""
    
    def __init__(self, input_dim: int = 400, hidden_dim: int = 1024, output_dim: int = 128):
        """
        初始化嵌入网络
        
        参数:
            input_dim: 输入特征维度，默认为400（更新）
            hidden_dim: 隐藏层维度，默认为1024
            output_dim: 输出特征维度，默认为128
        """
        super(EmbeddingNetwork, self).__init__()
        self.network = nn.Sequential(
            # 输入: [batch_size, input_dim]
            # 输出: [batch_size, hidden_dim]
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # 输入: [batch_size, hidden_dim]
            # 输出: [batch_size, output_dim]
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x: 输入特征张量，形状为 [batch_size, input_dim]
            
        返回:
            投影后的特征张量，形状为 [batch_size, output_dim]
        """
        return self.network(x)