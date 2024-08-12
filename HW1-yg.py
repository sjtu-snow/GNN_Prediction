import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# 加载Cora数据集
dataset = Planetoid(root='', name='Cora')
data = dataset[0]

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 定义GAT模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 定义GraphSage模型
class GraphSage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 定义模型、优化器和训练过程
class Model():
    def __init__(self, model_type, data, in_channels, hidden_channels, out_channels, heads, lr, weight_decay, epochs):
        self.data = data
        if model_type == 'GCN':
            self.model = GCN(in_channels, hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.model = GAT(in_channels, hidden_channels, out_channels, heads)
        elif model_type == 'GraphSage':
            self.model = GraphSage(in_channels, hidden_channels, out_channels)
        else:
            raise ValueError("Unsupported model type")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self):
        self.model.eval()
        logits, accs = self.model(self.data), []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs
    
    def evaluate(self):
        start_time = time.time()
        for epoch in range(self.epochs):
            loss = self.train()
            train_acc, val_acc, test_acc = self.test()
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        end_time = time.time()
        return test_acc, end_time - start_time

# 初始化模型参数
in_channels, hidden_channels, out_channels, heads, lr, weight_decay, epochs = dataset.num_node_features, 16, dataset.num_classes, 1, 0.01, 5e-4, 200

# 评估三个模型
model_types = ['GCN', 'GAT', 'GraphSage']
results = {}
for model_type in model_types:
    model = Model(model_type, data, in_channels, hidden_channels, out_channels, heads, lr, weight_decay, epochs)
    test_acc, time_cost = model.evaluate()
    results[model_type] = {'test_acc': test_acc, 'time_cost': time_cost}

# 输出最终结果
print('------------------Final Results------------------')
for model_type, result in results.items():
    print(f'{model_type} - Accuracy: {result["test_acc"]:.4f}, Time Cost: {result["time_cost"]:.4f}s')