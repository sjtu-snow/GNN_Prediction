import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.transforms import NormalizeFeatures

class MyNet(torch.nn.Module):
    def __init__(self, conv_list, func):
        super(MyNet, self).__init__()
        n = len(conv_list) - 1
        assert n >=1
        self.layers = nn.ModuleList()  
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(n-1):  
            self.layers.append( func(conv_list[i], conv_list[i+1]) )  
            self.activations.append(nn.ReLU())  
            self.dropouts.append(nn.Dropout(p=0.1))
        self.layers.append( func(conv_list[n-1], conv_list[n]) )
            
    def forward(self, x, edge_index):
        for layer, activation, dropout in zip(self.layers, self.activations, self.dropouts):  
            x = layer(x, edge_index)  
            x = activation(x)  
            x = dropout(x) 
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)  
    
    
class MyModel():
    def __init__(self, conv_list, func, data, device, optimizer=torch.optim.Adam, loss=F.nll_loss):
        self.device = device
        self.model = MyNet(conv_list, func).to(self.device)
        self.data = data
        self.optimizer = optimizer(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        self.loss = loss
        print(self.model)
        
    # 训练模型
    def train(self):
        x, edge_index, y, mask = self.data.x.to(self.device), self.data.edge_index.to(self.device), self.data.y.to(self.device), self.data.train_mask.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x, edge_index)
        loss = self.loss(out[mask], y[mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    # 测试模型
    def test(self):
        x, edge_index, y = self.data.x.to(self.device), self.data.edge_index.to(self.device), self.data.y
        self.model.eval()
        logits = self.model(x, edge_index).to('cpu')
        accs = []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs
        
    def run(self, epochs=100):
        train_acc, val_acc, test_acc = 0, 0, 0
        for epoch in range(epochs):
            loss = self.train()
            train_acc, val_acc, test_acc = self.test()
            if epoch%10==0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Final Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
 
    
# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora',transform=NormalizeFeatures())
# 加载数据
data = dataset[0]
# 设置层
con_list = [dataset.num_node_features, 163, dataset.num_classes]
# 设置GCN函数
func = GCNConv #GATConv, SAGEConv
# 设置运行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyModel(con_list, func, data, device)
model.run()