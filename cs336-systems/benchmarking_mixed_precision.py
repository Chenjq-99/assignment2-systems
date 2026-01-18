import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

class ToyModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=10):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out_fc1 = self.fc1(x)
        print(f"fc1 输出数据类型: {out_fc1.dtype}")
        
        out_relu = self.relu(out_fc1)
        out_ln = self.ln(out_relu)
        print(f"LayerNorm 输出数据类型: {out_ln.dtype}")
        
        logits = self.fc2(out_ln)
        print(f"模型最终logits数据类型: {logits.dtype}")
        return logits

device = torch.device("cuda")
print(f"使用设备: {device}")

input_dim = 1024
batch_size = 32
output_dim = 10

model = ToyModel(input_dim=input_dim, output_dim=output_dim).to(device)
print(f"模型参数（fc1权重）存储类型: {model.fc1.weight.dtype}\n")

x = torch.randn(batch_size, input_dim).to(device)
y = torch.randint(0, output_dim, (batch_size,)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

scaler = GradScaler('cuda')

def train_step(model, x, y, optimizer, criterion, scaler):
    model.train()
    optimizer.zero_grad()
    
    with autocast(device_type=device.type):
        logits = model(x)
        loss = criterion(logits, y) 
    
    print(f"损失值数据类型: {loss.dtype}\n")
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"fc1 权重梯度数据类型: {model.fc1.weight.grad.dtype}")
    return loss

loss = train_step(model, x, y, optimizer, criterion, scaler)
print(f"\n单步训练完成，损失值: {loss.item():.4f}")