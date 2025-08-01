import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from dyunet1 import DyUNet

model = DyUNet(in_channels=3, out_channels=1).cuda()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 假数据测试
x = torch.randn(10, 3, 256, 256).cuda()
y = torch.randint(0, 2, (10, 1, 256, 256)).float().cuda()
loader = DataLoader(TensorDataset(x, y), batch_size=2)

for epoch in range(2):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
