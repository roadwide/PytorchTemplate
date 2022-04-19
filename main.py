'''
Author: RoadWide
Date: 2022-04-14 19:03:57
LastEditTime: 2022-04-19 21:27:04
FilePath: /PytorchTemplate/main.py
Description: 
'''
import torch
import torch.nn.functional as F
import torch.optim as optim
from Model import Net
from DataSet import DataSet


batch_size = 256
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    DataSet('dataset.npz', train=True),
    batch_size = batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
        DataSet('dataset.npz', train=False),
        batch_size=batch_size, shuffle=True)

model = Net().to(device)
optimizer = optim.Adam(model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_num = batch_idx + 1
        total += len(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_num % 30 == 0 or batch_num == len(train_loader)):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, total, len(train_loader.dataset),
                100. * batch_num / len(train_loader), loss.item()))
    print(f'\nTrain Epoch: {epoch} Train Set Accuracy: {correct}/{len(train_loader.dataset)} ({correct/len(train_loader.dataset):.2%})')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1,epochs+1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

print("储存模型...")
model.save("savemodel.ckpt")
print("加载模型...")
model2 = Net().to(device)
model2.load("savemodel.ckpt")
test(model2, device, test_loader)