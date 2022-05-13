'''
Author: RoadWide
Date: 2022-04-14 19:03:57
LastEditTime: 2022-05-13 11:02:31
FilePath: /PytorchTemplate/main.py
Description: 
'''
import torch
import torch.nn.functional as F
import torch.optim as optim
from Model import Net
from DataSet import DataSet
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=10)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    # 设置随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1234)

train_loader = torch.utils.data.DataLoader(
    DataSet('dataset.npz', train=True),
    batch_size = args.batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
        DataSet('dataset.npz', train=False),
        batch_size=args.batch_size, shuffle=True)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# 每个epoch更新lr, 新的lr 为 初始lr / (epoch + 1), 例如lr = 0.1  0.05  0.033  0.025
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    loop = tqdm(train_loader, total =len(train_loader))
    for data, target in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss = loss.item(),acc = f"{correct/len(train_loader.dataset):.2%}")
    lr_scheduler.step()    # lr的调整应该在每个epoch结束之后，而不是每个batch结束之后

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

for epoch in range(1,args.epochs+1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

print("储存模型...")
model.save("savemodel.ckpt")
print("加载模型...")
model2 = Net().to(device)
model2.load("savemodel.ckpt")
test(model2, device, test_loader)