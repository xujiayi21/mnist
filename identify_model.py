#对于手写数据集mnist进行训练和测试 图片 1*28*28
import torchvision
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from model import *
#定义超参数
BATCHSIZE=64
LR=0.01
epoch=10
DEVICE=torch.device('cuda' if torch.cuda.is_available() else'cpu') #尽量用gpu跑，没有或者没空的话使用cpu进行
#获取数据
train_data=torchvision.datasets.MNIST('./data',train=True,transform=torchvision.transforms.ToTensor(),
                                      download=True)
test_data=torchvision.datasets.MNIST('./data',train=False,transform=torchvision.transforms.ToTensor(),
                                      download=True)
print(len(train_data))
print(len(test_data))
#加载数据
train_dataloader=DataLoader(train_data,batch_size=BATCHSIZE)
test_dataloader=DataLoader(test_data,batch_size=BATCHSIZE)

#模型定义
identify=Identify()
identify=identify.to(DEVICE)#尝试使用gpu

#定义损失函数和优化器
Crossloss=CrossEntropyLoss()
Crossloss=Crossloss.to(DEVICE)
optimizer=optim.SGD(identify.parameters(),lr=LR)

#开始训练模型
step_nums=0
identify.train()
for i in range(epoch):

    total_loss = 0
    print('------------开始第{}轮训练----------'.format(i+1))
    for data in train_dataloader:
        imgs, target = data
        imgs = imgs.to(DEVICE)
        target = target.to(DEVICE)
        output = identify(imgs)
        loss = Crossloss(output, target)
        total_loss=total_loss+loss

        # 优化更新参数
        optimizer.zero_grad()#梯度规律
        loss.backward()
        optimizer.step()
        step_nums=step_nums+1
        if step_nums%200==0:
            print('第{}次训练损失为{}'.format(step_nums,loss))
    print('训练集上第{}轮训练的总损失为{}'.format(i+1, total_loss))
    identify.eval()
    with torch.no_grad():
        print('---------------测试集开始第{}轮测试-------------'.format(i+1))
        total_loss=0
        total_accuracy=0
        for data in test_dataloader:
            imgs, target = data
            imgs=imgs.to(DEVICE)
            target=target.to(DEVICE)
            output = identify(imgs)
            loss = Crossloss(output, target)
            accuracy=(output.argmax(1)==target).sum()
            total_accuracy=total_accuracy+accuracy
            total_loss = total_loss + loss
        print('测试集上第{}轮训练的总损失为{}'.format(i+1, total_loss))
        print('测试集上第{}轮训练的精确率为{}'.format(i+1,total_accuracy.item()/len(test_data) ))
    torch.save(identify,"./identify_model{}".format(i))

