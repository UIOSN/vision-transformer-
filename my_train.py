import torch
import sys
from tqdm import tqdm
import math
import torch.optim as optim
from torchvision import transforms 
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from my_model import create_model
import os
def train_one_epoch(model,optimizer,data_loader,epoch,device):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader,file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num+=images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred,dim=1)[1]
        accu_num+=torch.eq(pred_classes,labels.to(device)).sum()
        loss = loss_function(pred,labels.to(device))
        loss.backward()
        
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               loss,
                                                                               accu_num.item() / sample_num)
        optimizer.step()
        optimizer.zero_grad()
        
    return accu_loss.item()/(step+1),accu_num.item()/sample_num

def evaluate(model,data_loader,device,epoch):
    with torch.no_grad():
        loss_function = torch.nn.CrossEntropyLoss()
        model.eval()
        accu_num = torch.zeros(1).to(device)
        accu_loss = torch.zeros(1).to(device)
        sample_num = 0
        data_loader = tqdm(data_loader,file=sys.stdout)
        for step,data in enumerate(data_loader):
            images,labels = data
            sample_num=images.shape[0]
            pred = model(images.to(device))
            pred_classes = torch.max(pred,dim=1)[1]
            accu_num+=torch.eq(pred_classes,labels.to(device)).sum()
            loss = loss_function(pred,labels.to(device))
            
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                        loss ,
                                                        accu_num.item() / sample_num)
        return accu_loss.item()/(step+1),accu_num.item()/sample_num
        

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop((224,224),scale=(0.05,1.0)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]),
        "val": transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])}
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar10',
                                             train=True,
                                             transform=data_transform['train'],
                                             download=True)
    val_dataset = torchvision.datasets.CIFAR10(root='./cifar10',
                                               train=False,
                                               transform=data_transform['val'],
                                               download=True)
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8,
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8, 
                                            )
    model = create_model(args.num_classes).to(device)#模型要记得todevice！
    root = "D:\\new\\vit\my vit\\weights\\model.pth"
    if os.path.exists(root):
        weights = torch.load(root)
        model.load_state_dict(weights)
    else:
        torch.save(model.state_dict(),root)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    tb_writer = SummaryWriter(log_dir="my vit/records")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model,optimizer,train_loader,epoch,device)
        torch.save(model.state_dict(),root)
        print("epoch {} 's weights have been saved.".format(epoch))
        scheduler.step()
        val_loss,val_acc = evaluate(model,val_loader,device,epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        

        
        






