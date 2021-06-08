
import os
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from models import *
from util import *
import time
from torchvision import transforms as tfs
import numpy as np

import time


# def main():
parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10 Distributed Training')
parser.add_argument("opts",
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--data', metavar='DIR', type=str,
                    help='path to dataset',default="./datasets/cifar10" )

parser.add_argument('--epochs',
                    default=10,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log_file_name',
                    metavar='NAME',
                    type=str,
                    default='test-cifar10',
                    help='log name')
parser.add_argument('--repeat',
                    metavar='R',
                    type=int,
                    default=1,
                    help='training repeat times')
parser.add_argument('--start_lr',
                    metavar='FACTOR',
                    type=float,
                    default=0.05,
                    help='start lr')
parser.add_argument('--model',
                    metavar='FACTOR',
                    type=str,
                    default='resnet',
                    help='training model')

parser.add_argument('--batch_size',
                    metavar='R',
                    type=int,
                    default=128,
                    help='batch size for training')


parser.add_argument('--workers',
                    metavar='R',
                    type=int,
                    default=2,
                    help='data loading workers')


args = parser.parse_args()



num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1


if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
    
if args.distributed and args.local_rank==0:
    log_file_name = args.log_file_name
    dirs = './logs/' + log_file_name
    if not os.path.exists(dirs):
        os.makedirs(dirs) 
    logger = get_log('./logs/' + log_file_name + '/' + 'train_log.txt')
synchronize()

best_acc=0.0

def evaluteTop1(net, loader, criterion):
    net.eval()  # self.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0.0
    total = len(loader.dataset)
    running_loss = 0.0  
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.no_grad():
            logits = net(x)
            loss = criterion(logits, y.long())
            running_loss += loss.mean().item()
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
    ### add the information about the test loss
    return correct / total


def evaluteTop5(net, loader, criterion):
    net.eval()
    correct = 0.0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.no_grad():
            logits = net(x)
            maxk = max((1, 5))
            y_resize = y.view(-1, 1) 
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total


def test(net, loader, criterion, validation):
    if validation: 
        top1 = evaluteTop1(net, loader, criterion)
        top5 = evaluteTop5(net, loader, criterion)
        if args.local_rank==0:
            logger.info("-------- validation ---------")
            logger.info("top1:%.4f   top5:%.4f" % (top1, top5))

    else:
        top1 = evaluteTop1(net, loader, criterion)
        top5 = evaluteTop5(net, loader, criterion)
        if args.local_rank==0:
            logger.info("---------- test -----------")
            logger.info("top1:%.4f   top5:%.4f" % (top1, top5))
    return top1, top5


def evaluate(net, testloader, criterion, best_acc):
    top1=0.0
    top1, top5 = test(net, testloader, criterion, False)
    if top1 > best_acc:
        # torch.save(net.state_dict(), PATH_best)
        best_acc = top1
        if args.local_rank==0:
            logger.info("best acc:%.4f" % (best_acc))
    return top1, top5, best_acc


def train(local_rank, distributed):
    global best_acc

    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)

    # load the model
    if args.model == 'mobilenet':
        net = MobileNetV2()
    elif args.model == 'resnet':
        net = ResNet18()
    elif args.model == 'vgg':
        net = VGG('VGG16')

    net.cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.start_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    MILESTONES = [30, 60]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=MILESTONES,
                                                     gamma=0.1)
    synchronize()
    # Wrap the model
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # test to false
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,find_unused_parameters=True
        )
    # Data loading code
    trainset = torchvision.datasets.CIFAR10(
        root=args.data,
        train=True,
        transform=tfs.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        download=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=train_sampler)
       
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data,
                                           train=False,
                                           download=True,
                                           transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)  

  
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch) 
        net.train()
        for i, (inputs, labels) in enumerate(trainloader, 0):

            inputs = inputs.reshape(-1, 3, 32, 32)
            inputs = inputs.cuda(non_blocking=True)  ###64,6,32,32
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.mean().backward()
            optimizer.step()
            
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()
            if i % 100 == 0:
                avg_loss = reduce_mean(loss.mean())
                avg_accuracy = reduce_mean(accuracy.mean())
                # print("avg_loss:{},avg_accuracy:{}".format(avg_loss, avg_accuracy))
                if local_rank == 0:
                    logger.info(
                        '[%d, %5d] loss: %.4f Acc: %.4f  lr:%.6f' %
                        (epoch + 1, i + 1,
                         avg_loss.item(), avg_accuracy.item(),
                         optimizer.state_dict()['param_groups'][0]['lr']))

    
        # works fine
        # top1, top5, best_acc =  evaluate(net, testloader, criterion, best_acc)
        
        # deadlock
        if args.local_rank == 0:
            top1, top5 = test(net, testloader, criterion, False)
            top1 = torch.tensor(top1).cuda(args.local_rank)
        else:
            top1 = torch.tensor(0.).cuda(args.local_rank)
        torch.distributed.broadcast(top1,src=0, async_op=False)
        print("local rank:{}, top1:{}".format(args.local_rank, top1.item()))

        scheduler.step()

    

if __name__ == '__main__':

    try:
        train(args.local_rank, args.distributed)
    except Exception as e:
        if args.local_rank==1:
            logger.error(e)
    
