'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

def neg_hscore(f,g):
    f0 = f - torch.mean(f,0)
    g0 = g - torch.mean(g,0)
    corr = torch.mean(torch.sum(f0*g0,1))
    cov_f = torch.mm(torch.t(f0),f0) / (f0.size()[0]-1.)
    cov_g = torch.mm(torch.t(g0),g0) / (g0.size()[0]-1.)
    return - corr + torch.trace(torch.mm(cov_f, cov_g)) / 2.

class aceModel_g(nn.Module):
    def __init__(self):
        super(aceModel_g, self).__init__()
        self.fc1 = nn.Linear(10, 512)

    def forward(self, y):
        g = self.fc1(y)
        return g


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

model_ace = ResNet50()
model_nn = ResNet50()
model_f = ResNet50Feature()
model_g = aceModel_g()

model_nn = model_nn.to(device)
model_f = model_f.to(device)
model_g = model_g.to(device)

if device == 'cuda':
    model_nn = torch.nn.DataParallel(model_nn)
    model_f = torch.nn.DataParallel(model_f)
    model_g = torch.nn.DataParallel(model_g)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_nn = optim.Adam(model_nn.parameters(),lr=args.lr)
optimizer_ace = optim.Adam(list(model_f.parameters())+list(model_g.parameters()), lr=args.lr)

# Training
# def train(epoch):
nb_epochs = 100
for epoch in range(nb_epochs):
    print('\nEpoch: %d' % epoch)
    nn_train_loss = 0
    correct = 0
    train_total = 0
    py = np.zeros((1,10))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets_1hot = torch.zeros(len(targets), 10).scatter_(1, targets.resize(len(targets),1), 1)
        inputs, targets, targets_1hot = inputs.to(device), targets.to(device), targets_1hot.to(device)

        optimizer_nn.zero_grad()
        optimizer_ace.zero_grad()
        logits_nn = model_nn(inputs)
        # pred = torch.max(logits_nn,1)[1]
        # acc = (pred==ys).sum()
        # print(xs[-1])
        f = model_f(inputs)
        g = model_g(targets_1hot)
        loss_ace = neg_hscore(f,g)
        loss_ace.backward()
        loss_nn = F.cross_entropy(logits_nn, targets)
        loss_nn.backward() 
        optimizer_nn.step()
        optimizer_ace.step() 

        nn_train_loss += loss_nn.item()
        _, predicted = logits_nn.max(1)
        train_total += len(inputs)
        correct += predicted.eq(targets).sum().item()

        py = py + torch.sum(targets_1hot,0).cpu().numpy()
        train_total += len(inputs)


        # optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (nn_train_loss/(batch_idx+1), 100.*correct/train_total, correct, train_total))

        print('Epoch {} finished.'.format(epoch))
        print("loss_nn:{},loss_ace:{}".format(loss_nn,loss_ace))
        py = py.astype(float) / train_total

        test_total = 0
        correct_ace = 0
        correct_nn = 0
        if torch.cuda.is_available():
            eye = torch.eye(10).cuda()
        else:
            eye = torch.eye(10)
        g_test = model_g(eye).data.cpu().numpy()
        g_test = g_test - np.mean(g_test, axis = 0)

        with torch.no_grad():
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                logits_nn = model_nn(inputs).data.cpu().numpy()
                # logits_nn_np = logits_nn.data.cpu().numpy()
                f_test = model_f(inputs).data.cpu().numpy()
                f_test = f_test - np.mean(f_test, axis = 0)

                # py = np.mean(y_train, axis = 0)
                pygx = py * (1 + np.matmul(f_test, g_test.T))
                # ace_acc = np.mean(np.argmax(pygx, axis = 1) == np.argmax(y_test, axis = 1))

                correct_ace += (np.argmax(pygx, axis = 1) == targets).sum()
                correct_nn += (np.argmax(logits_nn, axis=1) == targets).sum()
                total += len(inputs)

                nn_acc = float(correct_nn) / test_total
                ace_acc = float(correct_ace) / test_total

                print('NN test accuracy: %.2f%%' % (nn_acc * 100))
                print('ACE test accuracy: %.2f%%' % (ace_acc * 100))

model_f_dict = model_f.state_dict()
model_ace_dict = model_ace.state_dict()
model_f_dict = {key: value for key, value in model_f_dict.items() if key in model_ace_dict}
if torch.cuda.is_available():
    model_f_dict['linear.weight'] = torch.from_numpy((py * g_test.T).T).cuda()
    model_f_dict['linear.bias'] = torch.from_numpy(py).view(10).cuda()
else:
    model_f_dict['linear.weight'] = torch.from_numpy((py * g_test.T).T)
    model_f_dict['linear.bias'] = torch.from_numpy(py).view(10)           
model_ace_dict.update(model_f_dict)
model_ace.load_state_dict(model_f_dict)

torch.save(model_nn.state_dict(), 'model_nn_pytorch_{}epochs.pkl'.format(nb_epochs))
torch.save(model_ace.state_dict(), 'model_ace_pytorch_{}epochs.pkl'.format(nb_epochs))


# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.t7')
#         best_acc = acc


# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     # test(epoch)
    
