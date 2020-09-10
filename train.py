import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import os
import argparse
from model.resnet import resnet101
from dataset.GirlBoy import GirlBoy

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=8)
parser.add_argument('--epochs', type=int, default=21)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=str, default='0')
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

train_set = GirlBoy('./data/train/', train=True)
test_set = GirlBoy('./data/test/', test=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers)

model = resnet101(pretrained=True)
model.fc = nn.Linear(2048, 2)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=3)
criterion = nn.CrossEntropyLoss()
criterion.cuda()


def train(epoch):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	model.train()
	for batch_idx, (img, label) in enumerate(train_loader):
		image = Variable(img.cuda())
		label = Variable(label.cuda())
		optimizer.zero_grad()
		out = model(image)
		loss = criterion(out, label)
		loss.backward()
		optimizer.step()
		print("Epoch:%d [%d|%d] loss:%f" %(epoch, batch_idx, len(train_loader), loss.mean()))


def test(epoch):
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total = 0
	correct = 0
	with torch.no_grad():
		for batch_idx, (img, label) in enumerate(test_loader):
			image = Variable(img.cuda())
			label = Variable(label.cuda())
			out = model(image)
			_, predicted = torch.max(out.data, 1)
			total += image.size(0)
			correct += predicted.data.eq(label.data).cpu().sum()
	print("Acc: %f "% ((1.0*correct.numpy())/total))


def main():
	for epoch in range(opt.epochs):
		train(epoch)
		test(epoch)
	torch.save(model.state_dict(), 'ckp/model.pth')


if __name__ == '__main__':
	main()
