import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import argparse

from model.resnet import resnet101
from dataset.My_Test import MyTest

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


test_set = MyTest('./data/MyImage/', transform=True, test=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers)
model = resnet101(pretrained=True)
model.fc = nn.Linear(2048, 2)
model.load_state_dict(torch.load('./ckp/model.pth'))
model.cuda()
model.eval()


def main():
    with torch.no_grad():
        for image in test_loader:
            image = Variable(image.cuda())
            out = model(image)
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.data.cpu().numpy().tolist()
            print(predicted)


if __name__ == '__main__':
    main()
