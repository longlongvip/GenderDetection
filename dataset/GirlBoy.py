import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T


class GirlBoy(data.Dataset):
    def __init__(self, root, transform=True, train=False, test=False):
        self.train = train
        self.test = test
        self.transform = transform
        if self.train:
            images = [os.path.join(root, img) for img in os.listdir(root)]
        if self.test:
            images = [os.path.join(root, img) for img in os.listdir(root)]
        self.images = images

        if self.transform:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # 测试集
            if self.test:
                self.transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), normalize])
            # 训练集
            if self.train:
                self.transform = T.Compose(
                    [T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize])

    def __getitem__(self, index):
        img_path = self.images[index]
        if self.test:
            label = 1 if 'female' in img_path.split('/')[-1] else 0
        if self.train:
            label = 1 if 'female' in img_path.split('/')[-1] else 0
        img_data = Image.open(img_path)
        img_data = self.transform(img_data)
        return img_data, label

    def __len__(self):
        return len(self.images)
