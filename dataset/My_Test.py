import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T


class MyTest(data.Dataset):
    def __init__(self, root, transform=True, test=False):
        self.test = test
        self.transform = transform
        if self.test:
            images = [os.path.join(root, img) for img in os.listdir(root)]

        self.images = images

        if self.transform:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # 测试集
            if self.test:
                self.transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), normalize])

    def __getitem__(self, index):
        img_path = self.images[index]
        img_data = Image.open(img_path)
        img_data = self.transform(img_data)
        return img_data

    def __len__(self):
        return len(self.images)
