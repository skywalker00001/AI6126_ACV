from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from PIL import Image

from log import logger

'''
    mode="train/val/test"
    img.shape: torch.Size([3, 512, 512])
    label.shape: torch.Size([1, 512, 512])
'''
class FaceParse_Dataset(Dataset):
    def __init__(self, img_path, label_path, transform_img, transform_label, mode="train"): 
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()
        if mode == "train":
            self.num_images = len(self.train_dataset)
        elif mode == "val":
            self.num_images = len(self.val_dataset)
        else :
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            if self.mode == "train":
                label_path = os.path.join(self.label_path, str(i)+'.png')
                self.train_dataset.append([img_path, label_path])
            elif self.mode == "val":
                label_path = os.path.join(self.label_path, str(i)+'.png')
                self.val_dataset.append([img_path, label_path])
            elif self.mode == "test":
                self.test_dataset.append(img_path)
        logger.info(f'Finished preprocessing the CelebA dataset in {self.mode} mode...')

    def __getitem__(self, index):
        if self.mode == "test":
            dataset = self.test_dataset
            img_path = dataset[index]
            image = Image.open(img_path)
            return self.transform_img(image)
        else: 
            dataset = self.train_dataset if self.mode == "train" else self.val_dataset
            img_path, label_path = dataset[index]
            image = Image.open(img_path)
            label = Image.open(label_path)
            return self.transform_img(image), self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(T.CenterCrop(160))
        if resize:
            options.append(T.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(T.ToTensor())
        if normalize:
            options.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = T.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(T.CenterCrop(160))
        if resize:
            options.append(T.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(T.ToTensor())
        if normalize:
            options.append(T.Normalize((0, 0, 0), (0, 0, 0)))
        transform = T.Compose(options)
        return transform

    def loader(self):
        transform_img = self.transform_img(True, True, True, False) 
        transform_label = self.transform_label(True, True, False, False)  
        dataset = FaceParse_Dataset(self.img_path, self.label_path, transform_img, transform_label, self.mode)

        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            shuffle=(self.mode=="train"),
                            num_workers=2,
                            drop_last=False)
        return loader