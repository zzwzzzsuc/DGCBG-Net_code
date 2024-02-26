from random import shuffle
import torch
from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, petlist, labelList, boundList, mode = 'train'):
        self.imList = imList
        self.petlist = petlist
        self.labelList = labelList
        self.boundList = boundList
        self.mode = mode
        self.shuffle = shuffle

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):

        image_name = self.imList[idx]
        pet_name = self.petlist[idx]
        label_name = self.labelList[idx]
        bou_name = self.boundList[idx]

        image = Image.open(image_name)
        pet = Image.open(pet_name)
        label = Image.open(label_name)
        bou = Image.open(bou_name)

        Transform = []
        Transform.append(T.Resize((256, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        pet = Transform(pet)
        label = Transform(label)
        bou = Transform(bou)

        image = image.float()
        pet = pet.float()
        label = label.float()
        bound = bou.float()

        Norm_ = T.Normalize(([0.5]), ([0.5]))
        # Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        pet = Norm_(pet)

        return image, pet, label, bound


def get_loader(imList, petlist, labelList, boundList, batch_size, num_workers=1, mode='train', drop_last=True):
    """Builds and returns Dataloader."""

    dataset = MyDataset(imList, petlist, labelList, boundList, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader