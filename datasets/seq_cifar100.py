
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.incremental_dataset import getfeature_loader, IncrementalDataset, get_feature_extractor
from datasets.utils.incremental_dataset import get_previous_train_loader
from argparse import Namespace
from datasets.transforms.denormalization import DeNormalize


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        self.target_transform = target_transform
        self.attributes = []
        self.trans = []
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download=True)

    def set_att(self, att_name, att_data, att_transform=None):
        self.attributes.append(att_name)
        self.trans.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = (img, target, not_aug_img)
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]

            trans = self.trans[i]
            if trans:
                att_data = trans(att_data)

            ret_tuple += (att_data,)

        return ret_tuple


class SequentialCIFAR100(IncrementalDataset):
    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.nc = 100
        self.nt = 10
        self.n_channel = 3
        self.n_imsize1 = 32
        self.n_imsize2 = 32
        super(SequentialCIFAR100, self).__init__(args)

        if self.args.featureNet:
            self.args.transform = 'pytorch'
            self.extractor = get_feature_extractor(args)

        if self.args.transform == 'pytorch':
            self.normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.dnormalization_transform = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalization_transform])
            self.test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                self.normalization_transform])
        else:
            self.normalization_transform = None
            self.dnormalization_transform = None
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            self.test_transform = transforms.Compose([transforms.ToTensor()])

    def get_data_loaders(self):

        train_dataset = MyCIFAR100(self.args.root + 'CIFAR100', train=True, transform=self.train_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        self.test_transform, self.NAME)
        else:
            test_dataset = MyCIFAR100(self.args.root + 'CIFAR100', train=False, transform=self.test_transform)

        train, test = getfeature_loader(train_dataset, test_dataset, setting=self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(), self.normalization_transform])

        train_dataset = MyCIFAR100(self.args.root + 'CIFAR10', train=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.train_transform])
        return transform