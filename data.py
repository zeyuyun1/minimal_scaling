import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder


class SubsetWithTransform(Dataset):
    """Wraps a Subset so you can apply a different transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 3,
        img_size: int = 64,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        grayscale: bool = False,
        no_resize: bool = False,
        random_crop: bool = False,
    ):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.img_size    = img_size
        self.val_split   = val_split
        self.test_split  = test_split
        self.seed        = seed
        self.grayscale   = grayscale
        self.no_resize   = no_resize
        self.random_crop = random_crop

        # define transforms
        if self.no_resize:
            common_pre = []
        else:
            # If random_crop is enabled, we still resize the shorter side to img_size
            # to ensure the crop fits, then apply RandomCrop.
            if self.random_crop:
                common_pre = [
                    # transforms.Resize(self.img_size),
                    transforms.RandomCrop(self.img_size),
                ]
            else:
                common_pre = [
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),
                ]
        grayscale_tf = [transforms.Grayscale(num_output_channels=1)] if self.grayscale else []
        to_tensor_tf = [transforms.ToTensor()]

        self.train_transform = transforms.Compose(common_pre + grayscale_tf + to_tensor_tf)
        self.val_transform   = transforms.Compose(common_pre + grayscale_tf + to_tensor_tf)

    def prepare_data(self):
        # no download step for a local ImageFolder
        pass

    def setup(self, stage=None):
        # only split once
        if not hasattr(self, "train_dataset"):
            full = ImageFolder(self.data_dir, transform=None)
            total = len(full)
            val_len  = int(total * self.val_split)
            test_len = int(total * self.test_split)
            train_len = total - val_len - test_len

            generator = torch.Generator().manual_seed(self.seed)
            train_sub, val_sub, test_sub = random_split(
                full,
                [train_len, val_len, test_len],
                generator=generator
            )

            # wrap subsets to apply the correct transform
            self.train_dataset = SubsetWithTransform(train_sub, self.train_transform)
            self.val_dataset   = SubsetWithTransform(val_sub,   self.val_transform)
            self.test_dataset  = SubsetWithTransform(test_sub,  self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True,num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)