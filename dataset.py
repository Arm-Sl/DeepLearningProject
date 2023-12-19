from typing import List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    # SETUP DATASET MINST AVEC ROTATION
    def setup(self, stage: Optional[str] = None) -> None:
#       ========================ROT MNIST=======================================
        # DataSet d'entrainement MNIST
        train_transforms = transforms.Compose([
                                                transforms.Resize(self.patch_size),
                                                transforms.ToTensor()])
        self.train_dataset = MNIST(
                root="MNIST/raw/train-images-idx3-ubyte",
                train=True,
                download=True,
                transform = train_transforms)
        
        # DataSet de test MNIST
        val_transforms = transforms.Compose([
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor()])
        self.val_dataset = MNIST(
                root="MNIST/raw/train-images-idx3-ubyte",
                train=False,
                download=True,
                transform = val_transforms)
        
        # DataSet d'entrainement MNIST avec une rotation aléatoire entre -90 et 90
        train_transforms_rot = transforms.Compose([
                                                transforms.RandomRotation(degrees=90),
                                                transforms.Resize(self.patch_size),
                                                transforms.ToTensor()])
        self.train_dataset_rot = MNIST(
                root="MNIST/raw/train-images-idx3-ubyte",
                train=True,
                download=True,
                transform = train_transforms_rot)
        
        # DataSet de test MNIST avec une rotation aléatoire entre -90 et 90
        val_transforms_rot = transforms.Compose([
                                                transforms.RandomRotation(degrees=90),
                                                transforms.Resize(self.patch_size),
                                                transforms.ToTensor()])
        self.val_dataset_rot = MNIST(
                root="MNIST/raw/train-images-idx3-ubyte",
                train=False,
                download=True,
                transform = val_transforms_rot)

        # Dataset concaténé entre MNIST et Rotated MNIST
        self.train_dataset_concat = ConcatDataset([self.train_dataset, self.train_dataset_rot])
        self.val_dataset_concat = ConcatDataset([self.val_dataset, self.val_dataset_rot])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset_concat,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset_concat,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset_concat,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     