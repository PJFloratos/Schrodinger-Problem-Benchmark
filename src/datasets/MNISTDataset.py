import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, data_dir: str = "./data", train: bool = True) -> None:
        super().__init__()
        # Standardize MNIST to [-1, 1] to match the Gaussian prior
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        # Return only the image tensor, discarding the label
        img, _ = self.dataset[index]
        return img

    def __len__(self) -> int:
        return len(self.dataset)
