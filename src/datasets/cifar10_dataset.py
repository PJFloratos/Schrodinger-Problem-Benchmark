import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir: str = "./data", train: bool = True) -> None:
        super().__init__()
        # Standardize CIFAR-10 to [-1, 1] to match the Gaussian prior.
        # We pass three 0.5s for the mean and std because CIFAR-10 has 3 (RGB) channels.
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        # Return only the image tensor, discarding the label
        img, _ = self.dataset[index]
        return img

    def __len__(self) -> int:
        return len(self.dataset)
