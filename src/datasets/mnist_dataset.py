import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, data_dir: str = "./data", train: bool = True) -> None:
        super().__init__()
        # Keep MNIST in [0, 1] so background pixels have zero norm
        transform = transforms.ToTensor()

        self.dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        # Return only the image tensor, discarding the label
        img, _ = self.dataset[index]
        return img

    def __len__(self) -> int:
        return len(self.dataset)
