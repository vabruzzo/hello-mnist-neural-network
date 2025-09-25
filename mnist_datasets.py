import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.ToTensor()


mnist_train_dataset = datasets.load_dataset("mnist", split="train")
mnist_test_dataset = datasets.load_dataset("mnist", split="test")

train_dataset = MNISTDataset(mnist_train_dataset, transform=transform)
test_dataset = MNISTDataset(mnist_test_dataset, transform=transform)

batch_size = 64
training_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testing_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
