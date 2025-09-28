import torch
import torch.nn as nn
import torch.optim as optim
from mnist_datasets import training_dataset, testing_dataset


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, batch):
        """This function describes how data flows through our network"""
        flat = batch.view(batch.size(0), -1)
        hidden = self.relu(self.fc1(flat))
        output = self.fc2(hidden)

        return output


model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def evaluate_model():
    """Test how well our network performs on images it has never seen before"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testing_dataset:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def train_model():
    """Teach our network to recognize digits by showing it many examples"""
    model.train()

    running_loss = 0.0

    for i, (images, labels) in enumerate(training_dataset):
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 20 == 0:
            avg_loss = running_loss / (i + 1)
            accuracy = evaluate_model()
            print(
                f"Batch {i + 1}/{len(training_dataset)} - Loss: {loss.item():.4f} (avg {avg_loss:.4f}) - Test Acc: {accuracy:.2f}%"
            )


def main():
    """Main program: Test untrained network, train it, then test again"""
    print(f"Untrained accuracy: {evaluate_model():.2f}%")
    print("Now training the network...\n")

    train_model()

    final_accuracy = evaluate_model()
    print(f"\nFinal trained accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
