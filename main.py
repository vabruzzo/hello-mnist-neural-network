import torch
import torch.nn as nn
import torch.optim as optim
from mnist_datasets import batch_size, training_dataset, testing_dataset


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, batch):
        flat = batch.view(batch.size(0), -1)
        hidden = self.relu(self.fc1(flat))
        output = self.fc2(hidden)
        return output


model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def evaluate_model():
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
    model.train()

    running_loss = 0.0

    for i, (images, labels) in enumerate(training_dataset):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # classify a set of images
        loss = criterion(
            outputs, labels
        )  # calculate how wrong the classifications were

        # Backward pass and optimization
        loss.backward()  # compute gradient for every learnable parameter
        optimizer.step()  # use gradients to update the parameters

        running_loss += loss.item()

        # Print progress every 20 batches
        if (i + 1) % 20 == 0:
            # Compute running average loss so far
            avg_loss = running_loss / (i + 1)
            # Evaluate current test accuracy
            accuracy = evaluate_model()
            print(
                f"Batch {i + 1}/{len(training_dataset)} - Loss: {loss.item():.4f} (avg {avg_loss:.4f}) - Test Acc: {accuracy:.2f}%"
            )


def main():
    # untrained evaluation
    print(f"Untrained accuracy: {evaluate_model():.2f}%")

    train_model()

    # post-training evaluation
    final_accuracy = evaluate_model()
    print(f"\nFinal trained accuracy: {final_accuracy:.2f}%")
    print(f"Improvement: {final_accuracy - 10:.2f} percentage points!")


main()
