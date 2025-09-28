# Neural Network Training Script for MNIST Digit Recognition
# This program trains an artificial neural network to recognize handwritten digits (0-9)
# from the famous MNIST dataset used in machine learning education.

# Import the PyTorch library - a popular tool for building neural networks
import torch
import torch.nn as nn       # Neural network building blocks (layers, activations)
import torch.optim as optim # Optimization algorithms (how the network learns)
from mnist_datasets import training_dataset, testing_dataset


# Define our neural network architecture (the "brain" that will learn to recognize digits)
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Layer 1: Takes 784 inputs (28x28 pixel image flattened) and outputs 16 numbers
        # Think of this as 16 "feature detectors" that each look for different patterns
        self.fc1 = nn.Linear(784, 16)
        
        # Layer 2: Takes those 16 numbers and outputs 10 final scores (one for each digit 0-9)
        self.fc2 = nn.Linear(16, 10)
        
        # ReLU activation: Introduces non-linearity to the network
        # Without this, our network could only learn straight-line relationships
        # ReLU keeps positive numbers unchanged, sets negative numbers to 0
        # This simple function allows the network to learn complex, curved patterns
        self.relu = nn.ReLU()

    def forward(self, batch):
        """This function describes how data flows through our network"""
        # Step 1: Flatten the 28x28 images into a single row of 784 numbers
        # (Neural networks work with 1D lists of numbers, not 2D images directly)
        flat = batch.view(batch.size(0), -1)
        
        # Step 2: Send the flattened image through the first layer, then apply ReLU
        hidden = self.relu(self.fc1(flat))
        
        # Step 3: Send the result through the second layer to get final predictions
        # The output will be 10 numbers - higher numbers mean "more confident this is that digit"
        output = self.fc2(hidden)
        return output


# Create our neural network - like building the "brain" that will learn
model = MNISTNet()

# Define how we measure "wrongness" - CrossEntropyLoss is good for classification
# It converts raw network outputs to probabilities, then penalizes incorrect predictions
# The penalty is logarithmic: being confidently wrong hurts much more than being uncertain
criterion = nn.CrossEntropyLoss()

# Define the learning algorithm - Adam automatically adapts learning rates
# Unlike simple methods, Adam tracks each parameter individually and adjusts
# how much to change each one based on recent learning history
# lr=0.001 sets the base learning rate (Adam will fine-tune from there)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def evaluate_model():
    """Test how well our network performs on images it has never seen before"""
    # Put the model in "evaluation mode" - turns off learning-specific behaviors
    model.eval()
    correct = 0  # Count how many we get right
    total = 0    # Count total number of test images

    # torch.no_grad() tells PyTorch "we're just testing, don't prepare for learning"
    # This saves memory and makes testing faster
    with torch.no_grad():
        # Go through each batch of test images
        for images, labels in testing_dataset:
            # Ask our network: "What digit do you think this is?"
            outputs = model(images)
            
            # Find which digit got the highest score for each image
            # torch.max returns (highest_value, position_of_highest_value)
            # We only care about the position (which digit), so we use _ for the value
            _, predicted = torch.max(outputs.data, 1)
            
            # Keep track of our statistics
            total += labels.size(0)  # How many images in this batch
            correct += (predicted == labels).sum().item()  # How many we got right

    # Calculate percentage correct
    accuracy = 100 * correct / total
    return accuracy


def train_model():
    """Teach our network to recognize digits by showing it many examples"""
    # Put the model in "training mode" - enables learning-specific behaviors
    model.train()

    running_loss = 0.0  # Keep track of how "wrong" our guesses are, on average

    # Go through the training dataset in batches (small groups of images)
    # Like studying flashcards in small stacks rather than one giant pile
    # This is both organizationally manageable and computationally efficient -
    # modern hardware (GPUs) can process multiple images simultaneously
    for i, (images, labels) in enumerate(training_dataset):
        
        # STEP 1: Clear previous learning
        # Like erasing a whiteboard before solving a new math problem
        optimizer.zero_grad()

        # STEP 2: Make predictions ("Forward Pass")
        outputs = model(images)  # Ask network: "What digits do you see?"
        
        # STEP 3: Calculate how wrong we were
        # Compare our guesses (outputs) to the correct answers (labels)
        loss = criterion(outputs, labels)
        
        # STEP 4: Learn from mistakes ("Backward Pass")
        # This calculates how to adjust each part of the network to do better next time
        loss.backward()
        
        # STEP 5: Actually update the network based on what we learned
        # Like a student updating their approach after seeing the right answer
        optimizer.step()

        # Keep track of our progress
        running_loss += loss.item()

        # Print progress report every 20 batches (so we can see if we're improving)
        if (i + 1) % 20 == 0:
            avg_loss = running_loss / (i + 1)  # Average "wrongness" so far
            accuracy = evaluate_model()        # Test on images we haven't trained on
            print(
                f"Batch {i + 1}/{len(training_dataset)} - Loss: {loss.item():.4f} (avg {avg_loss:.4f}) - Test Acc: {accuracy:.2f}%"
            )


def main():
    """Main program: Test untrained network, train it, then test again"""
    
    # Before training: See how well our "untrained" network does
    # Should be around 10% (random guessing for 10 digits)
    print(f"Untrained accuracy: {evaluate_model():.2f}%")
    print("Now training the network...\n")

    # The main event: Train our network on thousands of digit examples
    train_model()

    # After training: See how much we improved
    final_accuracy = evaluate_model()
    print(f"\nFinal trained accuracy: {final_accuracy:.2f}%")
    print("The network learned to recognize digits!")


# This is where the program actually starts running
# Like the "main()" function in other programming languages
if __name__ == "__main__":
    main()
