# MNIST Neural Network Example

A beginner-friendly neural network implementation that learns to recognize handwritten digits (0-9) from the famous MNIST dataset. This project is designed for college students with no computer science background to understand how artificial neural networks work.

## What This Program Does

This neural network:

- Takes 28×28 pixel images of handwritten digits
- Learns patterns from 60,000 training examples
- Can then predict what digit is shown in new images it has never seen
- Achieves ~85-90% accuracy with a simple 2-layer network

## Prerequisites

- **Python 3.8+** (The programming language this is written in)
- **uv** (A fast Python package manager - we'll install this first)

### Installing uv

uv is a modern, fast Python package manager. Install it with:

**On macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal or run:

```bash
source $HOME/.cargo/env
```

## Installation & Setup

1. **Clone this repository:**

   ```bash
   git clone https://github.com/vabruzzo/hello-mnist-neural-network.git
   cd hello-mnist-neural-network
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```
   This automatically:
   - Creates a virtual environment
   - Installs PyTorch (the neural network library)
   - Installs the Datasets library (for MNIST data)
   - Sets up everything you need

## Running the Program

### Basic Training

```bash
uv run main.py
```

## What You'll See

When you run the program, you'll see output like:

```
Untrained accuracy: 9.85%
Now training the network...

Batch 20/938 - Loss: 1.2458 (avg 1.8234) - Test Acc: 45.23%
Batch 40/938 - Loss: 0.8934 (avg 1.6123) - Test Acc: 62.18%
Batch 60/938 - Loss: 0.5621 (avg 1.4567) - Test Acc: 74.35%
...

Final trained accuracy: 87.42%
The network learned to recognize digits!
```

## Understanding the Output

- **Untrained accuracy (~10%)**: Random guessing (1 out of 10 digits)
- **Loss**: How "wrong" the network's guesses are (lower = better)
- **Test Accuracy**: Percentage correct on images the network has never seen
- **Final accuracy (85-90%)**: Much better than random guessing!

## Project Structure

```
├── main.py              # Core neural network training script
├── main_annotated.py    # Same script with detailed educational comments
├── mnist_datasets.py    # Code to load and prepare the MNIST data
├── pyproject.toml       # Python project configuration
├── README.md            # This file
└── uv.lock              # Locked dependency versions
```

## Key Concepts Explained

### Neural Network Architecture

- **Input Layer**: 784 numbers (28×28 flattened image)
- **Hidden Layer**: 16 "feature detectors"
- **Output Layer**: 10 scores (one for each digit 0-9)

### Training Process

1. **Forward Pass**: Show the network an image, get its prediction
2. **Loss Calculation**: Measure how wrong the prediction was
3. **Backward Pass**: Calculate how to improve each part of the network
4. **Parameter Update**: Actually adjust the network to do better next time
5. **Repeat**: Do this for thousands of examples

### Why It Works

- The network finds patterns in pixels that humans use to recognize digits
- Through repetition, it learns features like curves, lines, and shapes
- Each training example helps it generalize to new, unseen digits

## Customization

You can experiment by editing `main.py`:

- Change network size: `nn.Linear(784, 16)` → `nn.Linear(784, 64)` for more neurons
- Adjust learning rate: `lr=0.001` → `lr=0.01` for faster learning
- Modify batch reporting: `% 20 == 0` → `% 50 == 0` for less frequent updates

## Troubleshooting

### "torch not found" or import errors

```bash
uv sync --refresh
```

### Out of memory

Reduce batch size in `mnist_datasets.py` or use a smaller network.

## Learning More

This is just the beginning! To dive deeper:

- Try different network architectures (add more layers)
- Experiment with other datasets (Fashion-MNIST, CIFAR-10)
- Learn about Convolutional Neural Networks (CNNs) for better image recognition
- Explore other machine learning libraries and frameworks
