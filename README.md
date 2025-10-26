# Tent: Test-Time Adaptation

This project implements the test-time adaptation in Medical imaging. Tent adapts a pre-trained model to a new target domain at test-time by minimizing the entropy of the model's predictions on the target data. This allows the model to improve its performance on the new domain without requiring labeled data.

## Project Summary

**Test-Time Adaptation for Medical Imaging using Controlled Shifts and Cross-Plane Adaptation**

*Author: Harsh Kumar 24576*

### Aim
- To implement and test TENT for 11-class medical imaging classification, addressing challenges from dataset shifts.
- To utilize Entropy and Maximum Mean Discrepancy (MMD) loss for unsupervised test-time learning.

### Setup
- **Model:** Resnet18
- **Parameters:** 33,689,931
- **Layers:** 87
- **Computing Resources:** NVIDIA A5000

### Performance
- **Training Time (25,211 samples):**
    - **Entropy loss:** 25.74 sec
    - **MMD loss:** 238.47 sec
- **Inference Time (25,211 samples):**
    - **Entropy loss:** 0.007418 sec
    - **MMD loss:** 0.007306 sec

### Results
- **Entropy Loss:**
    - **Accuracy:** +15.04% on average
    - **F1 Score:** +0.1397
- **MMD Loss:**
    - **Accuracy:** +15.94% on average
    - **F1 Score:** +0.1787

### Key Findings
- **Within-plane vs. Cross-plane Adaptation:** Cross-plane adaptation shows larger relative improvements, highlighting its value for domain transfer.
- **Noise Impact:** Adaptation effectiveness is correlated with corruption severity.
- **Entropy Adaptation:** Recovers almost original accuracy for Gaussian and blur noise.
- **MMD Adaptation:** Significantly improves accuracy from noisy baselines but doesn't fully recover to clean performance levels. It also requires 10x more training time.
- **MMD and Noise Types:** Gaussian noise achieves the highest accuracy, followed by Blurred noise. Poisson noise remains the most challenging.

## Files

- `tent.py`: The core implementation of the Tent algorithm. It contains the `tent` class which takes a model and an optimizer and adapts the model at test time.
- `train.py`: A script for training a model before applying Tent.
- `experiments.py`: A script to run adaptation experiments using Tent.
- `requirements.txt`: A list of python packages required to run the code.
- `*.ipynb`: Jupyter notebooks for analysis and visualization.
- `*.csv` and `*.txt`: Log files from experiments.

## Usage

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train a model:**
    Use `train.py` to train a source model.

3.  **Adapt the model:**
    Use `experiments.py` or `tent.py` to adapt the trained model to a new target dataset at test-time. The `tent` class in `tent.py` can be integrated into your own evaluation pipeline.

    Example usage of the `tent` class:
    ```python
    from tent import tent
    import torch

    # Load your pre-trained model
    model = ... 

    # Create an optimizer for the model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize Tent
    tenter = tent(model, optimizer)

    # Adapt the model on a batch of test data
    adapted_model = tenter.adapt(test_batch)

    # Make predictions with the adapted model
    predictions = adapted_model(test_batch)
    ```
