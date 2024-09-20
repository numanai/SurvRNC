# Model Architecture

## Overview
SurvRNC utilizes a combination of Convolutional Neural Networks (CNNs) and fully connected layers to process both medical images (CT and PT scans) and Electronic Health Records (EHR). The architecture is designed to perform survival analysis by predicting patient outcomes based on the integrated data.

## Components

### 1. CNN Encoder (net.py)
The CNN encoder processes the 3D medical images to extract meaningful features.

Layers:
- Convolutional layers with configurable kernel sizes
- Batch Normalization or Group Normalization
- Activation functions (ReLU, LeakyReLU, ELU)
- Dropout layers for regularization
- Max Pooling and Adaptive Average Pooling

### 2. EHR Encoder (net.py)
A fully connected layer processes the EHR data.

Layers:
- Linear layer transforming EHR features to a latent space

### 3. Feature Fusion (net.py)
Concatenates image features from the CNN encoder with EHR features.

### 4. Fully Connected Layers (net.py)
Additional dense layers to learn complex interactions between image and EHR features.

### 5. Output Layer (net.py)
Produces predictions for survival analysis based on the selected model type (deepmtlr, deephit, deepsurv, etc.).

## Custom Loss Functions (rnc_loss.py)
- RnCEHRLoss: This loss ignores censored patients data.
- ProgRnCLoss: This loss has been adapted to include the uncertain pairs for prognosis task.

## Training Pipeline (main.py)
- Data Loading: Utilizes HecktorDataset2Images to load and preprocess data.
- Model Initialization: Sets up the neural network architecture based on user configurations.
- Optimizer and Scheduler: Configurable optimizer (AdamW, Adam) and learning rate schedulers (StepLR, CosineAnnealingLR).
- Training Loop: Iterates over epochs, computes loss, performs backpropagation, and updates model weights.
- Evaluation: Assesses model performance using metrics like Concordance Index and Brier Score.
- Logging: Integrates with W&B for experiment tracking and visualization.

## Example Command
fix the code in this markdown file

The provided content in the markdown file appears to be mostly correct and well-structured. However, there are a few minor improvements we can make to enhance its formatting and readability. Here's the updated version with some fixes:
Architecture.md

```bash
python main.py \
--batch_size 32 \
--data_name hecktor_5_fold \
--dense_factor 3 \
--dropout_rate 0.5 \
--epochs 80 \
--k1 5 \
--k2 5 \
--label_num_duration 20 \
--loss_rnc 1.0 \
--lr 1e-04 \
--model_name deepmtlr \
--n_depth 3 \
--optimizer AdamW \
--temperature 3.9 \
--weight_decay 0.001 \
--seed 1406 \
--fold 4 \
--run_name deepmtlr_hktr_ProgRNC_f4_b32
```