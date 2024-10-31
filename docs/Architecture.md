# Model Architecture

## Table of Contents
1. [Overview](#overview)
2. [High-Level Diagram](#high-level-diagram)
3. [Components](#components)
   1. [CNN Encoder](#cnn-encoder)
   2. [EHR Encoder](#ehr-encoder)
   3. [Feature Fusion](#feature-fusion)
   4. [Fully Connected Layers](#fully-connected-layers)
   5. [Output Layer](#output-layer)
4. [Data Preprocessing](#data-preprocessing)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Custom Loss Functions](#custom-loss-functions)
7. [Training Pipeline](#training-pipeline)
8. [Extending the Architecture](#extending-the-architecture)
9. [Example Command](#example-command)

## Overview
SurvRNC utilizes a combination of Convolutional Neural Networks (CNNs) and fully connected layers to process both medical images (CT and PT scans) and Electronic Health Records (EHR). The architecture is designed to perform survival analysis by predicting patient outcomes based on the integrated data.

## High-Level Diagram
![High-Level Diagram](./high_level_diagram.png)

## Components

### 1. CNN Encoder (net.py)
The CNN encoder processes the 3D medical images to extract meaningful features.

Layers:
- Convolutional layers with configurable kernel sizes
- Batch Normalization or Group Normalization
- Activation functions (ReLU, LeakyReLU, ELU)
- Dropout layers for regularization
- Max Pooling and Adaptive Average Pooling

#### Example Code:
```python
def conv_3d_block(in_c, out_c, act='relu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)],
        ['elu', nn.ELU(inplace=True)],
    ])
    
    normalizations = nn.ModuleDict([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act],
    )
```

### 2. EHR Encoder (net.py)
A fully connected layer processes the EHR data.

Layers:
- Linear layer transforming EHR features to a latent space

#### Example Code:
```python
class EHR_Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(EHR_Encoder, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc(x)
```

### 3. Feature Fusion (net.py)
Concatenates image features from the CNN encoder with EHR features.

### 4. Fully Connected Layers (net.py)
Additional dense layers to learn complex interactions between image and EHR features.

### 5. Output Layer (net.py)
Produces predictions for survival analysis based on the selected model type (deepmtlr, deephit, deepsurv, etc.).

## Data Preprocessing
The data preprocessing steps involve loading and transforming the CT/PT images and EHR data to be fed into the model.

### Example Code:
```python
class HecktorDataset(Dataset):
    def __init__(self, clinical_data, transform, args):
        self.clinical_data = clinical_data
        self.transform = transform
        self.data_path = args['data_path']
    
    def __len__(self):
        return len(self.clinical_data)
    
    def __getitem__(self, idx):
        row_data = self.clinical_data.iloc[idx]
        patient_id = row_data['PatientID']
        
        ctpt = torch.load(os.path.join(self.data_path, 'processed', 'ctpt', f'{patient_id}_ctpt.pt'))
        
        if self.transform:
            data_dict = {'ctpt': ctpt}
            data_dict = self.transform(data_dict)
            ctpt = data_dict['ctpt']
        
        y = np.array([row_data['y_bin'], row_data['event'], row_data['duration']])
        
        x_ehr = np.array([
            row_data['Age'], row_data['Weight'],
            row_data['Chemotherapy'], row_data['Gender_M'],
            row_data['Performance_0.0'], row_data['Performance_1.0'], row_data['Performance_2.0'], 
            row_data['Performance_3.0'], row_data['Performance_4.0'],
            row_data['HPV_0.0'], row_data['HPV_1.0'],
            row_data['Surgery_0.0'], row_data['Surgery_1.0'],
            row_data['Tobacco_0.0'], row_data['Tobacco_1.0'], 
            row_data['Alcohol_0.0'], row_data['Alcohol_1.0'],
        ]).astype(np.float32)
        
        return (ctpt, x_ehr), y
```

## Evaluation Metrics
The evaluation metrics used to assess the model's performance include:

- Concordance Index (CI)
- Brier Score

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

## Extending the Architecture
To extend or modify the architecture for different use cases or datasets, consider the following:

- Adjust the CNN encoder layers to accommodate different image modalities or resolutions.
- Modify the EHR encoder to include additional clinical features.
- Experiment with different fusion techniques to combine image and EHR features.
- Implement custom loss functions tailored to specific survival analysis tasks.

## Example Command
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
