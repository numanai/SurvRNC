# Usage Guide

## Preprocessing Data

Before training, preprocess the CT and PT scans.

bash
python ctpt_preprocess.py --data_path ./data/raw --save_dir ./data/processed/ctpt --space_x 2 --space_y 2 --space_z 2 --a_min -250 --a_max 250 --b_min 0 --b_max 1 --seed 1234


## Training the Model

Run the training script with desired parameters.


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


### Available Arguments

- `--data_path`: Path to the raw data.
- `--data_name`: Name of the dataset (e.g., `support2`, `metabric`, `gbsg`, `hecktor_5_fold`).
- `--fold`: Fold number for cross-validation.
- `--data_split`: Train, validation, and test split ratios.
- `--label_num_duration`: Number of duration bins for labels.
- `--space_x`, `--space_y`, `--space_z`: Spatial dimensions for spacing.
- `--a_min`, `--a_max`, `--b_min`, `--b_max`: Normalization parameters.
- `--model_name`: Model architecture (`MTLR`, `deephit`, `deepsurv`, `Cox-PH`).
- `--activation`: Activation function (`ReLU`, `LeakyReLU`, etc.).
- `--dropout_rate`: Dropout rate.
- `--layer1_size`, `--layer2_size`: Sizes of the neural network layers.
- `--k1`, `--k2`: Kernel sizes for CNN blocks.
- `--n_depth`: Number of dense layers.
- `--dense_factor`: Factor to increase the size of dense layers.
- `--loss_rnc`: Weight for the RnC loss component.
- `--temperature`: Temperature parameter for RnC.
- `--loss_rnc_type`: Type of RnC loss (`RnCEHRLoss`, `ProgRnCLoss`).
- `--optimizer`: Optimizer choice (`Adam`, `AdamW`, etc.).
- `--weight_decay`: Weight decay for optimizer.
- `--seed`: Random seed for reproducibility.
- `--run_name`: Name for the experiment run.

## Evaluation

After training, evaluate the model's performance using the provided evaluation metrics.


bash
python main.py --evaluate --model_path ./models/your_model.pt


## Logging and Visualization

The project integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization. Ensure W&B is set up as per the [Installation Guide](./docs/Installation.md).

