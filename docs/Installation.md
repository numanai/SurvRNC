# Installation Guide

## Prerequisites

- Python 3.7+
- pip

## Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/numanai/SurvRNC.git
    cd SurvRNC
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download and Prepare Data**

    - Place your raw CT and PT scan data in the `data/raw/ct/` and `data/raw/pt/` directories respectively.
    - Run the preprocessing script:

    ```bash
    python ctpt_preprocess.py --data_path ./data/raw --save_dir ./data/processed/ctpt --space_x 2 --space_y 2 --space_z 2 --a_min -250 --a_max 250 --b_min 0 --b_max 1 --seed 1234
    ```

## Additional Setup

- **Weights & Biases (W&B) Integration**

    If you plan to use W&B for experiment tracking, ensure you have an account and set up the API key.

    ```bash
    wandb login
    ```

- **Ensure CUDA Availability**

    For GPU acceleration, ensure that CUDA is installed and compatible with your PyTorch version.