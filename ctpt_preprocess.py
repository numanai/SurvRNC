import argparse
import os
import yaml
import torch
from monai.transforms import Compose, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd, NormalizeIntensityd, SpatialPadd, ConcatItemsd
from tqdm import tqdm
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='CT/PT Preprocessing Script')
    parser.add_argument('--config', type=str, default='configs/preprocess.yaml', help='Path to the config file')
    parser.add_argument('--override', nargs=argparse.REMAINDER, help='Override config parameters. Example: --space_x 3')

    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.override:
        override_args = args.override
        for override in override_args:
            key, value = override.split('=')
            # Attempt to cast to int or float
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    value = value.lower() if value.lower() in ['true', 'false'] else value
            keys = key.split('.')
            cfg = config
            for k in keys[:-1]:
                cfg = cfg.setdefault(k, {})
            cfg[keys[-1]] = value

    # Access configuration parameters
    save_dir = config['save_dir']
    data_path = config['data_path']
    space_x = config['space_x']
    space_y = config['space_y']
    space_z = config['space_z']
    a_min = config['a_min']
    a_max = config['a_max']
    b_min = config['b_min']
    b_max = config['b_max']
    seed = config['seed']

    # Check and create save directory
    if os.path.isdir(save_dir):
        raise FileExistsError(f"Folder {save_dir} already exists ...")
    else:
        os.makedirs(save_dir, exist_ok=False)

    # Set seeds
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # Define transforms
    ctpt_transform = Compose([
        LoadImaged(keys=["ct", "pt"], ensure_channel_first=True),
        SpatialPadd(keys=["ct", "pt"], spatial_size=(176,176,176), method='end'),
        Orientationd(keys=["ct", "pt"], axcodes="RAS"),
        Spacingd(
            keys=["ct", "pt"],
            pixdim=(space_x, space_y, space_z),
            mode=("bilinear"),
        ), 
        ScaleIntensityRanged(
            keys=["ct"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),
        NormalizeIntensityd(keys=["pt"]),
        ConcatItemsd(keys=["pt", "ct"], name="ctpt"),
    ])
    
    # Process and save data
    for ct_name in tqdm(os.listdir(os.path.join(data_path, 'ct'))):
        patient_id = ct_name.split('_')[0]
        
        ctpt = {
            'ct': os.path.join(data_path, 'ct', ct_name),
            'pt': os.path.join(data_path, 'pt', f'{patient_id}_pt_trans.nii.gz'),
        }
        
        ctpt = ctpt_transform(ctpt)['ctpt']
        ctpt = torch.tensor(ctpt)
        
        torch.save(ctpt, os.path.join(save_dir, f'{patient_id}_ctpt.pt'))

if __name__ == '__main__':
    main()