import argparse
import os
import yaml
import sys
from pathlib import Path
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from pycox.evaluation import EvalSurv
from tqdm import tqdm, trange
import wandb
from umap import UMAP
from lifelines.utils import concordance_index

from monai.utils import set_determinism

from utils import *
from rnc_loss import RnCEHRLoss, ProgRnCLoss
from hecktor_dataset import HecktorDataset2Images

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed)
    torch.cuda.manual_seed_all(seed)

def init_wandb(config):
    os.environ["WANDB_API_KEY"] = config['wandb_api_key']
    wandb.login()
    wandb.init(project=config['wandb_project'], name=config['run_name'])

def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat
 
def pair_rank_mat(idx_durations, events, dtype='float32'):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.
    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.
    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})
    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat

def validate(model, loader, device, loss_fn, umap=False, surv_curve_samples=None, return_bs_auc=False, args=None):
    model.eval()
    
    meter_loss = AverageMeter()
       
    list_embs = []
    list_uc = []
    list_preds = []
    list_duration = []
    list_event = []

    with torch.no_grad():
        for x_val, y_val in loader:
            if isinstance(x_val, torch.Tensor):
                x_val = x_val.to(device)
            elif isinstance(x_val, list):
                x_val = [t.to(device) for t in x_val]
            else:
                raise ValueError(f"x_val should be a tensor or a list of tensors (currently it is {type(x_val)})")
            
            if isinstance(loader.dataset, HecktorDataset2Images):
                x_val = (
                    torch.cat((x_val[0], x_val[1]), dim=0),
                    x_val[-1].repeat(2,1),
                )
                if len(y_val.shape) == 1:
                    y_val = y_val[:, None]
                y_val = y_val.repeat(2,1)
                
            y_val = y_val.to(device)
            embs, y_pred = model(x_val)
            list_preds.append(y_pred.cpu().numpy())
            list_embs.append(embs[y_val[:,1]==1].cpu().numpy())
            list_uc.append(y_val[:,2][y_val[:,1]==1].cpu().numpy())
            list_duration.append(y_val[:,2].detach().cpu().numpy())
            list_event.append(y_val[:,1].cpu().numpy())
         
            if args['model_name'] == "deephit":
                rank_mat = pair_rank_mat(y_val[:,0], y_val[:,1])
                meter_loss.update(val=loss_fn(y_pred, y_val[:,0].long(), y_val[:,1], torch.tensor(rank_mat).to(device)), n=len(x_val))
            else:
                meter_loss.update(val=loss_fn(y_pred, y_val[:,0].long(), y_val[:,1]), n=len(x_val))

    embs = np.concatenate(list_embs)
    uc = np.concatenate(list_uc)
    preds = np.concatenate(list_preds)
    durations = np.concatenate(list_duration)
    events = np.concatenate(list_event)
  
    pred_risk = mtlr_risk(torch.tensor(preds))
    ci_harrell = concordance_index(durations, -pred_risk, events)

    surv = model.interpolate(10).predict_surv_df(preds)

    out = {'loss': meter_loss.avg}

    ev = EvalSurv(surv, durations, events, censor_surv='km')

    ci = ev.concordance_td('antolini')
    out['ci_antolini'] = ci
    out['ci_harrell'] = ci_harrell
    
    if return_bs_auc:
        surv = surv.loc[surv.index < max(durations)]
        eval_times = [surv.index[round(0.25*len(surv))], surv.index[round(0.5*len(surv))], surv.index[round(0.75*len(surv))]]
        filtered_surv = surv.loc[eval_times]
        bs = brier_score_at_times(durations, filtered_surv.T.values, events, eval_times)
        auc = roc_auc_at_times(durations, filtered_surv.T.values, events, eval_times)

        out['eval_times'] = eval_times
        out['bs'] = bs
        out['auc'] = auc
    
    if umap:
        ump = HelperUMAP(embs, n_components=2, init='random', random_state=0)
        umap_plot = ump(uc)
        out['umap'] = umap_plot
    
    if surv_curve_samples is not None:
        out['surv_curve'] = get_surv_curve(surv.iloc[:, :5])

    return out


def train(train_loader, val_loader, model, loss_fn, optimizer, device, args):
    reset_parameters(model)
    
    model.to(device)
    
    if args['loss_rnc_type'] == 'RnCEHRLoss':
        loss_rnc_fn = RnCEHRLoss(temperature=args['temperature'], label_diff='l1', feature_sim='l2').to(device)
    elif args['loss_rnc_type'] == 'ProgRnCLoss':
        loss_rnc_fn = ProgRnCLoss(temperature=args['temperature'], label_diff='l1_vec', feature_sim='l2').to(device)
    
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    pbar = trange(args['epochs'], desc='Training', leave=True)
    best_loss = float('inf')
    best_ci = float('-inf')
    best_weights = None
    for epoch in pbar:
        model.train()
        loss_meter = AverageMeter()
        for x_train, y_train in train_loader:
            y_train = y_train.to(device)
            if isinstance(x_train, torch.Tensor):
                x_train = x_train.to(device)
            elif isinstance(x_train, list):
                x_train = [t.to(device) for t in x_train]
            else:
                raise ValueError(f"x_train should be a tensor or a list of tensors (currently it is {type(x_train)})")
            
            if isinstance(train_loader.dataset, HecktorDataset2Images):
                x_train = (
                    torch.cat((x_train[0], x_train[1]), dim=0),
                    x_train[-1].repeat(2,1),
                )
                if len(y_train.shape) == 1:
                    y_train = y_train[:, None]
                y_train = y_train.repeat(2,1)
            
            embs, y_pred = model(x_train)
            
            optimizer.zero_grad()
            
            if args['model_name'] == "deephit":
                rank_mat = pair_rank_mat(y_train[:,0], y_train[:,1])
                loss = loss_fn(y_pred, y_train[:,0].long(), y_train[:,1], torch.tensor(rank_mat).to(device))
            else:
                loss = loss_fn(y_pred, y_train[:,0].long(), y_train[:,1])
            
            if args['loss_rnc'] > 0:
                if isinstance(loss_rnc_fn, RnCEHRLoss):
                    if sum(y_train[:,1].detach()==1) > 2:
                        loss_rnc = loss_rnc_fn(embs[y_train[:,1]==1], y_train[:,2][y_train[:,1]==1][:,None])
                        loss = loss + loss_rnc * args['loss_rnc']
                elif isinstance(loss_rnc_fn, ProgRnCLoss):
                    if sum(y_train[:,1].detach()==1) > 2:
                        loss_rnc = loss_rnc_fn(embs, y_train[:,2], y_train[:,1])
                        loss = loss + loss_rnc * args['loss_rnc']
                        
            loss.backward()
            optimizer.step()
            
            loss_meter.update(val=loss.item(), n=len(x_train))
        
        pbar.set_description(f"[epoch {epoch+1:4}/{args['epochs']}]")
        if (epoch+1) % args['validation_epoch'] == 0:
            model.eval()
            
            val_out = validate(model, val_loader, device, loss_fn, umap=False, args=args)
            val_loss = val_out['loss']
            val_ci = val_out['ci_antolini']
            val_ci_harrell = val_out['ci_harrell']
            
            if args['wandb']:
                log_dict = {
                    'Epoch': epoch+1,
                    'Val_loss': val_loss.item(),
                    'Val_CI_Antolini': val_ci,
                    'Val_CI_Harrell': val_ci_harrell,
                    'Train_loss': loss_meter.avg,
                }
                
                wandb.log(log_dict)

            pbar.set_postfix_str(f"Training loss: {loss_meter.avg:.4f}. Validation loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
            
            if val_ci_harrell > best_ci:
                best_ci = val_ci_harrell
                best_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                
        scheduler.step()
    
    model.eval()
    return model, best_weights

def process(config):
    # Prepare data
    data = preprocess_data(config)
    
    if data["type"] == "EHR&Image":
        print("Both 'EHR' and 'Image' are present.")
        config['in_features'] = data['train'][0][0][-1].shape[0] # number of clinical features
        train_ds = data['train']
        val_ds = data['val']
    elif data["type"] == "EHR":
        print("Only 'EHR' is present.")
        config['in_features'] = data['train'][0].shape[1]
        x_train, y_train = data['train']
        x_val, y_val = data['val']
        
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).permute(1,0)
        
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).permute(1,0)
        
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
    else:
        raise ValueError("Neither 'EHR' nor 'Image' are present. Check your data.")
    
    config['data_type'] = data["type"]
    
    # Define model and loss
    model, loss_fn = define_model_and_loss(config)
    
    # Define optimizer
    optimizer_name = getattr(torch.optim, config['optimizer'])
    optimizer = optimizer_name(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False, drop_last=True,
                              pin_memory=True, num_workers=config['num_workers'])
    
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=config['num_workers'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The model is using {device} for training!")
    
    model, best_weights = train(train_loader, val_loader, model, loss_fn, optimizer, device, config)
    torch.save({'weights': best_weights, 'lbl_cuts': config['lbl_cuts'], 'max_duration': config['max_duration']}, f'./submission_weights/model_{config["run_name"]}_fold_{config["fold"]}.pt')

    if config['wandb']:
        log_dict = {}
        
        model.load_state_dict(best_weights)
        
        out = validate(model, train_loader, device, loss_fn, umap=True, surv_curve_samples=5, return_bs_auc=True, args=config)
        train_loss, train_umap, train_ci, train_surv_curve, train_ci_ant = out['loss'], out['umap'], out['ci_harrell'], out['surv_curve'], out['ci_antolini']
        train_eval_times, train_bs, train_auc = out['eval_times'], out['bs'], out['auc']
        
        out = validate(model, val_loader, device, loss_fn, umap=True, surv_curve_samples=5, return_bs_auc=True, args=config)
        val_loss, val_umap, val_ci, val_surv_curve, val_ci_ant = out['loss'], out['umap'], out['ci_harrell'], out['surv_curve'], out['ci_antolini']
        val_eval_times, val_bs, val_auc = out['eval_times'], out['bs'], out['auc']
        
        log_dict.update({
            '(Best) Val loss': val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss,
            '(Best) Val CI_harrel': val_ci,
            '(Best) Val CI_antolini': val_ci_ant,
            '(Best) Val Surv Curve': wandb.Image(val_surv_curve),
            '(Best) Val UMAP': wandb.Image(val_umap),
            '(Best) Val Eval Times': wandb.Table(columns=list(range(len(val_eval_times))), data=[val_eval_times]),
            '(Best) Val Brier Score': wandb.Table(columns=list(range(len(val_eval_times))), data=[val_bs]),
            '(Best) Val AUC': wandb.Table(columns=list(range(len(val_eval_times))), data=[val_auc]),
            '(Best) Train loss': train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss,
            '(Best) Train CI': train_ci,
            '(Best) Train Surv Curve': wandb.Image(train_surv_curve),
            '(Best) Train UMAP': wandb.Image(train_umap),
        })
        
        wandb.log(log_dict)

def main():
    parser = argparse.ArgumentParser(description='SurvRNC Training Script')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to the config file')
    parser.add_argument('--override', nargs=argparse.REMAINDER, help='Override config parameters. Example: --optimizer SGD')

    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.override:
        override_args = args.override
        for override in override_args:
            key, value = override.split('=')
            # Attempt to cast to int or float
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = value.lower() if value.lower() in ['true', 'false'] else value
            keys = key.split('.')
            cfg = config
            for k in keys[:-1]:
                cfg = cfg.setdefault(k, {})
            cfg[keys[-1]] = value

    # Set seeds
    set_seeds(config['seed'])

    # Initialize WandB if enabled
    if config['wandb']:
        init_wandb(config)

    # Proceed with training using the loaded configuration
    process(config)

if __name__ == "__main__":
    main()


