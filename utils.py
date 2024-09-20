import os
from pycox.datasets import support, metabric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import brier_score_loss, roc_auc_score
import pandas as pd
import numpy as np

from pycox import models
from pycox.models import CoxPH, MTLR, DeepHitSingle
from pycox.models import utils
from pycox.models.interpolation import InterpolatePMF

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import torchtuples as tt # Some useful functions
import json

from collections import OrderedDict

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAffined,
    RandZoomd,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    Rand3DElasticd,
    RandShiftIntensityd,
)

from net import Deep_CNN
from hecktor_dataset import HecktorDataset, HecktorTestDataset, HecktorDataset2Images


def normalize(data, mean=None, std=None, skip_cols=[]):
    """Normalizes the columns of Pandas DataFrame to zero mean and unit
    standard deviation."""
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    if skip_cols is not None:
        mean[skip_cols] = 0
        std[skip_cols] = 1
    return (data - mean) / std, mean, std


def reset_parameters(model):
    """Resets the parameters of a PyTorch module and its children."""
    for m in model.modules():
        try:
            m.reset_parameters()
        except AttributeError:
            continue
    return model

def float_list(value):
    return [float(item) for item in value.split(',')]


def preprocess_data(args):
    ## dataset name
    data_name = args["data_name"].lower()
    
    if data_name == 'support2':
        data_type = "EHR"
        
        url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/support2csv.zip"
    
        # Remove other target columns and other model predictions
        cols_to_drop = [
            "hospdead",
            "slos",
            "charges",
            "totcst",
            "totmcst",
            "avtisst",
            "sfdm2",
            "adlp",
            "adls",
            "dzgroup",
            "sps",
            "aps",
            "surv2m",
            "surv6m",
            "prg2m",
            "prg6m",
            "dnr",
            "dnrday",
            "hday",
        ]

        # `death` is the overall survival event indicator
        # `d.time` is the time to death from any cause or censoring 
        data = (pd.read_csv(url)
                .drop(cols_to_drop, axis=1)
                .rename(columns={"d.time": "duration", "death": "event"}))
        data["event"] = data["event"].astype(int)
        
        data["ca"] = (data["ca"] == "metastatic").astype(int)

        # use recommended default values from official dataset description ()
        # or mean (for continuous variables)/mode (for categorical variables) if not given
        fill_vals = {
            "alb":     3.5,
            "pafi":    333.3,
            "bili":    1.01,
            "crea":    1.01,
            "bun":     6.51,
            "wblc":    9,
            "urine":   2502,
            "edu":     data["edu"].mean(),
            "ph":      data["ph"].mean(),
            "glucose": data["glucose"].mean(),
            "scoma":   data["scoma"].mean(),
            "meanbp":  data["meanbp"].mean(),
            "hrt":     data["hrt"].mean(),
            "resp":    data["resp"].mean(),
            "temp":    data["temp"].mean(),
            "sod":     data["sod"].mean(),
            "income":  data["income"].mode()[0],
            "race":    data["race"].mode()[0],
        }
        data = data.fillna(fill_vals)
        
        # one-hot encode categorical variables
        onehot_cols = ["sex", "dzclass", "income", "race"]
        data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
        
        eval_times = np.quantile(data.loc[data["event"] == 1, "duration"], [.25, .5, .75]).astype(int)
        
        df_train, df_val = train_test_split(data, test_size=args.data_split[1], random_state=args.seed)
        df_test = df_val
        #df_train, df_val = train_test_split(df_train, test_size=args.data_split[1], random_state=args.seed)
        
        df_train, mean_train, std_train = normalize(df_train, skip_cols=["duration", "event"])
        df_val, *_ = normalize(df_val, mean=mean_train, std=std_train, skip_cols=["duration", "event"])
        df_test, *_ = normalize(df_test, mean=mean_train, std=std_train, skip_cols=["duration", "event"])
        
        num_durations = args.label_num_duration
        lbltrans = MTLR.label_transform(num_durations, scheme='equidistant')
        get_target = lambda df: (df['duration'].values, df['event'].values)
        y_train = list(lbltrans.fit_transform(*get_target(df_train)))
        y_val = list(lbltrans.transform(*get_target(df_val)))
        
        
        # NOTE: WE ADD THE ACTUAL DURATION FOR RNC
        y_train.append(df_train['duration'].values)
        y_val.append(df_val['duration'].values)
        y_train = tuple(y_train)
        y_val = tuple(y_val)
        
        # NOTE:
        x_train = torch.tensor(df_train.drop(["duration", "event"], axis=1).values, dtype=torch.float)
        x_val   = torch.tensor(df_val.drop(["duration", "event"], axis=1).values, dtype=torch.float)
        x_test  = torch.tensor(df_test.drop(["duration", "event"], axis=1).values, dtype=torch.float)
        
        # NOTE:
        train = (x_train, y_train)
        val = (x_val, y_val)
        
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test) # NOTE: Why do we need this?
        
        args.lbl_cuts = lbltrans.cuts
        data_prep = {'type': data_type, 'train': train, 'val': val, 'duration_test': durations_test, 'event_test': events_test, 'eval_times': eval_times}
        
        return data_prep
    
    elif data_name == 'metabric':
        data_type = "EHR"
        data = metabric.read_df()

        eval_times = np.quantile(data.loc[data["event"] == 1, "duration"], [.25, .5, .75]).astype(int)
        df_train, df_val = train_test_split(data, test_size=args['data_split'][1], random_state=args['seed'])
        df_test = df_val
        
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        
        num_durations = args['label_num_duration']
        lbltrans = MTLR.label_transform(num_durations, scheme='equidistant')
        get_target = lambda df: (df['duration'].values, df['event'].values)
        y_train = list(lbltrans.fit_transform(*get_target(df_train)))
        y_val = list(lbltrans.transform(*get_target(df_val)))
        


        # NOTE: WE ADD THE ACTUAL DURATION FOR RNC
        y_train.append(df_train['duration'].values)
        y_val.append(df_val['duration'].values)
        y_train = tuple(y_train)
        y_val = tuple(y_val)
        
        train = (x_train, y_train)
        val = (x_val, y_val)
        
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_val) # NOTE: Why do we need this?
        
        args['lbl_cuts'] = lbltrans.cuts
        data_prep = {'type': data_type, 'train': train, 'val': val, 'eval_times': eval_times}
        
        return data_prep
        
    elif data_name == 'gbsg':
        raise ValueError("GBSG dataset is not available.")
    
    elif data_name == 'hecktor':
        data_type = 'EHR&Image'
        
        X = pd.read_csv(os.path.join(args.data_path, 'hecktor2022_endpoint_training.csv')) # ADAM
        y = pd.read_csv(os.path.join(args.data_path, 'hecktor2022_clinical_info_training.csv')) # ADAM
        df = pd.merge(X, y, on="PatientID")
        
        clinical_data = df

        clinical_data = clinical_data.rename(columns={"Relapse": "event", "RFS": "duration",
                                                      "Performance status": "Performance",
                                                      "HPV status (0=-, 1=+)": "HPV"}) # ADAM
        
        clinical_data = pd.get_dummies(clinical_data,
                                    columns=["Gender"],
                                    drop_first=True) # ADAM. Gender --> Gender_M (True/False)
        clinical_data = pd.get_dummies(clinical_data,
                               columns=["Performance", "HPV", "Surgery", "Tobacco", "Alcohol"], 
                               drop_first=False)
        
        # Drop some columns
        cols_to_drop = [
            # "Performance",
            # "HPV",
            # "Surgery", # ADAM. This exists in 2022 data but not in 2021 data
            "Task 1", # ADAM. I think we also have to filter this guy out
            "Task 2", # ADAM. I think we also have to filter this guy out
            # 'Tobacco',
            # 'Alcohol'
        ]
        clinical_data = clinical_data.drop(cols_to_drop, axis=1)
        
        # Fill missing values
        clinical_data['Weight'] = clinical_data['Weight'].fillna(75)
        # clinical_data['Tobacco'] = clinical_data['Tobacco'].fillna(-1)
        # clinical_data['Alcohol'] = clinical_data['Alcohol'].fillna(-1)
        # clinical_data['Tobacco'] = clinical_data['Tobacco'].replace(0, -1).fillna(0)
        # clinical_data['Alcohol'] = clinical_data['Alcohol'].replace(0, -1).fillna(0)
        
        # Create y_bin
        if args.model_name == "mtlr" or args.model_name == "deepmtlr":
            lbltrans = MTLR.label_transform(args.label_num_duration, scheme='equidistant')
            y_bins, y_events = lbltrans.fit_transform(clinical_data['duration'].values, clinical_data['event'].values)
        elif args.model_name == "deephit":
             lbltrans = DeepHitSingle.label_transform(args.label_num_duration)
             y_bins, y_events = lbltrans.fit_transform(clinical_data['duration'].values, clinical_data['event'].values)
        
        clinical_data['y_bin'] = y_bins
        
        ################################################################################################################################################################################################################################################################################################################################################################################################################################################################
        
        clinical_data = clinical_data[clinical_data['PatientID'] != 'MDA-036']
        ################################################################################################################################################################################################################################################################################################################################################################################################################################################################
        
        args.lbl_cuts = lbltrans.cuts
        args.max_duration = clinical_data['duration'].max()
        
        clinical_data['Weight'] = (clinical_data['Weight'] - clinical_data['Weight'].mean()) / clinical_data['Weight'].std()
        clinical_data['Age'] = (clinical_data['Age'] - clinical_data['Age'].mean()) / clinical_data['Age'].std()
        
        df_val = clinical_data[clinical_data['CenterID'].isin([1, 4])] #CHUM CHUV
        df_train = clinical_data[~clinical_data['CenterID'].isin([1, 4])] #CHUM CHUV
        #df_train = clinical_data

        # df_train, df_val = train_test_split(clinical_data, test_size=args.data_split[1], random_state=args.seed)
        print(f'Number of patients in train: {len(df_train)} and censoring rate: {1 - df_train["event"].mean()}')
        print(f'Number of patients in val: {len(df_val)} and censoring rate: {1 - df_val["event"].mean()}')
        
        #Calculate weights
        class_counts = df_train['event'].value_counts().to_dict()
        weights = [1.0 / class_counts[e] for e in df_train['event']]
        weights = torch.Tensor(weights)

        # Create sampler
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        #Define the transformations
        train_transforms = Compose([
            RandRotate90d(keys=["ctpt"], prob=0.1, spatial_axes=[0, 2]),  # 50% chance to rotate the images 90 degrees in the x-z plane
            RandFlipd(keys=["ctpt"], prob=0.1, spatial_axis=0),  # 50% chance to flip the images in the x axis
            RandFlipd(keys=["ctpt"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["ctpt"], prob=0.1, spatial_axis=2),
            # RandShiftIntensityd(keys=["ctpt"], offsets=0.10, prob=0.50,),
            # RandGaussianNoised(keys=["ctpt"], prob=0.25),  # 50% chance to add Gaussian noise to the images
            # RandAffined(keys=["ctpt"], prob=0.25, rotate_range=np.pi/12, translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1)),
            # RandZoomd(keys=["ctpt"], prob=0.5, min_zoom=0.9, max_zoom=1.1),
        ])
        # train_transforms = None
        val_transforms = None
        
        train = HecktorDataset2Images(df_train, train_transforms, args)
        val = HecktorDataset(df_val, val_transforms, args)
        
        return {'type': data_type, 'train': train, 'val': val, 'train_sampler': train_sampler}
    
    elif data_name == 'hecktor_test':
        data_type = 'EHR&Image'
        
        df = pd.read_csv(os.path.join(args.data_path, 'hecktor2022_clinical_info_testing.csv')) 
        clinical_data = df
        
        # clinical_data = pd.get_dummies(clinical_data,
        #                             columns=["Gender"],
        #                             drop_first=True) # ADAM. Gender --> Gender_M (True/False)
        clinical_data = clinical_data[clinical_data['Task 2'] == 1]
        clinical_data = clinical_data.rename(columns={
                                                      "Performance_status": "Performance",
                                                      "HPV status (0=-, 1=+)": "HPV"}) # ADAM
        
        clinical_data = pd.get_dummies(clinical_data,
                                    columns=["Gender"],
                                    drop_first=True) # ADAM. Gender --> Gender_M (True/False)
        clinical_data = pd.get_dummies(clinical_data,
                               columns=["Performance", "HPV", "Surgery", "Tobacco", "Alcohol"], 
                               drop_first=False)

        
        # Drop some columns
        cols_to_drop = [
            # "Performance_status",
            # "HPV status (0=-, 1=+)",
            # "Surgery", # ADAM. This exists in 2022 data but not in 2021 data
            "Task 1", # ADAM. I think we also have to filter this guy out
            "Task 2", # ADAM. I think we also have to filter this guy out
        ]
        clinical_data = clinical_data.drop(cols_to_drop, axis=1)
        
        # Fill missing values
        clinical_data['Weight'] = clinical_data['Weight'].fillna(75)
        # clinical_data['Tobacco'] = clinical_data['Tobacco'].fillna(-1)
        # clinical_data['Alcohol'] = clinical_data['Alcohol'].fillna(-1)
        clinical_data['Weight'] = (clinical_data['Weight'] - 81.86) /19.25
        clinical_data['Age'] = (clinical_data['Age'] - 61) / 9.1
        
        # args.lbl_cuts = lbltrans.cuts # ASK
        
        df_test = clinical_data
        
        test_transforms = None
        
        test = HecktorTestDataset(df_test, test_transforms, args)
        
        return {'type': data_type, 'test': test}
    
    
    elif data_name in ['hecktor_5_fold', 'hecktor_10_fold']:
        data_type = 'EHR&Image'
        
        X = pd.read_csv(os.path.join(args["data_path"], 'hecktor2022_endpoint_training.csv'))
        y = pd.read_csv(os.path.join(args["data_path"], 'hecktor2022_clinical_info_training.csv'))
        df = pd.merge(X, y, on="PatientID")
        
        json_file_path = os.path.join(args["data_path"], f'{data_name}_splits.json')
        
        clinical_data = df
        clinical_data = clinical_data.rename(columns={"Relapse": "event", "RFS": "duration",
                                                      "Performance status": "Performance",
                                                      "HPV status (0=-, 1=+)": "HPV"})
        
        clinical_data = pd.get_dummies(clinical_data,
                                    columns=["Gender"],
                                    drop_first=True) # ADAM. Gender --> Gender_M (True/False)
        clinical_data = pd.get_dummies(clinical_data,
                               columns=["Performance", "HPV", "Surgery", "Tobacco", "Alcohol"], 
                               drop_first=False)
        
        # Drop some columns
        cols_to_drop = [
            # "Performance",
            # "HPV",
            # "Surgery", # ADAM. This exists in 2022 data but not in 2021 data
            "Task 1", # ADAM. I think we also have to filter this guy out
            "Task 2", # ADAM. I think we also have to filter this guy out
            # 'Tobacco',
            # 'Alcohol'
        ]
        clinical_data = clinical_data.drop(cols_to_drop, axis=1)
        
        # Fill missing values
        clinical_data['Weight'] = clinical_data['Weight'].fillna(75)
        # clinical_data['Tobacco'] = clinical_data['Tobacco'].fillna(-1)
        # clinical_data['Alcohol'] = clinical_data['Alcohol'].fillna(-1)
        # clinical_data['Tobacco'] = clinical_data['Tobacco'].replace(0, -1).fillna(0)
        # clinical_data['Alcohol'] = clinical_data['Alcohol'].replace(0, -1).fillna(0)
        
        # Create y_bin
        if args["model_name"] == "mtlr" or args["model_name"] == "deepmtlr":
            lbltrans = MTLR.label_transform(args["label_num_duration"]) #, scheme='equidistant')
            y_bins, y_events = lbltrans.fit_transform(clinical_data['duration'].values, clinical_data['event'].values)
        elif args["model_name"] == "deephit":
             lbltrans = DeepHitSingle.label_transform(args["label_num_duration"])
             y_bins, y_events = lbltrans.fit_transform(clinical_data['duration'].values, clinical_data['event'].values)
        
        clinical_data['y_bin'] = y_bins
        
        ################################################################################################################################################################################################################################################################################################################################################################################################################################################################
        
        clinical_data = clinical_data[clinical_data['PatientID'] != 'MDA-036']
        ################################################################################################################################################################################################################################################################################################################################################################################################################################################################
        
        args["lbl_cuts"] = lbltrans.cuts
        args["max_duration"] = clinical_data['duration'].max()
        
        clinical_data['Weight'] = (clinical_data['Weight'] - clinical_data['Weight'].mean()) / clinical_data['Weight'].std()
        clinical_data['Age'] = (clinical_data['Age'] - clinical_data['Age'].mean()) / clinical_data['Age'].std()
        
        with open(json_file_path, 'r') as file:
            filename_folds = json.load(file)
        
        filename_folds_df = pd.DataFrame(filename_folds, columns=['PatientID', 'Fold'])
        clinical_data = pd.merge(clinical_data, filename_folds_df, on='PatientID')
        
        df_train = clinical_data[clinical_data['Fold'] != args['fold']]
        df_val = clinical_data[clinical_data['Fold'] == args['fold']]
        
        # df_train, df_val = train_test_split(clinical_data, test_size=args.data_split[1], random_state=args.seed)
        print(f'Number of patients in train: {len(df_train)} and censoring rate: {1 - df_train["event"].mean()}')
        print(f'Number of patients in val: {len(df_val)} and censoring rate: {1 - df_val["event"].mean()}')

        
        #Calculate weights
        class_counts = df_train['event'].value_counts().to_dict()
        weights = [1.0 / class_counts[e] for e in df_train['event']]
        weights = torch.Tensor(weights)

        # Create sampler
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        #Define the transformations
        train_transforms = Compose([
                RandRotate90d(keys=["ctpt"], prob=0.1, spatial_axes=[0, 2]),  # 50% chance to rotate the images 90 degrees in the x-z plane
                RandFlipd(keys=["ctpt"], prob=0.1, spatial_axis=0),  # 50% chance to flip the images in the x axis
                RandFlipd(keys=["ctpt"], prob=0.1, spatial_axis=1),
                RandFlipd(keys=["ctpt"], prob=0.1, spatial_axis=2),
                # RandGaussianSmoothd(keys=["ctpt"], prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                # RandAdjustContrastd(keys=["ctpt"], prob=0.5),
                # Rand3DElasticd(keys=["ctpt"], prob=0.5, sigma_range=(3,9), magnitude_range=(0.1, 0.2)),
                # RandShiftIntensityd(keys=["ctpt"], prob=0.5, offsets=0.1),
            ])
        
        train_transforms = train_transforms
        val_transforms = None
        
        train = HecktorDataset2Images(df_train, train_transforms, args)
        val = HecktorDataset(df_val, val_transforms, args)
        
        return {'type': data_type, 'train': train, 'val': val, 'train_sampler': train_sampler}
    
    elif data_name == 'None':
        data = None
        
def define_model_and_loss(args):
    model_name = args['model_name'].lower()
    
    if model_name == 'mtlr':
        model = MTLR
        
    elif model_name == 'deepmtlr':
        loss = models.loss.NLLMTLRLoss()
        if args['data_type'] == "EHR":
            model = Model(args)
        elif args['data_type'] == "EHR&Image":
            model = Deep_CNN(args)
            model.apply(init_params)
        
    elif model_name == 'deephit':
        loss = models.loss.DeepHitSingleLoss(alpha=0.2, sigma=0.1)
        if args['data_type'] == "EHR":
            model = Model(args)
        elif args['data_type'] == "EHR&Image":
            model = Deep_CNN(args)
            model.apply(init_params)
        
    elif model_name == 'deepsurv':
        loss = models.loss.CoxPHLoss()
        if args['data_type'] == "EHR":
            model = Model(args)
        elif args['data_type'] == "EHR&Image":
            model = Deep_CNN(args)
            model.apply(init_params)
    elif model_name == 'cox-ph':
        model = CoxPH
    
    return model, loss

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
 
        self.duration_index = args['lbl_cuts']
        activation = eval(f"torch.nn.{args['activation']}")
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(args['in_features'], args['layer1_size']),
            activation(), #torch.nn.ReLU(),
            torch.nn.BatchNorm1d(args['layer1_size']),
            torch.nn.Dropout(args['dropout_rate']),
            
            torch.nn.Linear(args['layer1_size'], args['layer2_size']),
            activation(), #torch.nn.ReLU(),
            torch.nn.BatchNorm1d(args['layer2_size'])
        )

        self.final_layer = nn.Sequential(
            nn.Dropout(args['dropout_rate']),
            torch.nn.Linear(args['layer2_size'], args['label_num_duration'])
        )
    
    def forward_enc(self, x):
        return self.encoder(x)  
    
    def forward(self, x):
        x_emb = self.encoder(x)
        out = self.final_layer(x_emb)

        return x_emb, out
    
    def predict_surv(self, input,  numpy=None):
        pmf = self.predict_pmf(input, False)
        surv = 1 - pmf.cumsum(1)
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        # preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        preds = torch.from_numpy(input)
        preds = utils.cumsum_reverse(preds, dim=1)
        pmf = utils.pad_col(preds).softmax(1)[:, :-1]
        return tt.utils.array_or_tensor(pmf, numpy, input)

    def predict_surv_df(self, input):
        surv = self.predict_surv(input, True)
        return pd.DataFrame(surv.transpose(), self.duration_index)
    
    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        """Use interpolation for predictions.
        There are only one scheme:
            `const_pdf` and `lin_surv` which assumes pice-wise constant pmf in each interval (linear survival).
        s
        Keyword Arguments:
            sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
                10 times the number of grid points than the original `duration_index` (default: {10}).
            scheme {str} -- Type of interpolation {'const_hazard', 'const_pdf'}.
                See `InterpolateDiscrete` (default: {'const_pdf'})
            duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
                only for setting index in `predict_surv_df` (default: {None})
        
        Returns:
            [InterpolationPMF] -- Object for prediction with interpolation.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolatePMF(self, scheme, duration_index, sub)

# ADAM
import matplotlib.pyplot as plt
from umap import UMAP
class HelperUMAP:
    def __init__(self, X, **kwargs):
        self.umap_obj = UMAP(**kwargs)
        self.embs = self.umap_obj.fit_transform(X)
    
    def __call__(self, labels):
        
        fig = plt.figure()
        ax = fig.add_subplot()
        points = ax.scatter(self.embs[:,0], self.embs[:,1], c=np.log(labels), s=20, cmap="Spectral")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(points)
        
        return fig

# https://discuss.pytorch.org/t/meaning-of-parameters/10655
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metric_at_times(metric, time_true, prob_pred, event_observed, score_times):
    """Helper function to evaluate a metric at given timepoints."""
    scores = []
    for time, pred in zip(score_times, prob_pred.T):
        target = time_true > time
        uncensored = target | event_observed.astype(bool)
        scores.append(metric(target[uncensored], pred[uncensored]))
        
    return scores


def brier_score_at_times(time_true, prob_pred, event_observed, score_times):
    scores = compute_metric_at_times(brier_score_loss, 
                                     time_true,
                                     prob_pred,
                                     event_observed,
                                     score_times)
    return scores


def roc_auc_at_times(time_true, prob_pred, event_observed, score_times):
    scores = compute_metric_at_times(roc_auc_score, 
                                     time_true,
                                     prob_pred, 
                                     event_observed,
                                     score_times)
    return scores

def get_surv_curve(surv):
    plot = surv.plot(drawstyle='steps-post')
    plot.set_ylabel('S(t | x)')
    plot.set_xlabel('Time')
    
    return plot

def mtlr_survival(logits: torch.Tensor) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    G = torch.tril(torch.ones(logits.size(1),
                              logits.size(1))).to(logits.device)
    density = torch.softmax(logits, dim=1)
    return torch.matmul(density, G)

def mtlr_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t,
    respectively.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits: torch.Tensor) -> torch.Tensor:
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)


def init_params(m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # initialize the final bias so that the predictied probability at
            # init is equal to the proportion of positive samples
            nn.init.constant_(m.bias, -1.5214691)

