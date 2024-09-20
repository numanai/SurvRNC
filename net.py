import torch
from torch import nn
import pandas as pd
import torchtuples as tt # Some useful functions
import numpy as np

from pycox.models import utils
from pycox.models.interpolation import InterpolatePMF

def conv_3d_block (in_c, out_c, act='relu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict ([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)],
        ['elu', nn.ELU(inplace=True)],
    ])
    
    normalizations = nn.ModuleDict ([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act],
    )

def flatten_layers(arr):
    return [i for sub in arr for i in sub]



class Deep_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.duration_index = args['lbl_cuts']
        # self.max_duration = args.max_duration

        self.ehr = nn.Linear(args['in_features'], 32)

        self.cnn = nn.Sequential(#block 1
                                 conv_3d_block(2, 32, kernel_size=args['k1']),
                                 nn.MaxPool3d(kernel_size=2, stride=2),
                                 conv_3d_block(32, 64, kernel_size=args['k1']),
                                 nn.MaxPool3d(kernel_size=2, stride=2),

                                 #block 2
                                 conv_3d_block(64, 128, kernel_size=args['k2']),
                                 conv_3d_block(128, 128, kernel_size=args['k2']),
                                 nn.MaxPool3d(kernel_size=2, stride=2),
                                 nn.AdaptiveAvgPool3d(1),
                                 nn.Flatten()                               
                            )

        if args['n_depth'] ==0:
            pass
        else:
            fc_layers = [[nn.Linear(128 + args['in_features'] , 128 * args['dense_factor']), 
                          nn.BatchNorm1d(128 * args['dense_factor']),
                          nn.ELU(inplace=True), 
                          nn.Dropout(args['dropout_rate'])]]   
            
            if args['n_depth'] > 1:    
                fc_layers.extend([[nn.Linear(128 * args['dense_factor'], 128 * args['dense_factor']),
                                   nn.BatchNorm1d(128 * args['dense_factor']),
                                   nn.ELU(inplace=True),
                                   nn.Dropout(args['dropout_rate'])] for _ in range(args['n_depth'] - 1)])
            
            fc_layers = flatten_layers(fc_layers)
            self.fc_emb = nn.Sequential(*fc_layers)

            if args['model_name'] in ['deephit', 'deepmtlr']:
                self.final_layer =  nn.Linear(128 * args['dense_factor'], args['label_num_duration'])
            elif args['model_name'] in ['deepsurv']:
                self.final_layer =  nn.Linear(128 * args['dense_factor'], 1, bias=False)
            else:
                raise ValueError(f'Unknown model name: {args["model_name"]}')
               
    def forward(self, x):
        img, clin_var = x
        cnn = self.cnn(img)
        # ehr = self.ehr(clin_var)
        
        ftr_concat = torch.cat((cnn, clin_var), dim=1)
        x_emb = self.fc_emb(ftr_concat)
        out = self.final_layer(x_emb)

        return x_emb, out
    
    ## Functions for prediction of survival curves for deephit, deepsurv and deepmtlr, Credits to PyCox
    
    def predict_surv(self, input,  numpy=None):
        if self.args['model_name'] in ['mtlr', 'deepmtlr']:
            pmf = self.predict_pmf(input, False)
            surv = 1 - pmf.cumsum(1)
            return tt.utils.array_or_tensor(surv, numpy, input)
        elif self.args['model_name'] in ['deephit']:
            pmf = self.predict_pmf(input, False)
            surv = 1 - pmf.cumsum(1)
            return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        if self.args['model_name'] in ['mtlr', 'deepmtlr']:
            preds = torch.from_numpy(input)
            preds = utils.cumsum_reverse(preds, dim=1)
            pmf = utils.pad_col(preds).softmax(1)[:, :-1]
            return tt.utils.array_or_tensor(pmf, numpy, input)
        elif self.args['model_name'] in ['deephit']:
            preds = torch.from_numpy(input)
            pmf = utils.pad_col(preds).softmax(1)[:, :-1]
            return tt.utils.array_or_tensor(pmf, numpy, input)

    def predict_surv_df(self, input):
        if self.args['model_name'] in ['mtlr', 'deepmtlr']:
            surv = self.predict_surv(input, True)
            return pd.DataFrame(surv.transpose(), self.duration_index)
        elif self.args['model_name'] in ['deephit']:
            surv = self.predict_surv(input, True)
            return pd.DataFrame(surv.transpose(), self.duration_index)
        elif self.args['model_name'] in ['deepsurv']:
            return np.exp(-self.predict_cumulative_hazards(input, max_duration, baseline_hazards_))

    
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
    
    def _compute_baseline_hazards(self, input, df_target, max_duration, batch_size, eval_=True, num_workers=0):
        if max_duration is None:
            max_duration = np.inf

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(self.predict(input, batch_size, True, eval_, num_workers=num_workers)))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col]/x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def target_to_df(self, target):
        durations, events = tt.tuplefy(target).to_numpy()
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events}) 
        return df

    def compute_baseline_hazards(self, input=None, target=None, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True, eval_=True, num_workers=0):
        """Computes the Breslow estimates form the data defined by `input` and `target`
        (if `None` use training data).

        Typically call
        model.compute_baseline_hazards() after fitting.
        
        Keyword Arguments:
            input  -- Input data (train input) (default: {None})
            target  -- Target data (train target) (default: {None})
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            sample {float or int} -- Compute estimates of subsample of data (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            set_hazards {bool} -- Set hazards in model object, or just return hazards. (default: {True})
        
        Returns:
            pd.Series -- Pandas series with baseline hazards. Index is duration_col.
        """
        if (input is None) and (target is None):
            if not hasattr(self, 'training_data'):
                raise ValueError("Need to give a 'input' and 'target' to this function.")
            input, target = self.training_data
        df = self.target_to_df(target)#.sort_values(self.duration_col)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)
        input = tt.tuplefy(input).to_numpy().iloc[df.index.values]
        base_haz = self._compute_baseline_hazards(input, df, self.max_duration)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def compute_baseline_cumulative_hazards(self, input=None, target=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None,
                                            eval_=True, num_workers=0):
        """See `compute_baseline_hazards. This is the cumulative version."""
        if ((input is not None) or (target is not None)) and (baseline_hazards_ is not None):
            raise ValueError("'input', 'target' and 'baseline_hazards_' can not both be different from 'None'.")
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(input, target, max_duration, sample, batch_size,
                                                             set_hazards=False, eval_=eval_, num_workers=num_workers)
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        bch = (baseline_hazards_
                .cumsum()
                .rename('baseline_cumulative_hazards'))
        if set_hazards:
            self.baseline_hazards_ = baseline_hazards_
            self.baseline_cumulative_hazards_ = bch
        return bch
    
    def _predict_cumulative_hazards(self, input, max_duration, baseline_hazards_,):
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False, 
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(input).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                            index=bch.index)
    
    def predict_cumulative_hazards(self, input, max_duration=None,baseline_hazards_=None):
        """See `predict_survival_function`."""
        if type(input) is pd.DataFrame:
            input = self.df_to_input(input)
        if baseline_hazards_ is None:
            if not hasattr(self, 'baseline_hazards_'):
                raise ValueError('Need to compute baseline_hazards_. E.g run `model.compute_baseline_hazards()`')
            baseline_hazards_ = self.baseline_hazards_
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        return self._predict_cumulative_hazards(input, max_duration, baseline_hazards_)


    
