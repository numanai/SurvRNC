import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class HecktorDataset(Dataset):
    def __init__(
        self,
        clinical_data,
        transform,
        args,
    ):
        self.clinical_data = clinical_data
        self.transform = transform
        self.data_path = args['data_path']
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.clinical_data)
    
    def get_targets(self):
        return self.clinical_data['event'].values
    
    def __getitem__(self, idx: int):
        """Get an input-target pair from the dataset.
        
        The images are assumed to be preprocessed and cached.
        
        Parameters
        ----------
        idx : int
            The index to retrieve (note: this is not the subject ID).
            
        Returns
        -------
        tuple
            ((ctpt, x_ehr), y) where:
            - ctpt is the preprocessed CT/PT image tensor
            - x_ehr is the clinical data as a numpy array
            - y is the target data as a numpy array
        """
        
        row_data = self.clinical_data.iloc[idx]
        patient_id = row_data['PatientID']
        
        # Load preprocessed CT/PT image
        ctpt = torch.load(os.path.join(self.data_path, 'processed', 'ctpt', f'{patient_id}_ctpt.pt'))
        
        if self.transform:
            # Apply transformation if specified
            data_dict = {'ctpt': ctpt}
            data_dict = self.transform(data_dict)
            ctpt = data_dict['ctpt']
        
        # Prepare target data
        y = np.array([row_data['y_bin'], row_data['event'], row_data['duration']])
        
        # Prepare clinical data
        # Note: Center ID is excluded as we're doing leave-one-center-out validation
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

class HecktorDataset2Images(HecktorDataset):
    def __getitem__(self, idx: int):
        """
        Get an input-target pair from the dataset with two augmented versions of the same image.
        
        Parameters
        ----------
        idx : int
            The index to retrieve (note: this is not the subject ID).
            
        Returns
        -------
        tuple
            ((ctpt1, ctpt2, x_ehr), y) where:
            - ctpt1, ctpt2 are two augmented versions of the CT/PT image tensor
            - x_ehr is the clinical data as a numpy array
            - y is the target data as a numpy array
        """
        row_data = self.clinical_data.iloc[idx]
        patient_id = row_data['PatientID']
        
        ctpt = torch.load(os.path.join(self.data_path, 'processed', 'ctpt', f'{patient_id}_ctpt.pt'))
        
        if self.transform:
            ctpt1 = self.transform({'ctpt': ctpt})['ctpt']
            ctpt2 = self.transform({'ctpt': ctpt})['ctpt']
        else:
            raise AttributeError("self.transform must be defined for augmentation!")
        
        # Prepare target and clinical data (same as in HecktorDataset)
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
        
        return (ctpt1, ctpt2, x_ehr), y

class HecktorTestDataset(HecktorDataset):
    def __getitem__(self, idx: int):
        """
        Get an input pair and patient ID from the test dataset.
        
        Parameters
        ----------
        idx : int
            The index to retrieve (note: this is not the subject ID).
            
        Returns
        -------
        tuple
            ((ctpt, x_ehr), patient_id) where:
            - ctpt is the preprocessed CT/PT image tensor
            - x_ehr is the clinical data as a numpy array
            - patient_id is the ID of the patient
        """
        row_data = self.clinical_data.iloc[idx]
        patient_id = row_data['PatientID']
        
        ctpt = torch.load(os.path.join(self.data_path, 'ctpt', f'{patient_id}_ctpt.pt'))
        if self.transform:
            ctpt = self.transform(ctpt)
        
        # Prepare clinical data (same as in HecktorDataset)
        x_ehr = np.array([
            row_data['Age'], row_data['Weight'],
            row_data['Chemotherapy'], row_data['Gender_M'],
            row_data['Performance_0.0'], row_data['Performance_1.0'], row_data['Performance_2.0'], 
            row_data['Performance_3.0'], row_data['Performance_4.0'],
            row_data['HPV_0.0'], row_data['HPV_1.0'],
            row_data['Surgery_0'], row_data['Surgery_1'],
            row_data['Tobacco_0.0'], row_data['Tobacco_1.0'], 
            row_data['Alcohol_0.0'], row_data['Alcohol_1.0'],
        ]).astype(np.float32)
        
        return (ctpt, x_ehr), patient_id

