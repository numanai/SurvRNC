!/bin/bash
Preprocess CT and PT images
python ctpt_preprocess.py \
--data_path ./data/raw \
--save_dir ./data/processed/ctpt \
--space_x 2 \
--space_y 2 \
--space_z 2 \
--a_min -250 \
--a_max 250 \
--b_min 0 \
--b_max 1 \
--seed 1234



*Make sure to give execute permissions to the script:*

bash
chmod +x scripts/preprocess.sh


### `tests/test_utils.py`

python:tests/test_utils.py
import unittest
import torch
from utils import normalize, reset_parameters
class TestUtils(unittest.TestCase):
def test_normalize(self):
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
normalized_data, mean, std = normalize(data, mean=torch.tensor([2.0, 3.0]), std=torch.tensor([1.0, 1.0]))
expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
self.assertTrue(torch.equal(normalized_data, expected))
def test_reset_parameters(self):
model = torch.nn.Linear(2, 2)
original_weights = model.weight.clone()
model.reset_parameters()
self.assertFalse(torch.equal(model.weight, original_weights))
if name == 'main':
unittest.main()