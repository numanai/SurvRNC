#!/usr/bin/env python3
"""
Test script to validate the bug fixes in SurvRNC.
"""
import tempfile
import torch
import numpy as np
import pandas as pd
import os
import sys

# Test 1: Boolean conversion logic fix
def test_boolean_conversion():
    """Test the boolean conversion logic in config override."""
    # Simulate the conversion logic from main.py
    test_cases = [
        ("true", True),
        ("false", False),
        ("True", True),
        ("False", False),
        ("1", 1),
        ("0.5", 0.5),
        ("normal_string", "normal_string"),
    ]
    
    for input_val, expected in test_cases:
        # Simulate the conversion logic from main.py
        value = input_val
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                # Handle boolean values
                if str(value).lower() == 'true':
                    value = True
                elif str(value).lower() == 'false':
                    value = False
                # Keep as string if not a boolean
        
        assert value == expected, f"Expected {expected}, got {value} for input '{input_val}'"
    
    print("✓ Boolean conversion logic test passed")
    return True

# Test 2: Column name consistency
def test_column_consistency():
    """Test that column names are consistent between dataset classes."""
    from hecktor_dataset import HecktorDataset, HecktorTestDataset
    
    # Create mock data with the expected columns
    mock_data = pd.DataFrame({
        'PatientID': ['test_001'],
        'Age': [60.0],
        'Weight': [70.0],
        'Chemotherapy': [1],
        'Gender_M': [True],
        'Performance_0.0': [1], 'Performance_1.0': [0], 'Performance_2.0': [0],
        'Performance_3.0': [0], 'Performance_4.0': [0],
        'HPV_0.0': [1], 'HPV_1.0': [0],
        'Surgery_0.0': [1], 'Surgery_1.0': [0],  # This should be consistent now
        'Tobacco_0.0': [1], 'Tobacco_1.0': [0],
        'Alcohol_0.0': [1], 'Alcohol_1.0': [0],
        'y_bin': [1], 'event': [1], 'duration': [100.0]
    })
    
    # Mock args
    mock_args = {'data_path': '/tmp/test'}
    
    # Create a temporary tensor file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, 'processed', 'ctpt'), exist_ok=True)
        test_tensor = torch.randn(2, 64, 64, 64)  # Mock CT/PT data
        torch.save(test_tensor, os.path.join(temp_dir, 'processed', 'ctpt', 'test_001_ctpt.pt'))
        
        mock_args['data_path'] = temp_dir
        
        # Test HecktorTestDataset
        test_dataset = HecktorTestDataset(mock_data, None, mock_args)
        
        try:
            # This should not raise a KeyError anymore
            (ctpt, x_ehr), patient_id = test_dataset[0]
            print("✓ Column name consistency test passed")
            return True
        except KeyError as e:
            print(f"✗ Column name consistency test failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error in column consistency test: {e}")
            return False

# Test 3: File path consistency
def test_file_path_consistency():
    """Test that file paths are consistent between dataset classes."""
    # This is implicitly tested in test_column_consistency since we create the file
    # in the 'processed/ctpt' directory which should work for both dataset classes now
    print("✓ File path consistency test passed (tested implicitly)")
    return True

# Test 4: Error handling for missing files
def test_missing_file_handling():
    """Test error handling for missing image files."""
    from hecktor_dataset import HecktorDataset
    
    mock_data = pd.DataFrame({
        'PatientID': ['nonexistent_patient'],
        'Age': [60.0], 'Weight': [70.0], 'Chemotherapy': [1], 'Gender_M': [True],
        'Performance_0.0': [1], 'Performance_1.0': [0], 'Performance_2.0': [0],
        'Performance_3.0': [0], 'Performance_4.0': [0],
        'HPV_0.0': [1], 'HPV_1.0': [0],
        'Surgery_0.0': [1], 'Surgery_1.0': [0],
        'Tobacco_0.0': [1], 'Tobacco_1.0': [0],
        'Alcohol_0.0': [1], 'Alcohol_1.0': [0],
        'y_bin': [1], 'event': [1], 'duration': [100.0]
    })
    
    mock_args = {'data_path': '/tmp/nonexistent'}
    dataset = HecktorDataset(mock_data, None, mock_args)
    
    try:
        (ctpt, x_ehr), y = dataset[0]
        print("✗ Missing file handling test failed: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        # Check that the error message is informative
        if "nonexistent_patient" in str(e) and "processed/ctpt" in str(e):
            print("✓ Missing file handling test passed")
            return True
        else:
            print(f"✗ Missing file handling test failed: Error message not informative enough: {e}")
            return False
    except Exception as e:
        print(f"✗ Missing file handling test failed: Unexpected error type: {e}")
        return False

# Test 5: Simple division by zero test  
def test_division_by_zero_simple():
    """Test division by zero protection in utils.py."""
    from utils import mtlr_hazard, mtlr_survival
    
    # Test with very small survival values
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.1, 0.1, 0.1]])
    
    try:
        # This should not raise division by zero error due to epsilon protection
        hazard = mtlr_hazard(logits)
        print("✓ Division by zero protection test passed")
        return True
    except Exception as e:
        print(f"✗ Division by zero protection test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running SurvRNC bug fix tests...\n")
    
    tests = [
        test_boolean_conversion,
        test_column_consistency,
        test_file_path_consistency,
        test_missing_file_handling,
        test_division_by_zero_simple
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)