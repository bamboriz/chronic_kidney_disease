import pytest
import pandas as pd

from app.data_utils import load_data, get_numeric_cols, get_categorical_cols
from app.data_utils import CATEGORY_THRESHOLD

# A few tests for project code

def test_dataset_loaded_properly():
    df = load_data()
    assert df.empty == False

@pytest.mark.parametrize("input", get_numeric_cols())
def test_numeric_cols_have_continuous_values(input):
    df = load_data()
    assert df[input].nunique() > CATEGORY_THRESHOLD

@pytest.mark.parametrize("input", get_categorical_cols())
def test_catergorical_cols_have_discrete_values(input):
    df = load_data()
    assert df[input].nunique() <= CATEGORY_THRESHOLD