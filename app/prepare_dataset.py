import pandas as pd
import numpy as np
import logging
import sys

from data_utils import DATA_URL

def load_data(path: str) -> pd.DataFrame:
    '''
    Reads .arff file and  outputs a cleaner .csv file in data folder
    '''
    data = []
    logging.info('Processing chronic_kidney_disease_full.arff file')
    with open(path, "r") as f:
        for line in f:
            line = line.replace('\n', '')
            data.append(line.split(','))

    names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu',  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc',
            'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane',
            'class', 'no_name']

    df = pd.DataFrame(data[145:], columns=names)
    df = df.drop(labels=[400, 401], axis=0)
    df = df.drop('no_name', axis=1)
    cols_names = get_full_col_names()
    df.rename(columns=cols_names, inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = fix_bad_data(df)
    df.to_csv(DATA_URL)
    logging.info('Processing complete')
    return df

def get_full_col_names() -> dict:
    '''
    Human readable names for better data exploration
    '''
    return { "age": "age", "bp":"blood_pressure", "sg":"specific_gravity", "al":"albumin", "su":"sugar", "rbc":"red_blood_cells",
            "pc":"pus_cell", "pcc":"pus_cell_clumps", "ba":"bacteria", "bgr":"blood_glucose_random", "bu":"blood_urea",
            "sc":"serum_creatinine", "sod":"sodium", "pot":"potassium", "hemo":"haemoglobin", "pcv":"packed_cell_volume",
            "wc":"white_blood_cell_count", "rc":"red_blood_cell_count", "htn":"hypertension", "dm":"diabetes_mellitus",
            "cad":"coronary_artery_disease", "appet":"appetite", "pe":"pedal_edema", "ane":"anemia", "class": "ckd"}

def fix_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['ckd'].isin(['ckd', 'notckd'])]
    df = df[df['coronary_artery_disease'].isin(['yes', 'no'])]
    df = df[df['diabetes_mellitus'].isin(['yes', 'no'])]
    return df

if __name__ == '__main__':
    if len(sys.argv) == 2:
        response = load_data(sys.argv[1])
    else:
        print('Please enter the path to the .arrf file')