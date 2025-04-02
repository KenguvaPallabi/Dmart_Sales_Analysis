import pandas as pd

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def clean_data(df):
    df.fillna(df.mean(), inplace=True)
    return df
