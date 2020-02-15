import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import TensorDataset


def adult_income():
    # load data-set, and remove columns which are irrelevant:
    #   fnlwgt - sampling weight. not something which should predict income
    #   education - string repr. of education-num. we don't need both.
    adult = pd.read_csv('adult.csv')
    adult = adult.drop(['fnlwgt', 'education'], axis=1)

    num_columns = adult.select_dtypes(include=['int']).columns
    cat_columns = adult.select_dtypes(include=['object']).columns

    # one-hot encode categorical columns
    adult = pd.get_dummies(adult, columns=cat_columns)
    # for categorical columns with only two values - drop one of the values
    adult = adult.drop(labels=['income_<=50K', 'gender_Female'], axis=1)

    # standardize (scale) numerical columns
    for col in num_columns:
        scaled = preprocessing.minmax_scale(adult[col])
        adult[col] = scaled

    # create training set
    X_tensor = torch.tensor(adult.drop(labels=['income_>50K'], axis=1).values)
    Y_tensor = torch.tensor(adult['income_>50K'].values)

    ds = TensorDataset(X_tensor, Y_tensor)
    dim = X_tensor.shape[1]

    return ds, dim
