import torch
import sklearn.datasets
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# create Neural Network to classify benign og malignant breast cancer in women

# Load the data
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)

drop_columns = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
diagnosis = {"M": 1, "B": 0}
y = data["diagnosis"].replace(diagnosis)


"""
dimensions_in = x_train.shape[1]
dimensions_out = 1
layer_dimensions = [dimensions_in, 25, 10, dimensions_out]
"""