from sklearn.datasets import load_breast_cancer
import pandas as pd
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# create Neural Network to classify benign og malignant breast cancer in women

# Load the data
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)
x = df.drop(["target"],axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=85)

scaler = StandardScaler()
transformed = scaler.fit_transform(x_train)

train = data_utils.TensorDataset(torch.from_numpy(transformed).float(),
                                 torch.from_numpy(y_train.as_matrix()).float())
dataloader = data_utils.DataLoader(train, batch_size=128, shuffle=False)

print(train)
dimensions_in = x_train.shape[1]
dimensions_out = 1
layer_dimensions = [dimensions_in, 25, 10, dimensions_out]
