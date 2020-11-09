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
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

# create Neural Network to classify benign og malignant breast cancer in women

device = torch.device('cpu')

# Load the data
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)

y = df['target'].to_numpy()
x = df.drop(["target"],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=85)

scaler = StandardScaler()
transformed_train = scaler.fit_transform(x_train)

train = data_utils.TensorDataset(torch.from_numpy(transformed_train).float(),
                                 torch.from_numpy(y_train).float())
dataloader = data_utils.DataLoader(train, batch_size=128, shuffle=False)

transformed_test = scaler.fit_transform(x_test)
test_set = torch.from_numpy(transformed_test).float()
test_valid = torch.from_numpy(y_test).float()
dimensions_in = x_train.shape[1]
dimensions_out = 1
layer_dimensions = [dimensions_in, 25, 10, dimensions_out]

x = torch.tensor(np.expand_dims(x_train,1), dtype=torch.float32, device=device)
y = torch.tensor(np.expand_dims(y_train,1), dtype=torch.float32, device=device)

H = 5

model = torch.nn.Sequential(
    torch.nn.Linear(30, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,1),
)
model.to(device)
loss_fn = torch.nn.MSELoss(reduction='mean')

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

T = 5000

Loss = np.zeros(T)

for t in range(T):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    Loss[t] = loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step() 

