from sklearn.datasets import load_breast_cancer
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_breast_cancer()
x, y = data.data, data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

dim_input = x_train.shape[1]

# Neural network model
model = torch.nn.Linear(dim_input, 1)

# Configure loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

# Number of epochs
num_epochs = 300
optimizer = torch.optim.Adam(model.parameters())


x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1,1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1,1)

nn_log = {"loss": [], "accuracy": [], "loss_value": [], "accuracy_value": []}

for epoch in range(num_epochs):
    y_pred = model(x_train)
    p_train = (y_pred.detach().numpy() > 0)
    train_accuracy = np.mean(y_train.numpy() == p_train)

    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_val_pred = model(x_test)
    loss_value = loss_fn(y_val_pred,y_test)
    p_test = (y_val_pred.detach().numpy() > 0)
    test_accuracy = np.mean(y_test.numpy() == p_test)

    # Append metrics for each epoch
    nn_log["loss"].append(loss.item())
    nn_log["accuracy"].append(train_accuracy*100)
    nn_log["loss_value"].append(loss_value.item())
    nn_log["accuracy_value"].append(test_accuracy*100)

    # Print the metrics for each epoch
    print("Loss, accuracy, val loss, val acc at epoch", epoch + 1, nn_log["loss"][-1],
          nn_log["accuracy"][-1], nn_log["loss_value"][-1], nn_log["accuracy_value"][-1])

# Plot accuracy by epochs
plt.plot(nn_log['accuracy'])
plt.plot(nn_log['accuracy_value'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot loss by epochs
plt.plot(nn_log['loss'])
plt.plot(nn_log['loss_value'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
