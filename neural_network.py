from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#               Create Neural Network to classify benign og malignant breast cancer in women

# pd.set_option('display.max_columns', None)

if torch.cuda.is_available():
    device = torch.device("cuda")

# Load the data
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = np.array(data.target)
# Create train/test split (80/20) using random_state to create replicable results
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

# Initiate scaler used for feature standardization
scaler = StandardScaler()

# Scale and transform training data
transformed = scaler.fit_transform(x_train)
train = data_utils.TensorDataset(torch.from_numpy(transformed).float(),torch.from_numpy(y_train).float())
dataloader = data_utils.DataLoader(train, batch_size=120, shuffle=False)

# Scale and transform test data
transformed = scaler.fit_transform(x_test)
test_set = torch.from_numpy(transformed).float()
test_valid = torch.from_numpy(y_test).float()

# Neural Network specifications
H = 6

dimemsions_in = x_train.shape[1]
dimemsions_out = 1

# The Neural Network
model = torch.nn.Sequential(
    torch.nn.Linear(dimemsions_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, dimemsions_out),)

# Optimizer, learning rate, loss function and number of iterations
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.0004
num_epochs = 200
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Run the NN and track accuracy, loss, etc. throughout each epoch.
nn_log = {"loss": [], "accuracy": [], "loss_value": [], "accuracy_value": []}
for epoch in range(num_epochs):
    loss = None
    for index, (batch, target) in enumerate(dataloader):
        y_pred = model(Variable(batch))
        loss = loss_fn(y_pred, Variable(target.float()))
        prediction = [1 if x > 0.5 else 0 for x in y_pred.data.numpy()]
        correct = (prediction == target.numpy()).sum()
        y_val_pred = model(Variable(test_set))
        loss_value = loss_fn(y_val_pred, Variable(test_valid.float()))
        prediction_value = [1 if x > 0.5 else 0 for x in y_val_pred.data.numpy()]
        correct_val = (prediction_value == test_valid.numpy()).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Append metrics for each epoch
    nn_log["loss"].append(loss.item())
    nn_log["accuracy"].append(100 * correct / len(prediction))
    nn_log["loss_value"].append(loss_value.item())
    nn_log["accuracy_value"].append(100 * correct_val / len(prediction_value))

    # Print the metrics for each epoch
    print("Loss, accuracy, val loss, val acc at epoch", epoch + 1, nn_log["loss"][-1],
          nn_log["accuracy"][-1], nn_log["loss_value"][-1], nn_log["accuracy_value"][-1])

# Find the highest accuracy value
index, value = max(enumerate(nn_log["accuracy_value"]), key=operator.itemgetter(1))

print("Higgest accuracy score: {} at iteration: {}".format(value, index))

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
