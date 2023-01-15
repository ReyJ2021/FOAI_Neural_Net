import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


#generating datasets
X_data = torch.linspace(0, 4, 400)
Y_data = 2 * X_data**3 - X_data**2 - 7*X_data + 6

#adding noise to the Y_data set
noise = torch.normal(mean=0, std=1, size=Y_data.shape)
Y = Y_data + noise

Y_Noisy = Y

X_train, X_val = X_data[:241], X_data[241:]
Y_train, Y_val = Y_Noisy[:241], Y_Noisy[241:]


# np.set_printoptions(precision=4)
# print("X_train:", X_train)
# print("X_val:", X_val)

# X_train_pd=pd.DataFrame(X_train)
# print(X_train_pd)

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        x = self.linear(x)
        return x


# Define a loss function and optimizer
model = RegressionModel()
criterion = nn.MSELoss()
mae_loss = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):

    inputs = X_train.reshape(-1,1)
    labels = Y_train.reshape(-1,1)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss_1 = mae_loss(outputs, labels)

    loss.backward()
    optimizer.step()


# model validation
with torch.no_grad():
    inputs = X_val.reshape(-1,1)
    labels = Y_val.reshape(-1,1)
    outputs = model(inputs)
    val_loss = criterion(outputs, labels)
    val_mae = mae_loss(outputs, labels)
    # predictions = outputs.numpy().flatten()
    # true_values = labels.numpy().flatten()

Y_pred = outputs.numpy().flatten()
Y_calc = Y_val.numpy()
error = Y_pred - Y_calc
accuracy = np.mean(error**2)


print(f'Mean squared error: {accuracy.item():.4f}')
print("Validation MAE:", val_mae)
# MAE produces a lower value than MSE metric

plt.scatter(X_val, true_values, label='True Values')
plt.scatter(X_val, predictions, label='Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
