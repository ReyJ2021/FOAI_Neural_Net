import torch
import numpy as np

# Generate dataset
X = torch.linspace(0, 4, 400)
Y = 2*X**3 - X**2 - 7*X + 6
Y_noisy = Y + torch.randn(Y.size())*1

# Split dataset into training and validation sets
X_train, Y_train, X_val, Y_val = X[:240], Y_noisy[:240], X[240:], Y_noisy[240:]

# Define the model
class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Instantiate the model
model = RegressionModel()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(X_train.reshape(-1, 1))
    labels = torch.from_numpy(Y_train.reshape(-1, 1))

    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward and optimize
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    inputs = torch.from_numpy(X_val.reshape(-1, 1))
    labels = torch.from_numpy(Y_val.reshape(-1, 1))
    outputs = model(inputs)
    error = outputs - labels
    mean_squared_error = (error ** 2).mean()
    print(f'Mean squared error: {mean_squared_error.item():.4f}')
