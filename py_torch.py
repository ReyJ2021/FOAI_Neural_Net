import torch
import numpy as np
import pandas as pd

#generating datasets
X_data = torch.linspace(0, 10, 1000)
Y_data = 2 * X_data**3 - X_data**2 - 7*X_data + 6

#adding noise to the Y_data set
noise = torch.normal(mean=0, std=1, size=Y_data.shape)
Y = Y_data + noise

Y_Noisy = Y

X_train, X_val = X_data[:600], X_data[600:]
Y_train, Y_val = Y_Noisy[:600], Y_Noisy[600:]


# np.set_printoptions(precision=4)
# print("X_train:", X_train)
# print("X_val:", X_val)

X_train_pd=pd.DataFrame(X_train)
print(X_train_pd)

