# FOAI_Neural_Net
Training a Neural net using PyTorch Library
demo- code-1
Explanation of the code:

First, we use torch.linspace to generate an array of X between 0 to 4 with a resolution of 0.01. Then use the provided equation to calculate the corresponding Y values.
Then we add noise to the Y array using the torch.randn function with a standard deviation of 1.
After that, we split the X and Y_noisy into training (60%) and validation (40%) sets.
Then, we defined the RegressionModel class which is inherited from the Pytorch nn.Module and it contains the linear model which has 1 input and 1 output neuron.
After that, we defined a mean squared error as a

