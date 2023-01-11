# FOAI_Neural_Net
Training a Neural net using PyTorch Library<br />
demo- code-1-Ram<br />
Explanation of the code: <br />

First, we use torch.linspace to generate an array of X between 0 to 4 with a resolution of 0.01. <br />
Then use the provided equation to calculate the corresponding Y values.<br />
Then we add noise to the Y array using the torch.randn function with a standard deviation of 1.<br />
After that, we split the X and Y_noisy into training (60%) and validation (40%) sets.<br />
Then, we defined the RegressionModel class which is inherited from the Pytorch nn.Module and it contains the linear model which has 1 input and 1 output neuron.<br />
After that, we defined a mean squared error as "a" <br />

