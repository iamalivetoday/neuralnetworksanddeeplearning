import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#we are training the model to learn a piecewise function where:


X = torch.arange(-30, 30, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0]) #tensor with same number of rows as X

Y[(X[:, 0] <= -10)] = 1.0 #sets Y to 1.0 where X is less than or equal to -10.
Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 0.5
Y[(X[:, 0] > 10)] = 0 #sets Y to 0 where X > 10

#plt.plot(X, Y)
#plt.show()

#class for single layer NN
class one_layer_net(torch.nn.Module):
  def __init__(self, input_size, hidden_neurons, output_size):
    super(one_layer_net, self).__init__()
    self.linear_one = torch.nn.Linear(input_size, hidden_neurons) #first hidden layer with input_size inputs and hidden_neurons outputs
    self.linear_two = torch.nn.Linear(hidden_neurons, output_size) #second hidden layer with hidden_neurons inputs and output_size outputs
    self.layer_in = None #intermediate layers 
    self.act = None
    self.layer_out = None

  def forward(self, x): #forward pass of network
    self.layer_in = self.linear_one(x) #first linear transformation to the input
    self.act = torch.sigmoid(self.layer_in) #sigmoid to the result of the first linear transformationgit 
    self.layer_out = self.linear_two(self.act) #second linear transformation to activated values
    y_pred = torch.sigmoid(self.linear_two(self.act)) #sigmoid activation function to output of second linear layer to get the final pred
    return y_pred 

model = one_layer_net(1, 2, 1)

#types of distribution
#bernoulli: p, 1-p, single trial
#uniform: all outcomes equally likely
#normal: binary data from an infinite sample (continuous)
#        use mean squared error
#binomial: binary data from a finite sample (discrete) --> getting r events out of n trials 
#          two outcomes possible, rate of failure is same for all trials
#          2 parameters: number of trials (n) and success probability (p)
#          mean = np, variance = npq
#          use cross entropy
#poisson: infinite outcomes possible 
#         probability of an event happening k times within a given interval of time or space
#          1 parameter: m (mean number of events)

#Cross-entropy is prefered for classification (predicting labels/categories), while mean squared error for regression (predicting continuous values)

def criterion(y_pred, y):
  #calculate the binary cross entropy
  #binary cross entropy is our loss function for sgd
  #remember, probabilities are between 0 and 1. log scales these values and spreads them out over a larger range, making the differences more significant, and penalizes wrong predictions more heavily, and makes convex loss landscape
  #convex loss landscape is a shape of the loss function that has a single global minimum and no local minima
  out = -1 * torch.mean(y * torch.log(y_pred) + (1-y) * torch.log(1 - y_pred))
  #y is the actual label
  return out
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#binary cross entropy loss measures the difference between predicted probabilities and actual binary labels
#basically MSE for classification 

epochs = 5000  # number of epochs to train the model
cost = []  # list to store the total loss for each epoch
total = 0  # variable to accumulate the total loss for each epoch

for epoch in range(epochs):
    print(str(epoch) + "epoch")
    total = 0  # reset the total loss for the current epoch
    epoch = epoch + 1  # increment the epoch count (not really necessary as range() handles this)
    
    for x, y in zip(X, Y):  # iterate over each data point (x, y) in the dataset
        yhat = model(x)  # forward pass: compute the model's prediction for input x
        loss = criterion(yhat, y)  # compute the loss between the prediction and the actual label
        loss.backward()  # backward pass: compute the gradients of the loss w.r.t. model parameters
        optimizer.step()  # update the model parameters using the gradients
        optimizer.zero_grad()  # reset the gradients to zero before the next iteration
        
        total += loss.item()  # accumulate the loss for the current epoch
    
    cost.append(total)  # store the total loss for the current epoch
    
    if epoch % 1000 == 0:
        print(str(epoch) + " epochs done!")  # print a message every 1000 epochs to track progress
        
        # visualize the results after every 1000 epochs
        plt.plot(X.numpy(), model(X).detach().numpy())  # plot the model's predictions
        plt.plot(X.numpy(), Y.numpy(), 'm')  # plot the actual labels
        plt.xlabel('x')  # label the x-axis
        plt.show()  # display the plot
