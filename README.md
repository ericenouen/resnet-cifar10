# ResNet Implementation for Cifar-10

The goal for this project was to create an implementation of ResNet as outlined in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

I took advantage of a program made from a user on [medium](https://medium.com/@joeyism/creating-alexnet-on-tensorflow-from-scratch-part-1-getting-cifar-10-data-46d349a4282f) in order to be able to download the Cifar-10 dataset and work with it easily.

From there I created the model in Keras. The paper outlined the structure of the model to follow the table below. Essentially, there are blocks that contain two convolutions and a skip connection. Stacks of these blocks following the instructions of the table below create the neural network.
![image](https://user-images.githubusercontent.com/54828661/94476241-39233000-019e-11eb-89fd-4d50d47cac87.png)

The following will analyze the difference between adding the skip connections compared to a regular deep neural network. As outlined in the paper, the regular deep network should experience problems optimizing as more and more layers are added on because the loss function becomes increasingly complex to minimize. The addition of skip connections, which allow the neural network to either disregard or minimize the results of any of the layers that have an accompanying skip connection.

# Graphical Comparison

## Percent Error Comparison from my Implementation
![alt text](Graph.jpeg)
## Percent Error Comparison from the Paper
![image](https://user-images.githubusercontent.com/54828661/94475184-b8176900-019c-11eb-8366-a140f9df1cea.png)

It's clear to see that in both graphs there is a clear difference between that of regular models and models with the residual connections added. It becomes incredibly hard to optimize a neural network the deeper it gets, and that is clearly shown in the explosion of the percent error in the plain networks.
Some of the main differences are that in my implementation I was not able to get the model to optimize as quickly as was described in the paper and I resorted to using more epochs to getting the same results. I also was not able to train the ResNet-44 and ResNet-56 nearly as well as the paper was and ResNet-32 was actually my best performing model. I assume that a large part of this was probably me not having the ability to repeatedly retrain the models until I received the results that I wanted, whereas the paper was able to do so. I also couldn't discover all of the tricks that they used and I was not able to train any of my models to the exactly correct accuracy, even though some are very close.

# Table Comparison

