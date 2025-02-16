# CS5720 - Neural Networks Assignment 1 - Question 4

## Overview
This repository contains the code for **Question 4** of the **CS5720 - Neural Networks** assignment. In this task, we explore the effects of different optimizers (Adam and SGD) on the training of an MNIST classifier.

## Objectives
- Train a model using two different optimizers: **Adam** and **SGD**.
- Compare the performance of these optimizers in terms of accuracy.
- Observe and analyze the training/validation accuracy curves.

## Questions to Answer
1. **What patterns do you observe in the training and validation accuracy curves?**
   Answer:

Typically, in the beginning, both the training and validation accuracy start lower and gradually increase.
If the training accuracy increases significantly while the validation accuracy plateaus or decreases, it might indicate overfitting, where the model is learning the training data too well but not generalizing to new data.
If both training and validation accuracy increase steadily and converge, it indicates the model is likely generalizing well.
If the validation accuracy is much lower than training accuracy, this might suggest that the model is underfitting, meaning it hasn’t learned enough from the training data.

   
2. **How can you use TensorBoard to detect overfitting?**
   Answer:

TensorBoard is a powerful tool that allows you to visualize various aspects of your model’s performance during training, such as loss, accuracy, and more.
To detect overfitting, you can monitor the training and validation loss or accuracy curves. If the training accuracy continues to improve while the validation accuracy stagnates or decreases, overfitting might be occurring.
By visualizing these curves in TensorBoard, you can easily detect a divergence between training and validation metrics, which is a clear sign of overfitting.
You can also track other metrics like learning rate and histograms of weights to understand if overfitting is happening, or if the model weights are becoming too large and unstable.

   
3. **What happens when you increase the number of epochs?**
  Answer:

Increasing the number of epochs allows the model to train for a longer time, which can lead to better performance as it learns more from the data.
However, after a certain point, the model may start overfitting if it’s trained for too many epochs, especially if the training loss keeps decreasing but the validation loss starts increasing.
In the case of a well-regularized model, increasing epochs could improve the model’s generalization ability if the model hasn’t converged yet.
It’s often a good idea to use techniques like early stopping to avoid overfitting while still benefiting from more epochs.
