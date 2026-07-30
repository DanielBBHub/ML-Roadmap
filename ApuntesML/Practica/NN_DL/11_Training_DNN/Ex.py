""" Exercises
1. What is the problem that Glorot initialization and He initialization aim to fix?

2. Is it OK to initialize all the weights to the same value as long as that value is
selected randomly using He initialization?

3. Is it OK to initialize the bias terms to 0?

4. In which cases would you want to use each of the activation functions we
discussed in this chapter?

5. What may happen if you set the momentum hyperparameter too close to 1 (e.g.,
0.99999) when using an SGD optimizer?

6. Name three ways you can produce a sparse model.

7. Does dropout slow down training? Does it slow down inference (i.e., making
predictions on new instances)? What about MC dropout?

8. Practice training a deep neural network on the CIFAR10 image dataset:
    a. Load CIFAR10 just like you loaded the FashionMNIST dataset in Chapter 10,
    but using torchvision.datasets.CIFAR10 instead of FashionMNIST. The
    dataset is composed of 60,000 32 × 32–pixel color images (50,000 for training,
    10,000 for testing) with 10 classes.

    b. Build a DNN with 20 hidden layers of 100 neurons each (that’s too many, but
    it’s the point of this exercise). Use He initialization and the Swish activation
    function (using nn.SiLU). Since this is a classification task, you will need an
    output layer with one neuron per class.

    c. Using NAdam optimization and early stopping, train the network on the
    CIFAR10 dataset. Remember to search for the right learning rate each time
    you change the model’s architecture or hyperparameters.

    d. Now try adding batch-norm and compare the learning curves: is it converging
    faster than before? Does it produce a better model? How does it affect training
    speed?

    e. Try replacing batch-norm with SELU, and make the necessary adjustments to
    ensure the network self-normalizes (i.e., standardize the input features, use
    LeCun normal initialization, make sure the DNN contains only a sequence of
    dense layers, etc.).

    f. Try regularizing the model with alpha dropout. Then, without retraining your
    model, see if you can achieve better accuracy using MC dropout.
    
    g. Retrain your model using 1cycle scheduling and see if it improves training
    speed and model accuracy.
    """