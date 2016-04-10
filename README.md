# Lernmi
Learn neural nets while using them

[![Build Status](https://travis-ci.org/samrussell/lernmi.svg?branch=master)](https://travis-ci.org/samrussell/lernmi)
[![Code Climate](https://codeclimate.com/github/samrussell/lernmi/badges/gpa.svg)](https://codeclimate.com/github/samrussell/lernmi)
[![Test Coverage](https://codeclimate.com/github/samrussell/lernmi/badges/coverage.svg)](https://codeclimate.com/github/samrussell/lernmi/coverage)

# Getting started
## What is a neural network?
A neural network is a collection of Neurons, connected by weighted Links. Signals propagate through the network from the input to the output, and we can train these neural networks to recognise certain patterns by training them on test data, and backpropagating the difference between what we expected the network to recognise, and what it actually thought it had found.

## How can I build one?
Look through the test apps in the main directory - they start off really simple, making some logic gates (NOT, OR, XOR), and then get into the MNIST handwriting recognition set. I know, it looks like a huge jump, but there's not really much middle ground. I've tried to make the code intuitive to follow, so if you're getting freaked out at the thought of training a neural network to recognise handwriting, try to spend a bit of time looking at the code in the simpler examples - the handwriting sample is the exact same thing, just with a lot more neurons and links.

# Moving parts in lernmi
## Neuron
A Neuron is responsible for combining values from Links, and providing this sum to anything that requests it. In practise, a Neuron receives the `#input` message from each Link connecting it to the previous layer, sums these values, applies an activation function, and provides the result to any Link that calls its `#output` method. In reverse, a neuron receives the `#submit_sensitivity` message from each Link that connects it to the subsequent layer, sums these values, and provides them through the `#previous_layer_sensitivity_sum` method (what a lovely name!).

To get things back to normal, send a `#reset` message.

If you want a bias neuron, send a `#bias` message.

## Link
A Link is responsible for propagating the weighted output from one neuron to another, backpropagating the weighted sensitivities from one neuron to another, and updating its own weight when backpropagating. Use the `#propagate` message to propagate, and the `#backpropagate` message to backpropagate.

## NeuronLayer
A NeuronLayer is responsible for grouping Neurons into layers. Initialize with however many learning neurons you want, it will create this many, plus a single bias neuron. Pass an array to the `#input` method to load all the learning neurons, and pass an array to the `#expected_output` method in the output layer as part of training. The `#reset` method resets all neurons in the layer.

## LinkLayer
A LinkLayer is responsible for grouping Links into layers. Initialize with an input and output NeuronLayer, and it will generate Links with default parameters. The `#propagate` and `#backpropagate` methods pass through to each Link in the LinkLayer.

## Network
A Network is responsible for managing a full neural network, via LinkLayers and NeuronLayers. It is initialized with an array of neurons per layer, and implements a `#learn` method that runs a single training epoch with provided data.

# Design decisions
## Using Neurons and Links
Most neural network implementations use a vector for each layer of neurons, and a matrix to hold the weights between any given two layers. This makes the math somewhat elegant and makes it more efficient to calculate (read: it runs really fast on a GPU), but it really raises the barrier to entry. I wanted to build a neural network that could be understood by someone who had a basic grasp of addition and multiplication, and matrix/vector arithmetic is quite intimidating if you're not used to using it.

Abstracting the network into Neurons and Links was useful. Neurons sum inputs and apply an activation fuction when propagating, and only need to sum sensitivities when backpropagating. A Link pulls a value from one Neuron and pushes it into another.

I had originally just had Neurons that each had references to multiple other Neurons, and then references backwards (to handle backpropagation), something which created a mess of circular dependencies and made Neurons quite confusing. Introducing the Link class solved the dependency problem and simplified Neurons a little bit.

## Having Links drive propagation/backpropagation
In real life, a neuron will emit a signal, which travels along some medium to another neuron. This abstraction is not helpful here. In a previous version, you would bump a Neuron to send data to its Link, then bump the Link to send data to its next Neuron, but this was quite yuck and still had a bunch of circular dependencies.

Under this model, Neurons hold data (and do a *little* bit of processing); Links move data (and also do a little processing where appropriate).

## Handling sensitivity outside of Neurons
The calculus is a bit complex and it took a couple weeks of reading to get my head around it. Fortunately, if we factor the math correctly, we can do a bunch of sensitivity calculation, pass these values to a Neuron, and all it has to do is sum the values and provide the result to the next Link in the chain. By doing it this way, propagation and backpropagation look very similar from the perspective of a Neuron.

## Splitting the activation function and its derivative across the Neuron and Link classes
Bias neurons.

If we look at the activation function derivative (sigmoid prime), we need to implement this in the Link class, as it's the only thing that knows the weight, the un-weighted output of the previous neuron (which has been passed through the sigmoid function already), and the summed sensitivities from the neuron in front. The Link is the logical home for the activation function derivative.

What about the activation function (sigmoid)? It also totally belongs in the Link class, but that makes bias neurons a bit yuck. You see, a bias neuron always outputs the value 1.0, but if the sigmoid function is in the Link, then we need the bias neuron to output a value x, such that sigmoid(x) == 1.0. That number is +infinity, although 999 and 9999 get pretty close. If we move the sigmoid function into the Neuron class, it's still in an okay place, but it now means our bias neurons can just output 1.0, and it feels a bit nicer.

I could be convinced to put the sigmoid into the Link class if there was a nice way to deal with bias neurons. We're a bit spoilt that sigmoid prime is a function of sigmoid(x)... if we were using tanh or something else then the Link class would need the raw output from Neuron for calculating sensitivities.

