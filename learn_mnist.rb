#!/usr/bin/env ruby

require "./lib/mnist_learner"

mnist_learner = MnistLearner.new("assets/t10k-images-idx3-ubyte.gz", "assets/t10k-labels-idx1-ubyte.gz")

mnist_learner.run
