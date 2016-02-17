#!/usr/bin/env ruby
require './lib/neuron'
require './lib/link'
require './lib/neuron_layer'
require './lib/link_layer'
require './lib/idx_loader'
require 'zlib'

# Train a neural net to learn MNIST numbers
# This has 28 x 28 = 784 input neurons, one bias neuron, and 10 output neurons
# Eventually it will converge

neuron_layers = [
  NeuronLayer.new(784),
  NeuronLayer.new(15),
  NeuronLayer.new(10)
]

link_layers = [
  LinkLayer.new(neuron_layers[0], neuron_layers[1]),
  LinkLayer.new(neuron_layers[1], neuron_layers[2])
]

#training_data = [
#  [[0.0, 0.0], 0.0],
#  [[1.0, 0.0], 1.0],
#  [[0.0, 1.0], 1.0],
#  [[1.0, 1.0], 0.0]
#]

puts "Loading MNIST data"

imagefile = File.open("assets/t10k-images-idx3-ubyte.gz")
imagestream = Zlib::GzipReader.new(imagefile)
images = IdxLoader.load(imagestream)

labelfile = File.open("assets/t10k-labels-idx1-ubyte.gz")
labelstream = Zlib::GzipReader.new(labelfile)
labels = IdxLoader.load(labelstream)

puts "Loaded"

training_data = images.zip(labels)

100.times do |trial|
  training_data.each.with_index do |(inputs, label), label_index|
    input_neurons = neuron_layers.first.learning_neurons

    input_neurons.each &:reset

    input_neurons.zip(inputs).each do |neuron, input|
      neuron.input input
    end

    link_layers.each do |link_layer|
      link_layer.propagate
    end

    output_neurons = neuron_layers.last.learning_neurons

    output_neurons.each.with_index do |neuron, index|
      expected_output = (index == label) ? 1.0 : 0.0
      neuron.submit_sensitivity (neuron.output - expected_output)
    end

    link_layers.reverse.each do |link_layer|
      link_layer.backpropagate
    end

    if label_index % 100 == 0
      output_map = output_neurons.map { |neuron| "%0.1f" % neuron.output }
      puts "Input output #{output_map} expected #{label}"
    end
  end
end

