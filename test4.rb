#!/usr/bin/env ruby
require './lib/neuron'
require './lib/link'
require './lib/neuron_layer'
require './lib/link_layer'

# Train a neural net to learn the XOR function
# This has two input neurons, one bias neuron, and two output neurons
# We train by setting the input neurons to (1, 0), (0, 1) and backpropagating 1
# We then set the inputs neuron to (0, 0), (1, 1) and backpropagate 0
# Eventually it will converge

neuron_layers = [
  NeuronLayer.new(2),
  NeuronLayer.new(3),
  NeuronLayer.new(1)
]

link_layers = [
  LinkLayer.new(neuron_layers[0], neuron_layers[1]),
  LinkLayer.new(neuron_layers[1], neuron_layers[2])
]

training_data = [
  [[0.0, 0.0], 0.0],
  [[1.0, 0.0], 1.0],
  [[0.0, 1.0], 1.0],
  [[1.0, 1.0], 0.0]
]

10000000.times do |trial|
  training_data.each do |inputs, expected_output|
    neuron_layers.each(&:reset)

    input_neurons = neuron_layers.first.learning_neurons

    input_neurons.zip(inputs).each do |neuron, input|
      neuron.input input
    end

    link_layers.each do |link_layer|
      link_layer.propagate
    end

    output_neurons = neuron_layers.last.learning_neurons

    output_neurons.each do |neuron|
      neuron.submit_sensitivity (neuron.output - expected_output)
    end

    link_layers.reverse.each do |link_layer|
      link_layer.backpropagate
    end

    if trial % 10000 == 0
      puts "Input #{inputs[0]} #{inputs[1]} output #{output_neurons.first.output} expected #{expected_output}"
    end
  end
end
