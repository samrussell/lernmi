#!/usr/bin/env ruby
require './lib/neuron'
require './lib/link'

# Train a neural net to learn the NOT function
# This has one input neuron, one bias neuron, and one output neuron
# We train by setting the input neuron to 1 and backpropagating 0
# We then set the input neuron to 0 and backpropagate 1
# Eventually it will converge

neuron1 = Neuron.new
neuron2 = Neuron.new
bias_neuron = Neuron.new
bias_neuron.input 1.0
initial_weight = 0.5
link = Link.new(initial_weight)
bias_link = Link.new(initial_weight)

link.input_neuron = neuron1
link.output_neuron = neuron2
bias_link.input_neuron = bias_neuron
bias_link.output_neuron = neuron2

1000000.times do |trial|
  neuron1.reset
  neuron2.reset
  neuron1.input 1.0
  link.propagate
  bias_link.propagate
  neuron2.submit_sensitivity (neuron2.output - 0.0)
  link.backpropagate
  bias_link.backpropagate
  puts "Neuron2 expected 0.0 received #{neuron2.output}, weight: #{link.weight} bias: #{bias_link.weight}" if trial % 10000 == 0
  neuron1.reset
  neuron2.reset
  neuron1.input 0.0
  link.propagate
  bias_link.propagate
  neuron2.submit_sensitivity (neuron2.output - 1.0)
  link.backpropagate
  bias_link.backpropagate
  puts "Neuron2 expected 1.0 received #{neuron2.output}, weight: #{link.weight} bias: #{bias_link.weight}" if trial % 10000 == 0
end
