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
bias_neuron.propagate 1.0
initial_weight = 0.5
link = Link.new(initial_weight)
bias_link = Link.new(initial_weight)

neuron1.output_links << link
neuron2.input_links << link
bias_neuron.output_links << bias_link
neuron2.input_links << bias_link
link.input_neurons << neuron1
link.output_neurons << neuron2
bias_link.input_neurons << bias_neuron
bias_link.output_neurons << neuron2

1000000.times do |trial|
  neuron1.reset
  neuron2.reset
  neuron1.propagate 1.0
  neuron1.feed_forward
  bias_neuron.feed_forward
  neuron2.backpropagate 0.0
  neuron2.feed_back
  puts "Neuron2 expected 0.0 received #{neuron2.value}, weight: #{link.weight} bias: #{bias_link.weight}" if trial % 10000 == 0
  neuron1.reset
  neuron2.reset
  neuron1.propagate 0.0
  bias_neuron.feed_forward
  neuron1.feed_forward
  neuron2.backpropagate 1.0
  neuron2.feed_back
  puts "Neuron2 expected 1.0 received #{neuron2.value}, weight: #{link.weight} bias: #{bias_link.weight}" if trial % 10000 == 0
end
