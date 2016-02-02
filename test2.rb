#!/usr/bin/env ruby
require './lib/neuron'
require './lib/link'

# Train a neural net to learn the OR function
# This has two input neurons, one bias neuron, and two output neurons
# We train by setting the input neurons to (1, 0), (1, 1), (0, 1) and backpropagating 1
# We then set the inputs neuron to (0, 0) and backpropagate 0
# Eventually it will converge

neuron0_0 = Neuron.new
neuron0_1 = Neuron.new
neuron0_2 = Neuron.new
neuron1_1 = Neuron.new
training_rate = 0.1

link0_0_1 = Link.new(0.4, training_rate)
link0_1_1 = Link.new(0.5, training_rate)
link0_2_1 = Link.new(0.6, training_rate)

neuron0_0.output_links << link0_0_1
neuron0_1.output_links << link0_1_1
neuron0_2.output_links << link0_2_1

neuron1_1.input_links << link0_0_1
neuron1_1.input_links << link0_1_1
neuron1_1.input_links << link0_2_1

link0_0_1.output_neurons << neuron1_1
link0_1_1.output_neurons << neuron1_1
link0_2_1.output_neurons << neuron1_1

link0_0_1.input_neurons << neuron0_0
link0_1_1.input_neurons << neuron0_1
link0_2_1.input_neurons << neuron0_2

training_data = [
  [[0.0, 0.0], 0.0],
  [[1.0, 0.0], 1.0],
  [[0.0, 1.0], 1.0],
  [[1.0, 1.0], 1.0]
]

bias_neurons = [neuron0_0]
bias_neurons.each { |neuron| neuron.propagate 1.0 }

input_neurons = [neuron0_1, neuron0_2]
feed_forward_neurons = [neuron0_0, neuron0_1, neuron0_2]
output_neurons = [neuron1_1]
feed_back_neurons = [neuron1_1]

temporary_neurons = input_neurons + output_neurons

10000000.times do |trial|
  training_data.each do |inputs, expected_output|
    temporary_neurons.each &:reset

    neuron0_1.propagate inputs[0]
    neuron0_2.propagate inputs[1]

    feed_forward_neurons.each &:feed_forward

    output_neurons.each do |neuron|
      neuron.backpropagate expected_output
    end

    feed_back_neurons.each &:feed_back

    if trial % 10000 == 0
      puts "Input #{inputs[0]} #{inputs[1]} output #{output_neurons.first.value} expected #{expected_output}"
      #puts "#{link0_0_1.weight}, #{link0_1_1.weight}, #{link0_2_1.weight}"
    end
  end

  if trial % 10000 == 0
    puts "#{link0_0_1.weight}, #{link0_1_1.weight}, #{link0_2_1.weight}"
  end
end
