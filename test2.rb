#!/usr/bin/env ruby
require './lib/neuron'
require './lib/link'

# Train a neural net to learn the OR function
# This has two input neurons, one bias neuron, and two output neurons
# We train by setting the input neurons to (1, 0), (1, 1), (0, 1) and backpropagating 1
# We then set the inputs neuron to (0, 0) and backpropagate 0
# Eventually it will converge

neuron1_1 = Neuron.new
training_rate = 0.1

link0_0_1 = Link.new(0.4, training_rate)
link0_1_1 = Link.new(0.5, training_rate)
link0_2_1 = Link.new(0.6, training_rate)

neuron1_1.input_links << link0_0_1
neuron1_1.input_links << link0_1_1
neuron1_1.input_links << link0_2_1

link0_0_1.output_neurons << neuron1_1
link0_1_1.output_neurons << neuron1_1
link0_2_1.output_neurons << neuron1_1

training_data = [
  [[0.0, 0.0], 0.0],
  [[1.0, 0.0], 1.0],
  [[0.0, 1.0], 1.0],
  [[1.0, 1.0], 1.0]
]
output_neurons = [neuron1_1]
feed_back_neurons = [neuron1_1]

10000000.times do |trial|
  training_data.each do |inputs, expected_output|
    #input_neurons.each &:reset
    #
    #inputs.zip(input_neurons).each do |input, neuron|
    #  neuron.propagate input
    #end
    #
    #feed_forward_neurons.each &:feed_forward

    link0_0_1.propagate 1.0
    link0_1_1.propagate inputs[0]
    link0_2_1.propagate inputs[1]

    output_neurons.each do |neuron|
      neuron.backpropagate expected_output
    end

    feed_back_neurons.each &:feed_back

    output_neurons.each &:reset

    if trial % 10000 == 0
      puts "Input #{inputs[0]} #{inputs[1]} output #{output_neurons.first.value} expected #{expected_output}"
    end
  end

  if trial % 10000 == 0
    puts link0_0_1.weight
    puts link0_1_1.weight
    puts link0_2_1.weight
  end
end
