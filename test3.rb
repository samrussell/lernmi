#!/usr/bin/env ruby
require './lib/neuron'
require './lib/link'

# Train a neural net to learn the XOR function
# This has two input neurons, one bias neuron, and two output neurons
# We train by setting the input neurons to (1, 0), (0, 1) and backpropagating 1
# We then set the inputs neuron to (0, 0), (1, 1) and backpropagate 0
# Eventually it will converge

neuron0_0 = Neuron.new
neuron0_1 = Neuron.new
neuron0_2 = Neuron.new
neuron1_0 = Neuron.new(true)
neuron1_1 = Neuron.new(true)
neuron1_2 = Neuron.new(true)
neuron1_3 = Neuron.new(true)
neuron2_1 = Neuron.new
training_rate = 0.1

link0_0_1 = Link.new(0.4, training_rate)
link0_1_1 = Link.new(0.4, training_rate)
link0_2_1 = Link.new(0.4, training_rate)
link0_0_2 = Link.new(0.4, training_rate)
link0_1_2 = Link.new(0.4, training_rate)
link0_2_2 = Link.new(0.4, training_rate)
link0_0_3 = Link.new(0.4, training_rate)
link0_1_3 = Link.new(0.4, training_rate)
link0_2_3 = Link.new(0.4, training_rate)
link1_0_1 = Link.new(0.4, training_rate)
link1_1_1 = Link.new(0.4, training_rate)
link1_2_1 = Link.new(0.4, training_rate)
link1_3_1 = Link.new(0.4, training_rate)

neuron0_0.output_links << link0_0_1
neuron0_1.output_links << link0_1_1
neuron0_2.output_links << link0_2_1
neuron0_0.output_links << link0_0_2
neuron0_1.output_links << link0_1_2
neuron0_2.output_links << link0_2_2
neuron0_0.output_links << link0_0_3
neuron0_1.output_links << link0_1_3
neuron0_2.output_links << link0_2_3
neuron1_0.output_links << link1_0_1
neuron1_1.output_links << link1_1_1
neuron1_2.output_links << link1_2_1
neuron1_3.output_links << link1_3_1

neuron1_1.input_links << link0_0_1
neuron1_1.input_links << link0_1_1
neuron1_1.input_links << link0_2_1
neuron1_2.input_links << link0_0_2
neuron1_2.input_links << link0_1_2
neuron1_2.input_links << link0_2_2
neuron1_3.input_links << link0_0_3
neuron1_3.input_links << link0_1_3
neuron1_3.input_links << link0_2_3
neuron2_1.input_links << link1_0_1
neuron2_1.input_links << link1_1_1
neuron2_1.input_links << link1_2_1
neuron2_1.input_links << link1_3_1

link0_0_1.output_neurons << neuron1_1
link0_1_1.output_neurons << neuron1_1
link0_2_1.output_neurons << neuron1_1
link0_0_2.output_neurons << neuron1_2
link0_1_2.output_neurons << neuron1_2
link0_2_2.output_neurons << neuron1_2
link0_0_3.output_neurons << neuron1_3
link0_1_3.output_neurons << neuron1_3
link0_2_3.output_neurons << neuron1_3
link1_0_1.output_neurons << neuron2_1
link1_1_1.output_neurons << neuron2_1
link1_2_1.output_neurons << neuron2_1
link1_3_1.output_neurons << neuron2_1

link0_0_1.input_neurons << neuron0_0
link0_1_1.input_neurons << neuron0_1
link0_2_1.input_neurons << neuron0_2
link0_0_2.input_neurons << neuron0_0
link0_1_2.input_neurons << neuron0_1
link0_2_2.input_neurons << neuron0_2
link0_0_3.input_neurons << neuron0_0
link0_1_3.input_neurons << neuron0_1
link0_2_3.input_neurons << neuron0_2
link1_0_1.input_neurons << neuron1_0
link1_1_1.input_neurons << neuron1_1
link1_2_1.input_neurons << neuron1_2
link1_3_1.input_neurons << neuron1_3

training_data = [
  [[0.0, 0.0], 0.0],
  [[1.0, 0.0], 1.0],
  [[0.0, 1.0], 1.0],
  [[1.0, 1.0], 0.0]
]

bias_neurons = [neuron0_0, neuron1_0]

input_neurons = [neuron0_1, neuron0_2]
feed_forward_layers = [
  [neuron0_0, neuron0_1, neuron0_2],
  [neuron1_0, neuron1_1, neuron1_2, neuron1_3]
]
output_neurons = [neuron2_1]
feed_back_layers = [
  [neuron2_1],
  [neuron1_1, neuron1_2, neuron1_3]
]

all_neurons = [ neuron0_0, neuron0_1, neuron0_2,
                neuron1_0, neuron1_1, neuron1_2, neuron1_3,
                neuron2_1
]

10000000.times do |trial|
  training_data.each do |inputs, expected_output|
    all_neurons.each &:reset
    bias_neurons.each { |neuron| neuron.propagate 1.0 }

    neuron0_1.propagate inputs[0]
    neuron0_2.propagate inputs[1]

    feed_forward_layers.each do |neurons|
      neurons.each &:feed_forward
    end

    output_neurons.each do |neuron|
      neuron.backpropagate expected_output
    end

    feed_back_layers.each do |neurons|
      neurons.each &:feed_back
    end

    if trial % 10000 == 0
      puts "Input #{inputs[0]} #{inputs[1]} output #{output_neurons.first.value} expected #{expected_output}"
      #puts "#{link0_0_1.weight}, #{link0_1_1.weight}, #{link0_2_1.weight}"
    end
  end

  if trial % 10000 == 0
    puts "#{link1_0_1.weight}, #{link1_1_1.weight}, #{link1_2_1.weight}, #{link1_3_1.weight}"
  end
end
