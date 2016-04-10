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
neuron1_0 = Neuron.new
neuron1_1 = Neuron.new
neuron1_2 = Neuron.new
neuron1_3 = Neuron.new
neuron2_1 = Neuron.new
training_rate = 0.1

weight_generator = Random.new(1)

link0_0_1 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_1_1 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_2_1 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_0_2 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_1_2 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_2_2 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_0_3 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_1_3 = Link.new(weight_generator.rand - 0.5, training_rate)
link0_2_3 = Link.new(weight_generator.rand - 0.5, training_rate)
link1_0_1 = Link.new(weight_generator.rand - 0.5, training_rate)
link1_1_1 = Link.new(weight_generator.rand - 0.5, training_rate)
link1_2_1 = Link.new(weight_generator.rand - 0.5, training_rate)
link1_3_1 = Link.new(weight_generator.rand - 0.5, training_rate)

link0_0_1.output_neuron = neuron1_1
link0_1_1.output_neuron = neuron1_1
link0_2_1.output_neuron = neuron1_1
link0_0_2.output_neuron = neuron1_2
link0_1_2.output_neuron = neuron1_2
link0_2_2.output_neuron = neuron1_2
link0_0_3.output_neuron = neuron1_3
link0_1_3.output_neuron = neuron1_3
link0_2_3.output_neuron = neuron1_3
link1_0_1.output_neuron = neuron2_1
link1_1_1.output_neuron = neuron2_1
link1_2_1.output_neuron = neuron2_1
link1_3_1.output_neuron = neuron2_1

link0_0_1.input_neuron = neuron0_0
link0_1_1.input_neuron = neuron0_1
link0_2_1.input_neuron = neuron0_2
link0_0_2.input_neuron = neuron0_0
link0_1_2.input_neuron = neuron0_1
link0_2_2.input_neuron = neuron0_2
link0_0_3.input_neuron = neuron0_0
link0_1_3.input_neuron = neuron0_1
link0_2_3.input_neuron = neuron0_2
link1_0_1.input_neuron = neuron1_0
link1_1_1.input_neuron = neuron1_1
link1_2_1.input_neuron = neuron1_2
link1_3_1.input_neuron = neuron1_3

links = [
  [
    link0_0_1,
    link0_1_1,
    link0_2_1,
    link0_0_2,
    link0_1_2,
    link0_2_2,
    link0_0_3,
    link0_1_3,
    link0_2_3
  ],
  [
    link1_0_1,
    link1_1_1,
    link1_2_1,
    link1_3_1
  ]
]

training_data = [
  [[0.0, 0.0], 0.0],
  [[1.0, 0.0], 1.0],
  [[0.0, 1.0], 1.0],
  [[1.0, 1.0], 0.0]
]

bias_neurons = [neuron0_0, neuron1_0]
bias_neurons.each { |neuron| neuron.input 999.0 }

input_neurons = [neuron0_1, neuron0_2]
output_neurons = [neuron2_1]

all_neurons = [ neuron0_0, neuron0_1, neuron0_2,
                neuron1_0, neuron1_1, neuron1_2, neuron1_3,
                neuron2_1
]

temporary_neurons = [ neuron0_1, neuron0_2,
                neuron1_1, neuron1_2, neuron1_3,
                neuron2_1
]

10000000.times do |trial|
  training_data.each do |inputs, expected_output|
    temporary_neurons.each &:reset

    neuron0_1.input inputs[0]
    neuron0_2.input inputs[1]

    links.each do |layer|
      layer.each &:propagate
    end

    output_neurons.each do |neuron|
      neuron.submit_sensitivity (neuron.output - expected_output)
    end

    links.reverse.each do |layer|
      layer.each &:backpropagate
    end

    if trial % 10000 == 0
      puts "Input #{inputs[0]} #{inputs[1]} output #{output_neurons.first.output} expected #{expected_output}"
    end
  end

  if trial % 10000 == 0
    puts "#{link1_0_1.weight}, #{link1_1_1.weight}, #{link1_2_1.weight}, #{link1_3_1.weight}"
  end
end
