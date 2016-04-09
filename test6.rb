#!/usr/bin/env ruby
require './lib/neuron'
require './lib/link'
require './lib/neuron_layer'
require './lib/link_layer'
require './lib/network'
require './lib/idx_loader'
require 'zlib'

# Train a neural net to learn MNIST numbers
# This has 28 x 28 = 784 input neurons, one bias neuron, and 10 output neurons
# Eventually it will converge

network = Network.new([784, 15, 10])

puts "Loading MNIST data"

imagefile = File.open("assets/t10k-images-idx3-ubyte.gz")
imagestream = Zlib::GzipReader.new(imagefile)
images = IdxLoader.load(imagestream)

labelfile = File.open("assets/t10k-labels-idx1-ubyte.gz")
labelstream = Zlib::GzipReader.new(labelfile)
labels = IdxLoader.load(labelstream)

puts "Loaded"

training_data = images.zip(labels)

def vote(output_neurons)
  votes = (0..9).to_a.zip(output_neurons.map(&:output))
  top_vote = votes.max {|a, b| a[1] <=> b[1]}
  puts "Guess: #{top_vote[0]} confidence #{top_vote[1]}"
end

def ascii_print(inputs)
  output = inputs.each_slice(28).map do |row|
    row.map do |darkness|
      darkness > 128 ? "X" : " "
    end.join
  end.join("\n")

  puts output
end

def expected_outputs_from_number(number)
  10.times.map {|i| i == number ? 1.0 : 0.0 }
end

100.times do |trial|
  training_data.each.with_index do |(inputs, label), label_index|
    network.learn(inputs, expected_outputs_from_number(label))
    
    if label_index % 100 == 0
      ascii_print(inputs)
      puts "expected #{label}"
      vote(network.neuron_layers.last.learning_neurons)
    end
  end
end
