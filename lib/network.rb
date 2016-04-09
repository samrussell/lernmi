require './lib/neuron_layer'
require './lib/link_layer'

class Network
  attr_reader :neuron_layers

  def initialize(neurons_in_each_layer)
    @neuron_layers = neurons_in_each_layer.map { |neuron_count| NeuronLayer.new(neuron_count) }
    @link_layers = @neuron_layers.each_cons(2).map { |neuron_layers| LinkLayer.new(*neuron_layers) }
  end

  def learn(inputs, expected_outputs)
    @neuron_layers.each &:reset
    @neuron_layers.first.input(inputs)
    @link_layers.each &:propagate 
    @neuron_layers.last.expected_output(expected_outputs)
    @link_layers.reverse.each &:backpropagate
  end
end
