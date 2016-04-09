require 'spec_helper'
require './lib/network'
require './lib/neuron_layer'
require './lib/link_layer'

describe Network do
  let(:neurons_in_each_layer) { [7, 3, 2] }
  let(:network) { Network.new(neurons_in_each_layer) }
  let(:fake_neuron_layers) { 3.times.map { instance_double(NeuronLayer) } }
  let(:layers_and_neuron_numbers) { fake_neuron_layers.zip(neurons_in_each_layer) }
  let(:fake_link_layers) { 2.times.map { instance_double(LinkLayer) } }

  before do
    layers_and_neuron_numbers.each do |neuron_layer, number_of_neurons|
      expect(NeuronLayer).to receive(:new).once.with(number_of_neurons).and_return(neuron_layer)
    end

    fake_link_layers.zip(fake_neuron_layers.each_cons(2).to_a).each do |link_layer, neuron_layers|
      expect(LinkLayer).to receive(:new).once.with(*neuron_layers).and_return(link_layer)
    end
  end

  describe "#learn" do
    let(:sample_inputs) { [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] }
    let(:expected_outputs) { [-0.2, 0.5] }

    it "propagates test values through the network" do
      expect(fake_neuron_layers[0]).to receive(:reset).ordered
      expect(fake_neuron_layers[1]).to receive(:reset).ordered
      expect(fake_neuron_layers[2]).to receive(:reset).ordered
      expect(fake_neuron_layers.first).to receive(:input).with(sample_inputs).ordered
      expect(fake_link_layers[0]).to receive(:propagate).ordered
      expect(fake_link_layers[1]).to receive(:propagate).ordered

      expect(fake_neuron_layers[-1]).to receive(:expected_output).with(expected_outputs).ordered
      expect(fake_link_layers[1]).to receive(:backpropagate).ordered
      expect(fake_link_layers[0]).to receive(:backpropagate).ordered

      network.learn(sample_inputs, expected_outputs)
    end
  end
end
