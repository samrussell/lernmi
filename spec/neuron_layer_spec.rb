require 'spec_helper'
require './lib/neuron'
require './lib/neuron_layer'

describe NeuronLayer do
  let(:neuron_layer) { NeuronLayer.new(5) }

  it "creates N+1 neurons" do
    expect(neuron_layer.all_neurons.count).to eq(6)
  end

  it "creates N learning neurons" do
    expect(neuron_layer.learning_neurons.count).to eq(5)
    neuron_layer.learning_neurons.each do |neuron|
      expect(neuron.output).to eq(0.5)
    end
  end

  it "creates 1 bias neuron" do
    expect(neuron_layer.all_neurons.first.output).to eq(1.0)
  end
end
