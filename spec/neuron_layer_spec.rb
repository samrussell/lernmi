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

  describe "#reset" do
    before do
      neuron_layer.learning_neurons.each do |neuron|
        neuron.input 1.0
      end
    end

    it "resets all learning neurons" do
      neuron_layer.learning_neurons.each do |neuron|
        expect(neuron.output).to be > 0.5
      end

      neuron_layer.reset

      neuron_layer.learning_neurons.each do |neuron|
        expect(neuron.output).to eq(0.5)
      end
    end

    it "doesn't reset bias neurons" do
      bias_neuron = neuron_layer.all_neurons.first
      expect(bias_neuron.output).to eq(1.0)

      neuron_layer.reset

      expect(bias_neuron.output).to eq(1.0)
    end
  end
end
