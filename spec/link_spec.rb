require 'spec_helper'
require './lib/link'

describe Link do
  describe '#propagate' do
    let(:test_value) { 0.8 }
    let(:weight) { 0.5 }
    let(:link) { Link.new(weight) }
    let(:output_neuron) { double }

    before do
      link.output_neurons << output_neuron
    end

    it 'applies its weight and propagates the value' do
      expect(output_neuron).to receive(:propagate).with(weight * test_value)

      link.propagate(test_value)
    end
  end

  describe '#backpropagate' do
    let(:test_value) { 0.1 }
    let(:weight) { 0.5 }
    let(:training_rate) { 0.1 }
    let(:link) { Link.new(weight, training_rate) }
    let(:input_neuron) { double }

    before do
      link.input_neurons << input_neuron
      link.propagate 1.0
    end

    it 'updates its weight and backpropagates the value' do
      expect(input_neuron).to receive(:backpropagate).with(test_value)

      link.backpropagate(test_value)

      expect(link.weight).to eq(0.51)
    end
  end
end
