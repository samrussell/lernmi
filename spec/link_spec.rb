require 'spec_helper'
require './lib/link'
require './lib/neuron'

describe Link do
  let(:test_value) { 0.8 }
  let(:weight) { 0.5 }
  let(:link) { Link.new(weight) }
  let(:output_neuron) { instance_double(Neuron) }
  let(:input_neuron) { instance_double(Neuron) }

  before do
    link.output_neuron = output_neuron
    link.input_neuron = input_neuron
  end

  def neuron_activation(value)
    1 / (1 + Math.exp(-value))
  end

  describe '#propagate' do
    it 'applies its weight and propagates the value' do
      expect(input_neuron).to receive(:output).and_return(test_value)

      expect(output_neuron).to receive(:input).with(weight * neuron_activation(test_value))

      link.propagate
    end
  end

  describe '#backpropagate' do
    it 'updates its weight and backpropagates the value' do
      expect(input_neuron).to receive(:backpropagate).with(test_value)

      link.backpropagate(test_value)

      expect(link.weight).to eq(0.51)
    end
  end
end
