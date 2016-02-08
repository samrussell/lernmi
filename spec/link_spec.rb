require 'spec_helper'
require './lib/link'
require './lib/neuron'

describe Link do
  let(:weight) { 0.5 }
  let(:training_rate) { 0.1 }
  let(:link) { Link.new(weight, training_rate) }
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
    let(:output_value) { 0.8 }

    it 'applies its weight and propagates the value' do
      expect(input_neuron).to receive(:output).and_return(output_value)
      expect(output_neuron).to receive(:input).with(weight * neuron_activation(output_value))

      link.propagate
    end
  end

  describe '#backpropagate' do
    let(:input_weighted_sensitivity) { 0.1 }
    let(:output_sensitivity) { input_weighted_sensitivity * output_value_gradient }
    let(:weight_update) { output_sensitivity * input_value * training_rate * -1 }
    let(:weighted_sensitivity) { output_sensitivity * weight }
    let(:input_value) { 0.3 }
    let(:output_value) { 0.8 }
    let(:output_value_gradient) { output_value * (1 - output_value) }

    it 'updates its weight and backpropagates the value' do
      expect(output_neuron).to receive(:output).and_return(output_value)
      expect(output_neuron).to receive(:get_sensitivity).and_return(input_weighted_sensitivity)
      expect(input_neuron).to receive(:output).and_return(input_value)
      expect(input_neuron).to receive(:set_sensitivity).with(weighted_sensitivity)

      expect { link.backpropagate }.to change { link.weight }.by(within(0.0001).of weight_update)
    end
  end
end
