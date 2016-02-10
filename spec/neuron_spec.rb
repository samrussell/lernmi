require 'spec_helper'
require './lib/neuron'

describe Neuron do
  let(:neuron) { Neuron.new }

  def activation_function(value)
    1 / (1 + Math.exp(-value))
  end

  describe "#reset" do
    it "resets the output and sensitivity" do
    neuron.input 0.3
    neuron.submit_sensitivity 0.3
    expect(neuron.output).to_not eq(activation_function(0.0))
    expect(neuron.previous_layer_sensitivity_sum).to_not eq(0.0)
    neuron.reset
    expect(neuron.output).to eq(activation_function(0.0))
    expect(neuron.previous_layer_sensitivity_sum).to eq(0.0)
    end
  end

  describe "#input" do
    it "adds to the input sum" do
      expect(neuron.output).to eq activation_function(0.0)
      neuron.reset
      neuron.input(0.3)
      expect(neuron.output).to eq activation_function(0.0 + 0.3)
      neuron.reset
      neuron.input(0.3)
      neuron.input(-0.2)
      expect(neuron.output).to eq activation_function(0.0 + 0.3 + -0.2)
    end
  end

  describe "#submit_sensitivity" do
    it "adds to the sensitivity" do
      expect(neuron.previous_layer_sensitivity_sum).to eq(0.0)
      neuron.reset
      neuron.submit_sensitivity(0.3)
      expect(neuron.previous_layer_sensitivity_sum).to eq(0.0 + 0.3)
      neuron.reset
      neuron.submit_sensitivity(0.3)
      neuron.submit_sensitivity(-0.2)
      expect(neuron.previous_layer_sensitivity_sum).to eq(0.0 + 0.3 + -0.2)
    end
  end
end
