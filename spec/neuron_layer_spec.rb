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

  describe "#input" do
    let(:input_values) { [0.1, 0.3, 0.5, 0.2, -0.4] }

    it "sets the input value for each learning neuron" do
      neuron_layer.learning_neurons.zip(input_values).each do |neuron, value|
        expect(neuron).to receive(:input).with(value)
      end

      neuron_layer.input(input_values)
    end
  end

  describe "#expected_output" do
    let(:expected_output_values) { [0.2, 0.3, 0.4, 0.5, 0.6] }
    let(:output_values) { [0.5, 0.4, 0.3, 0.2, 0.1] }
    let(:sensitivities) { [0.5 - 0.2, 0.4 - 0.3, 0.3 - 0.4, 0.2 - 0.5, 0.1 - 0.6] }

    it "sets the expected output value for each learning neuron" do
      neuron_layer.learning_neurons.zip(output_values).each do |neuron, value|
        expect(neuron).to receive(:output).and_return(value)
      end

      neuron_layer.learning_neurons.zip(sensitivities).each do |neuron, value|
        expect(neuron).to receive(:submit_sensitivity).with(value)
      end

      neuron_layer.expected_output(expected_output_values)
    end
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
