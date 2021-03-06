require 'spec_helper'
require './lib/neuron'
require './lib/link'
require './lib/neuron_layer'
require './lib/link_layer'

describe LinkLayer do
  let(:input_neuron_1) { instance_double(Neuron) }
  let(:input_neuron_2) { instance_double(Neuron) }
  let(:input_bias_neuron) { instance_double(Neuron) }
  let(:output_neuron_1) { instance_double(Neuron) }
  let(:output_neuron_2) { instance_double(Neuron) }
  let(:output_bias_neuron) { instance_double(Neuron) }
  let(:input_neuron_layer) { instance_double(NeuronLayer) }
  let(:output_neuron_layer) { instance_double(NeuronLayer) }
  let(:link_layer) { LinkLayer.new(input_neuron_layer, output_neuron_layer) }

  before do
    allow(input_neuron_layer).to receive(:all_neurons).and_return([input_bias_neuron, input_neuron_1, input_neuron_2])
    allow(output_neuron_layer).to receive(:learning_neurons).and_return([output_neuron_1, output_neuron_2])
  end

  it "creates links between the neurons" do
    expect(Link).to receive(:new).with(anything, anything, input_bias_neuron, output_neuron_1).and_call_original
    expect(Link).to receive(:new).with(anything, anything, input_bias_neuron, output_neuron_2).and_call_original
    expect(Link).to receive(:new).with(anything, anything, input_neuron_1, output_neuron_1).and_call_original
    expect(Link).to receive(:new).with(anything, anything, input_neuron_1, output_neuron_2).and_call_original
    expect(Link).to receive(:new).with(anything, anything, input_neuron_2, output_neuron_1).and_call_original
    expect(Link).to receive(:new).with(anything, anything, input_neuron_2, output_neuron_2).and_call_original

    link_layer
  end

  describe "#propagate" do
    let(:links) { 6.times.map { instance_double(Link) } }

    before do
      allow(Link).to receive(:new).and_return(*links)
    end

    it "calls propagate on each link" do
      links.each do |link|
        expect(link).to receive(:propagate)
      end

      link_layer.propagate
    end
  end

  describe "#backpropagate" do
    let(:links) { 6.times.map { instance_double(Link) } }

    before do
      allow(Link).to receive(:new).and_return(*links)
    end

    it "calls backpropagate on each link" do
      links.each do |link|
        expect(link).to receive(:backpropagate)
      end

      link_layer.backpropagate
    end
  end
end
