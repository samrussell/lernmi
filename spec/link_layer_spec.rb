require 'spec_helper'
require './lib/link'
require './lib/neuron_layer'
require './lib/link_layer'

describe LinkLayer do
  let(:input_neuron_layer) { instance_double(NeuronLayer) }
  let(:output_neuron_layer) { instance_double(NeuronLayer) }
  let(:link_layer) { LinkLayer.new(input_neuron_layer, output_neuron_layer) }

  xit "connects the neuron layers" do
  end
end
