require 'spec_helper'
require './lib/neuron'

describe Neuron do
  describe '#feed_forward' do
    let(:neuron) { Neuron.new }
    let(:output_link) { double }
    let(:expected_output) { 0.81757 }

    before do
      neuron.output_links << output_link

      # sigmoid(1.5) = 0.81757...
      # if we give the neuron a total of 1.5, it should propagate sigmoid(1.5)

      neuron.propagate 0.4
      neuron.propagate 0.5
      neuron.propagate 0.6
    end

    it 'sums its inputs, applies a sigmoid, and outputs the result to its links' do
      expect(output_link).to receive(:propagate).with(within(0.00001).of(expected_output))

      neuron.feed_forward
    end
  end

  describe '#feed_back' do
    let(:neuron) { Neuron.new }
    let(:input_link) { double }
    let(:expected_input) { -0.00510 }

    before do
      neuron.input_links << input_link

      # set neuron value to 0.5, then tell it the right answer was 0.6
      # make sure it correctly backpropagates the difference

      neuron.propagate 0.5
      neuron.backpropagate 0.6
    end

    it 'backpropagates the difference' do
      expect(input_link).to receive(:backpropagate).with(within(0.00001).of(expected_input))

      neuron.feed_back
    end
  end
end
