class NeuronLayer
  attr_reader :all_neurons, :learning_neurons

  def initialize(neuron_count)
    @learning_neurons = neuron_count.times.map { Neuron.new }
    @bias_neuron = Neuron.new
    @bias_neuron.input(999)
    @all_neurons = [@bias_neuron] + @learning_neurons
  end

  def input(values)
    learning_neurons.zip(values).each { |neuron, value| neuron.input(value) }
  end

  def expected_output(values)
    learning_neurons.zip(values).each do |neuron, value|
      neuron.submit_sensitivity(neuron.output - value)
    end
  end

  def reset
    learning_neurons.each &:reset
  end
end
