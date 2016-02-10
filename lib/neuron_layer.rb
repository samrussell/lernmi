class NeuronLayer
  attr_reader :all_neurons, :learning_neurons

  def initialize(neuron_count)
    @learning_neurons = neuron_count.times.map { Neuron.new }
    @bias_neuron = Neuron.new
    @bias_neuron.input(999)
    @all_neurons = [@bias_neuron] + @learning_neurons
  end
end
