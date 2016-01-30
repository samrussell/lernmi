class Link
  attr_accessor :weight
  attr_reader :output_neurons, :input_neurons

  def initialize(initial_weight, training_rate = 0.1)
    @output_neurons = []
    @input_neurons = []
    @weight = initial_weight
    @training_rate = training_rate
    @value = 0.0
  end

  def propagate(value)
    @output_neurons.each { |neuron| neuron.propagate(@weight * value) }
    @value = value
  end

  def backpropagate(error)
    @input_neurons.each { |neuron| neuron.backpropagate(error) }

    update_weight(error)
  end

  private

  def update_weight(error)
    @weight += error * @value * @training_rate
  end
end
