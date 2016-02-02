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
    @value = value
    @output_neurons.each { |neuron| neuron.propagate(@weight * value) }
  end

  def backpropagate(sensitivity)
    # TODO feedback to inner neurons
    #@input_neurons.each { |neuron| neuron.backpropagate(@weight * sensitivity) }

    update_weight(sensitivity)
  end

  private

  def update_weight(sensitivity)
    gradient = @value * sensitivity
    @weight -= gradient * @training_rate
  end

  def sigmoid(value)
    1 / (1 + Math.exp(-value))
  end

  def sigmoid_gradient(value)
    sigmoid(value) * (1 - sigmoid(value))
  end
end
