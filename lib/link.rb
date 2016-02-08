class Link
  attr_accessor :weight
  attr_accessor :output_neuron, :input_neuron

  def initialize(initial_weight, training_rate = 0.1)
    @output_neurons = []
    @input_neurons = []
    @weight = initial_weight
    @training_rate = training_rate
    @value = 0.0
  end

  def propagate
    value = @input_neuron.output

    @output_neuron.input(@weight * sigmoid(value))
  end

  def backpropagate(sensitivity)
    @input_neurons.each { |neuron| neuron.backpropagate(@weight * sensitivity) }

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
end
