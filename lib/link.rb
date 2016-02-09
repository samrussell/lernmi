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

    @output_neuron.input(value * @weight)
  end

  def backpropagate
    weighted_input_sensitivity = output_neuron.get_sensitivity

    output_sensitivity = weighted_input_sensitivity * gradient_of_sigmoid(output_neuron.output)

    input_neuron.submit_sensitivity(@weight * output_sensitivity)

    update_weight(output_sensitivity)
  end

  private

  def update_weight(sensitivity)
    gradient = input_neuron.output * sensitivity
    @weight -= gradient * @training_rate
  end

  def gradient_of_sigmoid(value)
    value * (1 - value)
  end
end
