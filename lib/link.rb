class Link
  attr_accessor :weight
  attr_accessor :output_neuron, :input_neuron

  def initialize(initial_weight, training_rate = 0.1, input_neuron = nil, output_neuron = nil)
    @weight = initial_weight
    @training_rate = training_rate
    @input_neuron = input_neuron
    @output_neuron = output_neuron
  end

  def propagate
    @output_neuron.input(@input_neuron.output * @weight)
  end

  def backpropagate
    output_sensitivity = output_neuron.previous_layer_sensitivity_sum * activation_function_sensitivity(output_neuron.output)

    input_neuron.submit_sensitivity(@weight * output_sensitivity)

    update_weight(input_neuron.output * output_sensitivity)
  end

  private

  def update_weight(sensitivity)
    @weight -= sensitivity * @training_rate
  end

  def activation_function_sensitivity(value)
    value * (1.0 - value)
  end
end
