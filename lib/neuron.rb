class Neuron
  attr_reader :previous_layer_sensitivity_sum

  def initialize
    reset
  end

  def bias
    @bias = true
  end

  def input(value)
    @input_sum += value
  end

  def output
    return 1.0 if @bias
    @output ||= activation_function(@input_sum)
  end

  def submit_sensitivity(value)
    @previous_layer_sensitivity_sum += value
  end

  def reset
    @input_sum = 0.0
    @previous_layer_sensitivity_sum = 0.0
    @output = nil
    @bias = false
  end

  def activation_function(value)
    1.0 / (1.0 + Math.exp(-value))
  end
end
