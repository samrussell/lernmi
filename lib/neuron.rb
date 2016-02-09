class Neuron
  def initialize
    reset
  end

  def input(value)
    @input += value
  end

  def output
    @output_value ||= sigmoid(@input)
  end

  def submit_sensitivity(value)
    @sensitivity += value
  end

  def get_sensitivity
    @output_sensitivity ||= @sensitivity
  end

  def reset
    @input = 0.0
    @sensitivity = 0.0
    @output_value = nil
    @output_sensitivity = nil
  end

  def sigmoid(value)
    1.0 / (1.0 + Math.exp(-value))
  end
end
