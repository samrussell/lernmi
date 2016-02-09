class Neuron
  def initialize
    reset
  end

  def input(value)
    @inputs << value
  end

  def output
    @output_value ||= sigmoid(@inputs.reduce(0.0, &:+))
  end

  def submit_sensitivity(value)
    @sensitivities << value
  end

  def get_sensitivity
    @output_sensitivity ||= @sensitivities.reduce(0.0, &:+)
  end

  def reset
    @inputs = []
    @sensitivities = []
    @output_value = nil
    @output_sensitivity = nil
  end

  def sigmoid(value)
    1 / (1 + Math.exp(-value))
  end
end
