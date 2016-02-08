class Neuron
  def initialize
    @inputs = []
    @sensitivities = []
  end

  def input(value)
    @inputs << value
  end

  def output
    @inputs.reduce(0.0, &:+)
  end

  def submit_sensitivity(value)
    @sensitivities << value
  end

  def get_sensitivity
    @sensitivities.reduce(0.0, &:+)
  end

  def reset
    @inputs = []
    @sensitivities = []
  end
end
