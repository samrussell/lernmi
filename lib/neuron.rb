class Neuron
  attr_reader :output_links, :input_links

  def initialize(hidden = false)
    @output_links = []
    @input_links = []
    @intermediate_value = 0.0
    @expected_value = 0.0
    @hidden = hidden
  end

  def input(value)
    raise MethodNotImplementedException
  end

  def output
    raise MethodNotImplementedException
  end

  def propagate(input_value)
    @intermediate_value += input_value
  end

  def backpropagate(input_value)
    @expected_value += input_value
  end

  def reset
    @intermediate_value = 0.0
    @expected_value = 0.0
  end

  def feed_forward
    @output_links.each { |link| link.propagate(value) }
  end

  def feed_back
    @input_links.each { |link| link.backpropagate(sensitivity) }
  end

  def value
    sigmoid(@intermediate_value)
  end

  private

  def sensitivity
    if @hidden
      @expected_value * value * (1 - value)
    else
      (value - @expected_value) * value * (1 - value)
    end
  end

  def sigmoid(value)
    1 / (1 + Math.exp(-value))
  end
end
