class Neuron
  attr_reader :output_links, :input_links

  def initialize
    @output_links = []
    @input_links = []
    @intermediate_value = 0.0
    @expected_value = 0.0
  end

  def propagate(input_value)
    @intermediate_value += input_value
  end

  def backpropagate(input_value)
    @expected_value = input_value
  end

  def reset
    @intermediate_value = 0.0
    @expected_value = 0.0
  end

  def feed_forward
    @output_links.each { |link| link.propagate(value) }
  end

  def feed_back
    @input_links.each { |link| link.backpropagate(error) }
  end

  def value
    sigmoid(@intermediate_value)
  end

  private

  def error
    (@expected_value - value) * sigmoid_gradient(value)
  end

  def sigmoid(value)
    1 / (1 + Math.exp(-value))
  end

  def sigmoid_gradient(value)
    sigmoid(value) * (1 - sigmoid(value))
  end
end
