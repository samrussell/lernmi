class LinkLayer
  def initialize(input_neuron_layer, output_neuron_layer)
    @input_neuron_layer = input_neuron_layer
    @output_neuron_layer = output_neuron_layer
    @links = input_neuron_layer.all_neurons.map do |input_neuron|
      output_neuron_layer.learning_neurons.map do |output_neuron|
        Link.new(generate_weight, 0.1, input_neuron, output_neuron)
      end
    end.flatten
  end

  def propagate
    @links.each &:propagate
  end

  def backpropagate
    @links.each &:backpropagate
  end

  private

  def generate_weight
    weight_generator.rand - 0.5
  end

  def weight_generator
    @weight_generator ||= Random.new(1)
  end
end
