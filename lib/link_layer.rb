class LinkLayer
  def initialize
    @links = []
  end

  def join(input_neuron_layer, output_neuron_layer)
    @links = input_neuron_layer.map do |input_neuron|
      output_neuron_layer.map do |output_neuron|
        link = Link.new(generate_weight, 0.1, input_neuron, output_neuron)
      end
    end.flatten
  end

  private

  def generate_weight
    weight_generator.rand - 0.5
  end

  def weight_generator
    @weight_generator ||= Random.new(1)
  end
end
