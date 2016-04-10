require './lib/neuron'
require './lib/link'
require './lib/neuron_layer'
require './lib/link_layer'
require './lib/network'
require './lib/idx_loader'
require 'zlib'

class MnistLearner
  def initialize(image_path, label_path)
    @image_path = image_path
    @label_path = label_path
  end

  def run
    puts "Loading images"
    load_images

    puts "Loading labels"
    load_labels

    puts "Initializing network"
    initialize_network

    puts "Running..."
    100.times do |trial|
      training_data.each.with_index do |(inputs, label), label_index|
        network.learn(inputs, expected_outputs_from_number(label))
        
        if label_index % 100 == 0
          ascii_print(inputs)
          puts "expected #{label}"
          vote(network.neuron_layers.last.learning_neurons)
        end
      end
    end
  end

  private

  attr_reader :network

  def initialize_network
    @network = Network.new([784, 15, 10])
  end

  def training_data
    @training_data ||= @images.zip(@labels)
  end

  def load_images
    @images = load_gzipped_idx_file(@image_path)
  end

  def load_labels
    @labels = load_gzipped_idx_file(@label_path)
  end

  def load_gzipped_idx_file(path)
    file = File.open(path)
    stream = Zlib::GzipReader.new(file)
    IdxLoader.load(stream)
  end

  def vote(output_neurons)
    votes = (0..9).to_a.zip(output_neurons.map(&:output))
    top_vote = votes.max {|a, b| a[1] <=> b[1]}
    puts "Guess: #{top_vote[0]} confidence #{top_vote[1]}"
  end

  def ascii_print(inputs)
    output = inputs.each_slice(28).map do |row|
      row.map do |darkness|
        darkness > 128 ? "X" : " "
      end.join
    end.join("\n")

    puts output
  end

  def expected_outputs_from_number(number)
    10.times.map {|i| i == number ? 1.0 : 0.0 }
  end
end
