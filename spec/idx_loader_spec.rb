require 'spec_helper'
require './lib/idx_loader'

def encode_bytes(string)
  string.force_encoding('ASCII-8BIT')
end

def encode_one_byte_integer(integer)
  [integer].pack('c')
end

def encode_four_byte_integer(integer)
  [integer].pack('L>')
end

def encode_byte_matrix(byte_matrix)
  byte_matrix.flatten.map { |b| encode_one_byte_integer b}.join
end

describe IdxLoader do
  context 'with an image file' do
    let(:first_two_zero_bytes) { encode_bytes "\x00\x00" }
    let(:file_contains_unsigned_bytes) { encode_bytes "\x08" }
    let(:file_is_a_list_of_matrices) { encode_bytes "\x03" }
    let(:number_of_matrices) { encode_four_byte_integer 2 }
    let(:rows_in_each_matrix) { encode_four_byte_integer 4 }
    let(:columns_in_each_matrix) { encode_four_byte_integer 4 }
    let(:matrix_bytes) { 
      [
        [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
      ]
    }
    let(:test_idx_data) { first_two_zero_bytes +
                          file_contains_unsigned_bytes +
                          file_is_a_list_of_matrices +
                          number_of_matrices +
                          rows_in_each_matrix +
                          columns_in_each_matrix +
                          encode_byte_matrix(matrix_bytes)
    }
    let(:test_idx_stream) { StringIO.new(test_idx_data) }

    it 'parses the idx stream into 2 images' do
      images = IdxLoader.load(test_idx_stream)

      expect(images.count).to eq(2)
      expect(images.first).to eq(matrix_bytes.first)
      expect(images.last).to eq(matrix_bytes.last)
    end
  end

  context 'with a label file' do
    let(:first_two_zero_bytes) { encode_bytes "\x00\x00" }
    let(:file_contains_unsigned_bytes) { encode_bytes "\x08" }
    let(:file_is_a_list_of_integers) { encode_bytes "\x01" }
    let(:number_of_integers) { encode_four_byte_integer 4 }
    let(:label_bytes) { [ 1, 6, 8, 23 ] }
    let(:test_idx_data) { first_two_zero_bytes +
                          file_contains_unsigned_bytes +
                          file_is_a_list_of_integers +
                          number_of_integers +
                          encode_byte_matrix(label_bytes)
    }
    let(:test_idx_stream) { StringIO.new(test_idx_data) }

    it 'parses the idx stream into 4 labels' do
      labels = IdxLoader.load(test_idx_stream)

      expect(labels.count).to eq(4)
      label_bytes.zip(labels).each do |expected_label, parsed_label|
        expect(expected_label).to eq(parsed_label)
      end
    end
  end
end
