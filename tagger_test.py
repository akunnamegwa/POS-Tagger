import os
import unittest
from unittest.mock import patch
from tagger import sentence_inferencing, viterbi_inference, write_solution, fine_tune

class TestTagger(unittest.TestCase):
    def setUp(self):
        # Use a sample test file based on your training data
        self.sample_test_file = ['Detective : NP0', 'Chief : NP0', 'Inspector : NP0', 'John : NP0',
                                 'McLeish : NP0', 'gazed : VVD', 'doubtfully : AV0', 'at : PRP',
                                 'the : AT0', 'plate : NN1', 'before : PRP', 'him : PNP', '. : PUN',
                                 'Having : VHG', 'thought : VVN', 'he : PNP', 'was : VBD', 'hungry : AJ0',
                                 ', : PUN', 'he : PNP', 'now : AV0', 'realized : VVD', 'that : CJT',
                                 'actually : AV0', 'he : PNP', 'needed : VVD', 'anything : PNI',
                                 'rather : AV0', 'than : CJS', 'the : AT0', 'overflowing : AJ0-VVG',
                                 'plate : NN1', 'of : PRF', 'cholesterol : NN1', 'the : AT0', 'canteen : NN1',
                                 'at : PRP', 'New : AJ0', 'Scotland : NP0', 'Yard : NN1', 'had : VHD',
                                 'provided : VVN', 'with : PRP', 'such : DT0', 'admirable : AJ0',
                                 'promptness : NN1', '. : PUN', '" : PUQ']

        self.output_file = 'test_output.txt'

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_sentence_inferencing(self):
        # Test with a sentence from the training data
        test_sentence = ['Detective', 'John', 'gazed', 'doubtfully', 'at', 'the', 'plate', '.']
        expected_result = [('Detective', 'NP0'), ('John', 'NP0'), ('gazed', 'VVD'), ('doubtfully', 'AV0'),
                           ('at', 'PRP'), ('the', 'AT0'), ('plate', 'NN1'), ('.', 'PUN')]
        result = sentence_inferencing(test_sentence)
        self.assertEqual(result, expected_result)

    def test_viterbi_inference(self):
        # Mocking input data for viterbi_inference
        test_data = [['Detective', 'John', 'gazed', 'doubtfully', 'at', 'the', 'plate', '.']]
        expected_result = [('Detective', 'NP0'), ('John', 'NP0'), ('gazed', 'VVD'), ('doubtfully', 'AV0'),
                           ('at', 'PRP'), ('the', 'AT0'), ('plate', 'NN1'), ('.', 'PUN')]

        with patch('builtins.input', side_effect=test_data):
            result = viterbi_inference(test_data)
        self.assertEqual(result, expected_result)

    def test_write_solution(self):
        # Test writing a solution to the output file
        test_data = [('Detective', 'NP0'), ('John', 'NP0'), ('gazed', 'VVD'), ('doubtfully', 'AV0'),
                     ('at', 'PRP'), ('the', 'AT0'), ('plate', 'NN1'), ('.', 'PUN')]
        write_solution(test_data, self.output_file)

        # Check if the file was created and contains the expected content
        with open(self.output_file, 'r') as file:
            content = file.read()
        expected_content = "Detective : NP0\nJohn : NP0\ngazed : VVD\ndoubtfully : AV0\nat : PRP\n" \
                           "the : AT0\nplate : NN1\n. : PUN\n"
        self.assertEqual(content, expected_content)

    def test_fine_tune(self):
        # Test fine-tuning based on English structure
        test_data = [('Detective', 'NP0'), ('gazed', 'VVD'), ('doubtfully', 'AV0'),
                     ('at', 'PRP'), ('the', 'AT0'), ('plate', 'NN1'), ('.', 'PUN')]
        fine_tune(test_data)

        # Check if the tags were updated as expected
        expected_result = [('Detective', 'NP0'), ('gazed', 'VVD'), ('doubtfully', 'AV0'),
                           ('at', 'PRP'), ('the', 'AT0'), ('plate', 'NN1'), ('.', 'PUN')]
        self.assertEqual(test_data, expected_result)

if __name__ == '__main__':
    unittest.main()

