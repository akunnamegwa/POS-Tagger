import os
import sys
import argparse
import numpy as np
from collections import Counter

# TODO: make training_list and testfile a result of argparse if_name block
# TODO: hardcode punctuation, of, to
# TODO: account for case sensitivity- except for proper nouns
# TODO: make sure that sentence stopping is in effect
# TODO: see if changing to general tagging makes a difference
# Your sentences need to take end markers inside of quotations or brackets into account
training_list = ['train1.txt', 'train2.txt']
testfile = 'tester.txt'
outputfile = 'please_work.txt'
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--trainingfiles",
#         action="append",
#         nargs="+",
#         required=True,
#         help="The training files."
#     )
#     parser.add_argument(
#         "--testfile",
#         type=str,
#         required=True,
#         help="One test file."
#     )
#     parser.add_argument(
#         "--outputfile",
#         type=str,
#         required=True,
#         help="The output file."
#     )
#     args = parser.parse_args()
#
#     training_list = args.trainingfiles[0]
#     testfile = args.testfile
#     outputfile = args.outputfile
#     print("training files are {}".format(training_list))
#     print("test file is {}".format(args.testfile))
#     print("output file is {}".format(args.outputfile))
#     print("Starting the tagging process.")


sentences = []  # [sentence, ...] | sentence = [(word, tag), ...]
tags = []  # [sentence tags, ...] | sentence tags = [tag, ]
word_bank = {}  # {tag : {word1: count, word2: count} }
words = set()
end_markers = ['.', '!', '?']
# make a list of sentences from each training file
for file in training_list:
    text = open(file)
    lines = text.readlines()
    sentence = []
    tag_list = []
    for word_plus_tag in lines:
        info = word_plus_tag.partition(':')
        word = info[0].strip()
        tag = info[2].strip('\n').strip()

        sentence.append((word, tag))  # add pair to sentence
        tag_list.append(tag)  # add tag to list of tags
        words.add(word)  # add word to word bank and set
        if tag in word_bank:
            word_bank[tag].update([word])
        else:
            word_bank[tag] = Counter([word])

        # if it's the end of a sentence
        if word in end_markers:
            # add the full sentence to global variable and reset local variable
            sentences.append(sentence)
            tags.append(tag_list)
            sentence = []
            tag_list = []

# create a Counter for all tags
tag_count = 0
counted_tags = Counter()
for taglst in tags:
    counted_tags.update(taglst)
    tag_count += len(taglst)

# make a dict for initial probabilities
first_tags = []
for sentence in sentences:
    first_tags.append(sentence[0][1])
counted_first_tags = Counter(first_tags)
first_tags2 = list(set(first_tags))
total_first_tags = sum(counted_first_tags.values())
initial_probabilities = {}
for tag in first_tags2:
    initial_probabilities[tag] = counted_first_tags[tag] / total_first_tags
print('\n', 'Initial Probabilities', initial_probabilities)


# function to find the transitional probabilities from one tag to another
def transition_prob(tag1, tag2):
    """
    :type tag1: str
    :type tag2: str
    :return: the transitional probability from tag1 to tag2
    """
    tag1_count = counted_tags[tag1]
    tag2_count = counted_tags[tag2]
    # return 0 if tag1 or tag2 does not exist in training file
    if tag1_count == 0 or tag2_count == 0:
        return 0
    transition_indices = []
    for taglst in tags:
        tag1_indices = [i for i, j in enumerate(taglst) if j == tag1]
        taglst_transition_indices = []
        for i in tag1_indices:
            if i + 1 < len(taglst):
                if taglst[i + 1] == tag2:
                    taglst_transition_indices.append(i)
        transition_indices.extend(taglst_transition_indices)
    return len(transition_indices) / tag1_count


# function to find observational probabilities for observed words
def observation_prob(tag, word):
    """
    :type tag: str
    :type word: str
    :return: The observational probability of word as tag
    """
    observations = word_bank[tag]  # count of all words with tag
    word_count = observations[word]
    return word_count / sum(observations.values())


observed_tags = sorted(set(counted_tags.elements()))
observed_words = sorted(words)

# create a matrix for transitional probabilities and observational probabilities
trans_matrix = []
observ_matrix = []
for tag1 in observed_tags:
    trans_for_tag = []
    for tag2 in observed_tags:
        trans_for_tag.append(transition_prob(tag1, tag2))
    trans_matrix.append(trans_for_tag)

    obs_for_tag = []
    for word in observed_words:
        obs_for_tag.append(observation_prob(tag1, word))
    observ_matrix.append(obs_for_tag)
trans_matrix = np.array(trans_matrix)  # array[tag1, tag2]
observ_matrix = np.array(observ_matrix)  # array[tag, word]


# TODO: split test words into sentences
# prep test file and tags for viterbi
text = open(testfile)
lines = text.readlines()
test_words = []
for word in lines:
    test_words.append(word.strip('\n'))
test_sentences = []
sent = []
for word in test_words:
    sent.append(word)
    if word in end_markers:
        test_sentences.append(sent)
        sent = []


def word_inference(word, prev_tag=None):
    """
    :type word: str
    :type prev_tag: str
    :return: The most likely tag for word given the training files and following Viterbi sequencing
    """
    # use prev word as S(t-1)
    best_prob = 0
    best_tag = 'NONE FOUND'
    if prev_tag is None:
        for t in initial_probabilities:
            t_index = observed_tags.index(t)
            initial = initial_probabilities[t]
            if word in words:
                word_index = observed_words.index(word)
                obs = observ_matrix[t_index, word_index]  # observation_prob(t, word)
            else:
                obs = 1
            total_prob = initial * obs
            # print('Tag is:', t, 'Trans is:', trans, 'Obs is:', obs)
            if total_prob > best_prob:
                best_prob = total_prob
                best_tag = t
    # if we are starting the sequence take initial probabilites into acount
    else:
        for t in observed_tags:
            prev_index = observed_tags.index(prev_tag)
            t_index = observed_tags.index(t)
            trans = trans_matrix[prev_index, t_index]
            if word in words:
                word_index = observed_words.index(word)
                obs = observ_matrix[t_index, word_index]  # observation_prob(t, word)
            else:
                obs = 1
            total_prob = trans * obs
            # print('Tag is:', t, 'Trans is:', trans, 'Obs is:', obs)
            if total_prob > best_prob:
                best_prob = total_prob
                best_tag = t
    # if a word has been observed and its observed tags haven't followed any tags NONE will be found
    if best_tag == 'NONE FOUND':
        if word in words:
            word_index = observed_words.index(word)
            obs = observ_matrix[:, word_index]  # observation_prob(t, word)
            max_obs = max(obs)
            i = np.where(observ_matrix==max_obs)[0][0]
            best_tag = observed_tags[i]
        else:  # I don't think it'll have the same problem for unobserved words
            pass

    return best_tag


def sentence_inferencing(sent):
    """
    :param sent: A test file sentence
    :type sent: list[str]
    :return: A list of words matched with their likely tags
    """
    inferences = []
    prev_tag = None
    for word in sent:
        tag = word_inference(word, prev_tag)
        pair = word, tag
        inferences.append(pair)
        prev_tag = tag
    return inferences


def viterbi_inference(test):
    """
    :type test: list[list[str]]
    :return:
    """
    result = []
    for sent in test:
        result.extend(sentence_inferencing(sent))
    return result


def write_solution(sentences, outpufile):
    """
    :type sentences: list[tuple[str,str]]
    :type outpufile: str
    :return: Writes a file with the complete POS tagging
    """
    file = open(outpufile, 'w')
    for pair in sentences:
        tagging = pair[0] + ' : ' + pair[1] + '\n'
        file.write(tagging)
    file.close()


result = viterbi_inference(test_sentences)
write_solution(result, outputfile)
# you can set then tuple tags here to create an ordered iterator for the tags
