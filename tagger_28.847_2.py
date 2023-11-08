# This is A4 for CSC384. It is a Part of Speech (POS) tagger using the Viterbi Algorithm
import argparse
import numpy as np
import time
from collections import Counter
import numba as nb
# numba is not working on Markus #TODO: remove all decorators and for loop

# TODO: account for case sensitivity- except for proper nouns
# TODO: see if changing to general tagging makes a difference
# TODO: Your sentences need to take end markers inside of quotations or brackets into account
# TODO: if there is an ambiguity tag use the one with the highest probability as the lead
training_list = []
testfile = ''
outputfile = ''
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    outputfile = args.outputfile
    training_list = args.trainingfiles[0]
    testfile = args.testfile
start = time.time()

sentences = []  # [sentence, ...] | sentence = [(word, tag), ...]
tags = []  # [sentence tags, ...] | sentence tags = [tag, ]
tag_bank = {}  # {tag : {word1: count, word2: count} }
words = set()  # set of all words in training files
end_markers = ['.', '!', '?']
punc = ['(', '[', '{', '}', ']', ')', '.', ',', '!', '?', ':', '-', ';', '\'', '\"']
punctuations = set(punc)
punc_tags = ['PUN', 'PUL', 'PUR', 'PUQ']
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
        if info[0] == '':  # account for when the word is ':'
            if info[1] == ':':
                word = ':'
            if info[2].strip() == ': PUN':
                tag = 'PUN'

        sentence.append((word, tag))  # add pair to sentence
        tag_list.append(tag)  # add tag to list of tags
        words.add(word)  # add word to word bank and set
        if tag in tag_bank:
            tag_bank[tag].update([word])
        else:
            tag_bank[tag] = Counter([word])
        # if it's the end of a sentence
        if word in end_markers:
            # add the full sentence to global variable and reset local variable
            sentences.append(sentence)
            tags.append(tag_list)
            sentence = []
            tag_list = []
    if sentence:
        sentences.append(sentence)

# create a Counter for all tags
tag_count = 0
counted_tags = Counter()
for taglst in tags:
    counted_tags.update(taglst)
    tag_count += len(taglst)

# lists for tag and word order (alphabetical | observed_tags => AJ0 ... ZZ0)
observed_tags = sorted(set(counted_tags.elements()))
observed_words = sorted(words)

# make an array for initial probabilities
first_tags = []
for sentence in sentences:
    first_tags.append(sentence[0][1])
counted_first_tags = Counter(first_tags)
observed_first_tags = list(set(first_tags))
total_first_tags = sum(counted_first_tags.values())
initial_probabilities = []
for tag in observed_tags:
    if tag in observed_first_tags:
        initial_probabilities.append(counted_first_tags[tag] / total_first_tags)
    else:
        initial_probabilities.append(1.0)
initial_probabilities = np.array(initial_probabilities)
point = time.time()
print('\n', 'Initial Probabilities made at', point - start)


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
    observations = tag_bank[tag]
    word_count = observations[word]
    return word_count / sum(observations.values())


# create a matrix for transitional probabilities and observational probabilities
trans_list = []
observ_list = []
for tag1 in observed_tags:
    # accumulate transitional probabilities
    trans_for_tag = []
    for tag2 in observed_tags:
        trans_for_tag.append(transition_prob(tag1, tag2))
    trans_list.append(trans_for_tag)

    # accumulate observational probabilities
    obs_for_tag = []
    for word in observed_words:
        obs_for_tag.append(observation_prob(tag1, word))
    observ_list.append(obs_for_tag)
trans_matrix = np.array(trans_list)  # array[tag1, tag2]
observ_matrix = np.array(observ_list)  # array[tag, word]

point1 = time.time()
print('Matrices made at', point1 - start)

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
if sent:
    test_sentences.append(sent)


def word_inference1(word, prev_tag=None):
    # make observations 1 if the word is unseen
    if word in words:
        word_index = observed_words.index(word)
        word_observations = observ_matrix[:, word_index]
    else:
        word_observations = np.ones_like(trans_matrix[0])

    # make transition probabilities equal to initial probabilities if it is the first word
    if prev_tag is None:
        diff = len(trans_matrix[0]) - len(initial_probabilities)
        diff_array = np.ones(diff)
        prev_tag_transitions = np.concatenate(
            (initial_probabilities, diff_array))

    else:
        prev_index = observed_tags.index(prev_tag)
        prev_tag_transitions = trans_matrix[prev_index, :]

    observed_tags2 = observed_tags.copy()
    # remove all punctuation tags if the word is not a punctuation
    if word not in punctuations:
        punc_indices = []
        seen_pun = set()
        for t in punc_tags:
            if t in observed_tags:
                punc_indices.append(observed_tags.index(t))
                seen_pun.add(t)
        observed_tags2 = sorted(set(observed_tags.copy()) - seen_pun)
        x = word_observations.copy()
        y = prev_tag_transitions.copy()
        word_observations = np.delete(x, punc_indices)
        prev_tag_transitions = np.delete(y, punc_indices)

    best_tag_index = np.argmax(prev_tag_transitions * word_observations)

    # account for seen words with non-zero observations and zero trans. probs
    if np.max(prev_tag_transitions * word_observations) == 0:
        best_tag_index = np.argmax(word_observations)

    best_tag = observed_tags2[best_tag_index]
    return best_tag


# NOT USING ANYMORE, FOR-LOOPS TAKE TOO LONG
def word_inference(word, prev_tag=None):
    """
    :type word: str
    :type prev_tag: str
    :return: The most likely tag for word given the training files, using Viterbi sequencing
    """
    # use prev_tag as S(t-1)
    best_prob = 0
    best_tag = 'NONE FOUND'
    # if we are starting the sequence take initial probabilities into account
    if prev_tag is None:
        for i in range(len(observed_first_tags)):
            # if it is not a punctuation word skip PUN, PUL, PUR, and PUQ
            if word not in punctuations and observed_first_tags[i] in punc_tags:
                continue
            # the word is a punctuation skip all non-PU0 tags
            if word in punctuations and observed_first_tags[i] not in punc_tags:
                continue
            # begin to infer regularly
            t_index = observed_tags.index(observed_first_tags[i])
            initial = initial_probabilities[i]
            if word in words:
                word_index = observed_words.index(word)
                obs = observ_matrix[
                    t_index, word_index]  # observation_prob(t, word)
            else:
                obs = 1
            total_prob = initial * obs
            # print('Tag is:', t, 'Trans is:', trans, 'Obs is:', obs)
            if total_prob > best_prob:
                best_prob = total_prob
                best_tag = observed_first_tags[i]
    else:
        for t in observed_tags:
            # if it is not a punctuation word skip PUN, PUL, PUR, and PUQ
            if word not in punctuations and t in punc_tags:
                continue
            # the word is a punctuation skip all non-PU0 tags
            if word in punctuations and t not in punc_tags:
                continue
            # begin to infer regularly
            prev_index = observed_tags.index(prev_tag)
            t_index = observed_tags.index(t)
            trans = trans_matrix[prev_index, t_index]
            if word in words:
                word_index = observed_words.index(word)
                obs = observ_matrix[
                    t_index, word_index]  # observation_prob(t, word)
            else:
                obs = 1
            total_prob = trans * obs
            # print('Tag is:', t, 'Trans is:', trans, 'Obs is:', obs)
            if total_prob > best_prob:
                best_prob = total_prob
                best_tag = t
            if total_prob == best_prob:
                if word in words:
                    if observ_matrix[t_index, word_index] > observ_matrix[prev_index, word_index]:
                        best_tag = t

    # if a word has been observed and its observed tags haven't followed any tags NONE will be found
    if best_tag == 'NONE FOUND':
        if word in words:
            word_index = observed_words.index(word)
            obs = observ_matrix[:, word_index]  # observation_prob(t, word)
            max_obs = max(obs)
            i = np.where(observ_matrix == max_obs)[0][0]
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
        tag = word_inference1(word, prev_tag)
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


def write_solution(sentences, outputfile):
    """
    :type sentences: list[tuple[str,str]]
    :type outputfile: str
    :return: Writes a file with the complete POS tagging
    """
    file = open(outputfile, 'w')
    for pair in sentences:
        tagging = pair[0] + ' : ' + pair[1] + '\n'
        file.write(tagging)
    file.close()


def fine_tune(sentences):
    """
    :type sentences: list[tuple[str, str]]
    :return: list of sentences with more accurate tags
    """
    for i in range(len(sentences)):
        word = sentences[i][0].strip()
        tag = sentences[i][1].strip()
        if word in punctuations:  # make sure all punctuations are correct
            if word in ['.', '!', ':', '?', ',', '-', ';']:
                sentences[i] = (word, 'PUN')
            elif word in ['(', '[', '{']:
                sentences[i] = (word, 'PUL')
            elif word in [')', ']', '}']:
                sentences[i] = (word, 'PUR')
            elif word in ['"']:
                sentences[i] = (word, 'PUQ')

        if len(word) > 3:  # infer -ing words to be VVG
            if tag != 'VVG' and word not in words:
                if word[-3:] == 'ing' and 'VVG' not in tag:
                    sentences[i] = (word, 'VVG')
                if word[-3:] == 'ing' and 'VVG' in tag:
                    trail_tag = tag.strip('-').strip('VVG')
                    new_tag = 'VVG' + '-' + trail_tag
                    sentences[i] = (word, new_tag)

        if word == '':  # account for blank spaces
            sentences[i] = (word, 'PUN')

        elif word == 'to' and tag != 'PRP':  # infer 'to' as preposition if a noun follows
            if i + 3 < len(sentences):
                tag2 = sentences[i + 2][1]
                tag3 = sentences[i + 3][1]
                if tag2 == 'NN1':
                    sentences[i] = (word, 'PRP')
                if tag3 in ['AJ0', 'AJC', 'AJS'] and tag3 == 'NN1':
                    sentences[i] = (word, 'PRP')
            elif i + 2 < len(sentences):
                tag2 = sentences[i + 2][1]
                if tag2 == 'NN1':
                    sentences[i] = (word, 'PRP')

        elif len(word) == 1:  # make sure single letters get ZZ0
            if word not in ['a', 'A', 'I'] and word.isalpha():
                sentences[i] = (word, 'ZZ0')

        elif word.upper() in ['NOT', "N'T", 'N"T']:  # infer negation
            sentences[i] = (word, 'XX0')

        elif word != '':
            if word not in words and word[0].isupper():  # infer proper nouns
                if i > 0:
                    if sentences[i - 1][0].strip() not in end_markers:
                        sentences[i] = (word, 'NP0')

            if tag == 'NN1' and word[-1] == 's' and word not in words:  # infer plural nouns
                sentences[i] = (word, 'NN2')


result = viterbi_inference(test_sentences)
fine_tune(result)

write_solution(result, outputfile)
end = time.time()
print('Finished at', end - start)
