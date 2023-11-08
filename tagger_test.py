# File with tests for Tagger.py


def create_test(text, output):
    """
    Create a test file based on text from a training file
    :param output:
    :type text: str
    :param text:
    :type output: str
    :return:
    """
    file = open(text)
    lines = file.readlines()
    file2 = open(output, 'w')
    for pair in lines:
        word = pair.partition(':')[0].strip()
        file2.write(word + '\n')

    file2.close()
    file.close()


def count_matches(correct_solution, tagger_solution):
    """
    :type correct_solution: file
    :type tagger_solution: file
    :return:
    """
    correct = []
    tagger = []
    c_lines = open(correct_solution).readlines()
    t_lines = open(tagger_solution).readlines()
    for i in range(len(c_lines)):
        c_info = c_lines[i].partition(':')
        c_word = c_info[0].strip()
        c_tag = c_info[2].strip('\n').strip()

        t_info = t_lines[i].partition(':')
        t_word = t_info[0].strip()
        t_tag = t_info[2].strip('\n').strip()

        correct.append((c_word, c_tag))
        tagger.append((t_word, t_tag))

    result = []
    for i in range(len(correct)):
        if correct[i] != tagger[i]:
            result.append((i, correct[i][0], correct[i][1], tagger[i][1]))

    accuracy = str(((len(result) / len(correct)) - 1) * -100
                   )
    # TODO: create a test output file instead of printing
    message = 'Accuracy is ' + accuracy + '\n' + 'The following are incorrect \n'
    for answer in result:
        mess = 'Line {0}: {1} should be {2}, got {3}'
        mess = mess.format(answer[0], answer[1], answer[2], answer[3])
        message += mess + '\n'

    return message


x = count_matches('training2.txt', 'init2_test.txt')
print(x)
