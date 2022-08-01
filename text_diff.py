from api.api import *
from textstat.textstat import legacy_round, textstatistics

# Feature: text length
def features_len_text(text):
    # Normalize to a number between 0..1
    len_text = [len(text) / 3000]
    return len_text


# Feature: sentence length
def features_len_sentence(text):
    replace_list = [
        '\n', ':', ';", '
        '', '--', ' $', ' he ', ' him ', ' she ', ' her ', ' the ', '1', '2',
        '3', '4', '5', '6', '7', '8', '9', '0', ' they ', ' them '
    ]

    for i in replace_list:
        text = text.replace(i, ' ')

    sentences = re.split('[.?!]', text)

    # Ignore all sentences shorter than 5 words
    sentences = [i for i in sentences if len(i) >= 5]

    num_of_sentence = len(sentences)
    len_sentence = [0] * num_of_sentence

    words_array = [(i.strip().split(" ")) for i in sentences]

    for i in range(0, num_of_sentence):
        for word in words_array[i]:
            if len(word) > 1:
                len_sentence[i] += 1

    # Ignore all sentences where length is bigger than 3-sigma
    len_sentence = mean_3_sigma(len_sentence)

    # Build a sentence length histogram with 20 buckets
    len_sentence_hist = [0] * 20
    for i in len_sentence:
        len_sentence_hist[min(20, i) % 20] += 1
    len_sentence_hist = [i / sum(len_sentence_hist) for i in len_sentence_hist]

    return len_sentence_hist


# Feature: word length
def features_len_word(text):

    # Split into single word, and convert every word to its original form, for example "apples" to "apple"
    words = splitwords(text)
    len_word = [len(i) for i in words]

    # Build a word length histogram with 5 buckets
    len_word_hist = [0] * 5
    for i in len_word:
        len_word_hist[min(21, i) % 5] += 1
    len_word_hist = [i / sum(len_word_hist) for i in len_word_hist]

    return len_word_hist


# Feature: number of sentences
def features_num_sentence(text):
    sentences = re.split('[.?!]', text)
    num_of_sentences = [len(sentences) / 32]

    return num_of_sentences


# Feature: syllables per word
def features_syllables_word(text):
    words = splitwords(text)
    syllable_word = [textstatistics().syllable_count(i) for i in words]

    # Make a histogram with 5 buckets
    syllable_word_hist = [0] * 5
    for i in syllable_word:
        syllable_word_hist[min(5, i) % 5] += 1
    syllable_word_hist = [
        i / sum(syllable_word_hist) for i in syllable_word_hist
    ]

    return syllable_word_hist


# Feature: number of unique words
def features_unique_words(text):
    words_array = splitwords(text)
    unique_words = []

    # Only count the unique words
    for word in words_array:
        if word not in unique_words:
            unique_words.append(word)

    len_unique_words = [len(unique_words) / 230]

    return len_unique_words


print("1. Assigning difficulty level to each word")
textbooks_data = load_textbooks_data()
diff_level = get_diff_level(textbooks_data)

print("2. Generating features from training data")
newFeatures = [
    features_len_text, features_len_sentence,
    features_len_word, features_num_sentence,
    features_syllables_word, features_unique_words
]

testtime = 10
order_acc_list = []
logic_acc_list = []

print("3. Training the regression models, and evaluating the accuracy")
for i in range(testtime):
    print("Start", i, "running ...")

    shuffle_data(10)

    train_data = load_train_data()
    train_feats, train_labels = get_feats_labels(train_data,
                                                 diff_level,
                                                 newFeatures=newFeatures,
                                                 diff_use=1)

    test_data = load_test_data()
    test_feats, test_labels = get_feats_labels(test_data,
                                               diff_level,
                                               newFeatures=newFeatures,
                                               diff_use=1)

    # Running the logistic regression model training and evaluation
    model = logistic_regression()
    model.train(train_feats, train_labels)

    pred_y = model.pred(test_feats)
    acc = accuracy(pred_y, test_labels)
    logic_acc_list.append(acc)

    # Running the order regression model training and evaluation
    model = order_regression()
    model.train(train_feats, train_labels)

    pred_y = model.pred(test_feats)
    acc = accuracy(pred_y, test_labels)
    order_acc_list.append(acc)

print("The final model outputs:")
print("10x logistic regression model evals: ", logic_acc_list)
print("10x order regression model evals: ", order_acc_list)
print("Logic_avg_acc     ", sum(logic_acc_list) / testtime)
print("Order_avg_acc     ", sum(order_acc_list) / testtime)
