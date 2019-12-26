import random
import numpy as np
import nltk
import sklearn
import operator
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import normalize


nltk.download('stopwords')  # If needed
nltk.download('punkt')  # If needed
nltk.download('wordnet')  # If needed

start = time.time()


# function read data from local txt file generating training set, test set and development set.
def loadDataSet():
    train_set_pos_url = "IMDb/train/imdb_train_pos.txt"
    train_set_neg_url = "IMDb/train/imdb_train_neg.txt"
    train_set_pos_open = open(train_set_pos_url).read()
    train_set_neg_open = open(train_set_neg_url).read()
    train_set_pos = train_set_pos_open.split("\n")
    train_set_neg = train_set_neg_open.split("\n")
    dataset_train = []
    for pos_train in train_set_pos: dataset_train.append((pos_train, 1))
    for neg_train in train_set_neg: dataset_train.append((neg_train, 0))

    test_set_pos_url = "IMDb/test/imdb_test_pos.txt"
    test_set_pos_open = open(test_set_pos_url).read()
    test_set_pos = test_set_pos_open.split("\n")
    test_set_neg_url = "IMDb/test/imdb_test_neg.txt"
    test_set_neg_open = open(test_set_neg_url).read()
    test_set_neg = test_set_neg_open.split("\n")
    dataset_test = []
    for pos_test in test_set_pos: dataset_test.append((pos_test, 1))
    for neg_test in test_set_neg: dataset_test.append((neg_test, 0))

    dev_set_pos_url = "IMDb/dev/imdb_dev_pos.txt"
    dev_set_pos_open = open(dev_set_pos_url).read()
    dev_set_pos = dev_set_pos_open.split("\n")
    dev_set_neg_url = "IMDb/dev/imdb_dev_neg.txt"
    dev_set_neg_open = open(dev_set_neg_url).read()
    dev_set_neg = dev_set_neg_open.split("\n")
    dataset_dev = []
    for pos_dev in dev_set_pos: dataset_dev.append((pos_dev, 1))
    for neg_dev in dev_set_neg: dataset_dev.append((neg_dev, 0))
    random.shuffle(dataset_train)
    random.shuffle(dataset_test)
    random.shuffle(dataset_dev)
    return dataset_train, dataset_test, dataset_dev


# use dictionary of pos and neg word
dict_pos_url = "opinion-lexicon-English/positive-words.txt"
dict_neg_url = "opinion-lexicon-English/negative-words.txt"
dict_pos = open(dict_pos_url).read()
dict_neg = open(dict_neg_url).read()
dict_pos = dict_pos.split("\n")
dict_neg = dict_neg.split("\n")

# initialize lemmatizer and stopword
lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("<")
stopwords.add(">")
stopwords.add("\"\"\"")


# Function count frequency of postive, negative words and the length of sentence
def get_count_vector(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    pos_word = 0
    neg_word = 0
    sentence_len = 0
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            token = lemmatizer.lemmatize(token).lower()
            sentence_len += 1
            if token not in stopwords:
                if token in dict_pos: pos_word+=1
                if token in dict_neg: neg_word+=1
    return [pos_word, neg_word, sentence_len]


# Function receive a review and return tokenize format
# Notice: This function code is retrived from lab code
def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    list_tokens = []
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens


# Function create adj verb dictionary of size "num_features" based on 1000 random review in dataset
# This function has some part similar to the lab code but was modified and add more code to satisfy
# the requirement.
def get_adj_verb_dictionary(dataset, num_features):
    dict_word_frequency = {}
    tag_list = ['JJ', 'JJR', 'JJS', 'VB']
    for instance in dataset[:1000]:
        list_tokens = get_list_tokens(instance[0])
        tag_set = nltk.pos_tag(list_tokens)
        for word, tag in tag_set:
            if word in stopwords: continue
            if tag in tag_list:
                if word not in dict_word_frequency:
                    dict_word_frequency[word] = 1
                else:
                    dict_word_frequency[word] += 1
            else:
                continue
    print(len(dict_word_frequency))
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
    vocabulary = []
    for word, frequency in sorted_list:
        vocabulary.append(word)
    return vocabulary


# Function generate adj, verb frequency based on dictionary generated below
def get_word_freq_vector(string):
    vector_text = []
    list_tokens_string = get_list_tokens(string)
    for i, word in enumerate(vocab):
        vector_text.append(list_tokens_string.count(word))
        """
        if word in list_tokens_string:
            vector_text.append(1)
        else:
            vector_text.append(0)
        """
    return vector_text


# Function combine 3 features vector: pos&neg word count, sentence length, adj and verb frequency
def combine_vector(dataset):
    X = np.zeros(shape=(len(dataset), size_of_vocab + 3))
    Y = []
    for i, instance in enumerate(dataset):
        pos_neg_sen = get_count_vector(instance[0])
        word_freq = get_word_freq_vector(instance[0])
        pos_neg_sen.extend(word_freq)
        X[i] = pos_neg_sen
        Y.append(instance[1])
    return X, Y


# Train SVM clf with feature selection by chi-square method
def train_svm_classifier(X, Y, kernel, gamma, paraC):
    svm_clf = sklearn.svm.SVC(kernel=kernel, gamma=gamma, C=paraC)
    svm_clf.fit(X, np.asarray(Y))
    return svm_clf


# Function using dev set to select best parameter for model including kernel and paraC and gamma
def select_best_para(dataset):
    list_paraC = [0.01,0.1, 1]
    list_kernel = ["rbf", "linear"]
    list_gamma = ['scale']
    best_accuracy_dev = 0.0
    X_dev_old, Y_dev = combine_vector(dataset)
    X_dev = fs.transform(X_dev_old)
    X_dev_train, X_dev_test, Y_dev_train, Y_dev_test = train_test_split(X_dev, Y_dev, test_size=0.2, random_state=0)
    for gamma in list_gamma:
        for paraC in list_paraC:
            for kernel in list_kernel:
                svm_dev = train_svm_classifier(X_dev_train, Y_dev_train, kernel, gamma, paraC)
                Y_dev_gold = np.asarray(Y_dev_test)
                Y_dev_predictions = svm_dev.predict(np.asarray(X_dev_test))
                accuracy_dev = accuracy_score(Y_dev_gold, Y_dev_predictions)
                print("Accuracy with " + str(kernel) + " kernel and gamma " + str(gamma) + " para C " + str(
                    paraC) + ": " + str(
                    round(accuracy_dev, 3)))
                if accuracy_dev >= best_accuracy_dev:
                    best_accuracy_dev = accuracy_dev
                    best_kernel = kernel
                    best_paraC = paraC
                    best_gamma = gamma
    print("\n Best accuracy overall in the dev set is " + str(round(best_accuracy_dev, 3)) + " with "
          + str(best_kernel) + " kernel and gamma " + str(best_gamma) + " para C " + str(best_paraC))
    return best_kernel, best_gamma, best_paraC


# Function generate test report based on test set
def score_svm(dataset):
    X_test_old, Y_test = combine_vector(dataset)
    X_test = fs.transform(X_test_old)
    Y_test_gold = np.asarray(Y_test)
    Y_text_predictions = svm_clf.predict(X_test)
    print(classification_report(Y_test_gold, Y_text_predictions))

# first: read data from file
dataset_train, dataset_test, dataset_dev = loadDataSet()
session0 = time.time()

# second: generate adj, verb vocabulary on training set
print("start generating vocabulary")
size_of_vocab = 1000
vocab = get_adj_verb_dictionary(dataset_train, size_of_vocab)
session1 = time.time()
print("generating vocabulary time: " + str(session1 - session0))

# third: combine three features vector in one, Notice: This step could take around 5 min to finish
print("start transforming vector")
X_train_old, Y_train = combine_vector(dataset_train)
session2 = time.time()
print("vector transforming time: " + str(session2 - session1))

# four: using f_classif to select most relevant features
print("start f_classif feature selection")
num_features=503
fs = SelectKBest(f_classif, k=num_features).fit(X_train_old, Y_train)
X_train = fs.transform(X_train_old)
session3 = time.time()
print("feature selection time: " + str(session3 - session2))

print("start selecting parameter on dev set")
kernel, gamma, paraC = select_best_para(dataset_dev)  # five: using dev set to select best kernel and para C value
session4 = time.time()
print("select parameter on dev set time: " + str(session4 - session3))

print("start training")
svm_clf = train_svm_classifier(X_train, Y_train, kernel, gamma, paraC)  # six: using optimized parameter to train model
session5 = time.time()
print("training time: " + str(session5 - session4))

print("start testing")
score_svm(dataset_test) # seven: generate classification report with trained model on test set
end = time.time()

print("initialize time: " + str(session0 - start))
print("generating vocabulary: " + str(session1 - session0))
print("vector transformation time: " + str(session2 - session1))
print("feature selection time: " + str(session3 - session2))
print("select parameter time: " + str(session4 - session3))
print("training time: " + str(session5 - session4))
print("testing time: " + str(end - session5))
print("total time: " + str(end - start))
# 546s fitst trial
# 349s Second trial

"""
------------------------------------------------------------------------------------------------------------------
following lines are test data trying to optimize time and performance and also find out the problem
of overfitting, feel free to check or ignore.
------------------------------------------------------------------------------------------------------------------

3000 vocab 500 features
finish initializing
finish generating vocabulary
generating vocabulary: 38.162075996398926
Accuracy with linear kernel and para C 0.01: 0.833
Accuracy with rbf kernel and para C 0.01: 0.504
Accuracy with linear kernel and para C 1: 0.862
Accuracy with rbf kernel and para C 1: 0.852
Accuracy with linear kernel and para C 10: 0.864
Accuracy with rbf kernel and para C 10: 0.94
 Best accuracy overall in the dev set is 0.94 with rbf kernel and para C 10
finish selecting parameter
select parameter time: 461.57385897636414
finish vectorizing
finish training
              precision    recall  f1-score   support
           0       0.84      0.79      0.81      2502
           1       0.80      0.84      0.82      2500
    accuracy                           0.82      5002
   macro avg       0.82      0.82      0.82      5002
weighted avg       0.82      0.82      0.82      5002
finish testing
initialize time: 0.12410688400268555
generating vocabulary: 38.162075996398926
select parameter time: 461.57385897636414
vectorize time: 275.4013319015503
training time: 187.68213319778442
testing time: 104.45805716514587
total time: 1067.4015641212463


2000 vocab 500 features
finish initializing
finish generating vocabulary
generating vocabulary: 24.23247003555298
Accuracy with linear kernel and para C 0.01: 0.835
Accuracy with rbf kernel and para C 0.01: 0.504
Accuracy with linear kernel and para C 1: 0.87
Accuracy with rbf kernel and para C 1: 0.854
Accuracy with linear kernel and para C 10: 0.868
Accuracy with rbf kernel and para C 10: 0.941
 Best accuracy overall in the dev set is 0.941 with rbf kernel and para C 10
finish selecting parameter
select parameter time: 526.6542026996613
finish vectorizing
finish training
              precision    recall  f1-score   support
           0       0.83      0.79      0.81      2502
           1       0.80      0.84      0.82      2500
    accuracy                           0.82      5002
   macro avg       0.82      0.82      0.81      5002
weighted avg       0.82      0.82      0.81      5002
finish testing
initialize time: 0.1568281650543213
generating vocabulary: 24.23247003555298
select parameter time: 526.6542026996613
vectorize time: 264.00054121017456
training time: 105.03458189964294
testing time: 107.90838503837585
total time: 1027.987009048462


1000 vocab 500 features
finish initializing
finish generating vocabulary
generating vocabulary: 15.126110792160034
Accuracy with linear kernel and para C 0.01: 0.834
Accuracy with rbf kernel and para C 0.01: 0.504
Accuracy with linear kernel and para C 1: 0.86
Accuracy with rbf kernel and para C 1: 0.864
Accuracy with linear kernel and para C 10: 0.865
Accuracy with rbf kernel and para C 10: 0.953
 Best accuracy overall in the dev set is 0.953 with rbf kernel and para C 10
finish selecting parameter
select parameter time: 414.48377323150635
finish vectorizing
finish training
              precision    recall  f1-score   support
           0       0.83      0.79      0.81      2502
           1       0.80      0.84      0.82      2500
    accuracy                           0.82      5002
   macro avg       0.82      0.82      0.81      5002
weighted avg       0.82      0.82      0.81      5002
finish testing
initialize time: 0.15787982940673828
generating vocabulary: 15.126110792160034
select parameter time: 414.48377323150635
vectorize time: 268.0847818851471
training time: 156.7295470237732
testing time: 118.50122690200806
total time: 973.0833196640015

rbf kernel with different gamma
finish initializing
finish generating vocabulary
generating vocabulary: 12.623382091522217
Accuracy with rbf kernel and gamma scale para C 1: 0.851
Accuracy with rbf kernel and gamma auto para C 1: 0.814
Accuracy with rbf kernel and gamma 0.001 para C 1: 0.788
Accuracy with rbf kernel and gamma 0.0001 para C 1: 0.597
 Best accuracy overall in the dev set is 0.851 with rbf kernel and gamma scale para C 1
finish selecting parameter
select parameter time: 232.87182092666626
finish vectorizing
finish training
              precision    recall  f1-score   support
           0       0.84      0.77      0.81      2502
           1       0.79      0.85      0.82      2500
    accuracy                           0.81      5002
   macro avg       0.82      0.81      0.81      5002
weighted avg       0.82      0.81      0.81      5002
finish testing
initialize time: 0.15407896041870117
generating vocabulary: 12.623382091522217
select parameter time: 232.87182092666626
vectorize time: 269.2015359401703
training time: 88.54790997505188
testing time: 110.84156203269958
total time: 714.2402899265289


rbf kernel with different gamma and linear kernel
finish initializing
finish generating vocabulary
generating vocabulary: 13.726391077041626
Accuracy with rbf kernel and gamma scale para C 1: 0.851
Accuracy with linear kernel and gamma scale para C 1: 0.858
Accuracy with rbf kernel and gamma 0.01 para C 1: 0.883
Accuracy with linear kernel and gamma 0.01 para C 1: 0.858
Accuracy with rbf kernel and gamma 0.1 para C 1: 0.996
Accuracy with linear kernel and gamma 0.1 para C 1: 0.858
 Best accuracy overall in the dev set is 0.996 with rbf kernel and gamma 0.1 para C 1
finish selecting parameter
select parameter time: 315.2392077445984
finish vectorizing
finish training
              precision    recall  f1-score   support
           0       0.73      0.81      0.77      2502
           1       0.79      0.70      0.74      2500
    accuracy                           0.75      5002
   macro avg       0.76      0.75      0.75      5002
weighted avg       0.76      0.75      0.75      5002
finish testing
initialize time: 0.1735210418701172
generating vocabulary: 13.726391077041626
select parameter time: 315.2392077445984
vectorize time: 271.67462515830994
training time: 325.8220431804657
testing time: 125.21120190620422
total time: 1051.84699010849

finish initializing
finish generating vocabulary
generating vocabulary: 13.596699714660645
Accuracy with linear kernel and gamma scale para C 1: 0.913
Accuracy with rbf kernel and gamma scale para C 1: 0.873
 Best accuracy overall in the dev set is 0.913 with linear kernel and gamma scale para C 1
finish selecting parameter
select parameter time: 212.10210609436035
finish vectorizing
finish training
              precision    recall  f1-score   support
           0       0.84      0.81      0.83      2502
           1       0.82      0.85      0.83      2500
    accuracy                           0.83      5002
   macro avg       0.83      0.83      0.83      5002
weighted avg       0.83      0.83      0.83      5002
finish testing
initialize time: 0.23033404350280762
generating vocabulary: 13.596699714660645
select parameter time: 212.10210609436035
vectorize time: 283.6328110694885
training time: 333.08015394210815
testing time: 121.63984704017639
total time: 964.2819519042969

finish initializing
6868
finish generating vocabulary
generating vocabulary: 12.53015923500061
finish vectorizing
vectorize time: 291.1285357475281
finish chi square testing
feature selection time: 0.3411550521850586
Accuracy with linear kernel and gamma scale para C 1: 0.877
Accuracy with rbf kernel and gamma scale para C 1: 0.909
 Best accuracy overall in the dev set is 0.909 with rbf kernel and gamma scale para C 1
finish selecting parameter
select parameter time: 153.64419603347778
finish training
training time: 91.02472519874573
              precision    recall  f1-score   support
           0       0.85      0.81      0.83      2502
           1       0.82      0.86      0.84      2500
    accuracy                           0.83      5002
   macro avg       0.84      0.83      0.83      5002
weighted avg       0.84      0.83      0.83      5002
finish testing
initialize time: 0.13055896759033203
generating vocabulary: 12.53015923500061
vectorize time: 291.1285357475281
chi square testing time: 0.3411550521850586
select parameter time: 153.64419603347778
training time: 91.02472519874573
testing time: 117.64035987854004
total time: 666.4396901130676


Add chi square testing selectkbest 1002 vocab 2000
finish initializing
7148
finish generating vocabulary
generating vocabulary: 13.105851888656616
finish vectorizing
vectorize time: 324.1820411682129
finish chi square testing
feature selection time: 0.6177289485931396
Accuracy with linear kernel and gamma scale para C 1: 0.915
 Best accuracy overall in the dev set is 0.915 with linear kernel and gamma scale para C 1
finish selecting parameter
select parameter time: 160.01560711860657
finish training
training time: 234.9708058834076
              precision    recall  f1-score   support
           0       0.85      0.82      0.84      2502
           1       0.83      0.85      0.84      2500
    accuracy                           0.84      5002
   macro avg       0.84      0.84      0.84      5002
weighted avg       0.84      0.84      0.84      5002
finish testing
initialize time: 0.12759804725646973
generating vocabulary: 13.105851888656616
vectorize time: 324.1820411682129
chi square testing time: 0.6177289485931396
select parameter time: 160.01560711860657
training time: 234.9708058834076
testing time: 142.2668318748474
total time: 875.2864649295807


finish initializing
6667
finish generating vocabulary
generating vocabulary: 13.683449745178223
finish vectorizing
vectorize time: 285.9945800304413
finish chi square testing
feature selection time: 0.35965633392333984
Accuracy with linear kernel and gamma scale para C 0.1: 0.809
Accuracy with rbf kernel and gamma scale para C 0.1: 0.753
Accuracy with linear kernel and gamma scale para C 1: 0.811
Accuracy with rbf kernel and gamma scale para C 1: 0.796
Accuracy with linear kernel and gamma scale para C 5: 0.797
Accuracy with rbf kernel and gamma scale para C 5: 0.79
 Best accuracy overall in the dev set is 0.811 with linear kernel and gamma scale para C 1
finish selecting parameter
select parameter time: 152.85847187042236
finish training
training time: 154.4430799484253
              precision    recall  f1-score   support
           0       0.85      0.82      0.83      2502
           1       0.83      0.86      0.84      2500
    accuracy                           0.84      5002
   macro avg       0.84      0.84      0.84      5002
weighted avg       0.84      0.84      0.84      5002
finish testing
initialize time: 0.18391108512878418
generating vocabulary: 13.683449745178223
vectorize time: 285.9945800304413
chi square testing time: 0.35965633392333984
select parameter time: 152.85847187042236
training time: 154.4430799484253
testing time: 114.85648679733276
total time: 722.379635810852

start vectorizing
vectorize time: 278.51699686050415
start chi sqaure feature selection
feature selection time: 0.28789496421813965
start selecting parameter on dev set
Accuracy with rbf kernel and gamma scale para C 0.1: 0.724
Accuracy with rbf kernel and gamma scale para C 1: 0.808
Accuracy with rbf kernel and gamma scale para C 5: 0.804
Accuracy with rbf kernel and gamma 0.01 para C 0.1: 0.733
Accuracy with rbf kernel and gamma 0.01 para C 1: 0.806
Accuracy with rbf kernel and gamma 0.01 para C 5: 0.816
 Best accuracy overall in the dev set is 0.816 with rbf kernel and gamma 0.01 para C 5
select parameter time: 147.8039951324463
start training svm model
training time: 78.09997797012329
start testing
              precision    recall  f1-score   support
           0       0.86      0.83      0.84      2502
           1       0.83      0.86      0.85      2500
    accuracy                           0.84      5002
   macro avg       0.84      0.84      0.84      5002
weighted avg       0.84      0.84      0.84      5002
initialize time: 0.16654610633850098
generating vocabulary: 12.838921070098877
vectorize time: 278.51699686050415
chi square testing time: 0.28789496421813965
select parameter time: 147.8039951324463
training time: 78.09997797012329
testing time: 111.95636487007141
total time: 629.6706969738007

start generating vocabulary
6715
generating vocabulary: 13.034069061279297
start vectorizing
vectorize time: 284.34582591056824
start chi sqaure feature selection
feature selection time: 0.2593369483947754
start selecting parameter on dev set
Accuracy with rbf kernel and gamma scale para C 0.1: 0.723
Accuracy with linear kernel and gamma scale para C 0.1: 0.804
Accuracy with rbf kernel and gamma scale para C 1: 0.806
Accuracy with linear kernel and gamma scale para C 1: 0.803
Accuracy with rbf kernel and gamma scale para C 5: 0.792
Accuracy with linear kernel and gamma scale para C 5: 0.8
Accuracy with rbf kernel and gamma scale para C 10: 0.784
Accuracy with linear kernel and gamma scale para C 10: 0.794
Accuracy with rbf kernel and gamma 0.01 para C 0.1: 0.731
Accuracy with linear kernel and gamma 0.01 para C 0.1: 0.804
Accuracy with rbf kernel and gamma 0.01 para C 1: 0.795
Accuracy with linear kernel and gamma 0.01 para C 1: 0.803
Accuracy with rbf kernel and gamma 0.01 para C 5: 0.805
Accuracy with linear kernel and gamma 0.01 para C 5: 0.8
Accuracy with rbf kernel and gamma 0.01 para C 10: 0.801
Accuracy with linear kernel and gamma 0.01 para C 10: 0.794
 Best accuracy overall in the dev set is 0.806 with rbf kernel and gamma scale para C 1
select parameter time: 268.4944612979889
start training svm model
training time: 87.89806175231934
start testing
              precision    recall  f1-score   support
           0       0.85      0.80      0.82      2502
           1       0.81      0.86      0.83      2500
    accuracy                           0.83      5002
   macro avg       0.83      0.83      0.83      5002
weighted avg       0.83      0.83      0.83      5002
initialize time: 0.15124106407165527
generating vocabulary: 13.034069061279297
vectorize time: 284.34582591056824
chi square testing time: 0.2593369483947754
select parameter time: 268.4944612979889
training time: 87.89806175231934
testing time: 126.39003801345825
total time: 780.5730340480804

tart generating vocabulary
6726
generating vocabulary time: 12.701534032821655
start vectorizing
vectorizing time: 294.58524107933044
start chi sqaure feature selection
chi square feature selection time: 0.2778339385986328
start selecting parameter on dev set
Accuracy with rbf kernel and gamma scale para C 0.1: 0.733
Accuracy with linear kernel and gamma scale para C 0.1: 0.823
Accuracy with rbf kernel and gamma scale para C 1: 0.806
Accuracy with linear kernel and gamma scale para C 1: 0.821
Accuracy with rbf kernel and gamma scale para C 5: 0.784
Accuracy with linear kernel and gamma scale para C 5: 0.818
 Best accuracy overall in the dev set is 0.823 with linear kernel and gamma scale para C 0.1
select parameter on dev set time: 160.0605571269989
start training
training time: 71.73239088058472
start testing
              precision    recall  f1-score   support
           0       0.85      0.81      0.83      2502
           1       0.82      0.86      0.84      2500
    accuracy                           0.84      5002
   macro avg       0.84      0.84      0.83      5002
weighted avg       0.84      0.84      0.83      5002
initialize time: 0.16392993927001953
generating vocabulary: 12.701534032821655
vectorize time: 294.58524107933044
chi square testing time: 0.2778339385986328
select parameter time: 160.0605571269989
training time: 71.73239088058472
testing time: 115.89063096046448
total time: 655.4121179580688


"""
