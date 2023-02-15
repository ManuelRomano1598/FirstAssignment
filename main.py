import nltk
import math
import random
import numpy as np
from nltk.corpus import europarl_raw
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# downloads list of european languages
nltk.download('europarl_raw')

#i get only a small sample of some of the languages to avoid to much differences in lenght
english = europarl_raw.english.chapters()
nonEnglish = europarl_raw.dutch.chapters() + europarl_raw.greek.chapters() + europarl_raw.finnish.chapters() + europarl_raw.french.chapters() + europarl_raw.german.chapters() + europarl_raw.italian.chapters() + europarl_raw.portuguese.chapters()

#function that removes punctuation from the corpus
def remove_punctuation(corpus):
    cleaned_corpus = []
    cleaned_sent= []
    cleaned_words = []
    for chapter in corpus:
        for sent in chapter:
            for word in sent:
                if word.isalpha():
                    cleaned_words.append(word)
            cleaned_sent.append(cleaned_words)
            cleaned_words=[]
        cleaned_corpus.append(cleaned_sent)
        cleaned_sent = []
    return cleaned_corpus

#remove punctuation to avoid false freq reading
english = remove_punctuation(english)
nonEnglish = remove_punctuation(nonEnglish)

#shuffle the corpus
random.shuffle(english)
random.shuffle(nonEnglish)

nonEnglish = nonEnglish[:100]

#divide the corpus in test and training sets
#keeping the distinction between english and not english to calculate the prior and the likelihoods
test_size = 0.2
eng_train, eng_test = train_test_split(english, test_size=test_size)
non_eng_train, non_eng_test = train_test_split(nonEnglish, test_size=test_size)

#DEBUG test to see if precison was always 100%
#non_eng_test =[[['Capital', 'tax', 'The', 'next', 'item', 'is', 'the', 'joint', 'debate', 'on', 'the', 'oral', 'question', 'by', 'Mr', 'Désir', 'and', 'others', 'to', 'the', 'Council', 'on', 'the', 'Council', 'position', 'on', 'the', 'idea', 'of', 'a', 'capital', 'tax', 'the', 'oral', 'question', 'by', 'Mr', 'Désir', 'and', 'others', 'to', 'the', 'Commission', 'on', 'the', 'Commission', 'position', 'on', 'the', 'idea', 'of', 'a', 'capital', 'tax']]] + non_eng_test

#DEBUG test with mixed sentence
#non_eng_test = [[['the','Capital', 'la', 'The', 'item', 'is', 'il', 'joint', 'debate', 'caso', 'the', 'oral', 'funziona', 'by', 'Mr', 'Désir', 'perchè', 'others', 'to', 'the', 'Council', 'on', 'the', 'Council', 'ringrazio', 'on', 'the', 'casa']]] + non_eng_test

#length of each corpus
english_len = len(eng_train)
non_english_len = len(non_eng_train)

#calculate the prior with chapters number
english_prior = english_len / (english_len + non_english_len)
non_english_prior = non_english_len / (english_len + non_english_len)

#for each corpus i get the frequencies of each word and the number of words
english_fd = {}
non_english_fd = {}
eng_word_count = 0
non_eng_word_count = 0

for chapter in eng_train:
    for sent in chapter:
        english_fd.update(FreqDist(sent))
        eng_word_count += len(sent)

for chapter in non_eng_train:
    for sent in chapter:
        non_english_fd.update(FreqDist(sent))
        non_eng_word_count += len(sent)

#transform from frequencies to disctionary of likelihoods
english_prob = {}
non_english_prob = {}

for word in english_fd:
    english_prob[word] = (english_fd[word] + 1) / eng_word_count

for word in non_english_fd:
    non_english_prob[word] = (non_english_fd[word] + 1) / non_eng_word_count


#this function simply gets the likelihood of a word avoiding unknown words
def trainer(target, dict_of_freq):
    #return one if UNK so that log(1)=0
    likelihood = 1

    if target in dict_of_freq:
        likelihood = dict_of_freq[target]

    return likelihood


#test fase
#merge the two corpus and label them with golden labels
#label 1 = english, label 0 =non english
test_corpus = eng_test + non_eng_test
labels = [1] * len(eng_test) + [0] * len(non_eng_test)

# Shuffle the corpus and the labels in the same order
#then unshuffle keeping relation with labels
combined = list(zip(test_corpus, labels))
random.shuffle(combined)
test_corpus[:], labels[:] = zip(*combined)

#likelihood of each class
likelihood1 = 0
likelihood2 = 0

#list of estimated y
guessed_labels = []

#add the log of the likelihood of each word in the sentence of a chapter
#then add the log of the prior
for chapter in test_corpus:
    for sent in chapter:
        for word in sent:
            likelihood1 += abs(math.log10(trainer(word, english_prob)))
    probClass1 = abs(math.log10(english_prior)) + likelihood1

    for sent in chapter:
        for word in sent:
            likelihood2 += abs(math.log10(trainer(word, non_english_prob)))
    probClass2 = abs(math.log10(non_english_prior)) + likelihood2

    #argmax of the probabilities and guess
    if probClass1 > probClass2:
        guessed_labels.append(1)
    else:
        guessed_labels.append(0)

    #set to zero to repeat for the next chapters
    probClass1 = 0
    probClass2 = 0
    likelihood2 = 0
    likelihood1 = 0

#calculate the confusion matrix
conf_matrix = confusion_matrix(labels, guessed_labels)

#flip the label to match the text theory (this function sets true positive at (1,1) instead of (0,0))
conf_matrix = np.fliplr(conf_matrix)
conf_matrix = np.flipud(conf_matrix)

#calculate evaluation metrics
precision = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[1][0])

#print
print("confusion matrix: \n", conf_matrix)
print("precision: ", precision)
print("recall: ", recall)
print("accuracy: ", accuracy)
