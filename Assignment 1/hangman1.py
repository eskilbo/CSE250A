import numpy as np
import pandas as pd
#import random

### Problem 1a ###

# Read lines from file and save as pandas dataframe. Separate lines into two columns. 
data = pd.read_csv('hw1_word_counts_05.txt', sep=" ", header=None, names=['Word', 'Count'])

# Compute prior probability and add as third column of dataframe for each word
data['P(W=w)'] = data['Count']*1.0/data['Count'].sum()

# Sort the dataframe by the word's prior probability (Second column of dataframe)
data_sorted = data.sort_values(by = ['P(W=w)'], ascending=False)

#print(data_sorted.head())

# Sanity check. 15 most frequent and 14 least frequent words. 
#print(data_sorted.head(15))
#print(data_sorted.tail(14))

### Problem 1b ###

# Initialize alphabet
#alphabet = np.ndarray(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

# Pick a random word to guess
#secret_word = random.choice(data['Word'])
#print(secret_word)

# Initialize the arrays for correct and incorrect guesses
correct_evidence = [None,None,None,None,None]
incorrect_evidence = ['A']

# Make words and prior probabilities from pandas dataframe to lists
words = data['Word'].tolist()
priors = data['P(W=w)'].to_numpy()
alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def isWordPossible(word, correct_evidence, incorrect_evidence):
    for i, letter in enumerate(word):
        if correct_evidence[i] == None:
            if letter in correct_evidence:
                return 0
        elif correct_evidence[i] != None:
            if letter != correct_evidence[i]:
                return 0
        elif letter in incorrect_evidence:
            return 0
    return 1

def posteriorDenominator(words, priors, correct_evidence, incorrect_evidence):
    denominator = 0.0
    for i, word in enumerate(words):
        denominator += isWordPossible(word,correct_evidence,incorrect_evidence)*priors[i]
    return denominator

def posteriorProbability(data, words, priors, correct_evidence, incorrect_evidence):
    posteriors = []
    denominator = posteriorDenominator(words, priors, correct_evidence, incorrect_evidence)
    for i, word in enumerate(words):
        nominator = isWordPossible(word,correct_evidence,incorrect_evidence)*priors[i]
        if not denominator:
            posteriors.append(0)
        else: 
            posteriors.append(nominator*1.0/denominator*1.0)
    data['Posterior'] = posteriors
    return data

def isLetterInWord(letter, word):
    for char in word:
        if letter == char:
            return 1
    return 0

def letterPredictiveProbability(data, letter, words, priors, correct_evidence, incorrect_evidence):
    post_data = posteriorProbability(data, words, priors, correct_evidence, incorrect_evidence)
    posteriors = post_data['Posterior'].to_numpy()
    letter_prob = 0.0
    for i, word in enumerate(words):
        letter_in_word = isLetterInWord(letter,word)
        letter_prob += letter_in_word*posteriors[i]
    if letter in correct_evidence or letter in incorrect_evidence:
        letter_prob = 0
    return letter_prob

def nextGuess(data, alphabet, words, priors, correct_evidence, incorrect_evidence):
    max_probability = 0.0
    max_letter = ''
    for i, alpha in enumerate(alphabet):
        temp_probability = letterPredictiveProbability(data,alpha,words,priors,correct_evidence,incorrect_evidence)
        if temp_probability > max_probability:
            max_probability = temp_probability
            max_letter = alphabet[i]
    print("Next letter: ", max_letter, " with probability: ", max_probability)



nextGuess(data,alphabet,words,priors,correct_evidence,incorrect_evidence)

