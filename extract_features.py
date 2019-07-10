#! /usr/local/bin/ python3.7
import os
import re

from nltk.stem import PorterStemmer
import numpy as np
from sklearn.svm import SVC

DATA_DIR = './data/'

# Load vocabulary
vocabulary = []
with open('vocab.txt', 'r') as f:
    lines = f.readlines()
    vocabulary = [line[:-1] for line in lines]


def process_email(email_contents):
    """ preprocesses a the body of an email and returns a list of word_indices """
    global vocabulary

    # ----- Preprocess email -----

    # Lower case
    email_contents = email_contents.lower()

    # Remove all HTML
    html = re.compile(r'<[^<>]+>')
    email_contents = html.sub('', email_contents)

    # Handle numbers
    numbers = re.compile(r'[0-9]+')
    email_contents = numbers.sub('number', email_contents)

    # Handle URLs
    urls = re.compile(r'(http|https)://[^\s]*')
    email_contents = urls.sub('httpaddr', email_contents)

    # Email
    email_addresses = re.compile(r'[^\s]+@[^\s]+')
    email_contents = email_addresses.sub('emailaddr', email_contents)

    # Dollar sign
    dollar_sign = re.compile(r'[$]+')
    email_contents = dollar_sign.sub('dollar', email_contents)

    # ----- Tokenize email -----
    tokens = []
    words = re.split(r"\s|\@|\$|\/|\#|\.|\-|\:|\&|\*|\+|\=|\[|\]|\?|\!|\(|\)|\{|\}|\,|\'|\'|\"|\>|\_|\<|\;|\%", email_contents)

    stemmer = PorterStemmer()

    for word in words:
        # Remove nonalphanumeric characters
        alphanumeric = re.compile(r'[^a-zA-Z0-9]')
        word = alphanumeric.sub('', word)

        # Stem the word
        word = stemmer.stem(word)

        # Get index if it exists
        if word in vocabulary:
            tokens.append(vocabulary.index(word))

    return tokens


def extract_features(tokens):
    global vocabulary
    num_words = len(vocabulary)
    features = np.zeros(num_words)
    features[tokens] = 1
    return features


def extract_features_for_emails_in_directory(path, num_items):
    features = np.zeros((num_items, 1899))
    i = 0

    for email in os.listdir(path):
        with open(os.path.join(path, email), 'r', encoding='unicode_escape') as email:
            try:
                email_contents = email.read()
            except UnicodeDecodeError:
                continue
            tokens = process_email(email_contents)
            email_features = extract_features(tokens)
            features[i, :] = email_features

            print('Processed {} {}'.format(i, email.name))
            i += 1
    
    return features


if __name__ == '__main__':
    num_non_spam = 2551
    num_spam = 501

    # Create data for non spam emails
    print('Creating data for non spam emails')
    non_spam_path = os.path.join(DATA_DIR, 'nonspam')
    non_spam_features = extract_features_for_emails_in_directory(non_spam_path, num_non_spam)

    # Create data for spam emails
    print('Creating data for spam emails')
    spam_path = os.path.join(DATA_DIR, 'spam')
    spam_features = extract_features_for_emails_in_directory(spam_path, num_spam)

    # Add labels
    print('Adding labels to data')
    non_spam_labels = np.zeros((num_non_spam, 1))
    non_spam_data = np.concatenate((non_spam_features, non_spam_labels), axis=1)
    spam_labels = np.ones((num_spam, 1))
    spam_data = np.concatenate((spam_features, spam_labels), axis=1)

    # Concatenate both data arrays
    data = np.concatenate((non_spam_data, spam_data))

    # Save data to disk
    print('Saving data')
    outfile = 'data/features.npz'
    np.savez(outfile, data=data)

