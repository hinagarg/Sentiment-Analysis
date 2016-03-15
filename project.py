
# coding: utf-8

# # CS579: Project
# 
# In this project, we will fit a text classifier to categorize product(iPhone) reviews by sentiment.

# In[2]:

from collections import Counter
from collections import defaultdict
import glob
import math
import operator
import hashlib
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tarfile
from pprint import pprint
from urllib import urlretrieve


# In[3]:

# Here is the path to the data directory.
path = 'data_proj'
print('subdirectories are:' + str(os.listdir(path)))


# In[4]:

def get_files(path):
    """ Returns a list of file names in this directory that end in .txt """
    
    flist = []
    for dirpath, dirnames, files in os.walk(path):
        for f in files:
            flist.append(os.path.join(dirpath, f))
    return sorted(flist)


# In[5]:

print (path + os.sep + 'train' + os.sep + 'pos')
pos_train_files = get_files(path + os.sep + 'train' + os.sep + 'pos')
neg_train_files = get_files(path + os.sep + 'train' + os.sep + 'neg')
all_train_files = pos_train_files + neg_train_files

print('found %d positive and %d negative training files' %
      (len(pos_train_files), len(neg_train_files)))
print('first positive file: %s' % pos_train_files[0])
print('first negative file: %s' % neg_train_files[0])


# In[6]:

def get_true_labels(file_names):
    """Returns a *numpy array* of ints for the true sentiment labels of each file.
    1 means positive, 0 means negative. 
    """
    labels_arr = []
    for name in file_names:
        if 'pos' in name:
            labels_arr.extend([1])
        elif 'neg' in name:
            labels_arr.extend([0])
        else:
            labels_arr.extend([-1])
    return np.array(labels_arr)
labels = get_true_labels(all_train_files)
print('first 3 and last 3 labels are: %s' % str(labels[[1,2,3,-3,-2,-1]]))


# In[7]:

# Here's what a positive review looks like.
def file2string(filename):
    return io.open(filename, encoding='utf8').readlines()[0]
    
file2string(pos_train_files[10])


# In[8]:

def tokenize(text):
    """Given a string, returns a list of tokens such that: (1) all
    tokens are lowercase, (2) all punctuation is removed (3) all url's are removed (4) all mentions are removed (5) all rt(retweet) symbol is removed. 
    """
    rt_removed = re.sub('rt', '', text.lower())
    url_removed = re.sub('http\S+', '', rt_removed)
    tweet = re.sub('@\S+', '', url_removed)
    return re.findall(r'\w+', tweet)

tokenize("RT @rxcknjh: iPhone 4s awful won't selfie quality https://t.co/hlpogBA5hs")


# In[9]:

def term_frequecies(all_train_files):
    """Given a document, returns a list of terms with respective term frequencies 
    """
    c = Counter()
    for doc in all_train_files:
        text = file2string(doc)
        tokenized = set(tokenize(text))
        c.update(tokenized)
    return c.most_common()[:5]
    
pprint(term_frequecies(all_train_files))


# In[10]:

def do_vectorize(filenames, tokenizer_fn=tokenize, min_df=1,
                 max_df=1., binary=True, ngram_range=(1,1)):
    """
    Converts a list of filenames into a sparse csr_matrix, where
    each row is a file and each column represents a unique word.
    Returns:
        A tuple (X, vec), where X is the csr_matrix of feature vectors,
        and vec is the CountVectorizer object.
    """
    from scipy.sparse import lil_matrix
    from collections import defaultdict
   
    vec = CountVectorizer(input = 'filename', min_df= min_df,
        ngram_range= ngram_range, max_df = max_df , binary=binary, tokenizer = tokenizer_fn, dtype = int)
    matrix = vec.fit_transform(filenames) 
    return matrix, vec
    
matrix, vec = do_vectorize(all_train_files)
print ('matrix represents %d documents with %d features' % (matrix.shape[0], matrix.shape[1]))
print('first doc has terms:\n%s' % (str(sorted(matrix[0].nonzero()[1]))))


# In[11]:

# This block returns the shuffled data.
def repeatable_random(seed):
    hash = str(seed)
    while True:
        hash = hashlib.md5(hash).digest()
        for c in hash:
            yield ord(c)

def repeatable_shuffle(X, y, filenames):
    r = repeatable_random(42) 
    indices = sorted(range(X.shape[0]), key=lambda x: next(r))
    return X[indices], y[indices], np.array(filenames)[indices]

X, y, filenames = repeatable_shuffle(matrix, labels, all_train_files)

print('first shuffled document %s has label %d and terms: %s' % 
      (filenames[0], y[0], sorted(X[0].nonzero()[1])))


# In[12]:

# This creates a LogsticRegression object, which
# we will use in the do_cross_validation method below.
def get_clf():
    return LogisticRegression(random_state=42)


# In[13]:

def do_cross_validation(X, y, n_folds=4, verbose=False):
    """
    Performs n-fold cross validation, calling get_clf() to train n
    different classifiers. 
    Returns:
        the average testing accuracy across all folds.
    """
    """ Computes average cross-validation acccuracy."""
        
    cv = KFold(len(y), n_folds, shuffle=False)
    accuracies = []
    for train_idx, test_idx in cv:
        clf = get_clf()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    if verbose:
        for i in range(0, len(accuracies)):
            print ('fold ' + str(i) + ' accuracy=%.4f'% accuracies[i] )   
    avg = np.mean(accuracies)
    return avg
    
print('average cross validation accuracy=%.4f' %
      do_cross_validation(X, y, verbose=True))


# In[14]:

def do_expt(filenames, y, tokenizer_fn=tokenize,
            min_df=1, max_df=1., binary=True,
            ngram_range=(1,1), n_folds=5):
    """
    Runs one experiment, which consists of vectorizing each file,
    performing cross-validation, and returning the average accuracy.
    Returns:
        the average cross validation testing accuracy.
    """
    X , vec = do_vectorize(filenames, tokenizer_fn, min_df,
                 max_df, binary , ngram_range)
    return do_cross_validation(X, y, n_folds)


# In[ ]:

print('accuracy using default settings: %.4g' % do_expt(filenames, y))


# In[ ]:

def compare_n_folds(filenames, y):
    """
    Varying the setting of n_folds parameter in the do_expt 
    function to be in [2,5,10,20] and plotting the accuracies for each setting.
    Returns:
        a list of average testing accuracies, one per fold.
    """
    n_folds = [2,5,10,20]
    accuracies = []
    for setting in n_folds:
        mean = do_expt(filenames, y, n_folds = setting)
        accuracies.append(mean)
    plt.figure()
    plt.plot([n for n in n_folds], [n for n in accuracies], 'bo-')
    plt.xlabel('n_folds')
    plt.ylabel('accuracy')
    plt.show()
    return accuracies
compare_n_folds(filenames, y)


# In[ ]:

def compare_binary(filenames, y):
    """
    Calling the do_expt twice, once with binary=True, and once with binary=False.
    Returns the average accuracies for each. Using the default parameters for the
    remaining arguments in do_expt.
    Returns:
        a list of average testing accuracies. The first entry
        is for binary=True, the second is for binary=False.
    """
    avg_accuracy_1 = do_expt(filenames, y, binary=True)
    avg_accuracy_2 = do_expt(filenames, y, binary=False)
    return [avg_accuracy_1, avg_accuracy_2]      
compare_binary(filenames, y)


# In[ ]:

def tokenize_with_punct(text):
    """Given a string, returns a list of tokens such that: (1) all
    tokens are lowercase, (2) all punctuation is kept as separate tokens.
    Returns:
        a list of tokens
    """
    return re.findall("\w+|[^A-Za-z0-9\s\x85]", text.lower())

tokenize_with_punct("Really wish I could have the iPhone 6 :-((")


# In[ ]:

def tokenize_with_appostrophe(text):
    """Does the same thing as tokenize, with the following difference:
    whenever the terms 'don't, won't aren't, can't cudn't' i.e. with an apostrophe appears, keep them while tokenizing.
    Returns:
        a list of tokens
    """
    rt_removed = re.sub('rt', '', text.lower())
    url_removed = re.sub('http\S+', '', rt_removed)
    tweet = re.sub('@\S+', '', url_removed)
    return re.findall(r"\w{2,}'\w|\w{2,}", tweet)

tokenize_with_appostrophe("RT @rxcknjh: iPhone 4s is  bad and doesn't have a good selfie quality, don't buy it  https://t.co/hlpogBA5hs")


# In[ ]:

def tokenize_with_negative_words(text):
    """Does the same thing as tokenize_with_appostrophe, with the following difference:
    whenever the term like:'don't, not, won't aren't, can't cudn't' i.eappears, change the two subsequent tokens to have the prefix
    'don't_' prior to the token. 
    Returns:
        a list of tokens
    """
    i = 0
    list_of_tokens = tokenize_with_appostrophe(text)
    matched = re.findall(r"not|\w{3,}'t"," ".join(list_of_tokens))
    if matched:
            while i < len(list_of_tokens):
                token = list_of_tokens[i]
                if token in matched:
                    if (i + 1) < len(list_of_tokens):
                        list_of_tokens[i + 1] = (token + '_' + list_of_tokens[i + 1])
                    if (i + 2) < len(list_of_tokens):
                        list_of_tokens[i + 2] = (token +  '_' + list_of_tokens[i + 2])
                i += 1      
    return list_of_tokens
    
tokenize_with_negative_words("This cn't movie is couldn't good. In fact, it is can't even really a movie good")


# In[ ]:

def tokenizer_expt(all_train_files, y):
    """
    An experiment to see how does the tokenizer affect results for four different tokenizers as below:
    1- tokenize
    2- tokenize_with_punct
    3- tokenize_with_appostrophe
    4- tokenize_with_negative_words
    Returns the average cross-validation accuracy for each approach,
    in the above order. Uses the default parameters for all other 
    arguments to do_expt.
    Returns:
        a list of average testing accuracies for each tokenizer.
    """
    tokenizers_mean = []
    tokenizers = [tokenize , tokenize_with_punct, tokenize_with_appostrophe, tokenize_with_negative_words]
    for tokenizer in tokenizers:
        result = do_expt(all_train_files, y, tokenizer_fn=tokenizer)
        tokenizers_mean.append(result)
    return tokenizers_mean   

tokenizer_expt(filenames, y)


# In[ ]:

def min_df_expt(filenames, y):
    """
    Varies the setting of min_df parameter in the do_expt 
    function to be ints in the range (1,10) (inclusive). Plots the accuracies for each setting.
    Also returns the list of accuracies. Uses the default value for all
    other arguments to the do_expt function, except that the tokenizer
    should be tokenize_with_negative_words_generalized.
    Returns:
        a list of average testing accuracies, one per min_df value.
    """
    min_df_mean = []
    for i in range(1,11):
        accuracy = do_expt(filenames, y, min_df = i, tokenizer_fn = tokenize_with_negative_words)
        min_df_mean.append(accuracy)
    plt.figure()
    plt.plot([n for n in range(1,11)], [n for n in min_df_mean], 'bo-')
    plt.xlabel('min_df')
    plt.ylabel('accuracy')
    plt.show()
    return min_df_mean

min_df_expt(filenames, y)


# In[ ]:

def max_df_expt(filenames, y):
    """
    Varies the setting of max_df parameter in the do_expt 
    function to be one of [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.].
    Plots the accuracies for each setting. Also returns the list of accuracies.
    Uses the default value for all other arguments to the do_expt function,
    except that the tokenizer=tokenize_with_negative_words_generalized.
    Returns:
        a list of average testing accuracies, one per max_df value.
    """
    max_df = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    max_df_mean = []
    for i in max_df:
        accuracy = do_expt(filenames, y, min_df = 1, max_df = i, tokenizer_fn = tokenize_with_negative_words)
        max_df_mean.append(accuracy)
    plt.figure()
    plt.plot([n for n in max_df], [n for n in max_df_mean], 'bo-')
    plt.xlabel('max_df')
    plt.ylabel('accuracy')
    plt.show()
    return max_df_mean
    
max_df_expt(filenames, y)


# In[ ]:

def n_grams_expt(filenames, y):
    """
    Vary the setting of n_grams parameter in the do_expt 
    function to be one of [(1,1), (1,2),(1,3),(1,4), (1,5), (1,6)].
    For each setting, call do_expt and store the resulting accuracy.
    Plot the accuracies for each setting. Also return the list of accuracies.
    Use the default value for all other arguments to the do_expt function,
    except that the tokenizer=tokenize_with_appostrophy and min_df=1.
    Params:
        filenames....list of training file names
        y............true labels for each file (a numpy array)
    Returns:
        a list of average testing accuracies, one per max_df value.
    """
    ngram_range=(1,1)
    ngram_range = [(1,1), (1,2) , (1,3),  (1,4),  (1,5),  (1,6)]
    ngram_range_mean = []
    for i in ngram_range:
        accuracy = do_expt(filenames, y, ngram_range = i, min_df = 1, max_df = 1., tokenizer_fn = tokenize_with_negative_words)
        ngram_range_mean.append(accuracy)
    plt.figure()
    plt.plot([n for n in ngram_range], [n for n in ngram_range_mean], 'bo-')
    plt.xlabel('ngram_range')
    plt.ylabel('accuracy')
    plt.show()
    return ngram_range_mean
    
n_grams_expt(filenames, y)


# ## Inspecting coefficients
# 
# Next we'll look at the coefficients learned by LogisticRegression to 
# determine how it is making its classification decisions.

# In[ ]:

# First, we'll train our final classifier using our best settings.
X, vec = do_vectorize(filenames, tokenizer_fn=tokenize_with_negative_words,
                     binary=True, min_df=2, max_df=.2)
clf = get_clf()
clf.fit(X, y)


# In[ ]:

print type(clf.coef_)
print clf.coef_.shape


# In[ ]:

# Here are the first 10 coefficients.
print(clf.coef_[0][:10])
# The features corresponding to them can be found using the vectorizer's get_feature_names method.
print(vec.get_feature_names()[:10])


# In[ ]:

def get_top_coefficients(clf, vec, n=10):
    """ Gets the top n coefficients for each class (positive/negative).
    Returns:
        Two lists of tuples. The first list containts the top terms for the positive
        class. Each entry is a tuple of (string, float) pairs, where
        string is the feature name and float is the coefficient.
        The second list is the same but for the negative class.
        In each list, entries are sorted in descending order of 
        absolute value."""
    
    pos_list = []
    neg_list = []
    
    coef = clf.coef_[0]
    srted = np.argsort(coef)
    topi = srted[::-1][:n]
    boti = srted[:n]
    terms = vec.get_feature_names()
    for n in topi:
        pos_list.append((terms[n], coef[n]))
    for i in boti:
        neg_list.append((terms[i], coef[i])) 
    return pos_list, neg_list
       
pos_coef, neg_coef = get_top_coefficients(clf, vec, n=5)
print('top positive coefs: %s' % str(pos_coef))
print('top negative coefs: %s' % str(neg_coef))


# ## Read testing data
# 
# Next, we'll read in the testing files (in the `test/` subdirectory) and compute our accuracy.

# In[ ]:

pos_test_files = get_files(path + os.sep + 'test' + os.sep + 'pos')
neg_test_files = get_files(path + os.sep + 'test' + os.sep + 'neg')
all_test_files = pos_test_files + neg_test_files
# Note that we call .transform, not .fit_transform, since we 
# don't want to learn a new vocabulary.
X_test = vec.transform(all_test_files)
y_test = np.array([1] * len(pos_test_files) + [0] * len(neg_test_files))
print('X_test represents %d documents with %d features' % (X_test.shape[0], X_test.shape[1]))
print('y_test has %d positive and %d negative labels' % (len(np.where(y_test==1)[0]),
                                                          len(np.where(y_test==0)[0])))
print('first testing file is %s' % all_test_files[0])
print('last testing file is %s' % all_test_files[-1])
print('testing accuracy=%.4g' % accuracy_score(y_test, clf.predict(X_test)))


# In[ ]:

def index_of_term(vec, term):
    """ This returns the column index corresponding to this term."""
    return vec.get_feature_names().index(term)

index_of_term(vec, 'got')


# In[ ]:

def train_after_removing_features(X, y, vec, features_to_remove):
    """
    Sets to 0 the columns of X corresponding to the terms in features_to_remove. 
    Then, trains a new classifier on X and y and return the result.
    Returns:
       The classifier fit on the modified X data.
    """
    M_X = X.copy()
    for term in features_to_remove:
        col_ind = index_of_term(vec, term)
        for row in range(0,len(y)):
            M_X[row , col_ind] = 0
            M_X[ row ,col_ind]
        clf = get_clf()
        clf.fit(M_X, y)
        return clf
    
clf = train_after_removing_features(X.copy(), y, vec, ['got'])
print('testing accuracy=%.5g' % accuracy_score(y_test, clf.predict(X_test)))


# In[ ]:

def get_top_errors(X_test, y_test, filenames, clf, n=10):
    """
    Uses clf to predict the labels of the testing data in X_test. 
    We want to find incorrectly predicted documents. Furthermore, we want to look at those 
    where the probability of the incorrect label, according to the classifier, is highest.
    Uses the .predict_proba method of the classifier to get the probabilities of
    each class label. Returns the n documents that were misclassified, sorted by the
    probability of the incorrect label. The returned value is a list of dicts, defined below.
    Returns:
        A list of n dicts containing the following key/value pairs:
           index: the index of this document (in the filenames array)
           probas: a numpy array containing the probability of class 0 and 1
           truth: the true label
           predicted: the predicted label
           filename: the path to the file for this document
    """
    predicted = clf.predict(X_test)
    error_result = []
    probabilities = clf.predict_proba(X_test)
    for label in range(0, len(predicted)):
        if predicted[label] != y_test[label]:
            dict_res = {}
            dict_res['index'] = label
            dict_res['probas'] = probabilities[label]
            dict_res['truth'] =  y_test[label]
            dict_res['predicted'] =  predicted[label]
            dict_res['filename'] =  filenames[label]
            error_result.append(dict_res)
    result_sorted = sorted(error_result, key=lambda item:item['probas'][0] * item['probas'][1])
    return (result_sorted)[:10]
errors = get_top_errors(X_test, y_test, all_test_files, clf)
errors


# In[ ]:

# Given a document, finds the term in it that is most strongly associated
# with a given class label, according to a trained classifier.
def most_predictive_term_in_doc(instance, clf, class_idx):
    """
    Params:
        instance....one row in the X csr_matrix, corresponding to a document.
        clf.........a trained LogisticRegression classifier
        class_idx...0 or 1. The class for which we should find the most 
                    predictive term in this document.
    Returns:
        The index corresponding to the term that appears in this instance
        and has the highest coefficient for class class_idx.
    """
    coef = clf.coef_[0]    
    indices = sorted(instance.indices, key=lambda x: coef[x])
    terms = vec.get_feature_names()
    if class_idx == 0:
        i = indices[0]
    else:
        i = indices[len(indices)-1]
    if coef[i] != 1:
        return i
            
neg_idx = most_predictive_term_in_doc(X_test[4], clf, 0)
pos_idx = most_predictive_term_in_doc(X_test[4], clf, 1)
print('for document %s, the term most predictive of class 0 is %s (index=%d)' %
      (all_test_files[7], vec.get_feature_names()[neg_idx], neg_idx))
print('for document %s, the term most predictive of class 1 is %s (index=%d)' %
      (all_test_files[7], vec.get_feature_names()[pos_idx], pos_idx))


# ## Implementing Naive Bayes
# 
# Next, we'll implement Naive Bayes and compute accuracy.

# In[ ]:

class Document(object):
    """ 
    The instance variables are:
    filename....The path of the file for this document.
    label.......The true class label ('neg' or 'pos'), determined by whether the filename contains the string 'neg'
    tokens......A list of token strings.
    """

    def __init__(self, filename, tokenizer_fn):
        self.filename = filename
        self.label = 'neg' if 'neg' in filename else 'pos'
        text = file2string(filename)
        self.tokenize_fn = tokenizer_fn(text)
        self.tokens = self.tokenize_fn


# In[ ]:

class NaiveBayes(object):

    def ExtractCalProbRequiredItems(self, documents):
        """
        Given a list of labeled Document objects,computes the required parameters for calculating word conditional probabilities
        """
        self.V = defaultdict(lambda : defaultdict(lambda : 0))
        self.sumTct = defaultdict(lambda : 0)
        self.C = defaultdict(lambda : 0)
        label_list = ["neg", "pos"]	
        for d in documents:
            self.C[d.label] += 1
            self.sumTct[d.label] += len(d.tokens)
            for token in d.tokens:
                if token not in self.V:
                    for label in label_list:
                        self.V[token][label] = 0
                self.V[token][d.label] +=  1
 
    def train(self, documents):
        """
        Given a list of labeled Document objects, computes the class priors and
        word conditional probabilities.
        """
        self.ExtractCalProbRequiredItems(documents)
        self.prior =  defaultdict(lambda: 0)
        self.condProb =  defaultdict(lambda: defaultdict(lambda: 0))
        self.N = len(documents)
        for c in self.C:
            self.prior[c] = self.C[c]/float(self.N)
        for t in self.V.keys():
            for label in self.V[t].keys():
                self.Tct = self.V[t][label]
                self.Tct_sum =  self.sumTct[label]
                self.condProb[t][label] = (self.Tct + 1)/ (float(self.Tct_sum) + len(self.V))

    def classify(self, documents):
        """
        Returns a list of strings, either 'neg' or 'pos', for each document.
        documents....A list of Document objects to be classified.
        """
        labels_flist = []
        for d in documents:
            score = defaultdict(lambda: 0)
            for c in self.C.keys():
                score[c] = score[c] +  math.log10(float(self.prior[c]))
                for t in d.tokens:
                    if t in self.V:
                        score[c] += math.log10(float(self.condProb[t][c]))
            labels_flist.append(max(score.iteritems(), key=operator.itemgetter(1))[0])

        return labels_flist


# In[ ]:

def evaluate(predictions, documents):
    """
    Evaluates the accuracy of a set of predictions.
    Prints the following:
    accuracy=xxx, yyy false neg, zzz false pos
    where
    xxx = percent of documents classified correctly
    yyy = number of pos documents incorrectly classified as neg
    zzz = number of neg documents incorrectly classified as pos
    
    """
    false_pos = 0
    false_neg = 0
    classified_correctly = 0
    for i in range(0,len(documents)):
        if documents[i].label == predictions[i]:
            classified_correctly += 1
        else:
            if predictions[i] == 'pos':
                false_pos += 1
            else:
                false_neg += 1
    print 'total errors ' + str(false_pos + false_neg)            
    print 'accuracy =' + str(classified_correctly/float(len(documents))) + ', ' + str(false_neg) + ' false neg' + ', ' + str(false_pos) + ' false pos' 


# In[ ]:

def main(tokenizer_fn = tokenize):
    
    pos_train_files = glob.glob(path + os.sep + 'train' + os.sep + 'pos' + os.sep + '*.txt')
    neg_train_files = glob.glob(path + os.sep + 'train' + os.sep + 'neg' + os.sep + '*.txt')
    all_train_files = pos_train_files + neg_train_files
  
    pos_test_files = glob.glob(path + os.sep + 'test' + os.sep + 'pos' + os.sep + '*.txt')
    neg_test_files = glob.glob(path + os.sep + 'test' + os.sep + 'neg' + os.sep + '*.txt')
    all_test_files = pos_test_files + neg_test_files
    
    train_docs = [Document(f, tokenizer_fn) for f in all_train_files]
    print 'read', len(train_docs), 'training documents.'
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(f, tokenizer_fn) for f in all_test_files]
    print 'read', len(test_docs), 'testing documents.'
    predictions = nb.classify(test_docs)
    evaluate(predictions, test_docs)

if __name__ == '__main__':
    main()


# In[ ]:

def naive_bayes_tokenizer_expt(all_train_files, y):
    """
    An experiment to see how does the tokenizer affect results for three different tokenizers as below:
    1- tokenize
    2- tokenize_with_punct
    3- tokenize_with_appostrophe
    4. tokenize_with_negative_words
    Returns the average cross-validation accuracy for each approach,
    in the above order. Uses the default parameters for all other 
    arguments to do_expt.
    Returns:
        a list of average testing accuracies for each tokenizer.
    """
    tokenizers_mean = []
    i = 1
    tokenizers = [tokenize , tokenize_with_punct, tokenize_with_appostrophe, tokenize_with_negative_words]
    for tokenizer in tokenizers:
        print "Results of expt: " + str(i)
        result = main(tokenizer_fn=tokenizer)
        print 
        i += 1
naive_bayes_tokenizer_expt(filenames, y)


# In[ ]:




# In[ ]:



