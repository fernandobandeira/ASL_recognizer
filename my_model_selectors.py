import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_components, minBIC = None, None
        for components in range(self.min_n_components, self.max_n_components+1):
            try:                
                logL = self.base_model(components).score(self.X, self.lengths)
                p = (components**2 - components) + (2*len(self.X[0]) * components)
                logN = np.log(len(self.X))
                BIC = -2 * logL + p * logN
                if minBIC is None or minBIC > BIC:
                    minBIC, best_num_components = BIC, components
            except:
                pass
        if best_num_components is None:
            best_num_components = self.n_constant
        return self.base_model(best_num_components)



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_components, maxDIC = None, None
        for components in range(self.min_n_components, self.max_n_components+1):
            try:
                logL = self.base_model(components).score(self.X, self.lengths)
                sumLogL = 0
                words = list(self.words.keys())
                M = len(words)
                words.remove(self.this_word)
                for word in words:
                    try:
                        modelSelector = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose)
                        sumLogL += modelSelector.base_model(components).score(modelSelector.X, modelSelector.lengths)
                    except:
                        M -= 1
                DIC = logL - sumLogL / (M-1)
                if maxDIC is None or maxDIC < DIC:
                    maxDIC, best_num_components = DIC, components
            except:
                pass
        if best_num_components is None:
            best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_components, maxAvgLogL = None, None
        for components in range(self.min_n_components, self.max_n_components+1):
            try:
                sumLogL = 0
                countLogL = 0
                split = KFold(5).split
                for train, test in split(self.sequences):
                    X, lengths = combine_sequences(train, self.sequences)
                    try:
                        sumLogL += self.base_model(components).score(X, lengths)
                        countLogL += 1
                    except:
                        pass
                if countLogL > 0:
                    avgLogL = sumLogL / countLogL
                    if maxAvgLogL is None or maxAvgLogL < avgLogL:
                        maxAvgLogL, best_num_components = avgLogL, components
            except:
                pass
        if best_num_components is None:
            best_num_components = self.n_constant
        return self.base_model(best_num_components)
