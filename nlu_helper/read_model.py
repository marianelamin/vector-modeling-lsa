import numpy as np
import pandas as pd
import re

from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlu_helper.utility import file2pickle, record_how_long_it_takes, show_message_log_console


class SentencePreProcessor:
    @staticmethod
    def configure_token_pattern(token_pattern=None) -> dict:
        # configuration of the doc by term matrix generation
        if token_pattern is None:
            token_pattern_ = {
                'token_pattern': r'(?u)(\b\w+\b|@\w+@)',  # accepts word and special @word@ as well
            }
        else:
            token_pattern_ = token_pattern
        return token_pattern_

    @staticmethod
    def pre_process_corpus(text):

        text = text.lower()

        # replaces any $33,333.34 for @price@ (including $ £ € )
        text = re.sub(r"[$£€]([,\d]+(\.)?\d+)", "@PRICE@", text)

        # replaces any 4 digits 1999 2001 2020 for @year@
        text = re.sub(r"[\d]{4}", "@YEAR@", text)

        # TODO: test this one, or for future studies
        # replaces any number of digits immediate followed by th as an @ordinal@
        # text = re.sub(r"[\d]{4}th", "@ORDINAL@", text)

        return text


class SentenceProcessor(SentencePreProcessor):
    #  Functions that process the sentences
    def __init__(self, model, weight_word_not_found=0.00001, logger=None):
        self.logger = logger
        self.args = model.args
        self.token_pattern = self.configure_token_pattern()
        self.lsa_model_pdf = model.as_pandas_data_frame()
        self.weight_word_not_found = weight_word_not_found

    @staticmethod
    def is_all_stop_words(st, stop_words):
        if stop_words is None:
            return False

        is_all_sw = True
        for w in st.split(" "):
            if w not in stop_words:
                is_all_sw = False
                # print(w, ' not a stop word')
        return is_all_sw

    def vector_of_a_word_not_found(self, value):
        return np.array([value for i in self.lsa_model_pdf.columns])

    def vector_of_an_existing_word(self, word):
        return self.lsa_model_pdf.loc[word].values

    def get_word_vector(self, word):
        """
        Searches the vector of a specified word, if the word does not exist, returns an array with the default
         word not found value, typically 0.00001
        :param word: a string containing he word to get the vector from
        :return: an numpy.array with the vector of the specified word
        """
        try:
            res = self.vector_of_an_existing_word(word)
        except KeyError:
            res = self.vector_of_a_word_not_found(self.weight_word_not_found)

            if self.logger is not None:
                show_message_log_console('\tNot found: ' + word, self.logger)

        return res

    def tokenize_sentence(self, sentence, pre_process_sentence=None) -> np.ndarray:
        """
        Creates an array that contains all the words that appear on the sentence, after being processed by the
        fit_transform
        The count vectorizer.fit_transform creates a document by term matrix with one document = the sentence
        """
        if pre_process_sentence is None:
            pre_process_method = self.pre_process_corpus
        else:
            pre_process_method = pre_process_sentence

        vectorizer = CountVectorizer(token_pattern=self.token_pattern['token_pattern'],
                                     preprocessor=pre_process_method,
                                     stop_words='english' if self.args['stopwords'] else None,
                                     )

        tokenized = list()
        # If there is at least one word that is not a stop word process down below
        if not self.is_all_stop_words(sentence, vectorizer.get_stop_words()) and len(sentence) != 0:
            bow = vectorizer.fit_transform([sentence])
            # print(bow)
            words = vectorizer.get_feature_names()
            # I need this print for testing purposes
            # print(pd.DataFrame(bow.toarray(), columns=words))

            i = 0
            for count in bow.data:
                # print(count)
                for ind in range(count.item()):
                    # print(words[i])
                    tokenized.append(words[i])
                i = i + 1

        if len(tokenized) == 0:
            print('tokenized is empty')
            tokenized.append('')

        return np.asarray(tokenized)

    def get_sentence_vector(self, tokenized_sentence):
        """
        Generates the vector associated with a new sentence, such vector sentence will be determined
        using the model loaded on this class
        :param tokenized_sentence: all the words that appear in the sentence, after being processed like the model.
        :return: a numpy.array with the average of all the words that exist on the sentence
        """
        sentence_matrix = list()

        for token in tokenized_sentence:
            word_vector = self.get_word_vector(token)
            sentence_matrix.append(word_vector)

            # if self.logger is not None:
            #     show_message_log_console(token + "\t" + str(word_vector), self.logger)

        sentence_matrix = np.asmatrix(sentence_matrix)

        return np.mean(sentence_matrix, axis=0)

    def get_sentence_match(self, real_answer: str, given_answer: str) -> float:
        # generate a list with the words that appear on the sentence
        g_tok = self.tokenize_sentence(given_answer)
        given_sentence_vector = self.get_sentence_vector(g_tok)

        r_tok = self.tokenize_sentence(real_answer)
        real_sentence_vector = self.get_sentence_vector(r_tok)
        return self.cos_sim(given_sentence_vector, real_sentence_vector)

    def cos_sim(self, given_sentence_vector, real_sentence_vector):
        # print('given_sentence_vector: ', type(given_sentence_vector), ' - ', given_sentence_vector)
        # print('real_sentence_vector: ', type(real_sentence_vector), ' - ', real_sentence_vector)
        return cosine_similarity(given_sentence_vector, real_sentence_vector).item()


class MatrixModel:
    """
    This class is in charged of reading from a file the model to compare two parragraphs.
    The model is saved using pickle, right after it is generated with the options saved in the json file.
    """

    def __init__(self, f_name, logger=None):
        """
        :param f_name: name of the file from which we want to load the data. In order to properly work this filename
        must have two extensions, .pkl and .json
        """
        self.filename = f_name
        self.logger = logger
        # loads from a pickle
        self.token_pattern = SentenceProcessor.configure_token_pattern()
        self.__matrix, self.__feature_words, self.args = self.__load_model()
        if issparse(self.__matrix):
            self.__matrix = self.__matrix.toarray()
        # print(self.__matrix)
        # self.token_pattern = None

    def __load_model(self) -> tuple:
        # return load_npz_model(self.filename)
        # return file2np(self.filename)
        if self.logger is None:
            return file2pickle(self.filename)
        else:
            return record_how_long_it_takes(method=lambda: file2pickle(self.filename),
                                            logger=self.logger,
                                            message='Loading the np.ndarray from a file (pickle.load()) . . .')

    def get_feature_words(self) -> list:
        return self.__feature_words

    def as_pandas_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.__matrix, index=self.__feature_words)

    def as_nd_array(self) -> np.ndarray:
        return self.__matrix
