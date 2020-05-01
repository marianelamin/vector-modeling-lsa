import os
from bs4 import BeautifulSoup
import time as time_module


from nltk.corpus import gutenberg
from nltk.corpus import sentence_polarity
from nltk.corpus import masc_tagged
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import europarl_raw
from nltk.corpus import state_union
from nltk.corpus import movie_reviews
from nltk.corpus import twitter_samples

# Select the correct corpora to load, looks at datasets from nltk.corpora
#  source: http://www.nltk.org/nltk_data/
#  with numbers:
#  23: id=movie_reviews
#  27: id=sentence_polarity,
#  26: id=masc_tagged,
#  28: id=webtext,
#  29: id=nps_chat,
#  31: id=europarl_raw,
#  40: id=state_union,
#  41: id=twitter_samples


class Corpus_movie_reviews:
    def __init__(self):
        self.number_id = 23
        self.source_id = "movie_reviews"
        self.titles = [name for name in movie_reviews.fileids()]
        self.data = [movie_reviews.raw(name) for name in self.titles]


class Corpus_sentence_polarity:
    def __init__(self):
        self.number_id = 27
        self.source_id = "sentence_polarity"
        self.titles = [name for name in sentence_polarity.fileids()]
        self.data = [sentence_polarity.raw(name) for name in self.titles]


class Corpus_masc_tagged:
    def __init__(self):
        self.number_id = 26
        self.source_id = "masc_tagged"
        self.titles = [name for name in masc_tagged.fileids()]
        self.data = [masc_tagged.raw(name) for name in self.titles]


class Corpus_webtext:
    def __init__(self):
        self.number_id = 28
        self.source_id = "webtext"
        self.titles = [name for name in webtext.fileids()]
        self.data = [webtext.raw(name) for name in self.titles]


class Corpus_nps_chat:
    def __init__(self):
        self.number_id = 29
        self.source_id = "nps_chat"
        self.titles = [name for name in nps_chat.fileids()]
        self.data = [nps_chat.raw(name) for name in self.titles]


class Corpus_europarl_raw:
    def __init__(self):
        self.number_id = 31
        self.source_id = "europarl_raw"
        self.titles = [name for name in europarl_raw.fileids()]
        self.data = [europarl_raw.raw(name) for name in self.titles]


class Corpus_state_union:
    def __init__(self):
        self.number_id = 40
        self.source_id = "state_union"
        self.titles = [name for name in state_union.fileids()]
        self.data = [state_union.raw(name) for name in self.titles]


class Corpus_twitter_samples:
    def __init__(self):
        self.number_id = 41
        self.source_id = "twitter_samples"
        self.titles = [name for name in twitter_samples.fileids()]
        self.data = [twitter_samples.raw(name) for name in self.titles]


class Corpus_ejemplo_clase:
    def __init__(self):
        self.number_id = -1
        self.source_id = "ejemplo_clase"
        self.data = ['The cow jumped over the moon',
                     "O'Leary's cow kicked the lamp",
                     'The kicked lamp started a fire',
                     'The cow on fire'
                     ]
        self.titles = [name for name in range(len(self.data))]


# class Corpus_paper:
#     def __init__(self):
#         self.number_id = -2
#         self.source_id = "paper"
#         self.data = ['Human machine interface for ABC computer applications',
#                      'A survey of user opinion of computer system response time',
#                      'The EPS user interface management system',
#                      'System and human system engineering testing of EPS',
#                      'Relation of user perceived response time to error measurement',
#                      'The generation of random, binary, ordered trees',
#                      'The intersection graph of paths in trees',
#                      'Graph minors IV: Widths of trees and well-quasi-ordering',
#                      'Graph minors IV: A survey']
#         self.titles = [name for name in range(len(self.data))]


class Corpus_all_senate_speeches:
    def __init__(self, divide_by=None, num_of_docs=None):
        self.number_id = -3
        self.source_id = "all_senate_speeches"
        filepath = os.path.join(os.path.dirname(__file__),
                                'corpora',
                                'all-senate-speeches.txt')

        # This is the entire corpus
        t_1 = time_module.time()
        entire_corpus = self.parse_senate_speeches(filepath)
        time2 = "%s s to load the entire corpus of %d documents" % (time_module.time() - t_1, len(entire_corpus))
        print(time2)

        if divide_by is None and num_of_docs is None:
            self.data = entire_corpus
        else:
            self.data = self.manipulate_corpus(entire_corpus, div=divide_by, n_docs=num_of_docs)

        # Create an array with the names of the different documents that were read
        # this instance variable might be useless
        self.titles = ["doc"+str(name) for name in range(len(self.data))]

        print('init corpus done!')

    def parse_senate_speeches(self, file_path) -> list:
        texts = list()
        with open(file_path) as fp:
            soup = BeautifulSoup(fp, 'xml')
            text_tag = soup.find_all('TEXT')
            for text in text_tag:
                texts.append(text.string)
                # print('\n****', text.string)

        return texts

    def manipulate_corpus(self, entire_corpus, div=None, n_docs=None):
        # Determine how much of the corpus will actually be selected to train our LSA model

        if div is not None:
            corpus_to_load = (len(entire_corpus) - 1) if (div is 1) else int(len(entire_corpus)/div)
            print('Getting {} documents as the corpus. div by {}'.format(corpus_to_load, div))
            # Select only a portion of the entire corpus, due to memory limitation of the current machine
            return entire_corpus[0:corpus_to_load]

        elif n_docs is not None:
            max_count = round(len(entire_corpus) / n_docs)
            new_corpus = list()
            text = ''
            count = 0
            for doc in entire_corpus:
                # print(doc)
                count = count + 1
                text = text + doc + "\n"
                if count == max_count:
                    new_corpus.append(text)
                    count = 0
                    text = ''
            new_corpus.append(text)
            return new_corpus

        else:
            return entire_corpus


if __name__ == '__main__':
    # cor = Corpus_all_senate_speeches(1)
    # print(cor.source_id, ' is ', len(cor.data), " elements long")

    print('\n\n*********************************\n\n')
    # print(cor.data[0:10])
    # for text in cor.data[0:5]:
    #     print("---->\n", text)
    #     print("---->\n", type(text))
    #     print("---->\n", text.string)