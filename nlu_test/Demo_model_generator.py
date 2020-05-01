import sys, os
PROJECT_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from nlu_helper.utility import script_timer, reformat_execution_time
from nlu_helper.model_generator import generate_a_model


if __name__ == '__main__':
    tiempo = script_timer(
        mysetup='''
from nlu_helper.find_corpus import Corpus_all_senate_speeches, Corpus_ejemplo_clase, Corpus_movie_reviews
from nlu_helper.read_model import SentenceProcessor
from nlu_helper.utility import log_and_print, check_input, pickle2file, np2file, write2file, commenter
from nlu_helper.utility import reformat_execution_time, create_a_logger

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.extmath import randomized_svd

import numpy as np
import sys
import time
        ''',
        mycode=generate_a_model)
    print('The program took {} seconds'.format(tiempo))
    d, h, m, s = reformat_execution_time(tiempo)
    duration = 'The program took {} day,  {} hr,  {} min,  {} seconds to complete'.format(d, h, m, s)
    print(duration)
