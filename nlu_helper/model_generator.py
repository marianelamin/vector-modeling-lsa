from nlu_helper.find_corpus import Corpus_all_senate_speeches, Corpus_ejemplo_clase, Corpus_movie_reviews
from nlu_helper.read_model import SentenceProcessor
from nlu_helper.utility import log_and_print, check_input, pickle2file, np2file, write2file, commenter
from nlu_helper.utility import reformat_execution_time, create_a_logger
from nlu_helper.utility import resources_folder, create_results_directory_if_needed

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.extmath import randomized_svd

import numpy as np
import sys
import os
import time


class ModelGenerator:
    """
        This class is in charge of generating a file with a term by document matrix, depending on the input.
    usage: python filename.py outputfile [options ]
    options are:
                --action= [lsa, docbyterm] latent semantic analysis, term by document matrix
                -tfidf term frequency invert document frequency matrix,
                -s stop words
                -o=<filename> if no specified it will generate a new one with all the options selected

    By default if there is no outputfile, one will be created, no stop words will be taken into account
    the simplest document by term matrix will be computed (bag of words) by counting.

    outpufFile will be stored in the resources folder.
    """

    def __init__(self, corpus_a, args):
        self.filename = args['output']
        self.extension = None
        self.corpus = corpus_a.data
        self.corpus_id = corpus_a.number_id
        self.corpus_source = corpus_a.source_id
        create_results_directory_if_needed(resources_folder(self.corpus_source))

        self.options = args
        self.feature_words = None
        self.logger = None

        self.token_pattern = SentenceProcessor.configure_token_pattern()
        self.ignored_words = None

    @staticmethod
    def pre_process_corpus(text):
        return SentenceProcessor.pre_process_corpus(text)

    def build_filename(self):
        """
        Depending on the options input by the user, this method returns the name of the file that will store the model
        :return: filename where the model output will be stored
        """
        if self.options['output'] is None:
            filename = (self.corpus_source if self.corpus_source is not None else '') \
                       + ('-min_df_' + str(self.options['min_document_frequency'])) \
                       + '-' + str(self.options['action']) \
                       + (('-dim' + str(self.options['svd_components'])) if (self.options['action'] == 'lsa') else '') \
                       + ('-' + str(self.options['score'])) \
                       + ('-stop' if self.options['stopwords'] else '')
        else:
            if '.' in self.options['output']:
                self.filename, self.extension = self.options['output'].split(".")
            # print('after splited: ', self.filename, self.extension)
            filename = self.options['output']
        return os.path.join(self.corpus_source, filename)

    def execute_action(self):
        """
        executes the corresponding method depending on the script parameters
        :return: a string with the information to be written on the output file
        """
        action = self.options['action']
        data_to_save = None

        if action == 'lsa':
            data_to_save = self.compute_lsa()
        elif action == 'docbyterm':
            csr_matrix = self.doc_by_term_matrix(pre_process_method=self.pre_process_corpus)
            data_to_save = csr_matrix  # .toarray()
            # truncate force model to have truncate_matrix columns
            # if self.options['truncate_matrix'] is not None:
            #     data_to_save = data_to_save[:, 0:self.options['truncate_matrix']]

        if data_to_save is not None:
            self.logger.debug('starting to save files . . .')
            self.save_files_report_times(data_to_save)
        else:
            self.logger.debug('action in input might have been wrong . . . There is no data to be saved.')

    def save_files_report_times(self, data_ndarray):
        print(data_ndarray.shape)
        size = "{}x{}".format(data_ndarray.shape[0], data_ndarray.shape[1])

        # save the matrix two ways and record how long it takes to do it
        m = 'Saving the ndarray using pickle . . . Saving json information to a file . . .'
        self.logger.info(m)
        print(m)
        start_time1 = time.time()
        pickle2file(self.build_filename() + "_" + size, data_ndarray,
                    self.feature_words, len(self.corpus), self.options, {
                        "token_pattern": self.token_pattern,
                        "ignored_words": self.ignored_words
                    })
        m = "took %s seconds to store data using pickle" % (time.time() - start_time1)
        self.logger.info(m)

        # message = 'Saving the np.ndarray using np.savez_compressed . . . Saving json information to a file . . .'
        # self.logger.info(message)
        # start_time2b = time.time()
        # print(message)
        # np2file("c_np2file_" + size + "_" + self.build_filename(), data_ndarray, self.feature_words, self.cols, self.options)
        # start_time3 = time.time()
        # time2 = ". . . %s seconds for (np2file np.savez_compressed)" % (start_time3 - start_time2b)
        # self.logger.info(time2)

    def doc_by_term_matrix(self, pre_process_method=None):
        """
        Generate document by term matrix based on the script parameters
        bag of words by counting or tfidf, with and without stopwords
        :return: a list of 2 elements, [vector of the doc by term matrix , its dataframe representation]
        """

        t_1 = time.time()
        if self.options['score'] == 'zeroone':
            # we do bag of words by placing zero if word does not appear, 1 if it does.
            if self.options['stopwords'] is False:
                # token_pattern=u'(?u)\b\w+\b'
                vectorizer = CountVectorizer(token_pattern=self.token_pattern['token_pattern'],
                                             min_df=self.options['min_document_frequency'],
                                             preprocessor=pre_process_method,
                                             dtype=np.float32,
                                             binary=True,
                                             )
            else:
                vectorizer = CountVectorizer(token_pattern=self.token_pattern['token_pattern'],
                                             min_df=self.options['min_document_frequency'],
                                             preprocessor=pre_process_method,
                                             dtype=np.float32,
                                             binary=True,
                                             stop_words='english',
                                             )
        elif self.options['score'] == 'tfidf':
            # we do doc by term using tfidf
            if self.options['stopwords'] is False:
                vectorizer = TfidfVectorizer(token_pattern=self.token_pattern['token_pattern'],
                                             min_df=self.options['min_document_frequency'],
                                             preprocessor=pre_process_method,
                                             dtype=np.float32,
                                             )
            else:
                vectorizer = TfidfVectorizer(token_pattern=self.token_pattern['token_pattern'],
                                             min_df=self.options['min_document_frequency'],
                                             preprocessor=pre_process_method,
                                             dtype=np.float32,
                                             stop_words='english',
                                             )
        else:
            # we do bag of words by counting
            if self.options['stopwords'] is False:
                # token_pattern=u'(?u)\b\w+\b'
                vectorizer = CountVectorizer(token_pattern=self.token_pattern['token_pattern'],
                                             min_df=self.options['min_document_frequency'],
                                             preprocessor=pre_process_method,
                                             dtype=np.float32,
                                             )
            else:
                vectorizer = CountVectorizer(token_pattern=self.token_pattern['token_pattern'],
                                             min_df=self.options['min_document_frequency'],
                                             preprocessor=pre_process_method,
                                             dtype=np.float32,
                                             stop_words='english',
                                             )

        bag_of_words = vectorizer.fit_transform(self.corpus)
        self.feature_words = vectorizer.get_feature_names()
        # words that were ignored because of their document frequency
        self.ignored_words = sorted(vectorizer.stop_words_)

        if bag_of_words.get_shape()[0] <= 100:
            self.options['truncate_matrix'] = None
            m = '\tNumber of documents is {}, hence Truncating must be None'.format(bag_of_words.get_shape()[0])
            self.logger.info(m)

        # commenter('term by doc matrix with options', lambda: print(bag_of_words.T))
        self.logger.info('term by doc matrix took {} seconds'.format(time.time() - t_1))
        return bag_of_words.T

    def compute_lsa(self):
        term_by_doc = self.doc_by_term_matrix(pre_process_method=self.pre_process_corpus)
        (n_words, n_docs) = term_by_doc.get_shape()
        # if n_docs exceeds 100, set the svd_components variable to 100 otherwise, set it to n_docs
        svd_components = (self.options['svd_components'] + 1) if n_docs > self.options['svd_components'] else n_docs

        # Computing with SVD
        for svd_method in ['TruncatedSVD']:  # , 'randomized_svd']:  # , 'svd_numpy']:
            start_time1 = time.time()
            print('Computing with ' + svd_method)
            if svd_method is 'TruncatedSVD':
                lsa = self.SVD_scikilearn(term_by_doc, svd_components - 1)
            # elif svd_method is 'randomized_svd':
            #     lsa = self.svd_randomized(term_by_doc, svd_components - 1)
            # elif svd_method is 'svd_numpy':
            #     lsa = self.singular_value_decomposition_numpy(term_by_doc, svd_components - 1)

            time1 = "\tSVD %s took %s seconds" % (svd_method, time.time() - start_time1)
            m = '\tlsa matrix calculation {} finished.'.format(svd_method)
            log_and_print(m, self.logger)
            log_and_print(time1, self.logger)

            log_and_print("\tsize {}".format(lsa.shape), self.logger)
            # self.logger.debug('starting to save files . . .')
            # self.save_files_report_times(lsa)

        return lsa

    def SVD_scikilearn(self, term_by_doc_matrix, svd_components):
        """
        Computing SVD with sci kit learn
        :param term_by_doc_matrix: a sparse matrix
        :param df: a data frame with the sparse matrix, indices and columns
        :param svd_components: number of components to select for the SVD
        :return: return lsa as ndarray, dataframe with the lsa model
        """

        svd = TruncatedSVD(n_components=svd_components, algorithm='arpack')
        # commenter('docbyterm....', lambda: print(type(term_by_doc_matrix)))
        # commenter('svd....', lambda: print(svd))
        print('\tperforming svd . . .')

        t_1 = time.time()
        u_sigma = svd.fit_transform(term_by_doc_matrix)
        m = '\tfit_transform took {} seconds, u_sigma contains {} MB'.format(time.time() - t_1,
                                                                             sys.getsizeof(u_sigma) / 1000000)
        log_and_print(m, self.logger)

        # truncate force my lsa to have truncate_matrix columns
        if self.options['truncate_matrix'] is not None:
            vt = svd.components_[:, 0:self.options['truncate_matrix']]
        else:
            vt = svd.components_

        m = '\tperforming lsa - np.dot(u_sigma {}, vt{}) . . . svd = {}'.format(u_sigma.shape,
                                                                                vt.shape, svd_components)
        log_and_print(m, self.logger)
        t_1 = time.time()
        lsa = np.dot(u_sigma, vt)
        m = '\tlsa = np.dot(u_s,vt) operation took {} seconds and lsa contains {} MB'.format(time.time() - t_1,
                                                                                             sys.getsizeof(
                                                                                                 lsa) / 1000000)
        log_and_print(m, self.logger)

        return lsa

    def svd_randomized(self, doc_by_term_matrix, svd_components):
        """
        Same as SVD_scikilearn when used with algorithm = randomized
        :param doc_by_term_matrix: a sparse matrix
        :param df: a data frame with the sparse matrix, indices and columns
        :param svd_components: number of components to select for the SVD
        :return: return lsa as ndarray, dataframe with the lsa model
        """

        t_1 = time.time()
        U, Sigma, VT = randomized_svd(doc_by_term_matrix, n_components=svd_components)

        # commenter('SCI --- \nU',
        #           lambda: print(pd.DataFrame(U), "\nSigma\n", pd.DataFrame(Sigma), "\nVT\n", pd.DataFrame(VT)))
        # commenter('vector U', lambda: print(type(U), len(U), U.shape))
        # Up, Sigmap, VTp = U, np.diag(Sigma), VT
        m = '\trandomized_svd() took {} seconds'.format(time.time() - t_1)
        log_and_print(m, self.logger)

        t_1 = time.time()
        if self.options['truncate_matrix'] is not None:
            lsa = (U * Sigma).dot(VT[:, 0:self.options['truncate_matrix']])
        else:
            lsa = (U * Sigma).dot(VT)

        m = '\tdot operation took {} seconds, generating an lsa of {} MB'.format(time.time() - t_1,
                                                                                 sys.getsizeof(lsa) / 1000000)
        log_and_print(m, self.logger)

        # docs = ["doc_" + str(i + 1) for i in range(lsa.shape[1])]
        # pdf_lsa = pd.DataFrame(lsa, columns=docs, index=df.index)
        # commenter('lsa randomized', lambda: print(pdf_lsa))

        return lsa

    def singular_value_decomposition_numpy(self, doc_by_term_matrix, svd_components):
        # NOT REALLY WORKING.. IT TAKES TOO LONG... :( OUTDATED!
        """
        Performs singular value decoposition, and then approximates the recontruction of the matrix
        by slicing the decomposed matrices.  We do slicing manually in this method
        :param doc_by_term_matrix: a sparse matrix with dtype D or F, but cannot be integer
        :param df: a data frame with the sparse matrix, indices and columns
        :param svd_components: number of components to select for the SVD
        :return: return lsa as ndarray, dataframe with the lsa model
        """
        print('\tperforming svd with numpy. . .')
        t_1 = time.time()

        u, s, vt = np.linalg.svd(doc_by_term_matrix.toarray())

        m = '\tnp.linealg.svd took {} seconds'.format(time.time() - t_1)
        log_and_print(m, self.logger)

        # commenter('u', lambda: print(pd.DataFrame(u), "\ns\n", pd.DataFrame(s), "\nvt\n", pd.DataFrame(vt)))
        # slicing on the number of topics or components

        print('\tslicing svd with numpy. . .')
        t_1 = time.time()

        up, sp, vtp = u[:, 0:svd_components], np.diag(s[0:svd_components]), vt[0:svd_components, :]

        m = '\tslicing operation took {} seconds'.format(time.time() - t_1)
        log_and_print(m, self.logger)

        # multiply back the matrix to get an approximation
        print('\tmultiplying svd with numpy. . .')
        t_1 = time.time()
        lsa = np.dot(np.dot(up, sp), vtp)
        m = '\tlsa = np.dot(n.dot(u, s), vt) operation took {} seconds. LSA contains {} MB'.format(time.time() - t_1,
                                                                                                   sys.getsizeof(
                                                                                                       lsa) / 1000000)
        log_and_print(m, self.logger)

        # docs = ["doc_" + str(i + 1) for i in range(lsa.shape[1])]
        # pdf_lsa = pd.DataFrame(lsa, columns=docs, index=df.index)
        # commenter('lsa numpy', lambda: print(pdf_lsa))

        return lsa


def generate_a_model():
    logger = create_a_logger('model_generator.log')

    arguments = check_input()
    # print(arguments)

    logger.debug('\n************************ LOADING CORPUS . . .')
    t1 = time.time()

    # corpus = Corpus_ejemplo_clase()
    # corpus = Corpus_movie_reviews()
    corpus = Corpus_all_senate_speeches()

    d, h, m, s = reformat_execution_time(time.time() - t1)
    log_and_print('. . . ' + corpus.source_id + ' loaded, it took {}d {}h {}m {} seconds to load'.format(d, h, m, s),
                  logger)
    log_and_print('{} docs loaded!'.format(len(corpus.data)), logger)

    generator = ModelGenerator(corpus, arguments)
    generator.logger = logger  # so the ModelGenerator class records its steps

    corpus = None  # free memory space

    print(generator.options['action'])

    logger.info('Filename: ' + generator.build_filename())
    logger.info('Options are: ' + str(generator.options))
    logger.debug('Starting to execute the action: ' + generator.options['action'])

    generator.execute_action()

    log_and_print('{} words in the model from {} documents'.format(len(generator.feature_words), len(generator.corpus)),
                  logger)
    d, h, m, s = reformat_execution_time(time.time() - t1)
    duration = '{} took {} day,  {} hr,  {} min,  {} seconds to complete'.format(generator.corpus_source.upper(),
                                                                                 int(d), int(h), int(m), int(s))
    logger.info(duration)
    logger.info(generator.corpus_source.upper() + '\n************************ . . . MODEL GENERATION COMPLETED!')


if __name__ == '__main__':
    generate_a_model()
