import argparse
import json
import re
import timeit
import os
import logging
import pickle
import time

import numpy as np
import pandas as pd
from scipy import sparse


def script_timer(mysetup, mycode):
    return timeit.timeit(setup=mysetup,
                         stmt=mycode,
                         number=1)


def create_results_directory_if_needed(out_folder):
    try:
        os.makedirs(out_folder)
        print(out_folder, ' has been created')
    except FileExistsError:
        print(out_folder, ' has already been created')


def resources_folder(filename):
    return os.path.join(os.path.dirname(__file__), 'resources', filename)


def logs_folder(filename):
    return os.path.join(os.path.dirname(__file__), 'logs', filename)


def looks_like_a_filename(text_alike_filename):
    """
    Return true or false if the text_alike_filename has the following format
     plain name or name.extension
     [a-zA-Z0-9]+\.[a-zA-Z0-9]
    regex:  ^[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)?$
    """
    pattern = re.compile(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9]+)?$')
    res = pattern.match(text_alike_filename)
    # print(res)
    return res


def filename_regex(arg_value):
    if not looks_like_a_filename(arg_value):
        msg = "it looks like %s is not a valid filename. No paths are allowed." % arg_value
        raise argparse.ArgumentTypeError(msg)
    return arg_value


def commenter(description, my_function) -> None:
    print('***** START *********************************************')
    print(description)
    my_function()
    print('***** END ***********************************************\n')


def show_message_log_console(message: str, logger: logging = None) -> None:
    if logger is not None:
        logger.debug(message)
    else:
        print(message)


def log_and_print(message, logger: logging = None) -> None:
    if logger is not None:
        logger.info(message)
    print(message)


def record_how_long_it_takes(method, logger, message):
    m_name = method.__name__
    # print(m_name)
    logger.info(m_name + " - " + message)
    start_time1 = time.time()
    print(message)
    ret = method()
    start_time2 = time.time()

    tiempo = start_time2 - start_time1
    d, h, m, s = reformat_execution_time(tiempo)
    duration = '{} took {} days {} : {} : {} seconds - ({} seconds total)'.format(m_name,
                                                                                  int(d), int(h), int(m), int(s),
                                                                                  tiempo)
    logger.info(duration)

    return ret


def reformat_execution_time(time_measured) -> tuple:
    """
    this function takes a time in seconds
    returns how many days, hours, minutes and seconds that is in a tuple
    """
    d = time_measured // (3600 * 24)
    t_left = time_measured % (3600 * 24)  # seconds left, not enough for a day
    h = t_left // 3600
    t_left = t_left % 3600  # seconds left, not enough for an hour
    m = t_left // 60
    t_left = t_left % 60  # seconds left, not enough for a minute
    s = t_left
    return d, h, m, s


def write2file(filename, data) -> None:
    f = open(resources_folder(filename), 'w')
    f.write(data)
    f.close()
    print("\nOutput file: " + filename)


def pickle2file(filename, data, indices, num_cols, arguments, vectorizer) -> None:
    save_json(filename, arguments, data.shape, num_cols, indices, vectorizer)

    with open(resources_folder(filename) + '.pkl', 'wb') as fp:
        pickle.dump(data, fp, protocol=4)
    print("Data in (pickle): " + filename + '.pkl')


def file2pickle(filename) -> tuple:
    # Load from the pickle file
    with open(resources_folder(filename) + '.pkl', 'rb') as f:
        loaded_matrix = pickle.load(f)
        # print(type(loaded_matrix))
    # commenter('loaded matrix type()', lambda: print(type(loaded_matrix)))
    js = load_json2py(filename)
    rows = js['rows']
    # pdf_matrix = pd.DataFrame(loaded_matrix, index=rows)

    return loaded_matrix, rows, js['args']


def sparse2file(filename, data, indices, columns, arguments) -> None:
    print("Saving the data . . .\nsize: " + str(data.size) + " type: ", type(data))
    print("shape: " + str(data.shape) + "\ntype: ", type(data))
    print("Saving the matrix using sparse.csc_matrix and sparse.save_npz . . .")
    sparse.save_npz(resources_folder(filename), sparse.csc_matrix(data))
    print("\nData in (sparse-save_npz): " + filename + '.npz')
    save_json(filename, arguments, columns, indices)


def file2sparse(filename):
    js = load_json2py(filename)
    rows = js['rows']
    cols = js['cols']

    loaded_matrix = sparse.load_npz(resources_folder(filename) + ".npz").toarray()
    commenter('loaded matrix type()', lambda: print(type(loaded_matrix)))
    commenter('loaded matrix', lambda: print(loaded_matrix))

    pd_dataframe = pd.DataFrame(loaded_matrix, index=rows, columns=cols)

    commenter('pandaDataFrame: ', lambda: print(pd_dataframe))
    return pd_dataframe


def save_json(filename, arguments, data_size, n_cols, indices, vectorizer):
    print("Saving the args, word-count, doc-count, rows(words) and columns(doc names) . . .")
    with open(resources_folder(filename) + '.json', 'w') as fp:
        json.dump({
            "args": arguments,
            "token_pattern": vectorizer['token_pattern'],
            "wordcount": len(indices),
            "nDocuments": n_cols,
            "dataSize": data_size,
            "rows": indices,
            "ignored": {
                "count": len(vectorizer['ignored_words']),
                "words": vectorizer['ignored_words']
            }
        }, fp)
    print("File information in: " + filename + '.json')


def load_json2py(filename_no_json_ext) -> dict:
    file = resources_folder(filename_no_json_ext)
    try:
        with open(file + '.json') as f:
            js = json.load(f)
        return js
    except FileNotFoundError:
        print(file + '\n is not a valid path :(')


def np2file(filename, data, indices, columns, arguments) -> None:
    # np.savez(resources_folder(filename), data=data.data)
    np.savez_compressed(resources_folder("compressed_" + filename), data=data.data)

    save_json("compressed_" + filename, arguments, columns, indices)
    print("np2file Output file: " + "compressed_" + filename)


def file2np(filename):
    loaded_matrix = np.load(resources_folder(filename) + ".npz")
    commenter('loaded matrix type()', lambda: print(type(loaded_matrix)))
    commenter('loaded matrix', lambda: print(loaded_matrix))
    # recontruct = sparse.csr_matrix((loaded_matrix['data'], loaded_matrix['indices'], loaded_matrix['indptr']),
    #                                shape=loaded_matrix['shape'])
    js = load_json2py(filename)
    rows = js['rows']
    cols = js['cols']
    print(len(rows))
    print(len(cols))
    print('matrix: ', loaded_matrix)
    print('shape: ', loaded_matrix.shape)
    pdf_matrix = pd.DataFrame(loaded_matrix, index=rows, columns=cols)

    return pdf_matrix, rows, cols, js['args']


def dataFrame2file(filename, data) -> None:
    data.to_csv(resources_folder(filename))
    print("\nOutput file: " + filename)


def load_csv_model(filename) -> tuple:
    """
    loads the data from the csv into the LSA model
    :return: void
    """
    dat_sci = pd.read_csv(resources_folder(filename), index_col=0)
    commenter('data from ' + filename, lambda: print(dat_sci))

    ind = dat_sci.index
    # commenter('index', lambda: print(ind))
    col = dat_sci.columns
    # commenter('columns', lambda: print(col))
    # self.data = np.asmatrix(dat_sci.values)
    # commenter('data', lambda: print(self.data))
    # print(type(dat_sci))

    return dat_sci, ind, col


def load_npz_model(filename) -> tuple:
    """ with no .npz or .json"""
    # load_row_cols_json
    js = load_json2py(filename)
    # print(js)
    # print(type(js['rows']))
    rows = js['rows']
    # print(type(js['cols']))
    documents = js['cols']

    loaded_matrix = pd.DataFrame(sparse.load_npz(resources_folder(filename) + ".npz").toarray(),
                                 index=rows,
                                 columns=documents)

    return loaded_matrix, rows, documents, js['args']


def check_input():
    parser = argparse.ArgumentParser(description='Generate an LSA model.')
    parser.add_argument('--action',
                        type=str,
                        choices=['lsa', 'docbyterm'],
                        default='docbyterm',
                        help='Generates a model using the specified action.  Latent Semantic Analisys = lsa, or '
                             'document by term matrix. Default: docbyterm')
    parser.add_argument('-min_df', '--min_document_frequency',
                        type=int,
                        default=1,
                        help='When building the vocabulary of words, ignore terms that have a document frequency '
                             'strictly lower than the given threshold. Default: 1')
    parser.add_argument('-s', '--stopwords',
                        action="store_true",
                        help='Determines whether or not to use stop words when processing the text.  Default: False')
    parser.add_argument('--score',
                        type=str,
                        choices=['tfidf', 'zeroone', 'count'],
                        default='count',
                        help='Generates a document by term matrix using count (frequency of the word in each document),'
                             ' tfidf (weighted frequency) or zeroone (discrete frequency, 1 is term appears, zero if it'
                             ' does not. Default: count')
    parser.add_argument('-svd_c', '--svd_components',
                        type=int,
                        default=100,
                        help='Number of components that will be used when processing SVD. Default: 100')
    parser.add_argument('-truncate', '--truncate_matrix',
                        type=int,
                        default=None,
                        help='Truncates the resulting matrix model.  When computing LSA: the (U * Sigma) * VT,'
                             ' only keeping the first -trucate columns of the VT matrix.  When computing only term '
                             'by document matrix: the resulting matrix will only contain the first -truncate columns.  '
                             'En either case, the matrix model will have a size of #totalWords by -truncate.  '
                             'Default: Does not truncate the model')
    parser.add_argument('-o', '--output',
                        type=filename_regex,
                        default=None,
                        help='Name of output file, must contain letters, numbers or underscore (_).  '
                             'Default: a descriptive name will be provided containing the options that were selected '
                             'when the script was executed')

    args = parser.parse_args()

    return vars(args)


def create_a_logger(logname):
    logger = logging.getLogger('main')

    # import os, sys
    # path, filename = os.path.split(sys.argv[0])
    # print(path)
    logging.basicConfig(filename=logs_folder(logname),
                        format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.WARNING)
    logger.addHandler(consoleHandler)
    consoleHandler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    return logger
