from traceback import print_exc
import numpy as np
import pandas as pd
import json
import sys
import os

PROJECT_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from nlu_helper.read_model import MatrixModel, SentenceProcessor
from nlu_helper.utility import load_json2py, create_a_logger, resources_folder, log_and_print
from nlu_helper.utility import create_results_directory_if_needed


class ModelApplier:

    def __init__(self, filename):
        self.logger = create_a_logger('read_model.log')
        self.logger.info('*************************************** STARTING TO APPLY MODEL')
        self.logger.info('model file used: ' + filename)
        self.logger.debug('creating the object of the MatrixModel')

        self.matrix_model = MatrixModel(filename, logger=self.logger)

        self.logger.debug('finish creating the MatrixModel object')
        # print(matrix_model.args)
        self.sp = SentenceProcessor(self.matrix_model, logger=self.logger)
        self.model_filename = filename

    def paragraph_vs_sentences_similarities(self, real_answer: str, given_answers: list) -> list:
        """
        :param real_answer: is a string containing the systems sentence/paragraph that we need to compare against
        :param given_answers: is a list of the possible answers that could have been given by the human user
        returns a list containing all cosine similarity values that were computed from comparing the answer to each
         possible answer on the question dictionary
        """
        r_tok = self.sp.tokenize_sentence(real_answer)
        real_sentence_vector = self.sp.get_sentence_vector(r_tok)
        array_sim = list()

        for given_answer in given_answers:
            # print(given_answer)
            g_tok = self.sp.tokenize_sentence(given_answer['text'])
            sim = self.sp.cos_sim(real_sentence_vector,
                                  self.sp.get_sentence_vector(g_tok))
            array_sim.append(sim)
            # print('result: ', sim)

        return array_sim

    def sort_similarity(self, cosine_similarity_results, question: dict) -> dict:
        # Add two more columns with 'id', 'text' tags
        ids = [q['id'] for q in question['possible_answers']]
        texts = [q['text'] for q in question['possible_answers']]
        pdf = pd.DataFrame({'id': ids, 'similarity': np.asarray(cosine_similarity_results), 'text': texts},
                           index=ids)

        sorted_pdf = pdf.sort_values(by='similarity', ascending=False)
        self.generate_log_similarities_report(question, sorted_pdf)

        final_order = [i + 1 for i in range(len(sorted_pdf.values))]
        sorted_pdf.insert(1, 'rank', final_order)

        item = {
            "item_id": question['lp_id'],
            "item_text": question['answer'],
            "result_as_pdf": json.loads(sorted_pdf.to_json()),
            "result_only_id": sorted_pdf.index.tolist()
        }
        return item

    def generate_log_similarities_report(self, question, pdf):
        m = '\n--------------------------------------- \nRESULTS \nCompare: ' + str(question['answer']) \
            + '\nTo: \n' + str(pdf) \
            + '\n---------------------------------------'
        log_and_print(m, self.logger)

    def save_model_results(self, output_model_results, items, path_testfile):
        with open(output_model_results + '.json', 'w') as fp:
            json.dump({
                "testfile": path_testfile,
                "model": {
                    "filename": self.matrix_model.filename,
                    "args": self.matrix_model.args
                },
                "items": items
            },
                fp)


def apply_model(path_filename, path_testfile, path_output_folder):
    ma = ModelApplier(path_filename)

    model_filename = extract_file_name(path_filename)
    test_filename = extract_file_name(path_testfile)
    print(model_filename)

    try:
        questionaire = load_json2py(path_testfile)

        items = list()
        for question in questionaire:
            possible_answers = question['possible_answers']
            answer = question['answer']

            if len(possible_answers) != 0:
                cos_sim_results = ma.paragraph_vs_sentences_similarities(real_answer=answer,
                                                                         given_answers=possible_answers)
                print(cos_sim_results)
                item = ma.sort_similarity(cos_sim_results, question)
                items.append(item)

        results_filename = path_output_folder + '/' + test_filename + '_VS_' + 'model_results_' + model_filename
        ma.save_model_results(results_filename, items, test_filename)

        ma.logger.info('FINISHED\n*************************************** We are DONE using MODEL')

    except FileNotFoundError:
        print('There is an error with the test data set file . . .')
        print_exc()


def list_models_on_directory(models_directory):
    models_path = os.path.join(resources_folder(models_directory))
    filename_list = os.listdir(models_path)
    print(filename_list)
    return get_rid_of_the_extension(filename_list)


def get_rid_of_the_extension(filename_with_extension):
    # ignore .DStore
    d = {name.split('.')[0] for name in filename_with_extension if name != '.DS_Store'}
    print(d)
    print(len(d))
    return d


def extract_file_name(path):
    if '/' in path:
        return path.split('/')[-1]
    else:
        return path


def main(test, models_parent_directory, out_folder):
    create_results_directory_if_needed(out_folder)
    all_models = sorted(list_models_on_directory(models_parent_directory))
    for model_name in all_models:
        print(model_name)
        print('\n\n--------------------------> START\nModel in: ', model_name)
        apply_model(models_parent_directory + '/' + model_name, test, out_folder)
        print('--------------------------> END\n\n')


if __name__ == '__main__':

    # TODO: add three arguments to be used as:
    #  1. fnameTest: relative or absolute path to the file that contains the test data set
    #  2. outFolder: the name of the folder that will contain the results or each model
    #  3. fnameModel: relative or absolute path to the file that contains all the models.
    #  It would be necessary to run some validation with the files, to make sure they are in the correct format
    #  before the model is applied.

    # Test data set: what is the json containing the sentences I want to test, within resource folder?
    # fnameTest = 'Question-Answer/qa_non_survey_related_1'
    # fnameTest = 'Question-Answer/landauer'
    fnameTest = 'Question-Answer/qa'

    # Folder where models are: where are my models? which folder within resource folder?
    models_parent_dir = 'all_senate_speeches'

    # Output folder: where do I want the similarities results? per model?
    # outFolder = 'apply_model_df_2'
    # outFolder = 'apply_model_trunc600_landauer'
    outFolder = 'apply_model_trunc1000'

    # ^----------------------- need to specify three variables above.

    main(fnameTest, models_parent_dir, outFolder)
