from traceback import print_exc
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import os, sys

PROJECT_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from nlu_test.compare_order import order_preservation_measure
from nlu_helper.utility import create_results_directory_if_needed


class HumanResults:
    def __init__(self, humans_results_folder):
        self.amt_directory = humans_results_folder
        self.results = dict()
        print('\n-------- files of Amazon Mechanical Turk results -------\n')

        self.cell_to_plot = 0
        self.rows, self.cols = 2, 5
        fig, self.axs = plt.subplots(self.rows, self.cols, sharey='row', figsize=(15, 8))
        plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9, right=0.95, bottom=0.1, left=0.05)

        perform_action_on_directory_files(self.amt_directory, self.read_from_amt)

        # plt.show()

    def save_plot(self, path):
        plt.savefig(os.path.join(path, 'human_rankings'))

    def read_from_amt(self, csv_file):
        print(csv_file)
        res = pd.read_csv(os.path.join(self.amt_directory, csv_file) + '.csv')
        lp_id = res['Input.LP'][0]
        rank_results = res[res['AssignmentStatus'].str.startswith('Approved')][
            ['Answer.s1', 'Answer.s2', 'Answer.s3', 'Answer.s4', 'Answer.s5', 'Answer.s6']]

        rank_results = rank_results.rename(
            columns={'Answer.s1': 'SP1', 'Answer.s2': 'SP2', 'Answer.s3': 'SP3', 'Answer.s4': 'SP4', 'Answer.s5': 'SP5',
                     'Answer.s6': 'SP6'})

        print('renamed: ', rank_results)
        r = rank_results.T
        r['av'] = rank_results.mean(0)
        r_sorted = r.sort_values(by="av")
        r_sorted['rank'] = [i + 1 for i in range(len(r_sorted['av']))]
        item = r_sorted['rank'].to_dict()
        self.results[lp_id] = item
        print(self.results)
        h_val, keys = r_sorted['rank'].values, r_sorted['rank'].index.values

        index_row = self.cell_to_plot // self.cols
        index_col = self.cell_to_plot % self.cols
        self.add_plot(h_val, keys, index_row, index_col, lp_id)

        for index, row in (r_sorted.iloc[:, 0:10:1].T).iterrows():
            print(row.values)
            self.axs[index_row][index_col].scatter(keys, row.values, s=100, c="#ffa500", alpha=0.2, marker='o')

        self.cell_to_plot = self.cell_to_plot + 1

    def add_plot(self, h_values, keys, row, col, title):
        plot_pair = pd.DataFrame({'human': h_values}, index=keys)
        plot_pair_sorted = plot_pair.sort_values(by="human")
        hu = plot_pair_sorted['human'].values
        x = plot_pair_sorted.index.values
        print('[' + str(row) + '][' + str(col) + ']')

        # self.axs[row][col].scatter(x, hu, marker='>', c="b", alpha=0.9)  # s=area, c=colors, alpha=0.5
        self.axs[row][col].set_ylabel('Ranking (1-more to 6-less similar)', fontsize=9)
        # self.axs[row][col].set_xlabel('Sentences of ' + title, fontsize=9)
        self.axs[row][col].set_title("Human ranking: " + title, fontsize=9)
        self.axs[row][col].grid(True)


class VsmResults:

    def __init__(self, models_result_folder):
        self.models_directory = models_result_folder
        print(self.models_directory)
        self.entire_report = list()
        self.all_results = list()

        print('\n-------- files of VSM models results -------\n')

        perform_action_on_directory_files(self.models_directory, self.merge_all_vsm)

        # to test only one model
        # self.merge_all_vsm('qa_VS_model_results_all_senate_speeches-min_df_1-tfidf-svd300-lsa_115168x150')
        print(self.all_results)

    def merge_all_vsm(self, filename):
        print('\t', self.models_directory + '/' + filename)
        with open(os.path.join(self.models_directory, filename + '.json')) as f:
            js = json.load(f)
        self.entire_report.append(js)
        # print(js)
        sim, partial_results = dict(), dict()
        for item in js['items']:
            print(item['item_id'])
            print(item['result_as_pdf']['rank'])
            partial_results[item['item_id']] = item['result_as_pdf']['rank']
            sim[item['item_id']] = item['result_as_pdf']['similarity']

        self.all_results.append({"model": js['model'], "results": partial_results, "sim": sim})

    def all_similarity_results(self, path):
        p = list()
        sim = dict()
        for m in self.entire_report:
            for item in m['items']:
                simL = {item['item_id'] + "-" + sp: r for sp, r in item['result_as_pdf']['similarity'].items()}
                sim = {**sim, **simL}
            p.append({"model": self.get_a_description_for_the_model(m['model']['args']), **sim})
            # print(r)
        pd.DataFrame(p).to_csv(path+'.csv')

    def get_a_name_for_the_model(self, d: dict) -> str:
        print(d)
        filename = 'model' \
                   + '-' + str(d['action']) \
                   + (('-dim' + str(d['svd_components'])) if (d['action'] == 'lsa') else '') \
                   + ('-' + str(d['score'])) \
                   + ('-stop' if d['stopwords'] else '')
        return filename

    def get_a_description_for_the_model(self, d: dict) -> str:
        # print(d)
        column_name = 'lsa:' + ('yes' if (d['action'] == 'lsa') else '-') \
                      + (' dim:' + (str(d['svd_components']) if (d['action'] == 'lsa') else '-')) \
                      + (' sco:' + str(d['score'])) \
                      + (' sw:' + ('yes' if d['stopwords'] else '-'))
        return column_name


class Evaluator:

    def __init__(self, vsm_object, amt_object, out_file):
        self.vsm = vsm_object
        self.amt = amt_object
        self.results_directory: str = out_file
        print('\n\n')
        self.data_set_names = None
        self.fig = None
        self.axs = None
        self.num_plot_rows, self.num_plot_cols = 2, 5
        self.cell_to_plot = 0
        self.data_set_names = {'x': 'more (1) to less similar (6)', 'y': 'Rank'}
        self.vsm.all_similarity_results(self.results_directory+'/cos_similarity_results')

    def plot_init(self, width=15, height=8):
        fig, axs = plt.subplots(self.num_plot_rows, self.num_plot_cols, sharey='row', figsize=(width, height))
        plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9, right=0.95, bottom=0.1, left=0.05)
        return fig, axs

    def compute_evaluation_with_one_model(self, human_results, key_model, val_model) -> tuple:
        corr = dict()
        spearman_results = dict()
        crssman_results = dict()
        kendall_results = dict()
        machine_result_vsm = val_model['results']
        # print(machine_result_vsm)

        fig, self.axs = self.plot_init(15, 8)
        self.cell_to_plot = 0
        for p_key, p_val in human_results.items():
            print('******* CELL ', self.cell_to_plot)
            pair, keys, h_values, m_values, sim_values = list(), list(), list(), list(), list()
            for s_key in human_results[p_key].keys():
                # print(s_key)
                keys.append(s_key)
                h_values.append(human_results[p_key][s_key])
                m_values.append(machine_result_vsm[p_key][s_key])
                sim_values.append(val_model['sim'][p_key][s_key])
                pair.append([human_results[p_key][s_key], machine_result_vsm[p_key][s_key]])

            # df =
            correlation, pval = spearmanr(pd.DataFrame(pair))
            print(f'correlation={correlation:.6f}, p-value={pval:.6f}')
            crssman = order_preservation_measure(h_values, m_values)
            print(f'position={crssman:.6f}')
            k_correlation, k_pval = kendalltau(h_values, m_values)
            print(f'kendall={k_correlation:.6f}, p-value={k_pval:.6f}')

            corr[p_key] = {
                "evaluation": {
                    "spearman_corr": correlation,
                    "spearman_p_value": pval,
                    "order_preservation_measure": crssman,
                    "kendall_corr": k_correlation,
                    "kendall_p_value": k_pval,
                }
            }
            spearman_results[p_key] = correlation
            crssman_results[p_key] = crssman
            kendall_results[p_key] = k_correlation

            d = {'human': h_values, 'machine': m_values, "sim": sim_values}

            self.add_plots(correlation, p_key, d, keys)
        plt.suptitle(key_model)

        # plt.show()
        plt.savefig(os.path.join(self.results_directory, key_model))
        return corr, [spearman_results, crssman_results, kendall_results]

    def add_plots(self, correlation, p_key, values, keys):
        plot_pair = pd.DataFrame(values, index=keys)
        plot_pair_sorted = plot_pair.sort_values(by="human")
        hu = plot_pair_sorted['human'].values
        ma = plot_pair_sorted["machine"].values
        sim = [(15*i)**2 for i in plot_pair_sorted["sim"].values]
        x = plot_pair_sorted.index.values
        row = self.cell_to_plot // self.num_plot_cols
        col = self.cell_to_plot % self.num_plot_cols
        print('[' + str(row) + '][' + str(col) + ']\t\t', self.cell_to_plot)
        self.cell_to_plot = self.cell_to_plot + 1
        self.axs[row][col].scatter(x, hu, label='human', s=15**2, c="#ffa500", alpha=0.5, marker='o')
        self.axs[row][col].scatter(x, ma, label='machine', s=sim, c="#2196f3", alpha=0.5, marker='o')
        self.axs[row][col].set_ylabel(self.data_set_names['y'])
        self.axs[row][col].set_xlabel(self.data_set_names['x'])
        self.axs[row][col].set_title(p_key + ': corr=' + str('%1.4f' % correlation),
                                     fontsize=9)  # + ' p-val=' + str('%1.4f' % pval),
        self.axs[row][col].grid(True)
        self.axs[row][col].legend(loc="upper left")


    def compute_evaluation(self, out_filename: str) -> None:
        eval_table = {"res_as_pdf_spearman": dict(),
                      "res_as_pdf_crssman": dict(),
                      "res_as_pdf_kendall": dict(),
                      "res": dict()}

        for val_model in self.vsm.all_results:
            figure_model_name = self.vsm.get_a_name_for_the_model(val_model['model']['args'])
            co, m_results = self.compute_evaluation_with_one_model(self.amt.results, figure_model_name, val_model)
            description_model = self.vsm.get_a_description_for_the_model(val_model['model']['args'])
            eval_table[description_model] = co
            eval_table["res_as_pdf_spearman"][description_model] = m_results[0]
            eval_table["res_as_pdf_crssman"][description_model] = m_results[1]
            eval_table["res_as_pdf_kendall"][description_model] = m_results[2]

        # save work on a json file
        with open(os.path.join(self.results_directory, out_filename) + '.json', 'w') as fp:
            json.dump(eval_table, fp)

        # save the correlation on a csv -  as table on the paper
        pd.DataFrame.from_dict(eval_table["res_as_pdf_spearman"]).T.to_csv(
            os.path.join(self.results_directory, out_filename + '_spearman') + '.csv')
        pd.DataFrame.from_dict(eval_table["res_as_pdf_crssman"]).T.to_csv(
            os.path.join(self.results_directory, out_filename + '_crssman') + '.csv')
        pd.DataFrame.from_dict(eval_table["res_as_pdf_kendall"]).T.to_csv(
            os.path.join(self.results_directory, out_filename + '_kendall') + '.csv')


def files_on_directory_no_ext(directory):
    nlu_test_path = os.path.dirname(__file__)
    path = os.path.join(nlu_test_path, directory)
    # print(path)
    filename_list = os.listdir(path)
    # print(filename_list)

    # exclude .DStore file
    return sorted([name.split('.')[0] for name in filename_list if name != '.DStore'])


def perform_action_on_directory_files(directory: str, action) -> None:
    all_files = files_on_directory_no_ext(directory)
    print(len(all_files))
    print(all_files)

    for file in all_files:
        if file is None: continue
        if len(file) == 0: continue
        action(file)


def main(results_directory_vsm):
    results_directory_amt = 'amt_results'
    output_folder = results_directory_vsm + '_eval'
    create_results_directory_if_needed(output_folder)
    try:
        h = HumanResults(results_directory_amt)
        h.save_plot(output_folder)

        k = VsmResults(results_directory_vsm)
        e = Evaluator(k, h, output_folder)
        e.compute_evaluation('results')

    except FileNotFoundError:
        print_exc()
        print('There was an error with the directories')


if __name__ == '__main__':
    # TODO: add two arguments to be used as:
    #  1. similarity_vsm_results: relative or absolute path to the folder that contains vector space models
    #  2. results_directory_amt: relative or absolute path to the folder containing the batch results from AMT
    #     outFolder: the name of the folder that will contain the results of the study. there will be graphs
    #     and a csv file with the correlation of each human's ranking against the machine's ranking
    #  It would be necessary to run some validation with the files, to make sure they are in the correct format
    #  before the model is evaluated.

    # Where are the similarity results from each model? which folder?
    # results_directory_vsm = 'apply_model_trunc600_landauer'
    similarity_vsm_results = 'apply_model_trunc1000'

    # ^----------------------- only need to specify where the results from each model are stored.

    main(similarity_vsm_results)