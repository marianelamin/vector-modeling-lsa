import pandas as pd
import json


def save_json(fname, json_obj):
    with open(fname, 'w') as fp:
        json.dump(json_obj, fp)

def read_csv_with_panda(filepath):
    data = pd.read_csv(filepath, index_col=0)
    return data

def separate_amt_input_tasks(csv_file):
    data = read_csv_with_panda(csv_file)
    print(data)
    # row = 0
    for row in range(len(data.values)):
        print(data[row:row+1])
        data[row:row+1].to_csv('amt_input/amt_input_' + str(row) + '.csv')


def create_QA_test_file(csv_in_file, json_out_file):
    print('creating the json file with testing . . .')
    qna_list = list()

    data = read_csv_with_panda(csv_in_file)
    print(data)
    # row = 0
    for row in range(len(data.values)):
        a_row = data[row:row + 1]
        answer = a_row.values[0][0]
        lp_id = a_row.index.values[0]
        possible_answers = [{"id": "SP1", "text": a_row.values[0][1]},
                            {"id": "SP2", "text": a_row.values[0][2]},
                            {"id": "SP3", "text": a_row.values[0][3]},
                            {"id": "SP4", "text": a_row.values[0][4]},
                            {"id": "SP5", "text": a_row.values[0][5]},
                            {"id": "SP6", "text": a_row.values[0][6]}
                            ]

        # print(possible_answers)
        element = {
            "answer": answer,
            "lp_id": lp_id,
            "possible_answers": possible_answers
        }
        qna_list.append(element)

    # print(qna_list)
    filename = json_out_file
    save_json(filename, qna_list)


if __name__ == '__main__':

    print('Welcome.  Paths in this script are relative to the current folder.')
    csv = 'amt_input/input_for_amazon_mechanical_turk - input.csv'
    # generating the 10 different batch files for the Amazon Mechanical Turk
    separate_amt_input_tasks(csv)

    # creating the Q&A json file that will be used by the system as a test input to compare paragraphs with a specific
    # Matrix model.
    json_file = '../nlu_helper/resources/Question-Answer/qa.json'
    create_QA_test_file(csv, json_file)


