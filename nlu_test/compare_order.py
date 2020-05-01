import pandas as pd

def order_preservation_measure(human_order: list, machine_order: list) -> float:
    """
    A method tht computer how well is an array ordered
    :param human_order: a list containing int numbers
    :returns: a number between 0 and 1, representing how well ordered is the list passed in
    """
    print('human. ', human_order)
    print('machine. ', machine_order)
    pair = pd.DataFrame({'human': human_order, 'machine': machine_order})
    list_order = pair.sort_values(by='human')['machine'].values

    if len(list_order) != 0:
        count, accumulation = 0, 0
        for i in range(len(list_order)):
            # print(i, list_order[i])
            # print('-----', range(i, len(list_order)))
            for j in range(i+1, len(list_order)):
                order = list_order[i] < list_order[j]
                point = 1 if order else 0
                count, accumulation = count + 1, accumulation + point
                print('(', i, ', ', j, ')', '\t', order, '\t ', list_order[i], ' < ', list_order[j], '\t', point)
        print(accumulation, '/', count)

        return accumulation/count
    else:
        raise ValueError('arr2 must contain at least one element')


if __name__ == '__main__':
    user = [
            {"text": "a paragraph that talks about something", "ind":4, "sim": 0.7},
            {"text": "a paragraph that talks about something", "ind":3, "sim": 0.6},
            {"text": "a paragraph that talks about something", "ind":2, "sim": 0.45},
            {"text": "a paragraph that talks about something", "ind":6, "sim": 0.44},
            {"text": "a paragraph that talks about something", "ind":1, "sim": 0.4},
            {"text": "a paragraph that talks about something", "ind":7, "sim": 0.34},
            {"text": "a paragraph that talks about something", "ind":5, "sim": 0.3}
            ]

    human_order = [d['ind'] for d in user]
    computer_order = [d['ind'] for d in user]

    print(order_preservation_measure(human_order, computer_order))




def nop(machine_order, human_order):
    panda_pair = pd.DataFrame({'human': human_order, 'machine': machine_order},
                        index=["sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "sp7"])
    panda_pair_sorted = panda_pair.sort_values(by="human")
    hu = panda_pair_sorted['human'].values
    ma = panda_pair_sorted["machine"].values
    x = panda_pair_sorted.index.values
