import lore

from prepare_dataset import *
from neighbor_generator import *
from evaluation import evaluate_explanation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def main():

    dataset_name = 'german_credit.csv'
    path_data = './datasets/'
    dataset = prepare_german_dataset(dataset_name, path_data)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    blackbox = RandomForestClassifier(n_estimators=20)
    blackbox.fit(X_train, y_train)

    X2E = X_test
    idx_record2explain = 1

    explanation, infos = lore.explain(idx_record2explain, X2E, dataset, blackbox,
                                      ng_function=genetic_neighborhood,
                                      discrete_use_probabilities=True,
                                      continuous_function_estimation=True,
                                      returns_infos=True)

    x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]

    print('x = %s' % x)
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)

    print('Evaluation')
    bb_outcome = infos['bb_outcome']
    cc_outcome = infos['cc_outcome']
    y_pred_bb = infos['y_pred_bb']
    y_pred_cc = infos['y_pred_cc']
    dfZ = infos['dfZ']
    dt = infos['dt']
    tree_path = infos['tree_path']
    leaf_nodes = infos['leaf_nodes']
    diff_outcome = infos['diff_outcome']

    print(evaluate_explanation(x, blackbox, dfZ, dt, tree_path, leaf_nodes, bb_outcome, cc_outcome,
                               y_pred_bb, y_pred_cc, diff_outcome, dataset, explanation[1]))


if __name__ == "__main__":
    main()
