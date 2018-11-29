import re
import lore
import datetime

from prepare_dataset import *
from neighbor_generator import *

from anchor import anchor_tabular
from statistics import mode

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from evaluation import hit_outcome


def fit_anchor(dataset, X_train, X_test, y_train, y_test, X2E):

    class_name = dataset['class_name']
    columns = dataset['columns']
    continuous = dataset['continuous']
    possible_outcomes = dataset['possible_outcomes']
    label_encoder = dataset['label_encoder']


    feature_names = list(columns)
    feature_names.remove(class_name)

    categorical_names = dict()
    idx_discrete_features = list()
    for idx, col in enumerate(feature_names):
        if col == class_name or col in continuous:
            continue
        idx_discrete_features.append(idx)
        categorical_names[idx] = label_encoder[col].classes_

    # Create Anchor Explainer
    explainer = anchor_tabular.AnchorTabularExplainer(possible_outcomes, feature_names, X2E, categorical_names)
    explainer.fit(X_train, y_train, X_test, y_test)

    return explainer


def anchor2arule(anchor_exp):
    anchor_dict = dict()
    for a in anchor_exp.names():
        k, v = None, None
        index_leq = a.find('<=')
        index_eq = a.find('=')
        index_geq = a.find('>=')

        if len(re.findall('.*<.*<=.*', a)):
            k = a.split('<')[1].strip().rstrip()
            v = a
        elif index_leq != -1 and index_leq < index_eq and '<=' in a:
            k = a.split('<=')[0].strip()
            v = '<=%s' % a.split('<=')[1].strip()
        elif '>' in a and not '>=' in a:
            k = a.split('>')[0].strip()
            v = '>%s' % a.split('>')[1].strip()
        elif '=' in a:
            k = a[:index_eq].strip()
            v = a[index_eq+1:].strip()

        if k is not None:
            anchor_dict[k] = v
    return anchor_dict


def run_experiment(blackbox, X2E, y2E, idx_record2explain, dataset, anchor_explainer, path_data, verbose=False):

    # class_name = dataset['class_name']
    # columns = dataset['columns']
    # features_type = dataset['features_type']
    # discrete = dataset['discrete']
    # continuous = dataset['continuous']
    # possible_outcomes = dataset['possible_outcomes']
    # label_encoder = dataset['label_encoder']

    # Remove From the Dataset to Explain x and return both them
    # starttime = datetime.datetime.now()
    # dfX2E, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)

    # Run Black Box on Instance to Explain
    bb_outcome = y2E[idx_record2explain] #blackbox.predict(x.reshape(1, -1))[0]
    # print(bb_outcome, type(bb_outcome))

    dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records')

    individual_hit_lore = 0
    fidelity_acc_lore = fidelity_f1_lore = coverage_lore = coverage_Z_lore = 0
    precision_lore = [0]
    individual_hit_anchor = fidelity_acc_anchor = fidelity_f1_anchor = coverage_anchor = coverage_Z_anchor = 0
    precision_anchor = [0]

    def eval(x, y):
        return 1 if x == y else 0

    print(datetime.datetime.now(), '\tLORE')
    attempt = 0
    while True:
        try:
            # Explanation with LORE
            lore_explanation, lore_info = lore.explain(idx_record2explain, X2E, dataset, blackbox,
                                                       ng_function=genetic_neighborhood, discrete_use_probabilities=True,
                                                       continuous_function_estimation=False,
                                                       returns_infos=True, path=path_data, sep=';', log=verbose)

            cc_outcome_lore = lore_explanation[0][0][dataset['class_name']]
            # print(cc_outcome_lore, type(cc_outcome_lore), bb_outcome, type(bb_outcome))
            # print(cc_outcome_lore == bb_outcome)
            individual_hit_lore = hit_outcome(bb_outcome, cc_outcome_lore)

            y_pred_bb_lore = lore_info['y_pred_bb']
            y_pred_cc_lore = lore_info['y_pred_cc']
            fidelity_acc_lore = accuracy_score(y_pred_bb_lore, y_pred_cc_lore)
            fidelity_f1_lore = f1_score(y_pred_bb_lore, y_pred_cc_lore)

            lrule = lore_explanation[0][1]
            # print(lrule)
            covered_lore = lore.get_covered(lrule, dfX2E, dataset)
            coverage_lore = len(covered_lore) / len(dfX2E)
            precision_lore = [1 - eval(v, cc_outcome_lore) for v in y2E[covered_lore]]
            covered_Z_lore = lore.get_covered(lrule, lore_info['dfZ'].to_dict('records'), dataset)
            coverage_Z_lore = len(covered_Z_lore) / len(lore_info['dfZ'])
            # print(coverage_lore)
            # print(covered_Z_lore)
            # print(coverage_Z_lore)

            if coverage_lore > 0.0 and coverage_Z_lore > 0.0:
                break

        except Exception:
            pass

        if attempt >= 5:
            break

        attempt += 1


    print(datetime.datetime.now(), '\tAnchor')
    attempt = 0
    while True:
        try:
            # Explanation with Anchor
            anchor_explanation, anchor_info = anchor_explainer.explain_instance(X2E[idx_record2explain].reshape(1, -1),
                                                                                blackbox.predict, threshold=0.95)

            Zanchor = anchor_info['state']['raw_data']
            y_pred_bb_anchor = blackbox.predict(Zanchor)
            y_pred_cc_anchor = blackbox.predict(Zanchor)
            fidelity_acc_anchor = accuracy_score(y_pred_bb_anchor, y_pred_cc_anchor)
            fidelity_f1_anchor = f1_score(y_pred_bb_anchor, y_pred_cc_anchor)

            arule = anchor2arule(anchor_explanation)
            # print(arule)

            covered_anchor = lore.get_covered(arule, dfX2E, dataset)
            coverage_anchor = len(covered_anchor) / len(dfX2E)
            if len(covered_anchor) > 0:
                if isinstance(y2E[0], str):
                    cc_outcome_anchor = mode(y2E[covered_anchor])
                else:
                    cc_outcome_anchor = int(np.round(y2E[covered_anchor].mean()))
            else:
                cc_outcome_anchor = bb_outcome

            # print(cc_outcome_anchor, type(cc_outcome_anchor))
            individual_hit_anchor = hit_outcome(bb_outcome, cc_outcome_anchor)
            precision_anchor = [1 - eval(v, cc_outcome_anchor) for v in y2E[covered_anchor]]

            dfZanchor = build_df2explain(blackbox, Zanchor, dataset).to_dict('records')[:1000]
            covered_Z_anchor = lore.get_covered(arule, dfZanchor, dataset)
            coverage_Z_anchor = len(covered_Z_anchor) / len(Zanchor)

        except Exception:
            pass

        if attempt >= 5:
            break

        attempt += 1

    res = '%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f' % (
        individual_hit_lore, fidelity_acc_lore, fidelity_f1_lore,
        coverage_lore, np.mean(precision_lore), coverage_Z_lore,
        individual_hit_anchor, fidelity_acc_anchor, fidelity_f1_anchor,
        coverage_anchor, np.mean(precision_anchor), coverage_Z_anchor,
    )
    return res


def main():

    start_index = -1

    path = './'
    path_data = path + 'datasets/'
    path_exp = path + 'experiments/'

    # dataset_name = 'german_credit.csv'
    # dataset = prepare_german_dataset(dataset_name, path_data)

    # blackbox = RandomForestClassifier(n_estimators=20)
    # blackbox.fit(X_train, y_train)

    datsets_list = {
        'german': ('german_credit.csv', prepare_german_dataset),
        # 'adult': ('adult.csv', prepare_adult_dataset),
        # 'compas': ('compas-scores-two-years.csv', prepare_compass_dataset)
    }

    blackbox_list = {
        'svm': LinearSVC,
        # 'dt': DecisionTreeClassifier,
        # 'nn': MLPClassifier,
        # 'rf': RandomForestClassifier,
        # 'lr': LogisticRegression,
    }

    d = list(datsets_list.keys())[0]
    b = list(blackbox_list.keys())[0]

    experiments_results = open(path_exp + 'lore_vs_anchor_coverage_precision_%s_%s.csv' % (d, b), 'a')

    for dataset_kw in datsets_list:
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]

        dataset = prepare_dataset_fn(dataset_name, path_data)
        X, y = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        X2E = X_test
        anchor_explainer = fit_anchor(dataset, X_train, X_test, y_train, y_test, X2E)

        for blackbox_name in blackbox_list:
            BlackBoxConstructor = blackbox_list[blackbox_name]
            if blackbox_name == 'nn':
                blackbox = BlackBoxConstructor(solver='lbfgs')
            else:
                blackbox = BlackBoxConstructor()

            blackbox.fit(X_train, y_train)
            y2E = blackbox.predict(X2E)
            y2E = np.asarray([dataset['label_encoder'][dataset['class_name']].classes_[i] for i in y2E])

            for idx_record2explain in range(len(X2E)):
                if idx_record2explain <= start_index:
                    continue
                print(datetime.datetime.now(), '%d - %.2f' % (idx_record2explain, idx_record2explain / len(X2E)))
                res = run_experiment(blackbox, X2E, y2E, idx_record2explain, dataset, anchor_explainer,
                                     path_data, verbose=False)
                res = '%d,%s,%s,%s\n' % (idx_record2explain, dataset_kw, blackbox_name, res)
                # print(res)
                experiments_results.write(res)
                experiments_results.flush()
                # break

    experiments_results.close()


if __name__ == "__main__":
    main()
