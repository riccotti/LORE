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

    nbr_run = 3

    print(datetime.datetime.now(), '\tLORE')

    features_lore = list()
    features_values_lore = list()
    nbr_features_lore = list()

    for k in range(nbr_run):
        print('%d, ' % k, end='')
        attempt = 0
        while True:
            # try:
                # Explanation with LORE
                lore_explanation, lore_info = lore.explain(idx_record2explain, X2E, dataset, blackbox,
                                                           ng_function=genetic_neighborhood, discrete_use_probabilities=True,
                                                           continuous_function_estimation=False,
                                                           returns_infos=True, path=path_data, sep=';', log=verbose)

                lrule = lore_explanation[0][1]
                features_lore.append(list(lrule.keys()))
                features_values_lore.append(lrule)
                nbr_features_lore.append(len(list(lrule.keys())))

            # except Exception:
            #     pass
            #
            # if attempt >= 3:
            #     break
            #
            # attempt += 1
    print('')

    print(datetime.datetime.now(), '\tAnchor')

    features_anchor = list()
    features_values_anchor = list()
    nbr_features_anchor = list()

    for k in range(nbr_run):
        print('%d, ' % k, end='')
        attempt = 0
        while True:
            try:
                # Explanation with Anchor
                anchor_explanation, anchor_info = anchor_explainer.explain_instance(X2E[idx_record2explain].reshape(1, -1),
                                                                                    blackbox.predict, threshold=0.95)

                arule = anchor2arule(anchor_explanation)
                features_anchor.append(list(arule.keys()))
                features_values_anchor.append(arule)
                nbr_features_anchor.append(len(list(arule.keys())))

            except Exception:
                pass

            if attempt >= 3:
                break

            attempt += 1
    print('')

    jaccard_features_lore = list()
    same_features_values_lore = list()
    deviation_nbr_features_lore = list()

    jaccard_features_anchor = list()
    same_features_values_anchor = list()
    deviation_nbr_features_anchor = list()

    # print(len(features_lore))
    # print(features_lore)

    # print(len(features_anchor))
    # print(features_anchor)

    for i1 in range(0, 10):
        for i2 in range(i1, 10):
            if len(features_lore) > i2:
                jl = len(set(features_lore[i1]) & set(features_lore[i2])) / len(set(features_lore[i1]) | set(features_lore[i2]))
                sl = 1 if features_values_lore[i1] == features_values_lore[i2] else 0
                dl = np.abs(nbr_features_lore[i1] - nbr_features_lore[i2])
                # print(jl,sl,dl)
                jaccard_features_lore.append(jl)
                same_features_values_lore.append(sl)
                deviation_nbr_features_lore.append(dl)
            if len(features_anchor) > i2:
                ja = len(set(features_anchor[i1]) & set(features_anchor[i2])) / len(set(features_anchor[i1]) | set(features_anchor[i2]))
                sa = 1 if features_values_anchor[i1] == features_values_anchor[i2] else 0
                da = np.abs(nbr_features_anchor[i1] - nbr_features_anchor[i2])
                # print(ja, sa, da)
                jaccard_features_anchor.append(ja)
                same_features_values_anchor.append(sa)
                deviation_nbr_features_anchor.append(da)

    res = '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f' % (
        np.mean(jaccard_features_lore), np.std(jaccard_features_lore),
        np.mean(same_features_values_lore), np.std(same_features_values_lore),
        np.mean(deviation_nbr_features_lore), np.std(deviation_nbr_features_lore),
        np.mean(jaccard_features_anchor), np.std(jaccard_features_anchor),
        np.mean(same_features_values_anchor), np.std(same_features_values_anchor),
        np.mean(deviation_nbr_features_anchor), np.std(deviation_nbr_features_anchor),
    )
    return res


def main():

    start_index = 0

    path = '/Users/riccardo/Documents/PhD/OpenTheBlackBox/code/LORE/'
    path_data = path + 'datasets/'
    path_exp = path + 'experiments/'

    # dataset_name = 'german_credit.csv'
    # dataset = prepare_german_dataset(dataset_name, path_data)

    # blackbox = RandomForestClassifier(n_estimators=20)
    # blackbox.fit(X_train, y_train)

    datsets_list = {
        # 'german': ('german_credit.csv', prepare_german_dataset),
        # 'adult': ('adult.csv', prepare_adult_dataset),
        'compas': ('compas-scores-two-years.csv', prepare_compass_dataset)
    }

    blackbox_list = {
        # 'svm': LinearSVC,
        # 'dt': DecisionTreeClassifier,
        # 'nn': MLPClassifier,
        'rf': RandomForestClassifier,
        # 'lr': LogisticRegression,
    }

    d = list(datsets_list.keys())[0]
    b = list(blackbox_list.keys())[0]

    experiments_results = open(path_exp + 'lore_vs_anchor_stability_%s_%s.csv' % (d, b), 'a')

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
                                     path_data, verbose=True)
                res = '%d,%s,%s,%s\n' % (idx_record2explain, dataset_kw, blackbox_name, res)
                # print(res)
                experiments_results.write(res)
                experiments_results.flush()
                # break

    experiments_results.close()


if __name__ == "__main__":
    main()
