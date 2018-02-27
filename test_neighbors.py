import random

from prepare_dataset import *
from neighbor_generator import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def main():
    random.seed(0)

    neighbor_generator_list = {
        'rd': real_data,
        'crd': closed_real_data,
        'rnd': random_neighborhood,
        'ros': random_oversampling,
        'ris': random_instance_selection,
        'gp': genetic_neighborhood,
    }

    dataset_name = 'german_credit.csv'
    path_data = './datasets/'
    dataset = prepare_german_dataset(dataset_name, path_data)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    blackbox = RandomForestClassifier(n_estimators=20)
    blackbox.fit(X_train, y_train)

    X2E = X_test

    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']

    dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous,
                                                         discrete_use_probabilities=False,
                                                         continuous_function_estimation=False)

    idx_record2explain = 1
    dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)

    for ng_name, ng_function in neighbor_generator_list.iteritems():

        print 'Neighbor generator: %s' % ng_name
        dfZ, Z = ng_function(dfZ, x, blackbox, dataset)

        class0_count = dfZ[class_name].value_counts().get(0, 0.0)
        class1_count = dfZ[class_name].value_counts().get(1, 0.0)

        print '\tneighbor size: %d' % len(dfZ)
        print '\tclass 0: %.2f (%d)' % (1.0 * class0_count / len(dfZ), class0_count)
        print '\tclass 1: %.2f (%d)' % (1.0 * class1_count / len(dfZ), class1_count)


if __name__ == "__main__":
    main()
