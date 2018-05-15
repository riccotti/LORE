import lore

from prepare_dataset import *
from neighbor_generator import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def main():

    dataset_name = 'german_credit.csv'
    path_data = '/Users/riccardo/Documents/PhD/OpenTheBlackBox/code/LORE/datasets/'
    dataset = prepare_german_dataset(dataset_name, path_data)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    blackbox = RandomForestClassifier(n_estimators=20)
    blackbox.fit(X_train, y_train)

    X2E = X_test
    idx_record2explain = 0

    explanation, infos = lore.explain(idx_record2explain, X2E, dataset, blackbox,
                                      ng_function=genetic_neighborhood,
                                      discrete_use_probabilities=True,
                                      continuous_function_estimation=False,
                                      returns_infos=True,
                                      path=path_data, sep=';', log=False)

    dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records')
    dfx = dfX2E[idx_record2explain]
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]

    print('x = %s' % dfx)
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)

    covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
    print(len(covered))
    print(covered)

    y2E = blackbox.predict(X2E)
    precision = [1-np.abs(v-explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
    print(precision)
    print(np.mean(precision), np.std(precision))


if __name__ == "__main__":
    main()
