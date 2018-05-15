from anchor import anchor_tabular

from prepare_dataset import *
from neighbor_generator import *

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
    idx_record2explain = 9

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

    print('Prediction: ', possible_outcomes[blackbox.predict(X2E[idx_record2explain].reshape(1, -1))[0]])

    exp, info = explainer.explain_instance(X2E[idx_record2explain].reshape(1, -1), blackbox.predict, threshold=0.95)

    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())

    # Get test examples where the anchora pplies# Get t
    fit_anchor = np.where(np.all(X2E[:, exp.features()] == X2E[idx_record2explain][exp.features()], axis=1))[0]
    print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(X2E.shape[0])))
    print('Anchor test precision: %.2f' % (np.mean(blackbox.predict(X2E[fit_anchor]) ==
                                                   blackbox.predict(X2E[idx_record2explain].reshape(1, -1)))))

    print(blackbox.predict(info['state']['raw_data']))


if __name__ == "__main__":
    main()
