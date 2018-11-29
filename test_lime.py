import lime
import lime.lime_tabular

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
    idx_record2explain = 1

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

    # Create Lime Explanator
    num_features = 5
    explainer = lime.lime_tabular.LimeTabularExplainer(X2E,
                                                       feature_names=feature_names,
                                                       class_names=possible_outcomes,
                                                       categorical_features=idx_discrete_features,
                                                       categorical_names=categorical_names,
                                                       verbose=False
                                                       )

    exp, Zlr, Z, lr = explainer.explain_instance(X2E[idx_record2explain], blackbox.predict_proba,
                                                 num_features=num_features, num_samples=1000)

    used_features_idx = list()
    used_features_importance = list()
    logic_explanation = list()
    for idx, weight in exp.local_exp[1]:
        used_features_idx.append(idx)
        used_features_importance.append(weight)
        logic_explanation.append(exp.domain_mapper.discretized_feature_names[idx])

    for feature, weight in zip(logic_explanation, used_features_importance):
        print(feature, weight)

    # bb_outcome = blackbox.predict(Z[0].reshape(1, -1))[0]
    # cc_outcome = np.round(lr.predict(Zlr[0, used_features_idx].reshape(1, -1))).astype(int)[0]
    #
    # y_pred_bb = blackbox.predict(Z)
    # y_pred_cc = np.round(lr.predict(Zlr[:, used_features_idx])).astype(int)


if __name__ == "__main__":
    main()
