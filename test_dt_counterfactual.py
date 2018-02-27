import random
import pyyadt

from prepare_dataset import *
from neighbor_generator import *

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


def main():
    random.seed(0)

    # X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
    # X = StandardScaler().fit_transform(X)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # class_name = 'class'
    # columns = ['class', 'X0', 'X1']
    # df = pd.DataFrame(np.concatenate((y_train.reshape(-1, 1), X_train), axis=1), columns=columns)
    #
    # features_type = {'X0': 'double', 'X1': 'double', 'class': 'string'}
    # discrete = ['class']
    # continuous = ['X0', 'X1']
    # discrete_no_class = list(discrete)
    # discrete_no_class.remove(class_name)
    # possible_outcomes = list(df[class_name].unique())
    # if features_type[class_name] == 'string':
    #     possible_outcomes = [str(po) for po in possible_outcomes]
    # _, label_encoder = label_encode(df, discrete)
    #
    # columns_tmp = list(columns)
    # columns_tmp.remove(class_name)
    # idx_features = {i: col for i, col in enumerate(columns_tmp)}
    #
    # dataset = {
    #     'class_name': class_name,
    #     'columns': columns,
    #     'features_type': features_type,
    #     'discrete': discrete,
    #     'continuous': continuous,
    #     'label_encoder': label_encoder,
    #     'possible_outcomes': possible_outcomes,
    #     'idx_features': idx_features,
    # }

    dataset_name = 'german_credit.csv'
    path_data = './datasets/'
    dataset = prepare_german_dataset(dataset_name, path_data)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    possible_outcomes = dataset['possible_outcomes']
    label_encoder = dataset['label_encoder']

    yX = np.concatenate((y_train.reshape(-1, 1), X_train), axis=1)
    data = list()
    for i, col in enumerate(columns):
        data_col = yX[:, i]
        data_col = data_col.astype(int) if col in discrete else data_col
        data_col = data_col.astype(int) if features_type[col] == 'integer' else data_col
        data.append(data_col)
    data = map(list, map(None, *data))
    dfZ = pd.DataFrame(data=data, columns=columns)
    dfZ = label_decode(dfZ, discrete, label_encoder)

    dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous,
                            filename='pyyadt_test', path='./', sep=';', log=False)

    dt_dot.write_png('pyyadt_test.png')
    # img = Image.open('pyyadt_test.png')
    # img.show()

    # y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
    #                                        discrete, continuous)

    idx_record2explain = 5  # 4
    x = dfZ.to_dict('records')[idx_record2explain]
    print x
    cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, x, class_name, features_type, discrete, continuous)

    print cc_outcome
    for k, v in rule[1].iteritems():
        print k, v
    print tree_path
    print '-------------------'

    diff_outcome = get_diff_outcome(cc_outcome, possible_outcomes)
    counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                 class_name, continuous, features_type)

    print counterfactuals

    for delta in counterfactuals:
        xcf = pyyadt.apply_counterfactual(x, delta, continuous, discrete, features_type)
        ycf, _, _ = pyyadt.predict_rule(dt, xcf, class_name, features_type, discrete, continuous)
        print delta, ycf, xcf


if __name__ == "__main__":
    main()
