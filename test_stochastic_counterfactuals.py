


from prepare_dataset import *
from neighbor_generator import *
from stochastic_counterfactuals import *

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
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']
    possible_outcomes = dataset['possible_outcomes']

    dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)
    bb_outcome = blackbox.predict(x.reshape(1, -1))[0]
    diff_outcome = get_diff_outcome(bb_outcome, possible_outcomes)
    diff_outcome = label_encoder[class_name].transform(np.array([diff_outcome]))[0]
    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset)  #.to_dict('records')[0]

    dfx, _ = label_encode(dfx, discrete, label_encoder)
    dfx = dfx.to_dict('records')[0]

    mad = list()
    for i in range(len(X2E[0])):
        median = np.median(X2E[:, i])
        mad.append(np.median([np.abs(x0 - median) for x0 in X2E[:, i]]))

    rnd_cfs_mxd = get_random_counterfactual(dfx, blackbox, diff_outcome, X2E, class_name, columns, discrete, continuous,
                                            features_type, label_encoder, mad=None, max_iter=100, tot_max_iter=10000)

    print rnd_cfs_mxd
    print apply_counterfactual(dfx, rnd_cfs_mxd[0], continuous, discrete)
    print '--------'

    rnd_cfs_mad = get_random_counterfactual(dfx, blackbox, diff_outcome, X2E, class_name, columns, discrete, continuous,
                                            features_type, label_encoder, mad=mad, max_iter=100, tot_max_iter=10000)

    print rnd_cfs_mad
    print apply_counterfactual(dfx, rnd_cfs_mad[0], continuous, discrete)
    print '--------'

    stc_cfs_mxd = get_stochastic_counterfactual(dfx, blackbox, X2E, diff_outcome, class_name, columns, discrete,
                                                continuous, features_type, label_encoder, mad=None, max_iter=1000)
    print stc_cfs_mxd
    print apply_counterfactual(dfx, stc_cfs_mxd[0], continuous, discrete)
    print '--------'

    stc_cfs_mad = get_stochastic_counterfactual(dfx, blackbox, X2E, diff_outcome, class_name, columns, discrete,
                                                continuous, features_type, label_encoder, mad=mad, max_iter=1000)
    print stc_cfs_mad
    print apply_counterfactual(dfx, stc_cfs_mad[0], continuous, discrete)
    print '--------'


if __name__ == "__main__":
    main()
