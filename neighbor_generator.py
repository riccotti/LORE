from gpdatagenerator import *
from distance_functions import *

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import CondensedNearestNeighbour


def genetic_neighborhood_old(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']
    feature_values = dataset['feature_values']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    Z = generate_data(x, feature_values, blackbox, discrete_no_class, continuous, class_name, idx_features,
                      distance_function, neigtype={'ss': 0.5, 'sd': 0.5}, population_size=1000, halloffame_ratio=0.1,
                      alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10)
    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z


def genetic_neighborhood(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']
    feature_values = dataset['feature_values']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    Z = generate_data(x, feature_values, blackbox, discrete_no_class, continuous, class_name, idx_features,
                      distance_function, neigtype={'ss': 0.5, 'sd': 0.5}, population_size=1000, halloffame_ratio=0.1,
                      alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10)

    zy = blackbox.predict(Z)
    # print(np.unique(zy, return_counts=True))
    if len(np.unique(zy)) == 1:
        # print('qui')
        label_encoder = dataset['label_encoder']
        dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
        neig_indexes = get_closest_diffoutcome(dfZ, dfx, discrete, continuous, class_name,
                                               blackbox, label_encoder, distance_function, k=100)
        Zn, _ = label_encode(dfZ, discrete, label_encoder)
        Zn = Zn.iloc[neig_indexes, Z.columns != class_name].values
        Z = np.concatenate((Z, Zn), axis=0)

    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z


def real_data(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']

    dfZ = dfZ
    Z, _ = label_encode(dfZ, discrete, label_encoder)
    Z = Z.iloc[:, Z.columns != class_name].values
    dfZ = build_df2explain(blackbox, Z, dataset)

    return dfZ, Z


def closed_real_data(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']
    continuous = dataset['continuous']

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
    neig_indexes = get_closest_diffoutcome(dfZ, dfx, discrete, continuous, class_name,
                                           blackbox, label_encoder, distance_function, k=100)

    dfZ = dfZ
    Z, _ = label_encode(dfZ, discrete, label_encoder)
    Z = Z.iloc[neig_indexes, Z.columns != class_name].values
    dfZ = build_df2explain(blackbox, Z, dataset)

    return dfZ, Z


def random_neighborhood(dfZ, x, blackbox, dataset, stratified=True):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']
    columns = dataset['columns']
    features_type = dataset['features_type']

    if stratified:

        def distance_function(x0, x1, discrete, continuous, class_name):
            return mixed_distance(x0, x1, discrete, continuous, class_name,
                                  ddist=simple_match_distance,
                                  cdist=normalized_euclidean_distance)

        dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
        neig_indexes = get_closest_diffoutcome(dfZ, dfx, discrete, continuous, class_name,
                                               blackbox, label_encoder, distance_function, k=100)

        Z, _ = label_encode(dfZ, discrete, label_encoder)
        Z = Z.iloc[neig_indexes, Z.columns != class_name].values
        Z = generate_random_data(Z, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True)
        dfZ = build_df2explain(blackbox, Z, dataset)

        return dfZ, Z

    else:

        Z, _ = label_encode(dfZ, discrete, label_encoder)
        Z = Z.iloc[:, Z.columns != class_name].values
        Z = generate_random_data(Z, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True)
        dfZ = build_df2explain(blackbox, Z, dataset)

        return dfZ, Z


def generate_random_data(X, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True):
    if isinstance(X, pd.DataFrame):
        X = X.values
    X1 = list()
    columns1 = list(columns)
    columns1.remove(class_name)
    for i, col in enumerate(columns1):
        values = X[:, i]
        diff_values = np.unique(values)
        prob_values = [1.0 * list(values).count(val) / len(values) for val in diff_values]
        if col in discrete:
            if uniform:
                new_values = np.random.choice(diff_values, size)
            else:
                new_values = np.random.choice(diff_values, size, prob_values)
        elif col in continuous:
            mu = np.mean(values)
            sigma = np.std(values)
            if sigma <= 0.0:
                new_values = np.array([values[0]] * size)
            else:
                new_values = np.random.normal(mu, sigma, size)
        if features_type[col] == 'integer':
            new_values = new_values.astype(int)
        X1.append(new_values)
    X1 = np.concatenate((X, np.column_stack(X1)), axis=0).tolist()
    if isinstance(X, pd.DataFrame):
        X1 = pd.DataFrame(data=X1, columns=columns1)
    return X1


def random_oversampling(dfZ, x, blackbox, dataset):
    dfZ, Z = random_neighborhood(dfZ, x, blackbox, dataset)
    y = blackbox.predict(Z)

    oversampler = RandomOverSampler()
    Z, _ = oversampler.fit_sample(Z, y)
    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z


def random_instance_selection(dfZ, x, blackbox, dataset):
    dfZ1, Z = random_neighborhood(dfZ, x, blackbox, dataset)
    y = blackbox.predict(Z)

    cnn = CondensedNearestNeighbour(return_indices=True)
    Z, _, _ = cnn.fit_sample(Z, y)
    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z

