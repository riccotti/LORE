import numpy as np
import pandas as pd
import _pickle as cPickle

from sklearn.preprocessing import LabelEncoder


def recognize_features_type(df, class_name):    
    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'
        
    return type_features, features_type


def set_discrete_continuous(features, type_features, class_name, discrete=None, continuous=None):
    
    if discrete is None and continuous is None:
        discrete = type_features['string']
        continuous = type_features['integer'] + type_features['double']
        
    if discrete is None and continuous is not None:
        discrete = [f for f in features if f not in continuous]
        continuous = list(set(continuous + type_features['integer'] + type_features['double']))
        
    if continuous is None and discrete is not None:
        continuous = [f for f in features if f not in discrete and (f in type_features['integer'] or f in type_features['double'])]
        discrete = list(set(discrete + type_features['string']))
    
    discrete = [f for f in discrete if f != class_name] + [class_name]
    continuous = [f for f in continuous if f != class_name]
    return discrete, continuous


def label_encode(df, columns, label_encoder=None):
    df_le = df.copy(deep=True)
    new_le = label_encoder is None
    label_encoder = dict() if new_le else label_encoder
    for col in columns:
        if new_le:
            le = LabelEncoder()
            df_le[col] = le.fit_transform(df_le[col])
            label_encoder[col] = le
        else:
            le = label_encoder[col]
            df_le[col] = le.transform(df_le[col])
    return df_le, label_encoder


def label_decode(df, columns, label_encoder):
    df_de = df.copy(deep=True)
    for col in columns:
        le = label_encoder[col]
        df_de[col] = le.inverse_transform(df_de[col])
    return df_de


def get_closest(df, x, discrete, continuous, class_name, distance_function, k=100):
    distances = list()
    for z in df.to_dict('records'):
        distances.append(distance_function(x, z, discrete, continuous, class_name))
        
    return np.argsort(distances).tolist()[:k]


def get_closest_diffoutcome(df, x, discrete, continuous, class_name, blackbox, label_encoder, distance_function,
                            k=100, diff_out_ratio=0.1):
    distances = list()
    distances_0 = list()
    idx0 = list()
    distances_1 = list()
    idx1 = list()
    Z, _ = label_encode(df, discrete, label_encoder)
    Z = Z.iloc[:, Z.columns != class_name].values

    idx = 0
    for z, z1 in zip(df.to_dict('records'), Z):
        d = distance_function(x, z, discrete, continuous, class_name)
        distances.append(d)
        if blackbox.predict(z1.reshape(1, -1))[0] == 0:
            distances_0.append(d)
            idx0.append(idx)
        else:
            distances_1.append(d)
            idx1.append(idx)
        idx += 1

    idx0 = np.array(idx0)
    idx1 = np.array(idx1)

    all_indexs = np.argsort(distances).tolist()[:k]
    indexes0 = list(idx0[np.argsort(distances_0).tolist()[:k]])
    indexes1 = list(idx1[np.argsort(distances_1).tolist()[:k]])

    if 1.0 * len(set(all_indexs) & set(indexes0)) / len(all_indexs) < diff_out_ratio:
        k_index = k - int(k * diff_out_ratio)
        final_indexes = all_indexs[:k_index] + indexes0[:int(k * diff_out_ratio)]
    elif 1.0 * len(set(all_indexs) & set(indexes1)) < diff_out_ratio:
        k_index = k - int(k * diff_out_ratio)
        final_indexes = all_indexs[:k_index] + indexes1[:int(k * diff_out_ratio)]
    else:
        final_indexes = all_indexs

    return final_indexes





def generate_artificial_features(size, class_name, columns, features_type, discrete, continuous, ratio=0.25):
    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    num_art_features = int(np.round(ratio * (len(columns) - 1)))
    num_disc_art_features = int(np.round(ratio * len(discrete_no_class)))
    num_cont_art_features = max(0, num_art_features - num_disc_art_features)

    disc_feature_values = dict()
    for i in range(num_disc_art_features):
        name = 'artificial_disc_%d' % i
        num_diff_values = np.random.choice([2, 3, 4, 5, 10])
        values = [j for j in range(num_diff_values)]
        disc_feature_values[name] = values

    cont_feature_fun = dict()
    for i in range(num_cont_art_features):
        name = 'artificial_cont_%d' % i
        fnidx = np.random.choice(np.arange(4))
        fn = [(np.random.chisquare, [1]),
              (np.random.exponential, [1]),
              (np.random.lognormal, [0, 1]),
              (np.random.normal, [0, 1])][fnidx]
        cont_feature_fun[name] = fn

    artificial_data = list()
    new_discrete = list()
    for artificial_feature in disc_feature_values:
        values = np.random.choice(disc_feature_values[artificial_feature], size).astype(int)
        artificial_data.append(values)

        features_type[artificial_feature] = 'integer'
        discrete.append(artificial_feature)
        columns.append(artificial_feature)
        new_discrete.append(artificial_feature)

    new_continuous = list()
    for artificial_feature in cont_feature_fun:
        fn = cont_feature_fun[artificial_feature][0]
        params = cont_feature_fun[artificial_feature][1]
        if len(params) == 1:
            values = fn(params[0], size)
        elif len(params) == 2:
            values = fn(params[0], params[1], size)
        artificial_data.append(values)

        features_type[artificial_feature] = 'double'
        continuous.append(artificial_feature)
        columns.append(artificial_feature)
        new_continuous.append(artificial_feature)

    # AF = {
    #     'AF': np.column_stack(artificial_data).tolist(),
    #     'columns': columns,
    #     'features_type': features_type,
    #     'discrete': discrete,
    #     'continuous': continuous
    # }
    # return AF
    return map(list, map(None, *artificial_data)), new_discrete, new_continuous


def build_df2explain(bb, X, dataset):
    
    columns = dataset['columns']
    features_type = dataset['features_type']
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']
    
    y = bb.predict(X)
    yX = np.concatenate((y.reshape(-1, 1), X), axis=1)
    data = list()
    for i, col in enumerate(columns):
        data_col = yX[:, i]
        data_col = data_col.astype(int) if col in discrete else data_col
        data_col = data_col.astype(int) if features_type[col] == 'integer' else data_col
        data.append(data_col)
    # data = map(list, map(None, *data))
    data = [[d[i] for d in data] for i in range(0, len(data[0]))]
    dfZ = pd.DataFrame(data=data, columns=columns)
    dfZ = label_decode(dfZ, discrete, label_encoder)
    return dfZ


def dataframe2explain(X2E, dataset, idx_record2explain, blackbox):
    # Dataset to explit to perform explanation (typically is the train or test set (real instances))
    Z = cPickle.loads(cPickle.dumps(X2E))

    # Select record to predict and explain
    x = Z[idx_record2explain]

    # Remove record to explain (optional) from dataset Z and convert into dataframe
    # Z = np.delete(Z, idx_record2explain, axis=0)
    dfZ = build_df2explain(blackbox, Z, dataset)
    
    return dfZ, x


def get_diff_outcome(outcome, possible_outcomes):
    return possible_outcomes[1] if outcome == possible_outcomes[0] else possible_outcomes[0]











