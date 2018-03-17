import pyyadt
import random

from neighbor_generator import *
from gpdatagenerator import calculate_feature_values


def explain(idx_record2explain, X2E, dataset, blackbox,
            ng_function=genetic_neighborhood,
            discrete_use_probabilities=False,
            continuous_function_estimation=False,
            returns_infos=False):

    random.seed(0)
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']
    possible_outcomes = dataset['possible_outcomes']

    # Dataset Preprocessing
    dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous,
                                                        discrete_use_probabilities=discrete_use_probabilities,
                                                        continuous_function_estimation=continuous_function_estimation)

    dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)

    # Generate Neighborhood
    dfZ, Z = ng_function(dfZ, x, blackbox, dataset)

    # Build Decision Tree
    dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous,
                            filename=dataset['name'], path='./', sep=';', log=False)

    # Apply Black Box and Decision Tree on instance to explain
    bb_outcome = blackbox.predict(x.reshape(1, -1))[0]

    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
    cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

    # Apply Black Box and Decision Tree on neighborhood
    y_pred_bb = blackbox.predict(Z)
    y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
                                           discrete, continuous)

    # Update labels if necessary
    if class_name in label_encoder:
        cc_outcome = label_encoder[class_name].transform(np.array([cc_outcome]))[0]

    if class_name in label_encoder:
        y_pred_cc = label_encoder[class_name].transform(y_pred_cc)

    # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(bb_outcome, possible_outcomes)
    counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                 class_name, continuous, features_type)

    explanation = (rule, counterfactuals)

    infos = {
        'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        'y_pred_bb': y_pred_bb,
        'y_pred_cc': y_pred_cc,
        'dfZ': dfZ,
        'Z': Z,
        'dt': dt,
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        'diff_outcome': diff_outcome
    }

    if returns_infos:
        return explanation, infos

    return explanation

