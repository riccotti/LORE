import pyyadt
import networkx as nx

from util import *

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def hit_outcome(bb_outcome, cc_outcome):
    return 1 if bb_outcome == cc_outcome else 0


def counterfactual_fairness(y, diff_outcome):
    return 1.0 * list(y).count(diff_outcome) / len(y)


def evaluate_explanation(x, bb, dfZ, dt, tree_path, leafnode_records, bb_outcome, cc_outcome,
                          y_pred_bb, y_pred_cc, diff_outcome, dataset, counterfactuals):

    class_name = dataset['class_name']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']

    individual_hit = hit_outcome(bb_outcome, cc_outcome)
    local_mimic_acc = accuracy_score(y_pred_bb, y_pred_cc)
    local_mimic_f1 = f1_score(y_pred_bb, y_pred_cc)

    records_index = [i for i, l in enumerate(leafnode_records) if l == tree_path[-1]]
    y_4lf_bb = y_pred_bb[records_index]
    y_4lf_cc = y_pred_cc[records_index]
    local_fairness_acc = accuracy_score(y_4lf_bb, y_4lf_cc)
    local_fairness_f1 = f1_score(y_4lf_bb, y_4lf_cc, pos_label=bb_outcome)

    explanation_size = len(tree_path) - 1
    local_logic_size = len(nx.dag_longest_path(dt))

    if len(records_index) > 0 and len(counterfactuals):
        counterfactual_size = int(np.round(np.mean([len(cf) for cf in counterfactuals])))
        cf_eval = evaluate_counterfactuals(counterfactuals, bb, x, dfZ, records_index, diff_outcome,
                                           class_name, discrete, continuous, features_type)
        local_counterfactual_fairness = cf_eval[0]
        individual_counterfactual_hit = cf_eval[1]
    else:
        counterfactual_size = 0.0
        local_counterfactual_fairness = 0.0
        individual_counterfactual_hit = 0.0

    eval_str = '%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%.6f' % (
        individual_hit, local_mimic_acc, local_mimic_f1, local_fairness_acc, local_fairness_f1,
        explanation_size, counterfactual_size, local_logic_size,
        local_counterfactual_fairness, individual_counterfactual_hit)
    return eval_str


def evaluate_counterfactuals(counterfactuals, bb, x, dfZ, records_index, diff_outcome,
                             class_name, discrete, continuous, features_type):
    lcf_list = list()
    ich_list = list()

    for delta in counterfactuals:

        xcf = pyyadt.apply_counterfactual(x, delta, continuous, discrete, features_type)
        dfx4cf = pd.DataFrame([xcf])
        dfx4cf, _ = label_encode(dfx4cf, discrete, None)
        x4cf = dfx4cf.iloc[:, dfx4cf.columns != class_name].values
        y_x4cf_bb_outcome = bb.predict(x4cf)[0]

        df4cf = dfZ.iloc[records_index]
        Zcf = [pyyadt.apply_counterfactual(z, delta, continuous, discrete, features_type)
               for z in df4cf.to_dict('records')]
        df4cf = pd.DataFrame(Zcf)
        df4cf, _ = label_encode(df4cf, discrete, None)
        X4cf = df4cf.iloc[:, df4cf.columns != class_name].values
        y_4cf_bb = bb.predict(X4cf)

        lcf_list.append(counterfactual_fairness(y_4cf_bb, diff_outcome))
        ich_list.append(hit_outcome(y_x4cf_bb_outcome, diff_outcome))

    return np.mean(lcf_list), np.mean(ich_list)
