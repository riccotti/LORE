import os
import re
import pydotplus
import subprocess

import numpy as np
import pandas as pd
import networkx as nx

from util import *

from collections import defaultdict


def fit(df, class_name, columns, features_type, discrete, continuous,
        filename='yadt_dataset', path='./', sep=';', log=False):
    
    data_filename = path + filename + '.data'
    names_filename = path + filename + '.names'
    tree_filename = path + filename + '.dot'
    
    df.to_csv(data_filename, sep=sep, header=False, index=False)
    
    names_file = open(names_filename, 'w')
    for col in columns:
        col_type = features_type[col]
        disc_cont = 'discrete' if col in discrete else 'continuous'
        disc_cont = 'class' if col == class_name else disc_cont 
        names_file.write('%s%s%s%s%s\n' % (col, sep, col_type, sep, disc_cont))
    names_file.close()
    
    cmd = 'yadt/dTcmd -fd %s -fm %s -sep %s -d %s' % (
        data_filename, names_filename, sep, tree_filename)
    output = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
    # cmd = r"dTcmd -fd %s -fm %s -sep '%s' -d %s" % (
    #     data_filename, names_filename, sep, tree_filename)
    # cmd = r'noah "%s"' % cmd
    # print(cmd)
    # output = subprocess.check_output(['noah', "%s" % cmd], stderr=subprocess.STDOUT)
    if log:
        print(cmd)
        print(output)

    dt = nx.DiGraph(nx.drawing.nx_pydot.read_dot(tree_filename))
    dt_dot = pydotplus.graph_from_dot_data(open(tree_filename, 'r').read())
    
    if os.path.exists(data_filename):
        os.remove(data_filename)
        
    if os.path.exists(names_filename):
        os.remove(names_filename)
        
    if os.path.exists(tree_filename):
        os.remove(tree_filename)
        
    return dt, dt_dot


def predict(dt, X, class_name, features_type, discrete, continuous, leafnode=True):
    edge_labels = get_edge_labels(dt)
    node_labels = get_node_labels(dt)
    node_isleaf = {k: v == 'ellipse' for k, v in nx.get_node_attributes(dt, 'shape').items()}
    
    y_list = list()
    lf_list = list()
    for x in X:
        y, tp = predict_single_record(dt, x, class_name, edge_labels, node_labels, node_isleaf,
                                      features_type, discrete, continuous)
        if y is None:
            continue
        y_list.append(y)
        lf_list.append(tp[-1])
    
    if leafnode:
        return np.array(y_list), lf_list
    
    return np.array(y_list)


def get_node_labels(dt):
    return {k: v.replace('"', '').replace('\\n', '') for k, v in nx.get_node_attributes(dt, 'label').items()}


def get_edge_labels(dt):    
    return {k: v.replace('"', '').replace('\\n', '') for k, v in nx.get_edge_attributes(dt, 'label').items()}
    
    
def predict_single_record(dt, x, class_name, edge_labels, node_labels, node_isleaf, features_type, discrete, continuous,
                          n_iter=1000):
    root = 'n0'
    node = root
    tree_path = list()
    count = 0
    while not node_isleaf[node]:
        att = node_labels[node]
        val = x[att]
        for child in dt.neighbors(node):
            count += 1
            edge_val = edge_labels[(node, child)]
            if att in discrete:
                val = val.strip() if isinstance(val, str) else val     
                if yadt_value2type(edge_val, att, features_type) == val:
                    tree_path.append(node)
                    node = child
                    break
            else:
                pyval = yadt_value2type(val, att, features_type)
                if '>' in edge_val:
                    thr = yadt_value2type(edge_val.replace('>', ''), att, features_type)
                    if pyval > thr:
                        tree_path.append(node)
                        node = child
                        break
                elif '<=' in edge_val:
                    thr = yadt_value2type(edge_val.replace('<=', ''), att, features_type)
                    if pyval <= thr:
                        tree_path.append(node)
                        node = child
                        break
        if count >= n_iter:
            print('Loop in Yadt prediction')
            return None, None
        count += 1

    tree_path.append(node)
    
    outcome = node_labels[node].split('(')[0]
    outcome = yadt_value2type(outcome, class_name, features_type)
        
    return outcome, tree_path
    

def predict_rule(dt, x, class_name, features_type, discrete, continuous):
    edge_labels = get_edge_labels(dt)
    node_labels = get_node_labels(dt)
    node_isleaf = {k: v == 'ellipse' for k, v in nx.get_node_attributes(dt, 'shape').items()}

    y, tree_path = predict_single_record(dt, x, class_name, edge_labels, node_labels, node_isleaf, 
                                         features_type, discrete, continuous)
    if y is None:
        return None, None, None
    
    rule = get_rule(tree_path, class_name, y, node_labels, edge_labels)
    
    return y, rule, tree_path


def get_covered_record_index(tree_path, leaf_nodes):
    return [i for i, l in enumerate(leaf_nodes) if l == tree_path[-1]]


def get_rule(tree_path, class_name, y, node_labels=None, edge_labels=None, dt=None):
    
    if node_labels is None:
        node_labels = get_node_labels(dt)
    
    if edge_labels is None:
        edge_labels = get_edge_labels(dt)

    ant = dict()
    for i in range(0, len(tree_path)-1):
        node = tree_path[i]
        child = tree_path[i + 1]
        if (node, child) in edge_labels:
            att = node_labels[node]
            val = edge_labels[(node, child)] 
        else:
            att = node_labels[child]
            val = edge_labels[(child, node)]

        if att in ant:
            val0 = ant[att]
            min_thr0 = None
            max_thr0 = None

            min_thr = None
            max_thr = None

            if len(re.findall('.*<.*<=.*', val0)):
                min_thr0 = float(val0.split('<')[0])
                max_thr0 = float(val0.split('<=')[1])
            elif '<=' in val0:
                max_thr0 = float(val0.split('<=')[1])
            elif '>' in val0:
                min_thr0 = float(val0.split('>')[1])

            if len(re.findall('.*<.*<=.*', val)):
                min_thr = float(val.split('<')[0])
                max_thr = float(val.split('<=')[1])
            elif '<=' in val:
                max_thr = float(val.split('<=')[1])
            elif '>' in val:
                min_thr = float(val.split('>')[1])

            new_min_thr = None
            new_max_thr = None

            if min_thr:
                new_min_thr = max(min_thr, min_thr0) if min_thr0 else min_thr

            if min_thr0:
                new_min_thr = max(min_thr, min_thr0) if min_thr else min_thr0

            if max_thr:
                new_max_thr = min(max_thr, max_thr0) if max_thr0 else max_thr

            if max_thr0:
                new_max_thr = min(max_thr, max_thr0) if max_thr else max_thr0

            if new_min_thr and new_max_thr:
                val = '%s< %s <=%s' % (new_min_thr, att, new_max_thr)
            elif new_min_thr:
                val = '>%s' % new_min_thr
            elif new_max_thr:
                val = '<=%s' % new_max_thr

        ant[att] = val
        
    cons = {class_name: y}
    
    weights = node_labels[tree_path[-1]].split('(')[1]
    weights = weights.replace(')', '')
    weights = [float(w) for w in weights.split('/')]
    
    rule = [cons, ant, weights]
    
    return rule


def yadt_value2type(x, attribute, features_type):

    if features_type[attribute] == 'integer':
        x = int(float(x))
    elif features_type[attribute] == 'double':
        x = float(x)
        
    return x


def get_counterfactuals(dt, tree_path, rule, diff_outcome, class_name, continuous, features_type):
    edge_labels = get_edge_labels(dt)
    node_labels = get_node_labels(dt)
    node_isleaf = {k: v == 'ellipse' for k, v in nx.get_node_attributes(dt, 'shape').items()}

    root = tree_path[0]

    node_diff_outcome_path = list()

    sp_from_root = nx.shortest_path(dt, root)
    for node in sp_from_root:
        if node == root or not node_isleaf[node]:
            continue

        sp_outcome = node_labels[node].split('(')[0]
        sp_outcome = yadt_value2type(sp_outcome, class_name, features_type)

        weights = node_labels[node].split('(')[1]
        weights = weights.replace(')', '')
        weight = [float(w) for w in weights.split('/')][0]

        if weight == 0.0:
            continue

        if sp_outcome == diff_outcome:
            node_diff_outcome_path.append(sp_from_root[node])

    cond = expand_rule(rule, continuous)
    clen = float('inf')
    counterfactuals = list()
    for ctp in node_diff_outcome_path:
        crule = get_rule(ctp, class_name, diff_outcome, node_labels, edge_labels)
        # delta = set(crule[1].items()) - set(rule[1].items())
        ccond = expand_rule(crule, continuous)
        delta, qlen = get_falsifeid_conditions(cond, ccond, continuous)
        if qlen < clen:
            clen = qlen
            counterfactuals = [delta]
        elif qlen == clen:
            counterfactuals.append(delta)

    return counterfactuals


def get_falsifeid_conditions(cond, ccond, continuous):
    # a condition falsified is not respect or not present in the verified conditions

    qlen = 0
    fcond = dict()
    for att, val in ccond.items():
        if att not in cond:
            if att in continuous:
                min_thr, max_thr = ccond[att]
                if min_thr > -np.inf and max_thr < np.inf:
                    val = '%s< %s <=%s' % (min_thr, att, max_thr)
                    qlen += 2
                elif min_thr > -np.inf:
                    val = '>%s' % min_thr
                    qlen += 1
                elif max_thr < np.inf:
                    val = '<=%s' % max_thr
                    qlen += 1
                fcond[att] = val
            else:
                fcond[att] = val
                qlen += 1
            continue

        if att in continuous:
            min_thr_c, max_thr_c = ccond[att]
            min_thr_r, max_thr_r = cond[att]

            if min_thr_c == min_thr_r and max_thr_c == max_thr_r:
                continue

            min_thr = None
            max_thr = None

            if min_thr_r < min_thr_c:
                min_thr = min_thr_c
                if max_thr_c < np.inf:
                    max_thr = max_thr_c

            if max_thr_r > max_thr_c:
                max_thr = max_thr_c
                if min_thr_c > -np.inf:
                    min_thr = min_thr_c

            if min_thr and max_thr:
                val = '%s< %s <=%s' % (min_thr, att, max_thr)
                qlen += 2
            elif min_thr:
                val = '>%s' % min_thr
                qlen += 1
            elif max_thr:
                val = '<=%s' % max_thr
                qlen += 1
            else:
                continue

            fcond[att] = val
        else:
            if val != cond[att]:
                fcond[att] = val
                qlen += 1

    return fcond, qlen


def expand_rule(rule, continuous):
    erule = dict()
    for sc in rule[1]:
        if sc in continuous:
            val = rule[1][sc]
            if len(re.findall('.*<.*<=.*', val)):
                min_thr0 = float(val.split('<')[0])
                max_thr0 = float(val.split('<=')[1])
                # erule[sc] = ['>%s' % min_thr0, '<=%s' % max_thr0]
                erule[sc] = [min_thr0, max_thr0]
            elif '<=' in val:
                max_thr0 = float(val.split('<=')[1])
                # erule[sc] = [-np.inf, '<=%s' % max_thr0]
                erule[sc] = [-np.inf, max_thr0]
            elif '>' in val:
                min_thr0 = float(val.split('>')[1])
                # erule[sc] = ['>%s' % min_thr0, np.inf]
                erule[sc] = [min_thr0, np.inf]
        else:
            erule[sc] = rule[1][sc]

    return erule


def apply_counterfactual(x, delta, continuous, discrete, features_type):
    xcf = cPickle.loads(cPickle.dumps(x))

    for att, val in delta.items():
        new_val = None
        if att in continuous:
            if '>' in val:
                thr = yadt_value2type(val.replace('>', ''), att, features_type)
                new_val = thr + 1
            elif '<=' in val:
                thr = yadt_value2type(val.replace('<=', ''), att, features_type)
                new_val = thr
        if att in discrete:
            new_val = yadt_value2type(val, att, features_type)

        xcf[att] = new_val

    return xcf
