from util import *
from distance_functions import *
from neighbor_generator import generate_random_data

from sys import maxint
from scipy.optimize import least_squares


def apply_counterfactual(x, delta, continuous, discrete):
    xcf = cPickle.loads(cPickle.dumps(x))

    for att, val in delta.items():
        new_val = None

        if att in continuous:
            delta = val
            new_val = xcf[att] - delta
        if att in discrete:
            new_val = val

        xcf[att] = new_val

    return xcf


def get_random_counterfactual(dfx, blackbox, diff_outcome, X2E, class_name, columns, discrete, continuous, features_type,
                              label_encoder, mad=None, max_iter=1000, tot_max_iter=100000):

    columns_no_class = list(columns)
    columns_no_class.remove(class_name)
    y1 = diff_outcome

    best_x1 = None
    min_dist = np.inf
    count = 0
    count_tot = 0

    if mad:
        def distance_function(x0, x1):
            return mad_distance(x0, x1, mad)
    else:
        def distance_function(x0, x1):
            return mixed_distance(x0, x1, discrete, continuous, class_name,
                                  ddist=simple_match_distance,
                                  cdist=normalized_euclidean_distance)
    while True:
        count_tot += 1
        x1 = np.array(generate_random_data(X2E, class_name, columns, discrete, continuous, features_type,
                                           size=1, uniform=True)[-1])
        df_x1 = pd.DataFrame(data=x1.reshape(1, -1), columns=columns_no_class).to_dict('records')[0]
        fwx1 = blackbox.predict(x1.reshape(1, -1))[0]
        if fwx1 == y1:
            count += 1
            if mad:
                dist = distance_function(dfx.values(), df_x1.values())
            else:
                dist = distance_function(dfx, df_x1)
            if dist < min_dist:
                min_dist = dist
                best_x1 = x1

        if count == max_iter or count_tot == tot_max_iter:
            break

    medoid = pd.DataFrame(data=best_x1.reshape(1, -1), columns=columns_no_class)
    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)
    medoid = label_decode(medoid, discrete_no_class, label_encoder)
    medoid = medoid.to_dict('records')[0]

    counterfactuals = list()
    counterfactual = dict()
    for att, val in medoid.items():
        if att == class_name:
            continue
        if att in discrete:
            if dfx[att] != val:
                counterfactual[att] = val
        elif att in continuous:
            if dfx[att] - val != 0.0:
                counterfactual[att] = dfx[att] - val
    counterfactuals.append(counterfactual)

    return counterfactuals


def fun_mixed(x1_lambda, y1, df_x, blackbox, discrete, continuous, class_name, columns_no_class):
    x1, lambdav = x1_lambda[:-1], x1_lambda[-1]
    df_x1 = pd.DataFrame(data=x1.reshape(1, -1), columns=columns_no_class).to_dict('records')[0]
    d = mixed_distance(df_x, df_x1, discrete, continuous, class_name,
                       ddist=simple_match_distance, cdist=normalized_euclidean_distance)
    fwx1 = blackbox.predict(x1.reshape(1, -1))[0]
    return lambdav*(fwx1-y1)**2 + d


def fun_mad(x1_lambda, y1, df_x, blackbox, columns_no_class, mad):
    x1, lambdav = x1_lambda[:-1], x1_lambda[-1]
    df_x1 = pd.DataFrame(data=x1.reshape(1, -1), columns=columns_no_class).to_dict('records')[0]
    d = mad_distance(df_x.values(), df_x1.values(), mad)
    fwx1 = blackbox.predict(x1.reshape(1, -1))[0]
    return lambdav*(fwx1-y1)**2 + d


def get_stochastic_counterfactual(dfx, blackbox, X2E, diff_outcome, class_name, columns, discrete, continuous,
                                  features_type, label_encoder, mad=None, max_iter=1000):

    columns_no_class = list(columns)
    columns_no_class.remove(class_name)
    y1 = diff_outcome

    min_bounds = np.append(np.min(X2E, axis=0), 0)
    max_bounds = np.append(np.max(X2E, axis=0), maxint)
    bounds = (min_bounds, max_bounds)

    niter = 0
    while True:
        x1 = np.array(generate_random_data(X2E, class_name, columns, discrete, continuous, features_type,
                                           size=1, uniform=True)[-1])
        x1_lambdav = np.append(x1, [np.random.randint(min_bounds[-1], max_bounds[-1])])

        fun = fun_mad if mad else fun_mixed

        d0 = fun(x1_lambdav, y1, dfx, blackbox, columns_no_class, mad) if mad else fun(
            x1_lambdav, y1, dfx, blackbox, discrete, continuous, class_name, columns_no_class)

        try:
            if mad:
                res = least_squares(fun, x1_lambdav, bounds=bounds, args=(y1, dfx, blackbox, columns_no_class, mad,),
                                    xtol=1e-4, max_nfev=1000)
            else:
                res = least_squares(fun, x1_lambdav, bounds=bounds, args=(
                    y1, dfx, blackbox, discrete, continuous, class_name, columns_no_class,),
                                xtol=1e-4, max_nfev=1000)
        except ValueError:
            continue

        d = fun(res.x, y1, dfx, blackbox, columns_no_class, mad) if mad else fun(res.x, y1, dfx, blackbox,
                                                discrete, continuous, class_name, columns_no_class)
        niter += 1

        if blackbox.predict(res.x[:-1].reshape(1, -1))[0] == y1 and d < d0:
            break

        if niter >= max_iter:
            break

    medoid = pd.DataFrame(data=res.x[:-1].astype(int).reshape(1, -1), columns=columns_no_class)
    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)
    medoid = label_decode(medoid, discrete_no_class, label_encoder)
    medoid = medoid.to_dict('records')[0]

    counterfactuals = list()
    counterfactual = dict()
    for att, val in medoid.items():
        if att == class_name:
            continue
        if att in discrete:
            if dfx[att] != val:
                counterfactual[att] = val
        elif att in continuous:
            if dfx[att] - val != 0.0:
                counterfactual[att] = dfx[att] - val
    counterfactuals.append(counterfactual)

    return counterfactuals
