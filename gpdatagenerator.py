import math
import warnings
import scipy.stats as st

from util import *
from deap import base, creator, tools, algorithms


def record_init(x):
    return x


def random_init(feature_values):
    individual = list()
    for feature_idx in feature_values:
        values = feature_values[feature_idx]
        val = np.random.choice(values, 1)[0]
        individual.append(val)
    return individual


def cPickle_clone(x):
    return cPickle.loads(cPickle.dumps(x))


def mutate(feature_values, indpb, toolbox, individual):
    new_individual = toolbox.clone(individual)
    for feature_idx in range(0, len(individual)):
        values = feature_values[feature_idx]
        if np.random.random() <= indpb:
            val = np.random.choice(values, 1)[0]
            new_individual[feature_idx] = val
    return new_individual,


def fitness_sso(x0, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # similar_same_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}
    
    # zero if is too similar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio >= eta else sim_ratio
    
    y0 = bb.predict(np.asarray(x0).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x1).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 == y1 else 0.0
    
    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,


def fitness_sdo(x0, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # similar_different_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}

    # zero if is too similar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio >= eta else sim_ratio

    y0 = bb.predict(np.asarray(x0).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x1).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 != y1 else 0.0

    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,


def fitness_dso(x0, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # dissimilar_same_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}

    # zero if is too dissimilar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio <= eta else 1.0 - sim_ratio
    
    y0 = bb.predict(np.asarray(x0).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x1).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 == y1 else 0.0
    
    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,


def fitness_ddo(x0, bb, alpha1, alpha2, eta, discrete, continuous, class_name, idx_features, distance_function, x1):
    # dissimilar_different_outcome
    x0d = {idx_features[i]: val for i, val in enumerate(x0)}
    x1d = {idx_features[i]: val for i, val in enumerate(x1)}
    
    # zero if is too dissimilar
    sim_ratio = 1.0 - distance_function(x0d, x1d, discrete, continuous, class_name)
    record_similarity = 0.0 if sim_ratio <= eta else 1.0 - sim_ratio
    
    y0 = bb.predict(np.asarray(x0).reshape(1, -1))[0]
    y1 = bb.predict(np.asarray(x1).reshape(1, -1))[0]
    target_similarity = 1.0 if y0 != y1 else 0.0
    
    evaluation = alpha1 * record_similarity + alpha2 * target_similarity
    return evaluation,


def setup_toolbox(record, feature_values, bb, init, init_params, evaluate, discrete, continuous, class_name,
                  idx_features, distance_function, population_size=1000, alpha1=0.5, alpha2=0.5, eta=0.3,
                  mutpb=0.2, tournsize=3):

    creator.create("fitness", base.Fitness, weights=(1.0,))
    creator.create("individual", list, fitness=creator.fitness)
    
    toolbox = base.Toolbox()
    toolbox.register("feature_values", init, init_params)
    toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)
    
    toolbox.register("clone", cPickle_clone)
    toolbox.register("evaluate", evaluate, record, bb, alpha1, alpha2, eta, discrete, continuous,
                     class_name, idx_features, distance_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, feature_values, mutpb, toolbox)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    
    return toolbox


def fit(toolbox, population_size=1000, halloffame_ratio=0.1, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False):
    
    halloffame_size = int(np.round(population_size * halloffame_ratio))
    
    population = toolbox.population(n=population_size)
    halloffame = tools.HallOfFame(halloffame_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                              stats=stats, halloffame=halloffame, verbose=verbose)
    
    return population, halloffame, logbook


def get_oversample(population, halloffame):
    fitness_values = [p.fitness.wvalues[0] for p in population]
    fitness_values = sorted(fitness_values)
    fitness_diff = [fitness_values[i+1] - fitness_values[i] for i in range(0, len(fitness_values)-1)]

    index = np.max(np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist())
    fitness_value_thr = fitness_values[index]
    
    oversample = list()
    
    for p in population:
        if p.fitness.wvalues[0] > fitness_value_thr:
            oversample.append(list(p))
            
    for h in halloffame:
        if h.fitness.wvalues[0] > fitness_value_thr:
            oversample.append(list(h))
            
    return oversample


def generate_data(x, feature_values, bb, discrete, continuous, class_name, idx_features, distance_function,
                  neigtype='all', population_size=1000, halloffame_ratio=0.1, alpha1=0.5, alpha2=0.5, eta1=1.0,
                  eta2=0.0, tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10, return_logbook=False):
    
    if neigtype == 'all':
        neigtype = {'ss': 0.25, 'sd': 0.25, 'ds': 0.25, 'dd': 0.25}
    
    size_sso = int(np.round(population_size * neigtype.get('ss', 0.0)))
    size_sdo = int(np.round(population_size * neigtype.get('sd', 0.0)))
    size_dso = int(np.round(population_size * neigtype.get('ds', 0.0)))
    size_ddo = int(np.round(population_size * neigtype.get('dd', 0.0)))
    
    Xgp = list()
    
    if size_sso > 0.0:
        toolbox_sso = setup_toolbox(x, feature_values, bb, init=record_init, init_params=x, evaluate=fitness_sso,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_sso, alpha1=alpha1, alpha2=alpha2, eta=eta1, mutpb=mutpb,
                                    tournsize=tournsize)
        population, halloffame, logbook = fit(toolbox_sso, population_size=size_sso, halloffame_ratio=halloffame_ratio, 
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xsso = get_oversample(population, halloffame)
        Xgp.append(Xsso)
    
    if size_sdo > 0.0:
        toolbox_sdo = setup_toolbox(x, feature_values, bb, init=record_init, init_params=x, evaluate=fitness_sdo,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_sdo, alpha1=alpha1, alpha2=alpha2, eta=eta1, mutpb=mutpb,
                                    tournsize=tournsize)
        population, halloffame, logbook = fit(toolbox_sdo, population_size=size_sdo, halloffame_ratio=halloffame_ratio, 
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xsdo = get_oversample(population, halloffame)
        Xgp.append(Xsdo)

    if size_dso > 0.0:
        toolbox_dso = setup_toolbox(x, feature_values, bb, init=record_init, init_params=x, evaluate=fitness_dso,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_dso, alpha1=alpha1, alpha2=alpha2, eta=eta2, mutpb=mutpb,
                                    tournsize=tournsize)
        population, halloffame, logbook = fit(toolbox_dso, population_size=size_dso, halloffame_ratio=halloffame_ratio, 
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xdso = get_oversample(population, halloffame)
        Xgp.append(Xdso)

    if size_ddo > 0.0:
        toolbox_ddo = setup_toolbox(x, feature_values, bb, init=record_init, init_params=x, evaluate=fitness_ddo,
                                    discrete=discrete, continuous=continuous, class_name=class_name,
                                    idx_features=idx_features, distance_function=distance_function,
                                    population_size=size_ddo, alpha1=alpha1, alpha2=alpha2, eta=eta2, mutpb=mutpb,
                                    tournsize=tournsize)
        population, halloffame, logbook = fit(toolbox_ddo, population_size=size_ddo, halloffame_ratio=halloffame_ratio, 
                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        Xddo = get_oversample(population, halloffame)
        Xgp.append(Xddo)

    Xgp = np.concatenate((Xgp), axis=0)

    if return_logbook:
        return Xgp, logbook

    return Xgp


def calculate_feature_values(X, columns, class_name, discrete, continuous, size=1000,
                             discrete_use_probabilities=False,
                             continuous_function_estimation=False):
    
    columns1 = list(columns)
    columns1.remove(class_name)
    feature_values = dict()
    
    for i, col in enumerate(columns1):
        values = X[:, i]
        if col in discrete:
            if discrete_use_probabilities:
                diff_values, counts = np.unique(values, return_counts=True)
                prob = 1.0 * counts / np.sum(counts)
                new_values = np.random.choice(diff_values, size=size, p=prob)
                new_values = np.concatenate((values, new_values), axis=0)
            else:
                diff_values = np.unique(values)
                new_values = diff_values
        elif col in continuous:
            if continuous_function_estimation:
                new_values = get_distr_values(values, size)
            else:  # suppose is gaussian
                mu = np.mean(values)
                sigma = np.std(values)
                new_values = np.random.normal(mu, sigma, size)
            new_values = np.concatenate((values, new_values), axis=0)
        
        feature_values[i] = new_values
        
    return feature_values


def get_distr_values(x, size=1000):
    nbr_bins = int(np.round(estimate_nbr_bins(x)))
    name, params = best_fit_distribution(x, nbr_bins)
    dist = getattr(st, name)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    distr_values = np.linspace(start, end, size)
    
    return distr_values


# Distributions to check
DISTRIBUTIONS = [st.uniform, st.dweibull, st.exponweib, st.expon, st.exponnorm, st.gamma, st.beta, st.alpha,
                 st.chi, st.chi2, st.laplace, st.lognorm, st.norm, st.powerlaw]


def freedman_diaconis(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    n = len(x)
    h = 2.0 * iqr / n**(1.0/3.0)
    k = math.ceil((np.max(x) - np.min(x))/h)
    return k


def struges(x):
    n = len(x)
    k = math.ceil( np.log2(n) ) + 1
    return k


def estimate_nbr_bins(x):
    if len(x) == 1:
        return 1
    k_fd = freedman_diaconis(x) if len(x) > 2 else 1
    k_struges = struges(x)
    if k_fd == float('inf') or np.isnan(k_fd):
        k_fd = np.sqrt(len(x))
    k = max(k_fd, k_struges)
    return k


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
                #print 'aaa'
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                # print distribution.name, sse
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution.name, best_params

