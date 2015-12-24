
"""
"""


def generador_sim(sim_f, params):
    n_sim = len(sim_f)
    for f, p in sim_f, params:
        f(*p)


def generator_res(sim_f, sim_p, desc_f, desc_p):
    results = []
    sim_info = product(sim_f, copy.copy(sim_p))
    for f, p in sim_info:
        v = f(*p)
        desc_info = product(desc_f, copy.copy(desc_p))
        for vf, vp in desc_info:
            results.append(vf(v, *vp))
    return results


def generator_descnames(sim_f, sim_p, desc_f, desc_p):
    results = []
    sim_info = product(sim_f, copy.copy(sim_p))
    for f, p in sim_info:
        desc_info = product(desc_f, copy.copy(desc_p))
        for vf, vp in desc_info:
            results.append([f, p, vf, vp])
    return results



def create_combination_parameters(l_parvals):
    n_params = len(l_parvals)
    return product(*l_parvals)



fs = [run_vots1, run_vots2]
params = [[5, 10, 15, 20, 25, 50], [10000]]

fs2 = [disc1, disc2, disc3]
params2 = [[[]]]

p1, p2 = product(*params), product(*params2)

r = generator_res(fs, p1, fs2, p2)
r2 = generator_descnames(fs, p1, fs2, p2)


def disc1(corrs, params=None):
    corrs = corrs/corrs[0, 0]
    corrs = corrs - np.eye(corrs.shape[0])
    return corrs.mean()


def disc2(corrs, params=None):
    corrs = corrs/corrs[0, 0]
    corrs = corrs - np.eye(corrs.shape[0])
    return corrs.std()


def disc3(corrs, params=None):
    corrs = corrs/corrs[0, 0]
    corrs = corrs - np.eye(corrs.shape[0])
    return np.sum(corrs, axis=1).std()


def run_vots1(n_parties, n_vots):
    v = np.random.randint(1, 3,(n_vots, n_parties)).astype(int)
    corr, v[:, 0] = np.zeros((n_parties, n_parties)), 0
    for i in xrange(n_vots):
        for j in range(n_parties):
            corr[j, v[i, j] == v[i, :]] += 1
    return corr

def run_vots2(n_parties, n_vots):
    v = np.ones((n_vots, n_parties)).astype(int)*np.random.randint(0, 3, n_parties)
    corr, v[:, 0] = np.zeros((n_parties, n_parties)), 0
    for i in xrange(n_vots):
        for j in range(n_parties):
            corr[j, v[i, j] == v[i, :]] += 1
    return corr




