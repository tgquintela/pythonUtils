
"""
test perturbations
------------------
Collection of tests which has to pass the perturbations functions and classes.

"""

import numpy as np
from perturbations import BasePerturbation, NonePerturbation, JitterLocations,\
    PermutationPerturbationLocations, PermutationPerturbationGeneration,\
    PermutationPerturbation, PartialPermutationPerturbationGeneration,\
    MixedFeaturePertubation, DiscreteIndPerturbation,\
    ContiniousIndPerturbation, PermutationIndPerturbation
from basecore_perturbations import BaseCorePermutationArray,\
    BaseCoreSubstitutionArray, BaseCoreJitteringArray
from filter_perturbations import sp_general_filter_perturbations,\
    feat_filter_perturbations, ret_filter_perturbations


def test():
    ###########################################################################
    ############################# Artificial data #############################
    ###########################################################################
    n = 1000
    ## Random data
    inds = np.arange(10)
    X_cont_ind = np.random.random((n, 1))
    X_cat_ind = np.random.randint(0, 10, (n, 1))
    X_cont = np.random.random((n, 1))
    X_cat = np.random.randint(0, 10, (n, 1))
    X_cat_ext = np.random.randint(0, 10, (n, 5))
    X_cont_ext = np.random.random((n, 5))
    X_mix = np.hstack([X_cont, X_cat])
    locs = np.random.random((n, 2))*100

    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
    k_perturb4 = k_perturb1+k_perturb2+k_perturb3
    probs = np.random.random((10, 10))
    probs = np.divide(probs, probs.sum(1)[:, None])

    probs_subs = [probs for i in range(5)]
    labels_subs = [np.arange(10) for i in range(5)]

    def f_coreperturb_test(perturb, mode='cont', args=[]):
        ## Preparation of inputs
        # Features definition
        if mode == 'cont':
            data = X_cont_ext
        elif mode == 'cat':
            data = X_cat_ext
        elif mode == 'mix':
            data = X_mix
        ## Computation testing
        perturb.apply(data, [0, 2], *args)
        if perturb.is_set():
            perturb._assert_set()
            args = perturb.get_parameters()
            perturb.set_parameters(*args)
        perturb.set_data(data)
        perturb._assert_set()
        args = perturb.get_parameters()
        perturb.set_parameters(*args)
        data_p = perturb.apply(data, [0, 2])
        assert(type(data_p) == np.ndarray)
        assert(data_p.shape[0] == n)
        assert(data_p.shape[1] == data.shape[1])
        assert(data_p.shape[2] == 2)

    def f_perturb_test(perturb, mode='cont', ind=False):
        "Test of perturbation."
        ## Preparation of inputs
        # Features definition
        if mode == 'cont':
            if ind:
                feats = X_cont_ind
            else:
                feats = X_cont
        elif mode == 'cat':
            if ind:
                feats = X_cat_ind
            else:
                feats = X_cat
        elif mode == 'mix':
            feats = X_mix
        ## Testing auxiliar functions
        ks = perturb._format_k_perturb(None)
        assert(type(ks) == list)
        ## Testing combinations of functions without precomputed
        loop_possible_test(perturb, feats)

        ## Testing general functions
        perturb.selfcompute_features(feats)
        assert(perturb.features_p is not None)
        assert(len(perturb.features_p.shape) == len(feats.shape)+1)
        perturb.selfcompute_locations(locs)
        assert(perturb.locations_p is not None)
        assert(len(perturb.locations_p.shape) == 3)
        #perturb.selfcompute_relations(relations)
        #perturb.selfcompute_discretizations(discretizations)
        loop_possible_test(perturb, feats)

    def loop_possible_test(perturb, feats):
        for k in range(perturb.k_perturb):
            ####################### Perturbation on data ######################
            ###################################################################
            ## Perturbation on indices
            inds_perturb = perturb.apply2indices(inds, k)
            assert(type(inds_perturb) == np.ndarray)
            assert(len(inds_perturb.shape) == 1)
            inds_perturb = perturb.apply2indices(inds, [k])
            assert(type(inds_perturb) == np.ndarray)
            assert(len(inds_perturb.shape) == 2)
            inds_perturb = perturb.apply2indices(list(inds), k)
            assert(type(inds_perturb) == list)
            inds_perturb = perturb.apply2indices(0, k)
            assert(inds_perturb - int(inds_perturb) <= np.finfo('f').eps)
#            assert(type(inds_perturb) == int)

            ## Perturbation on locations
            locs_perturb = perturb.apply2locations(locs, k)
            assert(type(locs_perturb) == np.ndarray)
            assert(len(locs_perturb.shape) == 2)
            locs_perturb = perturb.apply2locations(locs, [k])
            assert(type(locs_perturb) == np.ndarray)
            assert(len(locs_perturb.shape) == 3)

            ## Perturbation on features
            feats_perturb = perturb.apply2features(feats, k)
            assert(type(feats_perturb) == np.ndarray)
            assert(len(feats_perturb.shape) == len(feats.shape))
            feats_perturb = perturb.apply2features(feats, [k])
            assert(type(feats_perturb) == np.ndarray)
            assert(len(feats_perturb.shape) == len(feats.shape)+1)

            ## Perturbation on relations
            #perturb.apply2relations(relations, k)

            ## Perturbation on regions
            #perturb.apply2discretization(relations, k)

            ################### Perturbation on individuals ###################
            ###################################################################
            ## Perturbation location individuals
            locs_perturb = perturb.apply2locs_ind(locs, 0, k)
            assert(type(locs_perturb) == np.ndarray)
            assert(len(locs_perturb.shape) == 1)
            locs_perturb = perturb.apply2locs_ind(locs, [0], k)
            assert(type(locs_perturb) == np.ndarray)
            assert(len(locs_perturb.shape) == 2)
            locs_perturb = perturb.apply2locs_ind(locs, 0, [k])
            assert(type(locs_perturb) == np.ndarray)
            assert(len(locs_perturb.shape) == 2)
            locs_perturb = perturb.apply2locs_ind(locs, [0], [k])
            assert(type(locs_perturb) == np.ndarray)
            assert(len(locs_perturb.shape) == 3)

            ## Perturbation features individuals
            feats_perturb = perturb.apply2features_ind(feats, 0, k)
            if len(feats.shape) == 1:
                assert(type(feats_perturb) != np.ndarray)
            else:
                assert(type(feats_perturb) == np.ndarray)
                assert(len(feats_perturb.shape) == 1)
            feats_perturb = perturb.apply2features_ind(feats, [0], k)
            assert(type(feats_perturb) == np.ndarray)
            assert(len(feats_perturb.shape) == len(feats.shape))
            feats_perturb = perturb.apply2features_ind(feats, 0, [k])
            assert(type(feats_perturb) == np.ndarray)
            assert(len(feats_perturb.shape) == len(feats.shape))
            feats_perturb = perturb.apply2features_ind(feats, [0], [k])
            assert(type(feats_perturb) == np.ndarray)
            assert(len(feats_perturb.shape) == len(feats.shape)+1)

            #perturb.apply2relations_ind(relations, k)
            #perturb.apply2discretization_ind(relations, k)
        perturb.apply2indices(inds)
        perturb.apply2indices(0)
        perturb.apply2locations(locs)
        perturb.apply2features(feats)

    ###########################################################################
    ############################## Perturbations ##############################
    ###########################################################################
    ## CorePerturbations
    perm = BaseCorePermutationArray((n, 10))
    rei = perm.reindices
    args = (n, 10), False
    f_coreperturb_test(perm, 'mix', args)
    perm = BaseCorePermutationArray()
    f_coreperturb_test(perm, 'mix', args)
    perm = BaseCorePermutationArray((n, 10), False)
    f_coreperturb_test(perm, 'mix', args)
    perm = BaseCorePermutationArray(rei)
    f_coreperturb_test(perm, 'mix', args)
    perm = BaseCorePermutationArray(rei, False)
    f_coreperturb_test(perm, 'mix', args)

    jitt = BaseCoreJitteringArray()
    f_coreperturb_test(jitt, 'cont', [])
    jitt = BaseCoreJitteringArray()
    args = 0.2,
    f_coreperturb_test(jitt, 'cont', args)
    jitt = BaseCoreJitteringArray(0.2)
    f_coreperturb_test(jitt, 'cont', args)
    jitt = BaseCoreJitteringArray()
    f_coreperturb_test(jitt, 'cont', args)

    subs = BaseCoreSubstitutionArray()
    f_coreperturb_test(subs, 'cat', [])
    args = probs_subs, labels_subs
    subs = BaseCoreSubstitutionArray()
    f_coreperturb_test(subs, 'cat', args)
    subs = BaseCoreSubstitutionArray(probs_subs)
    f_coreperturb_test(subs, 'cat', args)
    subs = BaseCoreSubstitutionArray(probs_subs, labels_subs)
    f_coreperturb_test(subs, 'cat', args)

    ## Wrapper perturbations
    nonepert = NonePerturbation(10)
    f_perturb_test(nonepert)
    locsjitpert = JitterLocations(0.2, 5)
    f_perturb_test(locsjitpert)
    locspermpert = PermutationPerturbationLocations((n, 5))
    f_perturb_test(locspermpert)
    featspermpert = PermutationPerturbation((n, 5))
    f_perturb_test(featspermpert)
    permgener = PermutationPerturbationGeneration(n, k_perturb=2, seed=None)
    f_perturb_test(permgener)
    parpermgener = PartialPermutationPerturbationGeneration(n, rate_pert=.9,
                                                            k_perturb=2,
                                                            seed=None)
    f_perturb_test(parpermgener)

    # Individuals feature perturbations
    permpert_ind = PermutationIndPerturbation((n, 1))
    f_perturb_test(permpert_ind, 'cont', True)
    discpert_ind = DiscreteIndPerturbation(probs)
    f_perturb_test(discpert_ind, 'cat', True)
    contpert_ind = ContiniousIndPerturbation(0.05)
    f_perturb_test(contpert_ind, 'cont', True)
    mixpert_ind = MixedFeaturePertubation([contpert_ind, discpert_ind])
    f_perturb_test(mixpert_ind, 'mix', True)

    ###########################################################################
    ##################### Auxiliar perturbation functions #####################
    ###########################################################################
    sp_general_filter_perturbations(locsjitpert)
    feat_filter_perturbations(locsjitpert)
    ret_filter_perturbations(locsjitpert)
    sp_general_filter_perturbations(locspermpert)
    feat_filter_perturbations(locspermpert)
    ret_filter_perturbations(locspermpert)
    sp_general_filter_perturbations(featspermpert)
    feat_filter_perturbations(featspermpert)
    ret_filter_perturbations(featspermpert)
    sp_general_filter_perturbations([locsjitpert])
    feat_filter_perturbations([locsjitpert])
    ret_filter_perturbations([locsjitpert])
    sp_general_filter_perturbations([locspermpert])
    feat_filter_perturbations([locspermpert])
    ret_filter_perturbations([locspermpert])
    sp_general_filter_perturbations([featspermpert])
    feat_filter_perturbations([featspermpert])
    ret_filter_perturbations([featspermpert])

    perts = [PermutationPerturbation((n, 5)), NonePerturbation(5),
             JitterLocations(0.2, 5)]

    sp_general_filter_perturbations(perts)
    feat_filter_perturbations(perts)
    ret_filter_perturbations(perts)

#
#
#def test():
#    n = 1000
#    locs = np.random.random((n, 2))*100
#    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
#    k_perturb4 = k_perturb1+k_perturb2+k_perturb3
#
#    ## Perturbations features
#    feat_arr0 = np.random.randint(0, 20, (n, 1))
#    feat_arr1 = np.random.random((n, 10))
#    feat_arr = np.hstack([feat_arr0, feat_arr1])
#
#    ##########################################################################
#    #### GeneralPermutations
#    ## Create perturbations
#    class DummyPerturbation(BasePerturbation):
#        _categorytype = 'feature'
#        _perturbtype = 'dummy'
#
#        def __init__(self, ):
#            self._initialization()
#            self.features_p = np.random.random((10, 10, 10))
#            self.locations_p = np.random.random((100, 2, 5))
#            self.relations_p = np.random.random((100, 2, 5))
#    dummypert = DummyPerturbation()
#
#    # Testing main functions
#    dummypert.apply2indice(0, 0)
#    dummypert.apply2locations(locs)
#    dummypert.apply2locs_ind(locs, 0, 0)
#    dummypert.apply2features(feat_arr)
#    dummypert.apply2features_ind(feat_arr, 0, 0)
#    dummypert.apply2relations(None)
#    dummypert.apply2relations_ind(None, 0, 0)
#    dummypert.apply2discretizations(None)
#    dummypert.selfcompute_features(feat_arr)
#    dummypert.selfcompute_locations(locs)
#    dummypert.selfcompute_relations(None)
#    dummypert.selfcompute_discretizations(None)
#
#    ##########################################################################
#    #### Permutations
#    ## Create perturbations
#    perturbation1 = PermutationPerturbation((n, k_perturb1))
#    reind = np.vstack([np.random.permutation(n) for i in range(k_perturb1)])
#    perturbation1 = PermutationPerturbation(reind.T)
#
#    # Testing main functions individually
#    perturbation1.apply2indice(0, 0)
#    perturbation1.apply2locations(locs)
##    perturbation1.apply2locs_ind(locs, 0, 0)
#    perturbation1.selfcompute_locations(locs)
#    perturbation1.apply2features(feat_arr)
#    perturbation1.apply2features_ind(feat_arr, 0, 0)
#    perturbation1.selfcompute_features(feat_arr)
#
#    # Perturbations in Retriever
#    ret1 = KRetriever(locs)
#    ret2 = CircRetriever(locs)
#    ret1.add_perturbations(perturbation1)
#    ret2.add_perturbations(perturbation1)
#    assert(ret1.k_perturb == perturbation1.k_perturb)
#    assert(ret2.k_perturb == perturbation1.k_perturb)
#
#    ##########################################################################
#    #### NonePerturbation
#    ## Create perturbations
#    perturbation2 = NonePerturbation(k_perturb2)
#
#    # Testing main functions individually
#    perturbation2.apply2indice(0, 0)
#    perturbation2.apply2locations(locs)
##    perturbation2.apply2locs_ind(locs, 0, 0)
#    perturbation2.selfcompute_locations(locs)
#    perturbation2.apply2features(feat_arr)
##    perturbation2.apply2features_ind(feat_arr, 0, 0)
#    perturbation2.selfcompute_features(feat_arr)
#
#    # Perturbations in Retriever
#    ret1 = KRetriever(locs)
#    ret2 = CircRetriever(locs)
#    ret1.add_perturbations(perturbation2)
#    ret2.add_perturbations(perturbation2)
#    assert(ret1.k_perturb == perturbation2.k_perturb)
#    assert(ret2.k_perturb == perturbation2.k_perturb)
#
#    ##########################################################################
#    #### JitterPerturbations
#    ## Create perturbations
#    perturbation3 = JitterLocations(0.2, k_perturb3)
#
#    # Testing main functions individually
#    perturbation3.apply2indice(0, 0)
#    perturbation3.apply2locations(locs)
##    perturbation3.apply2locs_ind(locs, 0, 0)
#    perturbation3.selfcompute_locations(locs)
#    perturbation3.apply2features(feat_arr)
##    perturbation3.apply2features_ind(feat_arr, 0, 0)
#    perturbation3.selfcompute_features(feat_arr)
#
#    # Perturbations in Retriever
#    ret1 = KRetriever(locs)
#    ret2 = CircRetriever(locs)
#    ret1.add_perturbations(perturbation3)
#    ret2.add_perturbations(perturbation3)
#    assert(ret1.k_perturb == perturbation3.k_perturb)
#    assert(ret2.k_perturb == perturbation3.k_perturb)
#
#    ##########################################################################
#    #### CollectionPerturbations
#    ## Create perturbations
#    perturbation4 = [perturbation1, perturbation2, perturbation3]
#
#    # Perturbations in Retriever
#    ret1 = KRetriever(locs)
#    ret2 = CircRetriever(locs)
#    ret1.add_perturbations(perturbation4)
#    ret2.add_perturbations(perturbation4)
#    assert(ret1.k_perturb == k_perturb4)
#    assert(ret2.k_perturb == k_perturb4)
#
#    ##########################################################################
#    #### IndividualPerturbations
#    feat_perm = np.random.random((100, 1))
#    feat_disc = np.random.randint(0, 10, 100)
#    feat_cont = np.random.random((100, 1))
#
#    ### Reindices individually
#    # Individual perturbations
#    reind_ind = np.random.permutation(100).reshape((100, 1))
#
#    try:
#        boolean = False
#        perm_ind = PermutationIndPerturbation(list(reind_ind))
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    perm_ind = PermutationIndPerturbation(reind_ind)
#    perm_ind.reindices
#    # Testing main functions individually
#    perm_ind.apply2indice(0, 0)
#    perm_ind.apply2locations(locs)
##    perm_ind.apply2locs_ind(locs, 0, 0)
#    perm_ind.selfcompute_locations(locs)
#    perm_ind.apply2features(feat_perm)
#    perm_ind.apply2features(feat_perm, 0)
#    perm_ind.apply2features_ind(feat_perm, 0, 0)
#    perm_ind.selfcompute_features(feat_perm)
#
#    ### Continious individually
#    cont_ind = ContiniousIndPerturbation(0.5)
#    # Testing main functions individually
#    cont_ind.apply2indice(0, 0)
#    cont_ind.apply2locations(locs)
##    cont_ind.apply2locs_ind(locs, 0, 0)
#    cont_ind.selfcompute_locations(locs)
#    cont_ind.apply2features(feat_cont)
#    cont_ind.apply2features(feat_cont, 0)
##    cont_ind.apply2features_ind(feat_cont, 0, 0)
#    cont_ind.selfcompute_features(feat_cont)
#
#    ### Discrete individually
#    try:
#        boolean = False
#        disc_ind = DiscreteIndPerturbation(np.random.random((10, 10)))
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        probs = np.random.random((10, 10))
#        probs = (probs.T/probs.sum(1)).T
#        disc_ind = DiscreteIndPerturbation(probs[:8, :])
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    probs = np.random.random((10, 10))
#    probs = (probs.T/probs.sum(1)).T
#    disc_ind = DiscreteIndPerturbation(probs)
#    # Testing main functions individually
#    disc_ind.apply2indice(0, 0)
#    disc_ind.apply2locations(locs)
##    disc_ind.apply2locs_ind(locs, 0, 0)
#    disc_ind.selfcompute_locations(locs)
#    disc_ind.apply2features(feat_disc)
#    disc_ind.apply2features(feat_disc, 0)
##    disc_ind.apply2features_ind(feat_disc, 0, 0)
#    disc_ind.selfcompute_features(feat_disc)
#    try:
#        boolean = False
#        disc_ind.apply2features(np.random.randint(0, 40, 1000))
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#
#    ### Mix individually
#    mix_coll = MixedFeaturePertubation([perm_ind, cont_ind, disc_ind])
#    # Testing main functions individually
#    feat_mix = np.hstack([feat_perm, feat_cont, feat_disc.reshape((100, 1))])
#    mix_coll.apply2indice(0, 0)
#    mix_coll.apply2locations(locs)
##    mix_coll.apply2locs_ind(locs, 0, 0)
#    mix_coll.selfcompute_locations(locs)
#    mix_coll.apply2features(feat_mix)
##    mix_coll.apply2features_ind(feat_mix, 0, 0)
#    mix_coll.selfcompute_features(feat_mix)
#
#    try:
#        boolean = False
#        MixedFeaturePertubation(None)
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        MixedFeaturePertubation([None])
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        mix_coll.apply2features(None)
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        mix_coll.apply2features([None])
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#
#    ##########################################################################
#    ##################### Auxiliar perturbation functions ####################
#    ##########################################################################
#    sp_general_filter_perturbations(perturbation1)
#    feat_filter_perturbations(perturbation1)
#    ret_filter_perturbations(perturbation1)
#    sp_general_filter_perturbations(perturbation2)
#    feat_filter_perturbations(perturbation2)
#    ret_filter_perturbations(perturbation2)
#    sp_general_filter_perturbations(perturbation3)
#    feat_filter_perturbations(perturbation3)
#    ret_filter_perturbations(perturbation3)
#    sp_general_filter_perturbations([perturbation1])
#    feat_filter_perturbations([perturbation1])
#    ret_filter_perturbations([perturbation1])
#    sp_general_filter_perturbations([perturbation2])
#    feat_filter_perturbations([perturbation2])
#    ret_filter_perturbations([perturbation2])
#    sp_general_filter_perturbations([perturbation3])
#    feat_filter_perturbations([perturbation3])
#    ret_filter_perturbations([perturbation3])
#
#    perts = [PermutationPerturbation((n, 5)), NonePerturbation(5),
#             JitterLocations(0.2, 5)]
#
#    sp_general_filter_perturbations(perts)
#    feat_filter_perturbations(perts)
#    ret_filter_perturbations(perts)
