from sklearn.linear_model import Ridge
import scipy.stats
import time
import numpy as np
from flexridge import RidgeRegression, get_dip_values
import matplotlib.pyplot as plt

# trial solving constrained coefficients
np.random.seed(102)
n=70000
n_feats = 6
X = np.random.rand(n*n_feats).reshape(n,n_feats)
sample_weight = np.random.rand(n)# np.ones(n)
coefs_all = [np.array([0.02,0.05,0.5,-.4,0.25,0.1]),
            np.array([0.02,0.05,0.5,.4,0.25,0.1]),
         ]
unimodal_idx_groups = False
fit_intercept = True
alpha=0.01
n_trials = 10

def test_standard_fit_matches_sklearn():
    for i_coefs, coefs in enumerate(coefs_all):
        print('true coef:', coefs)
        y = np.dot(X, coefs
                   ).ravel() + scipy.stats.norm(0,1).rvs(n)

        # simple least squares
        start = time.time()
        for i in range(n_trials):
            reg = Ridge(alpha=alpha,#solver='lsqr', #normalize=False,
                        fit_intercept=fit_intercept,# positive=True
                        )
            reg.fit(X,y,sample_weight=sample_weight)
        time_s = np.round(time.time()-start,3)
        print('std ridge:',0.5*np.sum((y-reg.predict(X))**2), time_s, reg.intercept_,reg.coef_,
              )

        # custom Ridge
        start = time.time()
        for i in range(n_trials):
            reg_custom = RidgeRegression(alpha=alpha,
                                         fit_intercept=fit_intercept,
                        unimodal=unimodal_idx_groups)
            reg_custom.fit(X,y,sample_weight=sample_weight)
        time_s = np.round(time.time()-start,3)
        print('custom ridge:',0.5*np.sum((y-reg_custom.predict(X))**2),
              time_s,
              reg.intercept_, reg_custom.coef_,
              )
        # print(reg.intercept_, reg_custom.intercept_)
        assert np.allclose(reg.intercept_, reg_custom.intercept_,atol=1e-4)
        assert np.allclose(reg.coef_, reg_custom.coef_,atol=1e-4)
        assert np.allclose(reg.predict(X), reg_custom.predict(X),atol=1e-4)

def test_monotone_fit_matches_sklearn():
    for i_coefs, coefs in enumerate(coefs_all):
        y = np.dot(X, coefs
                   ).ravel() + scipy.stats.norm(0, 1).rvs(n)
        # simple least squares
        start = time.time()
        for i in range(n_trials):
            reg = Ridge(alpha=alpha,#solver='lsqr', #normalize=False,
                        fit_intercept=fit_intercept, positive=True
                        )
            reg.fit(X,y,sample_weight=sample_weight)
        time_s = np.round(time.time()-start,3)
        print('std ridge monotone:',0.5*np.sum((y-reg.predict(X))**2), time_s, reg.intercept_,reg.coef_,
              )

        # custom Ridge
        start = time.time()
        for i in range(n_trials):
            reg_custom = RidgeRegression(alpha=alpha,
                                         fit_intercept=fit_intercept,
                        unimodal=unimodal_idx_groups,
                                         monotone_constraints=[1]*(len(coefs)))
            reg_custom.fit(X,y,sample_weight=sample_weight)
        time_s = np.round(time.time()-start,3)
        print('custom ridge monotone:',0.5*np.sum((y-reg_custom.predict(X))**2),
              time_s,
              reg.intercept_, reg_custom.coef_,
              )
        assert np.allclose(reg.intercept_, reg_custom.intercept_,atol=1e-4)
        assert np.allclose(reg.coef_, reg_custom.coef_,atol=1e-4)
        assert np.allclose(reg.predict(X), reg_custom.predict(X),atol=1e-4)

def test_unimodal_solutions():
    correct_unimodal_coefs = [np.array([-0.15253716,  0.27612617,  0.27612618,  0.82850476, -0.22418673, -0.26600565]),
                np.array([-0.04950415, -0.04950415,  0.50304867,  0.50865376,  0.00694265, -0.12748278]),
                np.array([-0.04950671, -0.04950671,  0.50305287,  0.50865698,  0.25691133, -0.12748643]),
                np.array([-0.09941247, -0.09941247,  0.28908646,  0.28908646,  0.28908646,  0.36963751]),
                np.array([-0.13743962, -0.13743962,  0.50099497,  0.50307497,  0.01111545, -0.13402668]),
                # np.array([0.61969394 , 0.33860653 , 0.33860642 , 0.33860633 , 0.27177542 , 0.2717753 ]),
                # np.array([-0.05994643,  0.1801058 ,  0.70110524,  0.26040426 , 0.25248045,  0.25248045]),
                # np.array([0.73636598 , 0.73636598 , 0.90922988 , 0.99976239 , 0.49803944 , 0.49803944]),
                # np.array([0.42534316 , 0.60668949 , 0.78069408 , 0.33134332 , 0.33134333 , 0.15102768]),
                ]

    # trial solving constrained coefficients
    rand_seed = 121
    n = 900
    n_feats = 6
    np.random.seed(rand_seed)
    X = np.random.rand(n * n_feats).reshape(n, n_feats)
    coefs_all = [
        np.array([0., 0.75, 0., .75, 0., 0]),
        np.array([0.2, 0.05, 0.5, .4, 0.25, 0.1]),
        np.array([0.2, 0.05, 0.5, .4, 0.5, 0.1]),
        np.array([0.1, 0.05, 0.3, .2, 0.5, 0.6]),
        np.array([0.02, 0.05, 0.5, .4, 0.25, 0.1]),  # prob
        # np.random.rand(6),
        # np.random.rand(6),
        # np.random.rand(6),
        # np.random.rand(6),
    ]
    # unimodal_idx_groups = True
    fit_intercept = True
    alpha = 0.01
    for i_coefs, coefs in enumerate(coefs_all):
        np.random.seed(rand_seed)  # +i_coefs)
        print('true coef:', coefs)

        y = np.dot(X, coefs
                   ).ravel() + scipy.stats.norm(0, 1).rvs(n)

        # custom Ridge
        start = time.time()
        # for i in range(100):
        reg_custom = RidgeRegression(alpha=alpha,
                                     fit_intercept=fit_intercept,
                                     unimodal=False)
        reg_custom.fit(X, y)
        time_s = np.round(time.time() - start, 3)
        print('custom ridge:', 0.5 * np.sum((y - reg_custom.predict(X)) ** 2), np.max(get_dip_values(reg_custom.coef_)),
              time_s,
              reg_custom.intercept_, reg_custom.coef_,
              )
        fig, ax = plt.subplots()
        start = time.time()
        # for i in range(100):
        reg_custom_unimodal = RidgeRegression(alpha=alpha,
                                              fit_intercept=fit_intercept,
                                              unimodal=True)
        reg_custom_unimodal.fit(X, y)
        time_s = np.round(time.time() - start, 3)
        print('custom ridge unimodal:', 0.5 * np.sum((y - reg_custom_unimodal.predict(X)) ** 2),
              np.max(get_dip_values(reg_custom_unimodal.coef_)),
              time_s,
              reg_custom_unimodal.intercept_, reg_custom_unimodal.coef_,
              )
        #
        X_coef_locn = np.arange(len(coefs))

        plt.plot(X_coef_locn, coefs, marker='o', label='coefs true', color='k', linestyle='--')
        # plt.plot(X_coef_locn,coefs_est_ls, color='r',label='coefs ls', marker='x')
        plt.plot(X_coef_locn, reg_custom.coef_, label='coefs ridge', color='red', marker='x')
        plt.plot(X_coef_locn, reg_custom_unimodal.coef_, label='coefs ridge unimodal',
                 color='blue', marker='o', linestyle='--')
        print('unimodal coefs:', reg_custom_unimodal.coef_)
        # plt.plot(X_coef_locn,coefs_est_sp_uni, label='coefs spline constrained unimodal', color='g', marker='p')
        # plt.plot(x_all,coefs_full_spline_uni, label='coefs spline constrained unimodal', color='g', linestyle='--')
        # print(np.dot(cs.derivations_mat,spl_coefs_opt_uni ))
        # print(np.dot(constraints[-2,:],spl_coefs_opt_uni ))
        plt.legend()
        plt.grid()
        plt.ylim([0, 1])
        plt.show()

        # print('XX',correct_unimodal_coefs[i_coefs], reg_custom_unimodal.coef_)
        assert(np.allclose(correct_unimodal_coefs[i_coefs], reg_custom_unimodal.coef_, atol=1e-4))


if __name__ == '__main__':
    test_unimodal_solutions()
    test_standard_fit_matches_sklearn()
    test_monotone_fit_matches_sklearn()

