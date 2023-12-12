from flexridge import RidgeRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import numpy as np

class RidgeRegressionCV(object):
    """ Similar to Ridge CV, but with additional features of flexridge.RidgeRegression.

    """

    def __init__(self, monotone_constraints=None, # either: None=0, -1 (all non-increasing), +1 (all non-decreasing), or a list of +1/-1/0 for all feats
                 positive=None, # included for sklearn Ridge compatibiltiy. positive=True is equivalent to monotone_constraints=1.
                 alphas = (0.1, 1.0, 10.0),
                 fit_intercept=True,
                 unimodal = False, # either: True/False, OR a list of lists of indices of features to be made unimodal
                 alpha_optimisation = 'sklearn_ridge_cv', # only option at present
                 standardize = False, # normlise X columns to have unit variance and zero mean
                 cv = None, # for sklearn optimisation, defaults to optimised LOO
                 **kwargs):
        self.base_regressor_params_dict = {'monotone_constraints': None if monotone_constraints is None else np.asarray(monotone_constraints),
                                           'positive': positive,
                                           'fit_intercept': fit_intercept,
                                           'unimodal':unimodal}
        self.alphas = alphas
        self.alpha_optimisation = alpha_optimisation
        self.standardize = standardize
        self.standardizer = StandardScaler() if self.standardize else None
        self.cv = cv
        # self.regressor = linear_model.RidgeCV(**kwargs)
        # self.monotone_constraints = None if monotone_constraints is None else np.asarray(monotone_constraints)

    def fit(self, X, y):
        # standard is requested. there may be a teensy bit of leakage, I'm ok with that
        if self.standardize:
            X_t = self.standardizer.fit_transform(X)
        else:
            X_t = X.copy()

        # optimise for alpha
        if self.alpha_optimisation == 'sklearn_ridge_cv':
            regcv = RidgeCV(alphas=self.alphas, fit_intercept=self.base_regressor_params_dict['fit_intercept'], cv = self.cv)
            regcv.fit(X_t, y)
            self.alpha_ = regcv.alpha_
            print('Optimal alpha: {} ({}/{})'.format(self.alpha_, list(self.alphas).index(self.alpha_), len(self.alphas)))
        else:
            raise NotImplementedError('alpha_optimisation = {} not implemented'.format(self.alpha_optimisation))

        # fit the model
        param_dict = self.base_regressor_params_dict.copy()
        param_dict['alpha'] = self.alpha_
        self.regressor = RidgeRegression(**param_dict)
        self.regressor.fit(X_t, y)
        self.intercept_ = self.regressor.intercept_
        self.coef_ = self.regressor.coef_
        if self.standardize:
            self.coef_ = self.coef_ / self.standardizer.scale_
            self.intercept_ = self.intercept_ - np.sum(self.coef_ * self.standardizer.mean_)
        return


    def predict(self, X):
        if self.standardize:
            X_t = self.standardizer.transform(X)
        else:
            X_t = X
        res = self.regressor.predict(X_t)
        return res

    # def get_params(self, deep=True):
    #     dit = {
    #
    #         'monotone_constraints': self.monotone_constraints,
    #     }
    #     for k, v in self.regressor.get_params(deep=deep).items():
    #         dit[k] = v
    #     return dit
    #
    # def set_params(self, **kwargs):
    #     new_classifier_params = {}
    #     # print('set_params:',kwargs.items())
    #     for k, v in kwargs.items():
    #         if k in dir(self.regressor):
    #             new_classifier_params[k] = v
    #         if k in dir(self):
    #             setattr(self, k, v)
    #     self.regressor.set_params(**new_classifier_params)
    #     return self