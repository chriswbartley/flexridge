"""Flexible Ridge Regression (sklearn compatible).

This module implements a Ridge regression model in the sklearn pattern.
It is designed to be a plug in replacement to the sklearn Ridge model,
but with the following additional features:
    * Unimodal coefficients: groups of coefficients can be constrained to be
        unimodal.
    * Separately monotone coefficients: partial monotonicity can be imposed,
        with some coefficients can be constrained to be non-increasing or non-decreasing
        while others are not constrained. The sklearn implementation only allows
        all coefficients to be 'positive' (or no constraints).

Author:
    Chris Bartley, PhD chris@bartleys.net, 2023

The unimodal constraints are achieved by a novel approach that uses a quadratic
fit to the coefficients and then imposes a constraint on the curvature of the
quadratic. This prevents 'dips' in the coefficients, which are required for
multimodality.

Attributes:
    RidgeRegression: Main flexible ridge model
    QuadraticFit: Used to assess for coefficient unimodality.

Todo:
    * Replace r2_score with local definition to remove sklearn dependency.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
from sklearn.metrics import r2_score
import numpy as np

class QuadraticFit:
    critical_dip_value = 1*np.exp(-1*np.abs(0.5))
    def __init__(self, y1, y2, y3):
        self.y1 = y1
        self.y2= y2
        self.y3 = y3
        P1,P2,P3 = self.y1,self.y2,self.y3
        # calculate quadratic that goes through these points
        self.a = P1
        self.c = 0.5*(P3+P1-2*P2)
        self.b = P2-P1-self.c

        self.quad_pos_fn = lambda x_: self.a + self.b*x_ + self.c*x_**2
        if self.c !=0:
            self.x_min = -self.b/(2*self.c)
        else:
            self.x_min = np.inf
        # precalculate jacobian components
        self.denom = (self.y1-self.y3)**3
        self.factor = 2*self.y1-4*self.y2+ 2* self.y3
    def calc_value(self,x):
        return self.quad_pos_fn(x)
    def calc_dip_metric2(self):
        if self.c !=0:
            return np.sign(self.c)*1./(-self.b/2/self.c - 1)**2
        else:
            return 0.
    def calc_dip_metric2_jacobian(self):

        jac = np.sign(self.c)*self.factor/self.denom*np.array([
            8*(self.y2 - self.y3),
            8*(-self.y1 + self.y3),
            8*(self.y1 - self.y2),
        ])
        return jac
    def calc_dip_metric(self):
        if self.c !=0:
            return np.sign(self.c)*np.exp(-1*np.abs(-self.b/2/self.c-1))
        else:
            return 0.
    def calc_dip_metric_jacobian(self):
        val = self.calc_dip_metric()
        jac = -1*np.sign(self.x_min-1)*np.sign(self.c)/self.factor**2*np.exp(-1*np.abs(-self.b/2/self.c-1))*np.array([
            -4*(self.y2 - self.y3),
            -4*(-self.y1 + self.y3),
            -4*(self.y1 - self.y2),
        ])
        return jac

def get_dip_values(values, return_what = 'values'):
    if return_what == 'values':
        metrics = []
    else:
        metrics = np.zeros([300,len(values)])
    n_ = len(values)
    i_row = 0
    for i_start in range(n_-2):
        for i_end in range(i_start+2, n_):
            for i_mid in range(i_start+1, i_end):
                # print(i_start, i_end, i_mid)
                if return_what == 'values':
                    metrics.append(QuadraticFit(values[i_start],
                                         values[i_mid],
                                         values[i_end]).calc_dip_metric())
                else: # return jacobian
                    jac_local = QuadraticFit(values[i_start],
                                         values[i_mid],
                                         values[i_end]).calc_dip_metric_jacobian()
                    metrics[i_row,i_start] =jac_local[0]
                    metrics[i_row,i_mid] =jac_local[1]
                    metrics[i_row,i_end] =jac_local[2]

                    i_row =i_row +1
    if return_what !='values':
        metrics = metrics[:i_row]
    return np.array(metrics)

class RidgeRegression():
    def __init__(self,
                    monotone_constraints=None, # either: None=0, -1 (all non-increasing), +1 (all non-decreasing), or a list of +1/-1/0 for all feats
                 positive=None, # included for sklearn Ridge compatibiltiy. positive=True is equivalent to monotone_constraints=1.
                 alpha = 0.01,
                 fit_intercept=True,
                 unimodal = False, # either: True/False, OR a list of lists of indices of features to be made unimodal
                 ):
        self.monotone_constraints = monotone_constraints
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.unimodal = unimodal
        self.positive = positive


    def fit(self, X, y, sample_weight=None):
        if not self.positive is None:
            if self.positive:
                self.monotone_constraints = 1
        # construct optimisation problem with requested percentiles
        coefs_init_incl_intercept = np.zeros(X.shape[1]+(1 if self.fit_intercept else 0))
        X_ = X.copy()
        coef_end_idx = X_.shape[1]

        if sample_weight is None:
            sample_weight = np.ones(len(y))
        # standardise sampleweight so that sum equals len(y)
        sample_weight = sample_weight  / (np.sum(sample_weight)/len(y))
        # Add intercept
        if self.fit_intercept:
            X_ = np.hstack([ X_,np.ones([X.shape[0],1])])

        def loss_fn(coefs):
            loss =  0.5*np.sum(sample_weight*(y-np.dot(X_,coefs))**2)+ 0.5*self.alpha*np.sum(coefs[:coef_end_idx]**2)
            return loss

        def loss_fn_with_jac(coefs):
            loss =  0.5*np.sum(sample_weight*(y-np.dot(X_,coefs))**2)+ 0.5*self.alpha*np.sum(coefs[:coef_end_idx]**2)
            y_resid = np.dot(X_,coefs)-y
            jac = np.dot(X_.T, sample_weight*y_resid).ravel() +self.alpha*np.hstack([ coefs[:coef_end_idx],[0]])
            return loss,jac

        def loss_fn_with_jac_ls_coefs(coefs, reference_coefs):

            loss__ =  0.5*np.sum((coefs[:-1]-reference_coefs[:-1])**2)
            y_resid = coefs[:-1]-reference_coefs[:-1]
            jac = np.hstack([y_resid,[0]])
            return loss__,jac

        jac = True
        # initial unconstrained solve: L-BFGS-B is VERY fast
        res = minimize(loss_fn, x0 = coefs_init_incl_intercept, method = 'L-BFGS-B',
                       # options = {'maxiter':100},
                       )
        # update initial guess to be unconstrained fit
        coefs_init_incl_intercept = res['x']
        coefs_init = coefs_init_incl_intercept[:coef_end_idx]
        constraints = []
        if not self.monotone_constraints is None:
            if not hasattr(self.monotone_constraints,  '__iter__'):
                mono_constr = [self.monotone_constraints]*X.shape[1]
            else:
                mono_constr = self.monotone_constraints
            if self.fit_intercept:
                    mono_constr.append(0)
            for i, mono_constr_ in enumerate(mono_constr):
                A = np.zeros(len(coefs_init_incl_intercept))
                A[i] = 1
                if mono_constr_ == 1:
                    constraints.append(LinearConstraint(A, lb=0., ub=np.inf, keep_feasible=False))
                elif mono_constr_ == -1:
                    constraints.append(LinearConstraint(A, lb=-np.inf, ub=0, keep_feasible=False))
                elif mono_constr_ == 0:
                    pass
                else:
                    raise ValueError('monotone_constraints must be 1, -1 or 0: {}'.format())


        if not self.unimodal is None:
            # add unimodal constraints
            if type(self.unimodal) == bool:
                if self.unimodal:
                    unimodal_idx_groups = [np.arange(X.shape[1], dtype=int)]
                else:
                    unimodal_idx_groups = []
            else:
                unimodal_idx_groups = self.unimodal
            if len(unimodal_idx_groups)>0: # we have at least one group of unimodal features
                for unimodal_idx_grp in unimodal_idx_groups:
                    metric_dip = np.max(get_dip_values(coefs_init[unimodal_idx_grp]))
                    print('coefs init', coefs_init[unimodal_idx_grp])
                    print('metric_dip',metric_dip)
                    if metric_dip>QuadraticFit.critical_dip_value:
                        print('adding constraint')
                        constr_multiplier = 1
                        constr_fn = lambda coefs__incl_int: constr_multiplier*get_dip_values(coefs__incl_int[:-1][unimodal_idx_grp], return_what = 'values')#QuadraticFit
                        def constr_fn_jac(coefs__incl_int):
                            res_ = get_dip_values(coefs__incl_int[:-1][unimodal_idx_grp], return_what = 'jacobian')
                            jac_new = np.zeros([len(res_), len(coefs__incl_int)])
                            jac_new[:,unimodal_idx_grp] = res_
                            return jac_new

                        constraints.append(NonlinearConstraint(fun=constr_fn,
                                           lb=-np.inf, ub = constr_multiplier*QuadraticFit.critical_dip_value,
                                           jac = constr_fn_jac,#'cs',# constr_fn_jac, #'cs',# cs seems better as providing jacobian can result in
                                                               )
                                           )

        solve = lambda solver, maxiter, x0: minimize(loss_fn_with_jac,
                                        jac=jac, x0 = x0,
                       method = solver,#trust-constr',#SLSQP',#L-BFGS-B',#SLSQP', COBYLA
                       options = {'maxiter':maxiter,
                                  # 'ftol':1e-4,
                                  },
                       constraints = constraints)
        res = solve('SLSQP',1000,coefs_init_incl_intercept)

        if not res['success']:
            print('SLSQP failed, trying interior pt')
            # resolve init starting point
            loss_fn_init_jac_ = lambda coef_: loss_fn_with_jac_ls_coefs(coef_, coefs_init_incl_intercept)

            coefs_init__ = coefs_init_incl_intercept
            print('init coefs: ',coefs_init_incl_intercept)
            solve_init =  minimize(loss_fn_init_jac_,jac=True, x0 = coefs_init__,
                       method = 'SLSQP',
                       options = {'maxiter':1000,
                                  # 'ftol':1e-4,
                                  },
                       constraints = constraints)
            if not solve_init['success']:
                print('####### COULD NOT SOLVE INIT COEFS')
                print(solve_init)
            else:
                coefs_init_incl_intercept = solve_init['x']

            res = solve('trust-constr',10000,coefs_init_incl_intercept)
            if not res['success']:
                print(res)

        self.coef_ = res['x']
        self.intercept_ = 0.
        if self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:coef_end_idx]


    def predict(self, X):
        X_ = X.copy()
        coef_ = self.coef_
        if self.fit_intercept:
            X_ = np.hstack([ X_,np.ones([X.shape[0],1])])
            coef_ = np.array(list(coef_)+[self.intercept_])
        return np.dot(X_, coef_).ravel()

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` w.r.t. `y`.

        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """

        # from .metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def get_params(self, deep=False):
        dict = {
            'monotone_constraints' : self.monotone_constraints,
            'alpha': self.alpha,
        }
        return dict

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

