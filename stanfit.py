"""
A wrapper around PyStan's pystan.stan fitting method, providing a somewhat
more "Pythonic" interface to the fit results.

For PyStan info:

https://pystan.readthedocs.org/en/latest/getting_started.html

Created 2014-11-04 by Tom Loredo
"""

import matplotlib.pyplot as plt
import pystan


class ParamHandler(dict):
    """
    A container and handler for posterior sample data for a scalar parameter.

    This is mostly a dict-like object with access to data also possible via
    attributes, based on AttrDict from:

    http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
    """

    def __init__(self, *args, **kwargs):
        if not kwargs.has_key('fit'):
            raise ValueError('fit argument required!')
        if not kwargs.has_key('name'):
            raise ValueError('name argument required!')
        super(ParamHandler, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def trace(self, axes=None, xlabel=None, ylabel=None, **kwds):
        if axes is None:
            fig = plt.figure(figsize=(10,4))
            fig.subplots_adjust(bottom=.2, top=.9)
            axes = plt.subplot(111)
        axes.plot(self.chain, **kwds)
        if xlabel:
            axes.set_xlabel(xlabel)
        else:
            axes.set_xlabel('Sample #')
        if ylabel:
            axes.set_ylabel(ylabel)
        else:
            axes.set_ylabel(self.name)


class StanFit:
    """
    Helper class for PyStan model fitting.
    """

    # These keys are from the raw summary col names; hope they won't change!
    # Map them to valid Python attribute names.
    col_map = {'mean':'mean',
               'se_mean' : 'se_mean',
               'sd' : 'sd',
               '2.5%' : 'q025',
               '25%' : 'q25',
               '50%' : 'median',
               '75%' : 'q75',
               '97.5%' : 'q975',
               'n_eff' : 'ess',
               'Rhat' : 'Rhat'}

    def __init__(self, code, data, n_chains, n_iter, **kwds):
        """
        Run an initial Stan fit.

        Parameters
        ----------
        code : string
            String containing the Stan code for the model
        data : dict
            Dict of data corresponding to the model's data block
        n_chains : int
            Number of posterior sampler chains to run
        n_iter : int
            Number of iterations per chain for the initial run
        """
        self.code = code
        self.data = data
        self.n_chains = n_chains
        self.n_iter = n_iter
        # The actual fit!
        fit = pystan.stan(model_code=code, data=data, chains=n_chains,
                               iter=n_iter, **kwds)
    
        self.par_names = fit.model_pars  # unicode param names
        self.par_dims = {}
        for name, dim in zip(self.par_names, fit.par_dims):
            self.par_dims[name] = dim
    
        # Collect attribute names for storing param info, protecting from name
        # collision in this namespace.
        # *** Note this doesn't protect against future collisions! ***
        par_attr_names = {}
        for name in self.par_names:
            if hasattr(self, name):
                name_ = name + '_'
                if hasattr(self, name_):
                    raise ValueError('Cannot handle param name collision!')
                print '*** Access param "{0}" via "{0}_". ***'.format(name)
                par_attr_names[name] = name_
            else:
                par_attr_names[name] = name
        self.par_attr_names = par_attr_names

        # Collect info from the fit that shouldn't change if the fit is
        # extended.
        raw_summary = fit.summary()
        # Column names list the various types of statistics.
        self.sum_cols = raw_summary['summary_colnames']
        # Get indices into the summary table for the columns.
        self.col_indices = {}
        for i, name in enumerate(self.sum_cols):
            self.col_indices[name] = i
        # Row names list the parameters; convert from an ndarray to a list.
        self.sum_rows = [name for name in raw_summary['summary_rownames']]
        # Get indices for params; for vectors store the offset for 0th entry.
        self.par_indices = {}
        for name in self.par_names:
            if not self.par_dims[name]:  # scalar param
                self.par_indices[name] = self.sum_rows.index(name)
            else:  # vector
                self.par_indices[name] = self.sum_rows.index(name+'[0]')
        # Index for log_p:
        self.logp_indx = self.sum_rows.index('lp__')

        self._update_fit(fit, raw_summary)

    def _update_fit(self, fit, raw_summary):
        """
        Update attributes with results of a fit (an initial fit or an
        extension of a previous fit).
        """
        self.fit = fit
        self.chains = self.fit.extract(permuted=True)
        self.raw_summary = raw_summary  # dict of fit statistics
        self.summary = raw_summary['summary']

        # Populate namespace with handlers for each param, holding
        # various data from the fit.
        for name in self.par_names:
            attr_name = self.par_attr_names[name]
            row = self.par_indices[name]
            if not self.par_dims[name]:  # scalar param
                param = ParamHandler(fit=self.fit, name=name)
                setattr(self, attr_name, param)
                param['chain'] = self.chains[name]
                for stat in self.sum_cols:
                    col = self.col_indices[stat]
                    param[self.col_map[stat]] = self.summary[row,col]
                # 95% central credible interval:
                param['intvl95'] = (param['q025'], param['q975'])
            elif len(self.par_dims[name]) == 1:  # vector param
                l = []
                for i in xrange(self.par_dims[name][0]):
                    param = ParamHandler(fit=self.fit, name=name+'[%i]'%i)
                    param['chain'] = self.chains[name][:,i]
                    for stat in self.sum_cols:
                        col = self.col_indices[stat]
                        param[self.col_map[stat]] = self.summary[row+i,col]
                    param['intvl95'] = (param['q025'], param['q975'])
                    l.append(param)
                setattr(self, attr_name, l)
            else:
                # Could just direct user to summary attribute...
                raise NotImplementedError('Only scalar & vector params supported!')

        # Make a handler for log_p.
        param = ParamHandler(fit=self.fit, name='log_p')
        setattr(self, 'log_p', param)
        param['chain'] = self.chains['lp__']
        for stat in self.sum_cols:
            col = self.col_indices[stat]
            param[self.col_map[stat]] = self.summary[-1,col]
        # 95% central credible interval:
        param['intvl95'] = (param['q025'], param['q975'])

    def refit(self, n_iter=None, n_chains=None, data=None, **kwds):
        """
        Refit the current model, potentially using new data.

        The argument order was chosen to make it easiest to refit the same
        data with a longer run of the sampler; refit(n) does this.

        This skips the model compilation step, but otherwise runs a fresh
        fit.
        """
        if n_iter is None:
            n_iter = self.n_iter
        else:
            self.n_iter = n_iter
        if data is None:
            data = self.data
        else:
            self.data = data
        if n_chains is None:
            n_chains = self.n_chains
        else:
           self.n_chains = n_chains
        self.n_iter = n_iter
        # The actual fit!
        fit = pystan.stan(fit=self.fit, data=self.data, chains=self.n_chains,
                               iter=self.n_iter, **kwds)
        self._update_fit(fit, fit.summary())

    def stan_plot(self):
        self.fit.plot()

    def __str__(self):
        return str(self.fit)
