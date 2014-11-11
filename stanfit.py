"""
A wrapper around PyStan's compilation and fitting methods, providing a somewhat
more "Pythonic" interface to the fit results.

For PyStan info:

https://pystan.readthedocs.org/en/latest/getting_started.html

Created 2014-11-04 by Tom Loredo
"""

import cPickle, glob
from hashlib import md5

import numpy as np
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

    def trace(self, chain=None, step=True, axes=None,
                    xlabel=None, ylabel=None, **kwds):
        """
        Make a trace plot for the samples in chain `chain`.  If `chain` is None,
        show traces for all chains, iterating colors accorting to mpl defaults.

        By default, a step plot is used; set `step` to False for a line plot.
        """
        if axes is None:
            fig = plt.figure(figsize=(10,4))
            fig.subplots_adjust(bottom=.2, top=.9)
            axes = plt.subplot(111)
        if chain is None:
            if step:
                times = xrange(self.chains.shape[0])
                for c in range(self.chains.shape[1]):
                    axes.step(times, self.chains[:,c], where='pre',
                              label='%i'%c, **kwds)
            else:
                for c in range(self.chains.shape[1]):
                    axes.plot(self.chains[:,c], **kwds)
        else:
            if step:
                times = xrange(self.chains.shape[0])
                axes.step(times, self.chains[:,chain], where='pre',
                          label='%i'%c, **kwds)
            else:
                axes.plot(self.chains[:,chain], **kwds)
        if xlabel:
            axes.set_xlabel(xlabel)
        else:
            axes.set_xlabel('Sample #')
        if ylabel:
            axes.set_ylabel(ylabel)
        else:
            axes.set_ylabel(self.name)
        if chain is None:
            axes.legend(fontsize='small', labelspacing=.2, borderpad=.3)
        axes.figure.show()  # needed for display update with axes


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

    def __init__(self, source, data=None, n_chains=None, n_iter=None, 
                 name=None, **kwds):
        """
        Run an initial Stan fit.

        Parameters
        ----------
        source : string
            Path to a file (ending with ".stan") containing the Stan code for
            a model, or a string containing the code itself
        data : dict
            Dict of data corresponding to the model's data block
        n_chains : int
            Number of posterior sampler chains to run
        n_iter : int
            Number of iterations per chain for the initial run
        """
        if source.count('\n') == 0 and source[-5:] == '.stan':
            with open(source, 'r') as sfile:
                self.code = sfile.read()
        else:
            self.code = source
        self.name = name
        self._compile()

        self.data = data
        self.n_chains = n_chains
        self.n_iter = n_iter

        if data:
            self._set_data(data)

        # The actual fit!
        if data is not None and n_chains is not None and n_iter is not None:
            self.sample(data=data, chains=n_chains, iter=n_iter, **kwds)
        else:
            self.fit = None
    
    def _compile(self):
        """
        Compile a Stan model if necessary, loading a previously compiled
        version if available.
        """
        code_hash = md5(self.code.encode('ascii')).hexdigest()
        files = glob.glob('*-'+code_hash+'.pkl')
        if files:
            if len(files) != 1:
                raise RuntimeError('Cache collision---multiple matching cache files!')
            cache_path = files[0]
            self.name, self.model = cPickle.load(open(files[0], 'rb'))
            print 'Using cached StanModel "{0}" from {1}.'.format(self.name, files[0])
        else:
            self.model = pystan.StanModel(model_code=self.code)
            if self.name is None:
                cache_path = 'cached-model-{}.pkl'.format(code_hash)
            else:
                cache_path = 'cached-{}-{}.pkl'.format(name, code_hash)
            with open(cache_path, 'wb') as f:
                cPickle.dump((self.name, self.model), f)

    def _set_data(self, data):
        """
        Set the data info dictionary, and collect info about parameters for an
        application of the model to the dataset.

        Note that since hierarchical models are supported by Stan, the
        parameter space may not be completely defined until a dataset is
        specified (the dataset size determines the number of latent
        parameters in hierarchical models).
        """
        self.fit = self.model.fit_class(self.data)
        self.par_names = self.fit._get_param_names()  # unicode param names
        self.par_dims = {}
        for name, dim in zip(self.par_names, self.fit._get_param_dims()):
            self.par_dims[name] = dim

        # Make an index for accessing chains in fit.extract() results.
        # Note that 'lp__' is included here, and used in _make_param_handler.
        indx = 0
        self.par_indx = {}
        for name in self.par_names:
            self.par_indx[name] = indx
            dims = self.par_dims[name]
            if dims:
                indx += np.prod(dims)
            else:
                indx += 1  # technically could use prod(dims)=1. for dims=[]
        # self.log_p_indx = self.par_indx['lp__']

        # Stan includes log_prob in the param list; we'll track it separately
        # so remove it from the param info.
        indx_of_lp = self.par_names.index('lp__')
        del self.par_names[indx_of_lp]
        del self.par_dims['lp__']
        # del self.par_indx['lp__']

        # Collect attribute names for storing param info, protecting from name
        # collision in this namespace.
        # *** This doesn't protect against subsequent collision/overwriting. ***
        # TODO:  Make sure all needed class attributes are defined before this
        # runs, or otherwise protected.
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

    def _get_table_info(self):
        """
        Get information about the summary table from a fit to the current data.

        This information (largely dimensional/indexing) does not actually
        require a fit to be done, but it is only available from Stan post-fit.
        """
        # Collect info from the fit that shouldn't change if the fit is
        # re-run.
        self.raw_summary = self.fit.summary()  # dict of fit statistics (Rhat, ess...)
        # Column names list the various types of statistics.
        self.sum_cols = self.raw_summary['summary_colnames']
        # Get indices into the summary table for the columns.
        self.col_indices = {}
        for i, name in enumerate(self.sum_cols):
            self.col_indices[name] = i
        # Row names list the parameters; convert from an ndarray to a list.
        self.sum_rows = [name for name in self.raw_summary['summary_rownames']]
        # Get indices for params; for vectors store the offset for 0th entry.
        self.par_indices = {}
        for name in self.par_names:
            if not self.par_dims[name]:  # scalar param
                self.par_indices[name] = self.sum_rows.index(name)
            else:  # vector
                self.par_indices[name] = self.sum_rows.index(name+'[0]')

    def _update_fit_results(self):
        """
        Update attributes with results from the current fit.
        """
        # Extract chains, kept separate and ordered (permuted=False), with
        # burn-in discarded (inc_warmup=False), as an array indexed as
        # [sample #, chain #, param #]; note that log_p is added to the end
        # of the param list.
        self.chains = self.fit.extract(permuted=False)
        # Collect samples from the chains, merged via random permutation
        # (permuted=True), with burn-in discarded (inc_warmup=False), as a
        # param-keyed dict.
        self.samples = self.fit.extract(permuted=True)
        self.summary = self.raw_summary['summary']

        # Populate namespace with handlers for each param, holding
        # various data from the fit.
        for name in self.par_names:
            attr_name = self.par_attr_names[name]
            row = self.par_indices[name]
            if not self.par_dims[name]:  # scalar param
                param = self._make_param_handler(name, row)
                setattr(self, attr_name, param)
            elif len(self.par_dims[name]) == 1:  # vector param as list attr
                l = []
                for i in xrange(self.par_dims[name][0]):
                    param = self._make_param_handler(name, row, i)
                    l.append(param)
                setattr(self, attr_name, l)
            else:
                # Could just direct user to summary attribute...
                raise NotImplementedError('Only scalar & vector params supported!')

        # Make a handler for log_p, the last "parameter" in the Stan table.
        param = self._make_param_handler('log_p')
        setattr(self, 'log_p', param)
        param['chain'] = self.samples['lp__']
        for stat in self.sum_cols:
            col = self.col_indices[stat]
            param[self.col_map[stat]] = self.summary[-1,col]
        # 95% central credible interval:
        param['intvl95'] = (param['q025'], param['q975'])

    def _make_param_handler(self, name, row=None, item=None, log_p=False):
        """
        Create a ParamHandler instance for parameter name `name` and make
        it an attribute, using data from (row,col) in the fit summary table.

        Call with (name, row) for a scalar parameter.

        Call with (name, row, item) for an element of a vector parameter.

        Call with (name, log_p=True) for log_prob.
        """
        # Set the key to use for Stan table lookups.
        if name == 'log_p':
            key = 'lp__'
        else:
            key = name

        # Scalars and vectors handle names differently; vectors use `item`.
        if item is None:  # scalar case
            pname = name  # name to store in the handler
            prow = row
            samples = self.samples[key]
            chains = self.chains[:,:,self.par_indx[key]]
        else:  # vector case
            pname = name + '[%i]' % item
            prow = row + item
            samples = self.samples[key][:,item]
            chains = self.chains[:,:,self.par_indx[key]+item]

        param = ParamHandler(fit=self.fit, name=pname)
        param['samples'] = samples
        param['chains'] = chains
        for stat in self.sum_cols:
            col = self.col_indices[stat]
            param[self.col_map[stat]] = self.summary[prow,col]
        # 95% central credible interval:
        param['intvl95'] = (param['q025'], param['q975'])
        return param

    def sample(self, n_iter=None, n_chains=None, data=None, **kwds):
        """
        Run a posterior sampler using the compiled model, potentially using new
        data.

        The argument order was chosen to make it easiest to refit the same
        data with a longer run of the sampler; sample(n) does this.

        This skips the model compilation step, but otherwise runs a fresh
        MCMC chain.
        """
        if n_iter is None:
            n_iter = self.n_iter
        else:
            self.n_iter = n_iter
        if data is not None:
            self._set_data(data)
        if n_chains is None:
            n_chains = self.n_chains
        else:
           self.n_chains = n_chains
        self.n_iter = n_iter
        # The actual fit!
        self.fit = self.model.sampling(data=self.data, chains=self.n_chains,
                                      iter=self.n_iter, **kwds)
        # fit = pystan.stan(fit=self.fit, data=self.data, chains=self.n_chains,
        #                        iter=self.n_iter, **kwds)
        self._get_table_info()  # *** only needed for 1st fit to a data set
        self._update_fit_results()

    def stan_plot(self):
        self.fit.plot()

    def __str__(self):
        return str(self.fit)
