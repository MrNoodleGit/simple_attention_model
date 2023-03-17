from scipy.stats import halfcauchy
import numpy as np

def get_prior_samples(params, sim_settings):

    prior_mu = np.random.multivariate_normal(np.repeat(params['mean_mu'], sim_settings['num_features']), np.identity(sim_settings['num_features'])*params['mean_sd'], sim_settings['MCMC_sample_num']-sim_settings['MCMC_warmup'])
    prior_sigma = halfcauchy.rvs(loc = 0, scale = params['sd_scale'], size = (sim_settings['MCMC_sample_num']-sim_settings['MCMC_warmup'],sim_settings['num_features']))
        
    ## MIGHT NOT BE NECESSARY
    prior_z_rep = np.empty((sim_settings['MCMC_sample_num']-sim_settings['MCMC_warmup'], sim_settings['num_features']))
    for i in np.arange(0, sim_settings['MCMC_sample_num']-sim_settings['MCMC_warmup']):
        prior_z_rep[i,:] = np.random.multivariate_normal(prior_mu[i,:], np.identity(sim_settings['num_features'])*prior_sigma[i,:])

    prior_samples = np.hstack((prior_mu, prior_sigma))

    return prior_samples 