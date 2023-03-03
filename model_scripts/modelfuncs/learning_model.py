import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random, local_device_count
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

def run_inference(current_data, sim_data, params, sim_settings, model_run, pos_idx, exemplar_idx, sample_idx):
    
    nuts_kernel = NUTS(
    learning_model,
    target_accept_prob=sim_settings['accept_prob'],
    adapt_step_size=True,
    adapt_mass_matrix=False,
    max_tree_depth=sim_settings['max_tree_depth'],
    find_heuristic_step_size=True,
    )

    # find idx that corresponds to current sample
    sample_from_sim = np.shape(sim_data)[2] / sim_settings['exemplar_num'] * exemplar_idx + sample_idx 

    # append current sample to data
    current_data = np.append(current_data,sim_data[model_run, pos_idx, sample_from_sim, :])

    mcmc = MCMC(nuts_kernel, num_warmup=sim_settings['MCMC_warmup'], num_samples=sim_settings['MCMC_sample_num'], 
                num_chains=4, chain_method='parallel')
                
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, z=current_data, num_samples = np.shape(current_data)[1], exemplar_idx = exemplar_idx, 
            mean_mu= params['mean_mu'], mean_sd=params['mean_sd'], sd_scale=params['sd_scale'], noise_scale=params['noise_scale'], extra_fields=('potential_energy',))


def learning_model(z=None, num_samples=None, exemplar_idx=None, mean_mu=None, mean_sd=None, sd_scale=None, noise_scale=None):

    # Perceptual noise
    noise = numpyro.sample("noise", dist.HalfCauchy(noise_scale))

    # Concept
    mu = numpyro.sample("mu", dist.Normal(mean_mu, mean_sd))
    tau = numpyro.sample("tau", dist.HalfCauchy(sd_scale))

    with numpyro.plate("num_stim", exemplar_idx, dim=-1):
        with numpyro.handlers.reparam(config={'theta': TransformReparam()}):
            theta = numpyro.sample(
                'theta',
                dist.TransformedDistribution(
                    dist.Normal(0., 1.),
                    dist.transforms.AffineTransform(mu, tau)))

        with numpyro.plate("obs", num_samples, dim=-2):
            numpyro.sample("z", dist.Normal(theta, noise), obs=z)