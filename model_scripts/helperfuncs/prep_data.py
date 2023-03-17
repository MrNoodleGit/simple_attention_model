import numpy as np

def prep_data(params, sim_settings):

    sim_data = generate_data(sim_settings)
    
    model_input = {"mean_mu": params[0], "mean_sd": params[1], "sd_scale": params[2], "noise_scale": params[3], 
    "F": sim_settings['num_features']}

    return model_input, sim_data;

def generate_data(sim_settings):
        """generate data for model to see"""

        # preallocate sim data to appropriate size
        sim_data = np.empty(sim_settings['num_model_runs'], # TODO remove this dimension
                        len(sim_settings['deviant_pos']), 
                            sim_settings['num_features'], 
                            sim_settings['max_num_samples'])
        
        # allocation of samples to exemplars (1,1,1,2,2,2,2,...)
        exemplar_idx = np.repeat(np.arange(1, sim_settings['sequence_length']+1), 
                                            sim_settings['num_samples']/sim_settings['sequence_length'])
        
        # model runs
        for model_run in range(0, sim_settings['num_model_runs']):

            for pos_num, deviant_pos in enumerate(sim_settings['deviant_pos']):
                exemplar_means = np.tile(sim_settings['background_mean'], (sim_settings['sequence_length'], 1))
                exemplar_means[deviant_pos-1] = sim_settings['deviant_mean']

                # perceptual noise
                sig = np.identity(sim_settings['num_features']) * 1;

                # generate the data for each model run and deviant position
                sim_data = [np.random.multivariate_normal(exemplar_means[idx-1], sig) for idx in exemplar_idx] 
                sim_data[model_run, pos_num, :, :] = np.asmatrix(sim_data)

        return sim_data
        