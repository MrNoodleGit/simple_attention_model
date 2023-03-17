import argparse

from model_scripts.modelfuncs.get_prior_samples import get_prior_samples
from model_scripts.modelfuncs.main_loop import main_loop

from model_scripts.helperfuncs.prep_data import prep_data

from model_scripts.helperfuncs.param_funcs import get_params
from model_scripts.helperfuncs.param_funcs import get_sim_settings

from model_scripts.helperfuncs.save_model_outputs import save_behavior
from model_scripts.helperfuncs.save_model_outputs import save_plots

def run_model(args): 

    # get parameters 
    params = get_params(args.param_names_path, args.param_values_path) #DONE

    # get sim settings 
    sim_settings = get_sim_settings(args.sim_settings_path) #DONE
    
    # prepare data dict 
    #TODO
    #should check sim settings to choose between generating random data or get data from the embeddings
    data = prep_data(params, sim_settings)

    # generate prior samples
    # TODO
    # -get the code from OLD action loop notebook
    # -we need this just for the first step so we can compare prior and posterior
    prior_samples = get_prior_samples(params)

    # run main loop and return model behavior
    # TODO
    # - get code from action loop and numpyro_gpu_script
    model_behavior = main_loop(params, data, prior_samples, num_model_runs)
    
    # save model behaviors
    if args.save_behavior:
        save_behavior(params)

    # save plots of model behavior
    if args.save_plots:
        save_plots(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("param_values_path", type=str, help="Path to csv with parameters")
    parser.add_argument("param_names_path", type=str, help="Path to csv with parameter names")
    parser.add_argument("sim_settings_path", type=str, default=True, help="where other simulation parameters are")

    parser.add_argument("--save_behavior", type=str, default=True, help="whether to save model behavior")
    parser.add_argument("--save_plots", type=str, default=True, help="whether to visualize or not (default not)")

    args = parser.parse_args()
    run_model(args)
    