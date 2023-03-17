def main_loop(parameters, data, prior_samples, num_model_runs):
    model_behavior = None # this is the output of the main_loop

    for run in range(num_model_runs):
    
        print('model run: ', run)

        # generate the data
        # data = np.asmatrix(data)

        # Iterators
        samples_from_current_stim = 1
        total_samples = 1
        exemplar_num = 1

        # initialize empty data container
        sample_data = np.empty((num_samples,num_features))
        sample_data[:] = np.nan

        exemplar_labels = np.empty((num_samples,))
        exemplar_labels[:] = np.nan

        while sample or samples_from_current_stim > 1:
            
            print('stimulus: ', exemplar_num)
            
            print('sample: ', total_samples)

            
            # sample number
            data["M"] = total_samples

            # exemplar number 
            data["K"] = exemplar_num

            # add sim data
            sample_data[total_samples-1] = data[exemplar_idx == exemplar_num][samples_from_current_stim-1]
            data["z"] = np.transpose(sample_data[0:total_samples,:])

            # add exemplar for each id
            exemplar_labels[total_samples-1] = int(exemplar_num)
            data["exemplar_idx"] = [int(x) for x in exemplar_labels[~np.isnan(exemplar_labels)]]

            # get posterior samples
            # fit = sm.sampling(data=data, iter=num_iter, chains=1, warmup = num_warmup,control=dict(adapt_delta=0.95)); # TODO
            
            
            # posterior = np.hstack((fit['mu'][0:len(fit['mu']):thinning_factor], \
            #                        fit['sigma'][0:len(fit['mu']):thinning_factor])) # TODO

            vi = model.variational(data=data, output_samples=num_iter)

            posterior = vi.variational_sample

            break

            
            # fit gmms
            gmm_p = GaussianMixture(n_components=2, random_state=0).fit(posterior)
            gmm_q = GaussianMixture(n_components=2, random_state=0).fit(prior)

            if total_samples < 30:
                
                plt.figure(1)
                plt.rcParams['figure.figsize'] = [18, 12]
                plt.subplot(4,5,total_samples)
                plt.hist(np.hstack((prior[:,[0]],posterior[:,[0]])), bins = 100);
                plt.title(str(total_samples) + " , " + str(exemplar_num))
                plt.legend('prior', 'posterior')
                
                plt.figure(2)
                plt.rcParams['figure.figsize'] = [18, 12]
                plt.subplot(4,5,total_samples)
                plt.hist(np.hstack((prior[:,[1]],posterior[:,[1]])), bins = 100);
                plt.title(str(total_samples) + " , " + str(exemplar_num))
                plt.legend('prior', 'posterior')
            
            
            else: 
                plt.show()
                break;
                
            if policy is 'kl':

                X = gmm_p.sample(posterior.shape[0])

                log_p_X = gmm_p.score_samples(X[0])
                log_q_X = gmm_q.score_samples(X[0])

                # KL divergence between prior and posterior
                stim_info[run, total_samples-1] = log_p_X.mean() - log_q_X.mean()
                
                print('KL:',  stim_info[run,total_samples-1])

            elif policy is 'entropy':
                # reduction of entropy
                stim_info[run,total_samples-1] = np.abs(continuous.get_h(prior, k= 250) - continuous.get_h(posterior, k = 250))
                
                print('entropy change:',  stim_info[run,total_samples-1])
                
            elif policy is 'surprisal':   
                                
                # surprisal of current observation given prior
                stim_info[run, total_samples-1] = surprisal(np.squeeze(fit['z_rep'], axis = 1), sample_data[total_samples-1])
                
                print('surprisal:',  stim_info[run,total_samples-1])

            elif policy is 'EIG':
                
                hypothethical_grid = np.arange(-2, 2, 0.1)
                
                stim_info[run,total_samples-1] = EIG(posterior)

            # decision rule
            if stim_info[run,total_samples-1] < env_info:
                model_LT[run, exemplar_num-1] = samples_from_current_stim

                # reset/increment counters
                samples_from_current_stim = 1
                exemplar_num += 1

                if exemplar_num > sequence_length:
                    sample = False

            else:
                samples_from_current_stim += 1 

            if policy is 'kl' or policy is 'surprisal' or policy is 'entropy':
                prior = posterior
            
            total_samples += 1

        # start sampling for next model run
        sample = True

        break
        
        return model_behavior