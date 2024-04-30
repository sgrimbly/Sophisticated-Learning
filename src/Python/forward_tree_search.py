import numpy as np
import copy
from math_utils import normalise_matrix_columns, normalise_vector, round_half_up
from active_inference_utils import calculate_posterior, determine_observation_preference, calculate_novelty, G_epistemic_value
from spm_utils import spm_cross

# TODO (St John): Debug issue related to searching depth when compared to Matlab version. #bug

def forward_tree_search_SI(short_term_memory, O, P, a, A, y, B, b, imagined_t, search_horizon, time_since_resource, true_t, chosen_action, best_actions, weights, num_modalities, num_factors, num_states, num_resource_observations, G_prior, resource_constraints, memory_accessed):
    imagined_O = copy.deepcopy(O)
    imagined_P = copy.deepcopy(P)
    P_prior = copy.deepcopy(imagined_P) # Set to Q initially
    imagined_time_since_resource = copy.deepcopy(time_since_resource)
    imagined_chosen_action = copy.deepcopy(chosen_action)
    
    P = calculate_posterior(copy.deepcopy(imagined_P),y,imagined_O)
    G = G_prior
    
    # NOTE Doesn't seem necessary, don't think b changes
    bb = copy.deepcopy(b)
    bb[1] = normalise_matrix_columns(b[1])

    if imagined_t > true_t:
        # Refer to sophisticated inference paper for this logic
        
        # Increment the 'imagined' time since resource. This is useful since we are recursing down a level each time and we don't want to lose track of the time since resource.
        for resource, time_since in imagined_time_since_resource.items():
            imagined_time_since_resource[resource] = round_half_up(time_since + 1)

        # Calculate expected free energy (EFE)
        # 1. Epistemic term:
        epi = G_epistemic_value(y, P_prior)
        G += weights['Epistemic']*epi
        
        # 2. Novelty term:
        novelty = calculate_novelty(P, a, imagined_O, num_factors, weights)
        G += weights['Novelty']*novelty
        
        # 3. Pragmatic/extrinsic/reward term:
        C = determine_observation_preference(copy.deepcopy(imagined_time_since_resource), resource_constraints, weights)
        prag = imagined_O[1] @ C.T
        G += prag
        
        # Update the number of time steps since the agent last consumed a resource, based on its imagined observations
        imagined_time_since_resource['Food'] = round_half_up(imagined_time_since_resource['Food'] * (1 - imagined_O[1][1]))
        imagined_time_since_resource['Water'] = round_half_up(imagined_time_since_resource['Water'] * (1 - imagined_O[1][2]))
        imagined_time_since_resource['Sleep'] = round_half_up(imagined_time_since_resource['Sleep'] * (1 - imagined_O[1][3]))
        
        
    
    if imagined_t < search_horizon:
        # TODO: #rowan Why does this list of actions need to be randomised?
        #actions = np.random.permutation(5)  
        actions = np.arange(5)
        efe = np.array([0, 0, 0, 0, 0], dtype=float)
        for action in actions:
            Q_action = []
            Q_action.append((bb[0][:,:,action] @ P[0].T).T)
            Q_action.append((bb[1][:,:,0] @ P[1].T).T)
            qs = spm_cross(Q_action).ravel(order='F')

            # Find the indices where 'qs' is greater than 1/8
            likely_states = np.where(qs > 1/8)[0]

            # Check if 'likely_states' is empty
            if likely_states.size == 0:
                threshold = 1 / qs.size ** 2
                likely_states = np.where(qs > (1 / qs.size - threshold))[0]
            K = np.zeros(num_states*num_resource_observations)

            for state in likely_states:
                if short_term_memory[imagined_time_since_resource['Food'], \
                                    imagined_time_since_resource['Water'], \
                                    imagined_time_since_resource['Sleep'], state] != 0:
                    sh = short_term_memory[imagined_time_since_resource['Food'], \
                                    imagined_time_since_resource['Water'], \
                                    imagined_time_since_resource['Sleep'], state]
                    K[state] = sh
                else:
                    for modality in range(num_modalities):
                        imagined_O[modality] = normalise_vector(
                            y[modality].reshape(y[modality].shape[0],-1, order='F')[:,state]
                        )
                    
                    # Prior over next states given transition function (calculated earlier)
                    expected_free_energy, short_term_memory, best_actions, memory_accessed = forward_tree_search_SI(short_term_memory, imagined_O, Q_action, a, A, y, B, b, imagined_t+1, search_horizon, imagined_time_since_resource, true_t, imagined_chosen_action, best_actions, weights, num_modalities, num_factors, num_states, num_resource_observations, G_prior, resource_constraints, memory_accessed)
                    S = np.max(expected_free_energy)
                    K[state] = S
                    memory_accessed +=1
                    short_term_memory[imagined_time_since_resource['Food'], \
                                    imagined_time_since_resource['Water'], \
                                    imagined_time_since_resource['Sleep'], state] = S

            action_fe = K[likely_states] @ qs[likely_states]
            efe[action] += 0.7 * np.sum(action_fe)
            
        # Find the action with the maximum expected free energy
        imagined_chosen_action = np.argmax(efe)
        maxi = efe[imagined_chosen_action]

        # Update the global measure
        G += maxi  # Assuming 'G' is a variable tracking the global total.

        # Update the list of best actions
        best_actions.insert(0, imagined_chosen_action)  # Assuming 'best_actions' is a list; adds 'chosen_action' to the beginning.

    return G, short_term_memory, best_actions, memory_accessed
    
    
def forward_tree_search_SL():
    pass