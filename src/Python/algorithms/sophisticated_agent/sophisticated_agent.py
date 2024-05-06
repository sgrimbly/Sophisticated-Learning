# Importing standard library
import numpy as np
import random
import copy
import matplotlib
matplotlib.use('Qt5Agg')  # Choose the appropriate backend like 'Qt5Agg', 'Qt4Agg', etc.
import matplotlib.pyplot as plt

from datetime import datetime
import logging
import sys
sys.path.append('..')  # Adjust to go up two levels

# Importing local modules
from agent_utils import (
    update_environment, 
    is_alive,
    update_needs, 
    get_observations, 
    get_predicted_posterior,
    get_actual_posterior, 
    delete_short_term_memory,
    smooth_beliefs
)
from agent_utils import initialise_distributions
from forward_tree_search import forward_tree_search
from display import draw_gridworld
from math_utils import round_half_up

def initialise_experiment():
    # Initialize short_term_memory array
    # TODO: Why 35 shape? Maybe make this a hyperparameter in the notebook
    short_term_memory = np.zeros((35, 35, 35, 400))
    time_since_resource = {"Food": 0, "Water": 0, "Sleep": 0}
    # Initializing chosen_action
    chosen_action = []
    
    # Data structures for storing historical/previous agent models/observations and true states of the environment
    historical_predictive_observations_posteriors = [] # Agent belief about next state -> for state-prediction error calculation
    historical_agent_observations = [] 
    historical_agent_posterior_Q = [] 
    historical_true_states = [] 
    historical_chosen_action = []

    return chosen_action, short_term_memory, historical_predictive_observations_posteriors, historical_agent_observations, historical_agent_posterior_Q, historical_true_states, historical_chosen_action, time_since_resource

# Initialize the timers for each resource and the main time counter
def agent_loop(
    algorithm,
    a, 
    resource_constraints, 
    A, 
    B, 
    b, 
    D,
    t_constraint, 
    contextual_food_locations, 
    contextual_sleep_locations, 
    contextual_water_locations, 
    num_modalities, 
    num_states,
    num_factors,             
    num_contextual_states,   
    num_resource_observations,  
    num_context_observations,   
    hill_1,             
    weights,          
    G_prior,
    start_position,
    visualise,
    ax = None       
): 
    chosen_action, short_term_memory, historical_predictive_observations_posteriors, historical_agent_observations, historical_agent_posterior_Q, historical_true_states, historical_chosen_action, time_since_resource = initialise_experiment()
    
    trial_data = {
        "historical_predictive_observations_posteriors": [],
        "historical_agent_observations": [],
        "historical_agent_posterior_Q": [],
        "historical_true_states": [],
        "historical_chosen_action": [],
        "time_since_resource": [],
        "hill_memory_resets": [],
        "pe_memory_resets": [],
        "time_steps": [],
        "search_depth": [],
        "memory_accessed": []
    }
    
    # Main loop continues as long as time doesn't exceed constraint and agent's needs are met
    t = 0
    search_depth = 0
    pe_memory_resets = 0
    hill_memory_resets = 0
    memory_accessed = 0
    while t < t_constraint and is_alive(time_since_resource, resource_constraints):
    
        # Update the agent's model and the environment based on the last action
        historical_agent_posterior_Q, historical_true_states = update_environment(
            b, D, t, historical_chosen_action, historical_true_states, historical_agent_posterior_Q, start_position
        )
        Q = copy.deepcopy(historical_agent_posterior_Q[t])
        current_pos = np.argmax(np.cumsum(Q[0]) >= np.random.rand())
         
        # Update the timers for each resource based on the agent's actions
        time_since_resource = update_needs(
            historical_true_states, t, 
            contextual_food_locations, 
            contextual_water_locations, 
            contextual_sleep_locations, 
            time_since_resource
        )

        # Get the current observation from the environment
        O = get_observations(
            A, historical_true_states, t, num_modalities, num_states, num_resource_observations, num_context_observations
        )
        historical_agent_observations.append(O)

        # Determine predicted posterior
        rounded_predicted_P, historical_predictive_observations_posteriors = get_predicted_posterior(a, Q, O, historical_predictive_observations_posteriors)

        # Update the agent's beliefs, including backward smoothing for more accurate inferences
        smoothing_start = 0 if t <= 6 else t - 6
        for smoothing_t in range(smoothing_start, t+1): # In this case range() needs to be inclusive of t
            a = smooth_beliefs(historical_agent_observations, historical_agent_posterior_Q, A, a, b, smoothing_start, smoothing_t, t)
            
        # Determine actual posterior
        rounded_actual_P, P, y = get_actual_posterior(a, Q, O)

        # If there is a context prediction error larger than .1, or if the agent is on the hill, reset short-term memory
        no_state_prediction_error = np.array_equal(rounded_predicted_P, rounded_actual_P)  # This returns True if the two arrays have the same shape and elements, False otherwise
        if not no_state_prediction_error or current_pos == hill_1:
            short_term_memory = delete_short_term_memory()
            if not no_state_prediction_error: pe_memory_resets +=1
            if current_pos == hill_1: hill_memory_resets += 1

        # Determine the horizon for tree search based on the agent's current needs
        needs = {
            "Food":resource_constraints['Food'] - time_since_resource['Food'], 
            "Water":resource_constraints['Water'] - time_since_resource['Water'],
            "Sleep":resource_constraints['Sleep'] - time_since_resource['Sleep']
        }
        horizon = max(
            1, 
            min(
                9,
                needs["Food"],
                needs["Water"],
                needs["Sleep"]
            )
        )
        
        # Initialize best actions and perform tree search to find the best action
        best_actions = []
        tree_search_call_count = 0
        G, short_term_memory, best_actions, memory_accessed, tree_search_call_count = forward_tree_search(algorithm,
            args=(short_term_memory, historical_agent_observations, historical_agent_posterior_Q, a, A, y, B, b, t, t + horizon, 
            time_since_resource, t, chosen_action, 
            best_actions, weights, num_modalities, num_factors, num_states, num_resource_observations, G_prior, resource_constraints, memory_accessed, tree_search_call_count)
        )
        # G, best_actions = 0, [] 
        # if algorithm == "SI":
        #     G, short_term_memory, best_actions, memory_accessed = forward_tree_search_SI(
        #         short_term_memory, O, Q, a, A, y, B, b, t, t + horizon, 
        #         time_since_resource, t, chosen_action, 
        #         best_actions, weights, num_modalities, num_factors, num_states, num_resource_observations, G_prior, resource_constraints, memory_accessed
        #     )
        # elif algorithm == "SL":
        #     G, short_term_memory, best_actions, memory_accessed = forward_tree_search_SL(
        #         short_term_memory, O, Q, a, A, y, B, b, t, t + horizon, 
        #         time_since_resource, t, chosen_action, 
        #         best_actions, weights, num_modalities, num_factors, num_states, num_resource_observations, G_prior, resource_constraints, memory_accessed
        #     )
        
        
        # Save tree search depth for analytics
        search_depth += len(best_actions)
        
        # Add the latest posterior to the history
        historical_agent_posterior_Q[t] = copy.deepcopy(P)

        # Add the chosen action to the history
        historical_chosen_action.append(best_actions[0])
        
        # Prepare for the next iteration
        alive_status = is_alive(time_since_resource, resource_constraints)
        
        if not alive_status:
            print(f"At time {t} the agent is dead.", flush=True)
            print(f"The agent had: {resource_constraints['Food'] - time_since_resource['Food']} food, {resource_constraints['Water'] - time_since_resource['Water']} water, and {resource_constraints['Sleep'] - time_since_resource['Sleep']} sleep.", flush=True)
            print(f"The total tree search depth for this trial was {search_depth}.", flush=True)
            print(f"The agent accessed its memory {memory_accessed} times.", flush=True)
            print(f"The agent cleared its short-term memory {pe_memory_resets + hill_memory_resets} times.", flush=True)
            print(f"    State prediction error memory resets: {pe_memory_resets}.", flush=True)
            print(f"    Hill memory resets: {hill_memory_resets}.", flush=True)
            
        if visualise:
            # TODO: PyMDP has Gridworld environment that can be rendered.
            # Clear the axis, redraw and pause
            ax.clear()
            draw_gridworld(ax, current_pos, historical_true_states[t][1], needs,contextual_food_locations, contextual_water_locations, contextual_sleep_locations, hill_1)
            plt.pause(0.1)  # Adjust the time to be suitable for your loop speed
        
        t += 1
        
    trial_data["historical_predictive_observations_posteriors"] = historical_predictive_observations_posteriors
    trial_data["historical_agent_observations"] = historical_agent_observations
    trial_data["historical_agent_posterior_Q"] = historical_agent_posterior_Q
    trial_data["historical_true_states"] = historical_true_states
    trial_data["historical_chosen_action"] = historical_chosen_action
    trial_data["hill_memory_resets"] = hill_memory_resets
    trial_data["pe_memory_resets"] = pe_memory_resets
    trial_data["time_steps"] = t
    trial_data["search_depth"] = search_depth
    trial_data["memory_accessed"] = memory_accessed
    
    
    if visualise:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # This will now block until the window is closed.

    return a, trial_data, hill_memory_resets

def experiment(algorithm, seed, visualise=False):
    random.seed(seed)
    np.random.seed(seed)
    
    VISUALISE = visualise
    ax = None
    if VISUALISE:
        # Initial plot setup
        fig, ax = plt.subplots()
        plt.ion()  # Turn on interactive mode
        
    # Experimental/environmental hyperparams
    num_trials = 120
    num_states = 100
    num_factors = 2
    num_contextual_states = 4
    contextual_food_locations = [70,42,56,77]
    contextual_water_locations = [72,32,47,66]
    contextual_sleep_locations = [63,43,48,58] 
    hill_1 = 54 

    # Agentic hyperparams
    num_modalities = 3
    num_resource_observations = 4 # [none, food, water, sleep]
    num_context_observations = 5 # [summer, autumn, winter, spring, none]
    t_constraint = 100
    resource_constraints = {"Food":21,"Water":19,"Sleep":24}
    start_position = 50

    # Tree search hyperparams
    weights = {"Novelty":10, "Learning":40, "Epistemic":1, "Preference":10}
    G_prior = 0.02
        
    # Distributions
    A, a, B, b, D = initialise_distributions(
        num_states,
        contextual_food_locations,
        contextual_water_locations,
        contextual_sleep_locations,
        hill_1,
        start_position)

    # Experiments
    trials_data = []
    total_start_time = datetime.now()
    total_hill_resets = 0
    total_pe_resets = 0
    total_memory_accessed = 0
    total_search_depth = 0
    t_at25 = 0
    t_at50 = 0
    t_at75 = 0
    t_at100 = 0
    total_t = 0

    # Generate a timestamp in the format YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Configure logging to write to both the console and a log file
    log_file_path = f"{algorithm}_Seed_{seed}_{timestamp}.txt"
        
    log_format = '%(message)s'  # Only display the log message
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    for trial in range(num_trials):
        logging.info(f"\n{'-' * 40}\nTRIAL {trial+1} STARTED\n{'-' * 40}")
        trial_start_time = datetime.now()
        logging.info(f"Start Time: {trial_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if VISUALISE and ax is not None: ax.clear()
        a, trial_data, hill_memory_resets = agent_loop(
            algorithm,
            a, 
            resource_constraints, 
            A, 
            B, 
            b, 
            D,
            t_constraint, 
            contextual_food_locations, 
            contextual_sleep_locations, 
            contextual_water_locations, 
            num_modalities, 
            num_states,
            num_factors,             
            num_contextual_states,   
            num_resource_observations,  
            num_context_observations,   
            hill_1,             
            weights,          
            G_prior,
            start_position,     
            visualise = VISUALISE,     
            ax = ax
        )
        trials_data.append(trial_data)

        if trial_data["time_steps"] >= 24 and trial_data["time_steps"] < 49:
            t_at25 = t_at25 + 1
        elif trial_data["time_steps"] >= 49 and trial_data["time_steps"] < 74:
            t_at50 = t_at50 + 1
        elif trial_data["time_steps"] >= 74 and trial_data["time_steps"] < 99:
            t_at75 = t_at75 + 1
        elif trial_data["time_steps"] == 99:
            t_at100 = t_at100 + 1

        total_t += trial_data["time_steps"]-1
        total_hill_resets += trial_data["hill_memory_resets"]
        total_pe_resets += trial_data["pe_memory_resets"]
        total_search_depth += trial_data["search_depth"]
        total_memory_accessed += trial_data["memory_accessed"]
        
        # Calculate total runtime for the latest trial
        trial_end_time = datetime.now()    
        runtime = trial_end_time - trial_start_time
        minutes, seconds = divmod(runtime.seconds, 60)
        
        logging.info(f"TRIAL {trial+1} COMPLETE ✔")
        logging.info(f"End Time: {trial_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total runtime for this trial (minutes/seconds): {minutes:02d}:{seconds:02d}")
        logging.info(f"{'-' * 40}")
        logging.info(f"Total hill visits: {total_hill_resets}")
        logging.info(f"Total prediction errors: {total_pe_resets}")
        logging.info(f"Total search depth: {total_search_depth}")
        logging.info(f"Total times memory accessed: {total_memory_accessed}")
        logging.info(f"Total times 24 >= t < 49: {t_at25}")
        logging.info(f"Total times 49 >= t < 74: {t_at50}")
        logging.info(f"Total times 74 >= t < 99: {t_at75}")
        logging.info(f"Total times t == 99: {t_at100}")
        logging.info(f"Total time steps survived: {total_t}")
        # Calculate total run time so far
        runtime = trial_end_time - total_start_time
        hours, remainder = divmod(runtime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"Total runtime so far (hours/minutes/seconds): {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info(f"{'-' * 40}")

    # Calculate total run time for all trials
    total_end_time = datetime.now()
    runtime = total_end_time - total_start_time
    hours, remainder = divmod(runtime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"EXPERIMENT COMPLETE ✔.")
    logging.info(f"End Time: {total_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"TOTAL RUNTIME (hours/minutes/seconds): {hours:02d}:{minutes:02d}:{seconds:02d}")
    logging.info(f"AVERAGE RUNTIME PER TIME STEP: {round_half_up((runtime.total_seconds() / total_t), 3)} seconds")
    logging.info(f"Average hill visits per time step: {round_half_up(total_hill_resets / total_t,3)}")
    logging.info(f"Average prediction errors per time step: {round_half_up(total_pe_resets / total_t,3)}")
    logging.info(f"Average search depth per time step: {round_half_up(total_search_depth / total_t)}")
    logging.info(f"Average times memory accessed per time step: {round_half_up(total_memory_accessed / total_t)}")
    logging.info(f"{'-' * 40}")

# if __name__ == "__main__":
    