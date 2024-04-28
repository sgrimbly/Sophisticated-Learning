import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# TODO (St John): Fix potential threading and halting behaviour of rendering Matplotlib during code execution. #visualisation #bug

def index_to_position(index, size=10):
    return (index // size, index % size)

def get_contextual_position(context, locations):
    if 0 <= context < len(locations):
        return index_to_position(locations[context])
    return None

def update(frame):
    global agent_position, agent_needs

    ax.clear()
    # Assuming hill_position is either a global variable or part of the frame data
    draw_gridworld(ax, agent_position, context, agent_needs, contextual_food_locations, contextual_water_locations, contextual_sleep_locations, hill_position)

def draw_gridworld(ax, agent_index, context, needs, food_locations, water_locations, sleep_locations, hill_position):
    gridworld = np.zeros((10, 10))

    food_pos = get_contextual_position(context, food_locations)
    if food_pos:
        ax.text(*food_pos, 'F', ha='center', va='center', color='green')
    
    water_pos = get_contextual_position(context, water_locations)
    if water_pos:
        ax.text(*water_pos, 'W', ha='center', va='center', color='blue')

    sleep_pos = get_contextual_position(context, sleep_locations)
    if sleep_pos:
        ax.text(*sleep_pos, 'S', ha='center', va='center', color='red')

    agent_position_2d = index_to_position(agent_index)
    ax.text(*agent_position_2d, 'A', ha='center', va='center', color='black', backgroundcolor='white')

    # Hill position - directly converting the position index to coordinates
    hill_pos_2d = index_to_position(hill_position)
    ax.text(*hill_pos_2d, 'H', ha='center', va='center', color='brown')

    # Display agent's needs
    needs_text = f'Needs: Food {needs["Food"]}, Water {needs["Water"]}, Sleep {needs["Sleep"]}'
    ax.text(0.5, -1.5, needs_text, ha='center', va='center', color='black')

    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(axis=u'both', which=u'both',length=0)

    # Drawing the grid
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(axis=u'both', which=u'both',length=0)

# Rest of your code
