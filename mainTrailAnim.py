from funcTrailAnim import *
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import matplotlib.animation as animation

'''
TODO list in functions.py (list is not in any way ordered by priority)
'''


def run_simulation(max_steps, N_part, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, show_animation_on, save_animation, show_food,
                   bc_type, depT, steps_to_plot):
    '''
    Function that runs the simulation

    TODO: Write inputs and stuff here.
    '''
    start = timer()  # Timer starts

    if show_animation_on or save_animation:
        window_size = 600
        trail_imgs = []
        tk = Tk()
        tk.geometry(f'{window_size + 20}x{window_size + 20}')
        tk.configure(background='#000000')
        

        canvas_trail, canvas_img, palette = trail_animation_initialization(trailmap, tk, "magma")
    if steps_to_plot != []:
            p_fig, axs = trail_plot_initialization(steps_to_plot)
            max_steps = steps_to_plot[-1] + 1
            lengths = []
            

    x, y, theta = initialize_positions(mapsize, N_part, 80, 'c') # Position either list of coordinates or 'c' for center
    for step in range(max_steps):
        # print(step)
        theta += sense(x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, bc_type)
        x, y, theta = move(theta, x, y, SS)
        x, y, theta = boundary_conditions(x, y, theta, mapsize, bc_type)
        trailmap = deposit(x, y, trailmap, depT)
        trailmap = diffuse(trailmap)
        trailmap = decay(trailmap, decayT)
        if show_animation_on:
            if show_food:
                combined_map = np.maximum(trailmap, foodmap)
            else:
                combined_map = trailmap
            trail_update_animation(step, max_steps, combined_map, canvas_trail, canvas_img, tk, palette)
        if save_animation:
            if show_food:
                combined_map = np.maximum(trailmap, foodmap)
            else:
                combined_map = trailmap
            trail_img = store_trail_animation(step, combined_map, palette)
            trail_imgs.append(trail_img)
        if steps_to_plot != []:
            plot_trailmap(steps_to_plot, trailmap, step, axs)
            lengths = plot_total_length(max_steps,trailmap,step,lengths)

    if save_animation:
        save_trail_animation(trail_imgs, "test_anim.mp4")

    print(f'Time: {(timer() - start):.4}')  # Prints the time the run took



def main():
    # Initialization
    max_steps = 1000  # Number of steps to run simulation
    mapsize = 200  # Dimension of the squared arena.
    percent_p = 2  # Population as percentage of image area
    N_part = int(mapsize * mapsize * percent_p / 100)  # Number of particles.

    SS = 1  # Step size (how far agent moves per step)

    RA = np.pi / 4  # Agent rotation angle
    SA = np.pi / 4  # Sensor angle from forward position
    SO = 9  # Sensor offset distance

    decayT = 0.1  # Trail-map chemoattractant diffusion decay factor

    depT = 5  # Chemoattractant deposition per step

    bc_type = 'reflective'  # What type of boundary conditions to use (reflective or periodic)

    np.random.seed(1337) # Seed for random starting position

    

    

    # Maps
    trailmap = np.zeros([mapsize, mapsize])  # Empty trailmap

    food_str = 10
    std = 4
    mode = 'square' # Options on how to place food: 'none', 'random', 'manual', 'square', 'triangle'. Different modes require different inputs, see function description for more info
    coords = [[150, 150], [140, 30], [80, 160], [120, 90]] # Example of input to 'manual'
    width = 60 # Example of input to 'square' or 'triangle'
    mode_input = 5
    # mode_input = coords
    mode_input = width

    foodmap = place_food(food_str, std, mapsize, mode, mode_input)

    show_animation_on = True
    save_animation = False
    # For trail animation (way faster than agent animation). Also possible to save
    show_food = True # Shows food on trailmap but makes it harder to see trails
    save_trail_animation = False  # True will save the animation at end of run
    trail_animation_name = 'FFR120-Project/animations/trails_test.gif'  # name and path of saved animation (has to be .gif). Make sure to use a path that works for you.

    # For trail plots
    # steps_to_plot = [50, 100, 150, 200]  # list with times to plot. If you don't want to plot, make steps_to_plot = []
    # steps_to_plot = [2, 22, 99, 175, 367, 512, 1740, 4151]
    # steps_to_plot = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 15000]
    steps_to_plot = []

    run_simulation(max_steps, N_part, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, show_animation_on, save_animation, show_food,
                   bc_type, depT, steps_to_plot)

if __name__ == "__main__":
    main()