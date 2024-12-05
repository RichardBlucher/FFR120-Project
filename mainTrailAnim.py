from funcTrailAnim import *
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import matplotlib.animation as animation

'''
TODO list in functions.py (list is not in any way ordered by priority)
'''


def run_simulation(max_steps, x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, show_animation_on, save_animation, show_food,
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
        if steps_to_plot != []:
            p_fig, axs = trail_plot_initialization(steps_to_plot)
            max_steps = steps_to_plot[-1] + 1

        canvas_trail, canvas_img, palette = trail_animation_initialization(trailmap, tk, "viridis")

    for step in range(max_steps):

        theta += sense(x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, bc_type)
        x, y = move(theta, x, y, SS)
        x, y, theta = boundary_conditions(x, y, theta, mapsize, bc_type)
        trailmap = deposit(x, y, trailmap, depT)
        trailmap = diffuse(trailmap)
        trailmap = decay(trailmap, decayT)
        if show_animation_on:
            if show_food:
                combined_map = np.maximum(trailmap, foodmap)
            else:
                combined_map = trailmap
            trail_update_animation(step, combined_map, canvas_trail, canvas_img, tk, palette)
        if save_animation:
            if show_food:
                combined_map = np.maximum(trailmap, foodmap)
            else:
                combined_map = trailmap
            trail_img = store_trail_animation(step, combined_map, canvas_trail, canvas_img, palette)
            trail_imgs.append(trail_img)
        if steps_to_plot != []:
            plot_trailmap(steps_to_plot, trailmap, step, axs)

    if save_animation:
        save_trail_animation(trail_imgs, "test_anim.mp4")

    print(f'Time: {(timer() - start):.4}')  # Prints the time the run took



def main():
    # Initialization
    max_steps = 300  # Number of steps to run simulation
    mapsize = 200  # Dimension of the squared arena.
    percent_p = 5  # Population as percentage of image area
    N_part = int(mapsize * mapsize * percent_p / 100)  # Number of particles.

    SS = 1  # Step size (how far agent moves per step)

    RA = np.pi / 4  # Agent rotation angle
    SA = np.pi / 4  # Sensor angle from forward position
    SO = 9  # Sensor offset distance

    decayT = 0.1  # Trail-map chemoattractant diffusion decay factor

    depT = 5  # Chemoattractant deposition per step

    bc_type = 'periodic'  # What type of boundary conditions to use (reflective or periodic)

    # np.random.seed(2) # Seed for random starting position

    # Random position.
    x = (np.random.rand(N_part)) * mapsize
    y = (np.random.rand(N_part)) * mapsize

    # Random orientation.
    theta = 2 * (np.random.rand(N_part) - 0.5) * np.pi  # in [-pi, pi]

    # Maps
    trailmap = np.zeros([mapsize, mapsize])  # Empty trailmap
    foodmap = np.zeros([mapsize, mapsize])  # Empty foodmap

    food_val = 10
    # foodmap[30:40, 30:40] = food_val
    # foodmap[180:190, 20:30] = food_val
    foodmap[80:90, 70:80] = food_val
    foodmap[80:90, 150:160] = food_val
    foodmap[40:50, 140:150] = food_val
    foodmap[140:150, 180:190] = food_val
    # foodmap[70:80, 120:130] = food_val
    # foodmap[70:80, 70:80] = food_val
    # foodmap[120:130, 70:80] = food_val
    # foodmap[120:130, 120:130] = food_val


    show_animation_on = True
    save_animation = True
    # For trail animation (way faster than agent animation). Also possible to save
    show_food = True # Shows food on trailmap but makes it harder to see trails
    save_trail_animation = False  # True will save the animation at end of run
    trail_animation_name = 'FFR120-Project/animations/trails_test.gif'  # name and path of saved animation (has to be .gif). Make sure to use a path that works for you.

    # For trail plots
    # steps_to_plot = [50, 100, 150, 200]  # list with times to plot. If you don't want to plot, make steps_to_plot = []
    # steps_to_plot = [100, 200, 300, 400, 500]
    # steps_to_plot = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 15000]
    steps_to_plot = []

    run_simulation(max_steps, x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, show_animation_on, save_animation, show_food,
                   bc_type, depT, steps_to_plot)

if __name__ == "__main__":
    main()