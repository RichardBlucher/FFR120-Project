from functions import *
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import matplotlib.animation as animation

'''
TODO list in functions.py (list is not in any way ordered by priority)
'''

def run_simulation(max_steps, x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, animate_agents, N_skip, bc_type, depT, 
                   animate_trails, show_trail_animation, save_trail_animation, trail_animation_name, steps_to_plot):
    '''
    Function that runs the simulation

    TODO: Write inputs and stuff here.
    '''
    print('Simulation running')
    start = timer()  # Timer starts
    if steps_to_plot != []:
        p_fig, axs = trail_plot_initialization(steps_to_plot)
        max_steps = steps_to_plot[-1] + 1
    if animate_agents:
        particles, velocities, canvas, tk = agent_animation_initialization(x, y, theta, mapsize)
    
    if animate_trails:
        a_fig, ax, ims = trail_animation_initialization()

    for step in range(max_steps):
        
        theta += sense(x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, bc_type)
        x, y = move(theta, x, y, SS)
        x, y, theta = boundary_conditions(x, y, theta, mapsize, bc_type)
        trailmap = deposit(x, y, trailmap, depT)
        trailmap = diffuse(trailmap)
        trailmap = decay(trailmap, decayT)
        if animate_agents:
            agent_update_animation(step, N_skip, particles, velocities, canvas, tk, x, y, theta, mapsize)
        if animate_trails:
            trail_update_animation(ax, trailmap, step, ims)
        if steps_to_plot != []:
            plot_trailmap(steps_to_plot, trailmap, step, axs)
    
    print(f'Time: {(timer() - start):.4}')  # Prints the time the run took
    #print(f'max value of trailmap: {np.max(trailmap)})
    if animate_trails:
        trail_animation_show_save(a_fig, ims, show_trail_animation, save_trail_animation, trail_animation_name)
    




def main():
    
    # Initialization
    max_steps = 4151 # Number of steps to run simulation
    mapsize = 200  # Dimension of the squared arena.
    percent_p = 15 # Population as percentage of image area
    N_part = int(mapsize*mapsize * percent_p/100)   # Number of particles.

    SS = 1  # Step size (how far agent moves per step)
    
    
    
    RA = np.pi/4 # Agent rotation angle
    SA = np.pi/8 # Sensor angle from forward position
    SO = 9 # Sensor offset distance

    decayT = 0.1 # Trail-map chemoattractant diffusion decay factor

    depT = 5 # Chemoattractant deposition per step

    bc_type = 'periodic' # What type of boundary conditions to use (reflective or periodic)

    np.random.seed(5) # Seed for random starting position

    # Random position.
    x = (np.random.rand(N_part)) * mapsize
    y = (np.random.rand(N_part)) * mapsize

    # Random orientation.
    theta = 2 * (np.random.rand(N_part) - 0.5) * np.pi  # in [-pi, pi]

    # Maps
    trailmap = np.zeros([mapsize,mapsize]) # Empty trailmap
    foodmap = np.zeros([mapsize,mapsize]) # Empty foodmap
    '''#This was just a test for the food but it doesent really work well
    foodmap[70, 70] = 100
    foodmap[130, 130] = 100
    foodmap[70, 130] = 100
    foodmap[130, 70] = 100
    for i in range(5):
        foodmap = diffuse(foodmap)
    trailmap = deposit(x,y,trailmap,depT)
    trailmap = diffuse(trailmap)
    trailmap = decay(trailmap, decayT)

    #print(np.max(foodmap))
    #plt.imshow(foodmap)
    foodmap = foodmap*5
    print(np.max(foodmap))
    #plt.plot(foodmap[100,:]+trailmap[100,:])
    #plt.show()'''

    # For agent animation (really slow for many agents)
    animate_agents = False # True to animate agents, False to not
    N_skip = 1 # Number of steps to skip before animating a new frame

    # For trail animation (way faster than agent animation). Also possible to save
    animate_trails = True # True to animate trails, False to not
    show_trail_animation = False # True will show the animation at end of run
    save_trail_animation = True # True will save the animation at end of run
    trail_animation_name = 'FFR120-Project/animations/fig4.gif' # name and path of saved animation (has to be .gif). Make sure to use a path that works for you.

    # For trail plots
    steps_to_plot = [] #[2, 22, 99, 175, 367, 512, 1740, 4151] # list with times to plot. If you don't want to plot, make steps_to_plot = [] 

    
    



    run_simulation(max_steps, x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, animate_agents, N_skip, bc_type, depT, 
                   animate_trails, show_trail_animation, save_trail_animation, trail_animation_name, steps_to_plot)
    



if __name__ == "__main__":
    main()