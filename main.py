from functions import *
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import matplotlib.animation as animation

def run_simulation(max_steps, x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, animate_agents, N_skip, bc_type, depT, 
                   animate_trails, show_trail_animation, save_trail_animation, trail_animation_name):
    '''
    Function that runs the simulation

    TODO: Write inputs and stuff here.
    '''
    if animate_agents:
        particles, velocities, canvas, tk = agent_animation_initialization(x, y, theta, mapsize)
    
    if animate_trails:
        fig, ax, ims = trail_animation_initialization()

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
        
    if animate_trails:
        trail_animation_show_save(fig, ims, show_trail_animation, save_trail_animation, trail_animation_name)
    




def main():
    start = timer()  # Timer starts
    # Initialization
    max_steps = 100 # Number of steps to run simulation
    mapsize = 200  # Dimension of the squared arena.
    percent_p = 10 # Population as percentage of image area
    N_part = int(mapsize*mapsize * percent_p/100)   # Number of particles.

    SS = 1  # Step size (how far agent moves per step)
    
    
    
    RA = np.pi/4 # Agent rotation angle
    SA = np.pi/4 # Sensor angle from forward position
    SO = 9 # Sensor offset distance

    decayT = 0.1 # Trail-map chemoattractant diffusion decay factor

    depT = 5 # Chemoattractant deposition per step

    bc_type = 'periodic' # What type of boundary conditions to use (reflective or periodic)

    np.random.seed(2) # Seed for random starting position

    # Random position.
    x = (np.random.rand(N_part)) * mapsize  
    y = (np.random.rand(N_part)) * mapsize  

    # Random orientation.
    theta = 2 * (np.random.rand(N_part) - 0.5) * np.pi  # in [-pi, pi]

    # Maps
    trailmap = np.zeros([mapsize,mapsize]) # Empty trailmap
    foodmap = np.zeros([mapsize,mapsize]) # Empty foodmap
    ''' This was just a test for the food but it doesent relly work well
    foodmap[100, 70] = 200
    foodmap[100, 130] = 200
    for i in range(50):
        foodmap = diffuse(foodmap)'''

    # For agent animation (really slow for many agents)
    animate_agents = False # True to animate agents, False to not
    N_skip = 1 # Number of steps to skip before animating a new frame

    # For trail animation (way faster than agent animation). Also possible to save
    animate_trails = True # # True to animate trails, False to not
    show_trail_animation = False # True will show the animation at end of run
    save_trail_animation = True # True will save the animation at end of run
    trail_animation_name = 'FFR120-Project/animations/trails_test.gif' # name and path of saved animation (has to be .gif). Make sure to use a path that works for you.

    
    



    run_simulation(max_steps, x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, SS, decayT, animate_agents, N_skip, bc_type, depT, 
                   animate_trails, show_trail_animation, save_trail_animation, trail_animation_name)
    print(f'Time: {(timer() - start):.4}')  # Prints the time the run took



if __name__ == "__main__":
    main()