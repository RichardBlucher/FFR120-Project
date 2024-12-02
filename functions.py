import numpy as np
import scipy
from matplotlib import pyplot as plt
import math
from tkinter import *
import time


def sense(x, y, theta, SA, SO, trailmap, mapsize, foodmap, RA, bc_type):
    '''
    Function for agents to sense the environment and choose rotation.

    Input:
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    theta - np.array(N_particles), all particles direction of travel
    SA - float, Sensor angle from forward position
    SO - int,  Sensor offset distance
    trailmap - np.array(mapsize, mapsize), matrix with trail values
    mapsize - int, side length of environment box
    foodmap - np.array(mapsize, mapsize), matrix with food values
    RA - float,  Agent rotation angle

    Output:
    rotations - np.array(N_particles), rotations to be made
    '''
    N_particles = len(x) # Number of particles
    
    dthetas = [-RA, 0, RA] # To choose from (turn right, continue straight, turn left)
    rotations = np.zeros(N_particles) # To fill with all agents rotation

    # Find indexes to sense
    rx = np.round(x + SO * np.cos(theta-SA))
    ry = np.round(y + SO * np.sin(theta-SA))
    mx = np.round(x + SO * np.cos(theta))
    my = np.round(y + SO * np.sin(theta))
    lx = np.round(x + SO * np.cos(theta+SA))
    ly = np.round(y + SO * np.sin(theta+SA))

    rx, ry, _ = boundary_conditions(rx, ry, theta, mapsize, bc_type)
    mx, my, _ = boundary_conditions(mx, my, theta, mapsize, bc_type)
    lx, ly, _ = boundary_conditions(lx, ly, theta, mapsize, bc_type)
    

    for i in range(N_particles):
        
        
        inds_to_sense = [int(rx[i]), int(ry[i]), int(mx[i]), int(my[i]), int(lx[i]), int(ly[i])]
        rml = np.zeros(3)
        
        for ind in range(3):
            trail_val = 0
            if inds_to_sense[ind*2] >= mapsize or inds_to_sense[ind*2] < 0 or  inds_to_sense[ind*2+1] >= mapsize or inds_to_sense[ind*2+1] < 0: # If index to sense is out of range
                rml[ind] = 0 # set its value to 0
            else:
                trail_val = trailmap[inds_to_sense[ind*2], inds_to_sense[ind*2+1]] + foodmap[inds_to_sense[ind*2], inds_to_sense[ind*2+1]] # Sense trail+food
                
                rml[ind] = trail_val

        
        if rml[0] == rml[1] == rml[2]:
            rotations[i] = dthetas[1] # If all directions are same, continue forward
        else:
            rotations[i] = dthetas[np.argmax(rml)] # Go in direction of highest value

    return rotations

def deposit(x, y, trailmap, depT):
    '''
    Function for agents to deposit trail
    
    Input:
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    trailmap - np.array(mapsize, mapsize), matrix with trail values
    

    Output:
    trailmap - np.array(mapsize, mapsize), matrix with updated trail values
    '''
    for i in range(len(x)):
        trailmap[int(np.round(x[i])), int(np.round(y[i]))] = depT
    return trailmap

def diffuse(trailmap):
    '''
    Function to diffuse the trails

    Input: 
    trailmap - np.array(mapsize, mapsize), matrix with trail values
    #TODO: should also take SW (see J. Jones paper)
    #TODO: should also take diffK (see J. Jones paper)


    Output:
    trailmap - np.array(mapsize, mapsize), diffused trailmap
    '''
    return scipy.ndimage.uniform_filter(trailmap)

def decay(trailmap, decayT):
    '''
    Function to decay trails

    Input: 
    trailmap - np.array(mapsize, mapsize), matrix with trail values
    decayT - float, Trail-map chemoattractant diffusion decay factor

    Output:
    trailmap - np.array(mapsize, mapsize), decayed trailmap
    '''
    trailmap = trailmap*(1-decayT)

    return trailmap

def move(theta, x, y, SS):
    '''
    Function to move the particles
    Input:
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    theta - np.array(N_particles), all particles direction of travel
    SS - int, Step size (how far agent moves per step)
    
    Output:
    x - np.array(N_particles), all particles new x position
    ny - np.array(N_particles), all particles new y position
    '''
    x = x + SS * np.cos(theta)
    ny = y + SS * np.sin(theta)

    return x, ny

def boundary_conditions(x, y, theta, mapsize, bc_type):
    '''
    Applies boundary conditions

    Input:
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    theta - np.array(N_particles), all particles direction of travel
    mapsize - int, side length of environment box
    bc_type - string, indicating what type of boundary condition ('reflective' or 'periodic')

    Output:
    x - np.array(N_particles), all particles new x position
    y - np.array(N_particles), all particles new y position
    theta - np.array(N_particles), all particles new direction of travel
    '''
    N_part = len(x)
    vx = np.cos(theta)
    vy = np.sin(theta)
    if bc_type == 'reflective':
        for j in range(N_part):
            if x[j] < 0:
                x[j] = 0 + (0 - x[j])
                vx[j] = - vx[j]

            if x[j] > mapsize-1:
                x[j] = mapsize-1 - (x[j] - mapsize+1)
                vx[j] = - vx[j]

            if y[j] < 0:
                y[j] = 0 + (0 - y[j])
                vy[j] = - vy[j]
                
            if y[j] > mapsize-1:
                y[j] = mapsize-1 - (y[j] - mapsize+1)
                vy[j] = - vy[j]
        
        #nv = np.sqrt(vx ** 2 + vy ** 2)
        for i in range(N_part):
            theta[i] = math.atan2(vy[i], vx[i])

    elif bc_type == 'periodic':
        x = x % (mapsize-1)
        y = y % (mapsize-1)
    
    else:
        print(f'You\'ve typed {bc_type} as bc_type. Please choose reflective or periodic.')
    
    return x, y, theta

def animation_initialization(x, y, theta, mapsize):
    '''
    Function for initializing the agent animation.

    Inputs:
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    theta - np.array(N_particles), all particles direction of travel
    mapsize - int, side length of environment box

    Output:
    particles - list of tk objects or something?, holds all particles
    velocities - list of tk objects or something?, holds all particles' velocities
    canvas - i think this is the animation window
    tk - honestly i don't know but it is needed...
    '''
    N_part = len(x)
    
    window_size = 600

    rp = 0.5  # Plotting radius of a particle.
    vp = 1  # Length of the arrow indicating the velocity direction.
    line_width = 1  # Width of the arrow line.

    #N_skip = 1

    
    tk = Tk()
    tk.geometry(f'{window_size + 20}x{window_size + 20}')
    tk.configure(background='#000000')

    canvas = Canvas(tk, background='#ECECEC')  # Generate animation window 
    tk.attributes('-topmost', 0)
    canvas.place(x=10, y=10, height=window_size, width=window_size)

    particles = []
    for j in range(N_part):
        particles.append(
            canvas.create_oval(
                (x[j] - rp) / mapsize * window_size, 
                (y[j] - rp) / mapsize * window_size,
                (x[j] + rp) / mapsize * window_size, 
                (y[j] + rp) / mapsize * window_size,
                outline='#FF0000', 
                fill='#FF0000',
            )
        )

    velocities = []
    for j in range(N_part):
        velocities.append(
            canvas.create_line(
                x[j] / mapsize * window_size, 
                y[j] / mapsize * window_size,
                (x[j] + vp * np.cos(theta[j])) / mapsize * window_size, 
                (y[j] + vp * np.cos(theta[j])) / mapsize * window_size,
                width=line_width
            )
        )
    
    return particles, velocities, canvas, tk

def update_animation(step, N_skip, particles, velocities, canvas, tk, x, y, theta, mapsize):
    '''
    Function for updating the agent animation.

    Input:
    step - int, current step of the simulation
    N_skip - int, number of steps to skip before animating a new frame
    particles - list of tk objects or something?, holds all particles
    velocities - list of tk objects or something?, holds all particles' velocities
    canvas - i think this is the animation window
    tk - honestly i don't know but it is needed...
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    theta - np.array(N_particles), all particles direction of travel
    mapsize - int, side length of environment box

    Output:
    nothing, it just updates the animation window
    '''
    rp = 0.5  # Plotting radius of a particle.
    vp = 1  # Length of the arrow indicating the velocity direction.
    window_size = 600
    # Update animation frame.
    if step % N_skip == 0:        
        for j, particle in enumerate(particles):
            canvas.coords(
                particle,
                (x[j] - rp) / mapsize * window_size,
                (y[j] - rp) / mapsize * window_size ,
                (x[j] + rp) / mapsize * window_size ,
                (y[j] + rp) / mapsize * window_size ,
            )
                    
        for j, velocity in enumerate(velocities):
            canvas.coords(
                velocity,
                x[j] / mapsize * window_size,
                y[j] / mapsize * window_size,
                (x[j] + vp * np.cos(theta[j])) / mapsize * window_size,
                (y[j] + vp * np.sin(theta[j])) / mapsize * window_size,
            )
                    
        tk.title(f'Iteration {step}')
        tk.update_idletasks()
        tk.update()
        time.sleep(0)  # Increase to slow down the simulation.   

'''
TODO: Write function for plotting trailmaps at several times (like the figures in J. Jones)
TODO: Make the food work well
TODO: Optimizations
TODO: Other TODOs in the functions above
'''