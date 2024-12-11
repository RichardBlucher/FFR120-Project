import random

import numpy as np
import scipy
from matplotlib import pyplot as plt
import math
from tkinter import *
import time
import matplotlib.animation as animation
from PIL import Image, ImageTk
import cv2


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
    N_particles = len(x)  # Number of particles

    dthetas = [-RA, 0, RA]  # To choose from (turn right, continue straight, turn left)
    rotations = np.zeros(N_particles)  # To fill with all agents rotation

    # Find indexes to sense
    rx = np.round(x + SO * np.cos(theta - SA))
    ry = np.round(y + SO * np.sin(theta - SA))
    mx = np.round(x + SO * np.cos(theta))
    my = np.round(y + SO * np.sin(theta))
    lx = np.round(x + SO * np.cos(theta + SA))
    ly = np.round(y + SO * np.sin(theta + SA))

    rx, ry, _ = boundary_conditions(rx, ry, theta, mapsize, bc_type)
    mx, my, _ = boundary_conditions(mx, my, theta, mapsize, bc_type)
    lx, ly, _ = boundary_conditions(lx, ly, theta, mapsize, bc_type)

    for i in range(N_particles):

        inds_to_sense = [int(rx[i]), int(ry[i]), int(mx[i]), int(my[i]), int(lx[i]), int(ly[i])]
        rml = np.zeros(3)

        for ind in range(3):
            trail_val = 0
            if inds_to_sense[ind * 2] >= mapsize or inds_to_sense[ind * 2] < 0 or inds_to_sense[
                ind * 2 + 1] >= mapsize or inds_to_sense[ind * 2 + 1] < 0:  # If index to sense is out of range
                rml[ind] = 0  # set its value to 0
            else:
                trail_val = trailmap[inds_to_sense[ind * 2], inds_to_sense[ind * 2 + 1]] + foodmap[
                    inds_to_sense[ind * 2], inds_to_sense[ind * 2 + 1]]  # Sense trail+food

                rml[ind] = trail_val

        if rml[0] == rml[1] == rml[2]:
            rotations[i] = dthetas[1]  # If all directions are same, continue forward
        else:
            rotations[i] = dthetas[np.argmax(rml)]  # Go in direction of highest value

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
    trailmap = trailmap * (1 - decayT)

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
    N_part = len(x)
    nx = x + SS * np.cos(theta)
    ny = y + SS * np.sin(theta)
    same_pos_index = [0]
    tries = 0
    while len(same_pos_index) != 0 and tries < 3: # The "< 3" should not be needed here but sometimes it gets stuck with two particles in the same spot so i cheated a bit but will try to fix
        nxint = np.round(nx)
        nyint = np.round(ny)

        npos = np.vstack([nxint, nyint]).T
            
        dist = scipy.spatial.distance.cdist(npos, npos) + np.eye(N_part)
        same_pos_index = np.where(dist == 0)[0]

        nx[same_pos_index] = x[same_pos_index]
        ny[same_pos_index] = y[same_pos_index]
        theta[same_pos_index] = 2 * (np.random.rand(len(same_pos_index)) - 0.5) * np.pi
        tries += 1
    #print(same_pos_index) # Really useful print to check collisions





    return nx, ny, theta

def initialize_positions(mapsize, N_part, radius, position):
    '''
    Function for initializing the positions and orientations of the agents.

    Input:
    mapsize - int, side length of environment box
    N_part - int, number of agents.
    radius - radius of spawn circle
    position - list, center of circle

    Output:
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    theta - np.array(N_particles), all particles direction of travel

    '''

    x_c, y_c = position

    # Setting min radius for collisions
    min_radius = int(np.sqrt(N_part / np.pi) * 1.5)
    if radius < min_radius:
        print(f'Too small radius, setting radius to {min_radius}.')
        radius = min_radius

    # If circle out of bounds adjust position
    if x_c + radius > mapsize:
        x_c = mapsize - radius - 5
    if x_c - radius > mapsize:
        x_c = mapsize + radius + 5
    if y_c + radius > mapsize:
        y_c = mapsize - radius - 5
    if y_c - radius > mapsize:
        y_c = mapsize + radius + 5

    r = np.sqrt(np.random.rand(N_part)) * radius
    angle = np.random.rand(N_part) * 2 * np.pi

    x = x_c + r * np.cos(angle)
    y = y_c + r * np.sin(angle)

    same_pos_index = [0]
    while len(same_pos_index) != 0:
        xint = np.round(x)
        yint = np.round(y)
        pos = np.vstack([xint, yint]).T

        dist = scipy.spatial.distance.cdist(pos, pos) + np.eye(N_part)
        same_pos_index = np.where(dist == 0)[0]

        r = np.sqrt(np.random.rand(len(same_pos_index))) * radius
        angle = np.random.rand(len(same_pos_index)) * 2 * np.pi
        x[same_pos_index] = x_c + r * np.cos(angle)
        y[same_pos_index] = y_c + r * np.sin(angle)

    theta = 2 * (np.random.rand(N_part) - 0.5) * np.pi  # in [-pi, pi]

    return x, y, theta

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

            if x[j] > mapsize - 1:
                x[j] = mapsize - 1 - (x[j] - mapsize + 1)
                vx[j] = - vx[j]

            if y[j] < 0:
                y[j] = 0 + (0 - y[j])
                vy[j] = - vy[j]

            if y[j] > mapsize - 1:
                y[j] = mapsize - 1 - (y[j] - mapsize + 1)
                vy[j] = - vy[j]

        # nv = np.sqrt(vx ** 2 + vy ** 2)
        for i in range(N_part):
            theta[i] = math.atan2(vy[i], vx[i])

    elif bc_type == 'periodic':
        x = x % (mapsize - 1)
        y = y % (mapsize - 1)

    else:
        print(f'You\'ve typed {bc_type} as bc_type. Please choose reflective or periodic.')

    return x, y, theta

def place_food(food_str, std, mapsize, mode, mode_input):

    '''
    Places food based on inputs

    Input:
    food_str - int for value of food
    std - Standard deviation for gaussian of food
    mapsize - int, side length of environment box
    mode - string, how to place food - 'none', 'random', 'manual', 'square', 'triangle'
    mode_input - int or list depending on mode, 'random':int (number of random foods), 'manual':list(coordinates for food), 'square' or 'triangle':int(length of width)

    Output:
    foodmap - np.array of size (mapsize, mapsize), map for food with gaussian dist for each point
    '''


    foodmap = np.zeros([mapsize, mapsize])  # Empty foodmap

    if mode == 'none':
        pass
    if mode == 'random':
        try:
            food_positions = np.random.randint(10, mapsize-10, size=(2, mode_input))
            foodmap[food_positions[0, :], food_positions[1, :]] = food_str
        except:
            print('No food, enter a int value for number of random foods')
    if mode == 'manual':
        try:
            food_positions = np.array(mode_input)
            foodmap[food_positions[:, 0], food_positions[:, 1]] = food_str
        except:
            print('No food, enter a list of coordinates')
    if mode == 'square':
        try:
            width = mode_input
            p1 = (mapsize - width) // 2
            p2 = (mapsize - width) // 2 + width
            food_positions = np.array([[p1, p1], [p1 , p2], [p2, p1], [p2, p2]])
            foodmap[food_positions[:, 0], food_positions[:, 1]] = food_str
        except:
            print('No food, enter a int value for width')
    if mode == 'triangle':
        try:
            width = mode_input
            dc = width * np.sin(np.pi/3) // 2
            p1 = int(mapsize // 2 - dc)
            p2 = mapsize // 2
            p3 = int(mapsize // 2 + dc)
            food_positions = np.array([[p1, p2], [p3, p1], [p3, p3]])
            foodmap[food_positions[:, 0], food_positions[:, 1]] = food_str
        except:
            print('No food, enter a int value for width')
    if mode == 'dipper':
        # I know this is ugly, i just did it for fun
            
        star1 = np.round(np.array([0.077, 0.161])*mapsize*0.8 + 0.1*mapsize).astype(int)
        star2 = np.round(np.array([0.333, 0.221])*mapsize*0.8 + 0.1*mapsize).astype(int)
        star3 = np.round(np.array([0.440, 0.352])*mapsize*0.8 + 0.1*mapsize).astype(int)
        star4 = np.round(np.array([0.585, 0.509])*mapsize*0.8 + 0.1*mapsize).astype(int)
        star5 = np.round(np.array([0.553, 0.680])*mapsize*0.8 + 0.1*mapsize).astype(int)
        star6 = np.round(np.array([0.833, 0.809])*mapsize*0.8 + 0.1*mapsize).astype(int)
        star7 = np.round(np.array([0.956, 0.640])*mapsize*0.8 + 0.1*mapsize).astype(int)

        food_positions = np.flip(np.array([star1, star2, star3, star4, star5, star6, star7]))
        foodmap[food_positions[:, 0], food_positions[:, 1]] = food_str
        
    if mode not in ['none', 'random', 'manual', 'square', 'triangle']:
        print('No food, incorrect mode')

    foodmap = scipy.ndimage.gaussian_filter(foodmap, std)

    # Makes it so that value of food_str is the peak value, it's a bit wierd but it works (kinda)
    frac = (0.0003331331 + (1992766 - 0.0003331331) / (1 + (std / 0.008938102) ** 2.000044)) / 1000
    foodmap = foodmap / frac

    return foodmap

def trail_animation_initialization(trailmap, tk, colormap_name):
    """
    Initializes the tkinter Canvas and image for trailmap animation.

    Inputs:
    trailmap - trailmap
    tk - tk
    colormap_name - String for the colormap (viridis, plasma, magma etc)

    Outputs:
    canvas - Animation window
    canvas_img - Image placed on canvas
    palette - Color palette for the visualization

    """

    window_size = 600

    canvas = Canvas(tk)
    canvas.place(x=10, y=10, height=window_size, width=window_size)

    # Create image
    image = Image.fromarray(trailmap, mode='L')
    scaled_map = image.resize((window_size, window_size), Image.Resampling.NEAREST)
    tk_img = ImageTk.PhotoImage(scaled_map)

    # Add the image to the canvas
    canvas_img = canvas.create_image(0, 0, anchor='nw', image=tk_img)

    colormap = plt.cm.get_cmap(colormap_name)
    palette = []

    for i in range(256):
        rgba = colormap(i / 255.0)  # Get RGBA values from the colormap
        rgb = tuple((np.array(rgba[:3]) * 255).astype(int))  # Convert to RGB and scale to [0, 255]
        palette.append(rgb)


    return canvas, canvas_img, palette

def trail_update_animation(step, max_steps, combined_map, canvas, canvas_img, tk_trail, palette):
    """
    Updates the trailmap image on the trail canvas.

    Inputs:
    step - Fow showing iteration step in animation
    combined_map - trailmap or trailmap combined with food
    canvas - Animation window
    canvas_img - Image placed on canvas
    tk_trail - tk for the trailmap
    palette - Color palette for the visualization

    """

    window_size = 600

    # Normalize and convert trailmap to an image
    normalized_trailmap = ((combined_map - combined_map.min()) / (combined_map.max() - combined_map.min()) * 255).astype(np.uint8)
    image = Image.fromarray(normalized_trailmap, mode='P')
    image.putpalette([item for sublist in palette for item in sublist])
    scaled_map = image.resize((window_size, window_size), Image.Resampling.NEAREST)
    tk_img = ImageTk.PhotoImage(scaled_map)

    # Update the canvas image
    canvas.itemconfig(canvas_img, image=tk_img)
    canvas.tk_img = tk_img


    tk_trail.update()
    tk_trail.title(f'Iteration {step} of {max_steps}')

def store_trail_animation(step, combined_map, palette):
    """
    Returns the trailmap image for stroring.

    Inputs:
    step - Fow showing iteration step in animation
    combined_map - trailmap or trailmap combined with food
    palette - Color palette for the visualization

    Output:
    scaled_map - Image to be saved for making mp4

    """

    window_size = 600

    # Normalize and convert trailmap to an image
    normalized_trailmap = ((combined_map - combined_map.min()) / (combined_map.max() - combined_map.min()) * 255).astype(np.uint8)
    image = Image.fromarray(normalized_trailmap, mode='P')
    image.putpalette([item for sublist in palette for item in sublist])
    scaled_map = image.resize((window_size, window_size), Image.Resampling.NEAREST)

    scaled_map = scaled_map.convert("RGB")

    return scaled_map

def save_trail_animation(trail_imgs, filename):
    """
    Makes mp4 from trail images

    Inputs:
    trail_imgs - list of trail images
    filename - name of file

    """
    fps = 30
    frame_size = np.array(trail_imgs[0]).shape[:2][::-1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    for trail_img in trail_imgs:
        rgb_frame = cv2.cvtColor(np.array(trail_img), cv2.COLOR_RGB2BGR)
        out.write(rgb_frame)

    out.release()

def trail_plot_initialization(steps_to_plot):
    '''
    Function for initializing the trail plots.

    Input:
    steps_to_plot - list with ints, the desiered times to plot

    Output:
    fig - Figure object, for showing the trail plots
    axs - list of Axes objects, for making the trail plots
    '''
    nr_of_plots = len(steps_to_plot)
    fig, axs = plt.subplots(nrows=1, ncols=nr_of_plots, figsize=(nr_of_plots*3, 3), layout='constrained')
    #fig.suptitle('Configuration of the system at different t')
    i_fig = 0
    for step in steps_to_plot:
        axs[i_fig].set_title(f't = {step}')
        axs[i_fig].set_xlabel('x')
        axs[i_fig].set_ylabel('y')
        i_fig += 1
    
    return fig, axs

def plot_trailmap(steps_to_plot, trailmap, step, axs):
    '''
    Function to plot trailmaps at different times.

    Inputs:
    steps_to_plot - list with ints, the desiered times to plot
    trailmap - np.array(mapsize, mapsize), matrix with trail values
    step - int, current step (iteration) of the simulation
    axs - list of Axes objects, for making the trail plots

    Output:
    none
    '''
    if step in steps_to_plot:
        print(step)
        axs[steps_to_plot.index(step)].imshow(trailmap)
    if step == steps_to_plot[-1]:
        plt.show()


def plot_total_length(max_steps, trailmap, step, lengths):
    '''
    Idea for something similar to the total lenght (or cost in Tokyo paper)

    Inputs:
    max_steps - int, nr of steps the simulation runs for
    trailmap - np.array(mapsize, mapsize), matrix with trail values
    step - int, current step (iteration) of the simulation
    axs - list of Axes objects, for making the trail plots

    Output:
    none
    '''

    
    


    
    high_trail = trailmap > 2
    lengths.append(np.sum(high_trail)*100/(trailmap.shape[0]**2))
    
    if step == max_steps-1:
        plt.plot(lengths)
        plt.xlabel('Step')
        plt.ylabel('% of map containing trail')
        plt.show()

    return lengths


'''
TODO: Make the plot and animation functions better. Ex. could probably combine the initialization and actual plot functions to one, give more options (*kwargs?), better possibilities for saving, etc.
TODO: Make the food work well
TODO: Optimizations (find what is slow and make it faster)
TODO: Other TODOs in the functions above
TODO: Make it so that the function run_simulation in main.py doesn't need as many arguments
'''