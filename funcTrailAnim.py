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
    position - list, center of circle, enter 'c' for center of map

    Output:
    x - np.array(N_particles), all particles x position
    y - np.array(N_particles), all particles y position
    theta - np.array(N_particles), all particles direction of travel

    '''


    if position == "c":
        x_c, y_c = mapsize//2, mapsize//2
    else:
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
    
    if mode == 'gothenburg':

        g1=np.round(np.array([0.827272727, 0.054545455])*mapsize).astype(int)
        g2=np.round(np.array([0.554545455, 0.145454545])*mapsize).astype(int)
        g3=np.round(np.array([0.363636364, 0.163636364])*mapsize).astype(int)
        g4=np.round(np.array([0.472727273, 0.272727273])*mapsize).astype(int)
        g5=np.round(np.array([0.336363636, 0.327272727])*mapsize).astype(int)
        g6=np.round(np.array([0.058181818, 0.336363636])*mapsize).astype(int)
        g7=np.round(np.array([0.750909091, 0.218181818])*mapsize).astype(int)
        g8=np.round(np.array([0.654545455, 0.265454545])*mapsize).astype(int)
        g9=np.round(np.array([0.578181818, 0.290909091])*mapsize).astype(int)
        g10=np.round(np.array([0.654545455, 0.318181818])*mapsize).astype(int)
        g11=np.round(np.array([0.527272727, 0.349090909])*mapsize).astype(int)
        g12=np.round(np.array([0.454545455, 0.361818182])*mapsize).astype(int)
        g13=np.round(np.array([0.036363636, 0.645454545])*mapsize).astype(int)
        g14=np.round(np.array([0.172727273, 0.605454545])*mapsize).astype(int)
        g15=np.round(np.array([0.274545455, 0.545454545])*mapsize).astype(int)
        g16=np.round(np.array([0.318181818, 0.823636])*mapsize).astype(int)
        g17=np.round(np.array([0.452727273, 0.603636364])*mapsize).astype(int)
        g18=np.round(np.array([0.527272727, 0.461818182])*mapsize).astype(int)
        g19=np.round(np.array([0.589090909, 0.536363636])*mapsize).astype(int)
        g20=np.round(np.array([0.669090909, 0.463636364])*mapsize).astype(int)
        g21=np.round(np.array([0.76,	    0.387272727])*mapsize).astype(int)
        g22=np.round(np.array([0.829090909, 0.581818182])*mapsize).astype(int)
        g23=np.round(np.array([0.890909091, 0.792727273])*mapsize).astype(int)
        g24=np.round(np.array([0.878181818, 0.930909091])*mapsize).astype(int)
        g25=np.round(np.array([0.54,	    0.923636364])*mapsize).astype(int)

        food_positions = np.flip(np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21, g22, g23, g24, g25]))
        food_positions = np.round(food_positions*0.8+ 0.1*mapsize).astype(int)
        foodmap[food_positions[:, 0], food_positions[:, 1]] = food_str

    if mode == 'tokyo':

        t1=np.round(np.array([0.8125, 0.025])*mapsize).astype(int)
        t2=np.round(np.array([0.5, 0.1175])*mapsize).astype(int)
        t3=np.round(np.array([0.125, 0.15])*mapsize).astype(int)
        t4=np.round(np.array([0.1625, 0.115])*mapsize).astype(int)
        t5=np.round(np.array([0.2625, 0.125])*mapsize).astype(int)
        t6=np.round(np.array([0.31875, 0.15375])*mapsize).astype(int)
        t7=np.round(np.array([0.45625, 0.16875])*mapsize).astype(int)
        t8=np.round(np.array([0.74375, 0.15])*mapsize).astype(int)
        t9=np.round(np.array([0.275, 0.25])*mapsize).astype(int)
        t10=np.round(np.array([0.63125, 0.2875])*mapsize).astype(int)
        t11=np.round(np.array([0.3625, 0.35])*mapsize).astype(int)
        t12=np.round(np.array([0.385, 0.4])*mapsize).astype(int)
        t13=np.round(np.array([0.5325, 0.40625])*mapsize).astype(int)
        t14=np.round(np.array([0.76875,	0.385])*mapsize).astype(int)
        t15=np.round(np.array([0.9125, 0.5])*mapsize).astype(int)
        t16=np.round(np.array([0.95, 0.4875])*mapsize).astype(int)
        t17=np.round(np.array([0.6875, 0.45])*mapsize).astype(int)
        t18=np.round(np.array([0.65, 0.4875])*mapsize).astype(int)
        t19=np.round(np.array([0.525, 0.5])*mapsize).astype(int)
        t20=np.round(np.array([0.2375, 0.5])*mapsize).astype(int)
        t21=np.round(np.array([0.25625,	0.55625])*mapsize).astype(int)
        t22=np.round(np.array([0.5875, 0.55625])*mapsize).astype(int)
        t23=np.round(np.array([0.73125,	0.55625])*mapsize).astype(int)
        t24=np.round(np.array([0.575, 0.6125])*mapsize).astype(int)
        t25=np.round(np.array([0.3875, 0.6375])*mapsize).astype(int)
        t26=np.round(np.array([0.50625,	0.675])*mapsize).astype(int)
        t27=np.round(np.array([0.70625,	0.675])*mapsize).astype(int)
        t28=np.round(np.array([0.6875, 0.8])*mapsize).astype(int)
        t29=np.round(np.array([0.49, 0.90625])*mapsize).astype(int)
        t30=np.round(np.array([0.385, 0.75625])*mapsize).astype(int)
        t31=np.round(np.array([0.34375,	0.71875])*mapsize).astype(int)
        t32=np.round(np.array([0.28125,	0.69125])*mapsize).astype(int)
        t33=np.round(np.array([0.23125,	0.69375])*mapsize).astype(int)
        t34=np.round(np.array([0.1625, 0.725])*mapsize).astype(int)
        t35=np.round(np.array([0.11875,	0.925])*mapsize).astype(int)

    

        food_positions = np.flip(np.array([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35]))
        food_positions = np.round(food_positions*0.8+ 0.1*mapsize).astype(int)
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
    fig, axs = plt.subplots(nrows=1, ncols=nr_of_plots, figsize=(nr_of_plots*3, 3), layout='constrained', facecolor='#E7E9E7')
    #fig.suptitle('Configuration of the system at different t')
    i_fig = 0
    for step in steps_to_plot:
        axs[i_fig].set_title(f't = {step}')
        axs[i_fig].set_xticks([])
        axs[i_fig].set_yticks([])
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
        plt.figure(facecolor='#E7E9E7')
        plt.plot(lengths)
        plt.xlabel('Step')
        plt.ylabel('Area of map covered in trails [%]')
        plt.show()

    return lengths


'''
TODO: Make the plot and animation functions better. Ex. could probably combine the initialization and actual plot functions to one, give more options (*kwargs?), better possibilities for saving, etc.
TODO: Make the food work well
TODO: Optimizations (find what is slow and make it faster)
TODO: Other TODOs in the functions above
TODO: Make it so that the function run_simulation in main.py doesn't need as many arguments
'''
