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

def trail_update_animation(step, combined_map, canvas, canvas_img, tk_trail, palette):
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
    tk_trail.title(f'Iteration {step}')

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
        axs[steps_to_plot.index(step)].imshow(trailmap)
    if step == steps_to_plot[-1]:
        plt.show()


'''
TODO: Make the plot and animation functions better. Ex. could probably combine the initialization and actual plot functions to one, give more options (*kwargs?), better possibilities for saving, etc.
TODO: Make the food work well
TODO: Optimizations (find what is slow and make it faster)
TODO: Other TODOs in the functions above
TODO: Make it so that the function run_simulation in main.py doesn't need as many arguments
'''