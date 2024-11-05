""" Utilities for generating synthetic 1-photon calcium imaging data.

This module contains functions for generating synthetic 1-photon fluorescence imaging data, including ground truth temporal components, spatial components, and noisy videos.
Author: Ahmad Abdal Qader
Date: 2024-06-19
"""

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
# from scipy.integrate import solve_ivp
from scipy.spatial import distance
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.cm as cm
from tqdm.auto import tqdm
import numpy as np
import logging
import cv2
import yaml



def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def isvalid(point, frame_size, marginx, marginy):
    validx = range(marginx, int(frame_size[1]-marginx))
    validy = range(marginy, int(frame_size[0]-marginy))
    if np.logical_and(point[0] in validx, point[1] in validy):
        return True
    else:
        return False


def gauss(amplitude, center, sigma, X, Y, ):
    x0 = center[0]
    y0 = center[0]
    G = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return G

def create_background():
    
    pass


def exponential_decay_kernel(kernel_size, tau_rise_ms, tau_decay_ms, sampling_rate, amplitude=1):
    """
    Create an exponential decay kernel.

    ARGS:
        timesteps: Number of timesteps to calculate the kernel for.
        tau_rise_ms: Rise time constant in milliseconds.
        tau_decay_ms: Decay time constant in milliseconds.
        sampling_rate: Sampling rate in Hz (samples per second).
        amplitude: Amplitude of the kernel.

    Returns:
        A numpy array representing the kernel.
    """
    # Convert tau values from milliseconds to time bins
    tau_rise = tau_rise_ms / 1000 * sampling_rate
    if tau_rise < 1:
        print("rise time is smaller than possible resolution, effectively instantaneous")
    tau_decay = tau_decay_ms / 1000 * sampling_rate

    # Create the kernel
    t = np.arange(kernel_size)
    kernel = amplitude * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))
    return kernel


def create_spiking_component(duration=20, rate=0.1, sampling_rate=30):
    ''' 
    Creates a spiking component for a neuron.

    ARGS:
        duration (int): The duration of the simulation in seconds.
        rate (float): The firing rate for the neuron.
        sampling_rate (int): The sampling rate in Hz.

    Returns:
        np.array: An array representing the spike train for the neuron.
    '''

    num_samples = int(duration * sampling_rate)
    isi = np.random.exponential(1 / rate, size=num_samples)
    spike_times = np.cumsum(isi)

    # Create spike train
    spike_train = np.zeros(num_samples)
    for t in spike_times:
        if t < duration:
            index = int(t * sampling_rate)
            spike_train[index] = 1

    return spike_train


def initialize_neurons(number_nrn=50,
                      duration=20,
                      sampling_rate=30,
                      rate=0.2, 
                      perturb = False,
                      perturb_range = [0.04, 0.04]
                      ):
    
    '''
    Initializes neurons spiking trains for the simulation.

    ARGS:
        number_nrn (int): The number of neurons to simulate.
        duration (int): The duration of the simulation in seconds.
        sampling_rate (int): The sampling rate in Hz.
        rate (float): The firing rate for the neurons.

    Returns:
        np.array: An array of spiking trains neurons.
    '''

    if isinstance(rate, (int, float)):
        rate = [rate] * number_nrn
    if isinstance(rate, (list, np.ndarray)):
        assert len(rate) == int(number_nrn), "rates length must match number of neurons, or be a single value"
    if perturb:
        rate_noise = np.random.uniform(*perturb_range, number_nrn)
        rate += rate_noise
    
    out = np.zeros([number_nrn, (duration*sampling_rate)])
    for i in range(number_nrn):
        out[i] = create_spiking_component(duration=duration, rate = rate[i], sampling_rate=sampling_rate)
    return out


def get_temporal_components_gt(
                                duration=20,
                                number_nrns=10,
                                sampling_rate=30,
                                kernel_size=50,
                                tau_rise=50,
                                tau_decay=250,
                                rate=0.08,
                                perturb=True,
                                perturb_range=[0.04, 0.04],
                                ):
    ''' 
    Generates ground truth temporal components for a given number of neurons.

    ARGS:
        duration (int): in seconds, the duration of the simulation in seconds.
        number_nrn (int): The number of neurons to simulate.
        sampling_rate (int): The sampling rate in Hz.
        kernel_size (int): The size of the kernel to use for the convolution.
        tau_rise (int): The rise time constant for the exponential decay kernel.
        tau_decay (int): The decay time constant for the exponential decay kernel.
        rate (float): The firing rate for the neurons.
        perturb (bool): Whether to perturb the firing rate.
        perturb_range (list): The range of perturbation for the firing rate, uniform dist.

    Returns:
        np.array: An array of ground truth temporal components for each neuron.
    '''
    
    spikes = initialize_neurons(number_nrn = number_nrns,
                                duration=duration, 
                                sampling_rate=sampling_rate,
                                rate=rate,
                                perturb=perturb,
                                perturb_range=perturb_range
                               )

    kernel = exponential_decay_kernel(kernel_size, tau_rise, tau_decay, sampling_rate=sampling_rate, amplitude=1)
    fluo_gt = np.array([np.convolve(spk, kernel) for spk in spikes])
    
    return fluo_gt


def generate_neuron_shapes(number_nrns, x_shape_space = [2,3], y_shape_space = [2,3,4]):

    ''' 
    Generates the shapes for a given number of neurons.
    ARGS:
        number_nrns (int): The number of neurons to generate.
        x_shape_space (list): The range of standard deviations for the x-axis that can be selected
        y_shape_space (list): The range of standard deviations for the y-axis that can be selected
    Returns:
        list: A list of tuples representing the shapes of the neurons.
    
    '''
    all_combinations = list(product(x_shape_space, y_shape_space))
    sampled_combinations = np.random.choice(len(all_combinations), number_nrns, replace=True)
    neuron_shapes = [all_combinations[i] for i in sampled_combinations]
    return neuron_shapes


def generate_neuron_centers(number_nrns,
                            frame_size=[400,400],
                            min_distance=20,
                            spread=8):
    
    ''' Generates random centers for the neurons.
    Args:
        N (int): The number of neurons to generate.
        frame_size (tuple): The size of the frame.
        min_distance (int): The minimum distance in pxl sq between neurons.
        spread (float): scalar that determines how spread the neurons 
        are aroudn the center of the frame

    Returns:
        list: A list of tuples representing the centers of the neurons.
    '''
    
    points = []
    
    while len(points) < number_nrns:
        frame_center = [int(frame_size[0]/2), int(frame_size[1]/2)]
        # by default, std is 15% of the spatial dimension
        bd = 0.15 * np.array(frame_size) 
        o = np.random.multivariate_normal(frame_center[::-1], spread * np.array([[bd[0]**2, 0],
                                                                                 [0, bd[1]**2]]), )
        point = [int(o[0]), int(o[1])]
        if points == []:
            if isvalid(point, frame_size, marginx=10, marginy=10):
                points.append(point)
            else:
                print('point outside the frame limits', point)
        else:
            if isvalid(point, frame_size, marginx=10, marginy=10):
                if all(distance.euclidean(point, existing_point) >= min_distance for existing_point in points):
                    points.append(point)
                # TODO: maybe handle organized overlap here
            else:
                print('point outside the frame limits', point, len(points))

    return points


def add_overlap(points:np.array, 
                num_overlaps: int, 
                qualifier:float, 
                nozone:float, 
                non_overlap_min_distance:float,
                frame_size:list,
                marginx:int,
                marginy:int):
    """if selected by the user, intentionally adds overlaps
    between a select number of neurons.
    
    ARGS:
        points
        num_overlaps
        qualifier
        nozone

    """
    og_numpoints = len(points)
    print(og_numpoints)
    # select num_overlaps neurons and delete them
    to_delete = np.random.choice(np.arange(len(points)), num_overlaps)
    points = np.delete(points, to_delete, axis=0)
    print(points.shape)
    
    # pick a point to add overlap to, then pick an angle, then 
    # sample a magnitude between qualifier and nozone, then 
    # step that much in that direction
    num_added = 0
    overlapping_points = []
    while num_added<num_overlaps:
        # print('trying a new point')
        idx = np.random.randint(0, points.shape[0]-num_added)

        p = points[idx]
        
        theta = np.random.rand(1)*2*np.pi
        # magnitude = np.random.rand(1) * (1-(nozone/qualifier)) + (nozone/qualifier)
        magnitude = np.random.rand(1) * (1-nozone/qualifier) + nozone/qualifier
        magnitude *= qualifier
        
        x_step = int(np.ceil(magnitude * np.cos(theta)))
        y_step = int(np.ceil(magnitude * np.sin(theta)))
        new_point = p + [x_step, y_step]
        if isvalid(new_point, frame_size, marginx, marginy):
            if all(distance.euclidean(new_point, existing_point) \
                >= non_overlap_min_distance for existing_point in np.delete(points, idx, 0)):
                print(distance.euclidean(p, new_point))
                points = np.vstack([points, new_point])
                overlapping_points.append([p, new_point])
                num_added += 1
                # print('overlap successfull')
            else:
                logging.debug(f'while adding overlap {num_added+1}, overlap not permitted; trying a new point')
                pass
    print('total # of points', points.shape[0])
    return points, overlapping_points


def scale_fluo(fluo, min_s=0.2, max_s=1):
    
    ''' 
    Scales the fluorescence signal of a neuron.

    ARGS:
        fluo (np.array): The fluorescence signal to scale.
        min_s (float): The minimum scaling factor.
        max_s (float): The maximum scaling factor.

    Returns:
        np.array: The scaled fluorescence signal.
    '''
    s = np.random.uniform(min_s, max_s, fluo.shape[0])
    return fluo * s[:, np.newaxis]


def create_spatial(fluo,
                   neuron_shapes,
                   neuron_centers,
                   frame_size,
                   neighborhood,
                   shape_scalar=1):

    '''
    Creates a spatial representation of the neurons.

    Parameters:
        fluo (np.array): [neurons, time] The fluorescence signal of the neurons.
        neuron_shapes (list): The shapes of the neurons, given as standard dev in (x, y).
        neuron_centers (list): The centers of the neurons.
        frame_size (tuple): The size of the frame.
        neighborhood (int): The size of the neighborhood around each neuron to consider for the simulation.
        shape_scalar (float): controls the size of the neuron, higher = smaller neuron


    Returns:
        np.array: the movie with flashing neurons
    '''
    time = fluo.shape[1]
    out = np.ones([time, *frame_size]) * 0.96

    x = np.arange(frame_size[0])
    y = np.arange(frame_size[1])
    x, y = np.meshgrid(x, y, indexing='ij')
    
    for (f, s, l) in tqdm(zip(fluo, neuron_shapes, neuron_centers)):
        y0, x0 = l
        sigma_x, sigma_y = s
        
        x_start, x_end = max(0, x0-neighborhood), min(frame_size[0], x0+neighborhood)
        y_start, y_end = max(0, y0-neighborhood), min(frame_size[1], y0+neighborhood)
        
        # Create the mask for the neighborhood
        x_mask = (x >= x_start) & (x < x_end)
        y_mask = (y >= y_start) & (y < y_end)
        mask = x_mask & y_mask
            
        for t in range(time):
            out[t] += f[t] * np.exp( 
                -shape_scalar * (
                ((x-x0)**2) / (2 * sigma_x**2) +
                ((y-y0)**2) / (2 * sigma_y**2)
                )) #* mask

    return out


def add_noise(video, noise_type='gaussian', noise_level=0.1, seed=None, smoothness=0):
    """
    Add noise to a simulated video.

    ARGS:
        video: 3D numpy array of shape (num_frames, height, width) representing the video.
        noise_type: Type of noise to add ('gaussian', 'poisson', 'salt_and_pepper').
        noise_level: Intensity or variance of the noise.
        seed: Random seed for reproducibility.
        smooth: Smoothness of the noise in frames. if set to 0, no smoothing is applied.

    Returns:
    - Noisy video as a 3D numpy array.
    """
    if seed is not None:
        np.random.seed(seed)

    noisy_video = video.copy()

    if noise_type == 'gaussian':
        mean = 0
        std = noise_level * np.max(video)
        gaussian_noise = np.random.normal(mean, std, video.shape)
        if smoothness>0:
            noisy_video += gaussian_filter(gaussian_noise, sigma=1)
        noisy_video += gaussian_noise

    elif noise_type == 'poisson':
        noisy_video = np.random.poisson(video * noise_level) / noise_level
        if smoothness > 0:
            print('noise type not supported with smoothness, returning noisy video without smoothing')

    elif noise_type == 'salt_and_pepper':
        prob = noise_level
        salt_pepper_noise = np.random.choice([0, 1, 2], size=video.shape, p=[prob / 2, 1 - prob, prob / 2])
        noisy_video[salt_pepper_noise == 0] = 0
        noisy_video[salt_pepper_noise == 2] = np.max(video)
        if smoothness > 0:
            print('noise type not supported with smoothness, returning noisy video without smoothing')

    return np.clip(noisy_video, 0, 1)  # Clip values to maintain valid pixel range


def add_smooth_noise(video, noise_level=0.1, temporal_sigma=1.0, seed=None):
    """
    Add noise to a simulated video with smooth transitions over time.

    ARGS:
        video: 3D numpy array of shape (num_frames, height, width) representing the video.
        noise_type: Type of noise to add ('gaussian', 'poisson', 'salt_and_pepper').
        noise_level: Intensity or variance of the noise.
        temporal_sigma: Sigma for temporal smoothing of the noise.
        seed: Random seed for reproducibility.

    Returns:
    - Noisy video as a 3D numpy array.
    """
    if seed is not None:
        np.random.seed(seed)

    # Copy the video to avoid modifying the original
    noisy_video = video.copy()

    # Generate Gaussian noise
    mean = 0
    std = noise_level * np.max(video)
    noise = np.random.normal(mean, std, video.shape)
    
    # Smooth the noise over time
    smoothed_noise = gaussian_filter(noise, sigma=(temporal_sigma, 1, 1))
    
    # Add the smoothed noise to the video
    noisy_video += smoothed_noise

    return np.clip(noisy_video, 0, 1)  # Assuming pixel values are normalized between 0 and 1



def apply_vignette(movie, strength=0.5):
    """
    Apply a vignette effect to each frame of a movie.

    ARGS:
        movie: 3D numpy array representing the movie with shape [time, height, width].
        strength: Strength of the vignette effect (0 to 1). Higher values produce a stronger vignette.

    Returns:
    - Vignetted movie as a 3D numpy array.
    """
    num_frames, rows, cols = movie.shape
    center_x, center_y = cols / 2, rows / 2
    
    # Create a grid of (x, y) coordinates
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    
    # Compute the distance from the center
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette_mask = np.exp(-distance ** 2 / (2 * (strength * max_distance) ** 2))
    
    # Normalize the vignette mask to have a maximum value of 1
    vignette_mask /= np.max(vignette_mask)
    
    # Apply the vignette mask to each frame
    vignetted_movie = np.empty_like(movie)
    for t in range(num_frames):
        vignetted_movie[t] = movie[t] * vignette_mask
    
    return vignetted_movie


def frame_intensity_to_rgba(frame, minval=0, maxval=255, colormap=cv2.COLORMAP_TURBO):
    new_frame = np.ones((frame.shape[0], frame.shape[1], 4))
    disp_frame = frame.copy().astype("float")
    disp_frame -= minval
    disp_frame[disp_frame < 0] = 0
    disp_frame /= np.abs(maxval - minval)
    disp_frame[disp_frame >= 1] = 1
    disp_frame *= 255
    bgr_frame = cv2.applyColorMap(disp_frame.astype(np.uint8), colormap)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    new_frame[:, :, :3] = rgb_frame
    new_frame = new_frame.astype(np.uint8)
    return new_frame



def movie_intensity_to_rgba(movie_array, minval=0, maxval=255, colormap=cv2.COLORMAP_TURBO):
    time, height, width = movie_array.shape
    new_movie_array = np.ones((time, height, width, 4), dtype=np.uint8)
    for t in range(time):
        frame = movie_array[t]
        # Normalize the frame
        disp_frame = frame.copy().astype("float")
        disp_frame -= minval
        disp_frame[disp_frame < 0] = 0
        disp_frame /= np.abs(maxval - minval)
        disp_frame[disp_frame >= 1] = 1
        disp_frame *= 255
        bgr_frame = cv2.applyColorMap(disp_frame.astype(np.uint8), colormap)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        new_movie_array[t, :, :, :3] = rgb_frame
        new_movie_array[t, :, :, 3] = 255  # Set alpha channel to fully opaque
    
    return new_movie_array


def get_colormap_from_string(colormap_str):
    # Extract the colormap name from the string
    colormap_name = colormap_str.split('.')[-1]
    
    # Get the colormap constant from the cv2 module
    colormap_constant = getattr(cv2, colormap_name, None)
    
    if colormap_constant is None:
        raise ValueError(f"Colormap '{colormap_str}' not found in cv2 module.")
    
    return colormap_constant


def save_array_as_avi(array, filename, fps=30, iscolor=False):
    """
    Saves a 3D NumPy array as an MP4 video file.
    
    ARGS:
        array (numpy.ndarray): The input array of shape [time, height, width, (channels, if rgb)].
        filename (str): The name of the output MP4 file.
        fps (int): Frames per second for the video.
    """
    if array.ndim == 3:
        time, height, width = array.shape
        # if RGB:
        #     array = array[:, :, [1, 0, 2]]
    if array.ndim == 4:
        time, height, width, _ = array.shape
        # if RGB:
        #     array = array[:, :, :, [1, 0, 2]]        

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Use XVID codec
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=iscolor)

    for t in range(time):
        if iscolor:
            frame = array[t][:,:,[2,1,0]]
        else:
            frame = array[t]
            frame = cv2.convertScaleAbs(frame) 

        out.write(frame)
    
    out.release()
    print(f"Video saved as {filename}")
