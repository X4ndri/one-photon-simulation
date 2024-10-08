
from sim_utils import read_config
from pathlib import Path
from sim_utils import *
import numpy as np
import yaml
import time
import h5py

def run_simulation(config_path, save=['mp4', 'h5']):

    config = read_config(config_path)
    
    # general
    parent_path = config['general']['parent_path']
    duration = config['general']['duration']
    number_nrns = config['general']['number_nrns']
    frame_size = config['general']['frame_size']
    sampling_rate = config['general']['sampling_rate']
    seed = config['general']['seed']
    normalize_to = config['general']['normalize_to']
    dtype = eval(config['general']['dtype'])

    # spiking
    rate = config['spiking']['rate']
    perturb = config['spiking']['perturb']
    perturb_range = config['spiking']['perturb_range']

    # fluo
    kernel_size = config['fluo']['kernel_size']
    tau_rise = config['fluo']['tau_rise']
    tau_decay = config['fluo']['tau_decay']
    scale = config['fluo']['scale']

    # sptaial
    neighborhood = config['spatial']['neighborhood']
    x_shape_space = config['spatial']['x_shape_space']
    y_shape_space = config['spatial']['y_shape_space']
    min_distance = config['spatial']['min_distance']
    overlap_threshold = config['spatial']['overlap_threshold']
    overlap_tolerance = config['spatial']['overlap_tolerance']
    shape_scalar = config['spatial']['shape_scalar']

    # noise
    noise = config['noise']['added_noise']
    vignette = config['noise']['vignette']

    # visualization
    fps = config['visualization']['fps']
    cmap = get_colormap_from_string(config['visualization']['cmap'])
    minval = config['visualization']['minval']
    maxval = config['visualization']['maxval']

    # set seed
    if seed:
        np.random.seed(seed)
    # set up directories and names
    daystamp = time.strftime('%Y%m%d')
    timestamp = time.strftime('%H%M%S')
    day_outputs_path = Path(parent_path).joinpath(daystamp)
    day_outputs_path.mkdir(exist_ok=True, parents=True)
    if noise:
        noise_levels_info= ['-'.join([v['type'][0], str(v['level']), str(v['smoothness'])]) for k, v in noise.items()]
        noise_levels_str = '_'.join(noise_levels_info)
    else:
        noise_levels_str = 'noiseless'
    run_name = f'{daystamp}_{timestamp}_n{number_nrns}_{noise_levels_str}'

    # get temporal activities
    fluo_gt = get_temporal_components_gt(duration=duration,
                                         number_nrns=number_nrns,
                                         sampling_rate=sampling_rate,
                                         rate=rate,
                                         kernel_size=kernel_size,
                                         tau_rise=tau_rise,
                                         tau_decay=tau_decay,
                                         perturb=perturb,
                                         perturb_range=perturb_range,)
    
    # scale temporal activities
    fluo_scaled = scale_fluo(fluo_gt, *scale)

    # get neuron centers and shapes
    centers = generate_neuron_centers(number_nrns=number_nrns,
                                      frame_size=frame_size,
                                      min_distance=min_distance,
                                      overlap_threshold=overlap_threshold,
                                      overlap_tolerance=overlap_tolerance)
    
    shapes = generate_neuron_shapes(number_nrns=number_nrns,
                                    x_shape_space=x_shape_space,
                                    y_shape_space=y_shape_space)
    
    # create movie
    mov = create_spatial(fluo=fluo_scaled,
                         neuron_shapes=shapes,
                         neuron_centers=centers,
                         frame_size=frame_size,
                         neighborhood=neighborhood,
                         shape_scalar=shape_scalar)


    # add noise: the key of the noise dict is the order in which the noise is added
    order = 0
    # check if noise dictionary is empty
    if noise:
        for k, v in noise.items():
            if k == order:
                print(f'Adding noise of type {v["type"]} with level {v["level"]} and smoothness {v["smoothness"]}')
                mov = add_noise(mov, noise_type=v['type'], noise_level=v['level'], smoothness=v['smoothness'])
                order+=1
    

    if normalize_to:    
        mov *= normalize_to/mov.max()
        mov = mov.astype(dtype)

    if vignette:
        mov = apply_vignette(mov, vignette)
    
    if save is not None:
        if isinstance(save, str):
            save = [save]
        
        if 'h5' in save:
            # save movie as h5
            with h5py.File(day_outputs_path.joinpath(f'{run_name}.h5'), 'w') as file:
                file.create_dataset('mov', data=mov)
                file.create_dataset('fluo_gt', data=fluo_gt)
                file.create_dataset('fluo_scaled', data=fluo_scaled)
                file.create_dataset('centers', data=centers)
                file.create_dataset('shapes', data=shapes)
                file.create_dataset('config', data=yaml.dump(config))
        
        if 'mp4' in save:
            # convert to rgb before saving the mp4
            mov = movie_intensity_to_rgba(mov, colormap=cmap, minval=minval, maxval=maxval)[:,:,:,:3]
            save_array_as_mp4(mov, day_outputs_path.joinpath(f'{run_name}.mp4'), fps=fps, iscolor=True)
        
        # dump config to the run folder
        with open(day_outputs_path.joinpath(f'{run_name}.yaml'), 'w') as file:
            yaml.dump(config, file)

    return mov



if __name__ == '__main__':
    packdir = Path(__file__).parent
    config_path = packdir.joinpath('sim.yaml')
    run_simulation(config_path)