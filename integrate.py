import os
import h5py
import fabio
import numpy as np
from azint import AzimuthalIntegrator
import ipywidgets
from ipywidgets import Text, IntProgress
from IPython.display import display, clear_output

def integrate_file(fname, dset_name, config):
    if config['mask']:
        mask_fname = config['mask']
        ending = os.path.splitext(mask_fname)[1]
        if ending == '.npy':
            config['mask'] = np.load(mask_fname)
        else:
            config['mask'] = fabio.open(mask_fname).data 
    ai = AzimuthalIntegrator(**config)
        
    fh = h5py.File(fname, 'r')
    images = fh[dset_name]

    output_fname = fname.replace('raw', 'process/azint')
    root = os.path.splitext(output_fname)[0]
    output_fname = '%s_integrated.h5' %root
    output_folder = os.path.split(output_fname)[0]
    print(f'Integrating file: {os.path.split(fname)[1]}')
    print(f'Output file: {output_fname}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #output_fname = '/tmp/test.h5'

    output_fh = h5py.File(output_fname, 'w')
    shape = (len(images), *ai.output_shape)
    I_dset = output_fh.create_dataset('I', shape=shape, dtype=np.float32)
    if ai.error_model == 'poisson':
        sigma_dset = output_fh.create_dataset('sigma', shape=shape, dtype=np.float32)
    output_fh.create_dataset(ai.unit, data=ai.radial_axis)
    with open(config['poni_file'], 'r') as poni:
        p = output_fh.create_dataset('poni_file', data=poni.read())
        p.attrs['filename'] = config['poni_file']
    output_fh.create_dataset('mask_file', data=config['mask'])
    polarization_factor = config['polarization_factor'] 
    data = polarization_factor if polarization_factor is not None else 0
    output_fh.create_dataset('polarization_factor', data=data)
        
    progress = IntProgress(min=0, max=len(images))
    display(progress)

    for i, img in enumerate(images):
        if i % 10 == 0:
            progress.value = i
        I, sigma = ai.integrate(img)
        I_dset[i] = I
        if sigma is not None:
            sigma_dset[i] = sigma

    output_fh.close()
 
