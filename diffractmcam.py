import json
import os
from datetime import datetime as _datetime

import cv2
import jax.numpy as jnp
import jax.scipy
import numpy as np
import optax
import xarray as xr
from jax import grad, jit, random, vmap
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


def load_mcam_data(mcam_data_file, cam_x_slice, cam_y_slice, downsamp_factor, mcam_state, video_frame=None,
                   remove_static=False, percentile=.3, static_bkgd=None, use_jax=True, compute_by_cam_row=False,
                   rectify=False, interpolated_downsamp=True, weighted_pixel_bias=False):
    """
    Load and preprocess MCAM data (one frame at a time).

    Parameters:
    - mcam_data_file: str or dataset, filepath or preloaded dataset.
    - cam_x_slice, cam_y_slice: slice, camera slice range.
    - downsamp_factor: int, factor to downsample the data.
    - mcam_state: dict, state of the MCAM.
    - video_frame: int or array, specific frame(s) to load.
    - remove_static: bool, whether to remove static background.
    - percentile: float, percentile for background estimation.
    - static_bkgd: array, precomputed static background.
    - use_jax: bool, use JAX for computation.
    - compute_by_cam_row: bool, compute background row by row.
    - rectify: bool, clip data below 0.
    - interpolated_downsamp: bool, use interpolation for downsampling.
    - weighted_pixel_bias: bool, weight pixel bias based on static background.

    Returns:
    - mcam_images: 4Darray, processed MCAM images.
    - static_bkgd: array, static background.
    """

    # Load dataset, allow for preloading
    if isinstance(mcam_data_file, str):
        dataset = xr.load_dataset(mcam_data_file)
        dataset_images = dataset.images
    else:
        dataset_images = mcam_data_file

    # Process video frame, if an array of video_frame's is given, then it'll be collapsed to
    # one frame via the max operation.
    if video_frame is not None and not remove_static:
        dataset_images = dataset_images[video_frame]
        if np.shape(video_frame) != ():  # not a single value, i.e., an array
            dataset_images = dataset_images.max(0)
    mcam_images = np.asarray(dataset_images)

    # Rotate images 180 degrees
    mcam_images = mcam_images[..., ::-1, ::-1]

    # Downsample images
    if interpolated_downsamp and downsamp_factor != 1:
        new_shape = (*mcam_images.shape[:-2],
                     mcam_images.shape[-2] // downsamp_factor,
                     mcam_images.shape[-1] // downsamp_factor)
        downsampled = np.zeros(new_shape, dtype=np.uint8)
        if len(mcam_images.shape) == 4:  # no time dimension
            for i in tqdm(range(mcam_images.shape[0]), desc='downsampling mcam images'):
                for j in range(mcam_images.shape[1]):
                    downsampled[i, j] = cv2.resize(mcam_images[i, j], new_shape[-2:][::-1], cv2.INTER_LINEAR)
        elif len(mcam_images.shape) == 5:
            for i in tqdm(range(mcam_images.shape[0]), desc='downsampling mcam images'):
                for j in range(mcam_images.shape[1]):
                    for k in range(mcam_images.shape[2]):
                        downsampled[i, j, k] = cv2.resize(mcam_images[i, j, k], new_shape[-2:][::-1], cv2.INTER_LINEAR)
        else:
            raise Exception('invalid shape: ' + str(mcam_images.shape))
        mcam_images = downsampled
    else:
        mcam_images = mcam_images[..., ::downsamp_factor, ::downsamp_factor]

    # Compute static background if needed
    if remove_static:
        if static_bkgd is None:
            if use_jax:
                static_bkgd = jnp.zeros(mcam_images.shape[1:], dtype=jnp.float32)
                if not compute_by_cam_row:
                    for r_cam in tqdm(range(mcam_images.shape[1]), desc='computing background'):
                        for c_cam in range(mcam_images.shape[2]):
                            bkgd = jnp.percentile(mcam_images[:, r_cam, c_cam], percentile, axis=0)
                            static_bkgd = static_bkgd.at[r_cam, c_cam].set(bkgd)
                else:
                    for r_cam in tqdm(range(mcam_images.shape[1]), desc='computing background'):
                        for c_cam in range(mcam_images.shape[2]):
                            for row in range(mcam_images.shape[3]):
                                bkgd = jnp.percentile(mcam_images[:, r_cam, c_cam, row], percentile, axis=0)
                                static_bkgd = static_bkgd.at[r_cam, c_cam, row].set(bkgd)
            else:
                static_bkgd = np.percentile(mcam_images, percentile, axis=0).astype(np.float32)
        if np.shape(video_frame) == ():  # a single value
            mcam_images = mcam_images[video_frame].astype(np.float32) - static_bkgd
        else:
            mcam_images = mcam_images[video_frame].max(0).astype(np.float32) - static_bkgd
    else:
        static_bkgd = None
        mcam_images = mcam_images.astype(np.float32)

    # Apply pixel bias
    if weighted_pixel_bias:
        median_bkgd = np.median(static_bkgd, (2, 3))  # use one value per camera
        median_bkgd /= median_bkgd.max()
        mcam_images = mcam_images - mcam_state['pixel_bias'] * median_bkgd[:, :, None, None]
    else:
        mcam_images = mcam_images - mcam_state['pixel_bias']

    # Rectify images
    if rectify:
        mcam_images = np.maximum(mcam_images, 0)

    # Apply camera slices
    if cam_x_slice is not None:
        mcam_images = mcam_images[:, cam_x_slice]

    if cam_y_slice is not None:
        mcam_images = mcam_images[cam_y_slice]

    return mcam_images, static_bkgd


def generate_mcam_xy_coordinates(downsamp_factor, cam_x_slice, cam_y_slice, mcam_state, xy_center=np.array([0, 0])):
    """
    Generate global coordinates of all MCAM pixels.

    Parameters:
    - downsamp_factor: int, factor to downsample the data.
    - cam_x_slice, cam_y_slice: slice, camera slice range.
    - mcam_state: dict, state of the MCAM.
    - xy_center: 3D array, center coordinates for radial distortions.

    Returns:
    - xy_mcam: 5D array, global coordinates of MCAM pixels.
    """
    pixel_size = mcam_state['pixel_size_physical_um'] * downsamp_factor

    # physical coordinates without downsampling:
    num_y_pix = mcam_state['num_y_pix'] // downsamp_factor
    num_x_pix = mcam_state['num_x_pix'] // downsamp_factor

    x_pix = np.arange(num_x_pix) * pixel_size
    y_pix = np.arange(num_y_pix) * pixel_size
    x_pix, y_pix = np.meshgrid(x_pix, y_pix, indexing='xy')
    xy_pix = np.stack([x_pix, y_pix], axis=0)

    x_cam = np.arange(mcam_state['num_x_cam']) * mcam_state['camera_spacing_physical_um']
    y_cam = np.arange(mcam_state['num_y_cam']) * mcam_state['camera_spacing_physical_um']
    x_cam, y_cam = np.meshgrid(x_cam, y_cam, indexing='xy')
    xy_cam = np.stack([x_cam, y_cam], axis=0)

    if cam_x_slice is not None:
        xy_cam = xy_cam[:, :, cam_x_slice]

    if cam_y_slice is not None:
        xy_cam = xy_cam[:, cam_y_slice]

    xy = xy_cam[:, :, :, None, None] + xy_pix[
                                       :, None, None, :, :]  # shape: 2, num_x_cam, num_y_cam, num_x_pix, num_y_pix
    xy = xy - xy.mean((1, 2, 3, 4), keepdims=True)  # center the coordinates
    xy = xy + xy_center[:, None, None, None, None]  # for computing radial distortions

    xy_mcam = xy

    return xy_mcam


def visualize_mcam_data(mcam_images, xy_mcam, padding, recon_pixel_size_um, xy_ref=None, xy_FOV=None):
    """
    Visualize MCAM data by recreating the measurement.

    Parameters:
    - mcam_images: 4D array, MCAM images.
    - xy_mcam: 5D array, global coordinates of MCAM pixels.
    - padding: int or length-2 tuple, padding along each dimension.
    - recon_pixel_size_um: float, reconstruction pixel size in um.
    - xy_ref: 5D array, reference offset for centering coordinates. If not supplied, then computed as min.
    - xy_FOV: (2,) array, field of view in um. If not supplied, then computed as ranges.

    Returns:
    - measurement_recreated: 2D array, recreated measurement.
    """

    if xy_FOV is None:
        xy_FOV = xy_mcam.max((1, 2, 3, 4)) - xy_mcam.min((1, 2, 3, 4))  # field of view in um
    recon_dims = np.int32(np.ceil(np.ceil(xy_FOV / recon_pixel_size_um))) + 2 * padding + 1  # dims of the 2D array
    measurement_recreated = np.zeros(recon_dims)

    # convert xy coordinates to rc:
    if xy_ref is None:
        xy_ref = xy_mcam.min((1, 2, 3, 4), keepdims=True)
    cr = (xy_mcam - xy_ref) / recon_pixel_size_um + padding
    cr = cr.astype(np.int32)

    # scatter update the recon with mcam data:
    measurement_recreated[cr[0], cr[1]] = mcam_images
    
    # create nice plot of mcam measurement
    data_nan = measurement_recreated.copy().astype(np.float32)*np.nan
    data_nan[cr[0], cr[1]] = mcam_images

    plt.figure(figsize=(15,15))
    plt.imshow(data_nan, cmap='inferno')
    plt.clim([0, 10])
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return measurement_recreated


def get_psf_coordinates(mcam_state, theta=0, scale=1.005):
    """
    Get PSF coordinates with optional rotation and scaling. PSF coordinates are hard-coded, since this is our only PSF right now.

    Parameters:
    - mcam_state: dict, state of the MCAM.
    - theta: float, rotation angle in radians.
    - scale: float, scale factor for PSF coordinates.

    Returns:
    - xy_psf: array, PSF coordinates.
    """

    # psf coordinates (hard-coded, since this is our only psf right now)
    xcoord = np.array([31.0, 3.5, 13.5, 24.5, 3.5, 4.5, 27.5, 1.0, 29.5, 19.5, 6.5, 27.5, 26.5, 5.5,
                       16.0]) / 5 * 1000 * mcam_state['psf_flip_x']
    ycoord = np.array([34.5, 33.0, 33.5, 25.5, 17.0, 33.5, 32.5, 0.5, 3.0, 3.5, 9.5, 19.0, 3.5, 2.5,
                       18.0]) / 5 * 1000 * mcam_state['psf_flip_y']

    # the last coordinate is the zero order; make sure it's the center
    xcoord -= xcoord[-1]
    ycoord -= ycoord[-1]

    # this logic could be optimized later
    xcoord, ycoord = xcoord, -ycoord

    xy_psf = np.stack([xcoord, ycoord], axis=0)

    # correct rotation manually:
    rotmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    xy_psf = rotmat @ xy_psf

    return xy_psf


def get_recon_dims(xy_mcam, xy_psf, recon_pixel_size_um, padding):
    """
    Get reconstruction dimensions based on MCAM and PSF coordinates.

    Parameters:
    - xy_mcam: 5D array, global coordinates of MCAM pixels.
    - xy_psf: 3D array, PSF coordinates.
    - recon_pixel_size_um: float, reconstruction pixel size in um.
    - padding: int, padding for reconstruction.

    Returns:
    - recon: 2D array, blank reconstruction initialization.
    - xy_FOV: (2,) array, field of view in um.
    - xy_ref: 3D array, reference coordinates for reconstruction.
    - cr_all: 6D array, row-column coordinates for all the points contributing to the MCAM images according to PSF.
    - xy_all: 6D array, x-y coordinates for all the points contributing to the MCAM images according to PSF.
    """
    # all coordinates with multiplexing:
    xy_all = xy_mcam[:, :, :, :, :, None] + xy_psf[:, None, None, None, None, :]

    # origin reference to subtract xy coordinates
    xy_ref = xy_all.min((1, 2, 3, 4, 5), keepdims=True)

    # create reconstruction:
    xy_FOV = xy_all.max((1, 2, 3, 4, 5)) - xy_ref.squeeze()  # field of view in um
    recon_dims = np.int32(np.ceil(xy_FOV / recon_pixel_size_um)) + 2 * padding  # dims of the 2D array
    recon = np.zeros(recon_dims, dtype=np.float32)

    # convert xy coordinates to rc (not used if optimizing psf params):
    cr_all = (xy_all - xy_ref) / recon_pixel_size_um + padding
    cr_all = cr_all.astype(np.int32)

    return recon, xy_FOV, xy_ref, cr_all, xy_all


def get_mcam_subset(mcam_images, xy_mcam, xy_all, cr_all, recon, num_cam, cam_ul):
    """
    Get contiguous subsets of MCAM images and coordinates.

    Parameters:
    - mcam_images: array, MCAM images.
    - xy_mcam: array, global coordinates of MCAM pixels.
    - xy_all: array, all coordinates contributing to the MCAM images according to the PSF points.
    - cr_all: array, row-column coordinates for reconstruction.
    - recon: array, reconstruction data.
    - num_cam: tuple, number of cameras to slice.
    - cam_ul: tuple, upper left camera index.

    Returns:
    - mcam_images_subset: array, subset of MCAM images.
    - xy_mcam_subset: array, subset of MCAM coordinates.
    - xy_ref_subset: array, reference coordinates for subset.
    - recon_subset: array, subset of reconstruction data.
    - recon_x_subset: slice, x-axis slice for reconstruction.
    - recon_y_subset: slice, y-axis slice for reconstruction.
    """
    # select the camera subset:
    cam_x_subset = slice(cam_ul[0], cam_ul[0] + num_cam[0])
    cam_y_subset = slice(cam_ul[1], cam_ul[1] + num_cam[1])
    mcam_images_subset = mcam_images[cam_y_subset, cam_x_subset]
    xy_mcam_subset = xy_mcam[:, cam_y_subset, cam_x_subset]

    # select the recon crop (derive from coordinates):
    cr_min = cr_all[:, cam_y_subset, cam_x_subset].min(axis=(1, 2, 3, 4, 5))
    cr_max = cr_all[:, cam_y_subset, cam_x_subset].max(axis=(1, 2, 3, 4, 5))
    recon_x_subset = slice(cr_min[1], cr_max[1])
    recon_y_subset = slice(cr_min[0], cr_max[0])
    recon_subset = recon[recon_y_subset, recon_x_subset]

    xy_ref_subset = xy_all[:, cam_y_subset, cam_x_subset].min((1, 2, 3, 4, 5), keepdims=True)

    return mcam_images_subset, xy_mcam_subset, xy_ref_subset, recon_subset, recon_x_subset, recon_y_subset


def create_variables_and_optimizers(variable_initial_values, learning_rates,
                                    model_distortions, model_vignetting):
    """
    Create variables and optimizers for model training.

    Parameters:
    - variable_initial_values: dict, initial values for variables.
    - learning_rates: dict, learning rates for optimizers.
    - model_distortions: bool, model distortions flag.
    - model_vignetting: bool, model vignetting flag.
    - patch_accumulation_factor: int, factor for patch accumulation.

    Returns:
    - variables: dict, model variables.
    - optimizers: dict, model optimizers.
    - optimizer_states: dict, states of the optimizers.
    """
    optimizers = dict()
    variables = dict()
    for key in ['affinemat', 'psf_weights', 'recon', 'dc_offset']:
        variables[key] = variable_initial_values[key]
        optimizers[key] = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rates[key])

    if model_distortions:
        for key in ['xy_distort_center_tube_lens', 'xy_distort_center_objective',
                    'radial_distort_tube_lens', 'radial_distort_objective']:
            variables[key] = variable_initial_values[key]
            optimizers[key] = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rates[key])

    if model_vignetting:
        assert model_distortions  # need the distortion centers
        for key in ['sigma_tube_lens', 'sigma_objective']:
            variables[key] = variable_initial_values[key]
            optimizers[key] = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rates[key])

    optimizer_states = {key: optimizers[key].init(jnp.zeros_like(variables[key])) for key in optimizers.keys()}

    return variables, optimizers, optimizer_states


def distort_xy_coordinates(xy_batch, xy_psf, affinemat, xy0_tube, distort_tube, xy0_objective,
                            distort_objective):
    # Wrapper for _distort_xy_coordinates for external use to get spatially varying PSF. To do this, we need
    # the extra step of applying an affine transform to xy_psf to get xy_psf_warped (the input to
    # _distort_xy_coordinates).

    xy_psf_warped = affinemat @ xy_psf
    xy_all = _distort_xy_coordinates(xy_batch, xy_psf_warped, xy0_tube, distort_tube, xy0_objective,
                                     distort_objective)
    return xy_all


def _distort_xy_coordinates(xy_batch, xy_psf_warped, xy0_tube, distort_tube, xy0_objective,
                            distort_objective, return_radial_coordinates=False):
    # Internal function used by forward_model to apply distortion parameters to get the coordinates of the
    # spatially varying PSF.
    # xy_batch: the coordinates (in um) across the sensor
    # xy_psf_warped: nominal psf coordinates already warped by an affine transform (spatially-invariant)
    # xy0_tube, distort_tube, xy0_objective, distort_objective: the distortion parameters.
    # return_radial_coordinates: whether to return radial coordinates (to be used by vignetting optimization).

    # apply tube lens distortions:
    powers = jnp.arange(1, len(distort_tube) + 1)
    r2_image = jnp.sum((xy_batch / 40000 - xy0_tube[:, None, None, None, None]) ** 2, axis=0)
    M_tube = 1 + jnp.sum(distort_tube[:, None, None, None, None] *
                         jnp.power(r2_image[None, :, :, :, :], powers[:, None, None, None, None]), axis=0)
    xy_pupil_um = xy_batch * M_tube[None, :, :, :, :]  # same shape as xy_batch

    # undo DOE effects:
    xy_pupil_um = (xy_pupil_um[:, :, :, :, :, None] +
                   xy_psf_warped[:, None, None, None, None, :])  # shape: (*xy_batch.shape, num psf points (15))

    # apply objective distortions:
    powers = jnp.arange(1, len(distort_objective) + 1)
    r2_pupil = jnp.sum((xy_pupil_um / 40000 - xy0_objective[:, None, None, None, None, None]) ** 2, axis=0)
    M_objective = 1 + jnp.sum(distort_objective[:, None, None, None, None, None] *
                              jnp.power(r2_pupil[None, :, :, :, :, :], powers[:, None, None, None, None, None]),
                              axis=0)
    xy_all = xy_pupil_um * M_objective[None, :, :, :, :, :]  # same shape as xy_pupil_um

    # apply global affine:
    # xy_all = jnp.einsum('ij,jabcde->iabcde', global_affine, xy_all)

    if return_radial_coordinates:
        return xy_all, r2_image, r2_pupil
    else:
        return xy_all


def forward_model(variables, mcam_images, xy_mcam, xy_psf, regularization_dict,
                  padding, xy_ref, recon_pixel_size_um, mcam_state,
                  weighted_MSE=True, weight_delta=.1, interpolate=False, use_scatter=False,
                  model_distortions=True, model_vignetting=False, use_MAE=False):

    # use_scatter: don't use for training, only for diagnostics
    # weighted_MSE: give stronger weight to where data is closer to 0; discourages optimization from putting
    # weight_delta: the constant to add that tunes the relative weight of 0 data
    # patch_shape: relevant if batch_size_cam is used; then choose a random patch of shape patch_size (a 2-tuple), and
    # pick the same patch loction in every camera in the batch. If None, then use whole cameras.
    # accumulate_gradient: whether to accumulate gradient across multiple batches.
    # cr_blur_kernel: only relelvant if model_aberrations is not None. These are the col/row coords of the blur kernel.
    # L1 and TV reg coefficients.
    # use_MAE: otherwise, default is MSE

    if use_scatter:
        assert not interpolate

    if model_distortions or model_vignetting:  # need interpolation if optimizing calibration params
        if not interpolate:
            print('Warning: distortions are modeled but cant be optimized because you arent using pixel interpolation.')


    # unpack variables:
    recon = variables['recon']
    affinemat = variables['affinemat']
    psf_weights = variables['psf_weights']
    psf_weights = jnp.abs(psf_weights)  # force positive
    psf_weights = psf_weights / psf_weights.mean()  # normalize
    psf_weights = jnp.concatenate([psf_weights[0] * np.ones(xy_psf.shape[1] - 1, dtype=psf_weights.dtype),
                                   psf_weights[1] * np.ones(1, dtype=psf_weights.dtype)], axis=0)  # 0 vs 1st orders
    dc_offset = variables['dc_offset']
    if model_distortions:
        xy0_tube = variables['xy_distort_center_tube_lens']
        xy0_objective = variables['xy_distort_center_objective']
        distort_tube = variables['radial_distort_tube_lens']
        distort_objective = variables['radial_distort_objective']

    if model_vignetting:
        sigma_tube_lens = variables['sigma_tube_lens'] / 10000
        sigma_objective = variables['sigma_objective'] / 10000


    xy_batch = xy_mcam
    mcam_images_batch = mcam_images

    xy_psf_warped = affinemat @ xy_psf  # warp psf coordinates

    if model_distortions:  # code adapted from the point-based distortion estimation code, with shape modifications
        xy_all, r2_image, r2_pupil = _distort_xy_coordinates(xy_batch, xy_psf_warped, xy0_tube, distort_tube,
                                                             xy0_objective, distort_objective,
                                                             return_radial_coordinates=True)

    else:
        xy_all = (xy_batch[:, :, :, :, :, None] +
                  xy_psf_warped[:, None, None, None, None, :])  # all coordinates with multiplexing

    cr_all = (xy_all - xy_ref) / recon_pixel_size_um + padding

    if model_vignetting:
        assert model_distortions  # need r2_image and r2_pupil
        # two vignetting paths, depending on how far we're from the center:
        vignette_tube_lens = jnp.exp(-r2_image / 2 / sigma_tube_lens ** 2)
        vignette_objective = jnp.exp(-r2_pupil / 2 / sigma_objective ** 2)
        vignette = vignette_tube_lens[:, :, :, :, None] * vignette_objective
        psf_weights = psf_weights[None, None, None, None, :] * vignette

    if interpolate:
        # find neighboring pixels:
        cr_floor = jnp.floor(cr_all)
        cr_ceil = cr_floor + 1

        # distance to neighboring pixels:
        cr_floor_dist = cr_all - cr_floor
        cr_ceil_dist = cr_ceil - cr_all

        # cast
        cr_floor = cr_floor.astype(jnp.int32)
        cr_ceil = cr_ceil.astype(jnp.int32)

        # gather the 4 corners:
        recon_ff = recon[cr_floor[0], cr_floor[1]]  # shape: num camera, num pixels, num psf points(, num kernel points)
        recon_cc = recon[cr_ceil[0], cr_ceil[1]]
        recon_cf = recon[cr_ceil[0], cr_floor[1]]
        recon_fc = recon[cr_floor[0], cr_ceil[1]]

        # weights:
        weight_ff = cr_ceil_dist[0] * cr_ceil_dist[1]
        weight_cc = cr_floor_dist[0] * cr_floor_dist[1]
        weight_cf = cr_floor_dist[0] * cr_ceil_dist[1]
        weight_fc = cr_ceil_dist[0] * cr_floor_dist[1]

        # weighted average for interpolation:
        recon_gathered = recon_ff * weight_ff + recon_cc * weight_cc + recon_cf * weight_cf + recon_fc * weight_fc
        recon_gathered = recon_gathered * psf_weights[None, None, None, None, :]
        recon_gathered = recon_gathered.sum(-1)  # sum across all orders
    else:
        cr_all = cr_all.astype(jnp.int32)
        recon_gathered = recon[cr_all[0], cr_all[1]] * psf_weights[None, None, None, None, :]
        recon_gathered = recon_gathered.sum(-1)  # sum across all orders

    if use_scatter:  # use only for diagnostics, because inefficient (no need to create 'data' every time)
        xy_FOV_ = xy_mcam.max((1, 2, 3, 4)) - xy_mcam.min((1, 2, 3, 4))  # compute without PSF
        cr = (xy_mcam - xy_mcam.min((1, 2, 3, 4), keepdims=True)) / mcam_state['recon_pixel_size_um'] + padding
        cr = cr.astype(jnp.int32)
        prediction = jnp.zeros_like(recon)  # recon used to be measurement_recreated
        prediction = prediction.at[cr[0], cr[1]].add(recon_gathered)  # last dim of cr_all is the zero order
        data = jnp.zeros_like(recon)
        data = data.at[cr[0], cr[1]].add(mcam_images_batch)
        error = prediction - data - dc_offset
        MSE = jnp.mean(error ** 2)
    else:
        error = recon_gathered - mcam_images_batch - dc_offset
        if weighted_MSE:
            weight = 1 / (weight_delta + mcam_images_batch)
            if use_MAE:
                MSE = jnp.mean(weight * jnp.abs(error))
            else:
                MSE = jnp.mean(weight * error ** 2)
        else:
            if use_MAE:
                MSE = jnp.mean(jnp.abs(error))
            else:
                MSE = jnp.mean(error ** 2)
        prediction = recon_gathered

    loss_list = [MSE]

    # regularization:
    if 'L1' in regularization_dict:
        L1 = regularization_dict['L1'] * jnp.sum(jnp.sqrt(recon ** 2 + 1e-7))
        loss_list.append(L1)
    if 'L2' in regularization_dict:
        L2 = regularization_dict['L2'] * jnp.sum(recon ** 2)
        loss_list.append(L2)
    if 'TV' in regularization_dict:
        d0 = recon[1:, :-1] - recon[:-1, :-1]
        d1 = recon[:-1, 1:] - recon[:-1, :-1]
        TV = regularization_dict['TV'] * jnp.sum(jnp.sqrt(d0 ** 2 + d1 ** 2 + 1e-7))
        loss_list.append(TV)

    loss_list = jnp.array(loss_list)
    total_loss = jnp.sum(loss_list)

    return total_loss, (loss_list, prediction)


def get_acquisition_cam_slices(json_path):
    """
    Get slice objects for camera acquisition.

    Parameters:
    - json_path: str, path to the acquisition parameters JSON file.

    Returns:
    - x_slice: slice, x-axis slice for cameras.
    - y_slice: slice, y-axis slice for cameras.
    """
    if 'acquisition_parameters.json' not in json_path:
        json_path = os.path.join(json_path, 'acquisition_parameters.json')
    with open(json_path) as f:
        acquisition_params = json.load(f)
    y_slice = slice(acquisition_params['selection_slice'][0]['start'],
                    acquisition_params['selection_slice'][0]['stop'])
    x_slice = slice(acquisition_params['selection_slice'][1]['start'],
                    acquisition_params['selection_slice'][1]['stop'])
    return x_slice, y_slice


def filter_mcam_images(mcam_images, size=9, zero_frac=0.5):
    """
    Apply zero neighbor filter to MCAM images.

    Parameters:
    - mcam_images: array, MCAM images.
    - size: int, size of the neighborhood.
    - zero_frac: float, fraction of zero neighbors to filter.

    Returns:
    - mcam_images_filt: array, filtered MCAM images.
    """
    mcam_images_filt = np.zeros_like(mcam_images)
    for i in range(mcam_images.shape[0]):
        for j in range(mcam_images.shape[1]):
            mcam_images_filt[i, j] = zero_neighbor_filter(mcam_images[i, j], size, zero_frac)
    return mcam_images_filt


def zero_neighbor_filter(im, size=9, zero_frac=0.7):
    """
    Zero pixels based on zero neighbor fraction.

    Parameters:
    - im: array, input image.
    - size: int, size of the neighborhood.
    - zero_frac: float, fraction of zero neighbors to filter.

    Returns:
    - filtered_image: array, image after applying the filter.
    """
    assert size % 2 == 1  # must be odd

    # generate filter of ones except at the center
    kernel = np.ones((size, size), dtype=np.float32)
    kernel[size//2, size//2] = 0
    threshold = size ** 2 * (1-zero_frac)

    num_nonzero_neighbor = jax.scipy.signal.convolve2d(im != 0, kernel, 'same')  # number of nonzero neighbors
    return jnp.where(num_nonzero_neighbor <= threshold, 0, im)

def make_timestamp(datetime=None):
    """
    Make a timestamp with millisecond accuracy.

    Parameters:
    - datetime: datetime, datetime object to create the timestamp from.

    Returns:
    - timestamp: str, timestamp as a string with millisecond precision.
    """
    if datetime is None:
        datetime = _datetime.now()
    return datetime.strftime('%Y%m%d_%H%M%S_%f')[:-3]
