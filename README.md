# Large-scale compressive microscopy via diffractive multiplexing across a sensor array
We present a highly parallelized and multiplexed computational microscope that can achieve 3-μm resolution across a >5-cm<sup>2</sup> field of view at 120 fps, yielding a throughput of >25 gigapixels/sec.

For more details, see the accompanying [Nature Photonics paper](https://www.nature.com/articles/s41566-026-01974-4) or [arXiv preprint](https://arxiv.org/abs/2507.14437).

## Data
Two video datasets can be downloaded from [here](https://doi.org/10.5061/dryad.tmpg4f5b6) -- put these in `./data`. 

## Setting up your compute environment
Install docker and pull the following image:
```
docker pull ghcr.io/nvidia/jax:nightly-2023-12-12
```
This is the JAX version we used, but likely later versions will also work.
Use the provided dockerfile to create a custom image that contains other libraries needed to run the code (`cd` into the `./docker` directory):
```
sudo docker build -t jax:nightly-2023-12-12-custom .
```
For full reproducibility, we have included a `requirements-reproducibility.txt`. We used python 3.10.12, CUDA 12.2, NVIDIA Driver 535.288.01, Ubuntu 22.04, and a single NVIDIA GeForce RTX 3090 GPU. Pulling and creating the custom docker image should take a few mins, depending on your internet speed.

## Usage
Run the `reconstruction_without_fft.ipynb` Jupyter notebook and follow the instructions within. There are two rounds of optimization, one for estimating the spatially varying PSF across the extended field of view, the other for reconstructing the full video frame by frame. The first round generates a `.mat` calibration file to be used in the second round, but we also provide sample calibration files in `/distortion_params`. The bulk of the run time will be from the latter full-video optimization, which should take on the order of 5 sec per frame on a GPU (or up to an hour on a CPU) for a total of 1800 frames. Each frame can be reconstructed independently, so feel free to select a subset of the frames by adjusting `video_frames_to_recon` (e.g., every 10th frame: `np.arange(0, 1800, 10)`). The expected output is a sequence of reconstructed frames, saved as individual tiff files in `./recon` (e.g., `./recon/frame_0001.tiff`), which are featured in the supplementary videos of our [paper](https://www.nature.com/articles/s41566-026-01974-4).
