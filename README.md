# diffractive-compressive-array-microscopy
Large-scale compressive microscopy via diffractive multiplexing across a sensor array

## Data
Two video datasets can be downloaded from [here](https://datadryad.org/share/LINK_NOT_FOR_PUBLICATION/eScwGsVTxkEkyjRQuEdzRPoSBVwh5D_Q8TuqOLZKfe8) -- put these in `./data`.

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
