# Cognitive Robotics Lab 2023 - Denoising Diffusion Probabilistic Models for Binpicking Scenes

This repository provides dataset classes and some utility functions for the relevant datasets, as well as a place to develop and submit your implementation of the proposed method.

## Requirements

1. Docker and NVIDIA Container Toolkit.
2. An RTX 40XX or 30XX GPU
3. Our NVMe server *avola* mounted on */home/nfs/inf6/data/...*.

## Setup

1. Clone this repository.
2. Build the project. Be aware that the transfer of user permissions at the end of the build takes about 30 minutes.
```console
cd denoising_diffusion/Docker && ./build.sh
```
3. Run a container.
```console
./run.sh
```
4. (Optional) For quick prototyping setup jupyter notebook in the container.
```console
jupyter notebook password
```
```console
cd /repos
```
```console
jupyter notebook --no-browser --ip 0.0.0.0 --port 9999 &
```
(Optional) Connect to the machine hosting the container, for example using (adapt to your user, login server and host machine):
```console
ssh -L 9999:localhost:9999 -J USER@login-stud.informatik.uni-bonn.de USER@rob4
```
In either case, the notebook is now available locally at [http://localhost:9999/](http://localhost:9999/).

## Getting Started

Sanity-checking the proposed method on FashionMNIST is reasonable, please see *FashionMNIST_Dataset.ipynb* on how to get started with that dataset. For the Binpicking scene dataset, please see *BinsceneA_Dataset.ipynb*.
