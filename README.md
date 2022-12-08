# torch_filter
This is the pytorch differentiable filter for robot state estimation

## Getting Started
#### docker workflow

Clone the repo `git clone https://github.com/liuxiao1468/torch_filter.git`

#### Set the environment variables
Edit the `conf.sh` file to set the environment variables used to start the docker 
containers. 

```
IMAGE_TAG=  # unique tag to be used for the docker image.
CONTAINER_NAME=soft_robot  # name of the docker container.
DATASET_PATH=/home/xiao/datasets/  # Dataset path on the host machine.
CUDA_VISIBLE_DEVICES=0  # comma-separated list of GPU's to set visible.
```

#### Build the docker image
Run `./build.sh`

## Project Structure

#### Training or testing
Create or a modify a yaml file found in `./soft_robot` 
with the appropriate parameters. Set the mode parameter to perform the 
training or testing routine. 

```
mode:
    mode: 'train'  # 'train' | 'test'
```

Run the training and test script using the bash file `./run_filter.sh $CONFIG_FILE` 
where `$CONFIG_FILE` is the path to the config file. e.g. 
`./run_filter.sh ./config/tensegrity_xx.yaml`

View the logs with `docker logs -f $CONTAINER_NAME`

#### Tensorboard

Use the docker logs to copy the tensorboard link to a browser

```docker logs -f $CONTAINER_NAME-tensorboard```
 
#### Without Docker

If you don't want to use the docker container for training, you may directly use the
`train.py` script and pass in the config file. Make sure to have corresponding libraries and
dependencies installed on your local machine. Plase refer to `requirement.txt` and `Dockerfile` 
for those required packages.

Go to `./soft_robot` and then

Run `python train.py --config ./config/tensegrity_xx.yaml`


## Models
### DEnKF
Differentiable Ensemble Kalman Filter


## Datasets
#### Tensegrity_dataset
Ask IKemoto-san for the dataset.


## Results
TBD


## Model Zoo
TBD

