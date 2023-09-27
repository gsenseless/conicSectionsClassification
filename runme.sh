#!/bin/bash
echo $PWD

docker build -t cs_classification .
docker run -v $PWD:/home/jovyan/work cs_classification
### to run with GPU uncomment the line below and uncomment "ag_args_fit={'num_gpus': 1}"" in script
#docker run --gpus all -v $PWD:/home/jovyan/work cs_classification 
