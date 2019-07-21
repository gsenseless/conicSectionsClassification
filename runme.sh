#!/bin/bash
echo $PWD
chmod 777 $PWD

docker build -t cs_classification .
docker run -v $PWD:/home/jovyan/work cs_classification
