#!/usr/bin/env bash

# Check args
if [ "$#" -ne 0 ]; then
	  echo "usage: ./run.sh"
	    return 1
    fi

    # Get this script's path
    pushd `dirname $0` > /dev/null
    SCRIPTPATH=`pwd`
    popd > /dev/null

    set -e

    docker run\
	    --shm-size 12G\
	    --gpus all\
		--net host\
		-e SHELL\
        -e DISPLAY=$DISPLAY\
		-e DOCKER=1\
		--name binpicking_ddpm\
		-v $(dirname `pwd`):/repos/cognitive_robotics_lab_2023_binpicking_ddpm\
		-v /home/nfs/inf6/data/datasets/lab_cogrob23/:/data\
		-it binpicking_ddpm