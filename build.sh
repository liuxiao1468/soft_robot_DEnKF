#!/usr/bin/env bash
source conf.sh

if [[ -z $IMAGE_TAG ]]
then
  echo "No image tag provided. Not building image."
else
<<<<<<< HEAD
  docker build -t soft_robot/torch_filter:$IMAGE_TAG .
fi
=======
  docker build -t softrobot/torch_filter:$IMAGE_TAG .
fi
>>>>>>> af41d9018315342884776395f9ed09097ac8c9d8
