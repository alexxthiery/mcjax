#!/bin/bash

if_train=true
model_path='model_params.pkl'
if_animation=true
if_logZ=true

python dds.py \
  --if_train $if_train \
  --model_path $model_path \
  --if_animation $if_animation \
  --if_logZ $if_logZ