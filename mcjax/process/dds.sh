#!/bin/bash

if_train=true
model_path='model_params.pkl'
if_animation=true
add_score=true
variable_ts=false

python dds.py \
  --if_train $if_train \
  --model_path $model_path \
  --if_animation $if_animation \
  --add_score $add_score  \
  --variable_ts $variable_ts 
