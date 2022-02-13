# AMAL-Project

## Description
Expriment for arcticle 

**IMITATION LEARNING  BY REINFORCEMENT LEARNING**

Based on implementation for article

**Random Expert Distillation: Imitation Learning via Expert Policy Support
Estimation**

For environment run 

**pip install -r requirements** 

# Environment
We used four models of environments provided by GYM:

'ant-bullet-medium-v0', 'halfcheetah-bullet-medium-v0', 'hopper-bullet-medium-v0', 'walker2d-bullet-medium-v0'

# Expert agent
Expert agent is provide by **d4rl-pybullet**

https://github.com/takuseno/d4rl-pybullet

The experts are trained with SAC model.

# Baseline
We test RED and compare it with GAIL

# Run project
python main.py -m algorithm=ALG/ENV

ALG is in [GAIL|RED]
ENV is in [ant|halfcheetah|hopper|walker2d]

# Reference
This project is based on an open source project provided by @Kaixhin

https://github.com/Kaixhin/imitation-learning
