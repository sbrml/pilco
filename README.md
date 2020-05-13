# pilco

[![Build Status](https://travis-ci.org/sbrml/pilco.svg?branch=master)](https://travis-ci.org/sbrml/pilco)
[![Coverage Status](https://coveralls.io/repos/github/sbrml/pilco/badge.svg?branch=master)](https://coveralls.io/github/sbrml/pilco?branch=master)

Learn to balance baby.

![Balance](good_gifs/pendulum-test-20200422-081051.gif)

# Roadmap

Rename this repo. Candidates:
- Talos
- PRL
- IRL (Inference for RL)

## Priorities
- Run on more environments: Cartpole, Mountaincar
- Tensor shapes on all methods

## Clean up current code
- Define documentation layout
- Docstrings - Sphinx
- [Doctest](https://docs.python.org/3/library/doctest.html)?
- Remove all hacky stuff, like hard coded tensors
- Clean up pendulum.py: learning-dynamics, objective, optimisation, plotting.
- Batching in calls of our agents, policies and costs. Start with policies.
- Migrate to gpflow.
- Feasible/Initialisation space - this needs more specification, included it so we don't forget.

## Example notebooks
- Pendulum notebook

## Write derivations, including complexity
- Moment matching

## Profiling
- I have no clue how profiling works, let's research into how we should go about it

## Write tests (especially Monte Carlo - maybe one test for all moment matching)
- EQAgent
- EQCost
- Transforms
- EQPolicy and TransformedPolicy
- Util (cholesky update)

## Run on other environments
- [Deepmind Control suite](https://arxiv.org/pdf/1801.00690.pdf)
- [OpenAI gym](https://github.com/openai/gym/wiki)
- Cartpole
- Mountaincar
- Double pendulum
- Lunar lander

## Control something real
- Cartpole swing
- Lego mindstorms
- Ask robotics faculty

## Future algorithms
- Non-greedy exploration (see the [DL-algo](https://www.mlmi.eng.cam.ac.uk/files/disentangling_sources_of_uncertainty_for_active_exploration_reduced.pdf))
- Efficient Learning of Dynamics with an information based criterion (IRL)
- Posterior Sampling for RL
- Deep PILCO
- [Embed to control](http://papers.nips.cc/paper/5964-embed-to-control-a-locally-linear-latent-dynamics-model-for-control-from-raw-images.pdf)
- [PlaNet](https://arxiv.org/pdf/1811.04551.pdf)
