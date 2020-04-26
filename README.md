# pilco

Learn to balance baby.

![Balance](good_gifs/pendulum-test-20200422-081051.gif)

# Roadmap

Rename this repo. Candidates:
- Talos
- PRL
- IRL (Inference for RL)

## Clean up current code
- Docstrings - Sphinx
- Define documentation layout
- Remove all hacky stuff, like hard coded tensors
- Clean up pendulum.py: learning-dynamics, objective, optimisation, plotting

## Example notebooks
- Pendulum notebook

## Write derivations, including complexity
- Moment matching

## Profiling
- I have no clue how profiling works, let's research into how we should go about it

## Write tests (especially Monte Carlo - maybe one test for all moment matching)

## Run on other environments
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
