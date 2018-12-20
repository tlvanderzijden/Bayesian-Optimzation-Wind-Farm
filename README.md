# Thesis Bayesian Optimzation Wind Farms
This code belongs to the Bachelor Thesis: [A data-driven approach for wind farm power maximisation
using Bayesian Optimisation](Paper.pdf)
      

## Research
The objective of this research is to show that the power output of a scaled wind farm can be optimised, using a model
free optimisation algorithm. Traditionally, wind turbines are controlled to optimise their individual power production.
The result of this is that the first turbine has a high power output while the downstream turbines experience turbulent
wind, resulting in lower power as well as higher damage. A way to optimise the power production of the entire wind
farm is by misaligning some of the turbines in the yaw direction, thereby steering the turbulent wake away from
downstream turbines. In this study, Gaussian processes combined with Bayesian Optimisation have been used to approach the maximum power production of a scaled wind farm in as few measurements as possible. 

## Running the tests
[MultiDimensionalOptimization](MultiDimensionalOptimization.m) is the main script. Different optimization experiments can be executed with this script:
<li>Sample function: output of measurement point is sampled from a function (Branin)</li>
<li>Windtunnel: test in scaled wind farm as described in thesis. Power output of three wind turbines is measured</li>
<li>FLORIS: FLORIS is a model to estimate the behaviour of wakes in wind farms. A FLORIS model is defined and the power is sampled from the model.</li>

The variable `varExperiment` imports the experiment variables which are defined in a function. See folder 'Models' for different models/experiments

### Gaussian Processes and Bayesian Optimization 
More information about Gaussian Processes can be find in:<br>
[Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)-Rasmussen, C. E. (2004).<br>
And for Bayesian Optimization: <br>
[A Tutorial on Bayesian Optimization of
Expensive Cost Functions, with Application to
Active User Modeling and Hierarchical
Reinforcement Learning](https://arxiv.org/pdf/1012.2599.pdf)-Brochu, E., Cora, V. M., & De Freitas, N. (2010)<br>


## Deployment

Add additional notes about how to deploy this on a live system

## Authors
* **E.M. de Boer**
* **K.O. Koerten**
* **J. Langeveld**
* **T.L. van der Zijden**
## License

This project is licensed under the MIT License - see the [MultiDimensionalOptimization.m](MultiDimensionalOptimization.m) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
