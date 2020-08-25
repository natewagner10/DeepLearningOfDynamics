# Deep Learning Of Dynamics
Code base for the Practical Data Science course's group project.

Contributors: Manda Bucklin, Rosa Gradilla, Justin Tienken-Harder, Dominic Ventura, and Nate Wagner

## Problem Statement

Given an image (or a video) of a dynamical system such as a swinging pendulum, an oscillating spring-mass, or even a flapping bird (biological system), we are interested in learning the dynamics of the system - i.e. we are interested in predicting the changes in the system over time using deep neural networks (NN). Once trained, our model must be able to predict the next state of the dynamical system given the current state (or the past few states) for instance given positions of a pendulum at time t=1sec then what will be the position at times t=2sec, t=3sec, etc. We require that our neural network respects the underlying physics of our system (such as conservation of energy, etc).
