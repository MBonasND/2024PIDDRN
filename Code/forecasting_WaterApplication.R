######################################################
######################################################
### Forecasting with DESN, DDRN, PIDESN and PIDDRN ###
######################################################
######################################################

#clear environment
rm(list = ls()); gc()

#load libraries
library(tidyverse)
library(Matrix)
library(abind)
library(pracma)
library(reshape2)

#parallel libraries
library(doParallel)
library(parallel)
library(foreach)
library(doSNOW)

#load functions
source('data_processing.R')
source('deep_functions_physics_WaterApplication.R')

#specify cores
options(cores = 10)

#Define Physical Equation - NS
ns_equation = function(u_bar, yvals, nu, u_tau, delta, kappa = 0.15)
{
  u_y = pracma::gradient(u_bar, yvals)
  u_yy = pracma::gradient(u_y, yvals)
  
  term_1 = (1/delta)*(u_tau^2)
  term_2 = nu * u_yy
  term_3 = (kappa*yvals)^2 * abs(u_y) * u_y
  output = term_1 + term_2 + term_3
  
  return(output)
}



########################
### DESN Forecasting ###
########################

  
#Parameter specification
layers = 2
n.h = c(rep(150,layers))
nu = c(0.4, 0.8)
lambda.r = 1e-1
m = 0
alpha = 0.66
reduced.units = 30

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 100
tau = 10
validLen = 0
trainLen = 500-2*tau-validLen
testLen = tau
locations = dim(rawData)[2]

#Create training and testing sets
sets = cttv(rawData, tau, trainLen, testLen = testLen)
new.train = sets$yTrain
testindex = sets$xTestIndex

#Generating input data
input.dat = gen.input.data(trainLen = trainLen,
                           m = m,
                           tau = tau,
                           yTrain = (new.train),
                           rawData = (rawData),
                           locations = locations,
                           xTestIndex = testindex,
                           testLen = testLen)
y.scale = input.dat$y.scale
y.train = input.dat$in.sample.y
designMatrix = input.dat$designMatrix
designMatrixOutSample = input.dat$designMatrixOutSample
addScaleMat = input.dat$addScaleMat


start = proc.time()
testing = deep.esn(y.train = y.train,
                   x.insamp = designMatrix,
                   x.outsamp = designMatrixOutSample,
                   y.test = sets$yTest,
                   n.h = n.h,
                   nu = nu,
                   pi.w = pi.w, 
                   pi.win = pi.win,
                   eta.w = eta.w,
                   eta.win = eta.win,
                   lambda.r = lambda.r,
                   alpha = alpha,
                   m = m,
                   iter = iterations,
                   future = testLen,
                   layers = layers,
                   reduced.units = reduced.units,
                   startvalues = NULL,
                   activation = 'tanh',
                   distribution = 'Unif',
                   scale.factor = y.scale,
                   scale.matrix = addScaleMat,
                   fork = FALSE, 
                   parallel = TRUE,
                   verbose = FALSE, 
                   logNorm = FALSE)
proc.time()-start



esn_forcs = testing$forecastmean

  
########################
### DDRN Forecasting ###
########################

#LONG RUN TIME --- INCREASE CORES TO IMPROVE SPEED


#Parameter specification
layers = 2
n.h = c(rep(150,layers))
nu = c(0.4, 0.8)
lambda.r = 1e-1
m = 0
alpha = 0.66
reduced.units = 30

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 100
tau = 10
validLen = 0
trainLen = 500-2*tau-validLen
testLen = tau
locations = dim(rawData)[2]

#Spiking Parameters
timescale = 100
leakage = 0.9
threshold = 2.5
latency = 0
inhibitor = 1.0
resting = 0
stepmax = 200000
subsample = 0.05

#Create training and testing sets
sets = cttv(rawData, tau, trainLen, testLen = testLen)
new.train = sets$yTrain
testindex = sets$xTestIndex

#Generating input data
input.dat = gen.input.data(trainLen = trainLen,
                           m = m,
                           tau = tau,
                           yTrain = new.train,
                           rawData = rawData,
                           locations = locations,
                           xTestIndex = testindex,
                           testLen = testLen)
y.scale = input.dat$y.scale
y.train = input.dat$in.sample.y
designMatrix = input.dat$designMatrix
designMatrixOutSample = input.dat$designMatrixOutSample
addScaleMat = input.dat$addScaleMat


Sys.time()
start = proc.time()
testing = combo_spiking(y.train = y.train,
                        x.insamp = designMatrix,
                        x.outsamp = designMatrixOutSample,
                        y.test = sets$yTest,
                        n.h = n.h,
                        nu = nu,
                        pi.w = pi.w, 
                        pi.win = pi.win,
                        eta.w = eta.w,
                        eta.win = eta.win,
                        lambda.r = lambda.r,
                        alpha = alpha,
                        m = m,
                        timescale = timescale, 
                        threshold = threshold,
                        latency = latency,
                        leakage = leakage,
                        resting = resting,
                        inhibitor = inhibitor,
                        iter = iterations,
                        future = testLen,
                        layers = layers,
                        reduced.units = reduced.units,
                        startvalues = NULL,
                        activation = 'tanh',
                        distribution = 'Unif',
                        physics = FALSE,
                        PDE = ns_equation,
                        PDE_params = list('nu' = 1e-06, u_tau = 0.027, delta = 0.1, kappa = 0.15),
                        time_vals = NULL,
                        location_vals = yvals,
                        stepmax = stepmax,
                        subsample = subsample,
                        scale.factor = y.scale,
                        scale.matrix = addScaleMat,
                        fork = FALSE,
                        logNorm = FALSE)
end = proc.time()
end-start


ddrn_forcs = testing$forecastmean


  
##########################
### PIDESN Forecasting ###
##########################


#LONGER RUN TIME --- INCREASE CORES TO IMPROVE SPEED


#Parameter specification
layers = 2
n.h = c(rep(150,layers))
nu = c(0.4, 0.8)
lambda.r = 1e-1
m = 0
alpha = 0.66
reduced.units = 30
stepmax = 200000
subsample = 0.05
rho = 0.5 #chi in manuscript

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 100
tau = 10
validLen = 0
trainLen = 500-2*tau-validLen
testLen = tau
locations = dim(rawData)[2]


#Create training and testing sets
sets = cttv(rawData, tau, trainLen, testLen = testLen)
new.train = sets$yTrain
testindex = sets$xTestIndex

#Generating input data
input.dat = gen.input.data(trainLen = trainLen,
                           m = m,
                           tau = tau,
                           yTrain = (new.train),
                           rawData = (rawData),
                           locations = locations,
                           xTestIndex = testindex,
                           testLen = testLen)
y.scale = input.dat$y.scale
y.train = input.dat$in.sample.y
designMatrix = input.dat$designMatrix
designMatrixOutSample = input.dat$designMatrixOutSample
addScaleMat = input.dat$addScaleMat


start = proc.time()
testing = deep.esn.physics(y.train = y.train,
                           x.insamp = designMatrix,
                           x.outsamp = designMatrixOutSample,
                           y.test = sets$yTest,
                           n.h = n.h,
                           nu = nu,
                           pi.w = pi.w, 
                           pi.win = pi.win,
                           eta.w = eta.w,
                           eta.win = eta.win,
                           lambda.r = lambda.r,
                           alpha = alpha,
                           m = m,
                           iter = iterations,
                           future = testLen,
                           layers = layers,
                           reduced.units = reduced.units,
                           startvalues = NULL,
                           activation = 'tanh',
                           distribution = 'Unif',
                           physics = TRUE,
                           PDE = ns_equation,
                           PDE_params = list('nu' = 1e-06, u_tau = 0.027, delta = 0.1, kappa = 0.15),
                           time_vals = NULL,
                           location_vals = yvals,
                           stepmax = stepmax,
                           subsample = subsample,
                           rho = rho,
                           scale.factor = y.scale,
                           scale.matrix = addScaleMat,
                           fork = FALSE,
                           logNorm = FALSE)
proc.time()-start



pidesn_forcs = testing$forecastmean




##########################
### PIDDRN Forecasting ###
##########################

#LONGEST RUN TIME --- INCREASE CORES TO IMPROVE SPEED


#Parameter specification
layers = 2
n.h = c(rep(150,layers))
nu = c(0.4, 0.8)
lambda.r = 1e-1
m = 0
alpha = 0.66
reduced.units = 30

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 50
tau = 10
validLen = 0
trainLen = 500-2*tau-validLen
testLen = tau
locations = dim(rawData)[2]

#Spiking Parameters
timescale = 100
leakage = 0.9
threshold = 2.5
latency = 0
inhibitor = 1.0
resting = 0
stepmax = 200000
subsample = 0.05
rho = 0.5 #chi in manuscript

#Create training and testing sets
sets = cttv(rawData, tau, trainLen, testLen = testLen)
new.train = sets$yTrain
testindex = sets$xTestIndex

#Generating input data
input.dat = gen.input.data(trainLen = trainLen,
                           m = m,
                           tau = tau,
                           yTrain = new.train,
                           rawData = rawData,
                           locations = locations,
                           xTestIndex = testindex,
                           testLen = testLen)
y.scale = input.dat$y.scale
y.train = input.dat$in.sample.y
designMatrix = input.dat$designMatrix
designMatrixOutSample = input.dat$designMatrixOutSample
addScaleMat = input.dat$addScaleMat


Sys.time()
start = proc.time()
testing = combo_spiking(y.train = y.train,
                        x.insamp = designMatrix,
                        x.outsamp = designMatrixOutSample,
                        y.test = sets$yTest,
                        n.h = n.h,
                        nu = nu,
                        pi.w = pi.w, 
                        pi.win = pi.win,
                        eta.w = eta.w,
                        eta.win = eta.win,
                        lambda.r = lambda.r,
                        alpha = alpha,
                        m = m,
                        timescale = timescale, 
                        threshold = threshold,
                        latency = latency,
                        leakage = leakage,
                        resting = resting,
                        inhibitor = inhibitor,
                        iter = iterations,
                        future = testLen,
                        layers = layers,
                        reduced.units = reduced.units,
                        startvalues = NULL,
                        activation = 'tanh',
                        distribution = 'Unif',
                        physics = TRUE,
                        PDE = ns_equation,
                        PDE_params = list('nu' = 1e-06, u_tau = 0.027, delta = 0.1, kappa = 0.15),
                        time_vals = NULL,
                        location_vals = yvals,
                        stepmax = stepmax,
                        subsample = subsample,
                        rho = rho,
                        scale.factor = y.scale,
                        scale.matrix = addScaleMat,
                        fork = FALSE,
                        logNorm = FALSE)
end = proc.time()
end-start


piddrn_forcs = testing$forecastmean






