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
source('Code/data_processing.R')
source('Code/deep_functions_physics.R')

#specify cores
options(cores = 10)

#PDE
burger = function(u, xvals, time, nu)
{
  u_t = pracma::gradient(u, xvals, time)$Y
  u_x = pracma::gradient(u, xvals, time)$X
  u_xx = pracma::gradient(u_x, xvals, time)$X
  f = u_t + u*u_x - nu*u_xx
  
  return(f)
}

#load data and clean
load('Data/AllBurgerData.RData')
load('Data/AllBurgerDataNames.RData')
index = 3
dat = burger_data[[index]]
viscosity = as.numeric(str_split(burger_list_files, pattern = '_')[[index]][3])
total_locs = 30
wanted_locs = round(seq(1,50,length.out = total_locs))
xvals = seq(-1,1,length.out = 50)
rawData = dat[,wanted_locs]



########################
### DESN Forecasting ###
########################

  
#Parameter specification
layers = 2
n.h = c(rep(150,layers))
nu = c(0.8, 1.0)
lambda.r = 1e-3
m = 1
alpha = 0.02
reduced.units = 20

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 20
tau = 50
validLen = 0
trainLen = 450-tau-validLen
testLen = 50
locations = total_locs

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
nu = c(0.8,1.0)
lambda.r = 0.001
m = 1
alpha = 0.02
reduced.units = 20

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
eta.win = rep(1,layers)
pi.win = rep(0.1, layers)
iterations = 20
tau = 50
validLen = 0
trainLen = 450-tau-validLen
testLen = 50
locations = total_locs

#Spiking Parameters
timescale = 100
leakage = 0.3
threshold = 2
latency = 2
inhibitor = 0.3
resting = 0
stepmax = 200000
subsample = 0.15

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
                        PDE = burger,
                        PDE_params = list('nu' = viscosity/pi),
                        time_vals = 3/1000,
                        location_vals = xvals[wanted_locs],
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
nu = c(0.8, 1.0)
lambda.r = 1e-3
m = 1
alpha = 0.02
reduced.units = 20
stepmax = 200000
subsample = 0.15
rho = 0.5 #chi in manuscript

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 20
tau = 50
validLen = 0
trainLen = 450-tau-validLen
testLen = 50
locations = total_locs

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
                           PDE = burger,
                           PDE_params = list('nu' = viscosity/pi),
                           time_vals = 3/1000,
                           location_vals = xvals[wanted_locs],
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
nu = c(0.8,1.0)
lambda.r = 0.001
m = 1
alpha = 0.02
reduced.units = 20

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
eta.win = rep(1,layers)
pi.win = rep(0.1, layers)
iterations = 100
tau = 50
validLen = 0
trainLen = 450-tau-validLen
testLen = 50
locations = total_locs

#Spiking Parameters
timescale = 100
leakage = 0.3
threshold = 2
latency = 2
inhibitor = 0.3
resting = 0
stepmax = 200000
subsample = 0.15
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
                        PDE = burger,
                        PDE_params = list('nu' = viscosity/pi),
                        time_vals = 3/1000,
                        location_vals = xvals[wanted_locs],
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






