#################
#################
### Functions ###
#################
#################

#load libraries
library(tidyverse)
library(Matrix)
library(abind)

set.seed(NULL)


#######################################################
### Generate Training, Testing, and Validation Sets ###
#######################################################


cttv = function(rawData, tau, trainLen, testLen, validLen = NULL, valid.flag = FALSE)
{
  #Create training and testing sets
  totlength = trainLen + testLen + tau
  yTrain = rawData[(tau+1):(trainLen+tau),]
  yTest = rawData[(trainLen+tau+1):totlength,]
  xTestIndex = seq((trainLen+1), (totlength-tau), 1)
  
  #Create valid sets
  if(valid.flag)
  {
    xValTestIndex=(trainLen+1-validLen):(trainLen)
    yValTestIndex=(trainLen+tau+1-validLen):(trainLen+tau)
    yValid = rawData[yValTestIndex,]
  } else {
    yValid = NULL
    xValTestIndex = NULL
  }
  
  #Return list
  output = list('yTrain' = yTrain,
                'yTest' = yTest,
                'yValid' = yValid,
                'xTestIndex' = xTestIndex,
                'xValTestIndex' = xValTestIndex)
  return(output)
}


###########################
### Generate Input Data ###
###########################


gen.input.data = function(trainLen,
                          m,
                          tau,
                          yTrain,
                          rawData,
                          locations,
                          xTestIndex,
                          testLen)
{
  in.sample.len = trainLen - (m * tau)
  
  in.sample.x.raw = array(NA, dim = c(in.sample.len, m+1, locations))
  
  for(i in 1:in.sample.len)
  {
    in.sample.x.raw[i,,] = rawData[seq(i, (m*tau + i), by=tau), ]
  }
  
  #Scale in-sample x and y
  in.sample.y.raw = yTrain[(m*tau + 1):trainLen,]
  y.mean = mean(in.sample.y.raw)
  y.scale = sd(in.sample.y.raw)
  
  in.sample.y = (in.sample.y.raw - y.mean)/y.scale
  
  
  mean.train.x = mean(rawData[1:trainLen,])
  sd.train.x = sd(rawData[1:trainLen,])
  
  
  in.sample.x=(in.sample.x.raw - mean.train.x)/sd.train.x
  
  
  designMatrix = matrix(1,in.sample.len, (m + 1)*locations + 1)
  for(i in 1:in.sample.len){
    designMatrix[i,2:((m + 1)*locations + 1)] = as.vector(in.sample.x[i,,])
  }
  
  
  #Out-Sample
  out.sample.x.raw = array(NA, dim = c(testLen, m + 1, locations))
  for(i in 1:testLen)
  {
    out.sample.x.raw[i,,] = rawData[seq(xTestIndex[i]-(m*tau), xTestIndex[i], by=tau),]
  }
  
  
  #Scale out-sample x and y
  out.sample.x = (out.sample.x.raw - mean.train.x)/sd.train.x
  
  designMatrixOutSample = matrix(1, testLen, (m + 1)*locations + 1)
  for(i in 1:testLen)
  {
    designMatrixOutSample[i,2:((m + 1)*locations + 1)] = as.vector(out.sample.x[i,,])
  }
  
  
  
  #Additive scale matric
  addScaleMat = matrix(y.mean, locations, testLen)
  
  input.data.output = list('y.mean' = y.mean,
                           'y.scale' = y.scale,
                           'in.sample.y' = in.sample.y,
                           'in.sample.x' = in.sample.x,
                           'out.sample.x' = out.sample.x,
                           'in.sample.len' = in.sample.len,
                           'designMatrix' = designMatrix,
                           'designMatrixOutSample' = designMatrixOutSample,
                           'testLen' = testLen,
                           'addScaleMat' = addScaleMat)
  return(input.data.output)
}
