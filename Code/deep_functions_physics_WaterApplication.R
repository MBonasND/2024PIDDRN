#############################
#############################
### Forecasting Functions ###
#############################
#############################


#load libraries
library(tidyverse)
library(Matrix)
library(abind)
library(pracma)


###############################
### Deep Echo-State Network ###
###############################


deep.esn = function(y.train,
                    x.insamp,
                    x.outsamp,
                    y.test = NULL,
                    n.h,
                    nu,
                    pi.w, 
                    pi.win,
                    eta.w,
                    eta.win,
                    lambda.r,
                    alpha,
                    m,
                    iter,
                    future,
                    layers = 3,
                    reduced.units,
                    startvalues = NULL,
                    activation = 'tanh',
                    distribution = 'Normal',
                    logNorm = FALSE,
                    scale.factor,
                    scale.matrix,
                    parallel = F,
                    fork = F,
                    verbose = T)
{

  if(!parallel)
  {
    if(verbose)
    {
      prog.bar = txtProgressBar(min = 0, max = iter, style = 3)
    }
  }
  
  
  ###########################################
  ### Initial Conditions and Known Values ###
  ###########################################
  
  #Set training length and locations
  cap.t = dim(y.train)[1]
  locations = dim(y.train)[2]
  if(is.null(locations) | is.null(cap.t))
  {
    cap.t = length(y.train)
    locations = 1
  }
  
  #Get number of samples for weight matrices
  samp.w = list()
  samp.win = list()
  for(ell in 1:layers)
  {
    samp.w[[ell]] = n.h[ell] * n.h[ell]
    if(ell == 1)
    {
      n.x = (locations * (m+1)) + 1 #dim(x.insamp)[2]
      samp.win[[1]] = n.h[ell] * n.x
    } else {
      samp.win[[ell]] = n.h[ell] * (reduced.units+1)
    }
  }
  
  #Starting values of hidden units
  if(is.null(startvalues))
  {
    startvalues = list()
    for(ell in 1:layers)
    {
      startvalues[[ell]] = rep(0, n.h[ell])
    }
  }
  
  #Set the activation function
  if(activation == 'identity')
  {
    g.h = function(x)
    {
      return(x)
    } 
  } else if(activation == 'tanh') {
    g.h = function(x)
    {
      placeholder = tanh(x)
      return(placeholder)
    } 
  }
  
  #Set output array
  if(!parallel)
  {
    ensemb.mat = array(0, dim = c(locations, testLen, iter))
  }
  
  #########################
  ### Forecast Ensemble ###
  #########################
  set.seed(NULL)
  
  if(parallel)
  {
  
    #Specify Parallel clusters
    if(fork)
    {
      cl = parallel::makeForkCluster(getOption('cores')) 
    } else if(!fork)
    {
      cl = parallel::makeCluster(getOption('cores'))
    }
    
    #Activate clusters
    doParallel::registerDoParallel(cl)
    
    #Begin parallel iterations
    ensemb.mat = foreach::foreach(k = 1:iter,
                                  .combine = abind,
                                  .inorder = FALSE) %dopar%
      {
        set.seed(k)
        
        ##########################################
        ### Generate W and WIN weight matrices ###
        ##########################################
        W = list()
        WIN = list()
        lambda.w = c()
        for(ell in 1:layers)
        {
          #Set sparsity
          gam.w = purrr::rbernoulli(samp.w[[ell]], p = pi.w[ell])
          gam.win = purrr::rbernoulli(samp.win[[ell]], p = pi.win[ell])
          
          #Generate W
          if(distribution == 'Unif')
          {
            unif.w = runif(samp.w[[ell]], min = -eta.w[ell], max = eta.w[ell])
            W[[ell]] = Matrix::Matrix((gam.w == 1)*unif.w + (gam.w == 0)*0,
                                      nrow = n.h[ell], ncol = n.h[ell], sparse = T)
          } else if(distribution == 'Normal')
          {
            norm.w = rnorm(samp.w[[ell]], 0, 1)
            W[[ell]] = Matrix::Matrix((gam.w == 1)*norm.w + (gam.w == 0)*0,
                                      nrow = n.h[ell], ncol = n.h[ell], sparse = T)
          }
        
          #Generate W^in
          n.input = c(n.x, rep(reduced.units+1, (layers-1)))
          if(distribution == 'Unif')
          {
            unif.win = runif(samp.win[[ell]], min = -eta.win[ell], max = eta.win[ell])
            WIN[[ell]] = Matrix::Matrix((gam.win == 1)*unif.win + (gam.win == 0)*0,
                                 nrow = n.h[ell], ncol = n.input[ell], sparse = T)
          } else if(distribution == 'Normal')
          {
            norm.win = rnorm(samp.win[[ell]], 0, 1)
            WIN[[ell]] = Matrix::Matrix((gam.win == 1)*norm.win + (gam.win == 0)*0,
                                 nrow = n.h[ell], ncol = n.input[ell], sparse = T)
          }
          
          #Specify spectral radius
          lambda.w[ell] = max(abs(eigen(W[[ell]])$values))
    
        }
      
        
        ###############################
        ### Initialize Hidden Units ###
        ###############################
        h.prior = list()
        reservoir = list()
        h.forc.prior = list()
        forc.reservoir = list()
        Ident.Mat = diag(((layers-1)*reduced.units) + n.h[layers])
        
        for(ell in 1:layers)
        {
          h.prior[[ell]] = startvalues[[ell]]
          reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = cap.t)
          h.forc.prior[[ell]] = rep(0, n.h[ell])
          forc.reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = future)
        }
        
        
        ####################################
        ### Update Training Hidden Units ###
        ####################################
        input.data = list()
        input.data[[1]] = x.insamp
        output.data = list()
        output.data[[1]] = x.outsamp
        for(ell in 1:layers)
        {
          WIN.x.in.product = Matrix::tcrossprod(WIN[[ell]], input.data[[ell]])
          for(t in 1:cap.t)
          {
            omega = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.prior[[ell]] + WIN.x.in.product[,t]))
            h.prior[[ell]] = (1-alpha)*h.prior[[ell]] + alpha*omega
            reservoir[[ell]][,t] = as.numeric(h.prior[[ell]])
          } 
          
          h.forc.prior[[ell]] = reservoir[[ell]][,cap.t]
          WIN.x.out.product = Matrix::tcrossprod(WIN[[ell]], output.data[[ell]])
          for(fut in 1:future)
          {
            omega.hat = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.forc.prior[[ell]] + WIN.x.out.product[,fut]))
            h.forc.prior[[ell]] = (1-alpha)*h.forc.prior[[ell]] + alpha*omega.hat
            forc.reservoir[[ell]][,fut] = as.numeric(h.forc.prior[[ell]])
          } 
          
          
          #Dimension reduction to combine layers
          if(layers > 1)
          {
            placeholder = wql::eof(cbind(reservoir[[ell]], forc.reservoir[[ell]]), n = reduced.units, scale. = FALSE) 
            mean.pca = apply(placeholder$REOF[1:cap.t,], 2, mean)
            sd.pca = apply(placeholder$REOF[1:cap.t,], 2, sd)
            placeholder$REOF = (placeholder$REOF - matrix(mean.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)) / 
              matrix(sd.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)
            input.data[[ell+1]] = cbind(rep(1,cap.t), placeholder$REOF[1:cap.t,1:reduced.units])
            output.data[[ell+1]] = cbind(rep(1,future), placeholder$REOF[(cap.t+1):(cap.t+future),1:reduced.units])
          } else {
            input.data[[ell+1]] = NULL
            output.data[[ell+1]] = NULL
          }
          
        } 
        
        
        ###################################
        ### Estimate Coefficient Matrix ###
        ###################################
        
        #Get dimension reduced data on same scale
        if(layers > 1)
        {
          h.tild = matrix(NaN, nrow = cap.t, ncol = 1)
          for(ell in 2:layers)
          {
            h.tild = cbind(h.tild, g.h(input.data[[ell]][,-1]))
          }
          h.tild = h.tild[,-1]
        }
        
        #Estimate coefficients
        if(layers > 1)
        {
          final.design = rbind(reservoir[[layers]], t(h.tild))
        } else {
          final.design = rbind(reservoir[[layers]])
        }
        
        ridgeMat = lambda.r * Ident.Mat
        V = t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)
        
        
        ###########################
        ### Calculate Forecasts ###
        ###########################
        
        #Get dimension reduced data on same scale
        if(layers > 1)
        {
          h.tild.out = matrix(NaN, nrow = future, ncol = 1)
          for(ell in 2:layers)
          {
            h.tild.out = cbind(h.tild.out, g.h(output.data[[ell]][,-1]))
          }
          h.tild.out = h.tild.out[,-1]
        }
        
        #Create output design matrix
        if(layers > 1)
        {
          final.design.out = rbind(forc.reservoir[[layers]], t(h.tild.out))
        } else {
          final.design.out = rbind(forc.reservoir[[layers]])
        }
        
        
        #Generate forecasts
        if(logNorm)
        {
          exp((scale.factor * (V %*% final.design.out)) + scale.matrix)
        } else {
          (scale.factor * (V %*% final.design.out)) + scale.matrix
        }
      } 
   
     
  } else {
    
    
    
    #Begin non-parallel iterations
    for(k in 1:iter)
      {
        set.seed(NULL)
        
        ##########################################
        ### Generate W and WIN weight matrices ###
        ##########################################
        W = list()
        WIN = list()
        lambda.w = c()
        for(ell in 1:layers)
        {
          #Set sparsity
          gam.w = purrr::rbernoulli(samp.w[[ell]], p = pi.w[ell])
          gam.win = purrr::rbernoulli(samp.win[[ell]], p = pi.win[ell])
          
          #Generate W
          if(distribution == 'Unif')
          {
            unif.w = runif(samp.w[[ell]], min = -eta.w[ell], max = eta.w[ell])
            W[[ell]] = Matrix::Matrix((gam.w == 1)*unif.w + (gam.w == 0)*0,
                                      nrow = n.h[ell], ncol = n.h[ell], sparse = T)
          } else if(distribution == 'Normal')
          {
            norm.w = rnorm(samp.w[[ell]], 0, 1)
            W[[ell]] = Matrix::Matrix((gam.w == 1)*norm.w + (gam.w == 0)*0,
                                      nrow = n.h[ell], ncol = n.h[ell], sparse = T)
          }
          
          #Generate W^in
          n.input = c(n.x, rep(reduced.units+1, (layers-1)))
          if(distribution == 'Unif')
          {
            unif.win = runif(samp.win[[ell]], min = -eta.win[ell], max = eta.win[ell])
            WIN[[ell]] = Matrix::Matrix((gam.win == 1)*unif.win + (gam.win == 0)*0,
                                        nrow = n.h[ell], ncol = n.input[ell], sparse = T)
          } else if(distribution == 'Normal')
          {
            norm.win = rnorm(samp.win[[ell]], 0, 1)
            WIN[[ell]] = Matrix::Matrix((gam.win == 1)*norm.win + (gam.win == 0)*0,
                                        nrow = n.h[ell], ncol = n.input[ell], sparse = T)
          }
          
          #Specify spectral radius
          lambda.w[ell] = max(abs(eigen(W[[ell]])$values))
          
        }
        
        
        ###############################
        ### Initialize Hidden Units ###
        ###############################
        h.prior = list()
        reservoir = list()
        h.forc.prior = list()
        forc.reservoir = list()
        Ident.Mat = diag(((layers-1)*reduced.units) + n.h[layers])
        
        for(ell in 1:layers)
        {
          h.prior[[ell]] = startvalues[[ell]]
          reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = cap.t)
          h.forc.prior[[ell]] = rep(0, n.h[ell])
          forc.reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = future)
        }
        
        
        ####################################
        ### Update Training Hidden Units ###
        ####################################
        input.data = list()
        input.data[[1]] = x.insamp
        output.data = list()
        output.data[[1]] = x.outsamp
        for(ell in 1:layers)
        {
          WIN.x.in.product = Matrix::tcrossprod(WIN[[ell]], input.data[[ell]])
          for(t in 1:cap.t)
          {
            omega = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.prior[[ell]] + WIN.x.in.product[,t]))
            h.prior[[ell]] = (1-alpha)*h.prior[[ell]] + alpha*omega
            reservoir[[ell]][,t] = as.numeric(h.prior[[ell]])
          } 
          
          h.forc.prior[[ell]] = reservoir[[ell]][,cap.t]
          WIN.x.out.product = Matrix::tcrossprod(WIN[[ell]], output.data[[ell]])
          for(fut in 1:future)
          {
            omega.hat = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.forc.prior[[ell]] + WIN.x.out.product[,fut]))
            h.forc.prior[[ell]] = (1-alpha)*h.forc.prior[[ell]] + alpha*omega.hat
            forc.reservoir[[ell]][,fut] = as.numeric(h.forc.prior[[ell]])
          } 
          
          
          #Dimension reduction to combine layers
          if(layers > 1)
          {
            placeholder = wql::eof(cbind(reservoir[[ell]], forc.reservoir[[ell]]), n = reduced.units, scale. = FALSE) 
            mean.pca = apply(placeholder$REOF[1:cap.t,], 2, mean)
            sd.pca = apply(placeholder$REOF[1:cap.t,], 2, sd)
            placeholder$REOF = (placeholder$REOF - matrix(mean.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)) / 
              matrix(sd.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)
            input.data[[ell+1]] = cbind(rep(1, cap.t), placeholder$REOF[1:cap.t,1:reduced.units])
            output.data[[ell+1]] = cbind(rep(1,future), placeholder$REOF[(cap.t+1):(cap.t+future),1:reduced.units])
          } else {
            input.data[[ell+1]] = NULL
            output.data[[ell+1]] = NULL
          }
          
        } 
        
        
        ###################################
        ### Estimate Coefficient Matrix ###
        ###################################
        
        #Get dimension reduced data on same scale
        if(layers > 1)
        {
          h.tild = matrix(NaN, nrow = cap.t, ncol = 1)
          for(ell in 2:layers)
          {
            h.tild = cbind(h.tild, g.h(input.data[[ell]][,-1]))
          }
          h.tild = h.tild[,-1]
        }
        #Estimate coefficients
        if(layers > 1)
        {
          final.design = rbind(reservoir[[layers]], t(h.tild))
        } else {
          final.design = rbind(reservoir[[layers]])
          
        }
        ridgeMat = lambda.r * Ident.Mat
        V = t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)
        
        
        
        ###########################
        ### Calculate Forecasts ###
        ###########################
        
        #Get dimension reduced data on same scale
        if(layers > 1)
        {
          h.tild.out = matrix(NaN, nrow = future, ncol = 1)
          for(ell in 2:layers)
          {
            h.tild.out = cbind(h.tild.out, g.h(output.data[[ell]][,-1]))
          }
          h.tild.out = h.tild.out[,-1]
        }
        #Create output design matrix
        if(layers > 1)
        {
          final.design.out = rbind(forc.reservoir[[layers]], t(h.tild.out))
        } else {
          final.design.out = rbind(forc.reservoir[[layers]])
        }
        
        
        #Generate forecasts
        if(logNorm)
        {
          ensemb.mat[,,k] = exp((scale.factor * (V %*% final.design.out)) + scale.matrix)
        } else {
          ensemb.mat[,,k] = (scale.factor * (V %*% final.design.out)) + scale.matrix
        }
        
        #update progress bare
        if(verbose)
        {
          setTxtProgressBar(prog.bar, k)
        }
        
      } 
    
  }
  
  
  ########################
  ### Finalize Results ###
  ########################
  
  #Close parallel clusters
  if(!parallel)
  {
    if(verbose)
    {
      close(prog.bar)
    }
  } else if(parallel) {
    parallel::stopCluster(cl)
  }
  
  #Calculate forecast mean
  if(!parallel)
  {
    if(testLen > 1)
    {
      forc.mean = sapply(1:locations, function(n) rowMeans(ensemb.mat[n,,]))
    } else if(locations > 1) {
      forc.mean = apply(ensemb.mat[,1,], 1, mean)
    } else {
      forc.mean = mean(ensemb.mat[1,1,])
    }
  } else if(parallel) {
    if(locations > 1 & future == 1)
    {
      forc.mean = apply(ensemb.mat, 1, mean)
    } else if(locations == 1 & future > 1){
      forc.mean = (sapply(1:future, function(x) mean(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
    } else if(locations > 1 & future > 1) {
      forc.mean = t(sapply(1:future, function(x) rowMeans(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
    } else if(locations == 1 & future == 1) {
      forc.mean = mean(as.numeric(ensemb.mat))
    } else {
      forc.mean = NULL
    }
  }
  
  #Calculate MSE
  if(!is.null(y.test))
  {
    MSE=sum((y.test-forc.mean)^2)/(locations*future)
  } else {
    MSE = NULL
  }
  
  #Compile results
  esn.output = list('predictions' = ensemb.mat,
                    'forecastmean' = forc.mean,
                    'MSE' = MSE)
  return(esn.output)
}






#############################
### Physics Informed DESN ###
#############################


deep.esn.physics = function(y.train,
                            x.insamp,
                            x.outsamp,
                            y.test = NULL,
                            n.h,
                            nu,
                            pi.w, 
                            pi.win,
                            eta.w,
                            eta.win,
                            lambda.r,
                            alpha,
                            m,
                            iter,
                            future,
                            layers = 3,
                            reduced.units,
                            startvalues = NULL,
                            activation = 'tanh',
                            distribution = 'Normal',
                            physics = FALSE,
                            PDE = NULL,
                            PDE_params = list(),
                            time_vals = NULL,
                            location_vals = NULL,
                            stepmax = 50000,
                            subsample = 0,
                            rho = 0.5,
                            logNorm = FALSE,
                            scale.factor,
                            scale.matrix,
                            fork = F)
{
  
  ###########################################
  ### Initial Conditions and Known Values ###
  ###########################################
  
  #Set training length and locations
  cap.t = dim(y.train)[1]
  locations = dim(y.train)[2]
  if(is.null(locations) | is.null(cap.t))
  {
    cap.t = length(y.train)
    locations = 1
  }
  
  #Get number of samples for weight matrices
  samp.w = list()
  samp.win = list()
  for(ell in 1:layers)
  {
    samp.w[[ell]] = n.h[ell] * n.h[ell]
    if(ell == 1)
    {
      n.x = (locations * (m+1)) + 1 #dim(x.insamp)[2]
      samp.win[[1]] = n.h[ell] * n.x
    } else {
      samp.win[[ell]] = n.h[ell] * (reduced.units+1)
    }
  }
  
  #Starting values of hidden units
  if(is.null(startvalues))
  {
    startvalues = list()
    for(ell in 1:layers)
    {
      startvalues[[ell]] = rep(0, n.h[ell])
    }
  }
  
  #Set the activation function
  if(activation == 'identity')
  {
    g.h = function(x)
    {
      return(x)
    } 
  } else if(activation == 'tanh') {
    g.h = function(x)
    {
      placeholder = tanh(x)
      return(placeholder)
    } 
  }
  
  
  #########################
  ### Forecast Ensemble ###
  #########################
  set.seed(NULL)
  
  
  #Specify Parallel clusters
  if(fork)
  {
    cl = parallel::makeForkCluster(getOption('cores')) 
  } else if(!fork)
  {
    cl = parallel::makeCluster(getOption('cores'))
  }
  
  #Activate clusters
  doParallel::registerDoParallel(cl)
  
  #Begin parallel iterations
  ensemb.mat = foreach::foreach(k = 1:iter,
                                .combine = abind,
                                .inorder = FALSE) %dopar%
    { #begin parallel execution
      set.seed(k)
      
      ##########################################
      ### Generate W and WIN weight matrices ###
      ##########################################
      W = list()
      WIN = list()
      lambda.w = c()
      for(ell in 1:layers)
      {
        #Set sparsity
        gam.w = purrr::rbernoulli(samp.w[[ell]], p = pi.w[ell])
        gam.win = purrr::rbernoulli(samp.win[[ell]], p = pi.win[ell])
        
        #Generate W
        if(distribution == 'Unif')
        {
          unif.w = runif(samp.w[[ell]], min = -eta.w[ell], max = eta.w[ell])
          W[[ell]] = Matrix::Matrix((gam.w == 1)*unif.w + (gam.w == 0)*0,
                                    nrow = n.h[ell], ncol = n.h[ell], sparse = T)
        } else if(distribution == 'Normal')
        {
          norm.w = rnorm(samp.w[[ell]], 0, 1)
          W[[ell]] = Matrix::Matrix((gam.w == 1)*norm.w + (gam.w == 0)*0,
                                    nrow = n.h[ell], ncol = n.h[ell], sparse = T)
        }
        
        #Generate W^in
        n.input = c(n.x, rep(reduced.units+1, (layers-1)))
        if(distribution == 'Unif')
        {
          unif.win = runif(samp.win[[ell]], min = -eta.win[ell], max = eta.win[ell])
          WIN[[ell]] = Matrix::Matrix((gam.win == 1)*unif.win + (gam.win == 0)*0,
                                      nrow = n.h[ell], ncol = n.input[ell], sparse = T)
        } else if(distribution == 'Normal')
        {
          norm.win = rnorm(samp.win[[ell]], 0, 1)
          WIN[[ell]] = Matrix::Matrix((gam.win == 1)*norm.win + (gam.win == 0)*0,
                                      nrow = n.h[ell], ncol = n.input[ell], sparse = T)
        }
        
        #Specify spectral radius
        lambda.w[ell] = max(abs(eigen(W[[ell]])$values))
        
      } #end layers loop for obtaining hidden weights
      
      
      ###############################
      ### Initialize Hidden Units ###
      ###############################
      h.prior = list()
      reservoir = list()
      h.forc.prior = list()
      forc.reservoir = list()
      Ident.Mat = diag(((layers-1)*reduced.units) + n.h[layers])
      
      for(ell in 1:layers)
      {
        h.prior[[ell]] = startvalues[[ell]]
        reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = cap.t)
        h.forc.prior[[ell]] = rep(0, n.h[ell])
        forc.reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = future)
      }
      
      
      ####################################
      ### Update Training Hidden Units ###
      ####################################
      input.data = list()
      input.data[[1]] = x.insamp
      output.data = list()
      output.data[[1]] = x.outsamp
      for(ell in 1:layers)
      {
        WIN.x.in.product = Matrix::tcrossprod(WIN[[ell]], input.data[[ell]])
        for(t in 1:cap.t)
        {
          omega = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.prior[[ell]] + WIN.x.in.product[,t]))
          h.prior[[ell]] = (1-alpha)*h.prior[[ell]] + alpha*omega
          reservoir[[ell]][,t] = as.numeric(h.prior[[ell]])
        } 
        
        h.forc.prior[[ell]] = reservoir[[ell]][,cap.t]
        WIN.x.out.product = Matrix::tcrossprod(WIN[[ell]], output.data[[ell]])
        for(fut in 1:future)
        {
          omega.hat = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.forc.prior[[ell]] + WIN.x.out.product[,fut]))
          h.forc.prior[[ell]] = (1-alpha)*h.forc.prior[[ell]] + alpha*omega.hat
          forc.reservoir[[ell]][,fut] = as.numeric(h.forc.prior[[ell]])
        } 
        
        
        #Dimension reduction to combine layers
        if(layers > 1)
        {
          placeholder = wql::eof(cbind(reservoir[[ell]], forc.reservoir[[ell]]), n = reduced.units, scale. = FALSE) 
          mean.pca = apply(placeholder$REOF[1:cap.t,], 2, mean)
          sd.pca = apply(placeholder$REOF[1:cap.t,], 2, sd)
          placeholder$REOF = (placeholder$REOF - matrix(mean.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)) / 
            matrix(sd.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)
          input.data[[ell+1]] = cbind(rep(1,cap.t), placeholder$REOF[1:cap.t,1:reduced.units])
          output.data[[ell+1]] = cbind(rep(1,future), placeholder$REOF[(cap.t+1):(cap.t+future),1:reduced.units])
        } else {
          input.data[[ell+1]] = NULL
          output.data[[ell+1]] = NULL
        }
        
      } #end layers loop for updating the hidden nodes
      
      
      ###################################
      ### Estimate Coefficient Matrix ###
      ###################################
      
      #Get dimension reduced data on same scale
      if(layers > 1)
      {
        h.tild = matrix(NaN, nrow = cap.t, ncol = 1)
        for(ell in 2:layers)
        {
          h.tild = cbind(h.tild, g.h(input.data[[ell]][,-1]))
        }
        h.tild = h.tild[,-1]
      }
      
      #Compile design matrix for betas estimates
      if(layers > 1)
      {
        final.design = rbind(reservoir[[layers]], t(h.tild))
      } else {
        final.design = rbind(reservoir[[layers]])
      }
      
      
      
      #Calculate V (Beta) matrix
      if(!physics)
      {
        #Ridge Regression Estimate
        ridgeMat = lambda.r * Ident.Mat
        V = t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)
      } else {
        
        #Initialize V Matrix as Ridge estimates
        ridgeMat = lambda.r * Ident.Mat
        V = t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)
        
        #using R optim() function
        phold = as.numeric(V)
        df = list()
        df[[1]] = final.design
        df[[2]] = y.train
        
        set.seed(1)
        indices = sample(1:length(phold), round(length(phold)*subsample,0), replace = FALSE)
        
        fixed = phold[-indices]
        par = phold[indices]
        
        loss = function(data, par, fixed, indices, rho_weight)
        {
          
          temp = rep(0, length(c(par, fixed)))
          temp[indices] = par
          temp[-indices] = fixed
          
          V = matrix(temp, nrow = dim(data[[2]])[2], ncol = length(temp)/dim(data[[2]])[2])
          forcs = V %*% data[[1]]
          forcs_bar = apply(forcs, 1, mean)
          
          #Calculate of physics
          nu = PDE_params[[1]]
          u_tau = PDE_params[[2]]
          delta = PDE_params[[3]]
          kappa = PDE_params[[4]]
          f_physics_forcs = PDE(forcs_bar, location_vals, nu, u_tau, delta, kappa)
          
          
          #calulcate individual losses
          lu = apply((data[[2]] - t(forcs))^2, 2, mean)
          lf = f_physics_forcs^2
          
          #scale the losses
          min_u = min(lu)
          max_u = max(lu)
          min_f = min(lf)
          max_f = max(lf)
          
          scaled_u = (lu - min_u)/(max_u - min_u)
          scaled_f = (lf - min_f)/(max_f - min_f)
          
          scaled_loss = (1-rho_weight)*scaled_u + rho_weight*scaled_f
          
          return(mean(scaled_loss))
        }
        
        
        optimizer = optim(par = par, fn = loss, data = df, fixed = fixed, indices = indices, rho_weight = rho,
                          control = list('maxit' = stepmax))
        
        parameters = optimizer$par
        
        new_V = rep(0, length(phold))
        new_V[indices] = parameters
        new_V[-indices] = fixed
        
        V = matrix(new_V,
                   nrow = dim(df[[2]])[2],
                   ncol = dim(df[[1]])[1])
          
        
        
      } #end physics if statement
      
      
      
      
      
      
      
      
      
      
      ###########################
      ### Calculate Forecasts ###
      ###########################
      
      #Get dimension reduced data on same scale
      if(layers > 1)
      {
        h.tild.out = matrix(NaN, nrow = future, ncol = 1)
        for(ell in 2:layers)
        {
          h.tild.out = cbind(h.tild.out, g.h(output.data[[ell]][,-1]))
        }
        h.tild.out = h.tild.out[,-1]
      }
      
      #Create output design matrix
      if(layers > 1)
      {
        final.design.out = rbind(forc.reservoir[[layers]], t(h.tild.out))
      } else {
        final.design.out = rbind(forc.reservoir[[layers]])
      }
      
      
      #Generate forecasts
      if(logNorm)
      {
        exp((scale.factor * (V %*% final.design.out)) + scale.matrix)
      } else {
        (scale.factor * (V %*% final.design.out)) + scale.matrix
      }
      
    } #end parallel execution loop
  
  
  
  
  
  ########################
  ### Finalize Results ###
  ########################
  
  #Close parallel clusters
  parallel::stopCluster(cl)
  
  #Calculate forecast mean
  if(locations > 1 & future == 1)
  {
    forc.mean = apply(ensemb.mat, 1, mean)
  } else if(locations == 1 & future > 1){
    forc.mean = (sapply(1:future, function(x) mean(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
  } else if(locations > 1 & future > 1) {
    forc.mean = t(sapply(1:future, function(x) rowMeans(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
  } else if(locations == 1 & future == 1) {
    forc.mean = mean(as.numeric(ensemb.mat))
  } else {
    forc.mean = NULL
  }
  
  
  #Calculate MSE
  if(!is.null(y.test))
  {
    MSE=sum((y.test-forc.mean)^2)/(locations*future)
  } else {
    MSE = NULL
  }
  
  #Compile results
  esn.output = list('predictions' = ensemb.mat,
                    'forecastmean' = forc.mean,
                    'MSE' = MSE)
  return(esn.output)
  
  
}







######################################
### Generate Spike Trains Function ###
######################################


gen.spike.trains = function(data, timescale)
{
  #Generate Spike Trains via Poisson Process
  time = dim(data)[1]
  n.x = dim(data)[2]
  scaled.data = (data - min(data))/(max(data) - min(data))
  trains = array(NaN, dim = c(n.x, timescale, time))
  for(t in 1:time)
  {
    for(j in 1:n.x)
    {
      rand = runif(timescale)
      for(i in 1:timescale)
      {
        if(rand[i] < scaled.data[t,j]) 
        {
          trains[j,i,t] = 1
        } else {
          trains[j,i,t] = 0
        }
      }
    }
  }
  
  
  return(trains)
}



###############################
### DESN & DLSM Combo Model ###
###############################


combo_spiking = function(y.train,
                         x.insamp,
                         x.outsamp,
                         y.test = NULL,
                         n.h,
                         nu,
                         pi.w, 
                         pi.win,
                         eta.w,
                         eta.win,
                         lambda.r,
                         alpha,
                         m,
                         timescale = 100,
                         threshold,
                         latency,
                         leakage,
                         resting = 0,
                         inhibitor,
                         iter,
                         future,
                         layers = 3,
                         reduced.units,
                         startvalues = NULL,
                         activation = 'tanh',
                         distribution = 'Normal',
                         physics = FALSE,
                         PDE = NULL,
                         PDE_params = list(),
                         time_vals = NULL,
                         location_vals = NULL,
                         stepmax = 50000,
                         subsample = 0,
                         rho = 0.5,
                         logNorm = FALSE,
                         scale.factor,
                         scale.matrix,
                         fork = F)
{
  
  ###########################################
  ### Initial Conditions and Known Values ###
  ###########################################
  
  #Set training length and locations
  cap.t = dim(y.train)[1]
  locations = dim(y.train)[2]
  if(is.null(locations) | is.null(cap.t))
  {
    cap.t = length(y.train)
    locations = 1
  }
  
  #Get number of samples for weight matrices
  samp.w = list()
  samp.win = list()
  for(ell in 1:layers)
  {
    samp.w[[ell]] = n.h[ell] * n.h[ell]
    if(ell == 1)
    {
      n.x = (locations * (m+1)) + 1
      samp.win[[1]] = n.h[ell] * n.x
    } else {
      samp.win[[ell]] = n.h[ell] * (reduced.units+1)
    }
  }
  
  #Starting values of hidden units
  if(is.null(startvalues))
  {
    startvalues = list()
    for(ell in 1:layers)
    {
      startvalues[[ell]] = rep(0, n.h[ell])
    }
  }
  
  #Set the activation function
  if(activation == 'identity')
  {
    g.h = function(x)
    {
      return(x)
    } 
  } else if(activation == 'tanh') {
    g.h = function(x)
    {
      placeholder = tanh(x)
      return(placeholder)
    } 
  }
  
  #########################
  ### Forecast Ensemble ###
  #########################
  set.seed(NULL)
  
  
  #Specify Parallel clusters
  if(fork)
  {
    cl = parallel::makeForkCluster(getOption('cores')) 
  } else if(!fork)
  {
    cl = parallel::makeCluster(getOption('cores'))
  }
  
  #Activate clusters
  doParallel::registerDoParallel(cl)
  
  #Begin parallel iterations
  ensemb.mat = foreach::foreach(k = 1:iter,
                                .combine = abind,
                                .inorder = FALSE) %dopar%
    {
      set.seed(NULL)
      set.seed(k)
      
      ##########################################
      ### Generate W and WIN weight matrices ###
      ##########################################
      W = list()
      WIN = list()
      lambda.w = c()
      for(ell in 1:layers)
      {
        #Set sparsity
        gam.w = purrr::rbernoulli(samp.w[[ell]], p = pi.w[ell])
        gam.win = purrr::rbernoulli(samp.win[[ell]], p = pi.win[ell])
        
        #Generate W
        if(distribution == 'Unif')
        {
          unif.w = runif(samp.w[[ell]], min = -eta.w[ell], max = eta.w[ell])
          W[[ell]] = Matrix::Matrix((gam.w == 1)*unif.w + (gam.w == 0)*0,
                                    nrow = n.h[ell], ncol = n.h[ell], sparse = T)
        } else if(distribution == 'Normal')
        {
          norm.w = rnorm(samp.w[[ell]], 0, 1)
          W[[ell]] = Matrix::Matrix((gam.w == 1)*norm.w + (gam.w == 0)*0,
                                    nrow = n.h[ell], ncol = n.h[ell], sparse = T)
        }
        
        #Generate W^in
        n.input = c(n.x, rep(reduced.units+1, (layers-1)))
        if(distribution == 'Unif')
        {
          unif.win = runif(samp.win[[ell]], min = -eta.win[ell], max = eta.win[ell])
          WIN[[ell]] = Matrix::Matrix((gam.win == 1)*unif.win + (gam.win == 0)*0,
                                      nrow = n.h[ell], ncol = n.input[ell], sparse = T)
        } else if(distribution == 'Normal')
        {
          norm.win = rnorm(samp.win[[ell]], 0, 1)
          WIN[[ell]] = Matrix::Matrix((gam.win == 1)*norm.win + (gam.win == 0)*0,
                                      nrow = n.h[ell], ncol = n.input[ell], sparse = T)
        }
        
        #Specify spectral radius
        lambda.w[ell] = max(abs(eigen(W[[ell]])$values))
        
      }
      
      
      ###############################
      ### Initialize Hidden Units ###
      ###############################
      h.prior = list()
      reservoir = list()
      reservoir.spiking = list()
      h.forc.prior = list()
      h.forc.prior.spiking = list()
      forc.reservoir = list()
      forc.reservoir.spiking = list()
      Ident.Mat = diag((((layers-1)*reduced.units) + n.h[layers])*2)
      
      for(ell in 1:layers)
      {
        h.prior[[ell]] = startvalues[[ell]]
        reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = cap.t)
        reservoir.spiking[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = cap.t)
        h.forc.prior[[ell]] = rep(0, n.h[ell])
        h.forc.prior.spiking[[ell]] = rep(0, n.h[ell])
        forc.reservoir[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = future)
        forc.reservoir.spiking[[ell]] = matrix(NaN, nrow = n.h[ell], ncol = future)
      }
      
      
      ####################################
      ### Update Training Hidden Units ###
      ####################################
      input.data = list()
      input.data[[1]] = x.insamp
      output.data = list()
      output.data[[1]] = x.outsamp
      input.data.spiking = list()
      input.data.spiking[[1]] = x.insamp
      output.data.spiking = list()
      output.data.spiking[[1]] = x.outsamp
      for(ell in 1:layers)
      {
        reservoir.prior = rep(0, n.h[ell])
        #Generate Spike Trains via Poisson Process
        in.time = dim(input.data.spiking[[ell]])[1]
        in.n.x = dim(input.data.spiking[[ell]])[2]
        scaled.data = (input.data.spiking[[ell]] - min(input.data.spiking[[ell]]))/(max(input.data.spiking[[ell]]) - min(input.data.spiking[[ell]]))
        trains = array(NaN, dim = c(in.n.x, timescale, in.time))
        for(t in 1:in.time)
        {
          for(j in 1:in.n.x)
          {
            rand = runif(timescale)
            for(i in 1:timescale)
            {
              if(rand[i] < scaled.data[t,j]) 
              {
                trains[j,i,t] = 1
              } else {
                trains[j,i,t] = 0
              }
            }
          }
        }
        
        
        #loop through data in time
        input.trains = trains
        WIN.x.in.product = Matrix::tcrossprod(WIN[[ell]], input.data[[ell]])
        for(t in 1:cap.t)
        {
          #DESN
          omega = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.prior[[ell]] + WIN.x.in.product[,t]))
          h.prior[[ell]] = (1-alpha)*h.prior[[ell]] + alpha*omega
          reservoir[[ell]][,t] = as.numeric(h.prior[[ell]])
          
          
          #DLSM
          omega.spiking = matrix(NaN, nrow = n.h[ell], ncol = timescale)
          omega.prior.spiking = startvalues[[ell]]
          WIN.x.in.product.spiking = Matrix::Matrix(abs(WIN[[ell]]) %*% input.trains[,,t], sparse = T)
          
          spike.vec = matrix(0, nrow = n.h[ell], ncol = timescale)
          spike.vec.prior = rep(0, n.h[ell])
          for(t.star in 1:timescale)
          {
            omega.spiking[,t.star] = as.vector((as.matrix((nu[ell]/abs(lambda.w[ell])) * abs(W[[ell]]) %*% spike.vec.prior + WIN.x.in.product.spiking[,t.star] + omega.prior.spiking)))
            
            #enforce latency
            if(latency > 0)
            {
              for(d in 1:latency)
              {
                if(t.star-d > 0)
                {
                  flag <- spike.vec[,t.star-d] == 1
                  if(sum(flag) != 0)
                  {
                    omega.spiking[flag,t.star] = resting
                  }
                }
              }
            }
            
            #enforce lateral inhibition
            max.spike = which.max(omega.spiking[,t.star])
            max.spike.thresh = omega.spiking[max.spike, t.star] > threshold
            if(max.spike.thresh)
            {
              omega.spiking[-max.spike, t.star] = omega.spiking[-max.spike, t.star] - inhibitor
            }
            
            #update potential energy and output spikes
            flag.thresh <- omega.spiking[,t.star] > threshold
            if(sum(flag.thresh) != 0)
            {
              spike.vec[flag.thresh,t.star] = 1
              omega.prior.spiking[flag.thresh] = resting
            }
            
            if(sum(!flag.thresh) != 0)
            {
              spike.vec[!flag.thresh,t.star] = 0
            }
            
            #enforce leakage
            flag.omega <- omega.spiking[,t.star] != resting
            if(sum(flag.omega) != 0)
            {
              omega.prior.spiking[flag.omega] = omega.spiking[flag.omega,t.star] - leakage
            }
            
            if(sum(!flag.omega) != 0)
            {
              omega.prior.spiking[!flag.omega] = resting
            }
            
            spike.vec.prior = spike.vec[,t.star]
          } #end of DLSM loop
          
          reservoir.prior = (1-alpha)*reservoir.prior + alpha*rowSums(spike.vec)
          reservoir.spiking[[ell]][,t] = as.numeric(reservoir.prior)
          
          
          
        } #end time loop
        
        
        h.forc.prior.spiking[[ell]] = reservoir.spiking[[ell]][,cap.t] #DSESN
        
        out.time = dim(output.data.spiking[[ell]])[1]
        out.n.x = dim(output.data.spiking[[ell]])[2]
        scaled.data = (output.data.spiking[[ell]] - min(output.data.spiking[[ell]]))/(max(output.data.spiking[[ell]]) - min(output.data.spiking[[ell]]))
        out.trains = array(NaN, dim = c(out.n.x, timescale, out.time))
        for(t in 1:out.time)
        {
          for(j in 1:out.n.x)
          {
            rand = runif(timescale)
            for(i in 1:timescale)
            {
              if(rand[i] < scaled.data[t,j]) 
              {
                out.trains[j,i,t] = 1
              } else {
                out.trains[j,i,t] = 0
              }
            }
          }
        }
        
        #loop through future time points
        output.trains = out.trains
        h.forc.prior[[ell]] = reservoir[[ell]][,cap.t] 
        WIN.x.out.product = Matrix::tcrossprod(WIN[[ell]], output.data[[ell]])
        for(fut in 1:future)
        {
          #DESN
          omega.hat = g.h(as.matrix((nu[ell]/lambda.w[ell]) * W[[ell]] %*% h.forc.prior[[ell]] + WIN.x.out.product[,fut]))
          h.forc.prior[[ell]] = (1-alpha)*h.forc.prior[[ell]] + alpha*omega.hat
          forc.reservoir[[ell]][,fut] = as.numeric(h.forc.prior[[ell]])
          
          
          #DLSM
          omega.spiking = matrix(NaN, nrow = n.h[ell], ncol = timescale)
          omega.prior.spiking = startvalues[[ell]]
          WIN.x.out.product.spiking = Matrix::Matrix(abs(WIN[[ell]]) %*% output.trains[,,fut], sparse = T)
          
          spike.vec = matrix(0, nrow = n.h[ell], ncol = timescale)
          spike.vec.prior = rep(0, n.h[ell])
          for(t.star in 1:timescale)
          {
            omega.spiking[,t.star] = as.vector((nu[ell]/abs(lambda.w[ell])) * abs(W[[ell]]) %*% spike.vec.prior + WIN.x.out.product.spiking[,t.star] + omega.prior.spiking)
            
            
            #enforce latency
            if(latency > 0)
            {
              for(d in 1:latency)
              {
                if(t.star-d > 0)
                {
                  flag <- spike.vec[,t.star-d] == 1
                  if(sum(flag) != 0)
                  {
                    omega.spiking[flag,t.star] = resting
                  }
                }
              }
            }
            
            #enforce lateral inhibition
            max.spike = which.max(omega.spiking[,t.star])
            max.spike.thresh = omega.spiking[max.spike, t.star] > threshold
            if(max.spike.thresh)
            {
              omega.spiking[-max.spike, t.star] = omega.spiking[-max.spike, t.star] - inhibitor
            }
            
            
            #update potential energy and output spikes
            flag.thresh <- omega.spiking[,t.star] > threshold
            if(sum(flag.thresh) != 0)
            {
              spike.vec[flag.thresh,t.star] = 1
              omega.prior.spiking[flag.thresh] = resting
            }
            
            if(sum(!flag.thresh) != 0)
            {
              spike.vec[!flag.thresh,t.star] = 0
            }
            
            #enforce leakage
            flag.omega <- omega.spiking[,t.star] != resting
            if(sum(flag.omega) != 0)
            {
              omega.prior.spiking[flag.omega] = omega.spiking[flag.omega,t.star] - leakage
            }
            
            if(sum(!flag.omega) != 0)
            {
              omega.prior.spiking[!flag.omega] = resting
            }
            
            spike.vec.prior = spike.vec[,t.star]
          } #end DLSM loop
          
          h.forc.prior.spiking[[ell]] = (1-alpha)*h.forc.prior.spiking[[ell]] + alpha*rowSums(spike.vec)
          forc.reservoir.spiking[[ell]][,fut] = as.numeric(h.forc.prior.spiking[[ell]])
          
          
        } #end future loop
        
        
        
        
        #Dimension reduction to combine layers --- DESN
        if(layers > 1)
        {
          eof.dat = cbind(reservoir[[ell]], forc.reservoir[[ell]])
          constant.column = sum(apply(eof.dat, 2, var)==0)
          if(constant.column != 0)
          {
            columns = which(apply(eof.dat, 2, var)==0)
            for(clmn in 1:length(columns))
            {
              if(columns[clmn] == 1)
              {
                eof.dat[,columns[clmn]] = eof.dat[,(columns[clmn]+1)]
              } else if(columns[clmn] == (cap.t+future)){
                eof.dat[,columns[clmn]] = eof.dat[,(columns[clmn]-1)]
              } else {
                eof.dat[,columns[clmn]] = ceiling((eof.dat[,(columns[clmn]+1)]+eof.dat[,(columns[clmn]-1)])/2)
              }
            }
            
          }
          
          placeholder = wql::eof(eof.dat, n = reduced.units, scale. = FALSE) 
          mean.pca = apply(placeholder$REOF[1:cap.t,], 2, mean)
          sd.pca = apply(placeholder$REOF[1:cap.t,], 2, sd)
          placeholder$REOF = (placeholder$REOF - matrix(mean.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)) / 
            matrix(sd.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)
          input.data[[ell+1]] = cbind(rep(1,cap.t), placeholder$REOF[1:cap.t,1:reduced.units])
          output.data[[ell+1]] = cbind(as.matrix(rep(1,future), ncol = 1),
                                       matrix(placeholder$REOF[(cap.t+1):(cap.t+future),1:reduced.units], nrow = future, ncol = reduced.units))
        } else {
          input.data[[ell+1]] = NULL
          output.data[[ell+1]] = NULL
        }
        
        
        #Dimension reduction to combine layers --- DLSM
        if(layers > 1)
        {
          eof.dat = cbind(reservoir.spiking[[ell]], forc.reservoir.spiking[[ell]])
          constant.column = sum(apply(eof.dat, 2, var)==0)
          if(constant.column != 0)
          {
            columns = which(apply(eof.dat, 2, var)==0)
            for(clmn in 1:length(columns))
            {
              if(columns[clmn] == 1)
              {
                eof.dat[,columns[clmn]] = eof.dat[,(columns[clmn]+1)]
              } else if(columns[clmn] == (cap.t+future)){
                eof.dat[,columns[clmn]] = eof.dat[,(columns[clmn]-1)]
              } else {
                eof.dat[,columns[clmn]] = ceiling((eof.dat[,(columns[clmn]+1)]+eof.dat[,(columns[clmn]-1)])/2)
              }
            }
            
          }
          placeholder = wql::eof(eof.dat, n = reduced.units) 
          mean.pca = apply(placeholder$REOF[1:cap.t,], 2, mean)
          sd.pca = apply(placeholder$REOF[1:cap.t,], 2, sd)
          placeholder$REOF = (placeholder$REOF - matrix(mean.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)) / 
            matrix(sd.pca, nrow = cap.t+future, ncol = ncol(placeholder$REOF), byrow = TRUE)
          input.data.spiking[[ell+1]] = cbind(rep(1, cap.t), placeholder$REOF[1:cap.t,1:reduced.units])
          output.data.spiking[[ell+1]] = cbind(rep(1,future), placeholder$REOF[(cap.t+1):(cap.t+future),1:reduced.units])
        } else {
          input.data.spiking[[ell+1]] = NULL
          output.data.spiking[[ell+1]] = NULL
        }
        
        #print(ell)
        
      } #end layers loop
      
      
      ###################################
      ### Estimate Coefficient Matrix ###
      ###################################
      
      #Get dimension reduced data on same scale
      h.tild = matrix(NaN, nrow = cap.t, ncol = 1)
      for(ell in 2:layers)
      {
        #DESN
        h.tild = cbind(h.tild, g.h(input.data[[ell]][,-1]))
        
        #DLSM
        in.dat = input.data.spiking[[ell]][,-1]
        max.input = apply(in.dat, 1, max)
        min.input = apply(in.dat, 1, min)
        
        max.mat = matrix(max.input, nrow = cap.t, ncol = dim(in.dat)[2])
        min.mat = matrix(min.input, nrow = cap.t, ncol = dim(in.dat)[2])
        
        scaled.in = (in.dat - min.mat)/(max.mat-min.mat)
        
        h.tild = cbind(h.tild, scaled.in)
        
      }
      h.tild = h.tild[,-1]
      
      #DESN
      max.reserv = apply(reservoir[[layers]], 2, max)
      min.reserv = apply(reservoir[[layers]], 2, min)
      
      max.reserv.mat = matrix(max.reserv, nrow = n.h[layers], ncol = cap.t, byrow = T)
      min.reserv.mat = matrix(min.reserv, nrow = n.h[layers], ncol = cap.t, byrow = T)
      
      scaled.reserv = (reservoir[[layers]] - min.reserv.mat)/(max.reserv.mat - min.reserv.mat)
      
      #DLSM
      max.reserv = apply(reservoir.spiking[[layers]], 2, max)
      min.reserv = apply(reservoir.spiking[[layers]], 2, min)
      
      max.reserv.mat = matrix(max.reserv, nrow = n.h[layers], ncol = cap.t, byrow = T)
      min.reserv.mat = matrix(min.reserv, nrow = n.h[layers], ncol = cap.t, byrow = T)
      
      scaled.reserv.spiking = (reservoir.spiking[[layers]] - min.reserv.mat)/(max.reserv.mat - min.reserv.mat)
      
      
      #Estimate coefficients
      if(!physics)
      {
        #Ridge Regression
        final.design = rbind(scaled.reserv, scaled.reserv.spiking, t(h.tild))
        ridgeMat = lambda.r * Ident.Mat
        V = t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)
      } else {
        
        #Initialize V Matrix as Ridge estimates
        PIDESN_indices = c(1:n.h[layers], (2*n.h[layers]+1):(2*n.h[layers]+reduced.units))
        final.design = rbind(scaled.reserv, scaled.reserv.spiking, t(h.tild))
        ridgeMat = lambda.r * Ident.Mat
        #V = t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)
        
        #delcare loss function
        loss = function(data, par, fixed, indices, rho_weight)
        {
          
          temp = rep(0, length(c(par, fixed)))
          temp[indices] = par
          temp[-indices] = fixed
          
          V = matrix(temp, nrow = dim(data[[2]])[2], ncol = length(temp)/dim(data[[2]])[2])
          forcs = V %*% data[[1]]
          forcs_bar = apply(forcs, 1, mean)
          
          #Calculate of physics
          nu = PDE_params[[1]]
          u_tau = PDE_params[[2]]
          delta = PDE_params[[3]]
          kappa = PDE_params[[4]]
          f_physics_forcs = PDE(forcs_bar, location_vals, nu, u_tau, delta, kappa)
          
          lu = apply((data[[2]] - t(forcs))^2, 2, mean)
          min_u = min(lu)
          max_u = max(lu)
          lf = f_physics_forcs^2
          min_f = min(lf)
          max_f = max(lf)
          scaled_u = (lu - min_u)/(max_u - min_u)
          scaled_f = (lf - min_f)/(max_f - min_f)
          scaled_loss = (1-rho_weight)*scaled_u + rho_weight*scaled_f
          
          return(mean(scaled_loss))
        }
        
        
        #declare only PIDESN parameters to optimize
        V_PIDESN = t(y.train) %*%
          t(final.design[PIDESN_indices,]) %*%
          solve(Matrix::tcrossprod(final.design[PIDESN_indices,], final.design[PIDESN_indices,]) +
                  ridgeMat[PIDESN_indices, PIDESN_indices])
        phold = as.numeric(V_PIDESN)
        df = list()
        df[[1]] = final.design[PIDESN_indices,]
        df[[2]] = y.train
        
        set.seed(1)
        indices = sample(1:length(phold), round(length(phold)*subsample,0), replace = FALSE)
        fixed = phold[-indices]
        par = phold[indices]
        
        
        
        #start PIDESN optimization
        optimizer = optim(par = par, fn = loss, data = df, fixed = fixed, indices = indices, rho_weight = rho,
                          control = list('maxit' = stepmax))
        
        parameters = optimizer$par
        
        new_V_PIDESN = rep(0, length(phold))
        new_V_PIDESN[indices] = parameters
        new_V_PIDESN[-indices] = fixed
        
        V_PIDESN = matrix(new_V_PIDESN,
                          nrow = dim(df[[2]])[2],
                          ncol = length(PIDESN_indices))
        
        
        
        
        #initalize Matrix for PIDDRN to be optimized over
        V =  t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)
        V[,PIDESN_indices] = V_PIDESN
        V[,-PIDESN_indices] = 0
        
        
        
        #declare all parameters to optimize
        phold = as.numeric(V)
        df = list()
        df[[1]] = final.design
        df[[2]] = y.train
        
        set.seed(1)
        indices = sample(1:length(phold), round(length(phold)*subsample,0), replace = FALSE)
        fixed = phold[-indices]
        par = phold[indices]
        
        
        
        optimizer = optim(par = par, fn = loss, data = df, fixed = fixed, indices = indices, rho_weight = rho,
                          control = list('maxit' = stepmax))
        
        parameters = optimizer$par
        
        new_V = rep(0, length(phold))
        new_V[indices] = parameters
        new_V[-indices] = fixed
        
        V = matrix(new_V,
                   nrow = dim(df[[2]])[2],
                   ncol = 2*((layers-1)*reduced.units + n.h[layers]))
        
      } #end of V matrix estimation
      
      
      
      ###########################
      ### Calculate Forecasts ###
      ###########################
      
      #Get dimension reduced data on same scale
      h.tild.out = matrix(NaN, nrow = future, ncol = 1)
      for(ell in 2:layers)
      {
        #DESN
        h.tild.out = cbind(h.tild.out, g.h(output.data[[ell]][,-1]))
        
        #DLSM
        out.dat = output.data.spiking[[ell]][,-1]
        max.output = apply(out.dat, 1, max)
        min.output = apply(out.dat, 1, min)
        
        max.mat = matrix(max.output, nrow = future, ncol = dim(out.dat)[2])
        min.mat = matrix(min.output, nrow = future, ncol = dim(out.dat)[2])
        
        scaled.out = (out.dat - min.mat)/(max.mat-min.mat)
        
        h.tild.out = cbind(h.tild.out, scaled.out)
      }
      h.tild.out = h.tild.out[,-1]
      
      #DESN
      max.reserv = apply(forc.reservoir[[layers]], 2, max)
      min.reserv = apply(forc.reservoir[[layers]], 2, min)
      
      max.reserv.mat = matrix(max.reserv, nrow = n.h[layers], ncol = future, byrow = T)
      min.reserv.mat = matrix(min.reserv, nrow = n.h[layers], ncol = future, byrow = T)
      
      scaled.forc.reserv = (forc.reservoir[[layers]] - min.reserv.mat)/(max.reserv.mat - min.reserv.mat)
      
      #DLSM
      max.reserv = apply(forc.reservoir.spiking[[layers]], 2, max)
      min.reserv = apply(forc.reservoir.spiking[[layers]], 2, min)
      
      max.reserv.mat = matrix(max.reserv, nrow = n.h[layers], ncol = future, byrow = T)
      min.reserv.mat = matrix(min.reserv, nrow = n.h[layers], ncol = future, byrow = T)
      
      scaled.forc.reserv.spiking = (forc.reservoir.spiking[[layers]] - min.reserv.mat)/(max.reserv.mat - min.reserv.mat)
      
      #Create output design matrix
      final.design.out = rbind(scaled.forc.reserv, scaled.forc.reserv.spiking, t(h.tild.out))
      
      #Generate forecasts
      if(logNorm)
      {
        exp((scale.factor * (V %*% final.design.out)) + scale.matrix)
      } else {
        (scale.factor * (V %*% final.design.out)) + scale.matrix
      }
      
      
      
      
    } 
  
  
  
  
  
  ########################
  ### Finalize Results ###
  ########################
  
  #Close parallel clusters
  parallel::stopCluster(cl)
  
  #Calculate forecast mean
  if(locations > 1 & future == 1)
  {
    forc.mean = apply(ensemb.mat, 1, mean)
  } else if(locations == 1 & future > 1){
    forc.mean = (sapply(1:future, function(x) mean(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
  } else if(locations > 1 & future > 1) {
    forc.mean = t(sapply(1:future, function(x) rowMeans(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
  } else if(locations == 1 & future == 1) {
    forc.mean = mean(as.numeric(ensemb.mat))
  } else {
    forc.mean = NULL
  }
  
  
  #Calculate MSE
  if(!is.null(y.test))
  {
    MSE=sum((y.test-forc.mean)^2)/(locations*future)
  } else {
    MSE = NULL
  }
  
  #Compile results
  esn.output = list('predictions' = ensemb.mat,
                    'forecastmean' = forc.mean,
                    'MSE' = MSE)
  return(esn.output)
}
