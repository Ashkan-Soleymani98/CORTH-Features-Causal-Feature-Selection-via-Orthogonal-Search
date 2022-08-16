library(vctrs)
library(here)
library(mvtnorm)
library(randomForest)
library(glmnet)
library(bnlearn)
library(selectiveInference)

doubleML_parental_test <- function(z, d, y, regression_technique){
  N <- dim(z)[1] # Number of observations
  thetahat = 0
  runtime = 0
  if (regression_technique == "Random Forest"){
    start_time <- Sys.time()
    # Cross-fitting DML #
    # Split sample #
    I = sort(sample(1:N, N / 2))
    IC = setdiff(1:N, I)
    # compute ghat on both sample #
    model1 = randomForest(z[IC,], y[IC], maxnodes = 10)
    model2 = randomForest(z[I,], y[I], maxnodes = 10)
    G1 = predict(model1, z[I,])
    G2 = predict(model2, z[IC,])
    
    # Compute mhat and vhat on both samples #
    modeld1 = randomForest(z[IC,], d[IC], maxnodes = 10)
    modeld2 = randomForest(z[I,], d[I], maxnodes = 10)
    M1 = predict(modeld1, z[I,])
    M2 = predict(modeld2, z[IC,])
    V1 = d[I] - M1
    V2 = d[IC] - M2
    
  } else if (regression_technique == "Lasso"){ # Lasso with cross-validation for regularization estimator``
    start_time <- Sys.time()
    # Cross-fitting DML #
    # Split sample #
    I = sort(sample(1:N, N / 2))
    IC = setdiff(1:N, I)
    # compute ghat on both sample #
    cv.out <- cv.glmnet(z[IC,], y[IC], alpha = 1)
    best_lambda1 = cv.out$lambda.min
    cv.out <- cv.glmnet(z[I,], y[I], alpha = 1)
    best_lambda2 = cv.out$lambda.min
    model1 = glmnet(z[IC,], y[IC], alpha = 1, lambda = best_lambda1)
    model2 = glmnet(z[I,], y[I], alpha = 1, lambda = best_lambda2)
    G1 = predict(model1, s = best_lambda1, newx = z[I,])
    G2 = predict(model2, s = best_lambda2, newx = z[IC,])
    
    # Compute mhat and vhat on both samples #
    cv.out <- cv.glmnet(z[IC,], d[IC], alpha = 1)
    best_lambda1 = cv.out$lambda.min
    cv.out <- cv.glmnet(z[I,], d[I], alpha = 1)
    best_lambda2 = cv.out$lambda.min
    modeld1 = glmnet(z[IC,], d[IC], alpha = 1, lambda = best_lambda1)
    modeld2 = glmnet(z[I,], d[I], alpha = 1, lambda = best_lambda2)
    M1 = predict(modeld1, s = best_lambda1, newx = z[I,])
    M2 = predict(modeld2, s = best_lambda2, newx = z[IC,])
    V1 = d[I] - M1
    V2 = d[IC] - M2
    
  } else if (regression_technique == "Kernel Ridge Regression") {
    start_time <- Sys.time()
    # Cross-fitting DML #
    # Split sample #
    I = sort(sample(1:N, N / 2))
    IC = setdiff(1:N, I)
    # compute ghat on both sample #
    model1 = krr(z[IC,], y[IC])
    model2 = krr(z[I,], y[I])
    G1 = predict(model1, z[I,])
    G2 = predict(model2, z[IC,])
    
    # Compute mhat and vhat on both samples #
    modeld1 = krr(z[IC,], d[IC])
    modeld2 = krr(z[I,], d[I])
    M1 = predict(modeld1, z[I,])
    M2 = predict(modeld2, z[IC,])
    V1 = d[I] - M1
    V2 = d[IC] - M2
    
  }
  
  # Compute Cross-Fitting DML theta #
  theta1 = mean(V1 * (y[I] - G1)) / mean(V1 * d[I])
  theta2 = mean(V2 * (y[IC] - G2)) / mean(V2 * d[IC])
  
  theta_cf = mean(c(theta1, theta2))
  thetahat = theta_cf
  
  
  # Calculate Khi and Sigma on both samples#
  ## Indirect way ##
  # khi1 = mean(-y[I] * M1 - d[I] * G1 + M1 * G1 + y[I] * d[I])
  # khi2 = mean(-y[IC] * M2 - d[IC] * G2 + M2 * G2 + y[IC] * d[IC])
  # sigmatwo1 = mean((-y[I] * M1 - d[I] * G1 + M1 * G1 + y[I] * d[I] - khi1)^2)
  # sigmatwo2 = mean((-y[IC] * M2 - d[IC] * G2 + M2 * G2 + y[IC] * d[IC] - khi2)^2)
  
  ## Direct Way ##
  khi1 = mean((G1 - y[I]) * (d[I] - M1))
  khi2 = mean((G2 - y[IC]) * (d[IC] - M2))
  sigmatwo1 = mean(( (G1 - y[I]) * (d[I] - M1) - khi1)^2)
  sigmatwo2 = mean(( (G2 - y[IC]) * (d[IC] - M2) - khi2)^2)
  
  # Compute Cross-Fitting Khi and Sigma#
  khihat = (khi1 + khi2) / 2
  sigmatwohat = (sigmatwo1 + sigmatwo2) / 2
  
  runtime <- as.numeric(Sys.time() - start_time, units="secs")
  
  results = matrix(NA, 1, 4)
  colnames(results) <- c("Result", "Thetahat", "pValue", "Runtime")
  rownames(results) <- c("Cross-fitting DML")
  
  start_time <- Sys.time()
  results[,"Thetahat"] <- theta_cf
  results[,"pValue"] <- 2 * pnorm(abs(khihat), 0, sqrt(sigmatwohat) / sqrt(N), lower.tail = FALSE)
  results[,"Result"] <- (results[,"pValue"] < (0.1 / (dim(z)[2] + 1))) # Bonferroni Correction
  results[,"Runtime"] <- runtime + as.numeric(Sys.time() - start_time, units="secs")
  
  print(results)
  cat("\n")
  
  return(results)
}

CORTH_Features_find_parents <- function(data){
  data <- scale(as.matrix(data), TRUE, TRUE)
  
  k <- dim(data)[2]
  x <- data[,2:k-1]
  y <- data[,k]
  
  parental_state = matrix(0, 1, k)
  colnames(parental_state) <- colnames(data)
  rownames(parental_state) <- c("Result")
  regression_technique <- "Lasso"
  
  for (i in 1:(k-1)){ # one-vs-rest search
    print("-----")
    print(i)
    results <- doubleML_parental_test(x[,-i], x[,i], y, regression_technique)
    parental_state["Result", i] <- results[,"Result"]
  }
  return(parental_state)
}


data_address = "/DAG Samples/Random Structures/sparsity(0.2)n(5)observation_num(100)nonlinear_probability(0.3)a(0.5)b(1.5)theta(2)alpha(0.5)beta(0.5)/simulated_data10.csv"
data <- read.csv(paste(here(), data_address, sep=""), ,na.strings=c("", "NA") , header=TRUE, sep="\t")
shape <- dim(data)

CORTH_Features_find_parents(data.frame(as.matrix(data[,2:shape[2]])))

