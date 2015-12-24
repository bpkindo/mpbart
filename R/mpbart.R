

mpbart <- function(formula, train.data, test.data = NULL, base = NULL,
                   varying = NULL, sep=".", 
                   Prior = NULL, Mcmc = NULL, 
                   seedvalue = NULL)
{
#'Multinomial Probit Bayesian Additive Regression Trees
#'
#'Multinomial probit modeling using Bayesian Additive Regression Trees,
#'@param formula choice ~ demographic covariates. If there are no, demographic variables use y ~ 1,
#'@param train.data Training Data in wide format,
#'@param test.data Test Data in wide format, typically without the response,
#'@param base Base choic. Default is the highest class/choice,
#'@param varying the indexes of the variables that are alternative specific,
#'@param sep the seperator of the variable name and the alternative name,
#'@param Prior List of Priors for MPBART: e.g., Prior = list(nu=p+2,  V= diag(p - 1), ntrees=200,  kfac=2.0,  pbd=1.0, pb=0.5 , beta = 2.0, alpha = 0.95, nc = 100, priorindep = 0,  minobsnode = 10)
#'@param Mcmc List of MCMC starting values, burn-in ...: e.g.,     list(sigma0 = diag(p - 1), keep = 1, burn = 100, ndraws = 1000, keep_sigma_draws=FALSE)
#'@param seedvalue random seed value, default of 99 will be used if null,
#'@return class_prob_train training data choice/class probabilities,
#'@return predicted_class_train training data predicted choices/classes,
#'@return class_prob_test test data choice/class probabilities,
#'@return predicted_class_test test data predicted choices/classes,
#'@return sigmasample posterior samples of the latent variable covariance matrix.
#'@export


if(is.null(seedvalue)){
  seedvalue = 99
} else {
  set.seed(seedvalue)
}




out <- mpbart_call(formula = formula, 
                  data = train.data,
                  base = base, 
                  test.data = test.data,
                  Prior = Prior, Mcmc = Mcmc, 
                  varying = varying, sep = sep)
return(out)
}



