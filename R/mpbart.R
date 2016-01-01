

mpbart <- function(formula, train.data, test.data = NULL, base = NULL,
                   varying = NULL, sep=".", 
                   Prior = NULL, Mcmc = NULL, 
                   seedvalue = NULL)
{
#'Multinomial Probit Bayesian Additive Regression Trees
#'
#'Multinomial probit modeling using Bayesian Additive Regression Trees,
#'@param formula response ~ choice speccific covariates | demographic covariates. If there are no, demographic variables use  response ~ choice specific covariates| ~ 1. If there are no choice specific covariates, use  response ~ 1 | demographic covariates
#'@param train.data Training Data in wide format (for details on wide format, see documentation in R package \pkg{mlogit}),
#'@param test.data Test Data in wide format, typically without the response,
#'@param base base choice. Default is the highest class/choice,
#'@param varying The indeces of the variables that are alternative specific,
#'@param sep The seperator of the variable name and the alternative name in the choice specific covariates. For example a covariate name variabl1.choice1 indicates a separator of dot (.).
#'@param Prior List of Priors for MPBART: e.g., Prior = list(nu=p+2,  V= diag(p - 1), ntrees=200,  kfac=2.0,  pbd=1.0, pb=0.5 , beta = 2.0, alpha = 0.95, nc = 100, priorindep = FALSE,  minobsnode = 10).
#'The comonents of Prior are
#' \itemize{
#'\item nu 
#'}
#'@param Mcmc List of MCMC starting values, burn-in ...: e.g.,     list(sigma0 = diag(p - 1), keep = 1, burn = 100, ndraws = 1000, keep_sigma_draws=FALSE)
#'@param seedvalue random seed value, default of 99 will be used if null,
#'@return class_prob_train training data choice/class probabilities,
#'@return predicted_class_train training data predicted choices/classes,
#'@return class_prob_test test data choice/class probabilities,
#'@return predicted_class_test test data predicted choices/classes,
#'@return sigmasample posterior samples of the latent variable covariance matrix.
#'@export


if(is.null(seedvalue)){
  set.seed(99)
} else {
  set.seed(seedvalue)
}


mf <- match.call(expand.dots = FALSE)


out <- mpbart_call(formula = formula, 
                  data = train.data,
                  base = base, 
                  test.data = test.data,
                  Prior = Prior, Mcmc = Mcmc, 
                  varying = varying, sep = sep)
 return(out)
}



