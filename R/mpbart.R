

mpbart <- function(formula, train.data, test.data = NULL, base = NULL,
                   Prior = NULL, Mcmc = NULL, 
                   seedvalue = NULL)
{
#'@param formula choice ~ demographic covariates. If there are no, demographic variables use y ~ 1
#'@param data Training Data in wide format
#'@base Base choic. Default is the highest class/choice
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
                  Prior = Prior, Mcmc = Mcmc)
return(out)
}



