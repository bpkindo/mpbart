#' Multinomial Probit Bayesian Additive Regression Trees
#'
#' A function to implement multinomial probit regression via Bayesian Addition Regression Trees using partial marginal data augmentation.
#'
#' @useDynLib mpbart
#' @export


mpbart_call <- function(formula, data,base,test.data = NULL, 
                        Prior = NULL, Mcmc = NULL)
{
  


mf <- match.call(expand.dots = FALSE)
m <- match(c("formula", "data"), names(mf), 0L)
mf <- mf[c(1L, m)]
mf[[1L]] <- as.name("model.frame")
mf <- eval(mf, parent.frame())

Y <- model.response(mf)

Y <- as.factor(Y)
lev <- levels(Y)
p <- length(lev)
if (p < 3){
  stop("The number of alternatives should be at least 3.")
}

if (!is.null(base))
{  
  base <- base
} else {
  base <- lev[p]
}

if (base %in% lev) {
  Y <- relevel(Y, ref = base)
  lev <- levels(Y)
} else {
  stop(paste("Error: `base' does not exist in the response variable."))
}

base <- lev[1]
counts <- table(Y)
if (any(counts == 0)) {
  warning(paste("group(s)", paste(lev[counts == 0], collapse = " "), "are empty"))
  Y <- factor(Y, levels  = lev[counts > 0])
  lev <- lev[counts > 0]
}
Y <- as.numeric(unclass(Y)) - 1
Y <- ifelse(Y==0, p,Y)

Terms <- attr(mf, "terms")
X <- model.matrix.default(Terms, mf)

xcolnames <- colnames(X)
xcolnames <- xcolnames[-1]

if(length(xcolnames) ==1 ){
  X <- data.frame(X[,xcolnames])
  names(X) <- xcolnames[1]
  
} else {
  
  X <- X[,xcolnames]
}
if (!is.null(test.data)){
  
  if(length(xcolnames) ==1 ){
    Xtest <- data.frame(test.data[,xcolnames])
    names(Xtest) <- xcolnames[1]
  } else {
    Xtest <- test.data[,xcolnames]  
  }
  
  testXEx = NULL;
    for(i in 1:nrow(Xtest)){
      testXEx = rbind(testXEx, matrix(rep(Xtest[i,], p-1), byrow = TRUE, ncol = ncol(Xtest) ) )
    }
   
  
} else {
  testXEx = 0
}


XEx = NULL;
for(i in 1:nrow(X)){
  XEx = rbind(XEx, matrix(rep(X[i,], p-1), byrow = TRUE, ncol = ncol(X) ) )
}


Data = list(p=p,y=Y,X= XEx)
testData = list(p=p,X= testXEx)


cat("Table of y values",fill=TRUE)
print(table(as.integer(Data$y) ))
            
#print(table(as.integer(Data$y) , model.response(mf)))

n=length(Data$y)
pm1=p-1

if (!is.null(test.data)){
  testn <- nrow(testData$testXEx)/(p-1)
} else {
  testn <- 0
}

# X=createX(p,na=2,nd=NULL,Xa= traindata[,2:9],Xd=NULL, 
#           INT = FALSE,DIFF=TRUE,base=p)


if(missing(Prior)) 
{nu=pm1+3; V=nu*diag(pm1);
 ntrees=200; kfac=2.0;pbd=1.0;pb=0.5;beta = 2.0;alpha = 0.95; nc = 100; priorindep = 0; minobsnode = 10;
}
else 
{if(is.null(Prior$nu)) {nu=pm1+3} else {nu=Prior$nu}
 if(is.null(Prior$V)) {V=nu*diag(pm1)} else {V=Prior$V}
 if(is.null(Prior$ntrees)) {ntrees=200} else {ntrees=Prior$ntrees}
 if(is.null(Prior$kfac)) {kfac=2.0} else {kfac=Prior$kfac}
 if(is.null(Prior$pbd)) {pbd=1.0} else {pbd=Prior$pbd}
 if(is.null(Prior$pb)) {pb=0.5} else {pb=Prior$pb}
 if(is.null(Prior$beta)) {beta = 2.0} else {beta=Prior$beta}
 if(is.null(Prior$alpha)) {alpha = 0.95} else {alpha=Prior$alpha}
 if(is.null(Prior$nc)) {nc=100} else {nc=Prior$nc}
 if(is.null(Prior$priorindep)) {priorindep= FALSE} else {priorindep=Prior$priorindep}
 if(is.null(Prior$minobsnode)) {minobsnode= 10} else {minobsnode=Prior$minobsnode}
 
 
}

if(is.null(Mcmc$sigma0)) {sigma0=diag(pm1)} else {sigma0=Mcmc$sigma0}

if(is.null(Mcmc$keep)) {keep=1} else {keep=Mcmc$keep}
if(is.null(Mcmc$burn)) {burn=100} else {burn=Mcmc$burn}
if(is.null(Mcmc$ndraws)) {ndraws=1000} else {ndraws=Mcmc$ndraws}
if(is.null(Mcmc$keep_sigma_draws)) {keep_sigma_draws=FALSE} else {keep_sigma_draws=Mcmc$keep_sigma_draws}






C=chol(solve(sigma0))
#
#  C is upper triangular root of sigma^-1 (G) = C'C
#
sigmai=crossprod(C)

if( (priorindep ==TRUE) || (keep_sigma_draws==FALSE)){
  sigmasample = as.double(0);
  savesigma = 0;
} else {
  sigmasample = as.double(rep(sigma0, ndraws+burn));
  savesigma = 1;
}


res =   .C("rmnpMDA",w=as.double(rep(0,nrow(X))),
           trainx= as.double(t(X)), 
           testx= as.double(t(testX)),
           mu = as.double(rep(0,nrow(X))),
           sigmai = as.double(sigmai),
           V = as.double(V),
           n = as.integer(length(y)),
           n_dim = as.integer(ncol(sigmai)),
           y = as.integer(y), 
           n_cov = as.integer(k), 
           nu = as.integer(nu), 
           trainpred = as.double(rep(0,p*n)) , 
           testn = as.integer(testn), 
           testpred = as.double(rep(0,p*testn)), 
           ndraws = as.integer(ndraws), 
           burn = as.integer(burn),
           ntrees = as.integer(ntrees),
           kfac = as.double(kfac), 
           pbd = as.double(pbd), 
           pb = as.double(pb), 
           alpha = as.double(alpha),  
           beta =  as.double(beta),
           nc = as.integer(nc),
           savesigma = as.integer(savesigma),
           minobsnode = as.integer(minobsnode),
           sigmasample = sigmasample,
           PACKAGE="mpbart")      



return(list(out = res))

}