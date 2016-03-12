#include <iostream>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <R.h>
#include <Rmath.h>
#include "funs.h"
#include "tree.h"
#include "info.h"
#include "RcppArmadillo.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]


Rcpp::List rcpparma_rmnpMDA(arma::mat const trainx,
			arma::mat const testx , 
			arma::mat sigmai,
			arma::mat V, 
			int n,
			int n_dim, 
			arma::ivec y, 
			int pn_cov, 
			int nu, 
			int testn, 
			int pndraws, 
			int pburn,
			int pntrees,
			double pkfac, 
			double ppbd, 
			double ppb, 
			double palpha,
			double pbetap, 
			int pnc,
			int savesigma,
			int minobsnode){
			
			

dinfo di; dinfo dip;
di.n_samp = n; 
di.n_cov = pn_cov; 
di.n_dim = n_dim;
if(testn > 0){
	dip.n_samp = testn; dip.n_cov = pn_cov; 
	dip.n_dim = n_dim; dip.y=0;
}

arma::mat mtemp1(di.n_dim,di.n_dim);
arma::mat WishSample(di.n_dim,di.n_dim);
arma::mat WishSampleInv(di.n_dim,di.n_dim);
arma::mat SigmaTmp(di.n_dim,di.n_dim);
arma::mat SigmaTmpInv(di.n_dim,di.n_dim);
arma::mat WishSampleTildeInv(di.n_dim,di.n_dim);
arma::mat WishMat1(di.n_dim,di.n_dim);
//Initialize latents
arma::vec wold = arma::zeros<arma::vec>(di.n_samp*di.n_dim);
arma::vec mu = arma::zeros<arma::vec>(di.n_samp*di.n_dim);
arma::vec w=wold;
w = draww(wold,mu,sigmai,y);
std::vector<size_t> unqytrn;
double maxy;
maxy = R_NegInf;
for(size_t i=0; i<di.n_samp;i++){
	unqytrn.push_back(y(i));
	if(maxy<y(i)) maxy = y(i);
}


std::sort(unqytrn.begin(),unqytrn.end());
std::vector<size_t>::iterator it;
it = std::unique(unqytrn.begin(),unqytrn.end());
unqytrn.resize( std::distance(unqytrn.begin(),it) );

std::vector<std::vector<std::vector<double> > > XMat;
readx(XMat,di, trainx);

	std::vector<std::vector<std::vector<double> > > testXMat;
if(testn > 0){
	readx(testXMat,dip, testx);
}
std::vector<xinfo> xi;
xi.resize((int)di.n_dim);
int nc=pnc; // qually spaced cutpoints from min to max.
for(int j=0; j< (int)di.n_dim; j++){
	getcutpoints(nc, (int)di.n_cov, (int)di.n_samp,XMat[j],xi[j]);
}
std::vector<std::vector<double> > allfit; //sum of fit of all trees
std::vector<std::vector<double> > wtilde;

std::vector<std::vector<double> > ppredmeanvec;
if(testn > 0){
	ppredmeanvec.resize(dip.n_dim);
	for(size_t k=0; k<dip.n_dim; k++){
		ppredmeanvec[k].resize(dip.n_samp);
	}
}  
 //fit of current tree
std::vector<std::vector<double> > ftemp;
ftemp.resize(di.n_dim);
std::vector<double> fpredtemp;
if(testn > 0){
 //temporary fit vector to compute prediction
	fpredtemp.resize(dip.n_samp);
}
std::vector<double> mvnsample,mvnmean; //temp storage for mvn samples
mvnsample.resize(di.n_dim);
mvnmean.resize(di.n_dim);
allfit.resize(di.n_dim);
wtilde.resize(di.n_dim);
std::vector<std::vector<double> > predclasstrain;
predclasstrain.resize(di.n_samp);
for(size_t i=0; i<di.n_samp; i++){
	predclasstrain[i].resize(di.n_dim);
}


if(testn > 0){
	std::vector<std::vector<double> > predclasstest;
	predclasstest.resize(dip.n_samp);
	for(size_t i=0; i<dip.n_samp; i++){
		predclasstest[i].resize(dip.n_dim);
	}
}


for(size_t k=0; k<di.n_dim; k++){
  ftemp[k].resize(di.n_samp);
  allfit[k].resize(di.n_samp);
  wtilde[k].resize(di.n_samp);
}


std::vector<std::vector<double> > r;
std::vector<std::vector<double> > rtemp;
rtemp.resize(di.n_dim);
r.resize(di.n_dim);
for(size_t k=0; k<di.n_dim; k++){
 r[k].resize(di.n_samp);
 rtemp[k].resize(di.n_samp);
}

// priors and parameters
size_t burn = pburn; //number of mcmc iterations called burn-in
size_t nd = pndraws; //number of mcmc iterations
size_t m = pntrees;
double kfac = pkfac;

pinfo pi;
pi.pbd = ppbd; //prob of birth/death move
pi.pb = ppb; //prob of birth given  birth/death

pi.alpha = palpha; //prior prob a bot node splits is alpha/(1+d)^beta, d is depth of node
pi.betap = pbetap; //
pi.tau = (3.0)/(kfac*sqrt((double)m));
pi.sigma = 1.0;

  //storage for ouput
//in sample fit

std::vector<size_t> vec_class_pred_train, vec_class_pred_test;
double max_temp;// used to find the class membership
size_t pclass;// used to find class membership

   //initialize tree

std::vector<std::vector<tree> > t;
t.resize(m);
for(size_t i=0; i<m; i++){
	t[i].resize(di.n_dim);
}
for(size_t i=0; i<m; i++){
	for(size_t k=0;k< di.n_dim;k++) t[i][k].setm(0.00);
}

for(size_t k=0; k<di.n_dim; k++){
	for(size_t i=0;i<di.n_samp;i++) {
		allfit[k][i] = 0.0;
		wtilde[k][i] = 0.0;
	}
}

double alpha2, alpha2old;

std::vector<double> condsig;
condsig.resize(di.n_dim);
//MCMC
Rcpp::Rcout << "\nMCMC:\n";
time_t tp;
int time1 = time(&tp);
arma::mat WishMat1cholinv, WishMat1VInv; 
int sigdrawcounter = 0;

arma::mat psigmasample(nd+burn, di.n_dim*di.n_dim);

for(size_t loop=0;loop<(nd+burn);loop++) {
if(loop%100==0) Rprintf("iteration: %d of %d \n",loop, nd+burn);
wold = w;
w = draww(wold,mu,sigmai,y);
 if(loop==0){
  wold = w;
  w = draww(wold,mu,sigmai,y);
 }
mtemp1 = V*sigmai;
alpha2=arma::trace(mtemp1)/(double)R::rchisq((double)nu*di.n_dim);
for(size_t k=0; k<di.n_dim; k++){
	for(size_t i=0;i<di.n_samp;i++) {
		wtilde[k][i] = sqrt(alpha2) * (w(i*di.n_dim + k));
	}
}
	for(size_t ntree = 0 ; ntree <m; ntree++){
	for(size_t k=0; k<di.n_dim; k++){
		fit(t[ntree][k], XMat[k], di, xi[k], ftemp[k]);
		for(size_t i=0;i<di.n_samp;i++) {
			allfit[k][i] -= ftemp[k][i];
			rtemp[k][i] = wtilde[k][i] - allfit[k][i];
		}
	}
	getpseudoresponse(di, ftemp, rtemp, sigmai, r,condsig);
	for(size_t k=0; k<di.n_dim; k++){
		di.y = &r[k][0];
		pi.sigma = condsig[k];
		bd(XMat[k], t[ntree][k], xi[k], di, pi, minobsnode);
		drmu(XMat[k],t[ntree][k],xi[k],di,pi);
		fit(t[ntree][k], XMat[k], di, xi[k], ftemp[k]);
		for(size_t i=0;i<di.n_samp;i++) {
			allfit[k][i] += ftemp[k][i]	;
		}
	}
}
	
WishMat1.fill(0.0);
arma::mat epsilon(di.n_dim,di.n_samp);
for(size_t i=0;i<di.n_samp;i++){
	for(size_t j=0;j<di.n_dim;j++){
			epsilon(j,i) = (wtilde[j][i]-allfit[j][i]);
	}
}
WishMat1 = epsilon*arma::trans(epsilon);
WishMat1cholinv = arma::solve(arma::trimatu(arma::chol(V+WishMat1)), arma::eye(WishMat1.n_cols,WishMat1.n_cols));
WishMat1VInv = WishMat1cholinv*arma::trans(WishMat1cholinv);
	  
Rcpp::List WishSampletmp = rwishart(nu+di.n_samp,WishMat1VInv);
arma::mat C = Rcpp::as<arma::mat>(WishSampletmp["C"]);
sigmai = arma::trans(C)*C;

WishSampleTildeInv = Rcpp::as<arma::mat>(WishSampletmp["W"]);
arma::mat WishSampleTilde = Rcpp::as<arma::mat>(WishSampletmp["IW"]);

alpha2old = alpha2;
alpha2 = arma::trace(WishSampleTildeInv)/double(di.n_dim);

for(size_t i=0; i<di.n_samp; i++){
	for(size_t k=0; k < di.n_dim; k++){
	
		mu[i*di.n_dim + k] = allfit[k][i]/sqrt(alpha2); //divide allfit this to transform
		w(i*di.n_dim +k) = allfit[k][i]/sqrt(alpha2old) + (wtilde[k][i]-allfit[k][i]) /sqrt(alpha2) ; 
	}
}

if(savesigma==1){
	for(size_t j=0;j<di.n_dim;j++){
		for(size_t k=0;k<di.n_dim;k++){
			psigmasample(sigdrawcounter++) = WishSampleTilde(j,k)*alpha2;
			}
		}
	} else {
	psigmasample.reset();
	}
	
if(loop>=burn){
	 for(size_t k = 0; k <di.n_samp; k++){
		max_temp = R_NegInf;
		for(size_t l=0; l<di.n_dim; l++){
			mvnmean[l] = allfit[l][k];
		}
		
	  rMVN(mvnsample, mvnmean, WishSampleTildeInv,di.n_dim);
	  
		for(size_t l = 0 ; l < di.n_dim; l++){
			if(mvnsample[l] > max_temp){
					max_temp = mvnsample[l];
					pclass = l+1;
			}
		}
		if(max_temp <=0) {
			pclass = (size_t)maxy;
		}
		vec_class_pred_train.push_back(pclass);
		//Rcpp::Rcout << "pclass: " << pclass << std::endl;;
	 }
	 
	if(testn > 0) {
		
		for(size_t k=0; k<dip.n_dim; k++){
			for(size_t i=0;i<dip.n_samp;i++) {
			ppredmeanvec[k][i] = 0.0;

			}
		}
	
		for(size_t l = 0; l < dip.n_dim; l++){
			for(size_t j=0;j<m;j++) {
				fit(t[j][l], testXMat[l], dip, xi[l], fpredtemp);
				for(size_t k=0;k<dip.n_samp;k++) ppredmeanvec[l][k] += fpredtemp[k];
			}
		}
		
		for(size_t k = 0; k <dip.n_samp; k++){
			max_temp = R_NegInf;
			
			for(size_t l=0; l<di.n_dim; l++){
				mvnmean[l] = ppredmeanvec[l][k];
			}
		
		rMVN(mvnsample, mvnmean, WishSampleTildeInv,di.n_dim);

			for(size_t l = 0 ; l < dip.n_dim; l++){
				if(mvnmean[l] > max_temp){
						max_temp = mvnmean[l];
						pclass = l+1;
				}
			}
			if(max_temp <=0) {
				pclass = (size_t)maxy;
			}
			vec_class_pred_test.push_back(pclass);
			//Rcpp::Rcout << "pclass: " << pclass << std::endl;;
			}
		
	}
	 
}
	
}
 //end of loop

int time2 = time(&tp);
Rcpp::Rcout << "time for mcmc loop: " << time2-time1 << "  secs." << std::endl;
arma::mat ptrainpred(di.n_samp, di.n_dim+1);
ptrainpred.fill(0);
std::vector<size_t> temp_vec;
temp_vec.resize(nd-burn);
	for(size_t i =0; i <di.n_samp; i++){
		for(size_t loop=burn; loop<(size_t)nd; loop++){
			temp_vec[loop-burn] = vec_class_pred_train[loop*di.n_samp + i];
		}
		for(size_t k=0; k<=di.n_dim; k++){
			ptrainpred(i,k)  =  (double)std::count(temp_vec.begin(), temp_vec.end(), unqytrn[k])/(double)(nd-burn);
		}

	}
	
std::vector<size_t> temp_vec_test;
temp_vec_test.resize(nd-burn);
arma::mat ptestpred;
if(testn > 0) {
ptestpred.resize(dip.n_samp,dip.n_dim+1);
for(size_t i =0; i <dip.n_samp; i++){
	for(size_t loop=burn; loop<(size_t)nd; loop++){
		temp_vec_test[loop-burn] = vec_class_pred_test[loop*dip.n_samp + i];
	}
	for(size_t k=0; k<=dip.n_dim; k++){
		ptestpred(i,k)  =  (double)std::count(temp_vec_test.begin(), temp_vec_test.end(), unqytrn[k])/(double)(nd-burn);
	}
}

} else {
ptestpred.resize(1,1);
ptestpred.fill(0);
}


return Rcpp::List::create(Rcpp::Named("ptrainpred")=ptrainpred,
                         Rcpp::Named("ptestpred")=ptestpred, 
						  Rcpp::Named("psigmasample") = psigmasample);
							  
}
