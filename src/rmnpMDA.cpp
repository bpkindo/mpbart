#include <iostream>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <R.h>
#include <Rmath.h>
#include <R_ext/Lapack.h>
#include "funs.h"
#include "tree.h"
#include "info.h"


extern "C" {

void rmnpMDA(double *w, double  *pX, double *testpX, double *mu, double *sigmai, double *V, 
					int *pn, int *pn_dim, int *y, int *pn_cov, int *pnu, double *ptrainpred, 
					int *testn, double *ptestpred, 
					int *pndraws, 
					int *pburn,
					int *pntrees,
					double *pkfac, 
					double *ppbd, 
					double *ppb, 
					double *palpha,
					double *pbeta, 
					int *pnc,
					int *psavesigma,
					int *pminobsnode,
					double *psigmasample){
// w is the starting value of latens

//    *w is n_samp x n_dim vector
//    *pX is n_samp(n_dim) x n_cov  matrix. eg. row 0 corresponds to first choic, row 1 corresponds to 2nd choice
//     *y is multinomial 1,..., (n_dim + 1)
//  *sigmai is (n_dim) x (n_dim) 


dinfo di; dinfo dip;
di.n_samp = *pn; di.n_cov = *pn_cov; di.n_dim = *pn_dim;
if(*testn){
	dip.n_samp = *testn; dip.n_cov = *pn_cov; dip.n_dim = *pn_dim; dip.y=0;
}


//Initialize latents
int nn = *pn;
int nndim = *pn_dim;
int nu = *pnu;
int savesigma= *psavesigma;
size_t minobsnode = *pminobsnode;

draww(w, mu, sigmai, &nn,&nndim,y);

std::vector<size_t> unqytrn;
double maxy;
maxy = R_NegInf;
for(size_t i=0; i<di.n_samp;i++){
	unqytrn.push_back(y[i]);
	if(maxy<y[i]) maxy = y[i];
}


std::sort(unqytrn.begin(),unqytrn.end());
std::vector<size_t>::iterator it;
it = std::unique(unqytrn.begin(),unqytrn.end());
unqytrn.resize( std::distance(unqytrn.begin(),it) );


std::vector<std::vector<std::vector<double> > > XMat;
readx(XMat,di, pX);

	std::vector<std::vector<std::vector<double> > > testXMat;
if(*testn){
	readx(testXMat,dip, testpX);
}


std::vector<xinfo> xi;
xi.resize((int)di.n_dim);
int nc=*pnc; // qually spaced cutpoints from min to max.
for(int j=0; j< (int)di.n_dim; j++){
	getcutpoints(nc, (int)di.n_cov, (int)di.n_samp,XMat[j],xi[j]);
}


std::vector<std::vector<double> > allfit; //sum of fit of all trees
std::vector<std::vector<double> > wtilde;

std::vector<std::vector<double> > ppredmeanvec;
if(*testn){
	
	ppredmeanvec.resize(dip.n_dim);
	for(size_t k=0; k<dip.n_dim; k++){
		ppredmeanvec[k].resize(dip.n_samp);
	}
}  
 //fit of current tree
std::vector<std::vector<double> > ftemp;
ftemp.resize(di.n_dim);
std::vector<double> fpredtemp;
if(*testn){
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


if(*testn){
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


std::vector<double> condsig;
condsig.resize(di.n_dim);

// priors and parameters
size_t burn = *pburn; //number of mcmc iterations called burn-in
size_t nd = *pndraws; //number of mcmc iterations
size_t m=*pntrees;
double kfac=*pkfac;

pinfo pi;
pi.pbd=*ppbd; //prob of birth/death move
pi.pb=*ppb; //prob of birth given  birth/death

pi.alpha=*palpha; //prior prob a bot node splits is alpha/(1+d)^beta, d is depth of node
pi.beta=*pbeta; //
pi.tau=(3.0)/(kfac*sqrt((double)m));
pi.sigma=1.0;

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

std::vector<std::vector<double> > mtemp1;
std::vector<std::vector<double> > WishSample,WishSampleTilde,WishSampleInv, SigmaTmp, SigmaTmpInv;
WishSampleInv.resize(di.n_dim);
WishSample.resize(di.n_dim);
SigmaTmp.resize(di.n_dim);
SigmaTmpInv.resize(di.n_dim);

WishSampleTilde.resize(di.n_dim);

WishSampleInv.resize(di.n_dim);
mtemp1.resize(di.n_dim);
for(size_t j=0;j<di.n_dim;j++){
	WishSample[j].resize(di.n_dim);
	WishSampleTilde[j].resize(di.n_dim);
	mtemp1[j].resize(di.n_dim);
	WishSampleInv[j].resize(di.n_dim);
	SigmaTmp[j].resize(di.n_dim);
	SigmaTmpInv[j].resize(di.n_dim);

}



std::vector<std::vector<double> > WishMat1, WishMat1Inv, WishSampleTildeInv;

WishMat1.resize(di.n_dim);
WishSampleTildeInv.resize(di.n_dim);
for(size_t j=0;j<di.n_dim;j++){
	WishMat1[j].resize(di.n_dim);
	WishSampleTildeInv[j].resize(di.n_dim);
}



double alpha2, alpha2old, ss;
int sigdrawcounter = 0;
//MCMC


//cout << "\nMCMC:\n";
time_t tp;
int time1 = time(&tp);

for(size_t loop=0;loop<(nd+burn);loop++) {


if(loop%100==0) Rprintf("iteration: %d of %d \n",loop, nd+burn);


draww(w, mu, sigmai, &nn,&nndim,y);
if(loop==0) draww(w, mu, sigmai, &nn,&nndim,y);

ss=0;
for(size_t j=0;j<di.n_dim;j++){      
	for(size_t k=0;k<di.n_dim;k++) {
	mtemp1[j][k]=0;
	}
}

for(size_t i=0;i<di.n_dim;i++){
	for(size_t j=0;j<di.n_dim;j++){
		for(size_t k=0;k<di.n_dim;k++){
			mtemp1[j][k]+=V[j*di.n_dim + i]*sigmai[i*di.n_dim + k];
		}
	}
}

for(size_t j=0;j<di.n_dim;j++) ss+=mtemp1[j][j];
alpha2=ss/(double)rchisq((double)nu*di.n_dim);


for(size_t k=0; k<di.n_dim; k++){
	for(size_t i=0;i<di.n_samp;i++) {
		wtilde[k][i] = sqrt(alpha2) * (w[i*di.n_dim + k]);
	}
}


//done sampling alpha2, w

for(size_t ntree = 0 ; ntree <m; ntree++){
for(size_t k=0; k<di.n_dim; k++){
	fit(t[ntree][k], XMat[k], di, xi[k], ftemp[k]);
	for(size_t i=0;i<di.n_samp;i++) {
		allfit[k][i] -= ftemp[k][i];
		rtemp[k][i] = wtilde[k][i] - allfit[k][i];
	}
}


//get pseudo response
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


//done sampling (T,M)




for(size_t j=0;j<di.n_dim;j++){
	for(size_t k=0;k<di.n_dim;k++){
		WishMat1[j][k]=0.0;
	}
}

for(size_t i=0;i<di.n_samp;i++){
	for(size_t j=0;j<di.n_dim;j++){
		for(size_t k=0;k<di.n_dim;k++){
			WishMat1[j][k] += (wtilde[j][i]-allfit[j][i])* (wtilde[k][i] - allfit[k][i]);
		}
	}
}



dinv(WishMat1 ,di.n_dim,WishMat1Inv);
rWish(WishSampleTildeInv, WishMat1Inv, (int)(nu+di.n_samp),(int)di.n_dim);


dinv(WishSampleTildeInv ,di.n_dim,WishSampleTilde);
	
alpha2old = alpha2;
alpha2 = 0;
for(size_t j=0; j< di.n_dim; j++) alpha2 += (WishSampleTilde[j][j])/double(di.n_dim);

for(size_t i=0; i<di.n_samp; i++){
	for(size_t k=0; k < di.n_dim; k++){
	
		mu[i*di.n_dim + k] = allfit[k][i]/sqrt(alpha2); //divide allfit this to transform
		w[i*di.n_dim +k] = allfit[k][i]/sqrt(alpha2old) + (wtilde[k][i]-allfit[k][i]) /sqrt(alpha2) ; 
	}
}

	for(size_t j=0;j<di.n_dim;j++){
		for(size_t k=0;k<di.n_dim;k++){
			sigmai[j*di.n_dim + k] = WishSampleTildeInv[j][k]*alpha2;
			SigmaTmpInv[j][k] = WishSampleTildeInv[j][k]*alpha2;
			if(savesigma==1){
			psigmasample[sigdrawcounter++] = WishSampleTilde[j][k]/alpha2;
			}
		}
	}

if(loop>=burn){
	dinv(SigmaTmpInv ,di.n_dim,SigmaTmp);
	 for(size_t k = 0; k <di.n_samp; k++){
		max_temp = R_NegInf;
		for(size_t l=0; l<di.n_dim; l++){
			mvnmean[l] = allfit[l][k];
		}
		
	  rMVN(mvnsample, mvnmean, SigmaTmp,di.n_dim);
	  
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
		//cout << "pclass: " << pclass << endl;
	 }
	 
	if(*testn) {
		
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
		
		rMVN(mvnsample, mvnmean, SigmaTmp,di.n_dim);

			for(size_t l = 0 ; l < dip.n_dim; l++){
				if(mvnsample[l] > max_temp){
						max_temp = mvnsample[l];
						pclass = l+1;
				}
			}
			if(max_temp <=0) {
				pclass = (size_t)maxy;
			}
			vec_class_pred_test.push_back(pclass);
			//cout << "pclass: " << pclass << endl;
			}
		
	}
	 
}



} //end of loop

int time2 = time(&tp);
Rprintf("time for mcmc loop %d secs", time2-time1);

std::vector<size_t> temp_vec;
temp_vec.resize(nd-burn);
	for(size_t i =0; i <di.n_samp; i++){
		for(size_t loop=0; loop<(size_t)(nd-burn); loop++){
			temp_vec[loop] = vec_class_pred_train[loop*di.n_samp + i];

		}
		for(size_t k=0; k<=di.n_dim; k++){
			ptrainpred[i*(di.n_dim+1) + k]  =  (double)std::count(temp_vec.begin(), temp_vec.end(), unqytrn[k])/(double)(nd-burn);
		}

	}
	
std::vector<size_t> temp_vec_test;
temp_vec_test.resize(nd-burn);
if(*testn) {
for(size_t i =0; i <dip.n_samp; i++){
	for(size_t loop=0; loop<(size_t)(nd-burn); loop++){
		temp_vec_test[loop] = vec_class_pred_test[loop*dip.n_samp + i];
	}
	for(size_t k=0; k<=dip.n_dim; k++){
		ptestpred[i*(dip.n_dim+1) + k]  =  (double)std::count(temp_vec_test.begin(), temp_vec_test.end(), unqytrn[k])/(double)(nd-burn);
	}
}

}

}
};
