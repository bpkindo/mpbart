#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <R.h>
#include <Rmath.h>
#include <R_ext/Lapack.h>
#include "tree.h"
#include "info.h"
#include "bd.h"
#include "funs.h"

void condmom(double *x, double *mu, double *sigi, int p, int j, double *m, double *csig)
{
/*	function to compute moments of x[j] | x[-j]  */

	int ind,i,jm1;
	double csigsq;
	jm1=j-1;
	ind = p*jm1;
	csigsq = 1./sigi[ind+jm1];

	*m = 0.0;
	for (i=0 ; i < p; ++i)
		{
		if (i != jm1) 
			{*m +=  - csigsq*sigi[ind+i]*(x[i]-mu[i]);}
		}
	*m=mu[jm1]+*m ;
	*csig=sqrt(csigsq);
}

double rtrun(double mu, double sigma,double trunpt, int above) 
{
	double FA,FB,rnd,result,arg ;
	if (above) {
		FA=0.0; FB=pnorm(((trunpt-mu)/(sigma)),0.0,1.0,1,0);
			}
	else {
		FB=1.0; FA=pnorm(((trunpt-mu)/(sigma)),0.0,1.0,1,0);
		}
	
	GetRNGstate();
	rnd=unif_rand();
	arg=rnd*(FB-FA)+FA;
	if(arg > .999999999) arg=.999999999;
	if(arg < .0000000001) arg=.0000000001;
	result = mu + sigma*qnorm(arg,0.0,1.0,1,0);
	PutRNGstate();
	return result;
}
void drawwi(double *w, double *mu, double *sigmai,int *p, int *y)
{
/*	function to draw w_i by Gibbing's thru p vector   */

	int i,j,above;
	double bound;
	double mean, csig;

		for (i=0; i < *p; ++i) 
		{	
			bound=0.0;
		    	for (j=0; j < *p ; ++j) 
				{ if (j != i) {bound=fmax2(bound,w[j]); }}
			if (*y == i+1) 	
				above = 0;
			else 
				above = 1;

		condmom(w,mu,sigmai,*p,(i+1),&mean,&csig);
		w[i]=rtrun(mean,csig,bound,above);

		}
}

void draww(double *w, double *mu, double *sigmai, int *n, int *p, int *y) 
{
/*	function to gibbs down entire w vector for all n obs  */
	int i, ind;
	for (i=0; i < *n; ++i)
	{
		ind= *p * i;
		drawwi(w+ind,mu+ind,sigmai,p,y+i);
	}
}


//---------------------------------------------------------------
// get cut-points
void getcutpoints(int nc, int n_cov, int n_samp,
                 std::vector<std::vector<double> >& X, xinfo& xi){
   double xinc; //increments
   double xx;


   std::vector<double> minx(n_cov,INFINITY); // to store the minimum of each of the individual specific pred
   std::vector<double> maxx(n_cov,-INFINITY);// to store the max of each of the individual specific pred

      for(int j=0;j<n_cov;j++) {
      for(int i=0;i<n_samp;i++) {
         xx = X[i][j];
         if(xx < minx[j]) minx[j]=xx;
         if(xx > maxx[j]) maxx[j]=xx;
      }
   }



   //make grid of nc cutpoints between min and max for each x.
   xi.resize(n_cov);
   for(int i=0;i<n_cov;i++) {
      xinc = (maxx[i]-minx[i])/(nc+1.0);
      xi[i].resize(nc);
      for(int j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
   }


}

//--------------------------------------------------
//does this bottom node n have any variables it can split on.
bool cansplit(tree::tree_p n, xinfo& xi)
{
   int L,U;
   bool v_found = false; //have you found a variable you can split on
   size_t v=0;
   while(!v_found && (v < xi.size())) { //invar: splitvar not found, vars left
      L=0; U = xi[v].size()-1;
      n->rg(v,&L,&U);
      if(U>=L) v_found=true;
      v++;
   }
   return v_found;
}
//
void fit(tree& t, std::vector<std::vector<double> >& X, dinfo di, xinfo& xi, std::vector<double>& fv)
{
   double* xx = new double[di.n_cov];
   tree::tree_cp bn;

   for(size_t i=0;i<di.n_samp;i++) {
	for(size_t j=0;j<di.n_cov; j++){
		xx[j] = X[i][j];
	}
      
      bn = t.bn(xx,xi);
      fv[i] = bn->getm();
   }
}
// get pseudo response


void getpseudoresponse(dinfo& di, std::vector<std::vector<double> >& ftemp,  
						std::vector<std::vector<double> >& rtemp, double *sigmai,
						std::vector<std::vector<double> >& r, std::vector<double>& condsig){
double mean, csig;
double* tempres = new double[di.n_dim];
double* tempmean = new double[di.n_dim];
int itemp = 0;
for(size_t i=0; i<di.n_samp; i++){
	itemp = 0;
	//prediction from current tree for current i
	for(size_t k=0; k<di.n_dim; k++){
		tempmean[itemp++] = ftemp[k][i];
	}
	itemp = 0;
	//simulated latent for current i
	for(size_t k=0; k<di.n_dim; k++){
		tempres[itemp++] = rtemp[k][i];
	}

	for(size_t k=0; k<di.n_dim; k++){
		condmom(tempres,tempmean,sigmai,(int)di.n_dim,(int)(k+1),&mean,&csig);
		r[k][i] = tempres[k]- mean + tempmean[k];
		if(i==0) condsig[k] = csig;
	}

}

}



//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots)
{
   double pb;  //prob of birth to be returned
   tree::npv bnv; //all the bottom nodes
   t.getbots(bnv);
   for(size_t i=0;i!=bnv.size();i++)
      if(cansplit(bnv[i],xi)) goodbots.push_back(bnv[i]);
   if(goodbots.size()==0) { //are there any bottom nodes you can split on?
      pb=0.0;
   } else {
      if(t.treesize()==1) pb=1.0; //is there just one node?
      else pb=pi.pb;
   }
   return pb;
}
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars)
{
   int L,U;
   for(size_t v=0;v!=xi.size();v++) {//try each variable
      L=0; U = xi[v].size()-1;
      n->rg(v,&L,&U);
      if(U>=L) goodvars.push_back(v);
   }
}
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi)
{
   if(cansplit(n,xi)) {
      return pi.alpha/pow(1.0+n->depth(),pi.beta);
   } else {
      return 0.0;
   }
}
//--------------------------------------------------
//get sufficients stats for all bottom nodes
void allsuff(std::vector<std::vector<double> >& X, 
			tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, 
			std::vector<sinfo>& sv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observations
   size_t ni;         //the  index into vector of the current bottom node
     double* xx = new double[di.n_cov];
   double y;          //current y

   bnv.clear();
   x.getbots(bnv);

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   sv.resize(nb);

   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;

   for(size_t i=0;i<di.n_samp;i++) {
      for(size_t j=0;j<di.n_cov; j++){
		xx[j] = X[i][j];
	  }
      y=di.y[i];

      tbn = x.bn(xx,xi);
      ni = bnmap[tbn];

      ++(sv[ni].n);
      sv[ni].sy += y;
      sv[ni].sy2 += y*y;
   }
}
//--------------------------------------------------
//get sufficient stats for children (v,c) of node nx in tree x
void getsuff(std::vector<std::vector<double> >& X, 
			tree& x, tree::tree_cp nx, size_t v, size_t c, 
			xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr)
{
      double* xx = new double[di.n_cov];
   double y;  //current y
   sl.n=0;sl.sy=0.0;sl.sy2=0.0;
   sr.n=0;sr.sy=0.0;sr.sy2=0.0;

   for(size_t i=0;i<di.n_samp;i++) {
      for(size_t j=0;j<di.n_cov; j++){
		xx[j] = X[i][j];
	  }
	  
      if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node
         y = di.y[i];
         if(xx[v] < xi[v][c]) {
               sl.n++;
               sl.sy += y;
               sl.sy2 += y*y;
          } else {
               sr.n++;
               sr.sy += y;
               sr.sy2 += y*y;
          }
      }
   }
}
//--------------------------------------------------
//get sufficient stats for pair of bottom children nl(left) and nr(right) in tree x
void getsuff(std::vector<std::vector<double> >& X, 
			tree& x, tree::tree_cp nl, tree::tree_cp nr, 
			xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr)
{


     double* xx = new double[di.n_cov];
   double y;  //current y
   sl.n=0;sl.sy=0.0;sl.sy2=0.0;
   sr.n=0;sr.sy=0.0;sr.sy2=0.0;

   for(size_t i=0;i<di.n_samp;i++) {
      for(size_t j=0;j<di.n_cov; j++){
		xx[j] = X[i][j];
	  }
      tree::tree_cp bn = x.bn(xx,xi);
      if(bn==nl) {
         y = di.y[i];
         sl.n++;
         sl.sy += y;
         sl.sy2 += y*y;
      }
      if(bn==nr) {
         y = di.y[i];
         sr.n++;
         sr.sy += y;
         sr.sy2 += y*y;
      }
   }
}
//--------------------------------------------------
//log of the integrated likelihood
double lil(size_t n, double sy, double sy2, double sigma, double tau)
{
   double yb,yb2,S,sig2,d;
   double sum, rv;

   yb = sy/n;
   yb2 = yb*yb;
   S = sy2 - (n*yb2);
   sig2 = sigma*sigma;
   d = n*tau*tau + sig2;
   sum = S/sig2 + (n*yb2)/d;
   rv = -(n*LTPI/2.0) - (n-1)*log(sigma) -log(d)/2.0;
   rv = rv -sum/2.0;
   return rv;
}


void drmu(std::vector<std::vector<double> >& X, tree& t, xinfo& xi, dinfo& di, pinfo& pi)
{
GetRNGstate();

   tree::npv bnv;
   std::vector<sinfo> sv;
   allsuff(X, t,xi,di,bnv,sv);

   double a = 1.0/(pi.tau * pi.tau);
   double sig2 = pi.sigma * pi.sigma;
   double b,ybar;

   for(tree::npv::size_type i=0;i!=bnv.size();i++) {
      b = sv[i].n/sig2;
      ybar = sv[i].sy/sv[i].n;
      bnv[i]->setm(b*ybar/(a+b) + norm_rand()/sqrt(a+b));
   }

   
PutRNGstate();

}


void dinv(std::vector<std::vector<double> >& X,
	  int	size,
	  std::vector<std::vector<double> >& X_inv)
{
  int i,j, k, errorM;
  double* pdInv = new double[(int)(size * size)];
 X_inv.resize(size);
 for (j = 0; j < size; j++) X_inv[j].resize(size);
  

  for (i = 0, j = 0; j < size; j++) 
    for (k = 0; k <= j; k++) 
      pdInv[i++] = X[k][j];
  F77_CALL(dpptrf)("U", &size, pdInv, &errorM);
  if (!errorM) {
    F77_CALL(dpptri)("U", &size, pdInv, &errorM);
    if (errorM) {
      Rprintf("LAPACK dpptri failed, %d\n", errorM);
      
	  for(int jj=0; jj<size;jj++){
		for(int kk=0; kk<size;kk++){
		
	  }
	  }
	  
	  error("Exiting from dinv().\n");
	  
    }
  }
  else {
    Rprintf("LAPACK dpptrf failed, %d\n", errorM);
		  for(int jj=0; jj<size;jj++){
		for(int kk=0; kk<size;kk++){
			
	  }
	  }
    error("Exiting from dinv().\n");
  }
  for (i = 0, j = 0; j < size; j++) {
    for (k = 0; k <= j; k++) {
      X_inv[j][k] = pdInv[i];
      X_inv[k][j] = pdInv[i++];
    }
  }


}


void dcholdc(std::vector<std::vector<double> >& X, int size, std::vector<std::vector<double> >& L)
{
  int i, j, k, errorM;
  double* pdTemp = new double[(int)(size * size)];
  L.resize(size);
 
  for (j = 0; j < size; j++) L[j].resize(size);
  for (j = 0, i = 0; j < size; j++) 
    for (k = 0; k <= j; k++) 
      pdTemp[i++] = X[k][j];
  F77_CALL(dpptrf)("U", &size, pdTemp, &errorM);
  if (errorM) {
    Rprintf("LAPACK dpptrf failed, %d\n", errorM);
    error("Exiting from dcholdc().\n");
  }
  for (j = 0, i = 0; j < size; j++) {
    for (k = 0; k < size; k++) {
      if(j<k)
	L[j][k] = 0.0;
      else
	L[j][k] = pdTemp[i++];
    }
  }

} 



void rWish(std::vector<std::vector<double> >& Sample,        /* The matrix with to hold the sample */
	   std::vector<std::vector<double> >& S,             /* The parameter */
	   int df,                 /* the degrees of freedom */
	   int size)               /* The dimension */
{
GetRNGstate();

  int i,j,k;

  double* V = new double[(int)size];
  std::vector<std::vector<double> > B, C, N, mtemp;
	B.resize(size); C.resize(size); N.resize(size); mtemp.resize(size);
	for (j = 0; j < size; j++){
		B[j].resize(size); C[j].resize(size); N[j].resize(size); mtemp[j].resize(size);
	}
  
  for(i=0;i<size;i++) {
    V[i]=rchisq((double) df-i-1);
    B[i][i]=V[i];
    for(j=(i+1);j<size;j++)
      N[i][j]=norm_rand();
  }

  for(i=0;i<size;i++) {
    for(j=i;j<size;j++) {
      Sample[i][j]=0;
      Sample[j][i]=0;
      mtemp[i][j]=0;
      mtemp[j][i]=0;
      if(i==j) {
	if(i>0)
	  for(k=0;k<j;k++)
	    B[j][j]+=N[k][j]*N[k][j];
      }
      else { 
	B[i][j]=N[i][j]*sqrt(V[i]);
	if(i>0)
	  for(k=0;k<i;k++)
	    B[i][j]+=N[k][i]*N[k][j];
      }
      B[j][i]=B[i][j];
    }
  }
  
  dcholdc(S, size, C);
  for(i=0;i<size;i++)
    for(j=0;j<size;j++)
      for(k=0;k<size;k++)
	mtemp[i][j]+=C[i][k]*B[k][j];
  for(i=0;i<size;i++)
    for(j=0;j<size;j++)
      for(k=0;k<size;k++)
	Sample[i][j]+=mtemp[i][k]*C[j][k];
PutRNGstate();

}

void DrawSigma(dinfo& di, double *V, std::vector<std::vector<double> >& allfit, 
				double *w, std::vector<std::vector<double> >& WishSample, int nu)
{

std::vector<std::vector<double> > WishMat1;
std::vector<std::vector<double> > epsilon;
std::vector<std::vector<double> > WishMat1Inv;

epsilon.resize(di.n_dim);
WishMat1.resize(di.n_dim);
for(size_t j=0;j<di.n_dim;j++){
	WishMat1[j].resize(di.n_dim);
	epsilon[j].resize(di.n_samp);
}

for(size_t i=0; i<di.n_samp; i++){
	for(size_t k=0; k<di.n_dim; k++){
		epsilon[k][i] = w[i*di.n_dim + k] - allfit[k][i];
	}
}

for(size_t j=0;j<di.n_dim;j++){
	for(size_t k=0;k<di.n_dim;k++){
		WishMat1[j][k] = V[j*di.n_dim + k];
	}
}


for(size_t i=0; i<di.n_samp; i++){
	for(size_t j=0;j<di.n_dim;j++){
		for(size_t k=0;k<di.n_dim;k++){
		WishMat1[j][k] +=epsilon[j][i]*epsilon[k][i];
		}
	}	 
}

dinv(WishMat1 ,di.n_dim,WishMat1Inv);
rWish(WishSample, WishMat1Inv, (int)(nu+di.n_samp),(int)di.n_dim);

}


//read X 

void readx(std::vector<std::vector<std::vector<double> > >& XMat,dinfo& di, double *pX){
	XMat.resize(di.n_dim);
	for(size_t j=0; j < di.n_dim; j++){
	 XMat[j].resize(di.n_samp);
	}

	for(size_t j=0; j < di.n_dim; j++){
		for(size_t i=0; i < di.n_samp; i++){
			 XMat[j][i].resize(di.n_cov);
		}
	}


	int itemp = 0;
	for(size_t i=0; i < di.n_samp; i++){
		for(size_t j=0; j <  di.n_dim; j++){
			for(size_t k=0; k< di.n_cov; k++){
			 XMat[j][i][k] = pX[itemp++];
			}
		}
	}
}



/*  The Sweep operator */
void SWP(
	 std::vector<std::vector<double> >& X,             // The Matrix to work on 
	 size_t k,                  //The row to sweep 
	 size_t size)               // The dim. of X 
{

  if (X[k][k] < 10e-20) 
    error("SWP: singular matrix.\n");
  else
    X[k][k]=-1/X[k][k];
  for(size_t i=0;i<size;i++)
    if(i!=k){
      X[i][k]=-X[i][k]*X[k][k];
      X[k][i]=X[i][k];
    }
  for(size_t i=0;i<size;i++)
    for(size_t j=0;j<size;j++)
      if(i!=k && j!=k)
	X[i][j]=X[i][j]+X[i][k]*X[k][j]/X[k][k];
  
}

// draw from MVN -- adapted from R package MNP
void rMVN(                      
	  std::vector<double>& Sample,
	  std::vector<double>& mean,
	  std::vector<std::vector<double> >& Var,
	  size_t size)
{
	GetRNGstate();
	
  std::vector<std::vector<double> > Model;
  Model.resize(size +1);
  for(size_t j=0; j<= size; j++){
	  Model[j].resize(size + 1);
  }

  double cond_mean;
    
  /* draw from mult. normal using SWP */
  for(size_t j=1;j<=size;j++){       
    for(size_t k=1;k<=size;k++) {
      Model[j][k]=Var[j-1][k-1];
	}
    Model[0][j]=mean[j-1];
    Model[j][0]=mean[j-1];
  }
  Model[0][0]=-1;
  Sample[0]=(double)norm_rand()*sqrt(Model[1][1])+Model[0][1];
  for(size_t j=2;j<=size;j++){
    SWP(Model,j-1,size+1);
    cond_mean=Model[j][0];
    for(size_t k=1;k<j;k++) cond_mean+=Sample[k-1]*Model[j][k];
    Sample[j-1]=(double)norm_rand()*sqrt(Model[j][j])+cond_mean;
  }
  
PutRNGstate();
}



