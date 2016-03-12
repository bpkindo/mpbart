#include <iostream>
#include "tree.h"
#include "info.h"
#include <R.h>
#include <R.h>
#include <Rmath.h>
#include <RcppArmadillo.h>
#include <Rcpp.h>

#define LTPI 1.83787706640934536
arma::vec condmom(arma::vec const& x, arma::vec const& mu, arma::mat const& sigmai, int p, int j);

double rtrun(double mu, double sigma,double trunpt, int above);

arma::vec drawwi(arma::vec const& w, arma::vec const& mu, arma::mat const& sigmai, int p, int y);

arma::vec draww(arma::vec const& w, arma::vec const& mu, arma::mat const& sigmai, arma::ivec const& y);
// get cut-points
void getcutpoints(int nc, int n_cov, int n_samp,
                 std::vector<std::vector<double> >& X, xinfo& xi);


bool cansplit(tree::tree_p n, xinfo& xi);
//
void fit(tree& t, std::vector<std::vector<double> >& X, dinfo di, xinfo& xi, std::vector<double>& fv);

// get pseudo response
void getpseudoresponse(dinfo& di, std::vector<std::vector<double> >& ftemp,  
						std::vector<std::vector<double> >& rtemp, arma::mat& sigmai,
						std::vector<std::vector<double> >& r, std::vector<double>& condsig);
Rcpp::List rwishart(int const& nu, arma::mat const& V);
//--------------------------------------------------
//log of the integrated likelihood						
double lil(size_t n, double sy, double sy2, double sigma, double tau);
//--------------------------------------------------
//get sufficient stats for pair of bottom children nl(left) and nr(right) in tree x
void getsuff(std::vector<std::vector<double> >& X, 
				tree& x, tree::tree_cp nl, tree::tree_cp nr, 
				xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr);

//--------------------------------------------------
//get sufficient stats for children (v,c) of node nx in tree x
void getsuff(std::vector<std::vector<double> >& X, 
				tree& x, tree::tree_cp nx, size_t v, size_t c, 
				xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr);


//--------------------------------------------------
//get sufficients stats for all bottom nodes
void allsuff(std::vector<std::vector<double> >& X, 
			tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, 
			std::vector<sinfo>& sv);
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi);
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars);
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots);

void drmu(std::vector<std::vector<double> >& X, tree& t, xinfo& xi, dinfo& di, pinfo& pi);

/*
void dcholdc(std::vector<std::vector<double> >& X, int size, std::vector<std::vector<double> >& L); 
*/
void rWish(arma::mat& Sample,
	   arma::mat& S,
	   size_t df,
	   size_t size);		 
			
void readx(std::vector<std::vector<std::vector<double> > >& XMat,dinfo& di, arma::mat const& pX);
	   
void SWP(arma::mat& X,
	 size_t k,
	 size_t size);
	 
void rMVN(                      
	  std::vector<double>& Sample,
	  std::vector<double>& mean,
	  arma::mat& Var,
	  size_t size);