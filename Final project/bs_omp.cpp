#include <Rcpp.h>
#include <omp.h>

using namespace Rcpp;

NumericVector basis(NumericVector x, int degree, int i, NumericVector knots, int lenx){

    NumericVector B(lenx); // output variable
    NumericVector alpha1(lenx), alpha2(lenx);
         
     //NumericVector B 
    if(degree==0) {
        B = wrap(ifelse((x >= knots[i]) & (x < knots[i+1]), 1, 0));
    } else {
        
        //alpha1
        if((knots[degree+i] - knots[i]) == 0) {
            alpha1 = rep(0,lenx);
        } else {
              alpha1 = wrap((x - knots[i])/(knots[degree+i] - knots[i]));
        }

        //alpha2
        if((knots[i+degree+1] - knots[i+1]) == 0) {
            alpha2 = rep(0,lenx);
        } else {
            alpha2 = wrap((knots[i+degree+1] - x)/(knots[i+degree+1] - knots[i+1]));
        }
                                                                                                          
         B = (alpha1 * basis(x, (degree - 1), i, knots, lenx)) 
                + (alpha2 * basis(x, (degree - 1), (i + 1), knots, lenx));                                                                                             
    }//else
                      
    return B;
} //basis

//change name from matrix to something more meaningful
RcppExport SEXP formMatrix(SEXP x_, SEXP degree_, SEXP knots_, SEXP k_, SEXP lenx_){
    
    //convert data types from R to C++
    int degree = as<int>(degree_), 
    k = as<int>(k_), lenx = as<int>(lenx_);
    //SEXP to NumericVector: http://dirk.eddelbuettel.com/code/rcpp/Rcpp-quickref.pdf
    NumericVector x(x_);
    NumericVector knots(knots_);
    
    //output variable allocation:
    NumericMatrix out(lenx,k);
 
    #pragma omp parallel for
    for(int j = 0; j < k; j++) { //R equivalent:  for(j in 1:k){
        //Reference the jth column; changes propagate to matrix
        NumericMatrix::Column jvector = out(_,j);

        #pragma omp critical
        jvector = basis(x, degree, j, knots, lenx);
    }//end for(j)

    return out;
} //end matrix function
