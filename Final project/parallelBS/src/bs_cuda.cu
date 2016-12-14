#include <Rcpp.h>
#include <cuda.h>

using namespace Rcpp;
//indent = 4 spaces

//CUDA function
__global__ void degree0(float *B, float *knots, float *x, int i) {
		
		//current thread
		int me = blockIdx.x * blockDim.x + threadIdx.x;
		B[me] = ((x[me] >= knots[i]) && (x[me] < knots[i+1])) ? 1 : 0;
}

NumericVector c_basis(NumericVector x, int degree, int i, NumericVector knots, int lenx){

    NumericVector B(lenx); // output variable
    NumericVector alpha1(lenx), alpha2(lenx);
         
     //NumericVector B, alpha1, alpha2; 
    if(degree==0) {
        
			//blocks and threads for GPU
				int nthreads = min(lenx, 500);
				int nblocks = ceil(lenx/nthreads);

				//set up grid/block dimensions
				dim3 dimGrid(nblocks,1);
				dim3 dimBlock(nthreads,1,1);

				float *dB, *dknots, *dx;
			
				//copy to the GPU
				cudaMemcpy(dB,B,lenx,cudaMemcpyHostToDevice);
				cudaMemcpy(dknots, knots, knots.size(), cudaMemcpyHostToDevice);
				cudaMemcpy(dx, x, lenx, cudaMemcpyHostToDevice);
		
				degree0<<<dimGrid,dimBlock>>>(dB,dknots,dx,i);

				//copy back to B
				cudaMemcpy(B,dB,lenx,cudaMemcpyDeviceToHost);


				B = wrap(ifelse((x >= knots[i]) & (x < knots[i+1]), 1, 0));
    } //end if
    else {
        if((knots[degree+i] - knots[i]) == 0) {
            alpha1 = rep(0,lenx);
        }
        else {
              alpha1 = wrap((x - knots[i])/(knots[degree+i] - knots[i]));
        }

        if((knots[i+degree+1] - knots[i+1]) == 0) {
            alpha2 = rep(0,lenx);
        }//end if
        else {
            alpha2 = wrap((knots[i+degree+1] - x)/(knots[i+degree+1] - knots[i+1]));
        }//end else
                                                                                                          
         B = (alpha1 * c_basis(x, (degree - 1), i, knots, lenx)) 
                + (alpha2 * c_basis(x, (degree - 1), (i + 1), knots, lenx));
                                                                                             
    }//end else                  
    return B;
} //end basis function

//change name from matrix to something more meaningful
RcppExport SEXP c_formMatrix(SEXP x_, SEXP degree_, SEXP knots_, SEXP k_, SEXP lenx_){
    
    //convert data types from R to C++
    int degree = as<int>(degree_), 
    k = as<int>(k_), lenx = as<int>(lenx_);
    //SEXP to NumericVector: http://dirk.eddelbuettel.com/code/rcpp/Rcpp-quickref.pdf
    NumericVector x(x_);
    NumericVector knots(knots_);
    
    //output variable allocation:
    NumericMatrix out(lenx,k);
 			
		//blocks and threads for GPU
				int nthreads = min(lenx, 500);
				int nblocks = ceil(lenx/nthreads);

		//set up grid/block dimensions
				dim3 dimGrid(nblocks,1);
				dim3 dimBlock(nthreads,1,1);

				float *dB, *dknots, *dx;
				
				//make space on GPU
				cudaMalloc((void **) &dB,lenx*sizeof(float));
				cudaMalloc((void **) &dknots, knots.size()*sizeof(float));
				cudaMalloc((void **) &dx, lenx*sizeof(float));
	

    for(int j = 0; j < k; j++) { //R equivalent:  for(j in 1:k){
        //Reference the jth column; changes propagate to matrix
        NumericMatrix::Column jvector = out(_,j);

        jvector = c_basis(x, degree, j, knots, lenx);
    }//end for(j)

				cudaFree(dB);
				cudaFree(dknots);
				cudaFree(dx);				


    return out;
} //end matrix function
