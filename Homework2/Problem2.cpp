//c++ interface for mandelbrot set computation

#include <Rcpp.h>
#include <omp.h>

#include <complex>
#include <string>

using namespace Rcpp;

NumericVector seqcpp(float a, int size, float inc){
   //R's equivalent of the seq function
   NumericVector out(size); //allocate the right size
   for(int i=0; i<size; i++)
      out[i] = a+i*inc; //where a is either xl or yb
   return out;
}

int inset(std::complex<double> c, int maxiters){
   //adapted from ECS158 Textbook (Matloff)
   //tests whether c is in mandelbrot set after maxiters iterations
   int iters;
   float rl,im;
   std::complex<double> z = c; // initialization
   
   for(iters=0; iters < maxiters; iters++){
      z = z*z+c;
      rl = std::real(z);
      im = std::imag(z);
      if(rl*rl+im*im > 4) return 0; // check if |z|<=2
   }
   return 1;
} //end inset

RcppExport SEXP cmandel(SEXP nth_, SEXP xl_, SEXP xr_, SEXP yb_, SEXP yt_,
                        SEXP inc_, SEXP maxiters_, SEXP sched_, 
                        SEXP chunksize_){
   //adapted from ECS158 Quiz 2
   //convert inputs from R data types into C++ data types
   int nth = as<int>(nth_), maxiters = as<int>(maxiters_),
       chunksize = as<int>(chunksize_);
   float xl = as<float>(xl_), xr = as<float>(xr_),
         yb = as<float>(yb_), yt = as<float>(yt_),
         inc = as<float>(inc_);
   std::string sched = as<std::string>(sched_);

   int nxticks = (xr-xl)/inc+1;
   int nyticks = (yt-yb)/inc+1;
   NumericVector xticks = seqcpp(xl, nxticks, inc);
   NumericVector yticks = seqcpp(yb, nyticks, inc);
   //Eventual return value
   NumericMatrix m(nxticks,nyticks);

   #pragma omp parallel 
   {
      omp_set_num_threads(nth);

      if(sched.compare("static")) omp_set_schedule(omp_sched_static, chunksize);
      else if(sched.compare("dynamic")) omp_set_schedule(omp_sched_dynamic, chunksize);
      else if(sched.compare("auto")) omp_set_schedule(omp_sched_auto, chunksize);
      else omp_set_schedule(omp_sched_guided, chunksize);
      #pragma omp for

      //iterate through grid, set c to each grid point and see where z belongs
      //NOTE: this is probably where we will begin parallelizing?
      for(int i=0; i<nxticks; i++){
         float xti = xticks[i];
         for(int j=0; j<nyticks; j++){
            float ytj = yticks[j];
            std::complex<float> c (xti,ytj);
            m(i,j) = inset(c,maxiters);
         } //end j loop
      } //end i loop
      //once this is over, we have a matrix of 0 and 1, the "mandelbrot matrix"

   }// omp parrallel

   return m;
}//cmandel
