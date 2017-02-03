#include <thrust/device_vector.h>
//#include<iostream>

//using std::cout;
//using std::endl;

//functor
struct Smooth {
    //starting points of x, y, and m
    thrust::device_vector<float>::iterator dmIt;
    thrust::device_vector<float>::iterator dxIt;
    thrust::device_vector<float>::iterator dyIt;
    int n;
    float  h;

    //will be the arrays for x, y, and m after converted 
    //from the iterators
    float *x, *y, *m;

    //constructor for Smooth
    Smooth(thrust::device_vector<float>::iterator dm,
              thrust::device_vector<float>::iterator dx,
              thrust::device_vector<float>::iterator dy,
              int _n, float _h):
              dmIt(dm), dxIt(dx), dyIt(dy), n(_n), h(_h) {

                  //convert iterators to arrays so we can use brackets []
                  m = thrust::raw_pointer_cast(&dm[0]);
                  x = thrust::raw_pointer_cast(&dx[0]);
                  y = thrust::raw_pointer_cast(&dy[0]);
              };

    __device__ float myAbs(float value) {
        return (value < 0) ? (value * -1) : value;
    } //end abs function

    //overloaded operator function called implicity from for_each call
    __device__
    void operator() (const int me) {
        float xi = x[me];
        float sum = 0;
        float count = 0;

         for(int j=0; j<n; j++){
             //now iterate through the j values
             float xj = x[j]; //not needed - used for better visibility
             if(myAbs(xj-xi) < h){
                 sum += y[j];
                 count++;
             } //end abs if condition
         } //end j for loop

         //store the average of me in m
         m[me] = sum / count;
    }
};

void smootht(float *x, float *y, float *m, int n, float h){

    //copy data to the device
    thrust::device_vector<float> dx(x,x+n);
    thrust::device_vector<float> dy(y,y+n);

    //setup output vector
    thrust::device_vector<float> dm(n);

    //sequence iterators to go through for_each
    thrust::counting_iterator<float> seqb(0);
    thrust::counting_iterator<float> seqe = seqb + n;

    //loop through x, y, and m and find averages on device
    thrust::for_each(seqb, seqe, Smooth(dm.begin(), dx.begin(), dy.begin(), n, h));

    //copy averages from device to m
    thrust::copy(dm.begin(), dm.end(), m);
}

/*
int main(void) {

    int n = 600000;
    float x[n];
    float y[n];
    float h = 0.1;
    float xcount = 15000.0;
    //float ycount = 15.0;
    for(int i = 0; i < n; i++) {
        x[i] = y[i] = xcount;
        xcount -= 0.0001;
    }

     //return array
     float m[n];

     smootht(x, y, m, n, h);

     for(int i = 0; i < n; i++) {
         cout<<m[i]<<" ";
     } cout<<endl;
} 
*/
