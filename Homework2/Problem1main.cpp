#include <iostream>
#include <mpi.h>

using namespace std;

int *numcount(int *x, int n, int m);
void init();

int main(){
    
   int x[] = {3,4,5,12,13,4,5,12,4,5,6,3,4,5,13,4,5};
   int n = 17, m = 3;

   int * qq = numcount(x, n, m);
    // it's 49 for our example - different in general
    // however, printing the array is not what we care about


    if(qq != NULL) {
        for(int i=0;i<(qq[0]*(m+1));i++) cout << qq[i] << ",";
        cout << qq[qq[0]*(m+1)] << endl;
    }

} //end main
