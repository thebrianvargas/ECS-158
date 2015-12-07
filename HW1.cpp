#include <iostream>
#include <omp.h>

//prototypes
int* numcount(int *x, int n, int m);
bool portionMatchesPattern(int* mainArray, int* pattern, int portionStart, int portionEnd);
bool patternAlreadyFound(int* pattern);


int main(int argc, char **argv) {
    
    //test data
    int testArray[] = {3,4,5,12,13,4,5,12,4,5,6,3,4,5,13,4,5};
    
    int* returnArray = numcount(testArray, 17, 3);
}


//x = array, n = size of array, m = length of pattern to match
int* numcount(int* x, int n, int m) {
    int startIndex = 0; 
    int endIndex; 
    
    //array that holds the final patterns
    int* patterns = new int[1];
    patterns[0] = 0; //means no patterns were found
    int patternsSize = 0;
    
    #pragma omp parallel 
    {
        std::cout<<"just output to see if it's still running 0"<<std::endl;
        int currThread = omp_get_thread_num();
        int currStartIndex = startIndex++;
      
        //only run this part if within bounds of x
        if(currStartIndex < n - m) {
            int* currPortion = new int[m];
            for(int i = 0; i < m; i++) {
                currPortion[i] = x[startIndex + 1]; 
            }
            
            //check if pattern is in main array x
            //This loop goes through main array, and calls a function to check if the sub portion of the array matches the pattern
            for(int i = 0; i < n; i += m) {
                
                int portionStart = i; 
                int portionEnd = portionStart + m;
                if(portionMatchesPattern(x, currPortion, portionStart, portionEnd)) {
                    
                    //if a pattern is has not already been found, add it to return array
                    //else increment the count of the pattern foun in the return array
                    if(!patternAlreadyFound(currPortion)) {
std::cout<<"just output to see if it's still running 1"<<std::endl;
#pragma omp single
{
std::cout<<"in the single - thread: " << omp_get_thread_num()<<std::endl;
                        //increment the number of patterns found
std::cout<<"num patters -b: "<<patterns[0]<<std::endl;
                        patterns[0] = patterns[0]++;
std::cout<<"num patters -a: "<<patterns[0]<<std::endl;                        
                        
                        int* tempPatterns = new int[patternsSize + (m + 1)]; //add (m + 1) to store the count of appearance of the pattern
                        for(int j = 0; j <= m; j++) {
                            
                            //store the count of the appearance of pattern
                            if(j == m) {
                                tempPatterns[j] = 1;
                            } else {
                                tempPatterns[patternsSize + j] = currPortion[j];
                            }
                        }
                        
                        patterns = tempPatterns;
                        patternsSize += m;
                        delete [] tempPatterns;
std::cout<<"out of single 1/2"<<std::endl;
}
std::cout<<"out of single 2/2"<<std::endl;
                    } //else {
                        //for(int j = 0; j < patternsSize; j += (m + 1)) {
                          //still to do
                        //}
                    //}
                }
            }
            
            delete [] currPortion;
        } //if -- bounds check
        
    }
    std::cout<<"just output to see if it's still running 2"<<std::endl;
    //#pragma omp barrier
    
    std::cout<<"num patterns found: "<<patterns[0]<<std::endl;

    return patterns; 
} //numcount

bool portionMatchesPattern(int* mainArray, int* pattern, int portionStart, int portionEnd) {
    bool ret = false;
    
    int portionLength = portionEnd - portionStart;
    int* portion = new int[portionLength];
    for(int i = 0; i < portionLength; i++) {
        portion[i] = mainArray[portionStart + i];
    }
    
    for(int i = 0; i < portionLength; i++) {
        if(portion[i] == pattern[i]) {
            ret = true;
        } else{
            ret = false;
        }
    }
    
    delete [] portion;
    return ret;
}

bool patternAlreadyFound(int* pattern) {
    bool ret = false;
    
    return ret;
}
