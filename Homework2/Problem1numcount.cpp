#include <iostream>
#include <vector>
#include <map>
#include <mpi.h>

using std::map;
using std::vector;

#define DATA_MSG 0
#define DATA_MSG_END 2

#define MAX_PATTERN 1000

//initialize global variables
int nnodes, //total nodes
    me, //my node number
    *xCopy, //global copy of x
    *globalArray, //array that we are passed
    *patterns, //array that we return
    patternFromWorker[MAX_PATTERN], //patter we get from worker
    arraySize, //size of the array we are passed
    patternSize; //sixe of the patterns

void init() {
    patterns = new int[1];
    patterns[0] = 0;

   //insert first part
   MPI_Init(NULL,NULL);
   //put number of nodes in nnodes
   MPI_Comm_size(MPI_COMM_WORLD,&nnodes);
   //put node number of this node in me
   MPI_Comm_rank(MPI_COMM_WORLD,&me);

}

void addToPatterns(map<vector<int>, int>& mapPatterns) {
    //convert to vector to store as key
    vector<int> tmp;
    for(int i = 0; i < patternSize + 1; i++) {
        tmp.push_back(patternFromWorker[i]);
    }

    //store in map
    mapPatterns[tmp]++;
}

//node 0: manager node - distribute and collect
void node0() {
    map<vector<int>, int> mapPatterns;

    int start = 0;
    MPI_Status status;
    int arrayBound = arraySize - patternSize + 1;

    //initialize workers
    for(int i = 1; i < nnodes; i++) {
        MPI_Send(globalArray+start, patternSize, MPI_INT, i, DATA_MSG, MPI_COMM_WORLD);
        start++;
    }

    //keep workers working
    vector<int> tmpPattern;
    while(start != arrayBound) {
        //send and recieve to worker
        MPI_Recv(patternFromWorker, patternSize+1, MPI_INT, MPI_ANY_SOURCE, DATA_MSG, MPI_COMM_WORLD, &status);

        addToPatterns(mapPatterns);

        MPI_Send(globalArray+start, patternSize, MPI_INT, status.MPI_SOURCE, DATA_MSG, MPI_COMM_WORLD);
        start++;
    }

    //receive all workers when there's no more work to be done
    for (int i = 1; i < nnodes; i++) {
        MPI_Recv(patternFromWorker, patternSize+1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        addToPatterns(mapPatterns);
    }

    //kill all workers once they've finished working
    for (int i = 1; i < nnodes; i++) {
        MPI_Send(globalArray+start, patternSize, MPI_INT, i, DATA_MSG_END, MPI_COMM_WORLD);
    }

    //build return array
    int *out = new int[(mapPatterns.size()*(patternSize+1))+1];
    int outPlace = 1;
    out[0] = mapPatterns.size();
    for(map<vector<int>, int>::iterator it = mapPatterns.begin(); it != mapPatterns.end(); ++it) {
        for(int i = 0; i < it->first.size(); i++) {
            out[outPlace] = it->first[i];
            outPlace++;
        } //end for loop 2
    } //end for loop 1

    patterns = out;
}

void findPatternCount(int* pattern) {
    for(int i = 0; i < arraySize - patternSize + 1; i++){
        for(int j = 0; j < patternSize; j++) {
            if(xCopy[i+j] != pattern[j]) break;
            if(j == patternSize - 1) pattern[patternSize]++;
        }

    }
}

void workerNode() {
    int patternLength;
    MPI_Status status;

    //get work from node0 until there's none left
    while(true) {
        MPI_Recv(globalArray, patternSize, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if(status.MPI_TAG == DATA_MSG_END) return;

        MPI_Get_count(&status,MPI_INT, &patternLength);

        //store the pattern and the count of the pattern
        for(int i = 0; i < patternLength; i++) patternFromWorker[i] = globalArray[i];

        //start with count at 0
        patternFromWorker[patternLength] = 0;
        findPatternCount(patternFromWorker);

        MPI_Send(patternFromWorker, patternSize+1, MPI_INT, 0, DATA_MSG, MPI_COMM_WORLD);
    }
}

//sharing memory -- sorta
void globalize(int* x, int n, int m) {
    globalArray = x;
    arraySize = n;
    patternSize = m;

    xCopy = new int[n];
    for(int i = 0; i < n; i++) {
        xCopy[i] = x[i];
    }
}

int *numcount(int *x, int n, int m) {
    globalize(x, n, m);
    init();

    //all nodes run this program - make each do different actions
    if(me==0) node0();
    else workerNode();

    MPI_Finalize();

    if(me == 0){
        return patterns;
    }
    else{
        int* y = new int[1];
        y[0] = 0;
        return NULL;
    }
}
