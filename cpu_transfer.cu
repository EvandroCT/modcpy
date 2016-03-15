/*
 * Author: Evandro C Taquary
 * Compilation: nvcc -arch=sm_35 cpu_transfer.cu -o cpu
 * 
 * */

#include <iostream>
#include <sys/time.h>

using namespace std;

#define CHECK(call) \
{ \
        const cudaError_t error = call; \
        if (error != cudaSuccess) \
        { \
                cout << "Error: " << __FILE__ ": " << __LINE__ << ", "; \
                cout << "code: "<< error << ", reason: " << cudaGetErrorString(error) << endl; \
                exit(EXIT_FAILURE); \
        } \
}

typedef struct {
	short parent;
	short child1;
	short child2;
	double branch;
}node;

int main(int argc, char *argv[])
{	
	if(argc != 3){
		cout << "Usage: " << argv[0] << " #nodes #replications" << endl;
		exit(EXIT_FAILURE);
	}
	
	const int NODS = atoi(argv[1]);
	const int REPS = atoi(argv[2]);
	
	struct timeval begin, end;
	double time_spent;
	
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);	

	size_t repBytes = NODS * REPS * sizeof(node);
	
	node *h_replics = (node*) malloc(repBytes);
	node *d_replics;

	for(int i=0; i<REPS; i++)
		for(int j=0; j<NODS; j++)
		{
			h_replics[i*NODS+j].parent = j;
			h_replics[i*NODS+j].child1 = j;
			h_replics[i*NODS+j].child2 = j;
			h_replics[i*NODS+j].branch = j;
		}
	
	CHECK(cudaMalloc((void **) &d_replics, repBytes));
	
/**********************SERIAL MEASUREMENT**********************/
	gettimeofday(&begin, NULL);
	CHECK(cudaMemcpy(d_replics, h_replics, repBytes, cudaMemcpyHostToDevice));
	gettimeofday(&end, NULL);
	time_spent = (double) (end.tv_usec-begin.tv_usec)/1000 + (end.tv_sec-begin.tv_sec)*1000;
	cout << "Time spent:\t" << time_spent << "ms " <<  endl;
/**********************SERIAL MEASUREMENT**********************/
	
	for(int i=0; i<REPS; i++)
		for(int j=0; j<NODS; j++)
		{
			h_replics[i*NODS+j].parent = 0;
			h_replics[i*NODS+j].child1 = 0;
			h_replics[i*NODS+j].child2 = 0;
			h_replics[i*NODS+j].branch = 0;
		}
		
	CHECK(cudaMemcpy(h_replics, d_replics, repBytes, cudaMemcpyDeviceToHost));

	for(int i=0; i<REPS; i++)
		for(int j=0; j<NODS; j++)		
			if(	h_replics[i*NODS+j].parent != j ||
				h_replics[i*NODS+j].child1 != j ||
				h_replics[i*NODS+j].child2 != j ||
				h_replics[i*NODS+j].branch != j )
				{ 	
					cout << "Data doesn't match!" << endl;
					exit(1); 
				}
	cout << "Data does match!" << endl;
	cudaDeviceReset();	
  exit(EXIT_SUCCESS);
}
