/*
 * Author: Evandro C Taquary
 * Compilation: nvcc -arch=sm_35 gpu_replication.cu modcpy.cu -o gpu
 * 
 * */

#include <iostream>
#include <sys/time.h>
#include "modcpy.h"

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
	
	node tree[NODS];
	node *d_tree;

	size_t treeBytes = sizeof(tree);
	size_t repBytes = treeBytes * REPS;
	
	node *h_replics = (node*) malloc(repBytes);
	node *d_replic;

	int blockSize = device.warpSize*32;
	int gridSize = ((repBytes/sizeof(int4))/blockSize);
	dim3 grid = dim3(gridSize);
	dim3 block = dim3(blockSize);

	for(int i=0; i<NODS; i++)
	{
		tree[i].parent = i;
		tree[i].child1 = i;
		tree[i].child2 = i;
		tree[i].branch = i;
	}
	
	CHECK(cudaMalloc((void **) &d_tree, treeBytes));
	CHECK(cudaMemcpy(d_tree, &tree, treeBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void **) &d_replic, repBytes));
	
/**********************PARALLEL MEASUREMENT**********************/
	gettimeofday(&begin, NULL);
	modcpy<<<grid, block>>>(d_replic, d_tree, repBytes, treeBytes);
	CHECK(cudaDeviceSynchronize());
	gettimeofday(&end, NULL);
	time_spent = (double) (end.tv_usec - begin.tv_usec)/1000 + (end.tv_sec - begin.tv_sec)*1000;
	cout << "Time spent:\t" << time_spent << "ms " <<  endl;
/**********************PARALLEL MEASUREMENT**********************/		

	CHECK(cudaMemcpy(h_replics, d_replic, repBytes, cudaMemcpyDeviceToHost));

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
