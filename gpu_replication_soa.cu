/*************************************************************************
	
	Copyright (C) 2016	Evandro Taquary
	
	This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
	
*************************************************************************/

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

int main(int argc, char *argv[])
{
	
	cudaSetDevice(0);
	if(argc != 3){
		cout << "Usage: " << argv[0] << " #nodes #replications" << endl;
		exit(EXIT_FAILURE);
	}
	
	const int NODS = atoi(argv[1]);
	const int REPS = atoi(argv[2]);	
	
	size_t treeBytes = 3*sizeof(short)*NODS + sizeof(double)*NODS;
	int r = treeBytes%sizeof(int4);	
	treeBytes += r ? sizeof(int4)-r : 0;		
	size_t repBytes = treeBytes * REPS;
	
	void *d_tree;
	void *h_tree = (void*) malloc (treeBytes);
	memset(h_tree,0,treeBytes);
	short *parent = (short*) h_tree;
	short *child1 = parent+NODS;
	short *child2 = child1+NODS;
	double *branch = (double*) (child2+NODS);	
	
	for(short i=0; i<NODS; i++){
		parent[i] = i;
		child1[i] = i;
		child2[i] = i;
		branch[i] = i;
	}	
	
	void *h_replics = (void*) malloc(repBytes);
	void *d_replics;

	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	
	int blockSize = device.warpSize*32;
	int gridSize = (repBytes/sizeof(int4) + (blockSize-1)) / blockSize;
	dim3 grid = dim3(gridSize);
	dim3 block = dim3(blockSize);

	CHECK(cudaMalloc((void**) &d_tree, treeBytes));
	CHECK(cudaMemcpy(d_tree, h_tree, treeBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void **) &d_replics, repBytes));

	struct timeval begin, end;
	float time_spent;
	
/*******************************PARALLEL MEASUREMENT*******************************/
	
	gettimeofday(&begin, NULL);
	modcpy<<<grid, block>>>(d_replics, d_tree, repBytes, treeBytes);
	CHECK(cudaDeviceSynchronize());
	gettimeofday(&end, NULL);
	time_spent = (float) (end.tv_usec - begin.tv_usec)/1000 + (end.tv_sec - begin.tv_sec)*1000;
	cout << "Time spent:\t" << time_spent << "ms " <<  endl;
	
/*******************************PARALLEL MEASUREMENT*******************************/		

	CHECK(cudaMemcpy(h_replics, d_replics, repBytes, cudaMemcpyDeviceToHost));

	for(int i=0; i<REPS; i++){
		parent = (short*) (h_replics+treeBytes*i);
		child1 = parent+NODS;
		child2 = child1+NODS;
		branch = (double*) (child2+NODS);		
		for(int j=0; j<NODS; j++)
		{
			if(parent[j] != j || child1[j] != j || child2[j] != j || branch[j] != j ){
				cout << "Data doesn't match!" << endl;
				exit(EXIT_FAILURE);
			}
		}
	}	
	cout << "Data does match!" << endl;
	cudaDeviceReset();	
	exit(EXIT_SUCCESS);
}
