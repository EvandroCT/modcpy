/* GPU modular copy */

__global__ void modcpy(void *destination, void *source, size_t destination_size, size_t source_size){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *dp = (int*) destination;
	int *sp = (int*) source;
	int pos;

	for(int i=idx; i<destination_size/sizeof(int); i+=gridDim.x*blockDim.x){
		pos = i % (source_size/sizeof(int));
		dp[i] = sp[pos];		
	}
}
