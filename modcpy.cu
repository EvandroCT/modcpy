/* GPU modular copy */

__global__ void modcpy(void *destination, void *source, size_t destination_size, size_t source_size){

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int pos;
 
	int ds = destination_size/sizeof(int4), ss = source_size/sizeof(int4);
	for(int i = idx; i < ds; i += gridDim.x * blockDim.x){
		pos = i % ss;
		reinterpret_cast<int4*>(destination)[i] = reinterpret_cast<int4*>(source)[pos];  
	}
}
