/* GPU modular copy */

__global__ void modcpy(void *destination, void *source, size_t destination_size, size_t source_size){

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int pos;
 
	for(int i = idx; i < destination_size/sizeof(int4); i += gridDim.x * blockDim.x){
		pos = i % (source_size / sizeof(int4));
		reinterpret_cast<int4*>(destination)[i] = reinterpret_cast<int4*>(source)[pos];  
	}
}
