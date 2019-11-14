#define BLOCK_SIZE 512

__global__ void naiveReductionKernel(float *out, float *in, unsigned size)
{
    /********************************************************************
    Implement the naive reduction you learned in class.
    ********************************************************************/
    __shared__ float partialSum[2 * BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    unsigned int a = start + t;
    unsigned int b = blockDim.x + t;
    
    if (a < size)
        partialSum[t] = in[a];
    else
        partialSum[t] = 0.0;
    
    if (start + b < size)
        partialSum[b] = in[start + b];
    else
        partialSum[b] = 0.0;
    
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (t % stride == 0)
            partialSum[2 * t] += partialSum[2 * t + stride];
    }

    if (t == 0)
        out[blockIdx.x] = partialSum[0];
}

void naiveReducion(float *out, float *in_h, unsigned size)
{
    cudaError_t cuda_ret;    

    //int i;
    float *out_h, *in_d, *out_d;

    unsigned out_size = size / (BLOCK_SIZE<<1);
    if(size % (BLOCK_SIZE<<1)) out_size++;

    // Allocate Host memory ---------------------------------------------------
    out_h = (float*)malloc(out_size * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host in naive reduction.");

    // Allocate device variables ----------------------------------------------
    cuda_ret = cudaMalloc((void**)&in_d, size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in naive reduction.");

    cuda_ret = cudaMalloc((void**)&out_d, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in naive reduction.");

    // Copy host variables to device ------------------------------------------
    cuda_ret = cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device in naive reduction.");

    cuda_ret = cudaMemset(out_d, 0, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory in naive reduction.");
    
    // Launch kernel ----------------------------------------------------------
    dim3 dim_grid, dim_block;


    /********************************************************************
    Change dim_block & dim_grid.
    ********************************************************************/
    dim_block.x = BLOCK_SIZE; 
    dim_block.y = 1;
    dim_block.z = 1;
    
    dim_grid.x = out_size; 
    dim_grid.y = 1;
    dim_grid.z = 1;


    naiveReductionKernel<<<dim_grid, dim_block>>>(out_d, in_d, size);

    // Copy device variables from host ----------------------------------------
    cuda_ret = cudaMemcpy(out_h, out_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in naive reduction.");


    /********************************************************************
    For bonus implement this part on GPU. (2.5pts)
    HINT: Think recursively!!
    ********************************************************************/
    /* Accumulate partial sums on host */
    /*out[0] = 0;
    for( i = 0; i < out_size; i++ ) {
        out[0] += out_h[i];
    }*/
	if (out_size > 1)
        naiveReducion(out, out_h, out_size);
    else
        out[0] = out_h[0];
    
    // Free memory ------------------------------------------------------------
    free(out_h);
    cudaFree(in_d);
    cudaFree(out_d);
}

__global__ void improvedReductionKernel(float *out, float *in, unsigned size)
{
    /********************************************************************
    Implement the better reduction you learned in class.
    ********************************************************************/
    __shared__ float partialSum[2 * BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    unsigned int a = start + t;
    unsigned int b = blockDim.x + t;
    
    if (a < size)
        partialSum[t] = in[a];
    else
        partialSum[t] = 0.0;
    
    if (start + b < size)
        partialSum[b] = in[start + b];
    else
        partialSum[b] = 0.0;
    
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }

    if (t == 0)
        out[blockIdx.x] = partialSum[0];
    
}

void improvedReducion(float *out, float *in_h, unsigned size)
{
    cudaError_t cuda_ret;

    //int i;
    float *out_h, *in_d, *out_d;

    unsigned out_size = size / (BLOCK_SIZE<<1);
    if(size % (BLOCK_SIZE<<1)) out_size++;

    // Allocate Host memory ---------------------------------------------------
    out_h = (float*)malloc(out_size * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host in improved reduction.");

    // Allocate device variables ----------------------------------------------
    cuda_ret = cudaMalloc((void**)&in_d, size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in improved reduction.");

    cuda_ret = cudaMalloc((void**)&out_d, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in improved reduction.");

    // Copy host variables to device ------------------------------------------
    cuda_ret = cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device in improved reduction.");

    cuda_ret = cudaMemset(out_d, 0, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory in improved reduction.");
    
    // Launch kernel ----------------------------------------------------------
    dim3 dim_grid, dim_block;


    /********************************************************************
    Change dim_block & dim_grid.
    ********************************************************************/
    dim_block.x = BLOCK_SIZE; 
    dim_block.y = 1;
    dim_block.z = 1;
    
    dim_grid.x = out_size; 
    dim_grid.y = 1;
    dim_grid.z = 1;


    improvedReductionKernel<<<dim_grid, dim_block>>>(out_d, in_d, size);

    // Copy device variables from host ----------------------------------------
    cuda_ret = cudaMemcpy(out_h, out_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in improved reduction.");

    /********************************************************************
    For bonus implement this part on GPU. (2.5pts)
    HINT: This is exactly like naiveReduction so free bonus point!!
    ********************************************************************/
    /* Accumulate partial sums on host */
    /*out[0] = 0;
    for( i = 0; i < out_size; i++ )
        out[0] += out_h[i];*/
	if (out_size > 1)
        improvedReducion(out, out_h, out_size);
    else
        out[0] = out_h[0];

    // Free memory ------------------------------------------------------------
    free(out_h);
    cudaFree(in_d);
    cudaFree(out_d);
}

__global__ void ScanKernel(float *out, float *in, unsigned size)
{
    /********************************************************************
    Implement the scan algorithm you learned in class.
    ********************************************************************/
    
    __shared__ float T[BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int a = blockIdx.x * blockDim.x + t;
    
    if (a < size)
        T[t] = in[a];
    else
        T[t] = 0.0;

    int stride = 1;
    
    while(stride < BLOCK_SIZE)
    {
        __syncthreads();
		int index = (t + 1) * stride * 2 - 1;
        if(index < BLOCK_SIZE)
            T[index] += T[index - stride];
        stride = stride * 2;
    }
    
    stride = BLOCK_SIZE / 4;
    while(stride > 0)
    {
        __syncthreads();
		int index = (t + 1) * stride * 2 - 1;
        if(index + stride < BLOCK_SIZE)
            T[index + stride] += T[index];
        stride = stride / 2;
    }
    __syncthreads();
	if (a < size) 
        out[a] = T[t];
    
}

void Scan(float *out_h, float *in_h, unsigned size)
{
    cudaError_t cuda_ret;

    int i;
    float *in_d, *out_d;

    unsigned out_size = size;

    // Allocate device variables ----------------------------------------------
    cuda_ret = cudaMalloc((void**)&in_d, size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in scan.");

    cuda_ret = cudaMalloc((void**)&out_d, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in scan.");

    // Copy host variables to device ------------------------------------------
    cuda_ret = cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device in scan");

    cuda_ret = cudaMemset(out_d, 0, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory in scan");
    
    // Launch kernel ----------------------------------------------------------
    dim3 dim_grid, dim_block;

    /********************************************************************
    Change dim_block & dim_grid.
    ********************************************************************/
    dim_block.x = BLOCK_SIZE; 
    dim_block.y = 1;
    dim_block.z = 1;
    
    dim_grid.x = ceil((float)size/BLOCK_SIZE); 
    dim_grid.y = 1;
    dim_grid.z = 1;


    ScanKernel<<<dim_grid, dim_block>>>(out_d, in_d, size);

    // Copy device variables from host ----------------------------------------
    cuda_ret = cudaMemcpy(out_h, out_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in scan.");

    /********************************************************************
    For bonus implement this part on GPU. (5pts)
    HINT: This one is a little bit harder. You should write another
          GPU kernel.
    ********************************************************************/
    /* Accumulate partial sums on host */
    int c = 0;
    for( i = BLOCK_SIZE; i < out_size; i++ ) {
        if( i > ((c+1)*BLOCK_SIZE-1) ) {
            c++;
            out_h[i] += out_h[c*BLOCK_SIZE-1];
        } else {
            out_h[i] += out_h[c*BLOCK_SIZE-1];
        }
    }

    // Free memory ------------------------------------------------------------
    cudaFree(in_d);
    cudaFree(out_d);
}

