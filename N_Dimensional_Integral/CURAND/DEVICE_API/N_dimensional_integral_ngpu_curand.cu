#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_erf.h>

__device__ float Integrand (int, float *);
__device__ float Heat_Bath_Linear (int, double, double, float*);
__device__ float Heat_Bath_Exp (int, double, double, float*);
__device__ float Heat_Bath_Gaussian (int, double, double, float*);
__device__ float Weight_Linear (int, double, double, float *);
__device__ float Weight_Exp (int, double, double, float *);
__device__ float Weight_Gaussian (int, double, double, float *);

__global__ void SET_UP_KERNEL( long seed, curandState *state )
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init (seed, index, 0, &state[index]);
}

__global__ void SIMPLE_SAMPLING( int N_sample, int N_dimension, double *I_block, curandState *state )
{
	extern __shared__ double cache[];
	
	int cache_index, shift, layer;
	float x, temp;
	double I_sum;
	long index, N;

	index = threadIdx.x + blockIdx.x*blockDim.x;
	N = (long)(N_sample)*gridDim.x;  //gridDim.x = N_time/N_GPU
	cache_index = threadIdx.x;
	shift = blockDim.x*gridDim.x;
	I_sum = 0.0;
	curandState local = state[index];

	while ( index<N )
	{
		temp = 0.0;
		for (int i=0; i<N_dimension; i++)
		{
			x = curand_uniform(&local);
			temp += x*x;
		}
		temp = 1.0/(temp+1.0);
		I_sum += temp;
		index += shift;
	}
//	state[threadIdx.x+blockIdx.x*blockDim.x] = local;  // Don't use "index" since it has been shifted!

	cache[cache_index] = I_sum;
	layer = blockDim.x/2;
	__syncthreads();	

	while ( layer!=0 )
	{
		if (cache_index<layer)
			cache[cache_index] += cache[cache_index+layer];
		layer /= 2;
		__syncthreads();
	}

	if ( cache_index==0 )
	{
		I_block[blockIdx.x] = (double)(cache[0]/N_sample);
//		printf("%.6f\n", I_block[blockIdx.x]);
	}
}

__global__ void HEAT_BATH( char mode, int N_sample, int N_dimension, double C, double alpha, double *I_block, curandState *state )
{
	extern __shared__ double cache[];
	int cache_index, shift, layer;
	double temp;
	float *array_rand;
	long index, N;

	index = threadIdx.x + blockIdx.x*blockDim.x;
	N = long(N_sample)*gridDim.x;  //gridDim.x = N_time/N_GPU
	cache_index = threadIdx.x;
	shift = blockDim.x*gridDim.x;
	temp = 0.0;
	array_rand = (float*)malloc(N_dimension*sizeof(float));
	curandState local = state[index];
	
	if (mode=='l')
	{
		while (index<N)
		{
			for (int i=0; i<N_dimension; i++)
			{
				array_rand[i] = curand_uniform(&local);
			}
			temp += Heat_Bath_Linear(N_dimension, C, alpha, array_rand);
			index += shift;
		}
	}
	else if (mode=='e')
	{
		while (index<N)
		{
			for (int i=0; i<N_dimension; i++)
			{
				array_rand[i] = curand_uniform(&local);
			}
			temp += Heat_Bath_Exp(N_dimension, C, alpha, array_rand);
			index += shift;
		}
	}
	else
	{
		while (index<N)
		{
		for (int i=0; i<N_dimension; i++)
			{
				array_rand[i] = curand_uniform(&local);
			}
			temp += Heat_Bath_Gaussian(N_dimension, C, alpha, array_rand);
			index += shift;
		}
	}
//	state[threadIdx.x+blockIdx.x*blockDim.x] = local;  // Don't use "index" since it has been shifted!

	cache[cache_index] = temp;
	layer = blockDim.x/2;
	__syncthreads();	

	while ( layer!=0 )
	{
		if (cache_index<layer)
			cache[cache_index] += cache[cache_index+layer];
		layer /= 2;
		__syncthreads();
	}

	if ( cache_index==0 )
	{
		I_block[blockIdx.x] = (double)(cache[0]/N_sample);
//		printf("%.6f\n", I_block[blockIdx.x]);
	}
}

__global__ void IMPORTANT_SAMPLING(char mode, int N_sample, int N_time, int N_dimension, int N_thermal, int N_measure, double C, double alpha, double *I_block, curandState *state )
{
	int index, shift, count, total_sample;
	float weight_old, weight_new;
	float *array_rand_old, *array_rand_new;
	double I_sum;
	index = threadIdx.x + blockIdx.x*blockDim.x;
	shift = blockDim.x*gridDim.x;
	array_rand_old = (float*)malloc(N_dimension*sizeof(float));
	array_rand_new = (float*)malloc(N_dimension*sizeof(float));
	curandState local = state[index];
	
	if (mode=='l')
	{
		while (index<N_time)
		{
			I_sum = 0.0;
			for (int i=0; i<N_dimension; i++)
				array_rand_old[i] = curand_uniform(&local);
			weight_old = Weight_Linear(N_dimension, C, alpha, array_rand_old);

			for (int i=0; i<N_thermal; i++)
			{
				for (int j=0; j<N_dimension; j++)
					array_rand_new[j] = curand_uniform(&local);
				weight_new = Weight_Linear(N_dimension, C, alpha, array_rand_new);
				if (weight_new>weight_old || curand_uniform(&local)<weight_new/weight_old)
				{
					weight_old = weight_new;
					for (int j=0; j<N_dimension; j++)
						array_rand_old[j] = array_rand_new[j];
				}
			}
			count = 0;
			total_sample = 0;
			while (total_sample!=N_sample)
			{
				if (count%N_measure==0)
				{
					I_sum += Integrand(N_dimension, array_rand_old)/weight_old; 
					total_sample += 1;
					count = 1;
				}
				else
					count += 1;
				for (int j=0; j<N_dimension; j++)
					array_rand_new[j] = curand_uniform(&local);
				weight_new = Weight_Linear(N_dimension, C, alpha, array_rand_new);
				if (weight_new>weight_old || curand_uniform(&local)<weight_new/weight_old)
				{
					weight_old = weight_new;
					for (int j=0; j<N_dimension; j++)
						array_rand_old[j] = array_rand_new[j];
				}				
			}
			I_block[index] = I_sum/N_sample;
			index += shift;
		}
	}
	else if (mode=='e')
	{
		while (index<N_time)
		{
			I_sum = 0.0;
			for (int i=0; i<N_dimension; i++)
				array_rand_old[i] = curand_uniform(&local);
			weight_old = Weight_Exp(N_dimension, C, alpha, array_rand_old);

			for (int i=0; i<N_thermal; i++)
			{
				for (int j=0; j<N_dimension; j++)
					array_rand_new[j] = curand_uniform(&local);
				weight_new = Weight_Exp(N_dimension, C, alpha, array_rand_new);
				if (weight_new>weight_old || curand_uniform(&local)<weight_new/weight_old)
				{
					weight_old = weight_new;
					for (int j=0; j<N_dimension; j++)
						array_rand_old[j] = array_rand_new[j];
				}
			}
			count = 0;
			total_sample = 0;
			while (total_sample!=N_sample)
			{
				if (count%N_measure==0)
				{
					I_sum += Integrand(N_dimension, array_rand_old)/weight_old; 
					total_sample += 1;
					count = 1;
				}
				else
					count += 1;
				for (int j=0; j<N_dimension; j++)
					array_rand_new[j] = curand_uniform(&local);
				weight_new = Weight_Exp(N_dimension, C, alpha, array_rand_new);
				if (weight_new>weight_old || curand_uniform(&local)<weight_new/weight_old)
				{
					weight_old = weight_new;
					for (int j=0; j<N_dimension; j++)
						array_rand_old[j] = array_rand_new[j];
				}				
			}
			I_block[index] = I_sum/N_sample;
			index += shift;
		}
	}
	else
	{
		while (index<N_time)
		{
			I_sum = 0.0;
			for (int i=0; i<N_dimension; i++)
				array_rand_old[i] = curand_uniform(&local);
			weight_old = Weight_Gaussian(N_dimension, C, alpha, array_rand_old);

			for (int i=0; i<N_thermal; i++)
			{
				for (int j=0; j<N_dimension; j++)
					array_rand_new[j] = curand_uniform(&local);
				weight_new = Weight_Gaussian(N_dimension, C, alpha, array_rand_new);
				if (weight_new>weight_old || curand_uniform(&local)<weight_new/weight_old)
				{
					weight_old = weight_new;
					for (int j=0; j<N_dimension; j++)
						array_rand_old[j] = array_rand_new[j];
				}
			}
			count = 0;
			total_sample = 0;
			while (total_sample!=N_sample)
			{
				if (count%N_measure==0)
				{
					I_sum += Integrand(N_dimension, array_rand_old)/weight_old; 
					total_sample += 1;
					count = 1;
				}
				else
					count += 1;
				for (int j=0; j<N_dimension; j++)
					array_rand_new[j] = curand_uniform(&local);
				weight_new = Weight_Gaussian(N_dimension, C, alpha, array_rand_new);
				if (weight_new>weight_old || curand_uniform(&local)<weight_new/weight_old)
				{
					weight_old = weight_new;
					for (int j=0; j<N_dimension; j++)
						array_rand_old[j] = array_rand_new[j];
				}				
			}
			I_block[index] = (double)(I_sum/N_sample);
			index += shift;
		}
	}

	free(array_rand_old);
	free(array_rand_new);
//	state[threadIdx.x+blockIdx.x*blockDim.x] = local;  // Don't use "index" since it has been shifted!
}

__global__ void DATA_ANALYSIS( int N, double *I, double *ans_mean, double *ans_sigma )
{
	extern __shared__ double cache[];
	
	int index, cache_index, shift, layer;
	double mean, sigma;

	index = threadIdx.x + blockIdx.x*blockDim.x;
	cache_index = threadIdx.x;
	shift = blockDim.x*gridDim.x;
	mean = 0.0;
	sigma = 0.0;

	while ( index<N )
	{
		mean += I[index];
		sigma += I[index]*I[index];
		index += shift;
	}
	
	shift = blockDim.x;
	layer = blockDim.x/2;
	cache[cache_index] = mean;
	cache[cache_index+shift] = sigma;
	__syncthreads();
	
	while (layer!=0)
	{
		if (cache_index<layer)
		{
			cache[cache_index] += cache[cache_index+layer];
			cache[cache_index+shift] += cache[cache_index+shift+layer];
		}
		layer /= 2;
		__syncthreads();
	}

	if ( cache_index==0 )
	{
		ans_mean[blockIdx.x] = cache[0];
		ans_sigma[blockIdx.x] = cache[shift];
//		printf("Device: %.6f %.6f\n", ans_mean[blockIdx.x], ans_sigma[blockIdx.x]);
	}
}

int main(void)
{
	char mode;
	int N_GPU, tx, bx, cpu_thread_id, size_block, size_shared, size;
	int m, N_sample, N_dimension, N_time;
	int N_thermal, N_measure;
	float processing_time;
	double C, alpha; // Use weight function w(xi) = C*exp(-alpha*xi^2), where C is the normalized factor
	double mean, sigma;
	int *gid;
	double *h_mean, *h_sigma;
	double *sum_mean, *sum_sigma;
	long *seed;
	cudaEvent_t start, stop;

	puts("Use multi-GPU to calculate the N dimensional integral I( x1,x2,...,xN ) = integral(1/( 1+x1^2+x2^2+x3^2+...+xN^2 )dx1dx2dx3...dxN) by simple sampling and important sampling.\n");

	puts("Set the number of GPU.");
	scanf("%d", &N_GPU);
	printf("The number of GPU is %d .\n", N_GPU);

	gid = (int*)malloc(N_GPU*sizeof(int));

	puts("Set the GPU ID for each GPU.");
	for ( int i=0; i<N_GPU; i++ )
	{
		printf("Enter the GPU ID for #%d GPU.\n", i+1);
		scanf("%d", &gid[i]);
		printf("The GPU ID fot #%d GPU is %d .\n", i+1, gid[i]);
	}

	puts("Enter the dimension N .");
	scanf("%d", &N_dimension);
	if (N_dimension>0)
		printf("The dimension is %d .\n", N_dimension);
	else
	{
		puts("Dimension must be positive integer! Exit!");
		exit(0);
	}
	puts("Enter an integer m such that the number of sampling points N_sample = 2^m .");
	scanf("%d", &m);
	N_sample = (int)pow(2.0, m);
	if (N_sample>0)
		printf("The number of sample points is %d .\n", N_sample);
	else
	{
		printf("N_sample must be positive! Exit!");
		exit(0);
	}
	puts("Enter an integer m such that the total sampling time is 2^m (for statistic analysis, i.e. average and standard deviation).");
	scanf("%d", &m);
	N_time = (int)pow(2.0, m);
	if (N_time%N_GPU==0&&N_time>0)
	{
		if (N_time!=1)
			printf("The total sampling time is %d .\n", N_time);
		else
		{
			printf("N_time must be greater than 1 for stand deviation evaluation! Exit!");
			exit(0);
		}
	}
	else
	{
		puts("The number of sampling time must be divisible by N_GPU and positive! Exit!");
		exit(0);
	}

	puts("\nSet parameters for important sampling.");
	puts("Select the weight function type: l/e/g for linear, exponential decay or Gaussian.");
	scanf("%c", &mode); // to absorb the \n
	scanf("%c", &mode); 
	if (mode=='l')
		puts("Set the alpha of weight function w(xi) = C-alpha*x");
	else if (mode=='e')
		puts("Set the alpha of weight function w(xi) = C*exp(-alpha*xi), where C is the normalized factor.");
	else if (mode=='g')
		puts("Set the alpha of weight function w(xi) = C*exp(-alpha*xi^2), where C is the normalized factor.");
	else
	{
		puts("Must enter l, e of g! Exit!");
		exit(0);
	}
	scanf("%lf", &alpha);
	printf("The coefficients of weight function is alpha = %.4f .\n", alpha);
	seed = (long*)malloc(N_GPU*sizeof(long));
	for (int i=0; i<N_GPU; i++)
	{
		printf("Set the seed of random number generator for #%d CPU thread.\n", i+1);
		scanf("%ld", &seed[i]);
		printf("The seed of random number generator for #%d CPU thread is %ld .\n", i+1, seed[i]);
	}
	puts("Set the interval for measurement.");
	scanf("%d", &N_measure);
	printf("Interval is %d iterations for measurement.\n", N_measure);
	puts("Set the time of thermalization for important sampling (must be divisible by N_measure).");
	scanf("%d", &N_thermal);
	if (N_thermal%N_measure==0)
		printf("%d iteraions for thermalization.\n", N_thermal);
	else
	{
		puts("N_thermal must be divisible by N_measure! Exit!");
		exit(0);
	}
	printf("\n");

	puts("Set up the GPU.");
	puts("Set the value m such that threads per block tx=2^m .");
	scanf("%d", &m);
	tx = (int)pow(2.0, m);
	if (tx<=1024)
		printf("Thread per block is %d .\n", tx);
	else
	{
		puts("Thread per blocks must be smaller than 1024! Exit!");
		exit(0);
	}
	puts("Set the valure m such that the blocks per grid per GPU bx=2^m .");
	scanf("%d", &m);
	bx = (int)pow(2.0, m);
	if ( bx*tx>N_time/N_GPU )
	{
		bx = (N_time/N_GPU+tx-1)/tx;
		puts("------------------Blocks per grid too large! Self adjusted.------------------");
	}
	printf("Block per grid is %d .", bx);

	printf("\n");

	if (alpha!=0)
		if (mode=='l')
			C = 1.0/(1.0-alpha/2.0);
		else if (mode=='e')
			C = alpha/(1.0-exp(-alpha));
		else
			C = 2.0*sqrt(alpha/acos(-1.0))/gsl_sf_erf(sqrt(alpha));
	else
		C =1.0;

	size = N_time*sizeof(double);
	size_block = bx*sizeof(double);
	size_shared = tx*sizeof(double);

	h_mean = (double*)calloc(N_GPU*bx, sizeof(double));
	h_sigma = (double*)calloc(N_GPU*bx, sizeof(double));
	sum_mean = (double*)calloc(N_GPU, sizeof(double));
	sum_sigma = (double*)calloc(N_GPU, sizeof(double));

	omp_set_num_threads(N_GPU);

    #pragma omp parallel private(cpu_thread_id)
    {
        cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(gid[cpu_thread_id]);

		// Set up kernel for device API
		curandState *devstate;
		cudaMalloc((void**)&devstate, N_time/N_GPU*tx*sizeof(curandState));
		SET_UP_KERNEL <<<N_time/N_GPU, tx>>>(seed[cpu_thread_id], devstate);
		//

		double *d_integrand, *d_mean, *d_sigma; 

		cudaMalloc((void**)&d_integrand, size/N_GPU);
		cudaMalloc((void**)&d_mean, size_block);
		cudaMalloc((void**)&d_sigma, size_block);
		
		if ( cpu_thread_id==0 )
		{
		// Simple Sampling
			puts("\nUse simple sampling...");
			if (N_sample<tx)
				puts("------------------Threads per block too large! Automatically set to be equal to N_simple .------------------");
			puts("------------------The blocks per grid is automatically set to be equal to N_time/N_GPU .------------------");
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }
		if (N_sample<tx)
			SIMPLE_SAMPLING <<<N_time/N_GPU, N_sample, N_sample*sizeof(double)>>>( N_sample, N_dimension, d_integrand, devstate);
		else
			SIMPLE_SAMPLING <<<N_time/N_GPU, tx, size_shared>>>( N_sample, N_dimension, d_integrand, devstate);

//		double *h_integrand = (double*)malloc(size/N_GPU);
//		cudaMemcpy(h_integrand, d_integrand, size/N_GPU, cudaMemcpyDeviceToHost);
//		for (int i=0; i<N_time/N_GPU; i++)
//			printf("%.16e\n", h_integrand[i]);
	
		if ( cpu_thread_id==0 )
			puts("------------------The blocks per grid is switched back.------------------");
	
		DATA_ANALYSIS <<<bx, tx, 2*size_shared>>>( N_time/N_GPU, d_integrand, d_mean, d_sigma ); // to compute both mean and sigma
		
		cudaMemcpy(h_mean+bx*cpu_thread_id, d_mean, size_block, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sigma+bx*cpu_thread_id, d_sigma, size_block, cudaMemcpyDeviceToHost);
//		for (int i=0; i<bx; i++)
//			printf("%.6f %.6f\n", h_mean[i+bx*cpu_thread_id], h_sigma[i+bx*cpu_thread_id]);
		
		for (int i=0; i<bx; i++)
		{
			sum_mean[cpu_thread_id] += (double)h_mean[i+cpu_thread_id*bx];
			sum_sigma[cpu_thread_id] += (double)h_sigma[i+cpu_thread_id*bx];
		} 

		cudaFree(d_integrand);
		cudaFree(d_mean);
		cudaFree(d_sigma);
		cudaFree(devstate);
	} // End of OpenMP

	mean = 0.0;
	sigma = 0.0;
	for (int i=0; i<N_GPU; i++)
	{
		mean += sum_mean[i];
		sigma += sum_sigma[i];
	}

	mean /= N_time;
	sigma = sqrt((sigma - N_time*mean*mean)/(N_time-1.0));

	#pragma omp parallel private(cpu_thread_id)
	{
        cpu_thread_id = omp_get_thread_num();

		if (cpu_thread_id==0)
		{
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
            cudaEventElapsedTime( &processing_time, start, stop);
            printf("Total GPU time processing time is %.4f ms.\n", processing_time);
		}
		cudaDeviceReset();
	}

	printf("The mean of integral calculated by simple sampling with GPU is %.10f .\n", mean);
	printf("The standard deviation of integral calculated by simple sampling with GPU is %.10f .\n", sigma);

	// Simple Sampling Ends

	printf("\n");
	
	if (alpha==0)
	{
		puts("------------------For alpha = 0, heat bath and important sampling are equivalent to simple sampling! Exit!------------------");
		exit(0);
	}

	// Heat Bath
	h_mean = (double*)calloc(N_GPU*bx, sizeof(double));
	h_sigma = (double*)calloc(N_GPU*bx, sizeof(double));
	sum_mean = (double*)calloc(N_GPU, sizeof(double));
	sum_sigma = (double*)calloc(N_GPU, sizeof(double));

	#pragma omp parallel private(cpu_thread_id)
	{
        cpu_thread_id = omp_get_thread_num();
		cudaSetDevice(gid[cpu_thread_id]);

		// Set up kernel for device API
		curandState *devstate;
		cudaMalloc((void**)&devstate, N_time/N_GPU*tx*sizeof(curandState));
		SET_UP_KERNEL <<<N_time/N_GPU, tx>>>(seed[cpu_thread_id], devstate);
		//

		double *d_integrand, *d_mean, *d_sigma; 

		cudaMalloc((void**)&d_integrand, size/N_GPU);
		cudaMalloc((void**)&d_mean, size_block);
		cudaMalloc((void**)&d_sigma, size_block);

		if (cpu_thread_id==0)
		{
			puts("Use heat bath...");
			if ( mode=='l' )
				puts("Linear weight function is selected.");
			else if ( mode=='e' )
				puts("Exponential decay weight function is selected.");
			else
				puts("Gaussian weight function is selected.");
			if (N_sample<tx)
				puts("------------------Threads per block too large! Automatically set to be equal to N_simple .------------------");
			puts("------------------The blocks per grid is automatically set to be equal to N_time/N_GPU .------------------");
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
		}
		if (N_sample<tx)
			HEAT_BATH <<<N_time/N_GPU, N_sample, N_sample*size_shared>>>( mode, N_sample, N_dimension, C, alpha, d_integrand, devstate );
		else
			HEAT_BATH <<<N_time/N_GPU, tx, size_shared>>>( mode, N_sample, N_dimension, C, alpha, d_integrand, devstate );

//		double *h_integrand = (double*)malloc(size/N_GPU);
//		cudaMemcpy(h_integrand, d_integrand, size/N_GPU, cudaMemcpyDeviceToHost);
//		for (int i=0; i<N_time/N_GPU; i++)
//			printf("%.16e\n", h_integrand[i]);
  	
		if ( cpu_thread_id==0 )
			puts("------------------The blocks per grid is switched back.------------------");

		DATA_ANALYSIS <<<bx, tx, 2*size_shared>>>( N_time/N_GPU, d_integrand, d_mean, d_sigma ); // to compute both mean and sigma
		
		cudaMemcpy(h_mean+bx*cpu_thread_id, d_mean, size_block, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sigma+bx*cpu_thread_id, d_sigma, size_block, cudaMemcpyDeviceToHost);
//		for (int i=0; i<bx; i++)
//			printf("%.6f %.6f\n", h_mean[i+bx*cpu_thread_id], h_sigma[i+bx*cpu_thread_id]);

		
		for (int i=0; i<bx; i++)
		{
			sum_mean[cpu_thread_id] += (double)h_mean[i+cpu_thread_id*bx];
			sum_sigma[cpu_thread_id] += (double)h_sigma[i+cpu_thread_id*bx];
		} 

		cudaFree(d_integrand);
		cudaFree(d_mean);
		cudaFree(d_sigma);
		cudaFree(devstate);
  }  // end of OpenMP

	mean = 0.0;
	sigma = 0.0;
	for (int i=0; i<N_GPU; i++)
	{
		mean += sum_mean[i];
		sigma += sum_sigma[i];
	}

	mean /= N_time;
	sigma = sqrt((sigma - N_time*mean*mean)/(N_time-1.0));

	#pragma omp parallel private(cpu_thread_id)
	{
        cpu_thread_id = omp_get_thread_num();

		if (cpu_thread_id==0)
		{
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
            cudaEventElapsedTime( &processing_time, start, stop);
            printf("Total GPU time processing time is %.4f ms.\n", processing_time);
		}
		cudaDeviceReset();
	}

	printf("The mean of integral calculated by heat bath with GPU is %.10f .\n", mean);
	printf("The standard deviation of integral calculated by heat bath with GPU is %.10f .\n", sigma);

	// Heat Bath Ends

	printf("\n");

	// Imortant Sampling
	h_mean = (double*)calloc(N_GPU*bx, sizeof(double));
	h_sigma = (double*)calloc(N_GPU*bx, sizeof(double));
	sum_mean = (double*)calloc(N_GPU, sizeof(double));
	sum_sigma = (double*)calloc(N_GPU, sizeof(double));

	#pragma omp parallel private(cpu_thread_id)
	{
        cpu_thread_id = omp_get_thread_num();
		cudaSetDevice(gid[cpu_thread_id]);
		//Set up kernel
		curandState *devstate;
		cudaMalloc((void**)&devstate, bx*tx*sizeof(curandState*));
		SET_UP_KERNEL<<<bx, tx>>> (seed[cpu_thread_id], devstate);
		//
		
		double *d_integrand, *d_mean, *d_sigma; 

		cudaMalloc((void**)&d_integrand, size/N_GPU);
		cudaMalloc((void**)&d_mean, size_block);
		cudaMalloc((void**)&d_sigma, size_block);

		if (cpu_thread_id==0)
		{
			puts("Use important sampling...");
			if ( mode=='l' )
				puts("Linear weight function is selected.");
			else if ( mode=='e' )
				puts("Exponential decay weight function is selected.");
			else
				puts("Gaussian weight function is selected.");
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
		}

		IMPORTANT_SAMPLING <<<bx, tx>>>( mode, N_sample, N_time/N_GPU, N_dimension, N_thermal, N_measure, C, alpha, d_integrand, devstate );

//		double *h_integrand = (double*)malloc(size/N_GPU);
//		cudaMemcpy(h_integrand, d_integrand, size/N_GPU, cudaMemcpyDeviceToHost);
//		for (int i=0; i<N_time/N_GPU; i++)
//			printf("%.16e\n", h_integrand[i]);

		DATA_ANALYSIS <<<bx, tx, 2*size_shared>>> (N_time/N_GPU, d_integrand, d_mean, d_sigma);

		cudaMemcpy(h_mean+bx*cpu_thread_id, d_mean, size_block, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sigma+bx*cpu_thread_id, d_sigma, size_block, cudaMemcpyDeviceToHost);
//		for (int i=0; i<bx; i++)
//			printf("%.6f %.6f\n", h_mean[i+bx*cpu_thread_id], h_sigma[i+bx*cpu_thread_id]);

		for (int i=0; i<bx; i++)
		{
			sum_mean[cpu_thread_id] += (double)h_mean[i+cpu_thread_id*bx];
			sum_sigma[cpu_thread_id] += (double)h_sigma[i+cpu_thread_id*bx];
		} 

		cudaFree(d_mean);
		cudaFree(d_sigma);
		cudaFree(devstate);
	}

	mean = 0.0;
	sigma = 0.0;
	for (int i=0; i<N_GPU; i++)
	{
		mean += sum_mean[i];
		sigma += sum_sigma[i];
	}

	mean /= N_time;
	sigma = sqrt((sigma - N_time*mean*mean)/(N_time-1.0));

	#pragma omp parallel private(cpu_thread_id)
	{
        cpu_thread_id = omp_get_thread_num();

		if (cpu_thread_id==0)
		{
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
            cudaEventElapsedTime( &processing_time, start, stop);
            printf("Total GPU time processing time is %.4f ms.\n", processing_time);
		}
		cudaDeviceReset();
	}
	printf("The mean of integral calculated by important sampling with GPU is %.10f .\n", mean);
	printf("The standard deviation of integral calculated by important sampling with GPU is %.10f .\n", sigma);
	// Important Sampling Ends

	puts("\n==================================================================================================\n");

	free(h_mean);
	free(h_sigma);
	free(sum_mean);
	free(sum_sigma);
}

__device__ float Integrand (int N, float *array_random)
{
	float f = 0.0;
	for (int i=0; i<N; i++)
		f += powf(array_random[i], 2.0);
	
	f = 1.0/(1.0+f);

	return (f);
}

__device__ float Heat_Bath_Linear (int N, double C, double alpha, float *array_random)
{
	for (int i=0; i<N; i++)
		array_random[i] = (1.0-sqrtf(1.0-2.0*alpha*array_random[i]/C))/alpha;

	return( Integrand(N, array_random)/Weight_Linear(N, C, alpha, array_random) );
}

__device__ float Heat_Bath_Exp (int N, double C, double alpha, float *array_random)
{
	for (int i=0; i<N; i++)
		array_random[i] = -logf(1.0-alpha*array_random[i]/C)/alpha;

	return( Integrand(N, array_random)/Weight_Exp(N, C, alpha, array_random) );
}

__device__ float Heat_Bath_Gaussian (int N, double C, double alpha, float *array_random)
{
	float factor = C/2.0*sqrtf(acosf(-1.0)/alpha);
	for (int i=0; i<N; i++)
		array_random[i] = erfinvf(array_random[i]/factor)/sqrtf(alpha);

	return( Integrand(N, array_random)/Weight_Gaussian(N, C, alpha, array_random) );
}

__device__ float Weight_Linear (int N, double C, double alpha, float *array_random)
{
	float w = 1.0;
	for (int i=0; i<N; i++)
		w *= (1.0-alpha*array_random[i]);

	w = powf(C,N)*w;
	return(w);
}

__device__ float Weight_Exp (int N, double C, double alpha, float *array_random)
{
	float w = 0.0;
	for (int i=0; i<N; i++)
		w += array_random[i];
	
	w = powf(C,N)*exp(-alpha*w);
	return(w);
}

__device__ float Weight_Gaussian (int N, double C, double alpha, float *array_random)
{
	float w = 0.0;
	for (int i=0; i<N; i++)
		w += powf(array_random[i],2.0);
	
	w = powf(C,N)*exp(-alpha*w);
	return(w);
}
