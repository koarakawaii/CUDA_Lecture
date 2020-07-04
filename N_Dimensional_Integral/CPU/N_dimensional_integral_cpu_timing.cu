/*This version includes the timing for CPU calculation. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_cdf.h>

double Integrand (int, double *);
double Weight_Linear (int, double, double, double *);
double Weight_Exp (int, double, double, double *);
double Weight_Gaussian (int, double, double, double *);

int main(void)
{
	char mode;
	int m, N_dimension, N_sample, N_time, size_rn;
	int N_thermal, N_measure;
	float cpu_time;
	double C, alpha; // Use weight function w(xi) = C*exp(-alpha*xi^2), where C is the normalized factor
	double mean, sigma;
	long seed;
	double *set_of_rn, *I;
	gsl_rng *rng;
	cudaEvent_t start, stop;

	puts("Use simple sampling, heat bath and important sampling to calculate the N dimensional integral I( x1,x2,...,xN ) = integral(1/( 1+x1^2+x2^2+x3^2+...+xN^2 )dx1dx2dx3...dxN) .\n");

	puts("Enter the dimension N .");
	scanf("%d", &N_dimension);
	printf("The dimension is %d .\n", N_dimension);
	puts("Enter an integer m such that the number of sampling points N_sample = 2^m .");
	scanf("%d", &m);
	N_sample = (int)pow(2.0, m);
	printf("The number of sample points is %d .\n", N_sample);
	puts("Enter an integer m such that the total sampling time is 2^m (for statistic analysis, i.e. average and standard deviation).");
	scanf("%d", &m);
	N_time = (int)pow(2.0, m);
	printf("The total sampling time is %d .\n", N_time);

	puts("\nSet parameters for important sampling.");
	puts("Select the weight function type: l/e/g for linear, exponential decay or Gaussian.");
	scanf("%c", &mode); // to absorb the \n
	scanf("%c", &mode); 
	if (mode=='l')
		puts("Set the alpha of weight function w(xi) = C*(1-alpha*xi) .");
	else if (mode=='e')
		puts("Set the alpha of weight function w(xi) = C*exp(-alpha*xi), where C is the normalized factor.");
	else if (mode=='g')
		puts("Set the alpha of weight function w(xi) = C*exp(-alpha*xi^2), where C is the normalized factor.");
	else
	{
		puts("Must enter e of g! Exit!");
		exit(0);
	}
	scanf("%lf", &alpha);
	printf("The coefficients of weight function is alpha = %.4f .\n", alpha);
	puts("Set the seed of random number generator.");
	scanf("%ld", &seed);
	printf("The seed of random number generator is %ld .\n", seed);
	puts("Set the time of thermalization for important sampling.");
	scanf("%d", &N_thermal);
	printf("%d iteraions for thermalization.\n", N_thermal);
	puts("Set the interval for measurement.");
	scanf("%d", &N_measure);
	printf("Interval is %d iterations for measurement.\n", N_measure);

	printf("\n");

	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng,seed);
	cudaSetDevice(0);
	cudaEventCreate(&start)	;
	cudaEventCreate(&stop);

	if (alpha!=0)
		if (mode=='l')
			C = 1.0/(1.0-0.5*alpha);
		else if (mode=='e')
			C = alpha/(1.0-exp(-alpha));
		else
			C = 2.0*sqrt(alpha/acos(-1.0))/gsl_sf_erf(sqrt(alpha));
	else
		C =1.0;
	size_rn = N_dimension*sizeof(double);
	set_of_rn = (double*)malloc(size_rn);
	I = (double*)malloc(N_time*sizeof(double));

	puts("Use simple sampling...");
	cudaEventRecord(start,0);
	
	for (int k=0; k<N_time; k++)
	{
		mean = 0.0;
		for (int i=0; i<N_sample; i++)
		{
			for (int j=0; j<N_dimension; j++)
				set_of_rn[j] = gsl_rng_uniform(rng);
				
			mean += Integrand(N_dimension, set_of_rn);
		}
		I[k] = mean/N_sample;	
//		printf("%.6f\n", I[k]);
	}
	
	puts("Sampling ends. Start data analysis...");
	mean  = 0.0;
	for (int i=0; i<N_time; i++)
		mean += I[i];
	mean /= N_time;
	printf("The mean of integral calculated by simple sampling is %.10f .\n", mean);

	sigma = 0.0;
	if ( N_time>1 )
	{
		for (int i=0; i<N_time; i++)
			sigma += pow( I[i]-mean, 2.0 );
			sigma /= N_time-1.0;
			sigma = sqrt(sigma);
		printf("The standard deviation of integral calculated by simple sampling is %.10f .\n", sigma);
	}
	else
		puts("Stand deviation can only be calculated when sampling time are at least 2!");

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &cpu_time, start, stop);
	printf("Total CPU time for simple sampling is %.2f (ms).\n", cpu_time);
	printf("\n");

	puts("Use heat bath...");
	cudaEventRecord(start, 0);
	double y;
	
	if (mode=='l')
	{
		puts("Use inverse of linear function.");
		for (int k=0; k<N_time; k++)
		{
			mean = 0.0;
			for (int i=0; i<N_sample; i++)
			{
				for (int j=0; j<N_dimension; j++)
				{
					y = gsl_rng_uniform(rng)/C;
					set_of_rn[j] = (1.0-sqrt(1.0-2.0*alpha*y))/alpha;
				}
				mean += Integrand(N_dimension, set_of_rn)/Weight_Linear(N_dimension, C, alpha, set_of_rn);
			}
			I[k] = mean/N_sample;
		}

	}
	
	else if (mode=='e')
	{
		puts("Use inverse of exponetial function.");
		for (int k=0; k<N_time; k++)
		{
			mean = 0.0;
			for (int i=0; i<N_sample; i++)
			{
				for (int j=0; j<N_dimension; j++)
				{
					y = gsl_rng_uniform(rng);
					set_of_rn[j] = -log(1.0-alpha*y/C)/alpha;
				}
				mean += Integrand(N_dimension, set_of_rn)/Weight_Exp(N_dimension, C, alpha, set_of_rn);
			}
			I[k] = mean/N_sample;
		}
	}

	else
	{
		puts("Use inverse of Gaussian.");
		double factor = C*sqrt(acos(-1.0)/alpha);

		for (int k=0; k<N_time; k++)
		{
			mean = 0.0;
			for (int i=0; i<N_sample; i++)
			{
				for (int j=0; j<N_dimension; j++)
				{
					y = gsl_rng_uniform(rng)/factor;
					set_of_rn[j] = gsl_cdf_gaussian_Pinv( y+0.5 , 1.0/sqrt(2.0*alpha) );
				}
				mean += Integrand(N_dimension, set_of_rn)/Weight_Gaussian(N_dimension, C, alpha, set_of_rn);
			}
			I[k] = mean/N_sample;
		}
	}
	
	puts("Sampling ends. Start data analysis...");
	mean  = 0.0;
	for (int i=0; i<N_time; i++)
		mean += I[i];
	mean /= N_time;
	printf("The mean of integral calculated by simple sampling is %.10f .\n", mean);

	sigma = 0.0;
	if ( N_time>1 )
	{
		for (int i=0; i<N_time; i++)
			sigma += pow( I[i]-mean, 2.0 );
			sigma /= N_time-1.0;
			sigma = sqrt(sigma);
		printf("The standard deviation of integral calculated by simple sampling is %.10f .\n", sigma);
	}
	else
		puts("Stand deviation can only be calculated when sampling time are at least 2!");

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &cpu_time, start, stop);
	printf("Total CPU time for heat bath is %.2f (ms).\n", cpu_time);
	printf("\n");

	puts("Use important sampling...");	
	int iter, count;
	double weight_old, weight_new;
	double *set_of_rn_new = (double*)malloc(size_rn);
	cudaEventRecord(start, 0);

//	for (int i=0; i<N_dimension; i++)	
//		set_of_rn[i] = gsl_rng_uniform(rng);
	if (mode=='l')
	{
		puts("Use linear weight function.");
		for (int k=0; k<N_time; k++)
		{
			for (int i=0; i<N_dimension; i++)	
				set_of_rn[i] = gsl_rng_uniform(rng);
			// Set initial configuration 
			weight_old = Weight_Linear(N_dimension, C, alpha, set_of_rn);
			// end of setting initial configuraion

			iter = 0;
			count = 0;
			mean = 0.0;
			// Thermalization
			for (int i=1; i<N_thermal; i++)
			{
				for (int j=0; j<N_dimension; j++)	
					set_of_rn_new[j] = gsl_rng_uniform(rng);
				weight_new = Weight_Linear(N_dimension, C, alpha, set_of_rn_new);
				if ( weight_new>weight_old || gsl_rng_uniform(rng)<(weight_new/weight_old) )
				{
					weight_old = weight_new;
					memcpy(set_of_rn, set_of_rn_new, size_rn);
				}	
			}
			// end of thermalization
			// Measurement
			while (count<N_sample)
			{
				for (int j=0; j<N_dimension; j++)	
					set_of_rn_new[j] = gsl_rng_uniform(rng);
				weight_new = Weight_Linear(N_dimension, C, alpha, set_of_rn_new);
				if ( weight_new>weight_old || gsl_rng_uniform(rng)<(weight_new/weight_old) )
				{
					weight_old = weight_new;
					memcpy(set_of_rn, set_of_rn_new, size_rn);
				}
	
				if ( iter%N_measure==0 )
				{
					mean += Integrand(N_dimension, set_of_rn)/weight_old;
					count ++;
//					printf("%.6e %.6e %.6e\n", Integrand(N_dimension, set_of_rn), weight_old, mean);
				}
				iter ++;
			}
			// end of measurement
			I[k] = mean/N_sample;	
//			printf("%.6e\n", I[k]);
		}
	}

	else if (mode=='e')
	{
		puts("Use exponential decay weight function.");
		for (int k=0; k<N_time; k++)
		{
			for (int i=0; i<N_dimension; i++)	
				set_of_rn[i] = gsl_rng_uniform(rng);
			// Set initial configuration 
			weight_old = Weight_Linear(N_dimension, C, alpha, set_of_rn);
			// end of setting initial configuraion

			iter = 0;
			count = 0;
			mean = 0.0;
			// Thermalization
			for (int i=1; i<N_thermal; i++)
			{
				for (int j=0; j<N_dimension; j++)	
					set_of_rn_new[j] = gsl_rng_uniform(rng);
				weight_new = Weight_Exp(N_dimension, C, alpha, set_of_rn_new);
				if ( weight_new>weight_old || gsl_rng_uniform(rng)<(weight_new/weight_old) )
				{
					weight_old = weight_new;
					memcpy(set_of_rn, set_of_rn_new, size_rn);
				}	
			}
			// end of thermalization
			// Measurement
			while (count<N_sample)
			{
				for (int j=0; j<N_dimension; j++)	
					set_of_rn_new[j] = gsl_rng_uniform(rng);
				weight_new = Weight_Exp(N_dimension, C, alpha, set_of_rn_new);
				if ( weight_new>weight_old || gsl_rng_uniform(rng)<(weight_new/weight_old) )
				{
					weight_old = weight_new;
					memcpy(set_of_rn, set_of_rn_new, size_rn);
				}
	
				if ( iter%N_measure==0 )
				{
					mean += Integrand(N_dimension, set_of_rn)/weight_old;
					count ++;
//					printf("%.6e %.6e %.6e\n", Integrand(N_dimension, set_of_rn), weight_old, mean);
				}
				iter ++;
			}
			// end of measurement
			I[k] = mean/N_sample;	
//			printf("%.6e\n", I[k]);
		}
	}
	else
	{
		puts("Use Gaussian weight function.");
		for (int k=0; k<N_time; k++)
		{
			for (int i=0; i<N_dimension; i++)	
				set_of_rn[i] = gsl_rng_uniform(rng);
			// Set initial configuration 
			weight_old = Weight_Linear(N_dimension, C, alpha, set_of_rn);
			// end of setting initial configuraion

			iter = 0;
			count = 0;
			mean = 0.0;
			// Thermalization
			for (int i=1; i<N_thermal; i++)
			{
				for (int j=0; j<N_dimension; j++)	
					set_of_rn_new[j] = gsl_rng_uniform(rng);
				weight_new = Weight_Gaussian(N_dimension, C, alpha, set_of_rn_new);
				if ( weight_new>weight_old || gsl_rng_uniform(rng)<(weight_new/weight_old) )
				{
					weight_old = weight_new;
					memcpy(set_of_rn, set_of_rn_new, size_rn);
				}	
			}
			// end of thermalization
			// Measurement
			while (count<N_sample)
			{
				for (int j=0; j<N_dimension; j++)	
					set_of_rn_new[j] = gsl_rng_uniform(rng);
				weight_new = Weight_Gaussian(N_dimension, C, alpha, set_of_rn_new);
				if ( weight_new>weight_old || gsl_rng_uniform(rng)<(weight_new/weight_old) )
				{
					weight_old = weight_new;
					memcpy(set_of_rn, set_of_rn_new, size_rn);
				}
	
				if ( iter%N_measure==0 )
				{
					mean += Integrand(N_dimension, set_of_rn)/weight_old;
					count ++;
//					printf("%.6e %.6e %.6e\n", Integrand(N_dimension, set_of_rn), weight_old, mean);
				}
				iter ++;
			}
			// end of measurement
			I[k] = mean/N_sample;	
//			printf("%.6e\n", I[k]);
		}
	}
	// end of sampling	
	puts("Sampling ends. Start data analysis...");
	mean  = 0.0;
	for (int i=0; i<N_time; i++)
		mean += I[i];
	mean /= N_time;
	printf("The mean of integral calculated by simple sampling is %.10f .\n", mean);

	sigma = 0.0;
	if ( N_time>1 )
	{
		for (int i=0; i<N_time; i++)
			sigma += pow( I[i]-mean, 2.0 );
			sigma /= N_time-1.0;
			sigma = sqrt(sigma);
		printf("The standard deviation of integral calculated by simple sampling is %.10f .\n", sigma);
	}
	else
		puts("Stand deviation can only be calculated when sampling time are at least 2!");

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &cpu_time, start, stop);
	printf("Total CPU time for important sampling is %.2f (ms).\n", cpu_time);
	puts("\n======================================================================\n");
	free(set_of_rn);
	free(set_of_rn_new);
	free(I);
	cudaDeviceReset();
}

double Integrand (int N, double *array_random)
{
	double f = 0.0;
	for (int i=0; i<N; i++)
		f += pow(array_random[i], 2.0);
	
	f = 1.0/(1.0+f);

	return (f);
}

double Weight_Linear (int N, double C, double alpha, double *array_random)
{
    double w = 1.0;
    for (int i=0; i<N; i++)
        w *= (1.0-alpha*array_random[i]);
	w = pow(C,N)*w;
    return(w);
}


double Weight_Exp (int N, double C, double alpha, double *array_random)
{
	double w = 0.0;
	for (int i=0; i<N; i++)
		w += array_random[i];
	
	w = pow(C,N)*exp(-alpha*w);
	return(w);
}

double Weight_Gaussian (int N, double C, double alpha, double *array_random)
{
	double w = 0.0;
	for (int i=0; i<N; i++)
		w += pow(array_random[i],2.0);
	
	w = pow(C,N)*exp(-alpha*w);
	return(w);
}
