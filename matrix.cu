#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

#define WARP_SIZE 32
#define NODES 281903
#define EDGES 2312497
double h_data[EDGES+100];
int h_col[EDGES+100];
int h_ptr[NODES+100];
double h_deg_out[NODES+100];

__global__ void  //y=mat*x
spmv_csr_vector_kernel ( const int num_rows ,
                         const int * ptr ,
                         const int * indices ,
                         const double * data ,
                         const double * x ,
                         double * y)
{
    __shared__ double vals [WARP_SIZE];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
    int warp_id = thread_id / WARP_SIZE; // global warp index
    int lane = thread_id & (WARP_SIZE - 1); // thread index within the warp
    // one warp per row
    int row = warp_id ;
    if ( row < num_rows )
    {
        int row_start = ptr [ row ];
        int row_end = ptr [ row +1];
        // compute running sum per thread
        vals [ threadIdx.x ] = 0;
        for ( int jj = row_start + lane ; jj < row_end ; jj += WARP_SIZE)
        vals [ threadIdx.x ] += data [ jj ] * x [ indices [ jj ]];
        // parallel reduction in shared memory
        if ( lane < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
        if ( lane < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
        if ( lane < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
        if ( lane < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
        if ( lane < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];
        // first thread writes the result
        if ( lane == 0)
        y[ row ] += vals [ threadIdx.x ];
    }
}

__global__ void  //y=mat*x
spmv_csr_scalar_kernel ( const int num_rows ,
                         const int * ptr ,
                         const int * indices ,
                         const double * data ,
                         const double * x ,
                         double * y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if( row < num_rows )
    {
        double dot = 0;
        int row_start = ptr [ row ];
        int row_end = ptr [ row +1];
        for (int jj = row_start ; jj < row_end ; jj ++)
            dot += data [ jj ] * x[ indices [ jj ]];
        y[ row ] += dot ;
    }
}

__global__ void  //y=x*data
vector_mul_num ( const int num_rows ,
                 const double * data , 
                 const double x , 
                 double * y)
{
    int i= blockDim.x * blockIdx.x + threadIdx.x ;
    y[i]=data[i]*x;
}

__global__ void //z[i]=x[i]*y[i]
vector_mul_vector( const int num_rows ,
                   const double * x ,
                   const double * y ,
                   double * z)
{
    int i= blockDim.x * blockIdx.x + threadIdx.x ;
    z[i]=x[i]*y[i];
}

__device__ double d_ans;
__global__ void //return sum(data[i])
vector_sum ( const int num_rows ,
             const double * data)
{
    for(int i=0;i<num_rows;i++)
        d_ans+=data[i];
}

/*
void mat_x_vec()
{
    double h_data[32]={1,4,2,3,5,7,8,9,6};
    int h_col[32]={0,1,1,2,0,3,4,2,4};
    int h_ptr[32]={0,2,4,7,9};
    double h_x[32]={1,2,3,4,5};
    double h_y[32]={0,0,0,0};
    int num_rows=4;

    double *d_data;
    int *d_col;
    int *d_ptr;
    double *d_x;
    double *d_y;

    cudaMalloc((void**) &d_data,sizeof(double)*32);
    cudaMalloc((void**) &d_col,sizeof(int)*32);
    cudaMalloc((void**) &d_ptr,sizeof(int)*32);
    cudaMalloc((void**) &d_x,sizeof(double)*32);
    cudaMalloc((void**) &d_y,sizeof(double)*32);
    cudaMemcpy((void*)d_data, (void*)h_data, sizeof(double)*32, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_col, (void*)h_col, sizeof(int)*32, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_ptr, (void*)h_ptr, sizeof(int)*32, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_x, (void*)h_x, sizeof(double)*32, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)h_y, sizeof(double)*32, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.Start();
    spmv_csr_vector_kernel<<<num_rows,32>>>(num_rows,d_ptr,d_col,d_data,d_x,d_y);
    //spmv_csr_scalar_kernel<<<1,32>>>(num_rows,d_ptr,d_col,d_data,d_x,d_y);
    timer.Stop();
    printf("Duration: %g ms\n",timer.Elapsed());

    cudaMemcpy((void*)h_y, (void*)d_y, sizeof(double)*32, cudaMemcpyDeviceToHost);

    for(int i=0;i<num_rows;i++)
        printf("%.5f ",h_y[i]);
    printf("\n");
}

void vec_x_num()
{
    double num=233.33;
    double h_x[32]={1,2,3,4,5};
    double h_y[32]={0,0,0,0,0};
    int num_rows=5;

    double *d_x;
    double *d_y;
    cudaMalloc((void**) &d_x,sizeof(double)*32);
    cudaMalloc((void**) &d_y,sizeof(double)*32);
    cudaMemcpy((void*)d_x, (void*)h_x, sizeof(double)*32, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)h_y, sizeof(double)*32, cudaMemcpyHostToDevice);

    vector_mul_num<<<num_rows,32>>>(num_rows,d_x,num,d_y);

    cudaMemcpy((void*)h_y, (void*)d_y, sizeof(double)*32, cudaMemcpyDeviceToHost);

    for(int i=0;i<num_rows;i++)
        printf("%.5f ",h_y[i]);
    printf("\n");
}

void vec_x_vec()
{
    double h_x[32]={1,2,3,4,5};
    double h_y[32]={4,5,6,7,8};
    double h_z[32]={0};
    int num_rows=5;

    double *d_x;
    double *d_y;
    double *d_z;
    cudaMalloc((void**) &d_x,sizeof(double)*32);
    cudaMalloc((void**) &d_y,sizeof(double)*32);
    cudaMalloc((void**) &d_z,sizeof(double)*32);
    cudaMemcpy((void*)d_x, (void*)h_x, sizeof(double)*32, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)h_y, sizeof(double)*32, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_z, (void*)h_z, sizeof(double)*32, cudaMemcpyHostToDevice);

    vector_mul_vector<<<num_rows,32>>>(num_rows,d_x,d_y,d_z);

    cudaMemcpy((void*)h_z, (void*)d_z, sizeof(double)*32, cudaMemcpyDeviceToHost);

    for(int i=0;i<num_rows;i++)
        printf("%.5f ",h_z[i]);
    printf("\n");
}

void sum_vec()
{
    double h_x[32]={1,2,3,4,5};
    double h_ans=0;
    int num_rows=5;

    double *d_x;
    cudaMalloc((void**) &d_x,sizeof(double)*32);
    cudaMemcpy((void*)d_x, (void*)h_x, sizeof(double)*32, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ans, &h_ans, sizeof(double));
    vector_sum<<<1,1>>>(num_rows,d_x);
    cudaMemcpyFromSymbol(&h_ans, d_ans, sizeof(double));
    printf("ans= %.5f\n",h_ans);
    cudaDeviceReset();
}
*/

void input()
{
    ifstream fin;
    fin.open("eindices.txt", ios::in);
    char line[1024]= {0};
    int z;
    long long tot=0;
    while(fin.getline(line, sizeof(line)))
    {
        stringstream word(line);
        word >> z;
        h_col[tot]=z;
        tot++;
    }
    fin.clear();
    fin.close();
    cout<<"indices: "<<tot<<endl;

    fin.open("eindptr.txt", ios::in);
    tot=0;
    while(fin.getline(line, sizeof(line)))
    {
        stringstream word(line);
        word >> z;
        h_ptr[tot]=z;
        tot++;
    }
    fin.clear();
    fin.close();
    cout<<"ptr: "<<tot<<endl;

    fin.open("edata.txt", ios::in);
    tot=0;
    while(fin.getline(line, sizeof(line)))
    {
        stringstream word(line);
        word >> z;
        h_data[tot]=z;
        tot++;
    }
    fin.clear();
    fin.close();
    cout<<"data: "<<tot<<endl;

    fin.open("eout.txt", ios::in);
    tot=0;
    while(fin.getline(line, sizeof(line)))
    {
        stringstream word(line);
        word >> z;
        h_deg_out[tot]=z;
        tot++;
    }
    fin.clear();
    fin.close();
    cout<<"deg_out: "<<tot<<endl;
}

int main()
{
    input();

    return 0;
}

