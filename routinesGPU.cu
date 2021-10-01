#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define DEG2RAD 0.017453f
#define BLOCK_SIZE 16

/*
*	Práctica 1 GPU CUDA LINE ASSIST
* 	Código original: Carlos García Sánchez
* 	Modificaciones: Nicolás Pardina Popp
*	
*	Comentarios:
* 	He traspasado el código que más aprovecha la GPU a kernels de CUDA
*	Con	int i = blockIdx.x * blockDim.x + threadIdx.x;
*		int j = blockIdx.y * blockDim.y + threadIdx.y;
*	Consigo el mismo resultado que con CPU pero al cambiarlo para que los accesos sean
*	coalescentes a
*		int j = blockIdx.x * blockDim.x + threadIdx.x;
*		int i = blockIdx.y * blockDim.y + threadIdx.y;
*	mejora considerablemente el rendimiento.
*
*	Mirando el profiler se observa que los kernels que más tiempo consumen
*	son Intensity gradient, Noise reduction, y Hough Transform (sumados 87% aprox)
*	Los dos primeros obtendrían un gran beneficio del uso de memoria compartida ya que 
*	cada hilo tiene 25 accesos a la memoria principal del device, y se repiten accesos
*	entre hilos del mismo bloque.
*	
*/



__global__ void NoiseReduction(uint8_t *im, float *NR, int height, int width)
{
	// 2 bucles for anidados con accesos a submatriz 5x5 
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if( ( (i < height-2) && (i >= 2) ) && ( (j < width-2) && (j >= 2) ) )
	{
		NR[i*width+j] =	 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
						+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
						+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
						+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
						+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
						/159.0;
	}
}


/*
*	Kernel Gradiente de la imagen
*/

__global__ void IntensityGradient(float *NR, float *G, float *phi, float *Gx, float *Gy, int height, int width)
{
	float PI = 3.141593;
	// 2 bucles for anidados con accesos a matriz 5x5 (con una fila o columna de 0)
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( (i < height-2) && (i >= 2) ) && ( (j < width-2) && (j >= 2) ) )
	{
		// Intensity gradient of the image
		Gx[i*width+j] = 
			 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
			+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
			+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
			+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


		Gy[i*width+j] = 
			 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
			+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
			+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
			+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

		G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
		phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

		if(fabs(phi[i*width+j])<=PI/8 )
			phi[i*width+j] = 0;
		else if (fabs(phi[i*width+j])<= 3*(PI/8))
			phi[i*width+j] = 45;
		else if (fabs(phi[i*width+j]) <= 5*(PI/8))
			phi[i*width+j] = 90;
		else if (fabs(phi[i*width+j]) <= 7*(PI/8))
			phi[i*width+j] = 135;
		else phi[i*width+j] = 0;
	}
}

/*
*	Kernel Supresion de no-maximo
*/
__global__ void Edge(float *G, float *phi, uint8_t *pedge, int height, int width)
{
	// Edge
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( (i < height-3) && (i >= 3) ) && ( (j < width-3) && (j >= 3) ) )
	{
		pedge[i*width+j] = 0;
		if(phi[i*width+j] == 0)
		{
			if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
				pedge[i*width+j] = 1;
		} 
		else if(phi[i*width+j] == 45)
		{
			if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
				pedge[i*width+j] = 1;
		} 
		else if(phi[i*width+j] == 90)
		{
			if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
				pedge[i*width+j] = 1;
		} else if(phi[i*width+j] == 135) 
		{
			if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
				pedge[i*width+j] = 1;
		}
	}
}

/*
* Kernel Hysteresis
*/
__global__ void HysteresisThresholding(uint8_t *image_out, float *G, uint8_t *pedge, float lowthres, float hithres, int height, int width)
{
	// Hysteresis Thresholding
	int i, j;
	int ii, jj;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( (i < height-3) && (i >= 3) ) && ( (j < width-3) && (j >= 3) ) )
	{
		image_out[i*width+j] = 0;
		if(G[i*width+j]>hithres && pedge[i*width+j])
			image_out[i*width+j] = 255;
		else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
			// check neighbours 3x3
			for (ii=-1;ii<=1; ii++)
				for (jj=-1;jj<=1; jj++)
					if (G[(i+ii)*width+j+jj]>hithres)
						image_out[i*width+j] = 255;
	}
}

void canny(uint8_t *im, uint8_t *image_out,	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge, float level,	int height, int width, dim3 nBlocks, dim3 nThreads)
{
	int sizeOfImgUint8 = width * height * sizeof(uint8_t);
	int sizeOfImgFloat = width * height * sizeof(float);

	uint8_t *img_GPU;
	float *NR_GPU;

	// Load img to device memory
	cudaMalloc(&img_GPU, sizeOfImgUint8);
	cudaMemcpy(img_GPU, im, sizeOfImgUint8, cudaMemcpyHostToDevice);
	// Allocate NR in device memory
	cudaMalloc(&NR_GPU, sizeOfImgFloat);

	// Invoke kernel
	NoiseReduction<<<nBlocks, nThreads>>>(img_GPU, NR_GPU, height, width);
	cudaDeviceSynchronize();
	
	// Free device memory
	//cudaFree(img_GPU);

	float *G_GPU;
	float *phi_GPU;
	float *Gx_GPU;
	float *Gy_GPU;

	cudaMalloc(&G_GPU, sizeOfImgFloat);
	cudaMalloc(&phi_GPU, sizeOfImgFloat);
	cudaMalloc(&Gx_GPU, sizeOfImgFloat);
	cudaMalloc(&Gy_GPU, sizeOfImgFloat);
	// Invoke kernel
	IntensityGradient<<<nBlocks, nThreads>>>(NR_GPU, G_GPU, phi_GPU, Gx_GPU, Gy_GPU, height, width);
	cudaDeviceSynchronize();

	// Free device memory
	cudaFree(NR_GPU);
	cudaFree(Gx_GPU);
	cudaFree(Gy_GPU);	

	uint8_t *pedge_GPU;
	cudaMalloc(&pedge_GPU, sizeOfImgUint8);
	// Invoke kernel
	Edge<<<nBlocks, nThreads>>>(G_GPU, phi_GPU, pedge_GPU, height, width);
	cudaDeviceSynchronize();

	// Free device memory
	cudaFree(phi_GPU);

	float lowthres = level/2;
	float hithres = 2*(level);

	//uint8_t *image_out_GPU;
//	cudaMalloc(&image_out_GPU, sizeOfImgUint8);

	// Invoke kernel
	HysteresisThresholding<<<nBlocks, nThreads>>>(img_GPU, G_GPU, pedge_GPU, lowthres, hithres, height, width);
	
	cudaDeviceSynchronize();
	// Bring result from device memory
	cudaMemcpy(image_out, img_GPU, sizeOfImgUint8, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(G_GPU);
	cudaFree(pedge_GPU);
	//cudaFree(image_out_GPU);
}


void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

/*
*	Kernel Hough Transform
*/

__global__ void HoughTransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, float *sin_table, float *cos_table)
{
	int theta = 0;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);	// sqrt2 * la mayor de height/width

	if(i < accu_width*accu_height)
		accumulators[i]=0;	

	float center_x = width/2.0; 
	float center_y = height/2.0;

	if(i < height)
	{
		if(j < width)
		{
			if( im[ (i*width) + j] > 250 ) // Pixel is edge  
			{  
				
				for(theta=0;theta<180;theta++)  
				{  
					float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					atomicAdd(&accumulators[(int)((round(rho + hough_h) * 180.0)) + theta], 1);
				} 
			}
		}
	}

}

void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, float *sin_table, float *cos_table, dim3 nBlocks, dim3 nThreads)
{
	uint32_t *accumulators_GPU;
	float *sin_table_GPU; 
	float *cos_table_GPU;
	uint8_t *imEdge_GPU;

	cudaMalloc(&accumulators_GPU, accu_width * accu_height * sizeof(uint32_t));
	cudaMalloc(&sin_table_GPU, 180 * sizeof(float));
	cudaMalloc(&cos_table_GPU, 180 * sizeof(float));
	cudaMalloc(&imEdge_GPU, width*height*sizeof(uint8_t));


	cudaMemcpy(sin_table_GPU, sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cos_table_GPU, cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(accumulators_GPU, accumulators, accu_width * accu_height * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(imEdge_GPU, im, width*height*sizeof(uint8_t), cudaMemcpyHostToDevice);

	HoughTransform<<<nBlocks, nThreads>>>(imEdge_GPU, width, height, accumulators_GPU, accu_width, accu_height, sin_table_GPU, cos_table_GPU);
	cudaDeviceSynchronize();

	cudaMemcpy(accumulators, accumulators_GPU , accu_width * accu_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(imEdge_GPU);
	

	cudaFree(accumulators_GPU);
	cudaFree(sin_table_GPU);
	cudaFree(cos_table_GPU);
}


void line_asist_GPU(uint8_t *im, int height, int width,	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,	float *sin_table, float *cos_table, uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	int blockHeight;
	int blockWidth;
	if (height % BLOCK_SIZE)
		blockHeight = height/BLOCK_SIZE + 1;
	else 
		blockHeight = height/BLOCK_SIZE;
	if (width % BLOCK_SIZE)
		blockWidth = width/BLOCK_SIZE + 1;
	else 
		blockWidth = width/BLOCK_SIZE;
	
	dim3 nBlocks(blockHeight, blockWidth);
	dim3 nThreads(BLOCK_SIZE,BLOCK_SIZE);


	/* Canny */
	canny(im, imEdge, NR, G, phi, Gx, Gy, pedge, 1000.0f, height, width, nBlocks, nThreads);

	int threshold;
	/* hough transform */
	houghtransform(imEdge, width, height, accum, accu_width, accu_height, sin_table, cos_table, nBlocks, nThreads);

	if (width>height)
		threshold = width/6;
	else 
		threshold = height/6;

	getlines(threshold, accum, accu_width, accu_height, width, height, sin_table, cos_table, x1, y1, x2, y2, nlines);
}