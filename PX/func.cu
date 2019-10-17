/* Code completed by Antonio Marco Rodrigo Jimenez */

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <algorithm>
#include <array>
#include <iterator>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

//Subimos las dimensiones (cuadradas) del filtro a memoria de constantes. Cambiar aquí cuando se cambie la dimensión del filtro a utilizar
__constant__ const int constantKernelWidth = 5; //MODIFICAR SEGUN EL FILTRO
//Queremos subir el propio filtro a memoria de constantes, lo cual haremos más adelante con cudaMemcpyToSymbol
__constant__ float constantFilter[constantKernelWidth * constantKernelWidth];

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

//KERNELS

//Kernel que reqaliza una convolución para añadir filtros
__global__
void box_filter(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols, /*const float* const filter, */const int filterWidth) //Descomentar este parametro del metodo si se quiere usar sin memoria de constantes
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float result = 0.0f;
	for (int y = 0; y < filterWidth; y++)
	{
		for (int x = 0; x < filterWidth; x++)
		{
			//Con este doble for realizamos la convolucion. Guardamos aqui el valor de fila
			int row = (int)(thread_2D_pos.y + (y - filterWidth / 2));
			//Nos aseguramos que el valor calculado no se sale de los limites
			if (row < 0)
				row = 0;
			if (row > numRows - 1)
				row = numRows - 1;

			//Guardamos aqui el valor de columna
			int column = (int)(thread_2D_pos.x + (x - filterWidth / 2));
			//Nos aseguramos que el valor calculado no se sale de los limites
			if (column < 0)
				column = 0;
			if (column > numCols - 1)
				column = numCols - 1;

			//Devolvemos el valor de la multiplicacion final de la convolucion:

			//Comentar si se quiere usar sin memoria de constantes
			result += (float)constantFilter[y*filterWidth + x] * (float)(inputChannel[row*numCols + column]);

			//Descomentar si se quiere usar sin memoria de constantes
		    //result += (float)filter	   [y*filterWidth + x] * (float)(inputChannel[row*numCols + column]);
		}
	}
	//Nos aseguramos de que el color final se encuentra entre 0 y 255
	if (result < 0.0f)
		result = 0.0f;
	if (result > 255.0f)
		result = 255.0f;
	outputChannel[thread_1D_pos] = result;
}

//Kernel que invierte los colores de la imagen
__global__
void invert(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	outputChannel[thread_1D_pos] = -inputChannel[thread_1D_pos];
}

//Kernel que modifica el brillo de la imagen
__global__
void brightness(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols, int brightnessAmount)
{
	const int2 thread_2D_pos = make_int2 (blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float result;

	//Sumamos el brillo al canal de color
	result = inputChannel[thread_1D_pos] + brightnessAmount;

	//Comprobamos que nuestro color esta entre 0 y 255
	if (result < 0.0f)
		result = 0.0f;
	if (result > 255.0f)
		result = 255.0f;
	outputChannel[thread_1D_pos] = result;
}

//Kernel que modifica el contraste de la imagen de manera burda
__global__
void contrastLegacy(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols, int contrastLegacyAmount)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float result;

	//Si nos acercamos mas al negro, oscurecemos. Si nos acercamos mas al blanco, aclaramos
	if (inputChannel[thread_1D_pos] >= 127.0f)
		result = inputChannel[thread_1D_pos] + contrastLegacyAmount;
	else
		result = inputChannel[thread_1D_pos] - contrastLegacyAmount;

	//Comprobamos que nuestro color esta entre 0 y 255
	if (result < 0.0f)
		result = 0.0f;
	if (result > 255.0f)
		result = 255.0f;
	outputChannel[thread_1D_pos] = result;
}

//Kernel que modifica el contraste de la imagen de manera sofisticada
__global__
void contrast(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols, float avg, float contrastAmount)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float result;

	//Aplicamos el contraste al canal de color usando la media calculada avg
	result = contrastAmount * ((float)inputChannel[thread_1D_pos] - avg) + avg;

	//Comprobamos que nuestro color esta entre 0 y 255
	if (result < 0.0f)
		result = 0.0f;
	if (result > 255.0f)
		result = 255.0f;
	outputChannel[thread_1D_pos] = result;
}

//Kernel que modifica la saturacion de la imagen
__global__
void saturate(const uchar4* const inputImageRGBA, uchar4* const outputImageRGBA, int numRows, int numCols, int rsat, int gsat, int bsat)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float y;

	float ry1, gy1, by1;
	float ry,  gy,  by;
	float r,   g,   b;

	// Calculamos los componentes Red, Green y Blue
	ry1 = (70.0f  * inputImageRGBA[thread_1D_pos].x - 59.0f * inputImageRGBA[thread_1D_pos].y - 11.0f * inputImageRGBA[thread_1D_pos].z) / 100.0f;
	gy1 = (-30.0f * inputImageRGBA[thread_1D_pos].x + 41.0f * inputImageRGBA[thread_1D_pos].y - 11.0f * inputImageRGBA[thread_1D_pos].z) / 100.0f;
	by1 = (-30.0f * inputImageRGBA[thread_1D_pos].x - 59.0f * inputImageRGBA[thread_1D_pos].y + 89.0f * inputImageRGBA[thread_1D_pos].z) / 100.0f;

	// Utilizamos el modelo de color RGB 30-59-11
	y	= (30.0f  * inputImageRGBA[thread_1D_pos].x + 59.0f * inputImageRGBA[thread_1D_pos].y + 11.0f * inputImageRGBA[thread_1D_pos].z) / 100.0f;

	// Escalamos los valores calculados con las variables saturacion que queremos
	ry = (ry1 * rsat) / 100.0f;
	gy = (gy1 * gsat) / 100.0f;
	by = (by1 * bsat) / 100.0f;

	// Añadimos los colores calculados al componente de luz para generar los nuevos valores RGB
	r = ry + y;
	g = gy + y;
	b = by + y;

	//Comprobamos que nuestro color esta entre 0 y 255 en cada canal
	if (r < 0.0f)
		r = 0.0f;
	if (r > 255.0f)
		r = 255.0f;
	if (g < 0.0f)
		g = 0.0f;
	if (g > 255.0f)
		g = 255.0f;
	if (b < 0.0f)
		b = 0.0f;
	if (b > 255.0f)
		b = 255.0f;

	outputImageRGBA[thread_1D_pos].x = r;
	outputImageRGBA[thread_1D_pos].y = g;
	outputImageRGBA[thread_1D_pos].z = b;
}

//Kernel que modifica el tinte de la imagen para ajustar el tono a un color especifico
__global__
void tint(const uchar4* const inputImageRGBA, uchar4* const outputImageRGBA, int numRows, int numCols, int tintAmount)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float y;

	float ry1, by1;
	float ry,  gy,  by;
	float r,   g,   b;
	float c, s;
	float theta;

	// El valor tinte que queremos representa el angulo en el cual el tinte de la imagen es rotado
	theta = (3.14159 * tintAmount ) / 180;

	c = 256 * cos(theta);
	s = 256 * sin(theta);

	// Calculamos los componentes Red y Blue
	ry1 = (70.0f  * inputImageRGBA[thread_1D_pos].x - 59.0f * inputImageRGBA[thread_1D_pos].y - 11.0f * inputImageRGBA[thread_1D_pos].z) / 100.0f;
	by1 = (-30.0f * inputImageRGBA[thread_1D_pos].x - 59.0f * inputImageRGBA[thread_1D_pos].y + 89.0f * inputImageRGBA[thread_1D_pos].z) / 100.0f;

	// Utilizamos el modelo de color RGB 30-59-11
	y = (30.0f  * inputImageRGBA[thread_1D_pos].x + 59.0f * inputImageRGBA[thread_1D_pos].y + 11.0f * inputImageRGBA[thread_1D_pos].z) / 100.0f;

	// Rotamos los valores calculados segun el angulo que nos da la variable tintAmount
	by = (c * by1 - s * ry1) / 256;
	ry = (s * by1 + c * ry1) / 256;
	// Calculamos el componente Green a partir de los componentes Red y Blue ya rotados
	gy = (-51 * ry - 19 * by) / 100;
	
	// Recalculamos los nuevos valores RGB
	r = ry + y;
	g = gy + y;
	b = by + y;

	//Comprobamos que nuestro color esta entre 0 y 255 en cada canal
	if (r < 0.0f)
		r = 0.0f;
	if (r > 255.0f)
		r = 255.0f;
	if (g < 0.0f)
		g = 0.0f;
	if (g > 255.0f)
		g = 255.0f;
	if (b < 0.0f)
		b = 0.0f;
	if (b > 255.0f)
		b = 255.0f;

	outputImageRGBA[thread_1D_pos].x = r;
	outputImageRGBA[thread_1D_pos].y = g;
	outputImageRGBA[thread_1D_pos].z = b;
}

//Kernel que voltea la imagen horizontalmente
__global__
void flipHorizontal(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	// Alteramos el orden de los threads para que invierta las columnas
	outputChannel[thread_1D_pos] = inputChannel[thread_2D_pos.y * numCols - thread_2D_pos.x];
}

//Kernel que voltea la imagen verticalmente
__global__
void flipVertical(const uchar4* const inputImageRGBA, uchar4* const outputImageRGBA, int numRows, int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Devolvemos la imagen de salida tal cual a partir de la nueva input (d_l)
	outputImageRGBA[thread_1D_pos] = inputImageRGBA[thread_1D_pos];
}

//Kernel que genera numeros aleatorios por pixel
__global__ 
void generate(unsigned long seed, float* data, int numRows, int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Inicializamos la generacion de numeros aleatorios en cada fila
	curandState state;
	curand_init(seed, thread_1D_pos, 0, &state);

	//Parametro de correccion
	int offset = numCols / 10;
	//Por cada fila inicializada, generamos el valor aleatorio de los pixeles de todas las columnas de dicha fila
	for (int i = 0; i < numCols+offset; i++)
	{
		float ranv = curand_uniform(&state);
		data[thread_2D_pos.y * i + thread_2D_pos.x] = ranv;
	}
} 

//Kernel que aplica ruido salt&pepper a la imagen de manera aleatoria
__global__
void saltpepper(const uchar4* const inputImageRGBA, uchar4* const outputImageRGBA, int numRows, int numCols, float data1, float data2, float* data)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Para el efecto saltpepper, definimos el valor blanco y negro para los pixeles
	int salt = 255;
	int pepper = 0;

	//Obtenemos el valor de los numeros aleatorios generados en el kernel generate
	float ranv = data[thread_1D_pos];

	outputImageRGBA[thread_1D_pos] = inputImageRGBA[thread_1D_pos];
	//Segun la probabilidad, pintamos el pixel de blanco
	if (ranv >= 0.5 && ranv < data1)
	{
		outputImageRGBA[thread_1D_pos].x = (char)salt;
		outputImageRGBA[thread_1D_pos].y = (char)salt;
		outputImageRGBA[thread_1D_pos].z = (char)salt;
	}
	//Segun la probabilidad, pintamos el pixel de negro
	if (ranv >= data2 && ranv < 0.5)
	{
		outputImageRGBA[thread_1D_pos].x = (char)pepper;
		outputImageRGBA[thread_1D_pos].y = (char)pepper;
		outputImageRGBA[thread_1D_pos].z = (char)pepper;
	}
}

//Kernel que separa la imagen de entrada en sus 3 canales de color
__global__
void separateChannels(const uchar4* const inputImageRGBA, int numRows, int numCols, unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Dividimos la imagen de entrada en sus 3 canales de color RGB
	redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
	greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
	blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//Kernel que combina los 3 canales de color en la imagen final
__global__
void recombineChannels(const unsigned char* const redChannel, const unsigned char* const greenChannel, const unsigned char* const blueChannel, uchar4* const outputImageRGBA, int numRows, int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	//Alpha deberia ser 255 para que no haya transparencia
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
//Descomentar si se quiere usar sin memoria de constantes, y comentamos el filtro constant del define del principio
//float *d_filter;

//FUNCIONES

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage, const float* const h_filter, const size_t filterWidth)
{
	//Reservamos memoria para los 3 canales de color
	checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

	//Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
	//Descomentar si se quiere usar sin memoria de constantes
	//checkCudaErrors(cudaMalloc(&d_filter, filterWidth * filterWidth * sizeof(float)));
	
	//Copiar el filtro de la CPU host (h_filter) a memoria global de la GPU device (d_filter)
	//Descomentar si se quiere usar sin memoria de constantes
	//checkCudaErrors(cudaMemcpy(d_filter, h_filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
}

//Funcion que crea el filtro. Descomentar el que se quiere usar y comentar los demás (Cambiar el tamaño)
void create_filter(float **h_filter, int *filterWidth) {

	const int KernelWidth = constantKernelWidth; //TAMAÑO DEL FILTRO (Introducir a mano el numero en caso de no usar memoria de constantes//	
	*filterWidth = constantKernelWidth;

	// Creamos el filtro para rellenar
	*h_filter = new float[KernelWidth * KernelWidth];
	
	//Filtro gaussiano: blur
	/*
	const float KernelSigma = 2.;

	float filterSum = 0.f;

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
	  for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
		float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
		(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
		filterSum += filterValue;
	  }
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
	  for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
		(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
	  }
	}
	*/
	
	//Laplaciano 5x5
	
	(*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
	(*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
	(*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
	(*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
	(*h_filter)[20] = 1.;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;
	

	//Creamos diversos filtros para distintos propositos, comentar y descomentar el que se quiera utilizar

	//Aumentar nitidez 3x3
	/*
	(*h_filter)[0] = 0;   (*h_filter)[1] = -0.25;    (*h_filter)[2] = 0;
	(*h_filter)[3] = -0.25;  (*h_filter)[4] = 2.;  (*h_filter)[5] = -0.25;
	(*h_filter)[6] = 0; (*h_filter)[7] = -0.25; (*h_filter)[8] = 0;
	*/
	
	//Detección de línea horizontal - Line Detection Horizontal
	/*
	(*h_filter)[0] = -1.;   (*h_filter)[1] = -1.;    (*h_filter)[2] = -1.;
	(*h_filter)[3] = 2.;  (*h_filter)[4] = 2.;  (*h_filter)[5] = 2.;
	(*h_filter)[6] = -1.; (*h_filter)[7] = -1.; (*h_filter)[8] = -1.;
	*/

	/*
	//Suavizado - Smooth Arithmetic Mean
	(*h_filter)[0] = 0.111;   (*h_filter)[1] = 0.111;    (*h_filter)[2] = 0.111;
	(*h_filter)[3] = 0.111;  (*h_filter)[4] = 0.111;  (*h_filter)[5] = 0.111;
	(*h_filter)[6] = 0.111; (*h_filter)[7] = 0.111; (*h_filter)[8] = 0.111;
	*/

	//Subimos el filtro h_filter a memoria de constantes, como definimos al principio en constantFilter
	//Comentar en caso de que no queramos usar memoria de constantes
	cudaMemcpyToSymbol(constantFilter, *h_filter, sizeof(float) * KernelWidth * KernelWidth);
}

//Funcion que realiza las operaciones sobre la imagen de entrada y llama a los kernels
void operations(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA, uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols, unsigned char *d_redFiltered, unsigned char *d_greenFiltered, unsigned char *d_blueFiltered, const int filterWidth)
{
	///////////////////////////////////////////////////////////////////////////////
	//                    INSTRUCCIONES PARA USAR EL PROGRAMA                    //
	///////////////////////////////////////////////////////////////////////////////
	// Dar valor "true" a los efectos que se quieran aplicar y "false" a los que no

	// Box Filter
	bool b_boxFilter = false;
	// Brillo
	bool b_brightness = false;
	// Contraste legacy
	bool b_contrastLegacy = false;
	// Contraste
	bool b_contrast = false;
	// Saturacion
	bool b_saturation = false;
	// Tinte
	bool b_tint = true;
	// Voltear horizontalmente
	bool b_horizontalFlip = false;
	// Voltear verticalmente
	bool b_verticalFlip = false;
	// Invertir colores
	bool b_invert = false;
	// Aplicar ruido saltpepper (efecto granulado)
	bool b_saltpepper = false;

	// Dar valor recomendado int o float a los parametros de los efectos deseados

	// Cantidad de brillo a aumentar o disminuir en la imagen
	int brightnessAmount = -100; //(-255 , 255)
	// -255 = todo negro
	// 0 = no hay cambios
	// 255 = todo blanco

	// Cantidad de contraste Legacy
	int contrastLegacyAmount = 40; //(0, 127)
	// 0 = no hay cambios
	// 127 = contraste legacy maximo

	// Cantidad de contraste a cambiar
	float contrastAmount = 2.5; // (0, 255) (recomm <10)
	// 0 = todo gris
	// 0.5 = contraste reducido
	// 1 = no hay cambio
	// 10 = contraste muy alto

	// Saturacion de cada canal de color (rojo, verde y azul)
	// (0, +10000) (recom <5000)
	int rsat = 200; // Canal rojo
	int gsat = 200; // Canal verde
	int bsat = 200; // Canal azul
	// Todos a 0 = efecto blanco y negro
	// Todos a 100 = no hay cambios
	// Todos a 200 = saturacion aumentada
	// Todos a 0 menos uno = unico color saturado 

	//Valor de tinte circular. Altera el tono de la imagen a un color
	int tintAmount = -90; // (-180, 180)
	// 0 = no hay cambios
	// -180 = tinte azul
	// -90  = tinte morado
	// -30  = tinte rojo
	// 30   = tinte amarillo
	// 90   = tinte verde
	// 180  = tinte azul

	float saltpepperAmount = 0.5; // (0, 1)
	// 0 = no hay cambios, imagen limpia
	// 0.05 = filtro saltpepper noise optimo
	// 0.5 = filtro saltpepper muy alto
	// 1 = todo ruido
	///////////////////////////////////////////////////////////////////////////////
	//                    ***********************************                    //
	///////////////////////////////////////////////////////////////////////////////

	//Calculamos tamaños de bloque y grid
	const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	const dim3 gridSize(ceil(1.0f*numCols / blockSize.x), ceil(1.0f*numRows / blockSize.y));

	//OPERACIONES DE CANAL

	//Separamos en los 3 canales de color
	separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//Calculamos las operaciones de canal

	//Box Filter
	if (b_boxFilter)
	{
		box_filter << <gridSize, blockSize >> > (d_red,   d_redFiltered,   numRows, numCols, /*d_filter,*/ filterWidth);
		box_filter << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols, /*d_filter,*/ filterWidth);
		box_filter << <gridSize, blockSize >> > (d_blue,  d_blueFiltered,  numRows, numCols, /*d_filter,*/ filterWidth);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	//Invertir color
	if (b_invert)
	{
		invert << <gridSize, blockSize >> > (d_red,   d_redFiltered,   numRows, numCols);
		invert << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols);
		invert << <gridSize, blockSize >> > (d_blue,  d_blueFiltered,  numRows, numCols);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	//Brillo
	if (b_brightness)
	{
		brightness << <gridSize, blockSize >> > (d_red,   d_redFiltered,   numRows, numCols, brightnessAmount);
		brightness << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols, brightnessAmount);
		brightness << <gridSize, blockSize >> > (d_blue,  d_blueFiltered,  numRows, numCols, brightnessAmount);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	//Contraste Legacy
	if (b_contrastLegacy)
	{
		contrastLegacy << <gridSize, blockSize >> > (d_red,   d_redFiltered,   numRows, numCols, contrastLegacyAmount);
		contrastLegacy << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols, contrastLegacyAmount);
		contrastLegacy << <gridSize, blockSize >> > (d_blue,  d_blueFiltered,  numRows, numCols, contrastLegacyAmount);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	//Contraste
	if (b_contrast)
	{
		float sumR = 0.0f;
		float sumG = 0.0f;
		float sumB = 0.0f;

		float j = 0.0f;
		float avg = 0.0f;

		//Calculamos el valor de cada canal
		for (int y = 0; y <= numRows; y++)
		{
			for (int x = 0; x <= numCols; x++)
			{
				sumR += (float)h_inputImageRGBA[x].x;
				sumG += (float)h_inputImageRGBA[x].y;
				sumB += (float)h_inputImageRGBA[x].z;
				j++;
			}
		}

		//Hacemos la media para calcular el valor medio de la imagen
		avg = (sumR + sumG + sumB) / (3 * j);

		contrast << <gridSize, blockSize >> > (d_red,   d_redFiltered,   numRows, numCols, avg, contrastAmount);
		contrast << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols, avg, contrastAmount);
		contrast << <gridSize, blockSize >> > (d_blue,  d_blueFiltered,  numRows, numCols, avg, contrastAmount);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	//Flip horizontal
	if (b_horizontalFlip)
	{
		flipHorizontal << <gridSize, blockSize >> > (d_red,   d_redFiltered,   numRows, numCols);
		flipHorizontal << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols);
		flipHorizontal << <gridSize, blockSize >> > (d_blue,  d_blueFiltered,  numRows, numCols);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	//Recombinamos los 3 canales de color
	recombineChannels << <gridSize, blockSize >> > (d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	

	//OPERACIONES DE IMAGEN
	
	//Saturacion
	if (b_saturation)
	{
		saturate << <gridSize, blockSize >> > (d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, rsat, gsat, bsat);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	
	//Tinte
	if (b_tint)
	{
		tint << <gridSize, blockSize >> > (d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, tintAmount);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	
	//Flip vertical
	if (b_verticalFlip)
	{
		int tam = numRows * numCols;
		int k = numRows;
		uchar4 *l = new uchar4[tam];

		//Cambiamos las ultimas filas por las primeras de la imagen
		for (int y = 0; y < numRows; y++)
		{
			k--;
			for (int x = 0; x < numCols; x++)
				l[k*numCols + x] = h_inputImageRGBA[y*numCols + x];
		}

		uchar4 *d_l;
		checkCudaErrors(cudaMalloc(&d_l, tam * sizeof(uchar4)));
		checkCudaErrors(cudaMemcpy(d_l, l, tam * sizeof(uchar4), cudaMemcpyHostToDevice));

		flipVertical << <gridSize, blockSize >> > (d_l, d_outputImageRGBA, numRows, numCols);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaFree(d_l));
	}

	//Ruido salt & pepper
	if (b_saltpepper)
	{
		//Definimos la probabilidad con la cual añadiremos pixeles blancos o negros
		float probability = saltpepperAmount; //Optimo: 0.05

		float data, data1, data2;
		data = probability * 1 / 2;
		data1 = data + 0.5;
		data2 = 0.5 - data;

		float *d_data;
		checkCudaErrors(cudaMalloc(&d_data, numCols * numRows * sizeof(float)));

		//Generamos un numero aleatorio en cada pixel
		const dim3 gridSize2(ceil(1.0f / blockSize.x), ceil(1.0f*numRows / blockSize.y));
		generate << <gridSize2, blockSize >> > (1234, d_data, numRows, numCols);
		checkCudaErrors(cudaDeviceSynchronize());

		//Contrastamos el numero aleatorio del pixel con la probabilidad, para pintar o no pixeles blancos y negros
		saltpepper << <gridSize, blockSize >> > (d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, data1, data2, d_data);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_data));
	}
}

//Liberamos la memoria reservada para los canales de color
void cleanup() 
{
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
}