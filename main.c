#define CL_TARGET_OPENCL_VERSION 200

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else

#include <CL/cl.h>

#endif

#include <string.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define ASSERT_NOERROR(err) if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at line %d\n", err, __LINE__); exit(EXIT_FAILURE); } (void)0
#define PRINT_ERROR(err) if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at line %d\n", err, __LINE__); } (void)0

void
print_platform(
	cl_platform_id platform)
{
	char PLAT_INFO[255];
	
	cl_int ret;
	size_t plat_sz;
	
	ret = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 255, PLAT_INFO, &plat_sz);
	ASSERT_NOERROR(ret);
	printf("%s", PLAT_INFO);
	
	printf("  ---  ");
	
	ret = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 255, PLAT_INFO, &plat_sz);
	ASSERT_NOERROR(ret);
	printf("%s", PLAT_INFO);
}

cl_platform_id
ask_platform(void)
{
	cl_int ret;
	
	cl_uint num_platforms;
	ret = clGetPlatformIDs(0, NULL, &num_platforms);
	ASSERT_NOERROR(ret);
	
	cl_platform_id *platforms = calloc(num_platforms, sizeof(cl_platform_id));
	
	ret = clGetPlatformIDs(num_platforms, platforms, NULL);
	ASSERT_NOERROR(ret);
	
	if(num_platforms == 0)
	{
		exit(-1);
	}
	
	cl_platform_id platform_id;
	
	if(num_platforms == 1)
	{
		platform_id = platforms[0];
		
		printf("Automatically chose platform \"");
		print_platform(platform_id);
		printf("\"\n");
	}
	else
	{
		while(1)
		{
			printf("CHOOSE PLATFORM: \n");
			for(uint32_t i = 0; i < num_platforms; ++i)
			{
				printf("%d. ", i);
				print_platform(platforms[i]);
				printf("\n");
			}
			
			printf("PLATFORM> ");
			
			cl_uint choice;
			scanf("%u", &choice);
			
			if(choice < num_platforms)
			{
				platform_id = platforms[choice];
				break;
			}
		}
	}
	
	free(platforms);
	
	return platform_id;
}

void
print_device(
	cl_device_id dev)
{
	char DEV_INFO[255];
	
	cl_int ret;
	size_t str_sz;
	
	ret = clGetDeviceInfo(dev, CL_DEVICE_NAME, 255, DEV_INFO, &str_sz);
	ASSERT_NOERROR(ret);
	printf("%s", DEV_INFO);
}

cl_device_id
ask_device(
	cl_platform_id platform_id)
{
	cl_int ret;
	
	cl_uint num_devices;
	
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	ASSERT_NOERROR(ret);
	
	cl_device_id *devices = calloc(num_devices, sizeof(cl_device_id));
	
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices,
		devices, NULL);
	ASSERT_NOERROR(ret);
	
	if(num_devices == 0)
	{
		exit(-1);
	}
	
	cl_device_id device_id;
	
	if(num_devices == 1)
	{
		device_id = devices[0];
		
		printf("Automatically chose device \"");
		print_device(device_id);
		printf("\"\n");
	}
	else
	{
		while(1)
		{
			printf("CHOOSE DEVICE: \n");
			for(uint32_t i = 0; i < num_devices; ++i)
			{
				printf("%d. ", i);
				print_device(devices[i]);
				printf("\n");
			}
			
			printf("DEVICE> ");
			
			cl_uint choice;
			scanf("%u", &choice);
			
			if(choice < num_devices)
			{
				device_id = devices[choice];
				break;
			}
		}
	}
	
	free(devices);
	
	return device_id;
}

cl_program
create_program(
	cl_context ctx,
	cl_device_id dev)
{
	FILE *fp = fopen("kernel.cl", "r");
	if(!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	
	fseek(fp, 0, SEEK_END);
	size_t source_size = ftell(fp);
	rewind(fp);
	
	char *source_str = (char *) malloc(source_size);
	source_size = fread(source_str, 1, source_size, fp);
	fclose(fp);
	
	cl_int ret;
	
	cl_program program = clCreateProgramWithSource(ctx, 1,
		(const char **) &source_str, (const size_t *) &source_size, &ret);
	ASSERT_NOERROR(ret);
	
	// Build the program
	ret = clBuildProgram(program, 1, &dev, NULL, NULL, NULL);
	ASSERT_NOERROR(ret);
	
	return program;
}

int main(void)
{
	cl_int ret;
	
	// Get platform and device information
	cl_platform_id platform_id = ask_platform();
	cl_device_id device_id = ask_device(platform_id);
	
	printf("----------\n");
	
	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	ASSERT_NOERROR(ret);
	
	// Create a command queue
	cl_command_queue q = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
	ASSERT_NOERROR(ret);
	
	
	// Create a program from the kernel source
	cl_program program = create_program(context, device_id);
	
	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
	ASSERT_NOERROR(ret);
	
	// Create the two input vectors
	const int LIST_SIZE = 1024;
	int *A = (int *) malloc(sizeof(int) * LIST_SIZE);
	int *B = (int *) malloc(sizeof(int) * LIST_SIZE);
	for(int i = 0; i < LIST_SIZE; i++)
	{
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}
	
	// Create memory buffers on the device for each vector
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		LIST_SIZE * sizeof(int), NULL, &ret);
	ASSERT_NOERROR(ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		LIST_SIZE * sizeof(int), NULL, &ret);
	ASSERT_NOERROR(ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		LIST_SIZE * sizeof(int), NULL, &ret);
	ASSERT_NOERROR(ret);
	
	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &a_mem_obj);
	ASSERT_NOERROR(ret);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &b_mem_obj);
	ASSERT_NOERROR(ret);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &c_mem_obj);
	ASSERT_NOERROR(ret);
	
	int *C = (int *) malloc(sizeof(int) * LIST_SIZE);
    cl_event kernelWait;
	
	for(int i = 0; i < 10; i++)
	{
		ret = clEnqueueWriteBuffer(q, a_mem_obj, CL_TRUE, 0,
			LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
		ASSERT_NOERROR(ret);
		ret = clEnqueueWriteBuffer(q, b_mem_obj, CL_TRUE, 0,
			LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
		ASSERT_NOERROR(ret);
		
		size_t global_work_size = LIST_SIZE; // Process the entire lists
		size_t local_work_size = 64; // Process the entire lists
		
		ret = clEnqueueNDRangeKernel(
			q,
			kernel,
			1,
			NULL,
			&global_work_size,
			&local_work_size,
			0,
			NULL,
			&kernelWait);

		
		ASSERT_NOERROR(ret);

        ret = clWaitForEvents(1, &kernelWait);
		
		ret = clEnqueueReadBuffer(q, c_mem_obj, CL_TRUE, 0,
			LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
		ASSERT_NOERROR(ret);
		
		clFinish(q);
		
		int max = MIN(LIST_SIZE, 5);
		printf("loop [");
		for(int j = 0; j < max; ++j)
		{
			printf("%d", C[j]);
			if(j != max - 1)
			{
				printf(", ");
			}
		}
		printf(", ... ]\n");
	}
	// Clean up
	ret = clFlush(q);
	ASSERT_NOERROR(ret);
	ret = clFinish(q);
	ASSERT_NOERROR(ret);
	ret = clReleaseMemObject(a_mem_obj);
	ASSERT_NOERROR(ret);
	ret = clReleaseMemObject(b_mem_obj);
	ASSERT_NOERROR(ret);
	ret = clReleaseMemObject(c_mem_obj);
	ASSERT_NOERROR(ret);
	ret = clReleaseKernel(kernel);
	ASSERT_NOERROR(ret);
	ret = clReleaseProgram(program);
	ASSERT_NOERROR(ret);
	ret = clReleaseCommandQueue(q);
	ASSERT_NOERROR(ret);
	ret = clReleaseContext(context);
	ASSERT_NOERROR(ret);
	free(A);
	free(B);
	free(C);
	return 0;
}

