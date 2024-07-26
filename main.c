#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define ASSERT_NOERROR(err) if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at line %d\n", err, __LINE__); exit(EXIT_FAILURE); }
#define PRINT_ERROR(err) if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at line %d\n", err, __LINE__); }



int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ASSERT_NOERROR(ret);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &device_id, &ret_num_devices);
    ASSERT_NOERROR(ret);

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    ASSERT_NOERROR(ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    ASSERT_NOERROR(ret);



    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    ASSERT_NOERROR(ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    ASSERT_NOERROR(ret);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    ASSERT_NOERROR(ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &ret);
    ASSERT_NOERROR(ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    ASSERT_NOERROR(ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &ret);
    ASSERT_NOERROR(ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ASSERT_NOERROR(ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ASSERT_NOERROR(ret);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    ASSERT_NOERROR(ret);

    int *C = (int*)malloc(sizeof(int)* LIST_SIZE);

    for(int i = 0; i < 10; i++){
        ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
        ASSERT_NOERROR(ret);
        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
                LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
        ASSERT_NOERROR(ret);


        size_t global_item_size = LIST_SIZE; // Process the entire lists
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                &global_item_size, NULL, 0, NULL, NULL);
        PRINT_ERROR(ret);

        ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
                LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
        ASSERT_NOERROR(ret);

        ret = clFinish(command_queue);
        ASSERT_NOERROR(ret);
        printf("loop\n");
    }
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}


