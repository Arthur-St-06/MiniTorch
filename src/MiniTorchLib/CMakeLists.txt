project(MiniTorchLib)

set(MiniTorchLib_Src
	Tensor.h
	Tensor.cu
	cpu_kernels.h
	cpu_kernels.cpp
	cuda_support.h
	cuda_support.cu
	cuda_kernels.h
	cuda_kernels.cu
	errors_support.h
)

add_library(MiniTorchLib ${MiniTorchLib_Src})