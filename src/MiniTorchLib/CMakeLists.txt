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
	common.h
	cuda_common.h
	helper_functions.h
	rand.h
	nn_init.h
	nn_init.cpp
)

add_library(MiniTorchLib ${MiniTorchLib_Src})