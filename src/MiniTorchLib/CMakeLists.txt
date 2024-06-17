project(MiniTorchLib)

set(MiniTorchLib_Src
	Tensor.h
	Tensor.cu
	CPUTensorFunctions.h
	CPUTensorFunctions.cpp
	cuda_support.h
	cuda_support.cu
	cuda_kernels.h
	cuda_kernels.cu
)

add_library(MiniTorchLib ${MiniTorchLib_Src})