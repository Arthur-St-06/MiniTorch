project(MiniTorchLib)

set(MiniTorchLib_Src
	Tensor.h
	Tensor.cpp
	CPUTensorFunctions.h
	CPUTensorFunctions.cpp
)

add_library(MiniTorchLib ${MiniTorchLib_Src})