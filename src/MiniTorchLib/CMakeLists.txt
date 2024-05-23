project(MiniTorchLib)

set(MiniTorchLib_Src
	Tensor.h
	Tensor.cpp
)

add_library(MiniTorchLib ${MiniTorchLib_Src})