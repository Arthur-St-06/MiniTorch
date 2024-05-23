cmake_minimum_required(VERSION 3.20)

project(MiniTorchLib_Tests VERSION 1.0)

include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE  # Set timestamp to time of extraction
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(${PROJECT_NAME} Tensor_Test.cpp)
target_link_libraries(${PROJECT_NAME}
	MiniTorchLib
	gtest_main
)
add_test(NAME Tensor_Test COMMAND ${PROJECT_NAME})