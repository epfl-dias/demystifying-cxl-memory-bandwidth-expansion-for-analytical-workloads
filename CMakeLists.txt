cmake_minimum_required(VERSION 3.16)
project(cxl-bandwidth-expansion C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -Wall -Wextra -Werror")
find_package(OpenMP REQUIRED)

foreach(benchmark sequential join filter partition)
  add_executable(${benchmark} src/${benchmark}.c src/common.c)
  target_link_libraries(${benchmark} OpenMP::OpenMP_C)
endforeach()
