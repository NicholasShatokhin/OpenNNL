# Works with 3.15 and tested through 3.30
cmake_minimum_required(VERSION 3.15...3.30)

file(GLOB library_SRC
     "opennnl.h"
     "opennnl.cpp"
)

file(GLOB project_SRC
     "*.h"
     "*.cpp"
)

list(REMOVE_ITEM project_SRC "opennnl.cpp")
list(REMOVE_ITEM project_SRC "opennnl.h")

add_library(OpenNNL ${library_SRC})

add_executable(OpenNNL-demo ${project_SRC})

target_link_libraries(OpenNNL-demo PRIVATE OpenNNL)
