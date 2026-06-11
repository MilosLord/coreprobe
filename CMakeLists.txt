cmake_minimum_required(VERSION 3.16)
project(coreprobe LANGUAGES CXX)

add_executable(coreprobe coreprobe.cpp)

set_target_properties(coreprobe PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(coreprobe PRIVATE /W4)
    target_compile_definitions(coreprobe PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(coreprobe PRIVATE -Wall -Wextra -Wpedantic)
    find_package(Threads REQUIRED)
    target_link_libraries(coreprobe PRIVATE Threads::Threads)
endif()

install(TARGETS coreprobe RUNTIME DESTINATION bin)
