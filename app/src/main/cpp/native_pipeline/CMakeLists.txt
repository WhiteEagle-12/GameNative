cmake_minimum_required(VERSION 3.22.1)

project(NativePipeline)

set(CMAKE_CXX_STANDARD 17)

# Paths to SDKs (Pointed to provided resources)
set(QNN_SDK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../resources\ for\ ai\ agent/qairt/2.45.0.260326)

include_directories(
    include
    ${QNN_SDK_ROOT}/include/QNN
)

add_library(native_pipeline SHARED
    src/QnnUpscaler.cpp
    src/AfmeManager.cpp
    src/BufferInterceptor.cpp
    src/pipeline_jni.cpp
)

# Link against provided libraries
set(QNN_LIB_DIR ${QNN_SDK_ROOT}/lib/aarch64-android)

target_link_libraries(native_pipeline
    log
    android
    ${QNN_LIB_DIR}/libQnnHtp.so
    ${QNN_LIB_DIR}/libQnnSystem.so
)
