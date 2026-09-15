cmake_minimum_required(VERSION 3.22.1)
project(voicedesign_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384")

set(MNN_SRC "E:/AndroidStudioProjects/MNN-3.6.1")
set(PREBUILT ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

add_library(voicedesign_jni SHARED voicedesign_jni.cpp vd_text.cpp)
target_include_directories(voicedesign_jni PRIVATE ${MNN_SRC}/include ${MNN_SRC}/express ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(voicedesign_jni PRIVATE VDT_NO_MAIN)

add_library(mnn SHARED IMPORTED)
set_target_properties(mnn PROPERTIES IMPORTED_LOCATION ${PREBUILT}/libMNN.so)
add_library(mnn_express SHARED IMPORTED)
set_target_properties(mnn_express PROPERTIES IMPORTED_LOCATION ${PREBUILT}/libMNN_Express.so)

target_link_libraries(voicedesign_jni mnn mnn_express log)
