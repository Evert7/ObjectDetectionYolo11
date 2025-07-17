
cmake_minimum_required(VERSION 3.10)
project(yolo11_detector)

set(CMAKE_CXX_STANDARD 17)

# Path to ONNX Runtime
set(ONNXRUNTIME_DIR "/home/hvant/onnxruntime-linux-x64-1.17.0")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# add_executable(yolo11_detector src/main.cpp)    # main.cpp
# add_executable(yolo11_detector src/image_inference.cpp)   # image_inference.cpp
add_executable(yolo11_detector src/video_inference.cpp src/Tracker.cpp)   # image_inference.cpp
# add_executable(yolo11_detector src/camera_inference.cpp)   # image_inference.cpp



target_link_libraries(yolo11_detector
    ${OpenCV_LIBS}
    onnxruntime
)




