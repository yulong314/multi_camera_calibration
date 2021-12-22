set(the_description "Custom Calibration Pattern")
find_package(OpenCV)
#ocv_define_module(ccalib opencv_core opencv_imgproc opencv_calib3d opencv_features2d opencv_highgui WRAP python)
add_library(ccalib src/ccalib.cpp src/multicalib.cpp src/omnidir.cpp src/randpattern.cpp src/mymulticalib.cpp src/doubleSide.cpp)
include_directories( ${OpenCV_INCLUDE_DIRS} include/opencv2/ccalib/ include/opencv2 /usr/include/eigen3)
add_executable(multi_cameras_calibration samples/multi_cameras_calibration.cpp)
target_link_libraries(multi_cameras_calibration ccalib ${OpenCV_LIBS} )


add_executable(testDerive src/testDerive.cpp)
