#ifndef MULTICAMERA_
#define MULTICAMERA_
/*
        Hardcoded two cameras get images with non-blocking sense.
*/

#include <visp3/core/vpImage.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/sensor/vpRealSense2.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <vector>

#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
 #include <visp3/core/vpGaussianFilter.h>
 #include <visp3/core/vpHistogram.h>
 #include <visp3/core/vpImageConvert.h>
 #include <visp3/core/vpImageFilter.h>
 #include <visp3/core/vpMath.h>
 #include <visp3/imgproc/vpImgproc.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

/*
        Get frames from several RealSense cameras
        and puts in VISP containers.
*/
class MulticameraRealsense : public vpRealSense2 {
   public:
    std::vector<std::string> CAMERAS_SERIALS{"044322070354", "944622072831"};  // robot, free
    vpRealSense2 rs_robot;
    rs2::config config_robot;
    rs2::pipeline_profile *profile_robot;
    rs2::pipeline *pipe_robot;
    cv::Mat mat_robot;
    vpImage<unsigned char> I_robot;
    vpDisplayX d_robot;

    vpRealSense2 rs_fly;
    rs2::config config_fly;
    rs2::pipeline_profile *profile_fly;
    rs2::pipeline *pipe_fly;
    cv::Mat mat_fly;
    vpImage<vpRGBa> I_fly;
    vpDisplayX d_fly;

    MulticameraRealsense();

    ~MulticameraRealsense();

    /* Hardcoded two cameras */
    void get_config_for(rs2::config &config, vpImage<unsigned char> &I, std::string serial);

    /* function from librealsense api examples */
    void frame_to_mat(const rs2::frame &f, cv::Mat &img);

    /* function from VISP api */
    void getColorFrame(const rs2::frame &frame, vpImage<vpRGBa> &color);

    void test_1();
    void test_2();
    void test_3();
};

#endif  // MULTICAMERA_