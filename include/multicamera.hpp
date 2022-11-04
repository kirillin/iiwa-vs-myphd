#ifndef MULTICAMERA_
#define MULTICAMERA_
/*
	Hardcoded two cameras get images with non-blocking sense.
*/

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include <visp3/core/vpImage.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/core/vpImageConvert.h>

std::vector<std::string> CAMERAS_SERIALS{"044322070354", "944622072831"}; // robot, free

/*
	Get frames from several RealSense cameras
	and puts in VISP containers.
*/
class MulticameraRealsense : public vpRealSense2
{

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

public:
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
};

#endif // MULTICAMERA_