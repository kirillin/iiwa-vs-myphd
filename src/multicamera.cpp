#include "multicamera.hpp"

MulticameraRealsense::MulticameraRealsense()
{
	// robot camera
	config_robot.enable_device(CAMERAS_SERIALS[0]);
	config_robot.disable_all_streams();
	config_robot.enable_stream(RS2_STREAM_INFRARED, 1, 848, 100, RS2_FORMAT_Y8, 300);
	rs_robot.open(config_robot);

	profile_robot = &rs_robot.getPipelineProfile();
	pipe_robot = &rs_robot.getPipeline();
	auto prof_robot = profile_robot->get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
	mat_robot.create(prof_robot.height(), prof_robot.width(), CV_8UC1);
	I_robot.init(100, 848);
	d_robot.init(I_robot, 50, 50, "robot_camera");

	// fly camera
	config_fly.enable_device(CAMERAS_SERIALS[1]);
	config_fly.disable_all_streams();
	config_fly.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGBA8, 30);
	rs_fly.open(config_fly);
	profile_fly = &rs_fly.getPipelineProfile();
	pipe_fly = &rs_fly.getPipeline();
	auto prof_fly = profile_fly->get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
	mat_fly.create(prof_fly.height(), prof_fly.width(), CV_8UC4);
	I_fly.init(480, 640);
	d_fly.init(I_fly, 50, 100, "fly_camera");
}

MulticameraRealsense::~MulticameraRealsense()
{
	delete profile_robot;
	delete pipe_robot;
}

/* function from librealsense api examples */
void MulticameraRealsense::frame_to_mat(const rs2::frame &f, cv::Mat &img)
{
	auto vf = f.as<rs2::video_frame>();
	const int w = vf.get_width();
	const int h = vf.get_height();
	const int size = w * h;
	if (f.get_profile().format() == RS2_FORMAT_BGR8)
	{
		memcpy(static_cast<void *>(img.ptr<cv::Vec3b>()), f.get_data(), size * 3);
	}
	else if (f.get_profile().format() == RS2_FORMAT_RGB8)
	{
		cv::Mat tmp(h, w, CV_8UC3, const_cast<void *>(f.get_data()), cv::Mat::AUTO_STEP);
		cv::cvtColor(tmp, img, cv::COLOR_RGB2BGR);
	}
	else if (f.get_profile().format() == RS2_FORMAT_Y8)
	{
		memcpy(img.ptr<uchar>(), f.get_data(), size);
	}
}

/* function from VISP api */
void MulticameraRealsense::getColorFrame(const rs2::frame &frame, vpImage<vpRGBa> &color)
{
	auto vf = frame.as<rs2::video_frame>();
	unsigned int width = (unsigned int)vf.get_width();
	unsigned int height = (unsigned int)vf.get_height();
	color.resize(height, width);

	if (frame.get_profile().format() == RS2_FORMAT_RGB8)
	{
		vpImageConvert::RGBToRGBa(const_cast<unsigned char *>(static_cast<const unsigned char *>(frame.get_data())),
								  reinterpret_cast<unsigned char *>(color.bitmap), width, height);
	}
	else if (frame.get_profile().format() == RS2_FORMAT_RGBA8)
	{
		memcpy(reinterpret_cast<unsigned char *>(color.bitmap),
			   const_cast<unsigned char *>(static_cast<const unsigned char *>(frame.get_data())),
			   width * height * sizeof(vpRGBa));
	}
	else if (frame.get_profile().format() == RS2_FORMAT_BGR8)
	{
		vpImageConvert::BGRToRGBa(const_cast<unsigned char *>(static_cast<const unsigned char *>(frame.get_data())),
								  reinterpret_cast<unsigned char *>(color.bitmap), width, height);
	}
	else
	{
		throw vpException(vpException::fatalError, "RealSense Camera - color stream not supported!");
	}
}

/* blocking test*/
void MulticameraRealsense::test_1()
{

	double loop_start_time = 0;
	while (true)
	{
		loop_start_time = vpTime::measureTimeMs();

		auto data_robot = pipe_robot->wait_for_frames();
		frame_to_mat(data_robot.get_infrared_frame(1), mat_robot);
		vpImageConvert::convert(mat_robot, I_robot);
		vpDisplay::display(I_robot);
		vpDisplay::flush(I_robot);
		// cv::imshow("Display window", mat_robot);

		auto data_fly = pipe_fly->wait_for_frames();
		getColorFrame(data_fly.get_color_frame(), I_fly);
		vpDisplay::display(I_fly);
		vpDisplay::flush(I_fly);

		std::cout << vpTime::measureTimeMs() - loop_start_time << std::endl;
	}
}

/* non-blocking test*/
void MulticameraRealsense::test_2()
{

	double loop_start_time = 0;
	while (true)
	{
		loop_start_time = vpTime::measureTimeMs();

		rs2::frameset data_robot;
		if (pipe_robot->poll_for_frames(&data_robot))
		{
			frame_to_mat(data_robot.get_infrared_frame(1), mat_robot);
			vpImageConvert::convert(mat_robot, I_robot);
			vpDisplay::display(I_robot);
			vpDisplay::flush(I_robot);
		}

		rs2::frameset data_fly;
		if (pipe_fly->poll_for_frames(&data_fly))
		{
			getColorFrame(data_fly.get_color_frame(), I_fly);
			vpDisplay::display(I_fly);
			vpDisplay::flush(I_fly);
		}

		std::cout << vpTime::measureTimeMs() - loop_start_time << std::endl;
	}
}

int main(int argc, char **argv)
{

	MulticameraRealsense mcamera;
	// mcamera.test_1();
	mcamera.test_2();

	return 0;
}
