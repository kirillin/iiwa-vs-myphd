#include "multicamera.hpp"
#include <opencv2/features2d.hpp>

MulticameraRealsense::MulticameraRealsense() {
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

    rs2::device selected_device = profile_robot->get_device();
    auto depth_sensor = selected_device.first<rs2::depth_sensor>();

    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);
    }

    // if (depth_sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
    // 	depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, false);
    // }

    // fly camera
    config_fly.enable_device(CAMERAS_SERIALS[1]);
    config_fly.disable_all_streams();
    config_fly.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGBA8, 60);
    rs_fly.open(config_fly);
    profile_fly = &rs_fly.getPipelineProfile();
    pipe_fly = &rs_fly.getPipeline();
    auto prof_fly = profile_fly->get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    mat_fly.create(prof_fly.height(), prof_fly.width(), CV_8UC4);
    I_fly.init(480, 640);
    d_fly.init(I_fly, 1000, 50, "fly_camera");


}

MulticameraRealsense::~MulticameraRealsense() {
    delete profile_robot;
    delete pipe_robot;
    delete profile_fly;
    delete pipe_fly;
}

/* function from librealsense api examples */
void MulticameraRealsense::frame_to_mat(const rs2::frame &f, cv::Mat &img) {
    auto vf = f.as<rs2::video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();
    const int size = w * h;
    if (f.get_profile().format() == RS2_FORMAT_BGR8) {
        memcpy(static_cast<void *>(img.ptr<cv::Vec3b>()), f.get_data(), size * 3);
    } else if (f.get_profile().format() == RS2_FORMAT_RGB8) {
        cv::Mat tmp(h, w, CV_8UC3, const_cast<void *>(f.get_data()), cv::Mat::AUTO_STEP);
        cv::cvtColor(tmp, img, cv::COLOR_RGB2BGR);
    } else if (f.get_profile().format() == RS2_FORMAT_Y8) {
        memcpy(img.ptr<uchar>(), f.get_data(), size);
    }
}

/* function from VISP api */
void MulticameraRealsense::getColorFrame(const rs2::frame &frame, vpImage<vpRGBa> &color) {
    auto vf = frame.as<rs2::video_frame>();
    unsigned int width = (unsigned int)vf.get_width();
    unsigned int height = (unsigned int)vf.get_height();
    color.resize(height, width);

    if (frame.get_profile().format() == RS2_FORMAT_RGB8) {
        vpImageConvert::RGBToRGBa(const_cast<unsigned char *>(static_cast<const unsigned char *>(frame.get_data())),
                                  reinterpret_cast<unsigned char *>(color.bitmap), width, height);
    } else if (frame.get_profile().format() == RS2_FORMAT_RGBA8) {
        memcpy(reinterpret_cast<unsigned char *>(color.bitmap),
               const_cast<unsigned char *>(static_cast<const unsigned char *>(frame.get_data())),
               width * height * sizeof(vpRGBa));
    } else if (frame.get_profile().format() == RS2_FORMAT_BGR8) {
        vpImageConvert::BGRToRGBa(const_cast<unsigned char *>(static_cast<const unsigned char *>(frame.get_data())),
                                  reinterpret_cast<unsigned char *>(color.bitmap), width, height);
    } else {
        throw vpException(vpException::fatalError, "RealSense Camera - color stream not supported!");
    }
}

/* blocking test*/
void MulticameraRealsense::test_1() {
    double loop_start_time = 0;
    while (true) {
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
void MulticameraRealsense::test_2() {
    double loop_start_time = 0;
    while (true) {
        loop_start_time = vpTime::measureTimeMs();

        rs2::frameset data_robot;
        if (pipe_robot->poll_for_frames(&data_robot)) {
            frame_to_mat(data_robot.get_infrared_frame(1), mat_robot);
            vpImageConvert::convert(mat_robot, I_robot);
            vpDisplay::display(I_robot);
            vpDisplay::flush(I_robot);
        }

        rs2::frameset data_fly;
        if (pipe_fly->poll_for_frames(&data_fly)) {
            getColorFrame(data_fly.get_color_frame(), I_fly);
            vpDisplay::display(I_fly);
            vpDisplay::flush(I_fly);
        }

        std::cout << vpTime::measureTimeMs() - loop_start_time << std::endl;
    }
}


/* experiment measurments emulator */
void MulticameraRealsense::test_3() {
    double loop_start_time = 0;
    
    // vpDisplayX d;

    double center_x = 0;
    double center_y = 0;
    double radius_small = 0;
    double radius_big = 0;

    double num_circles_1 = 0;
    double num_circles_2 = 0;


    double t0 = vpTime::measureTimeSecond();
    double t = 0;
    double initialisation_time = 5;

    while (t < initialisation_time) {
        t = vpTime::measureTimeSecond() - t0;
        
        rs2::frameset data_fly;
        if (pipe_fly->poll_for_frames(&data_fly)) {
            getColorFrame(data_fly.get_color_frame(), I_fly);
        }

        cv::Mat image;
        vpImageConvert::convert(I_fly, image);
        cv::medianBlur(image, image, 7);
        
        // finding circles
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::medianBlur(gray, gray, 3);

        std::vector<cv::Vec3f> circles1;
        cv::HoughCircles(gray, circles1, cv::HOUGH_GRADIENT, 1,
                    100,  // change this value to detect circles with different distances to each other
                    100, 30, 100, 300 // change the last two parameters
        );
        std::vector<cv::Vec3f> circles2;
        cv::HoughCircles(gray, circles2, cv::HOUGH_GRADIENT, 1,
                    100,  // change this value to detect circles with different distances to each other
                    100, 30, 1, 30 // change the last two parameters
        );

        if (circles1.size() > 0 && circles2.size() > 0) {

            center_x += circles1[0][0] + circles2[0][0];
            center_y += circles1[0][1] + circles2[0][1];

            radius_big += circles1[0][2];
            radius_small += circles2[0][2];

            cv::circle( image, cv::Point(circles1[0][0], circles1[0][1]), circles1[0][2], cv::Scalar(0,255,0), 1, cv::LINE_AA);
            cv::circle( image, cv::Point(circles2[0][0], circles2[0][1]), circles2[0][2], cv::Scalar(0,255,0), 1, cv::LINE_AA);
        }
        num_circles_1 += circles1.size();
        num_circles_2 += circles2.size();

        cv::imshow("sss", image);
        int k = cv::waitKey(1); 
    }

    center_x = center_x / (num_circles_1 + num_circles_2);
    center_y = center_y / (num_circles_1 + num_circles_2);
    radius_big = radius_big / num_circles_1;
    radius_small = radius_small / num_circles_2;


    int x_blob, y_blob; // global vars

    // main control loop with time = \approx 1.5 ms
    while (true) {
        loop_start_time = vpTime::measureTimeMs();

        rs2::frameset data_fly;
        if (pipe_fly->poll_for_frames(&data_fly)) {
            getColorFrame(data_fly.get_color_frame(), I_fly);
            
            cv::Mat image;
            vpImageConvert::convert(I_fly, image);
            if (image.empty()){
                continue;
            }
            // cv::medianBlur(image, image, 3);
            cv::Mat hsv, mask;
            cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
            // cv::inRange(hsv, cv::Scalar(1, 100, 100), cv::Scalar(210,255,255), mask);
            cv::inRange(hsv, cv::Scalar(1, 0, 207), cv::Scalar(180,77,255), mask); // a red laser blob
            cv::Moments m = cv::moments(mask, false);
            x_blob = m.m10/m.m00;
            y_blob = m.m01/m.m00;

            // cv::circle( image, cv::Point(x_blob, y_blob), 5, cv::Scalar(255,0,0), 3, cv::LINE_AA);
        
            // // // ploting
            // cv::circle( image, cv::Point(center_x, center_y), radius_small, cv::Scalar(0,255,0), 1, cv::LINE_AA);
            // cv::circle( image, cv::Point(center_x, center_y), radius_big, cv::Scalar(0,0,255), 1, cv::LINE_AA);

            // // cv::imshow("mask", mask);
            // // cv::imshow("image2_with", im_with_keypoints);
            // cv::imshow("image", image);
            // int k = cv::waitKey(1); 

        }

        std::cout << "Loop time: " << vpTime::measureTimeMs() - loop_start_time << std::endl;
    }

}

// int main(int argc, char **argv) {
//     MulticameraRealsense mcamera;
//     // mcamera.test_1();
//     // mcamera.test_2();
//     mcamera.test_3();

//     return 0;
// }
