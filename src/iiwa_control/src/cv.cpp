#include <ros/ros.h>
#include <stdio.h>
#include <visp3/blob/vpDot2.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpImageFilter.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/gui/vpPlot.h>
#include <visp3/imgproc/vpImgproc.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/vision/vpPose.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/vs/vpServoDisplay.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>
// #include <boost/scoped_ptr.hpp>
#include <boost/units/conversion.hpp>
#include <boost/units/io.hpp>
#include <boost/units/systems/angle/degrees.hpp>
#include <boost/units/systems/si/angular_velocity.hpp>
#include <boost/units/systems/si/velocity.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>  // std::mutex
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <typeinfo>
#include <vector>

#include "fine_measuarments_emulator.hpp"
#include "geometry_msgs/Twist.h"
#include "multicamera.hpp"
#include "std_msgs/Int32.h"
#include "utils.hpp"

class CV {
    std::mutex mtx;
    int CV_STATE;
    
    std::vector<double> ss;
    int kk = 0;
    Eigen::Matrix<double, 2, 1> shift;

    MulticameraRealsense mcamera;

    // 2 stage -- snake
    FineMeasurementsEmulalor fm_emulator;
    bool system_state;                    // [0, 1, 2] visp, snake, off
    Eigen::Matrix<double, 2, 1> state_0;  // get from CV

    // camera snake algorithm detection
    double center_x = 0;
    double center_y = 0;
    double radius_small = 0;
    double radius_big = 0;

    double num_circles_1 = 0;
    double num_circles_2 = 0;

    int x_blob, y_blob;  // global vars

    double initialisation_time = 1;

    Eigen::Matrix<double, 2, 1> y;

    ros::NodeHandle nh;
    ros::Publisher vel_pub;
    ros::Publisher state_pub;

    // bool is_init = true;

    double opt_tagSize;
    bool display_tag;
    int opt_quad_decimate;
    bool opt_verbose;
    bool opt_adaptive_gain;
    bool opt_task_sequencing;
    double convergence_threshold;

    bool final_quit;
    bool has_converged;  // to test secont stage!
    bool send_velocities;
    bool servo_started;
    static double t_init_servo;

   public:
    CV() {
        CV_STATE = 0;  // 0, 1, 2, 3 ???

        vel_pub = nh.advertise<geometry_msgs::Twist>("iiwa_vel", 1, false);
        state_pub = nh.advertise<std_msgs::Int32>("iiwa_state", 1, true);

        fm_emulator.init(10);
        fm_emulator.init_isotropic_surface_data(10);
        system_state = 1;  // [0, 1, 2] visp, snake, off
        state_0.setZero();
        y = state_0;
    }

    ~CV() {
        publish_stop();
        vel_pub.shutdown();
        state_pub.shutdown();
        // delete mcamera;
    }

    void computePose(std::vector<vpPoint> &point, const std::vector<vpImagePoint> &ip, const vpCameraParameters &cam, bool init, vpHomogeneousMatrix &cMo) {
        vpPose pose;
        double x = 0, y = 0;
        for (unsigned int i = 0; i < point.size(); i++) {
            vpPixelMeterConversion::convertPoint(cam, ip[i], x, y);
            point[i].set_x(x);
            point[i].set_y(y);
            pose.addPoint(point[i]);
        }

        if (init == true) {
            vpHomogeneousMatrix cMo_dem;
            vpHomogeneousMatrix cMo_lag;
            pose.computePose(vpPose::DEMENTHON, cMo_dem);
            pose.computePose(vpPose::LAGRANGE, cMo_lag);
            double residual_dem = pose.computeResidual(cMo_dem);
            double residual_lag = pose.computeResidual(cMo_lag);
            if (residual_dem < residual_lag)
                cMo = cMo_dem;
            else
                cMo = cMo_lag;
        }
        pose.computePose(vpPose::VIRTUAL_VS, cMo);
    }

    void make_twist_msg(geometry_msgs::Twist &twist, double x, double y, double z, double wx, double wy, double wz) {
        twist.linear.x = x;
        twist.linear.y = y;
        twist.linear.z = z;
        twist.angular.x = wx;
        twist.angular.y = wy;
        twist.angular.z = wz;
    }

    void publish_stop() {
        {
            std::cout << "STOP ROBOT\n";

            ros::NodeHandle nh;
            ros::Publisher vel_pub;
            ros::Publisher state_pub;

            vel_pub = nh.advertise<geometry_msgs::Twist>("iiwa_vel", 1, true);
            state_pub = nh.advertise<std_msgs::Int32>("iiwa_state", 1, true);

            double start_time = ros::Time::now().toSec();
            double t = 0;
            ros::Rate R(100);
            // while (nh.ok() && t < 3.0) {
            while (nh.ok()) {
                std::cout << "stoping...\n";

                t = ros::Time::now().toSec() - start_time;

                geometry_msgs::Twist twist_stop;
                make_twist_msg(twist_stop, 0, 0, 0, 0, 0, 0);
                vel_pub.publish(twist_stop);

                std_msgs::Int32 state;
                state.data = -1;  // stop robot state
                state_pub.publish(state);

                ros::spinOnce();
                R.sleep();
            }

            vel_pub.shutdown();
            state_pub.shutdown();
        }
    }

    void publish(double x, double y, double z, double wx, double wy, double wz, int cv_state) {
        geometry_msgs::Twist twist;
        make_twist_msg(twist, x, y, z, wx, wy, wz);
        vel_pub.publish(twist);

        std_msgs::Int32 state;
        state.data = cv_state;
        state_pub.publish(state);
    }

    int spin() {
        std::cout << "Loop started cv" << std::endl;
        double t0 = utils::nowtime();
        double t_prev = 0;
        try {
            //////////////////
            // start init();
            //////////////////
            opt_tagSize = 0.120;
            display_tag = true;
            opt_quad_decimate = 2;
            opt_verbose = false;
            opt_adaptive_gain = false;
            opt_task_sequencing = true;
            convergence_threshold = 0.175;

            rs2::frameset data_robot;
            if (mcamera.pipe_robot->poll_for_frames(&data_robot)) {
                mcamera.frame_to_mat(data_robot.get_infrared_frame(1), mcamera.mat_robot);
                vpImageConvert::convert(mcamera.mat_robot, mcamera.I_robot);
                vpDisplay::display(mcamera.I_robot);
                vpDisplay::flush(mcamera.I_robot);
            }

            rs2::frameset data_fly;
            if (mcamera.pipe_fly->poll_for_frames(&data_fly)) {
                mcamera.getColorFrame(data_fly.get_color_frame(), mcamera.I_fly);
                vpDisplay::display(mcamera.I_fly);
                vpDisplay::flush(mcamera.I_fly);
            }

            // Get camera extrinsics
            vpPoseVector ePc;
            // Set camera extrinsics default values
            // -0.0175, -0.08, 0.05;
            ePc[0] = -0.0175;
            ePc[1] = -0.08;
            ePc[2] = 0.05;
            ePc[3] = 0.0;
            ePc[4] = 0.0;
            ePc[5] = 0.0;

            vpHomogeneousMatrix eMc(ePc);
            std::cout << "eMc:\n"
                      << eMc << "\n";

            // Get camera intrinsics
            vpCameraParameters cam = mcamera.rs_robot.getCameraParameters(RS2_STREAM_INFRARED, vpCameraParameters::perspectiveProjWithDistortion);
            std::cout << "cam:\n"
                      << cam << "\n";

            vpDetectorAprilTag::vpAprilTagFamily tagFamily = vpDetectorAprilTag::TAG_36h11;
            vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS;
            // vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::BEST_RESIDUAL_VIRTUAL_VS;
            vpDetectorAprilTag detector(tagFamily);
            detector.setAprilTagPoseEstimationMethod(poseEstimationMethod);
            detector.setDisplayTag(display_tag);
            detector.setAprilTagQuadDecimate(opt_quad_decimate);

            // Servo
            vpHomogeneousMatrix cdMc, cMo, oMo;

            // Desired pose used to compute the desired features
            vpHomogeneousMatrix cdMo(vpTranslationVector(0, 0, opt_tagSize * 3),  // 3 times tag with along camera z axis
                                     vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));

            // Create visual features
            std::vector<vpFeaturePoint> p(4), pd(4);  // We use 4 points

            // Define 4 3D points corresponding to the CAD model of the Apriltag
            std::vector<vpPoint> point(4);
            point[0].setWorldCoordinates(-opt_tagSize / 2., -opt_tagSize / 2., 0);
            point[1].setWorldCoordinates(opt_tagSize / 2., -opt_tagSize / 2., 0);
            point[2].setWorldCoordinates(opt_tagSize / 2., opt_tagSize / 2., 0);
            point[3].setWorldCoordinates(-opt_tagSize / 2., opt_tagSize / 2., 0);

            vpServo task;
            // Add the 4 visual feature points
            for (size_t i = 0; i < p.size(); i++) {
                task.addFeature(p[i], pd[i]);
            }
            task.setServo(vpServo::EYEINHAND_CAMERA);
            task.setInteractionMatrixType(vpServo::CURRENT);

            if (opt_adaptive_gain) {
                vpAdaptiveGain lambda(0.01, 0.01, 0.1);  // lambda(0)=4, lambda(oo)=0.4 and lambda'(0)=30
                task.setLambda(lambda);
            } else {
                task.setLambda(0.008);
            }

            final_quit = false;
            has_converged = false;  // to test secont stage!
            send_velocities = false;
            servo_started = false;
            static double t_init_servo = vpTime::measureTimeMs();
            std::cout << "Init IBVS completed!\n";

            // snake init
            double t0 = vpTime::measureTimeSecond();
            double t = 0;
            while (t < initialisation_time) {
                t = vpTime::measureTimeSecond() - t0;

                rs2::frameset data_fly;
                if (mcamera.pipe_fly->poll_for_frames(&data_fly)) {
                    mcamera.getColorFrame(data_fly.get_color_frame(), mcamera.I_fly);
                }

                cv::Mat image;
                vpImageConvert::convert(mcamera.I_fly, image);

                cv::medianBlur(image, image, 7);

                // finding circles
                cv::Mat gray;
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
                cv::medianBlur(gray, gray, 3);

                std::vector<cv::Vec3f> circles1;
                cv::HoughCircles(gray, circles1, cv::HOUGH_GRADIENT, 1,
                                 100,               // change this value to detect circles with different distances to each other
                                 100, 30, 100, 300  // change the last two parameters
                );
                std::vector<cv::Vec3f> circles2;
                cv::HoughCircles(gray, circles2, cv::HOUGH_GRADIENT, 1,
                                 100,            // change this value to detect circles with different distances to each other
                                 100, 30, 1, 30  // change the last two parameters
                );

                if (circles1.size() > 0 && circles2.size() > 0) {
                    center_x += circles1[0][0] + circles2[0][0];
                    center_y += circles1[0][1] + circles2[0][1];

                    radius_big += circles1[0][2];
                    radius_small += circles2[0][2];

                    cv::circle(image, cv::Point(circles1[0][0], circles1[0][1]), circles1[0][2], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    cv::circle(image, cv::Point(circles2[0][0], circles2[0][1]), circles2[0][2], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                }
                num_circles_1 += circles1.size();
                num_circles_2 += circles2.size();

                // cv::imshow("sss", image);
                // int k = cv::waitKey(1);
            }

            center_x = center_x / (num_circles_1 + num_circles_2);
            center_y = center_y / (num_circles_1 + num_circles_2);
            radius_big = radius_big / num_circles_1;
            radius_small = radius_small / num_circles_2;

            int size = 2 * radius_big / sqrt(2);  // squere inside circle
            int grid = 1;
            fm_emulator.init_isotropic_surface_data(size / grid);
            bool is_init_snake = false;

            //////////////////
            // end init();
            //////////////////

            ros::Rate R(600);
            while (nh.ok()) {
                try {
                    double t_start = vpTime::measureTimeMs();
                    double start_time = ros::Time::now().toSec();

                    const int64_t utime = utils::micros();
                    double t = utils::nowtime() - t0;
                    double dt = t - t_prev;
                    t_prev = t;

                    // CV_STATE = 0;
                    // publish(-0.02, 0, 0, 0, 0, 0, 1);
                    /////////////
                    // start update();
                    /////////////

                    vpColVector v_c(6);
                    Eigen::Matrix<double, 6, 1> v_c_snake;

                    if (CV_STATE == 1) {  // 2 stage -- ibvs

                        rs2::frameset data_robot;
                        if (mcamera.pipe_robot->poll_for_frames(&data_robot)) {
                            mcamera.frame_to_mat(data_robot.get_infrared_frame(1), mcamera.mat_robot);
                            vpImageConvert::convert(mcamera.mat_robot, mcamera.I_robot);
                        }

                        vpDisplay::display(mcamera.I_robot);
                        std::vector<vpHomogeneousMatrix> cMo_vec;
                        detector.detect(mcamera.I_robot, opt_tagSize, cam, cMo_vec);

                        {
                            std::stringstream ss;
                            ss << "Left click to " << (send_velocities ? "stop the robot" : "servo the robot") << ", right click to quit.";
                            vpDisplay::displayText(mcamera.I_robot, 20, 20, ss.str(), vpColor::white);
                        }

                        // Only one tag is detected
                        if (cMo_vec.size() == 1) {
                            cMo = cMo_vec[0];

                            static bool first_time = true;
                            if (first_time) {
                                // Introduce security wrt tag positionning in order to avoid PI rotation
                                std::vector<vpHomogeneousMatrix> v_oMo(2), v_cdMc(2);
                                v_oMo[1].buildFrom(0, 0, 0, 0, 0, M_PI);
                                for (size_t i = 0; i < 2; i++) {
                                    v_cdMc[i] = cdMo * v_oMo[i] * cMo.inverse();
                                }
                                if (std::fabs(v_cdMc[0].getThetaUVector().getTheta()) < std::fabs(v_cdMc[1].getThetaUVector().getTheta())) {
                                    oMo = v_oMo[0];
                                } else {
                                    std::cout << "Desired frame modified to avoid PI rotation of the camera" << std::endl;
                                    oMo = v_oMo[1];  // Introduce PI rotation
                                }

                                // Compute the desired position of the features from the desired pose
                                for (size_t i = 0; i < point.size(); i++) {
                                    vpColVector cP, p_;
                                    point[i].changeFrame(cdMo * oMo, cP);
                                    point[i].projection(cP, p_);

                                    pd[i].set_x(p_[0]);
                                    pd[i].set_y(p_[1]);
                                    pd[i].set_Z(cP[2]);
                                }
                            }

                            // Get tag corners
                            std::vector<vpImagePoint> corners = detector.getPolygon(0);

                            // Update visual features
                            for (size_t i = 0; i < corners.size(); i++) {
                                // Update the point feature from the tag corners location
                                vpFeatureBuilder::create(p[i], cam, corners[i]);
                                // Set the feature Z coordinate from the pose
                                vpColVector cP;
                                point[i].changeFrame(cMo, cP);

                                p[i].set_Z(cP[2]);
                            }

                            if (opt_task_sequencing) {
                                if (!servo_started) {
                                    if (send_velocities) {
                                        servo_started = true;
                                    }
                                    t_init_servo = vpTime::measureTimeMs();
                                }
                                v_c = task.computeControlLaw((vpTime::measureTimeMs() - t_init_servo) / 1000.);
                            } else {
                                v_c = task.computeControlLaw();
                            }

                            // Display the current and desired feature points in the image display
                            vpServoDisplay::display(task, cam, mcamera.I_robot);
                            for (size_t i = 0; i < corners.size(); i++) {
                                std::stringstream ss;
                                ss << i;
                                // Display current point indexes
                                vpDisplay::displayText(mcamera.I_robot, corners[i] + vpImagePoint(15, 15), ss.str(), vpColor::white);
                                // Display desired point indexes
                                vpImagePoint ip;
                                vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), ip);
                                vpDisplay::displayText(mcamera.I_robot, ip + vpImagePoint(15, 15), ss.str(), vpColor::white);
                            }

                            if (opt_verbose) {
                                std::cout << "v_c: " << v_c.t() << std::endl;
                            }

                            double error = task.getError().sumSquare();
                            std::stringstream ss;
                            ss << "error: " << error;
                            vpDisplay::displayText(mcamera.I_robot, 20, static_cast<int>(mcamera.I_robot.getWidth()) - 150, ss.str(), vpColor::white);

                            if (opt_verbose)
                                std::cout << "error: " << error << std::endl;

                            if (error < convergence_threshold) {
                                has_converged = true;
                                std::cout << "Servo task has converged"
                                          << "\n";
                                vpDisplay::displayText(mcamera.I_robot, 100, 20, "Servo task has converged", vpColor::white);
                            }
                            if (first_time) {
                                first_time = false;
                            }
                        } else {
                            v_c = 0;
                        }

                        if (!send_velocities) {
                            v_c = 0;
                        }

                        // PUBLISH VELOCITIES
                        publish(v_c[0], v_c[1], v_c[2], v_c[3], v_c[4], v_c[5], CV_STATE);

                        {
                            std::stringstream ss;
                            ss << "Loop time: " << vpTime::measureTimeMs() - t_start << " ms";
                            vpDisplay::displayText(mcamera.I_robot, 40, 20, ss.str(), vpColor::white);
                        }

                        vpDisplay::flush(mcamera.I_robot);
                        vpMouseButton::vpMouseButtonType button;
                        if (vpDisplay::getClick(mcamera.I_robot, button, false)) {
                            switch (button) {
                                case vpMouseButton::button1:
                                    send_velocities = !send_velocities;
                                    break;

                                case vpMouseButton::button3:
                                    final_quit = true;
                                    v_c = 0;
                                    break;

                                default:
                                    break;
                            }
                        }

                    } else if (CV_STATE == 2) {  // 2 stage --snake

                        Eigen::Matrix<double, 6, 1> v_cs;
                        v_cs.setZero();

                        rs2::frameset data_fly;
                        if (mcamera.pipe_fly->poll_for_frames(&data_fly)) {
                            mcamera.getColorFrame(data_fly.get_color_frame(), mcamera.I_fly);

                            cv::Mat src;
                            vpImageConvert::convert(mcamera.I_fly, src);
                            if (src.empty()) {
                                continue;
                            }

                            // cv::Mat image = src(cv::Range(center_y - radius_big, center_y + radius_big), cv::Range(center_x - radius_big, center_x + radius_big));
                            cv::Mat image = src(cv::Range(center_y - size / 2, center_y + size / 2), cv::Range(center_x - size / 2, center_x + size / 2));
                            cv::flip(image, image, 1);
                            // std::cout << "***" << std::endl;
                            // std::cout << center_x << "\t" << center_y << std::endl;
                            // std::cout << radius_big << "\t" << size <<  std::endl;

                            // cv::medianBlur(image, image, 3);
                            cv::Mat hsv, mask;
                            cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
                            // cv::inRange(hsv, cv::Scalar(1, 100, 100), cv::Scalar(210,255,255), mask);
                            cv::inRange(hsv, cv::Scalar(1, 0, 207), cv::Scalar(180, 77, 255), mask);  // a red laser blob
                            cv::Moments m = cv::moments(mask, false);
                            x_blob = m.m10 / m.m00;
                            y_blob = m.m01 / m.m00;

                            cv::circle(image, cv::Point(x_blob, y_blob), 5, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);

                            // // ploting
                            // cv::circle(image, cv::Point(center_x, center_y), radius_small, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                            // cv::circle(image, cv::Point(center_x, center_y), radius_big, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

                            // cv::imshow("mask", mask);
                            // cv::imshow("image2_with", im_with_keypoints);
                            // cv::imshow("image_laser", image);
                            // int k = cv::waitKey(1);

                            // end of CV


                            // if bloob in ROI
                            y << 0, 0;
                            if ((y_blob > 0 && y_blob < size) && (x_blob > 0 && x_blob < size)) {
                                y << y_blob, x_blob;
                                // std::cout << y.transpose() << std::endl;


                                if (!is_init_snake) {
                                    state_0(0) = y(0);
                                    state_0(1) = y(1);
                                    is_init_snake = true;
                                }

                                double omega_x = 0;
                                double omega_y = 0;

                                // std::cout << "INSIDE ROI\n";

                                // if inside big circle
                                if (x_blob < size && y_blob < size) {
                                    // std::cout << "INSIDE BIG CIRCLE\n";std::cout << "INSIDE BIG CIRCLE\n";

                                    // laser inside big circle
                                    if (abs(x_blob - size / 2) < radius_small && abs(y_blob - size / 2) < radius_small) {
                                        std::cout << "INSIDE SMALL CIRCLE\n";
                                        // laser inside small circle
                                        CV_STATE = 3;  // goal achived
                                        std::cout << "[SNAKE] Goal achived!\n";
                                    } else {
                                        // std::cout << "INSIDE SNAKE SEARCH\n";

                                        // wait_while abs(dq.sum()) < 0.001
                                        if (!fm_emulator.is_inited) {
                                            fm_emulator.search_init(state_0);
                                            // y = state_0;
                                        }
                                        // std::cout << y.transpose() << std::endl;
                                        // std::cout << state_0.transpose() << std::endl;

                                        double s;
                                        fm_emulator.get_isotropic_surface_data(s, y);

                                        bool arrived;
                                        

                                        ss.push_back(s);
                                        kk++;
                                        if (kk > 10) {
                                            s = s / 10;
                                            fm_emulator.update(s, y, shift, arrived);
                                        }


                                        // std::cout << y.transpose() << "\t" << s << "\t" << arrived <<"\n";

                                        // // robot control using `y`
                                        // double omega_x = copysign(1.0, y(0));
                                        // double omega_y = copysign(1.0, y(1));
                                        double omega_x = shift(0);
                                        double omega_y = shift(1);

                                        v_cs << 0, 0, 0, 0.001 * omega_x, 0.001 * omega_y, 0;
                                        publish(v_cs(0), v_cs(1), v_cs(2), v_cs(3), v_cs(4), v_cs(5), CV_STATE);
                                        std::this_thread::sleep_for(std::chrono::milliseconds(50));

                                    }
                                }
                            }
                            cv::imshow("image_laser", image);
                            int k = cv::waitKey(1);

                        }
                        // PUBLISH VELOCITIES
                        // v_cs << 0, 0, 0, 0, 0, 0;
                        // publish(v_cs(0), v_cs(1), v_cs(2), v_cs(3), v_cs(4), v_cs(5), CV_STATE);
                        // std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    } else {
                        std::cout << "[FSM] CV_STATE in unknown: " << CV_STATE << std::endl;
                    }

                    // FSM
                    if (!has_converged && CV_STATE < 3) {
                        is_init_snake = false;
                        CV_STATE = 1;
                        // std::cout << "[FSM] CV_STATE: " << CV_STATE << " -- 1 stage -- `ibvs` algorithm!\n";
                    } else if (has_converged && CV_STATE < 3) {
                        CV_STATE = 2;
                        // std::cout << "[FSM] CV_STATE: " << CV_STATE << " -- 2 stage -- `snake` algorithm!\n";
                    } else {
                        // std::cout << "[FSM] CV_STATE: " << CV_STATE << "\n";
                    }
                    CV_STATE = 2;

                    /////////////
                    // end update();
                    /////////////

                    ros::spinOnce();
                    R.sleep();
                    // std::cout << "[CV node] freq: " << 1 / (ros::Time::now().toSec() - start_time) << std::endl;
                } catch (...) {
                    std::cout << "Try...catch\n";
                    publish_stop();
                }
            }

            std::cout << "Stop the robot " << std::endl;
            // PUBLISH VELOCITIES (ZEROS)
            publish_stop();
        } catch (const vpException &e) {
            std::cout << "ViSP exception: " << e.what() << std::endl;
            std::cout << "Stop the robot " << std::endl;
            publish_stop();
            return EXIT_FAILURE;
        } catch (const std::exception &e) {
            std::cout << "Exception: " << e.what() << std::endl;
            publish_stop();
            return EXIT_FAILURE;
        }
        return 0;
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "cv_node");

    CV cv;
    cv.spin();

    return 0;
}