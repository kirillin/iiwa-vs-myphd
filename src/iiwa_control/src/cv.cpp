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
#include <boost/scoped_ptr.hpp>
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

    MulticameraRealsense mcamera;

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

        vel_pub = nh.advertise<geometry_msgs::Twist>("iiwa_vel", 1, true);
        state_pub = nh.advertise<std_msgs::Int32>("iiwa_state", 1, true);

    }

    ~CV() {
        publish_stop();
        vel_pub.shutdown();
        state_pub.shutdown();
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

    void display_point_trajectory(const vpImage<unsigned char> &I, const std::vector<vpImagePoint> &vip,
                                  std::vector<vpImagePoint> *traj_vip) {
        for (size_t i = 0; i < vip.size(); i++) {
            if (traj_vip[i].size()) {
                // Add the point only if distance with the previous > 1 pixel
                if (vpImagePoint::distance(vip[i], traj_vip[i].back()) > 1.) {
                    traj_vip[i].push_back(vip[i]);
                }
            } else {
                traj_vip[i].push_back(vip[i]);
            }
        }
        for (size_t i = 0; i < vip.size(); i++) {
            for (size_t j = 1; j < traj_vip[i].size(); j++) {
                vpDisplay::displayLine(I, traj_vip[i][j - 1], traj_vip[i][j], vpColor::green, 2);
            }
        }
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
        stc::cout << "STOP ROBOT\n";
        
        double start_time = ros::Time::now().toSec();
        double t = 0;
        ros::Rate R(100);
        while (nh.ok() && t < 3.0) {
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
            std::cout << "Init completed!\n";
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

                    CV_STATE = 0;
                    publish(-0.02, 0, 0, 0, 0, 0, 1);
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
                        rs2::frameset data_fly;
                        if (mcamera.pipe_fly->poll_for_frames(&data_fly)) {
                            mcamera.getColorFrame(data_fly.get_color_frame(), mcamera.I_fly);
                            vpDisplay::display(mcamera.I_fly);
                            vpDisplay::flush(mcamera.I_fly);
                        }

                        Eigen::Matrix<double, 6, 1> v_cs;
                        v_cs.setZero();

                        // PUBLISH VELOCITIES
                        publish(v_cs(0), v_cs(1), v_cs(2), v_cs(3), v_cs(4), v_cs(5), CV_STATE);
                    } else {
                        std::cout << "[FSM] CV_STATE in unknown: " << CV_STATE << std::endl;
                    }

                    // FSM
                    if (!has_converged) {
                        CV_STATE = 1;
                        std::cout << "[FSM] CV_STATE: " << CV_STATE << " -- 1 stage -- `ibvs` algorithm!\n";
                    } else {
                        CV_STATE = 2;
                        std::cout << "[FSM] CV_STATE: " << CV_STATE << " -- 2 stage -- `snake` algorithm!\n";
                    }
                    /////////////
                    // end update();
                    /////////////


                    ros::spinOnce();
                    R.sleep();
                    std::cout << "[CV node] freq: " << 1 / (ros::Time::now().toSec() - start_time) << std::endl;
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