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
#include <lcm/lcm-cpp.hpp>
#include <map>
#include <mutex>  // std::mutex
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <typeinfo>
#include <vector>

#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"

#include "iiwa_kinematics.hpp"
#include "multicamera.hpp"
#include "utils.hpp"

using drake::lcmt_iiwa_command;
using drake::lcmt_iiwa_status;

const char *kLcmStatusChannel = "IIWA_STATUS";
const char *kLcmCommandChannel = "IIWA_COMMAND";
const int kNumJoints = 7;

typedef Eigen::Matrix<double, kNumJoints, 1> Vector7d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

class Controller {
    std::ofstream log_file;
    double t_ramp;
    double t_ramp0;
    bool init_ramp;
    int ramp_state;
    
    bool DEBUG = true;
    bool is_init = true;

    // iiwa driver interface
    lcm::LCM lcm;
    lcmt_iiwa_status lcm_status{};
    lcmt_iiwa_command lcm_command{};

    // robot state
    Eigen::Matrix<double, kNumJoints, 1> q;
    Eigen::Matrix<double, kNumJoints, 1> dq;

    // stiffness and damping parameters
    Eigen::Matrix<double, kNumJoints, 1> k;
    Eigen::Matrix<double, kNumJoints, 1> b;
    Eigen::Matrix<double, kNumJoints, kNumJoints> K;
    Eigen::Matrix<double, kNumJoints, kNumJoints> B;

    // robot desired values
    Eigen::Matrix<double, kNumJoints, 1> q_des;
    Eigen::Matrix<double, kNumJoints, 1> dq_des;
    // Vector6d v_des;
    Eigen::Matrix<double, 6, kNumJoints> J;
    Eigen::Matrix<double, kNumJoints, 6> pinvJ;
    Eigen::Quaterniond quat;
    Eigen::Matrix<double, 3, 1> fpe;
    Eigen::Matrix<double, 3, 3> fRe;

    Eigen::Matrix<double, kNumJoints, 1> tau_fb;
    Eigen::Matrix<double, kNumJoints, 1> tau_fb0;
    Eigen::Matrix<double, kNumJoints, 1> tau_fb_prev;
    bool is_ramp = true;

    Eigen::Matrix<double, kNumJoints, 1> Q_DES_0;
    Eigen::Matrix<double, kNumJoints, 1> q_delta;
    Eigen::Matrix<double, kNumJoints, 1> q_delta_prev;

    Eigen::Matrix<double, kNumJoints, 1> ramp_q_des_0;
    Eigen::Matrix<double, kNumJoints, 1> q_des_prev;
    
    Eigen::Matrix<double, kNumJoints, 1> q_error;    
    Eigen::Matrix<double, kNumJoints, 1> ramp_q_error;    
    

    MulticameraRealsense mcamera;

    vpColVector taus_plot;

    std::mutex mtx;
    

   public:
    Controller() {
        log_file.open("log_control.txt");
        
        q_error.setZero();
        ramp_q_error.setZero();
        ramp_state = 0;
        tau_fb_prev.setZero();
        tau_fb0.setZero();

        ramp_q_des_0.setZero();
        q_des_prev.setZero();

        t_ramp = 0;
        t_ramp0 = 0;
        bool init_ramp = false;

        q.setZero();
        dq.setZero();
        K.setZero();
        B.setZero();

        K.diagonal() << 2, 3, 1, 2, 0.2, 0.1, 0.1;
        B.diagonal() << 2, 3, 1, 2, 0.2, 0.1, 0.1;
        K = K * 1;
        B = B * 1;

        // Klambda.diagonal() << 1, 1, 1, 1, 1, 1, 1, 1;
        // v_des << 0, 0, 0, 0, 0, 0;

        J.setZero();
        fpe.setZero();
        fRe.setZero();
        q_delta.setZero();
        Q_DES_0.setZero();
        tau_fb.setZero();

        taus_plot.resize(7, 1, true);
    }

    ~Controller() {
        log_file.close();

        // stop robot any way
        if (!lcm.good())
            std::cout << "[~Controller()] There is a problem with lcm\n";
        lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);
        if (0 == lcm.handle()) {
            const int64_t utime = utils::micros();
            Eigen::Matrix<double, 7, 1> tau_fb;
            tau_fb.setZero();
            publish(tau_fb, utime);
        }
    }

    void handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg) {
        mtx.lock();
        lcm_status = *msg;
        for (int i = 0; i < 7; i++) {
            q(i) = lcm_status.joint_position_measured[i];
            dq(i) = lcm_status.joint_velocity_estimated[i];
        }
        mtx.unlock();
    }

    // void compute_error(std::vector<vpImagePoint> &ip, std::vector<vpImagePoint> &ipd, Eigen::Matrix<double, 8, 1> &error) {
    //     error = Eigen::Matrix<double, 8, 1>::Zero();
    //     for (int i = 0; i < Np; i++) {
    //         // error in image space for each point
    //         double ex = ip[i].get_u() - ipd[i].get_u();
    //         double ey = ip[i].get_v() - ipd[i].get_v();
    //         error(0 + 2 * i) = ex;
    //         error(1 + 2 * i) = ey;
    //     }
    // }

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

    /* Publish command to robot */
    void publish(Eigen::Matrix<double, 7, 1> &tau_fb, int64_t utime) {
        // std::cout << "tau_fb::";
        // for (int i = 0; i < 7; i++) {
        //     std::cout << std::setw(2) << tau_fb(i) << std::setprecision(8) <<  " \t";
        // }
        // std::cout << std::endl;

        // mtx.lock();

        lcm_command.utime = utime;
        lcm_command.num_joints = kNumJoints;
        lcm_command.num_torques = kNumJoints;
        lcm_command.joint_position.resize(kNumJoints, 0);
        lcm_command.joint_torque.resize(kNumJoints, 0);
        for (int i = 0; i < kNumJoints; i++) {
            lcm_command.joint_position[i] = lcm_status.joint_position_measured[i];
            if (abs(tau_fb(i)) > 50) {
                std::cout << "TORQUE UPPER 50 - joint: " << i << std::endl;
                // for (int j = 0; j < kNumJoints; j++)
                lcm_command.joint_torque[i] = 0;
                // break;
            } else {
                lcm_command.joint_torque[i] = tau_fb[i];
            }
        }

        lcm.publish("IIWA_COMMAND", &lcm_command);

        mtx.unlock();
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

    int loop_vs_april() {
        try {
            // connect to iiwa-driver
            if (!lcm.good())
                return 1;

            lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

            double opt_tagSize = 0.120;
            bool display_tag = true;
            int opt_quad_decimate = 2;
            bool opt_verbose = false;
            bool opt_plot = false;
            bool opt_adaptive_gain = false;
            bool opt_task_sequencing = true;
            double convergence_threshold = 0.175;

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

            // vpImage<unsigned char> I(height, width);

            // vpDisplayX dc(I, 10, 10, "Color image");

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

            vpPlot *plotter = nullptr;
            int iter_plot = 0;

            if (opt_plot) {
                plotter = new vpPlot(3, static_cast<int>(250 * 3), 500, static_cast<int>(mcamera.I_robot.getWidth()) + 80, 10,
                                     "Real time curves plotter");
                plotter->setTitle(0, "Visual features error");
                plotter->setTitle(1, "Camera velocities");
                plotter->setTitle(2, "Torques");
                plotter->initGraph(0, 8);
                plotter->initGraph(1, 6);
                plotter->initGraph(2, 7);
                plotter->setLegend(0, 0, "error_feat_p1_x");
                plotter->setLegend(0, 1, "error_feat_p1_y");
                plotter->setLegend(0, 2, "error_feat_p2_x");
                plotter->setLegend(0, 3, "error_feat_p2_y");
                plotter->setLegend(0, 4, "error_feat_p3_x");
                plotter->setLegend(0, 5, "error_feat_p3_y");
                plotter->setLegend(0, 6, "error_feat_p4_x");
                plotter->setLegend(0, 7, "error_feat_p4_y");
                plotter->setLegend(1, 0, "vc_x");
                plotter->setLegend(1, 1, "vc_y");
                plotter->setLegend(1, 2, "vc_z");
                plotter->setLegend(1, 3, "wc_x");
                plotter->setLegend(1, 4, "wc_y");
                plotter->setLegend(1, 5, "wc_z");
                plotter->setLegend(2, 0, "tau_1");
                plotter->setLegend(2, 1, "tau_2");
                plotter->setLegend(2, 2, "tau_3");
                plotter->setLegend(2, 3, "tau_4");
                plotter->setLegend(2, 4, "tau_5");
                plotter->setLegend(2, 5, "tau_6");
                plotter->setLegend(2, 6, "tau_7");
            }

            bool final_quit = false;
            // convergence!
            bool has_converged = true;
            bool send_velocities = false;
            bool servo_started = false;
            std::vector<vpImagePoint> *traj_corners = nullptr;  // To memorize point trajectory

            static double t_init_servo = vpTime::measureTimeMs();

            // robot.set_eMc(eMc); // Set location of the camera wrt end-effector frame
            // robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);
            std::cout << "Init completed!\n";

            double t = 0;
            double t0 = utils::nowtime();
            double t_prev = 0;
            double dt = 0.002;

            bool is_init_state_1 = false;

            //////////////////////
            //////////////////////
            //////////////////////
            while (0 == lcm.handle() || (!has_converged && !final_quit)) {
                // timers
                double t_start = vpTime::measureTimeMs();
                const int64_t utime = utils::micros();  // for lcm msg
                t = utils::nowtime() - t0;              // for control system
                dt = t - t_prev;
                t_prev = t;

                if (is_init == true) {
                    for (int i = 0; i < kNumJoints; i++) {
                        Q_DES_0(i) = q(i);
                    }
                    q_delta.setZero();
                    is_init = false;
                    is_ramp = true;
                    std::cout << "Current robot position in JS is " << Q_DES_0.transpose() << std::endl;
                }

                rs2::frameset data_robot;
                if (mcamera.pipe_robot->poll_for_frames(&data_robot)) {
                    mcamera.frame_to_mat(data_robot.get_infrared_frame(1), mcamera.mat_robot);
                    vpImageConvert::convert(mcamera.mat_robot, mcamera.I_robot);
                    // vpDisplay::display(mcamera.I_robot);
                    // vpDisplay::flush(mcamera.I_robot);
                }

                rs2::frameset data_fly;
                if (mcamera.pipe_fly->poll_for_frames(&data_fly)) {
                    mcamera.getColorFrame(data_fly.get_color_frame(), mcamera.I_fly);
                    vpDisplay::display(mcamera.I_fly);
                    vpDisplay::flush(mcamera.I_fly);
                }

                vpDisplay::display(mcamera.I_robot);

                std::vector<vpHomogeneousMatrix> cMo_vec;
                detector.detect(mcamera.I_robot, opt_tagSize, cam, cMo_vec);

                {
                    std::stringstream ss;
                    ss << "Left click to " << (send_velocities ? "stop the robot" : "servo the robot") << ", right click to quit.";
                    vpDisplay::displayText(mcamera.I_robot, 20, 20, ss.str(), vpColor::red);
                }

                vpColVector v_c(6);

                if (!has_converged) {
                    // Only one tag is detected
                    if (cMo_vec.size() == 1) {
                        cMo = cMo_vec[0];

                        // // experiment stuff
                        // vpHomogeneousMatrix Hrot;
                        // Hrot.buildFrom(0, 0, 0, 0, 0, -M_PI/2);

                        // cMo = Hrot * cMo;

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
                            vpDisplay::displayText(mcamera.I_robot, corners[i] + vpImagePoint(15, 15), ss.str(), vpColor::red);
                            // Display desired point indexes
                            vpImagePoint ip;
                            vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), ip);
                            vpDisplay::displayText(mcamera.I_robot, ip + vpImagePoint(15, 15), ss.str(), vpColor::red);
                        }
                        if (first_time) {
                            traj_corners = new std::vector<vpImagePoint>[corners.size()];
                        }
                        // Display the trajectory of the points used as features
                        // display_point_trajectory(mcamera.I_robot, corners, traj_corners);

                        // if (opt_plot) {
                        //     plotter->plot(0, iter_plot, task.getError());
                        //     plotter->plot(1, iter_plot, v_c);
                        //     iter_plot++;
                        // }

                        if (opt_verbose) {
                            std::cout << "v_c: " << v_c.t() << std::endl;
                        }

                        double error = task.getError().sumSquare();
                        std::stringstream ss;
                        ss << "error: " << error;
                        vpDisplay::displayText(mcamera.I_robot, 20, static_cast<int>(mcamera.I_robot.getWidth()) - 150, ss.str(), vpColor::red);

                        if (opt_verbose)
                            std::cout << "error: " << error << std::endl;

                        if (error < convergence_threshold) {
                            has_converged = true;
                            std::cout << "Servo task has converged"
                                      << "\n";
                            vpDisplay::displayText(mcamera.I_robot, 100, 20, "Servo task has converged", vpColor::red);
                        }
                        if (first_time) {
                            first_time = false;
                        }
                    }  // end if (cMo_vec.size() == 1)
                    else {
                        v_c = 0;
                    }

                    if (!send_velocities) {
                        v_c = 0;
                    }

                    // set velocities

                    Eigen::Matrix<double, 6, 1> v_c_copy;
                    for (int i = 0; i < 6; i++) {
                        v_c_copy(i) = v_c[i];
                    }

                    Eigen::Matrix<double, 6, 6> Projection;
                    Projection.diagonal() << 1, 1, 0, 1, 1, 1;
                    v_c_copy = Projection * v_c_copy;

                    for (int j = 0; j < 6; j++) {
                        if (abs(v_c_copy(j)) > 0.02) {
                            v_c_copy(j) = copysign(0.02, v_c_copy(j));
                        }
                    }

                    // Send to the robot
                    set_robot_velocity(v_c_copy, t, dt, utime);

                    if (opt_plot) {
                        plotter->plot(0, iter_plot, task.getError());
                        plotter->plot(1, iter_plot, v_c);
                        plotter->plot(2, iter_plot, taus_plot);
                        iter_plot++;
                    }

                    std::cout << v_c_copy << std::endl;

                    {
                        std::stringstream ss;
                        ss << v_c_copy;
                        vpDisplay::displayText(mcamera.I_robot, 80, 20, ss.str(), vpColor::green);
                    }

                    {
                        std::stringstream ss;
                        ss << "Loop time: " << vpTime::measureTimeMs() - t_start << " ms";
                        vpDisplay::displayText(mcamera.I_robot, 40, 20, ss.str(), vpColor::red);
                    }
                    vpDisplay::flush(mcamera.I_robot);

                    vpMouseButton::vpMouseButtonType button;
                    if (vpDisplay::getClick(mcamera.I_robot, button, false)) {
                        switch (button) {
                            case vpMouseButton::button1:
                                send_velocities = !send_velocities;
                                if (abs(Q_DES_0.sum() - q.sum()) > 0.001) {
                                    is_init = true;
                                }
                                break;

                            case vpMouseButton::button3:
                                final_quit = true;
                                v_c = 0;
                                break;

                            default:
                                break;
                        }
                    }
                } else {
                    std::cout << "Converged!" << std::endl;
                    if (!is_init_state_1) {
                        double t0 = utils::nowtime();
                        is_init_state_1 = true;
                    }
                    Eigen::Matrix<double, 6, 1> v_c_copy;
                    v_c_copy.setZero();
                    set_robot_velocity(v_c_copy, t, dt, utils::micros());
                }
            }

            std::cout << "Stop the robot " << std::endl;
            Eigen::Matrix<double, 6, 1> v_c_copy;
            v_c_copy.setZero();
            set_robot_velocity(v_c_copy, 0, 0.002, utils::micros());

            if (opt_plot && plotter != nullptr) {
                delete plotter;
                plotter = nullptr;
            }

            // if (!final_quit) {
            //     while (!final_quit) {
            //         // rs.acquire(I);
            //         vpDisplay::display(I);

            //         vpDisplay::displayText(I, 20, 20, "Click to quit the program.", vpColor::red);
            //         vpDisplay::displayText(I, 40, 20, "Visual servo converged.", vpColor::red);

            //         if (vpDisplay::getClick(I, false)) {
            //             final_quit = true;
            //         }

            //         vpDisplay::flush(I);
            //     }
            // }
            if (traj_corners) {
                delete[] traj_corners;
            }
        } catch (const vpException &e) {
            std::cout << "ViSP exception: " << e.what() << std::endl;
            std::cout << "Stop the robot " << std::endl;
            Eigen::Matrix<double, 6, 1> v_c_copy;
            v_c_copy.setZero();
            set_robot_velocity(v_c_copy, 0, 0.002, utils::micros());
            return EXIT_FAILURE;
        } catch (const std::exception &e) {
            std::cout << "Exception: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    int set_robot_velocity(const Eigen::Matrix<double, 6, 1> &v_c, double t, double dt = 0.002, int64_t utime = 0) {
        tau_fb.setZero();

        bool is_v_c_zero = true;
        for (int i = 0; i < 6; i++) {
            if (abs(v_c(i)) > 0.000001) {
                is_v_c_zero = false;
                break;
            }
        }

        // Jacobian test
        Eigen::Matrix<double, 6, 1> v_c_copy;

        // v_c_copy << 0.0, 0.00, 0.00, 0.00, change_sign * 0.08, 0.00;
        v_c_copy << 0.0, 0.00, 0.00, 0.00, 0.08 * sin(0.5 * t), 0.00;
        is_v_c_zero = false;

        if (!is_v_c_zero) {
            // make adjoint
            Eigen::Matrix<double, 6, 6> fVe;  // world to ee
            iiwa_kinematics::forwarkKinematics(quat, fpe, q);
            fRe = quat.normalized().toRotationMatrix();
            utils::make_adjoint(fVe, fpe, fRe);

            // get J w.r.t ee and inverse it
            iiwa_kinematics::jacobian(J, q);
            J = fVe * J;
            utils::pinv2(J, pinvJ);

            // test to condition number of inversed Jacobian
            Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(pinvJ);
            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
            if (cond > 50) {
                std::cerr << "[ERROR] COND JACOBI: " << cond << std::endl;
            }

            // velocity kinematics
            dq_des = pinvJ * v_c_copy;
            // dq_des = pinvJ * v_c;
            q_delta += dq_des * dt;

        } else {
            q_delta.setZero();
        }

        // q_delta.setZero();
        q_des = Q_DES_0;  // + q_delta;
        tau_fb = 1000 * K * (q_des - q) - 20 * B * dq;

        log_file << tau_fb.transpose() << " " << (q_des - q).transpose() << " " << dq.transpose() << " " << q_delta.transpose() << " " << t << "\n";

        publish(tau_fb, utime);

        for (int i = 0; i < kNumJoints; i++)
            taus_plot[i] = tau_fb(i);

        // // jerk compensation
        // if (is_ramp) {
        //     tau_fb0 = tau_fb;
        //     is_ramp = false;
        // }

        // Eigen::Matrix<double, 7, 1> tau_cmd;
        // tau_cmd = tau_fb - tau_fb0 * exp(- 4 * t);

        // publish(tau_cmd, utime);

        return 0;
    }

    int test_set_robot_velocity(const Eigen::Matrix<double, 6, 1> &v_c, double t, double dt = 0.002, int64_t utime = 0) {
        tau_fb.setZero();

        /* stabilisation test*/
        // {
        //     q_des = Q_DES_0;
        //     tau_fb = 700 * K * (q_des - q) - 40 * B * dq;
        // }

        /* low speed traking test*/
        // {
        //     q_delta << 0, 0, 0, 0, 0, 0.4 * sin(t), 0;
        //     q_des = Q_DES_0 + q_delta;
        //     tau_fb = 700 * K * (q_des - q) - 40 * B * dq;
        // }

        /* low speed traking test 2*/
        // {
        //     q_delta << 0, 0, 0, 0, 0, 0.4 * sin(t), 0.4 * sin(t);
        //     q_des = Q_DES_0 + q_delta;
        //     tau_fb = 700 * K * (q_des - q) - 40 * B * dq;
        // }

        /* add big delta to `q` test*/
        // {
        //     q_delta << 0, 0, 0, 0, 0, 0.5, 0;
        //     q_des = Q_DES_0 + q_delta;
        //     tau_fb = 700 * K * (q_des - q) - 40 * B * dq;
        // }


        /**/
        /* Add big delta to `q` with "ramped" `q_des` test*/
        /**/
        // {
        //     std::stringstream ss; ss << t << "(ramp_state: "<< ramp_state << ")\t";
        //     q_delta << 0, 0, 0, 0, 0, 1.0, 0;
        //     q_des = Q_DES_0 + q_delta;
        //     q_error = q_des - q;
        //     // int ramp_state = 0; // [0, 1, 2] -- no ramp, init, ramp -- it's a global variable

        //     // check if q_error too big, then turn off ramp
        //     for (int i = 0; i < kNumJoints; i++) {
        //         // if ((tau_fb(i) - tau_fb_prev(i)) > 3.0) {
        //         if ((abs(q_error[i]) > 0.3) && ramp_state == 0 ) { // (t - t_ramp > 2.0)WARNING: hardcoded `5` depends to exponentiol ramp parameter
        //             // init_ramp = true;
        //             ramp_state = 1;
        //             ss << "too big q detected (0) \t";
        //             break;
        //         }
        //     }

        //     // initialize ramp parameters
        //     if (ramp_state == 1) {
        //         ss << "ramp init (1)\t";
        //         ramp_q_error = q_des - q;
        //         t_ramp0 = t;
        //         Eigen::Matrix<double, 7, 1> t_ramp_vec;
        //         t_ramp = -999;
        //         for (int i = 0; i < kNumJoints; i++) {
        //             double val = ((q_des(i) - q(i)) / ramp_q_error(i));
        //             if (val > 0) {
        //                 t_ramp_vec(i) =  log(val) / 6;
        //             } else {
        //                 t_ramp_vec(i) = 2.0; // WARNING: what if `val` is negative? check it!
        //             }
        //             if (t_ramp < t_ramp_vec(i)) {
        //                     t_ramp = t_ramp_vec(i);
        //             }
        //         }
        //         ramp_state = 2;
        //         // init_ramp = false;
        //     }

        //     if (ramp_state == 2) {
        //         if (t - t_ramp0 > t_ramp) {
        //             ramp_state = 0;
        //             ss << "ramp_state -> 0 \t";
        //         }
                
        //         q_error = q_des - q - ramp_q_error * exp( -6 * (t - t_ramp0)); // `1`-`1.5` seconds smoth increasing (bigger value`6` -> larger smooth transietn)
        //     }
            
        //     // q_des_prev = q_des;

        //     // ss << "compute control\t\n";
        //     tau_fb = 700 * K * q_error - 40 * B * dq;
        //     std::cout << ss.str() << std::endl;
        // }

        /**/
        /* Cartesian velocity with last works test*/
        /**/
        {
            // std::stringstream ss; ss << t << "(ramp_state: "<< ramp_state << ")\t";
            Eigen::Matrix<double, 6, 1> v_c;

            v_c << 0.0, 0.0, 0.0, 0.3 * sin(t), 0.3 * cos(t), 0.0;

            // make adjoint
            Eigen::Matrix<double, 6, 6> fVe;  // world to ee
            iiwa_kinematics::forwarkKinematics(quat, fpe, q);
            fRe = quat.normalized().toRotationMatrix();
            utils::make_adjoint(fVe, fpe, fRe);

            // get J w.r.t ee and inverse it
            iiwa_kinematics::jacobian(J, q);
            J = fVe * J;
            utils::pinv2(J, pinvJ);

            // test to condition number of inversed Jacobian
            Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(pinvJ);
            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
            if (cond > 20) {
                std::cerr << "[ERROR] COND JACOBI: " << cond << std::endl;
            }

            // velocity kinematics
            dq_des = pinvJ * v_c;
            q_delta += dq_des * dt;

            q_des = Q_DES_0 + q_delta;
            q_error = q_des - q;
            // int ramp_state = 0; // [0, 1, 2] -- no ramp, init, ramp -- it's a global variable

            // check if q_error too big, then turn off ramp
            for (int i = 0; i < kNumJoints; i++) {
                // if ((tau_fb(i) - tau_fb_prev(i)) > 3.0) {
                if ((abs(q_error[i]) > 0.3) && ramp_state == 0 ) { // (t - t_ramp > 2.0)WARNING: hardcoded `5` depends to exponentiol ramp parameter
                    // init_ramp = true;
                    ramp_state = 1;
                    break;
                }
            }

            // initialize ramp parameters
            if (ramp_state == 1) {
                ramp_q_error = q_des - q;
                t_ramp0 = t;
                Eigen::Matrix<double, 7, 1> t_ramp_vec;
                t_ramp = -999;
                for (int i = 0; i < kNumJoints; i++) {
                    double val = ((q_des(i) - q(i)) / ramp_q_error(i));
                    if (val > 0) {
                        t_ramp_vec(i) =  log(val) / 6;
                    } else {
                        t_ramp_vec(i) = 2.0; // WARNING: what if `val` is negative? check it!
                    }
                    if (t_ramp < t_ramp_vec(i)) {
                            t_ramp = t_ramp_vec(i);
                    }
                }
                ramp_state = 2;
                // init_ramp = false;
            }

            if (ramp_state == 2) {
                if (t - t_ramp0 > t_ramp) {
                    ramp_state = 0;
                }
                q_error = q_des - q - ramp_q_error * exp( -6 * (t - t_ramp0)); // `1`-`1.5` seconds smoth increasing (bigger value`6` -> larger smooth transietn)
            }
            
            tau_fb = 700 * K * q_error - 40 * B * dq;
        }


        
        log_file << tau_fb.transpose() << " " << q_error.transpose() << " " << dq.transpose() << " " << q_delta.transpose() << " " << t << "\n";
        publish(tau_fb, utime);

        return 0;
    }

    void test_ctrl() {
        if (!lcm.good())
            std::cout << "!!!!!!!!!!";

        lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

        bool init = true;

        std::cout << "Loop started vel cart" << std::endl;
        double t0 = utils::nowtime();
        double t_prev = 0;
        while (0 == lcm.handle()) {
            double loop_time = vpTime::measureTimeMs();

            const int64_t utime = utils::micros();
            double t = utils::nowtime() - t0;
            double dt = t - t_prev;
            t_prev = t;
            // double freq = 1.0 / dt;

            if (init == true) {
                init = false;
                for (int i = 0; i < 7; i++) {
                    q_des(i) = lcm_status.joint_position_measured[i];
                    Q_DES_0(i) = lcm_status.joint_position_measured[i];
                }
                q_des_prev= Q_DES_0;
            }
            Eigen::Matrix<double, 6, 1> v_c_copy;
            v_c_copy.setZero();
            test_set_robot_velocity(v_c_copy, t, dt, utils::micros());

            std::cout << "Loop time: " << vpTime::measureTimeMs() - loop_time << std::endl;
        }
    }
};

int main(int argc, char **argv) {
    std::cout << "Node started" << std::endl;

    Controller controller;
    // controller.loop_vs();
    // controller.loop_vs_april();
    controller.test_ctrl();

    return 0;
}