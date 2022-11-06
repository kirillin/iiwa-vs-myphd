#ifndef LOOP_H
#define LOOP_H

#include <ros/ros.h>
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <algorithm>
#include <boost/units/conversion.hpp>
#include <boost/units/io.hpp>
#include <boost/units/systems/angle/degrees.hpp>
#include <boost/units/systems/si/angular_velocity.hpp>
#include <boost/units/systems/si/velocity.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <lcm/lcm-cpp.hpp>
#include <mutex>  // std::mutex
#include <vector>

#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"

#include "std_msgs/Int32.h"
#include "geometry_msgs/Twist.h"



using namespace std;
using namespace Eigen;

using drake::lcmt_iiwa_command;
using drake::lcmt_iiwa_status;

const char *kLcmStatusChannel = "IIWA_STATUS";
const char *kLcmCommandChannel = "IIWA_COMMAND";
const int kNumJoints = 7;

class VelocityController {
    ros::NodeHandle nh;

    ros::Subscriber vel_sub;
    ros::Subscriber state_sub;

    lcm::LCM lcm;
    lcmt_iiwa_status lcm_status{};
    lcmt_iiwa_command lcm_command{};

    Matrix<double, 7, 1> q;
    Matrix<double, 7, 1> dq;

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

    std::mutex mtx;
    Eigen::Matrix<double, 6, 1> v_c;
    Eigen::Matrix<double, 6, 6> Projection;

    double t_ramp;
    double t_ramp0;
    bool init_ramp;
    int ramp_state;
    
    bool DEBUG = true;
    bool is_init = true;

    int feedback_state;

   public:
    VelocityController();
    ~VelocityController();

    void handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg);

    void vel_callback(const geometry_msgs::Twist &msg);
    void state_callback(const std_msgs::Int32 &msg);

    void publish(Eigen::Matrix<double, 7, 1> &tau_fb, int64_t utime);
    int set_robot_velocity(const Eigen::Matrix<double, 6, 1> &v_c, double t, double dt, int64_t utime);
    void loop();
};

#endif
