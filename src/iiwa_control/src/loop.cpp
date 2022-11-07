#include <loop.hpp>

#include "iiwa_kinematics.hpp"
// #include "multicamera.hpp"
#include "utils.hpp"

VelocityController::VelocityController() {
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

    vel_sub = nh.subscribe("iiwa_vel", 10, &VelocityController::vel_callback, this);
    state_sub = nh.subscribe("iiwa_state", 10, &VelocityController::state_callback, this);

    feedback_state = 0;

    v_c.setZero();

    Projection.diagonal() << 1, 1, 0, 1, 1, 1;

    ROS_INFO("Velocity controller initialised.");
}

VelocityController::~VelocityController() {
    // TODO stop robot
}

void VelocityController::handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg) {
    mtx.lock();
    lcm_status = *msg;
    for (int i = 0; i < 7; i++) {
        q(i) = lcm_status.joint_position_measured[i];
        dq(i) = lcm_status.joint_velocity_estimated[i];
    }
    mtx.unlock();
}

void VelocityController::vel_callback(const geometry_msgs::Twist &msg) {
    mtx.lock();
    v_c(0) = msg.linear.x;
    v_c(1) = msg.linear.y;
    v_c(2) = msg.linear.z;
    v_c(3) = msg.angular.x;
    v_c(4) = msg.angular.y;
    v_c(5) = msg.angular.z;
    mtx.unlock();
}

void VelocityController::state_callback(const std_msgs::Int32 &msg) {
    mtx.lock();
    feedback_state = msg.data;
    mtx.unlock();
}

/* Publish command to robot */
void VelocityController::publish(Eigen::Matrix<double, 7, 1> &tau_fb, int64_t utime) {
    std::cout << "tau_fb::";
    for (int i = 0; i < 7; i++) {
        std::cout << std::setw(2) << tau_fb(i) << std::setprecision(8) <<  " \t";
    }
    std::cout << std::endl;

    mtx.lock();

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

int VelocityController::set_robot_velocity(const Eigen::Matrix<double, 6, 1> &v_c, double t, double dt, int64_t utime) {
    tau_fb.setZero();

    /**/
    /* Cartesian velocity with last works test*/
    /**/
    {
        // std::stringstream ss; ss << t << "(ramp_state: "<< ramp_state << ")\t";
        // Eigen::Matrix<double, 6, 1> v_c;

        // v_c << 0.0, 0.0, 0.0, 0.3 * sin(t), 0.3 * cos(t), 0.0;

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

        // // check if q_error too big, then turn off ramp
        // for (int i = 0; i < kNumJoints; i++) {
        //     // if ((tau_fb(i) - tau_fb_prev(i)) > 3.0) {
        //     if ((abs(q_error[i]) > 0.3) && ramp_state == 0) {  // (t - t_ramp > 2.0)WARNING: hardcoded `5` depends to exponentiol ramp parameter
        //         // init_ramp = true;
        //         ramp_state = 1;
        //         break;
        //     }
        // }

        // // initialize ramp parameters
        // if (ramp_state == 1) {
        //     ramp_q_error = q_des - q;
        //     t_ramp0 = t;
        //     Eigen::Matrix<double, 7, 1> t_ramp_vec;
        //     t_ramp = -999;
        //     for (int i = 0; i < kNumJoints; i++) {
        //         double val = ((q_des(i) - q(i)) / ramp_q_error(i));
        //         if (val > 0) {
        //             t_ramp_vec(i) = log(val) / 6;
        //         } else {
        //             t_ramp_vec(i) = 2.0;  // WARNING: what if `val` is negative? check it!
        //         }
        //         if (t_ramp < t_ramp_vec(i)) {
        //             t_ramp = t_ramp_vec(i);
        //         }
        //     }
        //     ramp_state = 2;
        //     // init_ramp = false;
        // }

        // if (ramp_state == 2) {
        //     if (t - t_ramp0 > t_ramp) {
        //         ramp_state = 0;
        //     }
        //     q_error = q_des - q - ramp_q_error * exp(-6 * (t - t_ramp0));  // `1`-`1.5` seconds smoth increasing (bigger value`6` -> larger smooth transietn)
        // }

        tau_fb = 1000 * K * q_error - 40 * B * dq;
    }

    // log_file << tau_fb.transpose() << " " << q_error.transpose() << " " << dq.transpose() << " " << q_delta.transpose() << " " << t << "\n";
    publish(tau_fb, utime);

    return 0;
}

void VelocityController::loop() {
    try {
        if (!lcm.good())
            std::cout << "lcm problem ad lcm.good()\n";

        lcm.subscribe(kLcmStatusChannel, &VelocityController::handleFeedbackMessage, this);

        bool init = true;

        std::cout << "Loop started vel cart" << std::endl;
        double t0 = utils::nowtime();
        double t_prev = 0;

        ros::Rate R(600);
        while (nh.ok()) {
            double start_time = ros::Time::now().toSec();

            const int64_t utime = utils::micros();
            double t = utils::nowtime() - t0;
            double dt = t - t_prev;
            t_prev = t;

            if (0 == lcm.handle()) {
                if (init == true) {
                    init = false;
                    for (int i = 0; i < 7; i++) {
                        q_des(i) = lcm_status.joint_position_measured[i];
                        Q_DES_0(i) = lcm_status.joint_position_measured[i];
                    }
                    q_des_prev = Q_DES_0;
                }

                if (feedback_state == 1) {
                    v_c = Projection * v_c;
                    set_robot_velocity(v_c, t, dt, utils::micros());
                } else if (feedback_state == 2) {
                    set_robot_velocity(v_c, t, dt, utils::micros());
                } else {
                    Eigen::Matrix<double, 6, 1> v_c;
                    v_c.setZero();
                    set_robot_velocity(v_c, t, dt, utils::micros());
                }
            }

            ros::spinOnce();
            R.sleep();
            std::cout << "freq: " << 1 / (ros::Time::now().toSec() - start_time) << "\t state: " << feedback_state << std::endl;
        }
    } catch (...) {
        std::cout << "Try...catch\n";
    }
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "velocity_controller_node");

    VelocityController vc;
    vc.loop();

    return 0;
}