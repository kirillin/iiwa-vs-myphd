
#include <stdio.h>
#include <iostream>
#include <lcm/lcm-cpp.hpp>
#include <cassert>
#include <chrono>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"

using drake::lcmt_iiwa_status;
using drake::lcmt_iiwa_command;

const char* kLcmStatusChannel = "IIWA_STATUS";
const char* kLcmCommandChannel = "IIWA_COMMAND";
const int kNumJoints = 7;

typedef Eigen::Matrix<double, kNumJoints, 1> Vector7d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

class Controller {

	// robot state
	Eigen::Matrix<double, kNumJoints, 1>  q;
	Eigen::Matrix<double, kNumJoints, 1>  dq;

	// stiffness and damping parameters
	Eigen::Matrix<double, kNumJoints, 1>  k;
	Eigen::Matrix<double, kNumJoints, 1>  b;
	Eigen::Matrix<double, kNumJoints, kNumJoints>  K;
	Eigen::Matrix<double, kNumJoints, kNumJoints>  B;

	// robot desired position
	// Eigen::Matrix<double, kNumJoints, 1>  q_des;
	Vector7d error;

	lcmt_iiwa_status lcm_status{};
	lcmt_iiwa_command lcm_command{};

	lcm::LCM lcm;

 public:
	Controller(){

		q.setZero();
		dq.setZero();
		K.setZero();
		B.setZero();

		K.diagonal() << 2, 2, 2, 1, 0.5, 0.5, 0.25;
		K = K * 50;
		B.diagonal() << 2, 2, 2, 1, 0.5, 0.5, 0.25;
		B = B * 0;
	}
	~Controller() {};

	void handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan,
						const drake::lcmt_iiwa_status *msg) {
		lcm_status = *msg;

		for (int i = 0; i < kNumJoints; i++) {
			q[i] = lcm_status.joint_position_measured[i];
			dq[i] = lcm_status.joint_velocity_estimated[i];
		}

		// Vector7d q_des; q_des << 1.9554,	-0.494669,	-0.121013,	1.27,	-0.0204903,	0.0,	-0.0243785;
		// error = q_des - q;
		// Vector7d tau = K * error - B * dq;

		// lcm_command.utime = micros();
		// lcm_command.num_joints = kNumJoints;
 		// lcm_command.num_torques = kNumJoints;
		// lcm_command.joint_position.resize(kNumJoints, 0);
		// lcm_command.joint_torque.resize(kNumJoints, 0);
		// for (int i = 0; i < kNumJoints; i++ ) {
		// 	lcm_command.joint_position[i] = lcm_status.joint_position_measured[i];
		// 	lcm_command.joint_torque[i] = tau[i];
		// 	// std::cout << lcm_command.joint_torque[i] << "\t" << std::endl;
		// }

		// lcm.publish("IIWA_COMMAND", &lcm_command);
	}

	/*
		Impedance controller
			- read: q, dq, q_des
			- u = K (q_des - q) - B dq
	*/
	Vector7d pd_controller(const Vector7d &q_des) {
		for (int i = 0; i < kNumJoints; i++) {
			q[i] = lcm_status.joint_position_measured[i];
			dq[i] = lcm_status.joint_velocity_estimated[i];
		}
		error = q_des - q;
		Vector7d tau = K * error - B * dq;
		return tau;
	}

	void print_vector(const Vector7d vec) {
		for (int i = 0; i < kNumJoints; i++) {
			std::cout << vec[i] << "\t";
		}
		std::cout << std::endl;
	}

	void print_matrix() {
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				std::cout << K(i,j) << "\t";
			}
			std::cout << std::endl;
		}
	}


	lcmt_iiwa_command update(int64_t utime, Eigen::Matrix<double, kNumJoints, 1>  q_des) {
		// Vector7d* u;
		Vector7d u = pd_controller(q_des);
						
		lcm_command.utime = utime;
		lcm_command.num_joints = kNumJoints;
 		lcm_command.num_torques = kNumJoints;
		lcm_command.joint_position.resize(kNumJoints, 0);
		lcm_command.joint_torque.resize(kNumJoints, 0);
		for (int i = 0; i < kNumJoints; i++ ) {
			lcm_command.joint_position[i] = lcm_status.joint_position_measured[i];
			lcm_command.joint_torque[i] = u[i];
			// std::cout << lcm_command.joint_torque[i] << "\t" << std::endl;
		}
		return lcm_command;
	}

	int64_t micros() {
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

};



double nowtime() {
	auto current_time = std::chrono::system_clock::now();
	auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
	double num_seconds = duration_in_seconds.count();
	return  num_seconds;
}



int main(int argc, char **argv)
{
	lcm::LCM lcm;

	if (!lcm.good())
		return 1;
	
	std::cout << "Node started" << std::endl;

	Controller controller;
	controler.loop();
	
	lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, &controller);

	double t0 = nowtime();
	double t_prev = 0;
	while (0 == lcm.handle()) {
		// double t = nowtime() - t0;
		// double dt = t - t_prev;
		// t_prev = t;
		// double freq = 1.0 / dt;
		// std::cout << t << "\t(" << dt <<  ")\t freq: " << freq << std::endl;

		// const int64_t utime = micros();
		// Vector7d q_des; q_des << 1.9554,	-0.494669,	-0.121013,	1.27,	-0.0204903,	0.0,	-0.0243785;
		// lcmt_iiwa_command cmd = controller.update(utime, q_des);
		// lcm.publish("IIWA_COMMAND", &cmd);

	}

	return 0;
}
