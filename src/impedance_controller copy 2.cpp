
#include <fstream>
#include <visp3/core/vpImage.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/gui/vpDisplayX.h>

#include <visp3/vision/vpPose.h>
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpPixelMeterConversion.h>

#include <visp3/imgproc/vpImgproc.h>
#include <visp3/core/vpImageFilter.h>

#include <visp3/blob/vpDot2.h>

#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/vs/vpServo.h>

#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/io/vpImageIo.h>

#include <boost/units/io.hpp>
#include <boost/units/systems/si/angular_velocity.hpp>
#include <boost/units/systems/si/velocity.hpp>
#include <boost/units/systems/angle/degrees.hpp>
#include <boost/units/conversion.hpp>


#include <stdio.h>
#include <iostream>
#include <lcm/lcm-cpp.hpp>
#include <cassert>
#include <chrono>
#include <cmath>

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

	bool DEBUG = false;

	double START_TIME = 0;
	int RATE = 30;
	int Np = 4;

	double opt_square_width = 0.06;
	double L = opt_square_width / 2.;
	double distance_same_blob = 10.; // 2 blobs are declared same if their distance is less than this value

	Vector6d v_des;

	// std::ofstream file_joint_state;
	// std::ofstream file_vs;

	double l = 0;
	double vcx = 0, vcy = 0, vcz = 0;


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

		v_des << 0, 0, 0, 0, 0, 0;
	}
	~Controller() {};

	void handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan,
						const drake::lcmt_iiwa_status *msg) {
		lcm_status = *msg;
		// std::cout << "handler " << nowtime() << std::endl;
	}


	

	/*
	* Pseudo inverse image jacobian.
	*/
	void pinv(Eigen::Matrix<double, 8, 6>& L, Eigen::Matrix<double, 6, 8>& pinvL, double alpha0 = 0.001, double w0 = 0.0001) {
			double w = 0, alpha = 0;

			double detL = (L * L.transpose()).determinant();
			if (detL < 1.0e-10) {
					w = 1.0e-5;
			} else {
					w = sqrt(detL);
			}

			if (w >= w0) {
					alpha = 0;
			} else {
					alpha = alpha0 * (1.0 - w / w0) * (1 - w / w0);
			}

			// 6x8 = 6x8 * (8x6 * 6x8)
			pinvL = L.transpose() * (L * L.transpose() - alpha * Eigen::MatrixXd::Identity(8, 8)).inverse();
	}

	void compute_error(std::vector<vpImagePoint>& ip, std::vector<vpImagePoint>& ipd, Eigen::Matrix<double, 8, 1>& error) {
			error = Eigen::Matrix<double, 8, 1>::Zero();
			for (int i = 0; i < Np; i++) {
					// error in image space for each point
					double ex = ip[i].get_u() - ipd[i].get_u();
					double ey = ip[i].get_v() - ipd[i].get_v();
					error(0 + 2 * i) = ex;
					error(1 + 2 * i) = ey;
			}
	}


	/* Impedance controller tau = K (q_des - q) - B dq */
	Vector7d pd_controller(const Vector7d &q_des) {
		for (int i = 0; i < kNumJoints; i++) {
			q[i] = lcm_status.joint_position_measured[i];
			dq[i] = lcm_status.joint_velocity_estimated[i];
		}
		error = q_des - q;
		Vector7d tau = K * error - B * dq;
		return tau;
	}
	/* Gravity compensation tau = 0 */
	Vector7d gravity_controller() {
		Vector7d tau; tau.setZero();
		return tau;
	}


	lcmt_iiwa_command update(int64_t utime, double t, double dt) {
		
		// IC
		Vector7d q_des; q_des << 1.9554,	-0.494669,	-0.121013,	1.27,	0.0,	0.0,	0.0;
		Vector7d u = pd_controller(q_des);
		
		// Gravity
		// Vector7d u = gravity_controller();
						
		lcm_command.utime = utime;
		lcm_command.num_joints = kNumJoints;
 		lcm_command.num_torques = kNumJoints;
		lcm_command.joint_position.resize(kNumJoints, 0);
		lcm_command.joint_torque.resize(kNumJoints, 0);
		for (int i = 0; i < kNumJoints; i++ ) {
			lcm_command.joint_position[i] = lcm_status.joint_position_measured[i];
			lcm_command.joint_torque[i] = u[i];
		}
		return lcm_command;
	}

	int64_t micros() {
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

	double nowtime() {
		auto current_time = std::chrono::system_clock::now();
		auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
		double num_seconds = duration_in_seconds.count();
		return  num_seconds;
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

	int loop() {
		
		if (!lcm.good())
			return 1;
		lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

		std::cout << "Loop started" << std::endl;
		double t0 = nowtime();
		double t_prev = 0;
		while (0 == lcm.handle()) {
			// std::cout << "loop " << nowtime() << std::endl;
			double t = nowtime() - t0;
			double dt = t - t_prev;
			t_prev = t;
			double freq = 1.0 / dt;
			std::cout << t << "\t(" << dt <<  ")\t freq: " << freq << std::endl;

			const int64_t utime = micros();
			lcmt_iiwa_command cmd = update(utime, t, dt);
			lcm.publish("IIWA_COMMAND", &cmd);
		}
		return 0;
	}

};


int main(int argc, char **argv) {
	// double lambda = std::atof(argv[1]);

	// double vcx = std::atof(argv[2]);
	// double vcy = std::atof(argv[3]);
	// double vcz = std::atof(argv[4]);
	std::cout << "Node started" << std::endl;

	Controller controller;
	int code = controller.loop();
	
	return code;
}
