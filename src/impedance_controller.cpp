
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
#include <visp3/core/vpImageConvert.h>

#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/io/vpImageIo.h>

#include <boost/units/io.hpp>
#include <boost/units/systems/si/angular_velocity.hpp>
#include <boost/units/systems/si/velocity.hpp>
#include <boost/units/systems/angle/degrees.hpp>
#include <boost/units/conversion.hpp>
#include <boost/scoped_ptr.hpp>

#include <stdio.h>
#include <iostream>
#include <lcm/lcm-cpp.hpp>
#include <cassert>
#include <chrono>
#include <thread>
#include <cmath>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"

#include "kinematics.h"
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>
#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntspaceinertiamatrix.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/treejnttojacsolver.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>

 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

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
	Eigen::Matrix<double, kNumJoints, 1>  q_des;
	
	Kinematics ks;

	lcmt_iiwa_status lcm_status{};
	lcmt_iiwa_command lcm_command{};

	lcm::LCM lcm;

	bool DEBUG = false;

	double START_TIME = 0;
	int RATE = 30;
	int Np = 4;
	Eigen::Matrix<double, 8, 8>  Klambda;
	

	double opt_square_width = 0.06;
	double L = opt_square_width / 2.;
	double distance_same_blob = 10.; // 2 blobs are declared same if their distance is less than this value

	Vector6d v_des;

	// std::ofstream file_joint_state;
	// std::ofstream file_vs;

	double l = 0;
	double vcx = 0, vcy = 0, vcz = 0;

	boost::scoped_ptr<KDL::ChainFkSolverPos> jnt_to_pose_solver;
	boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver;
	
	// The variables (which need to be pre-allocated).                                                                                                                              
	KDL::JntArray  kdl_q;            // Joint positions                                                                                                                                
	KDL::JntArray  kdl_q0;           // Joint initial positions                                                                                                                        
	KDL::JntArray  kdl_dq;      // Joint velocities                                                                                                                               
	KDL::JntArray  kdl_tau;          // Joint torques                                                                                                                                  

	KDL::Frame     kdl_x;            // Tip pose                                                                                                                                       
	KDL::Frame     kdl_xd;           // Tip desired pose                                                                                                                               
	KDL::Frame     kdl_x0;           // Tip initial pose                                                                                                                               

	KDL::Twist     kdl_xerr;         // Cart error                                                                                                                                     
	KDL::Twist     kdl_dx;         // Cart velocity                                                                                                                                  
	KDL::Wrench    kdl_F;            // Cart effort                                                                                                                                    
	KDL::Jacobian  kdl_J;            // Jacobian                                                                                                                                       

	// Note the gains are incorrectly typed as a twist,                                                                                                                             
	// as there is no appropriate type!                                                                                                                                             
	KDL::Twist     kdl_Kp;           // Proportional gains                                                                                                                             
	KDL::Twist     kdl_Kd;           // Derivative gains                   

	KDL::Tree iiwa_tree;
	KDL::Chain iiwa_chain;
	std::string base_link;
	std::string tool_link;
	double circle_phase;
	Eigen::Matrix<double, 7, 1> TAU;

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

		Klambda.diagonal() << 1, 1, 1, 1, 1, 1, 1, 1;

		v_des << 0, 0, 0, 0, 0, 0;

		// Get parameters for kinematic from setup.yaml file
		std::string urdf_file = "/home/iiwa/phd_code/ws/labs-robots-files/kuka-iiwa/iiwa_model_simscape/iiwa_description/urdf/iiwa14.urdf";
		urdf::Model model;
		if (!model.initFile(urdf_file)){
			printf("Failed to parse urdf file");
		}
		
		base_link = "world";
		tool_link = "iiwa_link_ee_kuka";
		
		if (!kdl_parser::treeFromUrdfModel(model, iiwa_tree)){
			printf("Failed to construct kdl tree");
		}

		if(!iiwa_tree.getChain(base_link, tool_link, iiwa_chain)){
			printf("Failed to get KDL chain from tree ");
		}
		
		jnt_to_pose_solver.reset(new KDL::ChainFkSolverPos_recursive(iiwa_chain));
		jnt_to_jac_solver.reset(new KDL::ChainJntToJacSolver(iiwa_chain));

		// Resize (pre-allocate) the variables in non-realtime.  
		std::cout << "N of Joints " << iiwa_chain.getNrOfJoints() << std::endl;                                                                                                                       
		kdl_q.resize(iiwa_chain.getNrOfJoints());
		kdl_q0.resize(iiwa_chain.getNrOfJoints());
		kdl_dq.resize(iiwa_chain.getNrOfJoints());
		kdl_tau.resize(iiwa_chain.getNrOfJoints());
		kdl_J.resize(iiwa_chain.getNrOfJoints());

		// Pick the gains.                                                                                                                                                              
		kdl_Kp.vel(0) = 100.0;  kdl_Kd.vel(0) = 1.0;        // Translation x                                                                                                                  
		kdl_Kp.vel(1) = 100.0;  kdl_Kd.vel(1) = 1.0;        // Translation y                                                                                                                  
		kdl_Kp.vel(2) = 100.0;  kdl_Kd.vel(2) = 1.0;        // Translation z                                                                                                                  
		kdl_Kp.rot(0) = 100.0;  kdl_Kd.rot(0) = 1.0;        // Rotation x                                                                                                                     
		kdl_Kp.rot(1) = 100.0;  kdl_Kd.rot(1) = 1.0;        // Rotation y                                                                                                                     
		kdl_Kp.rot(2) = 100.0;  kdl_Kd.rot(2) = 1.0;        // Rotation z     
		circle_phase = 0;
		TAU.setZero();
		
		// // test jacobian
		// KDL::JntArray q;
		// KDL::Jacobian J;
		// q.resize(iiwa_chain.getNrOfJoints());
		// J.resize(iiwa_chain.getNrOfJoints());
		// jnt_to_jac_solver->JntToJac(q, J);

		// std::cout << J.columns() << " " << J.rows() << std::endl;			
		// for (int i = 0; i < 6; i++) {
		// 	for (int j = 0; j < 7; i++) {
		// 		std::cout << J(i, j) << " ";
		// 	}
		// 	std::cout << std::endl;
		// }

	}
	~Controller() {
		// stop robot any way
		if (!lcm.good())
			std::cout << "CAN NOT STOP THE ROBOT!" << std::endl;
		lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);
		while (0 == lcm.handle()) {
			const int64_t utime = micros();
			Eigen::Matrix<double, 7, 1> tau_fb; tau_fb.setZero();
			publish(tau_fb, utime, 0, 0);
		}
	};

	int velocity_cart_controller() {
				
		if (!lcm.good())
			return 1;

		lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

		// std::this_thread::sleep_for(std::chrono::milliseconds(x));

		bool init = true;
		
		Eigen::Matrix<double, 7, 1> dei; dei.setZero();

		std::cout << "Loop started vel cart" << std::endl;
		double t0 = nowtime();
		double t_prev = 0;
		while (0 == lcm.handle()) {
			const int64_t utime = micros();
			double t = nowtime() - t0;
			double dt = t - t_prev;
			t_prev = t;
			double freq = 1.0 / dt;
			// std::cout << t << std::endl;

			if (init == true) {
				init = false;
				for (int i = 0; i < 7; i++) {
					q_des(i) = lcm_status.joint_position_measured[i];
					kdl_q(i) = lcm_status.joint_position_measured[i];
					kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
				}
				jnt_to_pose_solver->JntToCart(kdl_q, kdl_x0);
			}

			for (int i = 0; i < kNumJoints; i++) {
				kdl_q(i) = lcm_status.joint_position_measured[i];
				kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
			}
			jnt_to_pose_solver->JntToCart(kdl_q, kdl_x);
			jnt_to_jac_solver->JntToJac(kdl_q, kdl_J);
			
			Eigen::Matrix<double, 7, 1> tau_fb; tau_fb.setZero();

			Eigen::Matrix<double, 6, 7> Jacobian;
			Eigen::Matrix<double, 7, 7> Kvc; Kvc.setZero();
			Eigen::Matrix<double, 7, 7> Ivc; Ivc.setZero();
			Eigen::Matrix<double, 6, 1> delta_x;
			Eigen::Matrix<double, 7, 1> dq_des;
			Eigen::Matrix<double, 7, 1> e;

			Kvc.diagonal() << 100, 100, 100, 100, 50, 40, 20;
			// Ivc.diagonal() << 10, 10, 10, 10, 10, 10, 10;
			delta_x << 0.02, 0.00, 0.0, 0.0, 0.0, 0.0;

			jnt_to_pose_solver->JntToCart(kdl_q, kdl_x);
			kdl_J.changeRefFrame(kdl_x);

			for (int i = 0; i < 6; i++) {
				for (int j = 0; j < 7; j++) {
					Jacobian(i, j) = kdl_J(i, j);
				}
			}

			Eigen::Matrix<double, 7,6> pinvJ;
			pinv2(Jacobian, pinvJ);
			dq_des =  pinvJ * delta_x;

			q_des += dq_des * dt;
			e = q_des - q;
			tau_fb = Kvc * e;

			for (int i = 0; i < 7; i ++) {
				// tau_fb(i) = kdl_tau(i);
				std::cout << tau_fb(i) << "\t";
			}
			std::cout << std::endl;
			publish(tau_fb, utime, t, dt);
		}
		return 0;
	}

	int circle_controller() {
				
		if (!lcm.good())
			return 1;

		lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

		// std::this_thread::sleep_for(std::chrono::milliseconds(x));

		bool init = true;
		
		Eigen::Matrix<double, 7, 1> dei; dei.setZero();

		std::cout << "Loop started vel cart" << std::endl;
		double t0 = nowtime();
		double t_prev = 0;
		while (0 == lcm.handle()) {
			const int64_t utime = micros();
			double t = nowtime() - t0;
			double dt = t - t_prev;
			t_prev = t;
			double freq = 1.0 / dt;
			// std::cout << t << std::endl;

			if (init == true) {
				init = false;
				for (int i = 0; i < 7; i++) {
					q_des(i) = lcm_status.joint_position_measured[i];
					kdl_q(i) = lcm_status.joint_position_measured[i];
					kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
				}
				jnt_to_pose_solver->JntToCart(kdl_q, kdl_x0);
			}

			for (int i = 0; i < kNumJoints; i++) {
				kdl_q(i) = lcm_status.joint_position_measured[i];
				kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
			}
			jnt_to_pose_solver->JntToCart(kdl_q, kdl_x);
			jnt_to_jac_solver->JntToJac(kdl_q, kdl_J);
			// kdl_J.changeRefFrame(kdl_x);

			Eigen::Matrix<double, 7, 1> tau_fb; tau_fb.setZero();

			Eigen::Matrix<double, 6, 7> Jacobian;
			Eigen::Matrix<double, 7, 7> Kvc; Kvc.setZero();
			Eigen::Matrix<double, 7, 7> Ivc; Ivc.setZero();
			Eigen::Matrix<double, 6, 1> delta_x;
			Eigen::Matrix<double, 7, 1> dq_des;
			Eigen::Matrix<double, 7, 1> e;



			for (unsigned int i = 0 ; i < 6 ; i++) {
				kdl_dx(i) = 0;
				for (unsigned int j = 0 ; j < iiwa_chain.getNrOfJoints() ; j++) {
					kdl_dx(i) += kdl_J(i,j) * kdl_dq(j);
				}
			}

			// Follow a circle of 10cm at 3 rad/sec.                                                                                                                                        
			circle_phase += 0.1 * dt;
			KDL::Vector  circle(0,0,0);
			circle(0) = 0.1 * cos(circle_phase);
			circle(1) = 0.1 * sin(circle_phase);

			kdl_xd = kdl_x0;
			kdl_xd.p += circle;

			// Calculate a Cartesian restoring force.                                                                                                                                       
			kdl_xerr.vel = kdl_x.p - kdl_xd.p;
			kdl_xerr.rot = 0.1 * (kdl_xd.M.UnitX() * kdl_x.M.UnitX() +
	   							  kdl_xd.M.UnitY() * kdl_x.M.UnitY() +
								  kdl_xd.M.UnitZ() * kdl_x.M.UnitZ());


 			for (unsigned int i = 0 ; i < 6 ; i++)
				kdl_F(i) = - kdl_Kp(i) * kdl_xerr(i); // - kdl_Kd(i) * kdl_dx(i);

			// Convert the force into a set of joint torques.                                                                                                                               
			for (unsigned int i = 0 ; i < iiwa_chain.getNrOfJoints() ; i++)
			{
				kdl_tau(i) = 0;
				for (unsigned int j = 0 ; j < 6 ; j++)
					kdl_tau(i) += kdl_J(j,i) * kdl_F(j);
			}


			for (int i = 0; i < 7; i ++) {
				tau_fb(i) = kdl_tau(i);
				std::cout << tau_fb(i) << "\t";
			}
			std::cout << std::endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
			publish(tau_fb, utime, t, dt);
		}
		return 0;
	}

	void handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan,
						const drake::lcmt_iiwa_status *msg) {
		lcm_status = *msg;
		for (int i = 0; i < 7; i++) {
			q(i) = lcm_status.joint_position_measured[i];
			dq(i) = lcm_status.joint_velocity_estimated[i];
		}
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

	void pinv2(Eigen::Matrix<double, 6, 7>& L, Eigen::Matrix<double, 7, 6>& pinvL, double alpha0 = 0.001, double w0 = 0.0001) {
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

		// 7x6 = 7x6 * (6x7 * 7x6)
		pinvL = L.transpose() * (L * L.transpose() - alpha * Eigen::MatrixXd::Identity(6, 6)).inverse();
	}
	
	void pinv3(Eigen::Matrix<double, 8, 7>& L, Eigen::Matrix<double, 7, 8>& pinvL, double alpha0 = 0.001, double w0 = 0.0001) {
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

		// 7x6 = 7x6 * (6x7 * 7x6)
		pinvL = L.transpose() * (L * L.transpose() - alpha * Eigen::MatrixXd::Identity(6, 6)).inverse();
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

    void computePose(std::vector <vpPoint> &point, const std::vector <vpImagePoint> &ip,
                     const vpCameraParameters &cam, bool init, vpHomogeneousMatrix &cMo) {
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

	/* Impedance controller tau = K (q_des - q) - B dq */
	Vector7d pd_controller(const Vector7d &q_des) {
		for (int i = 0; i < kNumJoints; i++) {
			q[i] = lcm_status.joint_position_measured[i];
			dq[i] = lcm_status.joint_velocity_estimated[i];
		}
		Vector7d error = q_des - q;
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

	int loop_vs() {
		
		if (!lcm.good())
			return 1;
		lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

		try {
 			vpImage<unsigned char> I;

            /* CAMERA SETUP */
            vpRealSense2 g;

            rs2::config config;
            config.disable_stream(RS2_STREAM_DEPTH);
            config.disable_stream(RS2_STREAM_INFRARED);
			// config.enable_stream(RS2_STREAM_INFRARED, 1, 848, 100, RS2_FORMAT_Y8, 300);
			// config.disable_stream(RS2_STREAM_COLOR);
            config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_Y16, 60);

            g.open(config);

            g.acquire(I);

			// vpCameraParameters cam(848, 100, I.getWidth() / 2, I.getHeight() / 2);
            // cam = g.getCameraParameters(RS2_STREAM_INFRARED, vpCameraParameters::perspectiveProjWithoutDistortion);

            vpCameraParameters cam(840, 840, I.getWidth() / 2, I.getHeight() / 2);
            cam = g.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithoutDistortion);

            std::clog << cam << std::endl;

            vpImagePoint germ[Np];
            vpDot2 blob[Np];

            for (int i = 0; i < Np; i++) {
                blob[i].setGraphics(true);
                blob[i].setGraphicsThickness(1);
            }

            vpDisplayX d;
            d.init(I, 20, 40, "Original");

            std::vector<vpImagePoint> ip;
            std::vector<vpImagePoint> ipd;

            double w, h; h = I.getWidth() / 2; w = I.getHeight() / 2;
            //ipd.push_back(vpImagePoint(cam.get_v0() - 100, cam.get_u0() - 100)); // left top
            //ipd.push_back(vpImagePoint(cam.get_v0() - 100, cam.get_u0() + 100)); // right top
            //ipd.push_back(vpImagePoint(cam.get_v0() + 100, cam.get_u0() + 100)); // right bottom
            //ipd.push_back(vpImagePoint(cam.get_v0() + 100, cam.get_u0() - 100)); // left bottom

            ipd.push_back(vpImagePoint(300.954, 326.775)); // left top
            ipd.push_back(vpImagePoint(299.619, 408.009)); // rifht top                        
            ipd.push_back(vpImagePoint(382.439, 410.426)); // right bottom
			ipd.push_back(vpImagePoint(382.761, 328.593)); // right top
            
            

            
            std::vector<vpPoint> point;
            point.push_back(vpPoint(-L, -L, 0));
            point.push_back(vpPoint( L, -L, 0));
            point.push_back(vpPoint( L,  L, 0));
            point.push_back(vpPoint(-L,  L, 0));

            vpHomogeneousMatrix cMo, cdMo;
            //computePose(point, ipd, cam, true, cdMo);

            bool init_cv = true;   // initialize tracking and pose computation
            bool learn = true;
            bool isTrackingLost = false;
            bool send_velocities = false;
            bool final_quit = false;

            int k = 0;
            double t;
			double dt;
			double freq;
            double loop_start_time;

            double Z = 1;
            double f = 0.00193; // fo            cal length of RSD435
            double fx = cam.get_px();
            double fy = cam.get_py();
            double rhox = f / fx;
            double rhoy = f / fy;
            Eigen::Matrix<double, 2, 2> diagrho;
            diagrho << rhox, 0, 0, rhoy;

            double lambda = l;
            double lambda_0 = vcx;      // 4
            double lambda_inf = vcy;    // 1
            double lambda_0l = vcz;     // 600
            double mu = l;          // 4

            Eigen::Matrix<double, 3, 3> Rzero = Eigen::Matrix<double, 3, 3>::Zero();
            Eigen::Matrix<double, 3, 3> cRe;
            // roatate to pi/2 around Z-axis
            // cRe <<  0,  0, -1,
            //         1,  0,  0,
            //         0, -1,  0;
			//z
			cRe <<  0, -1, 0,
                    1, 0,  0,
                    0, 0,  1;
			cRe <<  1, 0, 0,
                    0, 1,  0,
                    0, 0,  1;					
			// //zx
			cRe <<  1, 0, 0,
                    0, -1, 0,
                    0, 0,  -1;					


            Eigen::MatrixXd cVe(6, 6);
            cVe << cRe, Rzero, Rzero, cRe;
			Eigen::Matrix<double, 7, 7> Kvc; Kvc.setZero();
			Eigen::Matrix<double, 7, 7> Ivc; Ivc.setZero();
			Eigen::Matrix<double, 7, 1> dq_des;

			Kvc.diagonal() << 100, 100, 100, 100, 50, 40, 20;

			std::cout << "Loop started" << std::endl;
			double t0 = nowtime();
			double t_prev = 0;
			while (0 == lcm.handle()) {
				std::cout << "***" << std::endl;
// std::cout << " 1 " << nowtime() - t0 << "\t started" <<std::endl;
				// TIME
				const int64_t utime = micros();
				t = nowtime() - t0;
				double loop_start_time = nowtime();
				dt = t - t_prev;
				t_prev = t;
				freq = 1.0 / dt;
				std::cout << t << "\t(" << dt <<  ")\t freq: " << freq << std::endl;

				Eigen::Matrix<double, 7, 1> tau_fb = Eigen::Matrix<double, 7, 1>::Zero();

                try {
                    g.acquire(I);
                    vpDisplay::display(I);
// std::cout << " 2 " << nowtime() - t0 << "\t g.acquiered" <<  std::endl;
                    std::stringstream ss; ss << (send_velocities ? "Click to START" : "Click to STOP");
                    vpDisplay::displayText(I, 20, 20, ss.str(), vpColor(254,188,0));

                    if (!learn) {
                        vpDisplay::displayText(I, vpImagePoint(80, 20), "Tracking is ok!", vpColor::green);
                        try {

                            Eigen::Matrix<double, 8, 1> error;
                            Eigen::Matrix<double, 8, 6> L = Eigen::Matrix<double, 8, 6>::Zero();
                            Eigen::Matrix<double, 6, 8> pinvL;

                            Eigen::Matrix<double, 6, 7> J;
                            Eigen::Matrix<double, 7, 6> pinvJ;

							Eigen::Matrix<double, 8, 7> Jp; 

                            Eigen::Matrix<double, 6, 1> v_c;
                            Eigen::Matrix<double, 6, 1> v_c0;
                            Eigen::Matrix<double, 6, 1> v_e;

                            std::vector <vpImagePoint> ip(Np);

							// std::cout << "***" << std::endl;
                            for (int i = 0; i < Np; i++) {
                                blob[i].track(I);
                                ip[i] = blob[i].getCog();

                                if (!init_cv) {
                                    vpColVector cP;
                                    point[i].changeFrame(cMo, cP);
                                    Z = cP[2]; //FIXME: can be estimated from square dims???
//                                    std::cout << "Z" << i << ": " << Z << std::endl;
                                }

                                double x = (ip[i].get_u() - cam.get_u0()) * rhox / f;
                                double y = (ip[i].get_v() - cam.get_v0()) * rhoy / f;

								// std::cout << ip[i].get_u() << " " << ip[i].get_v() << std::endl;

                                Eigen::Matrix<double, 2, 6> Lx;
                                Lx << 1 / Z,      0, -x / Z, -x * y,        (1 + x * x), -y,
                                        0,    1 / Z, -y / Z, -(1 + y * y),        x * y, x;
                                Lx = -1 * f * diagrho * Lx;

                                // Copy one point-matrix to full image-jacobian
                                for (int j = 0; j < 6; j++) {
                                    for (int k = 0; k < 2; k++) {
                                        L(k + 2 * i, j) = Lx(k, j);
                                    }
                                }

                                if (DEBUG) {
                                    std::stringstream ss; ss << i;
                                    vpDisplay::displayText(I, blob[i].getCog(), ss.str(), vpColor::white); // number of point
                                    vpDisplay::displayLine(I, ipd[i], ip[i], vpColor(254,188,0), 1); // line between current and desired points
                                }
                            }
// std::cout << " 3 " << nowtime() - t0 << "\t befor comute error" <<  std::endl;
                            compute_error(ip, ipd, error);

							for (int i = 0; i < kNumJoints; i++) {
								kdl_q(i) = lcm_status.joint_position_measured[i];
								kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
							}
							jnt_to_pose_solver->JntToCart(kdl_q, kdl_x);
							jnt_to_jac_solver->JntToJac(kdl_q, kdl_J);
							kdl_J.changeRefFrame(kdl_x);							
                            for (int i = 0; i < 6; i++) {
								for (int j = 0; j < 7; j++) {
									J(i, j) = kdl_J(i, j);
								}
							}

                            computePose(point, ip, cam, init_cv, cMo);

							Jp = L * cVe * J;

							Eigen::Matrix<double, 8, 6> tempL;
							tempL = L * cVe;
							
							pinv(tempL, pinvL);
							v_c = pinvL * error;
							
							// Eigen::Matrix<double, 7,6> pinvJ;
							pinv2(J, pinvJ);

							dq_des =  pinvJ * v_c;

							q_des += dq_des * dt;
							
							tau_fb = Kvc * (q_des - q);


							// tau_fb = 1000000 * Jp.transpose() * Klambda * error; // - B * dq;
							// std::cout << "tau_fb: ";
							// for (int i = 0; i < 7; i++) {
							// 	std::cout << tau_fb(i) << " \t";
							// }
							// std::cout << std::endl;

                            // if ((!send_velocities) || (error.lpNorm<Eigen::Infinity>() <= 0.015)) {
							if ((!send_velocities)) {
								tau_fb.setZero();
                                t0 = nowtime();
								// std::cout << "STOPED" << std::endl;
                            }
// std::cout << " 4 " << nowtime() - t0 << "\t tau_fb comuted" <<  std::endl;
                            publish(tau_fb, utime, t, dt);

                            vpDisplay::displayArrow(I, vpImagePoint(cam.get_v0(), cam.get_u0()),
                                                    vpImagePoint(cam.get_v0(), cam.get_u0() + v_c[0] * 10000),
                                                    vpColor::red);
                            vpDisplay::displayArrow(I, vpImagePoint(cam.get_v0(), cam.get_u0()),
                                                    vpImagePoint(cam.get_v0() + v_c[1] * 10000, cam.get_u0()),
                                                    vpColor::green);

                            vpDisplay::displayArrow(I, vpImagePoint(cam.get_v0(), cam.get_u0()),
                                                    vpImagePoint(cam.get_v0() + v_c[1] * 10000,
                                                                 cam.get_u0() + v_c[0] * 10000), vpColor(254,188,0));

                            if (DEBUG) {
                                vpDisplay::displayFrame(I, cdMo, cam, opt_square_width, vpColor::none, 1);
                                vpDisplay::displayFrame(I, cMo, cam, opt_square_width, vpColor::none, 2);
                                // write error to file
                            }

                        } catch (...) {
                            std::cout << "Computer vision failure.\n";
							// isTrackingLost = true;
                        }
                    } else {
                        if (vpDisplay::getClick(I, germ[k], false)) {
                            blob[k].initTracking(I, germ[k]);
                            k++;
                        }
                        if (k == Np) {
                            learn = false;
                            k = 0;
							for (int i = 0; i < 7; i++) {
								q_des(i) = lcm_status.joint_position_measured[i];
							}
                        }
                    }

                    vpDisplay::flush(I);

                    ss.str("");
                    ss << "Loop time: " << nowtime() - loop_start_time << " ms";
                    vpDisplay::displayText(I, 40, 20, ss.str(), vpColor(254, 188, 0));
                    vpDisplay::flush(I);

// std::cout << " 5 " << nowtime() - t0 << "\t image flushed" <<  std::endl;

                    vpMouseButton::vpMouseButtonType button;
                    if (vpDisplay::getClick(I, button, false)) {
						std::cout << "Click mouse -> publish cmd 0" << std::endl;
     					tau_fb.setZero();
                        publish(tau_fb, utime, t, dt);
                        
						switch (button) {
                            case vpMouseButton::button1:
                                send_velocities = !send_velocities;
                                std::cout << "Send velocities mode changed." << std::endl;
                                break;
                            case vpMouseButton::button2:
                                init_cv = true;   // initialize tracking and pose computation
                                learn = true;
                                send_velocities = false;
                                std::cout << "Reseted." << std::endl;
                                break;
                            case vpMouseButton::button3:
                                final_quit = true;
                                std::cout << "Quited." << std::endl;
                                break;
                            default:
                                break;
                        }
                    }

                } catch (...) {
                    isTrackingLost = true;
                    std::cout << "Tracking lost. Finding blobs..    .\r";
					std::cout << "Click mouse -> publish cmd 0" << std::endl;
					tau_fb.setZero();
					publish(tau_fb, utime, t, dt);
                }
				std::this_thread::sleep_for(std::chrono::milliseconds(30));
			}
		} catch (const vpException &e) {
            std::stringstream ss;
            ss << "vpException: " << e;
            std::cout << ss.str() << std::endl;
        }
		return 0;
	}

	// PUBLISH COMMAND
	void publish(Eigen::Matrix<double, 7, 1> &tau_fb , int64_t utime, double t, double dt) {
		lcm_command.utime = utime;
		lcm_command.num_joints = kNumJoints;
 		lcm_command.num_torques = kNumJoints;
		lcm_command.joint_position.resize(kNumJoints, 0);
		lcm_command.joint_torque.resize(kNumJoints, 0);
		for (int i = 0; i < kNumJoints; i++ ) {
			lcm_command.joint_position[i] = lcm_status.joint_position_measured[i];
			lcm_command.joint_torque[i] = tau_fb[i];
		}
		std::cout << "tau_fb::";
		for (int i = 0; i < 7; i++) {
			std::cout << tau_fb(i) << " \t";
		}
		std::cout << std::endl;

		lcm.publish("IIWA_COMMAND", &lcm_command);
	}

	int test_camera() {
		
		vpImage<unsigned char> I;
		
		rs2::pipeline pipe;
    	pipe.start();
		
		
		
		// vpRealSense2 g;

		// rs2::config config;
		// config.disable_stream(RS2_STREAM_DEPTH);
		// // config.disable_stream(RS2_STREAM_INFRARED);
		// config.enable_stream(RS2_STREAM_INFRARED, 1, 848, 100, RS2_FORMAT_Y8, 300);
		// config.disable_stream(RS2_STREAM_COLOR);
		// // config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_Y16, 60);

		// g.open(config);

		// g.acquire(I);

		// vpCameraParameters cam(848, 100, I.getWidth() / 2, I.getHeight() / 2);
		// cam = g.getCameraParameters(RS2_STREAM_INFRARED, vpCameraParameters::perspectiveProjWithoutDistortion);
		return 0;
	}

};


// int main(int argc, char **argv) {
// 	// double lambda = std::atof(argv[1]);

// 	// double vcx = std::atof(argv[2]);
// 	// double vcy = std::atof(argv[3]);
// 	// double vcz = std::atof(argv[4]);
// 	std::cout << "Node started" << std::endl;

// 	// Controller controller;
// 	// // int code = controller.loop_vs();
// 	// // int code = controller.velocity_cart_controller();
// 	// // int code = controller.circle_controller();
// 	// int code = controller.test_camera();
		

// 	vpImage<unsigned char> I;

//     // Declare depth colorizer for pretty visualization of depth data
//     rs2::colorizer color_map;
//     // Declare rates printer for showing streaming rates of the enabled streams.
//     rs2::rates_printer printer;

// 	rs2::config* config = new rs2::config();
// 	config->enable_stream(RS2_STREAM_DEPTH, 0, 848, 100, RS2_FORMAT_Z16, 300);
// 	rs2::pipeline pipe;
// 	pipe.start(*config);

// 	while (1==1) // Application still alive?
// 	{
// 		rs2::frameset data = pipe.wait_for_frames().    // Wait for next set of frames from the camera
// 							apply_filter(printer).     // Print each enabled stream frame rate
// 							apply_filter(color_map);   // Find and colorize the depth data
// 		rs2::video_frame frame = data.get_infrared_frame();
		
// 	}

// 	int code = 0;
// 	return code;
// }


namespace
{

void frame_to_mat(const rs2::frame &f, cv::Mat &img)
{
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
} // namespace
 
int main()
{
//   const int width = 640, height = 480, fps = 60;
  vpRealSense2 rs;
  rs2::config config;
  config.enable_stream(RS2_STREAM_INFRARED, 1, 848, 100, RS2_FORMAT_Y8, 300);
  rs.open(config);
 
  rs2::pipeline_profile &profile = rs.getPipelineProfile();
  rs2::pipeline &pipe = rs.getPipeline();
 
  auto infrared_profile = profile.get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
  cv::Mat mat_infrared1(infrared_profile.height(), infrared_profile.width(), CV_8UC1);


vpImage<unsigned char> I(100, 848);
vpDisplayX d(I);

  while (true) {
 
    auto data = pipe.wait_for_frames();
    frame_to_mat(data.get_infrared_frame(1), mat_infrared1);

	
	vpImageConvert::convert(mat_infrared1, I);

	vpDisplay::display(I);
	vpDisplay::flush(I);

  }
 
  return EXIT_SUCCESS;
}