
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
#include <boost/scoped_ptr.hpp>

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
		
		std::string base_link;
		base_link = "world";
		std::string tool_link;
		tool_link = "iiwa_link_ee";

		KDL::Tree iiwa_tree;
		if (!kdl_parser::treeFromUrdfModel(model, iiwa_tree)){
			printf("Failed to construct kdl tree");
		}
		KDL::Chain iiwa_chain;
		if(!iiwa_tree.getChain(base_link, tool_link, iiwa_chain)){
			printf("Failed to get KDL chain from tree ");
		}

		boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver;
		jnt_to_jac_solver.reset(new KDL::ChainJntToJacSolver(iiwa_chain));
		KDL::JntArray q;
		KDL::Jacobian J;
		q.resize(iiwa_chain.getNrOfJoints());
		J.resize(iiwa_chain.getNrOfJoints());
		jnt_to_jac_solver->JntToJac(q, J);

		std::cout << J.columns() << " " << J.rows() << std::endl;			
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 7; i++) {
				std::cout << J(i, j) << " ";
			}
			std::cout << std::endl;
		}

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
            config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGBA8, 30);
            g.open(config);

            g.acquire(I);

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
//            eRc << 0, 1, 0,
//                  -1, 0, 0,
//                   0, 0, 1;
//            eRc << 0, 0, 1,
//                  -1, 0, 0,
//                   0, -1, 0;
            cRe <<  0,  0, -1,
                    1,  0,  0,
                    0, -1,  0;


            Eigen::MatrixXd cVe(6, 6);
            cVe << cRe, Rzero, Rzero, cRe;


			std::cout << "Loop started" << std::endl;
			double t0 = nowtime();
			double t_prev = 0;
			while (0 == lcm.handle()) {
				// TIME
				const int64_t utime = micros();
				t = nowtime() - t0;
				dt = t - t_prev;
				t_prev = t;
				freq = 1.0 / dt;
				// std::cout << t << "\t(" << dt <<  ")\t freq: " << freq << std::endl;

				Eigen::Matrix<double, 7, 1> tau_fb = Eigen::Matrix<double, 7, 1>::Zero();

                try {
                    g.acquire(I);
                    vpDisplay::display(I);

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

							std::cout << "***" << std::endl;
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

								std::cout << ip[i].get_u() << " " << ip[i].get_v() << std::endl;

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

                            compute_error(ip, ipd, error);
                            ks.get_jacobe(q, J);
                            computePose(point, ip, cam, init_cv, cMo);


							Jp = L * cVe * J;
							tau_fb = Jp.transpose() * Klambda * error - B * dq;


                            if ((!send_velocities) || (dq.lpNorm<Eigen::Infinity>() <= 0.015)) {
								tau_fb = Eigen::Matrix<double, 7, 1>::Zero();
                                t0 = nowtime();
                            }

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
                        }
                    }

                    vpDisplay::flush(I);

                    ss.str("");
                    ss << "Loop time: " << 1/freq << " ms";
                    vpDisplay::displayText(I, 40, 20, ss.str(), vpColor(254, 188, 0));
                    vpDisplay::flush(I);

                    vpMouseButton::vpMouseButtonType button;
                    if (vpDisplay::getClick(I, button, false)) {
     					tau_fb = Eigen::Matrix<double, 7, 1>::Zero();
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
					tau_fb = Eigen::Matrix<double, 7, 1>::Zero();
					publish(tau_fb, utime, t, dt);
                }

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
		// lcm.publish("IIWA_COMMAND", &cmd);
	}

};


int main(int argc, char **argv) {
	// double lambda = std::atof(argv[1]);

	// double vcx = std::atof(argv[2]);
	// double vcy = std::atof(argv[3]);
	// double vcz = std::atof(argv[4]);
	std::cout << "Node started" << std::endl;

	Controller controller;
	int code = controller.loop_vs();
	
	return code;
}
