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
	
	bool DEBUG = true;
	
	// iiwa driver interface
	lcm::LCM lcm;	
	lcmt_iiwa_status lcm_status{};
	lcmt_iiwa_command lcm_command{};

	// robot state
	Eigen::Matrix<double, kNumJoints, 1>  q;
	Eigen::Matrix<double, kNumJoints, 1>  dq;

	// stiffness and damping parameters
	Eigen::Matrix<double, kNumJoints, 1>  k;
	Eigen::Matrix<double, kNumJoints, 1>  b;
	Eigen::Matrix<double, kNumJoints, kNumJoints>  K;
	Eigen::Matrix<double, kNumJoints, kNumJoints>  B;

	// robot desired values
	Eigen::Matrix<double, kNumJoints, 1>  q_des;
	Vector6d v_des;

	// IBVS parameters
	Eigen::Matrix<double, 8, 8>  Klambda;
	int Np = 4;
	double opt_square_width = 0.11;
	double L = opt_square_width / 2.;
	double distance_same_blob = 1.; // 2 blobs are declared same if their distance is less than this value
	double l = 0;
	double vcx = 0, vcy = 0, vcz = 0;

	// double lambda = 10;
	double lambda_0 = 30;      // 4
	double lambda_inf = 5;    // 1
	double lambda_0l = 6000;     // 600
	// double mu = 4;          // 4
	// double lambda = (lambda_0 - lambda_inf) * exp( - lambda_0l * v_c.lpNorm<Eigen::Infinity>() / (lambda_0 - lambda_inf)) + lambda_inf;

	// KINEMATICS
	KDL::Tree iiwa_tree;
	KDL::Chain iiwa_chain;
	std::string base_link;
	std::string tool_link;

	boost::scoped_ptr<KDL::ChainFkSolverPos> jnt_to_pose_solver;
	boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver;
	
	KDL::JntArray  kdl_q;            // Joint positions                                                                                                                                
	KDL::JntArray  kdl_q0;           // Joint initial positions                                                                                                                        
	KDL::JntArray  kdl_dq;      	 // Joint velocities                                                                                                                               
	KDL::JntArray  kdl_tau;          // Joint torques                                                                                                                                  
	KDL::Frame     kdl_x;            // Tip pose                                                                                                                                       
	KDL::Frame     kdl_xd;           // Tip desired pose                                                                                                                               
	KDL::Frame     kdl_x0;           // Tip initial pose                                                                                                                               
	KDL::Twist     kdl_dx;         	 // Cart velocity                                                                                                                                  
	KDL::Wrench    kdl_F;            // Cart effort                                                                                                                                    
	KDL::Jacobian  kdl_J;            // Jacobian                                                                                                                                       

 public:
	Controller(){

		q.setZero();
		dq.setZero();
		K.setZero();
		B.setZero();

		K.diagonal() << 2, 2, 2, 1, 0.5, 0.5, 0.25;
		B.diagonal() << 2, 2, 2, 1, 0.5, 0.5, 0.25;
		K = K * 50;
		B = B * 1;

		Klambda.diagonal() << 1, 1, 1, 1, 1, 1, 1, 1;

		v_des << 0, 0, 0, 0, 0, 0;

		// kinematics initialization
		std::string urdf_file = "/home/iiwa/phd_code/ws/labs-robots-files/kuka-iiwa/iiwa_model_simscape/iiwa_description/urdf/iiwa14.urdf";
		base_link = "world";
		// tool_link = "iiwa_link_ee_kuka";
		tool_link = "iiwa_link_ee";
		
		urdf::Model model;
		if (!model.initFile(urdf_file)){
			printf("Failed to parse urdf file");
		}
		
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
		kdl_J.resize(iiwa_chain.getNrOfJoints());
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

	void handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg) {
		lcm_status = *msg;
		for (int i = 0; i < 7; i++) {
			q(i) = lcm_status.joint_position_measured[i];
			dq(i) = lcm_status.joint_velocity_estimated[i];
		}
	}

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

    void computePose(std::vector <vpPoint> &point, const std::vector <vpImagePoint> &ip, const vpCameraParameters &cam, bool init, vpHomogeneousMatrix &cMo) {
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

	int64_t micros() {
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

	double nowtime() {
		auto current_time = std::chrono::system_clock::now();
		auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
		double num_seconds = duration_in_seconds.count();
		return  num_seconds;
	}

	void frame_to_mat(const rs2::frame &f, cv::Mat &img) {
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

	int loop_vs() {
		
		// iiwa driver interface 
		if (!lcm.good())
			return 1;

		lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

		try {
			vpRealSense2 g;
			rs2::config config;
			config.disable_all_streams();
			config.enable_stream(RS2_STREAM_INFRARED, 1, 848, 100, RS2_FORMAT_Y8, 300);
			g.open(config);

			rs2::pipeline_profile &profile = g.getPipelineProfile();
			rs2::pipeline &pipe = g.getPipeline();

			rs2::device selected_device = profile.get_device();
			auto depth_sensor = selected_device.first<rs2::depth_sensor>();

			if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
				depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);
			}

			// if (depth_sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
			// 	depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, false);
			// }

			auto infrared_profile = profile.get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
			cv::Mat mat_infrared1(infrared_profile.height(), infrared_profile.width(), CV_8UC1);
			vpImage<unsigned char> I(100, 848);

			vpCameraParameters cam = g.getCameraParameters(RS2_STREAM_INFRARED, vpCameraParameters::perspectiveProjWithoutDistortion);
			// vpCameraParameters cam = g.getCameraParameters(RS2_STREAM_INFRARED, vpCameraParameters::perspectiveProjWithDistortion);
            std::clog << cam << std::endl;

			// FIRST FRAME INIT
			auto data = pipe.wait_for_frames();
			frame_to_mat(data.get_infrared_frame(1), mat_infrared1);
			vpImageConvert::convert(mat_infrared1, I);
            vpImagePoint germ[Np];
            vpDot2 blob[Np];

            for (int i = 0; i < Np; i++) {
                blob[i].setGraphics(true);
                blob[i].setGraphicsThickness(1);
            }

            vpDisplayX d(I);
            // d.init(I, 20, 40, "Original");

            std::vector<vpImagePoint> ip;
            std::vector<vpImagePoint> ipd;

            double w, h; h = I.getWidth() / 2; w = I.getHeight() / 2;

			double a = 30;
            ipd.push_back(vpImagePoint(50 - a, 424 - a)); // left top
            ipd.push_back(vpImagePoint(50 - a, 424 + a)); // rifht top                        
            ipd.push_back(vpImagePoint(50 + a, 424 + a)); // right bottom
			ipd.push_back(vpImagePoint(50 + a, 424 - a)); // right top
                        
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

            double Z = 2.0;
            double f = 0.00193; // fo            cal length of RSD435
            double fx = cam.get_px();
            double fy = cam.get_py();
            double rhox = f / fx;
            double rhoy = f / fy;
            Eigen::Matrix<double, 2, 2> diagrho;
            diagrho << rhox, 0, 0, rhoy;

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
			//zx
			// cRe <<  1, 0, 0,
            //         0, -1, 0,
            //         0, 0,  -1;					


            Eigen::MatrixXd cVe(6, 6);
            cVe << cRe, Rzero, Rzero, cRe;
			Eigen::Matrix<double, 7, 7> Kvc; Kvc.setZero();
			Eigen::Matrix<double, 7, 7> Ivc; Ivc.setZero();
			Eigen::Matrix<double, 7, 1> dq_des;

			Kvc.diagonal() << 100, 100, 100, 100, 50, 40, 20;

			std::cout << "Loop started" << std::endl;
			double t0 = nowtime();
			double t_prev = 0;
			double loop_start_time;
			Eigen::Matrix<double, 7, 1> tau_fb = Eigen::Matrix<double, 7, 1>::Zero();
			Eigen::Matrix<double, 8, 1> error;
			Eigen::Matrix<double, 8, 6> L = Eigen::Matrix<double, 8, 6>::Zero();
			Eigen::Matrix<double, 6, 8> pinvL;

			Eigen::Matrix<double, 6, 7> J;
			Eigen::Matrix<double, 7, 6> pinvJ;

			Eigen::Matrix<double, 8, 7> Jp; 

			Eigen::Matrix<double, 6, 1> v_c;
			Eigen::Matrix<double, 6, 1> v_c0;
			Eigen::Matrix<double, 6, 1> v_e;
			double x;
			double y;
			Eigen::Matrix<double, 2, 6> Lx;
			Eigen::Matrix<double, 7, 1> Q_DES_0; Q_DES_0.setZero();
			Eigen::Matrix<double, 7, 1> q_delta;
			q_delta.setZero();

// START LOOP
			while (0 == lcm.handle()) {
				std::cout << "***" << std::endl;

				// timers
				loop_start_time = vpTime::measureTimeMs();
				const int64_t utime = micros();
				t = nowtime() - t0;
				dt = t - t_prev;
				t_prev = t;

				tau_fb.setZero();
				L.setZero();
				Lx.setZero();

				try {
					// get new image
					auto data = pipe.wait_for_frames();
					frame_to_mat(data.get_infrared_frame(1), mat_infrared1);
					vpImageConvert::convert(mat_infrared1, I);

                    vpDisplay::display(I);
                    std::stringstream ss; ss << (send_velocities ? "Click to STOP" : "Click to START");
                    vpDisplay::displayText(I, 20, 20, ss.str(), vpColor(254,188,0));

                    if (!learn) {
                        try {

                            std::vector <vpImagePoint> ip(Np);

                            for (int i = 0; i < Np; i++) {
                                blob[i].track(I);
                                ip[i] = blob[i].getCog();
                                if (!init_cv) {
                                    vpColVector cP;
                                    point[i].changeFrame(cMo, cP);
                                    Z = cP[2]; //FIXME: can be estimated from square dims???
                                }

                                x = (ip[i].get_u() - cam.get_u0()) * rhox / f;
                                y = (ip[i].get_v() - cam.get_v0()) * rhoy / f;

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

							for (int i = 0; i < kNumJoints; i++) {
								kdl_q(i) = lcm_status.joint_position_measured[i];
								// kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
							}
							jnt_to_pose_solver->JntToCart(kdl_q, kdl_x);
							jnt_to_jac_solver->JntToJac(kdl_q, kdl_J);
							kdl_J.changeRefFrame(kdl_x);
                          
						    for (int i = 0; i < 6; i++) {
								for (int j = 0; j < 7; j++) {
									J(i, j) = kdl_J(i, j);
								}
							}

							if (DEBUG) {
                            	computePose(point, ip, cam, init_cv, cMo);
							}

							// Jp = L * cVe * J;

							Eigen::Matrix<double, 8, 6> tempL;
							tempL = L * cVe;
							
							pinv(tempL, pinvL);
							v_c = pinvL * error;


							v_c(0) = 0.01;
							v_c(1) = 0.0;
							v_c(2) = 0;
							v_c(3) = 0;
							v_c(4) = 0;
							v_c(5) = 0;
							std::cout << "v_c(0:1): " << v_c(0) << "\t" << v_c(1) << "\t" << v_c(2) << std::endl;

							double a = lambda_0 - lambda_inf;
							double lambda = a * exp( - lambda_0l * v_c.lpNorm<Eigen::Infinity>() / a) + lambda_inf;

							pinv2(J, pinvJ);							

							dq_des =  lambda * pinvJ * v_c;
							q_delta += dq_des * dt;
							
							std::cout << "q_delta:   ";
							for (int i = 0; i < 7; i++) {
								if (abs(q_delta(i)) > 1.0) {
									q_delta(i) + copysign(1.0, q_delta(i));
								}
								std::cout << q_delta(i) << "\t";
							}
							std::cout << std::endl;

							q_des = Q_DES_0 + q_delta;
							tau_fb = 8 * K * (q_des - q) - 10 * B * dq;

							std::cout << "tau: ";
							for (int i = 0; i < 7; i++) {
								std::cout << tau_fb(i) << " \t";
							}
							std::cout << std::endl;

							// Eigen::Matrix<double, 7, 7> P;
							// P.diagonal() << 0.5, 1, 1, 1, 0.4, 0.3, 0.1;
							// tau_fb = 2000000 * Jp.transpose() * Klambda * error; // - B * dq;
							// tau_fb.setZero();

                            // if ((!send_velocities) || (error.lpNorm<Eigen::Infinity>() <= 0.015)) {
							if ((!send_velocities)) {
								tau_fb.setZero();
								tau_fb = 8 * K * (Q_DES_0 - q) - 10 * B * dq;								
                                t0 = nowtime();
                            }

                            publish(tau_fb, utime, t, dt);

                            if (DEBUG) {
								vpDisplay::displayArrow(I, vpImagePoint(cam.get_v0(), cam.get_u0()),
														vpImagePoint(cam.get_v0(), cam.get_u0() + v_c[0] * 10000),
														vpColor::red);
								vpDisplay::displayArrow(I, vpImagePoint(cam.get_v0(), cam.get_u0()),
														vpImagePoint(cam.get_v0() + v_c[1] * 10000, cam.get_u0()),
														vpColor::green);

								vpDisplay::displayArrow(I, vpImagePoint(cam.get_v0(), cam.get_u0()),
														vpImagePoint(cam.get_v0() + v_c[1] * 10000,
																	cam.get_u0() + v_c[0] * 10000), vpColor(254,188,0));
																									
                                vpDisplay::displayFrame(I, cdMo, cam, opt_square_width, vpColor::none, 1);
                                vpDisplay::displayFrame(I, cMo, cam, opt_square_width, vpColor::none, 2);
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
						for (int i = 0; i < k; i++) {
							std::stringstream ss; ss << i;
							vpDisplay::displayText(I, blob[i].getCog(), ss.str(), vpColor::white); // number of point
						}
                        if (k == Np) {
                            learn = false;
                            k = 0;
							for (int i = 0; i < 7; i++) {
								q_des(i) = lcm_status.joint_position_measured[i];
								Q_DES_0(i) = lcm_status.joint_position_measured[i];
							}
                        }
                    }

                    vpDisplay::flush(I);

					double loop_deltat = vpTime::measureTimeMs() - loop_start_time;
                    ss.str("");
                    ss << "Loop time: " << loop_deltat << " ms";
					// std::cout << ss.str() << std::endl;
                    vpDisplay::displayText(I, 40, 20, ss.str(), vpColor(254, 188, 0));
                    vpDisplay::flush(I);
					
					if (loop_deltat < 2.0) {
						int64_t loop_deltat_add = (2.0 - loop_deltat) * 1000;
						std::this_thread::sleep_for(std::chrono::microseconds(loop_deltat_add));
					}

					loop_deltat = vpTime::measureTimeMs() - loop_start_time;
                    ss.str("");
                    ss << "Loop time: " << loop_deltat << " ms";
					// std::cout << ss.str() << std::endl;

                    vpMouseButton::vpMouseButtonType button;
                    if (vpDisplay::getClick(I, button, false)) {
						std::cout << "Click mouse -> publish cmd 0" << std::endl;
     					// tau_fb.setZero();
                        // publish(tau_fb, utime, t, dt);
                        q_delta.setZero();
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
					// tau_fb.setZero();
					// publish(tau_fb, utime, t, dt);
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

		std::cout << "tau_fb::";
		for (int i = 0; i < 7; i++) {
			std::cout << tau_fb(i) << " \t";
		}
		std::cout << std::endl;

		lcm_command.utime = utime;
		lcm_command.num_joints = kNumJoints;
 		lcm_command.num_torques = kNumJoints;
		lcm_command.joint_position.resize(kNumJoints, 0);
		lcm_command.joint_torque.resize(kNumJoints, 0);
		for (int i = 0; i < kNumJoints; i++ ) {
			lcm_command.joint_position[i] = lcm_status.joint_position_measured[i];
			if (abs(tau_fb(i)) > 30) {
				std::cout << "TORQUE UPER 30" << std::endl;
				lcm_command.joint_torque[i] = 0;
			} else {
				lcm_command.joint_torque[i] = tau_fb[i];
			}
		}

		lcm.publish("IIWA_COMMAND", &lcm_command);
	}

	int camera_test() {
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
	return 0;
	}

	int vel_ctrl_test() {
		if (!lcm.good())
			return 1;

		lcm.subscribe(kLcmStatusChannel, &Controller::handleFeedbackMessage, this);

		// std::this_thread::sleep_for(std::chrono::milliseconds(x));

		bool init = true;
		
		Eigen::Matrix<double, 7, 1> dei; dei.setZero();
		Eigen::Matrix<double, 7, 1> tau_fb; tau_fb.setZero();

		Eigen::Matrix<double, 6, 7> Jacobian;
		Eigen::Matrix<double, 7, 7> Kvc; Kvc.setZero();
		Eigen::Matrix<double, 7, 7> Ivc; Ivc.setZero();
		Eigen::Matrix<double, 6, 1> delta_x;
		Eigen::Matrix<double, 7, 1> dq_des;
		Eigen::Matrix<double, 6, 1> v_c; v_c.setZero();
		Eigen::Matrix<double, 7,6> pinvJ;
		Eigen::Matrix<double, 7, 1> Q_DES_0; Q_DES_0.setZero();
		Eigen::Matrix<double, 7, 1> q_delta;
		q_delta.setZero();


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
					Q_DES_0(i) = lcm_status.joint_position_measured[i];
					kdl_q(i) = lcm_status.joint_position_measured[i];
					kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
				}
				// jnt_to_pose_solver->JntToCart(kdl_q, kdl_x0);
			}

			for (int i = 0; i < kNumJoints; i++) {
				kdl_q(i) = lcm_status.joint_position_measured[i];
				kdl_dq(i) = lcm_status.joint_velocity_estimated[i];
			}
			jnt_to_pose_solver->JntToCart(kdl_q, kdl_x);
			jnt_to_jac_solver->JntToJac(kdl_q, kdl_J);
			// kdl_J.changeRefFrame(kdl_x);
			// kdl_J.changeRefPoint(kdl_x);

			// bool 	KDL::changeBase (const Jacobian &src1, const Rotation &rot, Jacobian &dest)
			// bool 	KDL::changeRefPoint (const Jacobian &src1, const Vector &base_AB, Jacobian &dest)
			// bool 	KDL::changeRefFrame (const Jacobian &src1, const Frame &frame, Jacobian &dest)


			for (int i = 0; i < 3; i++) {
				std::cout << kdl_x.p(i) << std::endl;
			}
			double roll,pitch,yaw;
			// kdl_x.M.GetRPY(roll,pitch,yaw);
			kdl_x.M.GetEulerZYX(roll,pitch,yaw);
			std::cout << "RPY: " << roll * 180/3.1415 << "\t" << pitch * 180/3.1415 << "\t" << yaw * 180/3.1415 << std::endl;
			
			std::cout << "***\n";
			std::cout << iiwa_chain.getNrOfJoints() << std::endl;
 			std::cout << iiwa_chain.getNrOfSegments() << std::endl;

			for (int i = 0; i < iiwa_chain.getNrOfSegments(); i++ ) {
				std::cout << iiwa_chain.getSegment(i).getName() << std::endl;
			}

			// KDL::Frame p_out = KDL::Frame::Identity();
			// int j=0;
			// for(unsigned int i=0; i<iiwa_chain.getNrOfSegments();i++){
			// 	if(iiwa_chain.getSegment(i).getJoint().getType()!=KDL::Joint::None){
			// 		p_out = p_out*iiwa_chain.getSegment(i).pose(q(j));
			// 		j++;
			// 	}else{
			// 		p_out = p_out*iiwa_chain.getSegment(i).pose(0.0);
			// 	}
			// }

			// for (int i = 0; i < 3; i++) {
			// 	std::cout << p_out.p(i) << std::endl;
			// }
			
			// // kdl_x.M.GetRPY(roll,pitch,yaw);
			// p_out.M.GetEulerZYX(roll,pitch,yaw);
			// std::cout << "RPY: " << roll * 180/3.1415 << "\t" << pitch * 180/3.1415 << "\t" << yaw * 180/3.1415 << std::endl;

			std::cout << "***\n";

			for (int i = 0; i < 6; i++) {
				for (int j = 0; j < 7; j++) {
					Jacobian(i, j) = kdl_J(i, j);
				}
			}


			Eigen::Matrix<double, 3, 3> Rzero = Eigen::Matrix<double, 3, 3>::Zero();
            Eigen::Matrix<double, 3, 3> R;
			Eigen::Matrix<double, 3, 3> Rp;
			Eigen::Matrix<double, 3, 3> skp;
			R << kdl_x.M.data[0], kdl_x.M.data[1], kdl_x.M.data[2], kdl_x.M.data[3], kdl_x.M.data[4], kdl_x.M.data[5], kdl_x.M.data[6], kdl_x.M.data[7], kdl_x.M.data[8];
			skp << 0,	-kdl_x.p(2),	kdl_x.p(1),
				kdl_x.p(2),	0,	-kdl_x.p(0),
				kdl_x.p(1), kdl_x.p(0), 0;
			Rp = skp * R;

			Eigen::Matrix<double, 6, 6> V;	// world to ee
            V << R, Rp, Rzero, R;
			
			Jacobian = V * Jacobian;

			pinv2(Jacobian, pinvJ);
			
			v_c << 0.00, 0.00, 0.01, 0.0, 0.0, 0.00;
			dq_des =  pinvJ * v_c;
			q_delta += dq_des * dt;
			
			std::cout << "q_delta:   ";
			for (int i = 0; i < 7; i++) {
				if (abs(q_delta(i)) > 0.5) {
					q_delta(i) + copysign(0.5, q_delta(i));
				}
				std::cout << q_delta(i) << "\t";
			}
			std::cout << std::endl;

			q_des = Q_DES_0 + q_delta;
			tau_fb = 10 * K * (q_des - q) - 10 * B * dq;
			
			publish(tau_fb, utime, t, dt);
		}
		return 0;
	}

};

int main(int argc, char **argv)
{
	std::cout << "Node started" << std::endl;

	Controller controller;
	// controller.loop_vs();
	// controller.camera_test();
	controller.vel_ctrl_test();

	return 0;
}