
int int main(int argc, char const *argv[])
{
    

     vpRealSense2 rs;
     rs2::config config;
     unsigned int width = 640, height = 480;
     config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGBA8, 30);
     config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
     config.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8, 30);
     rs.open(config);
  
     // Get camera extrinsics
     vpPoseVector ePc;
     // Set camera extrinsics default values
     ePc[0] = 0.0337731;
     ePc[1] = -0.00535012;
     ePc[2] = -0.0523339;
     ePc[3] = -0.247294;
     ePc[4] = -0.306729;
     ePc[5] = 1.53055;
  
     // If provided, read camera extrinsics from --eMc <file>
     if (!opt_eMc_filename.empty()) {
       ePc.loadYAML(opt_eMc_filename, ePc);
     } else {
       std::cout << "Warning, opt_eMc_filename is empty! Use hard coded values."
                 << "\n";
     }
     vpHomogeneousMatrix eMc(ePc);
     std::cout << "eMc:\n" << eMc << "\n";
  
     // Get camera intrinsics
     vpCameraParameters cam =
         rs.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);
     std::cout << "cam:\n" << cam << "\n";
  
     vpImage<unsigned char> I(height, width);
  
 #if defined(VISP_HAVE_X11)
     vpDisplayX dc(I, 10, 10, "Color image");
 #elif defined(VISP_HAVE_GDI)
     vpDisplayGDI dc(I, 10, 10, "Color image");
 #endif
  
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
     vpHomogeneousMatrix cdMo(vpTranslationVector(0, 0, opt_tagSize * 3), // 3 times tag with along camera z axis
                              vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));
  
     // Create visual features
     std::vector<vpFeaturePoint> p(4), pd(4); // We use 4 points
  
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
       vpAdaptiveGain lambda(1.5, 0.4, 30); // lambda(0)=4, lambda(oo)=0.4 and lambda'(0)=30
       task.setLambda(lambda);
     } else {
       task.setLambda(0.5);
     }
  
     vpPlot *plotter = nullptr;
     int iter_plot = 0;
  
     if (opt_plot) {
       plotter = new vpPlot(2, static_cast<int>(250 * 2), 500, static_cast<int>(I.getWidth()) + 80, 10,
                            "Real time curves plotter");
       plotter->setTitle(0, "Visual features error");
       plotter->setTitle(1, "Camera velocities");
       plotter->initGraph(0, 8);
       plotter->initGraph(1, 6);
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
     }
  
     bool final_quit = false;
     bool has_converged = false;
     bool send_velocities = false;
     bool servo_started = false;
     std::vector<vpImagePoint> *traj_corners = nullptr; // To memorize point trajectory
  
     static double t_init_servo = vpTime::measureTimeMs();
  
     robot.set_eMc(eMc); // Set location of the camera wrt end-effector frame
     robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);





    return 0;
}
