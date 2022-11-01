#ifndef KINEMATICS_
#define KINEMATICS_

#include <iostream>
#include <iomanip>
#include <limits>

#include <Eigen/Geometry>
#include <Eigen/Dense>

class Kinematics {
 public:
  Kinematics();
  ~Kinematics();

//   Vector6d forward(VectorNd q, int from=0, int to=5);
//   ConfigurationsOfManipulator inverse(Vector6d s);

  void get_jacobe(Eigen::Matrix<double, 7, 1> q, Eigen::Matrix<double, 6, 7>& J);
  double normalized_angle(double q, int j);
};

#endif
