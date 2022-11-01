#include "kinematics.h"


Kinematics::Kinematics() {};
Kinematics::~Kinematics() {};

/* q \in \mathbb{R} */
double Kinematics::normalized_angle(double q, int j) {
  if (j != 2) {
    while (q < 0.0) {
      q += 2. * M_PI;
    }
    while (q >= 2 * M_PI) {
      q -= 2. * M_PI;
    }
  } else {
    while (q > 0.0) {
      q -= 2. * M_PI;
    }
    while (q <= -2 * M_PI) {
      q += 2. * M_PI;
    }
   }
  return q;
}

void Kinematics::get_jacobe(Eigen::Matrix<double, 7, 1> q, Eigen::Matrix<double, 6, 7>& J) {
  Eigen::Matrix<double, 7, 1> theta;
  theta << 0, 0, 0, 0, 0, 0, 0;
  // todo: chech offsets and directin of matlab and real iiwa
  double theta_1 = theta(0) - q(0);
  double theta_2 = theta(1) - q(1);
  double theta_3 = theta(2) - q(2);
  double theta_4 = theta(3) - q(3);
  double theta_5 = theta(4) - q(4);
  double theta_6 = theta(5) - q(5);
  double theta_7 = theta(6) - q(6);

  
}
