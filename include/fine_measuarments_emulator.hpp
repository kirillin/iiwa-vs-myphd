#ifndef FINE_MEASUREMENTS_EMULATOR_
#define FINE_MEASUREMENTS_EMULATOR_

#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class FineMeasurementsEmulalor {
  private:
    double m_0;

    Eigen::Matrix<double, 2 ,1> state;
    int l;
    double s_prev;
    int m;
    Eigen::Matrix<double, 2 ,1> shift;
    Eigen::Matrix<double, 2 ,2> R;
    bool is_arrived;

    Eigen::MatrixXd Z;
  
  public:
    bool is_inited;
    FineMeasurementsEmulalor();
    FineMeasurementsEmulalor(int);
    
    ~FineMeasurementsEmulalor();

    void init_isotropic_surface_data();
    void get_isotropic_surface_data(double&, Eigen::Matrix<double, 2 ,1>);

    void search_init(const Eigen::Matrix<double, 2 ,1> &state_0);
    void update(double s, Eigen::Matrix<double, 2 ,1> &y, bool &arrived);

    // /* test isotropic surface data*/
    // static void test_1();

    // /* test snake search algorithm */
    // static void test_2();

};

#endif
