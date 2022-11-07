#include "fine_measuarments_emulator.hpp"


FineMeasurementsEmulalor::FineMeasurementsEmulalor() {
    is_inited = false;
    R << 0, 1, -1, 0;

    Eigen::Matrix<double, 2, 1> state_0;
    state_0.setZero();
    search_init(state_0);
}

FineMeasurementsEmulalor::FineMeasurementsEmulalor(int _m_0) : m_0(_m_0) {
    is_inited = false;
    R << 0, 1, -1, 0;

    Eigen::Matrix<double, 2, 1> state_0;
    state_0.setZero();
    search_init(state_0);
}

FineMeasurementsEmulalor::~FineMeasurementsEmulalor() {}

void FineMeasurementsEmulalor::init(int _m_0) {
    m_0 = _m_0;
}

void FineMeasurementsEmulalor::init_isotropic_surface_data(const size_t size) {
    // isotropic config
    // const size_t size_x = 10;
    // const size_t size_y = 10;

    size_t size_x = size;
    size_t size_y = size;

    double A = 100.0;
    double x0 = 0.0;
    double y0 = 0.0;
    double sigma_X = 0.08;
    double sigma_Y = 0.08;

    Eigen::Matrix<double, Eigen::Dynamic, 1> xg;
    xg.setLinSpaced(size_x, -10.0f, 10.0f);
    xg = xg * 0.01;

    Eigen::Matrix<double, Eigen::Dynamic, 1> yg;
    yg.setLinSpaced(size_y, -10.0f, 10.0f);
    yg = yg * 0.01;

    // // xg.setLinSpaced(size_x,-1,1);
    // // yg.setLinSpaced(size_y,-1,1);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X;
    X = DynamicXXd::Zero(size_y, size_x);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Y;
    Y = DynamicXXd::Zero(size_y, size_x);

    for (int i = 0; i < size_y; i++) {
        for (int j = 0; j < size_x; j++) {
            X(i, j) = xg(j);
            Y(i, j) = yg(i);
        }
    }

    double theta = M_PI;

    double a = cos(theta) * cos(theta) / (2 * sigma_X * sigma_X) + sin(theta) * sin(theta) / (2 * sigma_Y * sigma_Y);
    double b = -sin(2 * theta) / (4 * sigma_X * sigma_X) + sin(2 * theta) / (4 * sigma_Y * sigma_Y);
    double c = sin(theta) * sin(theta) / (2 * sigma_X * sigma_X) + cos(theta) * cos(theta) / (2 * sigma_Y * sigma_Y);

    std::cout << "a,b,c = " << a << " " << b << " " << c << std::endl;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> aXXdet;
    aXXdet = DynamicXXd::Zero(size_y, size_x);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bbXYdet;
    bbXYdet = DynamicXXd::Zero(size_y, size_x);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cYYdet;
    cYYdet = DynamicXXd::Zero(size_y, size_x);

    Z.resize(size_y, size_x);
    Z.setZero();  // must be squere
    for (int i = 0; i < size_y; i++) {
        for (int j = 0; j < size_x; j++) {
            aXXdet(i, j) = a * (X(i, j) - x0) * (X(i, j) - x0);
            bbXYdet(i, j) = (2 * b) * (X(i, j) - x0) * (Y(i, j) - y0);
            cYYdet(i, j) = c * (Y(i, j) - y0) * (Y(i, j) - y0);
            Z(i, j) = A - A * exp(-(aXXdet(i, j) + bbXYdet(i, j) + cYYdet(i, j)));
        }
    }
    std::cout << Z << std::endl;
}

// void FineMeasurementsEmulalor::init_isotropic_surface_data(const size_t size) {
//     // isotropic config
//     // const size_t size_x = 10;
//     // const size_t size_y = 10;

//     const size_t size_x = size;
//     const size_t size_y = size;

//     double A = 100.0;
//     double x0 = 0.0;
//     double y0 = 0.0;
//     double sigma_X = 0.04;
//     double sigma_Y = 0.04;

//     Eigen::Matrix<double, size_x, 1> xg;
//     xg.setLinSpaced(size_x, -10.0f, 10.0f);
//     xg = xg * 0.01;
//     Eigen::Matrix<double, size_y, 1> yg;
//     yg.setLinSpaced(size_y, -10.0f, 10.0f);
//     yg = yg * 0.01;

//     // xg.setLinSpaced(size_x,-1,1);
//     // yg.setLinSpaced(size_y,-1,1);

//     Eigen::Matrix<double, size_y, size_x> X;
//     X.setZero();
//     Eigen::Matrix<double, size_y, size_x> Y;
//     Y.setZero();
//     for (int i = 0; i < size_y; i++) {
//         for (int j = 0; j < size_x; j++) {
//             X(i, j) = xg(j);
//             Y(i, j) = yg(i);
//         }
//     }

//     double theta = M_PI;

//     double a = cos(theta) * cos(theta) / (2 * sigma_X * sigma_X) + sin(theta) * sin(theta) / (2 * sigma_Y * sigma_Y);
//     double b = -sin(2 * theta) / (4 * sigma_X * sigma_X) + sin(2 * theta) / (4 * sigma_Y * sigma_Y);
//     double c = sin(theta) * sin(theta) / (2 * sigma_X * sigma_X) + cos(theta) * cos(theta) / (2 * sigma_Y * sigma_Y);

//     std::cout << "a,b,c = " << a << " " << b << " " << c << std::endl;

//     Eigen::Matrix<double, size_y, size_x> aXXdet;
//     aXXdet.setZero();
//     Eigen::Matrix<double, size_y, size_x> bbXYdet;
//     bbXYdet.setZero();
//     Eigen::Matrix<double, size_y, size_x> cYYdet;
//     cYYdet.setZero();

//     Z.resize(size_y, size_x);
//     Z.setZero();  // must be squere
//     for (int i = 0; i < size_y; i++) {
//         for (int j = 0; j < size_x; j++) {
//             aXXdet(i, j) = a * (X(i, j) - x0) * (X(i, j) - x0);
//             bbXYdet(i, j) = (2 * b) * (X(i, j) - x0) * (Y(i, j) - y0);
//             cYYdet(i, j) = c * (Y(i, j) - y0) * (Y(i, j) - y0);
//             Z(i, j) = A - A * exp(-(aXXdet(i, j) + bbXYdet(i, j) + cYYdet(i, j)));
//         }
//     }
//     // std::cout << Z << std::endl;
// }


void FineMeasurementsEmulalor::get_isotropic_surface_data(double &s, Eigen::Matrix<double, 2, 1> y) {
    s = Z(y(0), y(1));
}

void FineMeasurementsEmulalor::search_init(const Eigen::Matrix<double, 2, 1> &state_0) {
    state.setZero();
    state = state_0;
    shift.setZero();
    shift << 1, 0;
    l = 0;
    s_prev = 9999;
    m = 10;
    is_arrived = false;

    is_inited = true;
}

void FineMeasurementsEmulalor::update(double s, Eigen::Matrix<double, 2, 1> &y, Eigen::Matrix<double, 2, 1> &shift_,  bool &arrived) {
    if (s >= 2) {
        std::cout << l << "\t" << m << std::endl;
        if (l > m) {
            shift = R * shift;
            l = 0;
            // if (m > 1) {
            //     m = m - 1;
            // }
        } else {
            state = state + shift;
            if (s > s_prev) {
                l = l + 1;
            }
            s_prev = s;
        }
    } else {
        is_arrived = true;
    }

    arrived = is_arrived;
    y = state;
    
    shift_ = shift;
}

// /* test isotropic surface data*/
// void test_1() {
//     FineMeasurementsEmulalor fm_emulator(1);
//     fm_emulator.init_isotropic_surface_data(10);

//     double s;
//     Eigen::Matrix<double, 2, 1> y;
//     y << 4, 6;
//     fm_emulator.get_isotropic_surface_data(s, y);
//     std::cout << "Emulated signal is " << s << std::endl;
// }

// /* test snake search algorithm */
// void test_2() {
//     FineMeasurementsEmulalor fm_emulator(1);
//     fm_emulator.init_isotropic_surface_data(10);

//     Eigen::Matrix<double, 6, 1> v_c;

//     bool system_state = 1;                // [0, 1, 2] visp, snake, off
//     Eigen::Matrix<double, 2, 1> state_0;  // get from CV
//     state_0.setZero();

//     Eigen::Matrix<double, 2, 1> y(state_0);

//     bool arrived = false;
//     while (!arrived) {
//         if (system_state == 1) {
//             // wait_while abs(dq.sum()) < 0.001
//             if (!fm_emulator.is_inited) {
//                 fm_emulator.search_init(state_0);
//                 y = state_0;
//             }

//             double s;
//             y(0) = std::rand() % 10;
//             y(1) = std::rand() % 10;
//             fm_emulator.get_isotropic_surface_data(s, y);
//             fm_emulator.update(s, y, arrived);
//             std::cout << y.transpose() << "\t" << s << "\n";

//             if (arrived) {
//                 system_state = 2;
//             } else {
//                 // robot control using `y`
//                 double x_offset = y(0);
//                 double y_offset = y(1);

//                 v_c.setZero();
//                 v_c(4) = x_offset;
//                 v_c(5) = y_offset;
//             }
//         }
//         std::this_thread::sleep_for(std::chrono::seconds(1));
//     }
// }

// int main() {
//     test_1();
//     test_2();

//     return 0;
// }
