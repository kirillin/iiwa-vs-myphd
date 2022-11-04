#ifndef UTILS_
#define UTILS_

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace utils {

int64_t micros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double nowtime() {
    auto current_time = std::chrono::system_clock::now();
    auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
    double num_seconds = duration_in_seconds.count();
    return num_seconds;
}

void pinv(Eigen::Matrix<double, 8, 6> &L, Eigen::Matrix<double, 6, 8> &pinvL, double alpha0 = 0.001, double w0 = 0.0001) {
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

void pinv2(Eigen::Matrix<double, 6, 7> &L, Eigen::Matrix<double, 7, 6> &pinvL, double alpha0 = 0.001, double w0 = 0.0001) {
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

void pinv3(Eigen::Matrix<double, 8, 7> &L, Eigen::Matrix<double, 7, 8> &pinvL, double alpha0 = 0.001, double w0 = 0.0001) {
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

int skew(Eigen::Matrix<double, 3, 1> fpe, Eigen::Matrix<double, 3, 3> &skew_fpe) {
    skew_fpe << 0, -fpe(2), fpe(1),
        fpe(2), 0, -fpe(0),
        -fpe(1), fpe(0), 0;
    return 0;
}

int make_adjoint(Eigen::Matrix<double, 6, 6> &fVe, const Eigen::Matrix<double, 3, 1> &fpe, const Eigen::Matrix<double, 3, 3> &fRe) {
    Eigen::Matrix<double, 3, 3> Rzero = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 3, 3> pR;
    Eigen::Matrix<double, 3, 3> skew_fpe;
    utils::skew(fpe, skew_fpe);
    pR = -skew_fpe * fRe.transpose();
    fVe << fRe.transpose(), Rzero, pR, fRe.transpose();
    return 0;
}

};  // namespace utils

#endif