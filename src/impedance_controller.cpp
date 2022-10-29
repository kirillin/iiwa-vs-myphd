
#include <stdio.h>
#include <lcm/lcm-cpp.hpp>
#include <cassert>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"

using drake::lcmt_iiwa_command;
using drake::lcmt_iiwa_status;

const char* kLcmStatusChannel = "IIWA_STATUS";
const char* kLcmCommandChannel = "IIWA_COMMAND";
const int kNumJoints = 7;

class ImpedanceController {

  public:
    ~ImpedanceController() {
        // robot state
        Eigen::Matrix<double, kNumJoints, 1>  q;
        Eigen::Matrix<double, kNumJoints, 1>  dq;
        
        q.setZero();
        dq.setZero();

        // stiffness and damping parameters
        Eigen::Matrix<double, kNumJoints, 1>  k;
        Eigen::Matrix<double, kNumJoints, 1>  b;
        Eigen::Matrix<double, kNumJoints, kNumJoints>  K;
        Eigen::Matrix<double, kNumJoints, kNumJoints>  B;

        k << 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0;
        b << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        K = k.asDiagonal();
        B = b.asDiagonal();

        // robot desired position
        Eigen::Matrix<double, kNumJoints, 1>  q_des;

        q_des << 2.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0;

    }

    void handleFeedbackMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan,
                       const drake::lcmt_iiwa_status *msg) {

        int t = msg->utime;
        if (msg->num_joints == kNumJoints) {
            double msg->joint_position_measured[num_joints];
            double msg->joint_velocity_estimated[num_joints];
            double msg->joint_position_commanded[num_joints];
            double msg->joint_position_ipo[num_joints];
            double msg->joint_torque_measured[num_joints];
            double msg->joint_torque_commanded[num_joints];
            double msg->joint_torque_external[num_joints];



        } else {
            assert(("Number joint in message from FRI faild!"));
        }



        // cur_t = time.time()
        // print("freq ",1.0/(cur_t-prev_t), "\tt: ", t)
        // prev_t = cur_t

        // printf("Received message on channel \"%s\":\n", chan.c_str());
        // printf("  timestamp   = %lld\n", (long long) msg->timestamp);
        // printf("  position    = (%f, %f, %f)\n", msg->position[0], msg->position[1],
        //        msg->position[2]);
        // printf("  orientation = (%f, %f, %f, %f)\n", msg->orientation[0], msg->orientation[1],
        //        msg->orientation[2], msg->orientation[3]);
        // printf("  ranges:");
        // for (i = 0; i < msg->num_ranges; i++)
        //     printf(" %d", msg->ranges[i]);
        // printf("\n");
        // printf("  name        = '%s'\n", msg->name.c_str());
        // printf("  enabled     = %d\n", msg->enabled);
    }

    void update(double dt) {
       

    }

};

int main(int argc, char **argv)
{
    lcm::LCM lcm;

    if (!lcm.good())
        return 1;

    ImpedanceController impedanceController;
    lcm.subscribe(kLcmStatusChannel, &ImpedanceController::handleFeedbackMessage, &impedanceController);

    while (0 == lcm.handle()) {
        // Do nothing
    }

    return 0;
}
