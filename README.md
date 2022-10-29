### Two-Stage Motion Control System for KUKA lbr iiwa

Two-Stage positioning system using exteroceptive sensors and finita state machine:
- stage 1 - Image based visual servoing via eye-in-hand schema
- stage 2 - Heuristic search algorithm with unknown gradient

## Setup and install

1. Find the KUKA lbr iiwa robot
2. Connect it to a external rt-linux pc to iiwa KONI interface via an ethernet cable 5e+ category
3. Connect [INTEL RealSense](https://github.com/IntelRealSense/librealsense) camera to the robot and to the external computer
4. Install [drake-iiwa-driver](https://github.com/RobotLocomotion/drake-iiwa-driver)
5. Build this repository

    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```
6. Setup the environment with the robot, a visual marker, and any second sensor


## Run

```bash
kuka_driver
iiwa-vs-myphd
```

### Authors

- **Kirill Artemov** - [kirillin](https://github.com/kirillin)
- **Sergey Kolyubin** - [se-ko](https://github.com/se-ko)

