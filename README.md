<div align="center">
  <h1>DCH-SLAM: A CUDA-Accelerated Dynamic-Aware Visual SLAM<br>for Real-Time Deployment on Edge GPUs</h1>
  <strong>ICTA 2026 (under review)</strong>
  <br>
    <a href="#" target="_blank">Dat Bui Minh Tuan</a><sup>1</sup>,
    <a href="#" target="_blank">Cuong Phan Trong</a><sup>1</sup>,
    <a href="#" target="_blank">Hung Nguyen Viet</a><sup>1</sup>,
    <a href="#" target="_blank">Mai Cao Van</a><sup>1</sup>
  <p>
    <h5>
      <sup>1</sup>FPT University, Vietnam
    </h5>
  </p>

  [<img src="https://img.shields.io/badge/GitHub-DCH--SLAM-black?logo=github" alt="GitHub">](https://github.com/datrich/DCH-SLAM)
  [<img src="https://img.shields.io/badge/Platform-Jetson_Orin_NX-green?logo=nvidia" alt="Jetson">](#)
  [<img src="https://img.shields.io/badge/CUDA-12.6-blue?logo=nvidia" alt="CUDA">](#)

</div>

<p align="center">
  <img src="assets/bonn_crowd_small.gif" alt="Bonn crowd" width="240">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/bonn_mov_small.gif" alt="Bonn moving" width="240">
</p>

**DCH-SLAM** is a visual SLAM system for dynamic environments built on top of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) and [NGD-SLAM](https://github.com/yuhaozhang7/NGD-SLAM). It unifies three lines of work:

- **CUDA-accelerated ORB front-end** — Ported from [Jetson-SLAM](https://github.com/KumarRobotics/jetson-slam): bounded rectification, pyramidal feature culling & aggregation (PyCA), and zero-copy shared-memory design for high-frame-rate feature extraction on embedded GPUs.
- **Adaptive Barron robust kernel** — Replaces the fixed Huber kernel in Local Bundle Adjustment (LBA) with an adaptive Barron kernel whose shape parameter α is estimated online from the residual distribution, down-weighting both known and unknown dynamic-object outliers.
- **Asynchronous YOLO semantic thread** — Follows the NGD-SLAM architecture: YOLO detection runs in a separate thread and is queried every *k* = 3 frames, decoupling inference latency from tracking and enabling real-time operation on edge hardware.

### Results on Jetson Orin NX (TUM fr3/walking_xyz, 3 runs)

| Pipeline | RMSE (cm) | Track. latency | FPS |
|----------|-----------|---------------|-----|
| NGD-SLAM baseline (Huber + CPU ORB + sync YOLO) | 3.46 | 61.9 ms | 16.2 |
| **DCH-SLAM** (Barron + CUDA ORB + async YOLO) | **3.51** | **50.7 ms** | **19.7** |

---

# 1. Prerequisites

Tested on **Ubuntu 20.04** and **22.04**. For the CUDA front-end, an NVIDIA GPU with CUDA ≥ 11.0 is required (Jetson Orin NX uses SM87; desktop RTX uses SM89/SM120).

## C++17 Compiler
```bash
sudo apt install build-essential
```

## CUDA Toolkit
Required for the CUDA ORB front-end (`USE_CUDA_FRONTEND=ON`). Install from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- Jetson Orin NX: CUDA 12.6 (via JetPack / L4T R36.5)
- Desktop: CUDA 11.8 or later

## Pangolin
Used for visualization. Download and install from [github.com/stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin).

## OpenCV ≥ 4.4
```bash
sudo apt install libopencv-dev
```

## Eigen3 ≥ 3.1
```bash
sudo apt install libeigen3-dev
```

## YOLO (included in `Thirdparty/`)
Uses the [C++ OpenCV DNN version](https://github.com/hpc203/yolov34-cpp-opencv-dnn) of [YOLO-fastest](https://github.com/dog-qiuqiu/Yolo-Fastest). Model config and weights are in `Thirdparty/YOLO/` and loaded via OpenCV (CPU backend). The semantic thread runs asynchronously every *k* frames — tracking never blocks on inference.

## DBoW2 and g2o (included in `Thirdparty/`)
Modified versions of [DBoW2](https://github.com/dorian3d/DBoW2) (place recognition) and [g2o](https://github.com/RainerKuemmerle/g2o) (nonlinear optimization). The g2o fork adds `robust_kernel_barron.h/.cpp` implementing the adaptive Barron loss.

## Python
Required for trajectory alignment with ground truth.
```bash
pip install numpy evo
```

---

# 2. Building DCH-SLAM

```bash
git clone https://github.com/datrich/DCH-SLAM.git DCH-SLAM
cd DCH-SLAM
```

Extract the ORB vocabulary:
```bash
cd Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd ..
```

### CPU-only build (Barron kernel + async YOLO, no CUDA)
```bash
chmod +x build.sh
./build.sh
```

### CUDA build (Barron kernel + CUDA ORB + async YOLO)
```bash
chmod +x build.sh
USE_CUDA_FRONTEND=ON ./build.sh
```
Or manually:
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA_FRONTEND=ON -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="87"   # change to 89 for RTX 40xx, 120 for RTX 50xx
make -j$(nproc)
```

---

# 3. Running Examples

## TUM RGB-D Dynamic Sequences

Download from [cvg.cit.tum.de/data/datasets/rgbd-dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Example for `freiburg3_walking_xyz`:

```bash
./Examples/RGB-D/rgbd_tum \
  ./Vocabulary/ORBvoc.txt \
  ./Examples/RGB-D/TUM3.yaml \
  /path/to/rgbd_dataset_freiburg3_walking_xyz \
  ./Examples/RGB-D/associations/fr3_walk_xyz.txt
```

## Bonn RGB-D Dynamic Dataset

Download from [www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset](http://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/). Example for the balloon sequence:

```bash
./Examples/RGB-D/rgbd_tum \
  ./Vocabulary/ORBvoc.txt \
  ./Examples/RGB-D/BonnRGBD.yaml \
  /path/to/rgbd_bonn_balloon \
  /path/to/rgbd_bonn_balloon/associations.txt
```

---

# 4. Key Differences from NGD-SLAM

| Component | NGD-SLAM | DCH-SLAM |
|-----------|----------|----------|
| ORB front-end | CPU | CPU **or** CUDA (USE_CUDA_FRONTEND=ON) |
| Robust kernel (LBA) | Fixed Huber | **Adaptive Barron** (α estimated online) |
| YOLO invocation | Synchronous (blocks tracking) | **Asynchronous** (every k=3 frames) |
| Target hardware | Laptop CPU | Laptop CPU **+ Jetson / desktop GPU** |

---

# 5. Citation

If you find DCH-SLAM useful in your research, please cite:

```bibtex
@inproceedings{bui2026dchslam,
  title={{DCH-SLAM}: A CUDA-Accelerated Dynamic-Aware Visual SLAM for Real-Time Deployment on Edge GPUs},
  author={Bui Minh Tuan, Dat and Phan Trong, Cuong and Nguyen Viet, Hung and Cao Van, Mai},
  booktitle={Proceedings of the International Conference on Intelligent Technologies and Applications (ICTA)},
  year={2026}
}
```

DCH-SLAM builds on top of the following works — please also cite them:

```bibtex
@inproceedings{zhang2025ngdslam,
  title={{NGD-SLAM}: Towards Real-Time Dynamic SLAM without GPU},
  author={Zhang, Yuhao and Bujanca, Mihai and Luj{\'a}n, Mikel},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3467--3473},
  year={2025},
  doi={10.1109/IROS60139.2025.11246202}
}

@article{campos2021orbslam3,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM},
  author={Campos, Carlos and Elvira, Richard and Rodr{\'\i}guez, Juan J G{\'o}mez and Montiel, Jos{\'e} MM and Tard{\'o}s, Juan D},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={6},
  pages={1874--1890},
  year={2021},
  doi={10.1109/TRO.2021.3075644}
}

@article{kumar2023jetsonslam,
  title={High-Speed Stereo Visual SLAM for Low-Powered Computing Devices},
  author={Kumar, Ashish and Park, Jaehyun and Behera, Laxmidhar},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={2},
  pages={499--506},
  year={2023},
  doi={10.1109/LRA.2022.3228183}
}
```
