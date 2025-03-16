# MegaStation - NSDI'25

Code base for [Building Massive MIMO Baseband Processing on a Single-Node Supercomputer](https://www.usenix.org/conference/nsdi25/presentation/xie).

## Prerequisite

- cmake >= 3.27
- gcc >= 13 (with C++20 support)
- cuda toolkit >= 12.4
- python >= 3.9
- MATLAB >= 2024b
- Access to NVIDIA [Aerial](https://developer.nvidia.com/aerial) source code (Aerial-cuBB-source-23.03.0-Rel-23-3.66-x86_64.tar.gz)
- [gdrcopy](https://github.com/NVIDIA/gdrcopy)

*Note*: We don't have right to release Aerial's source code, even partially. In order to run the code, please first apply NVIDIA Aerial developer program to get Aerial's code.

You should apply [patches](third_party/aerial-ldpc-encode.patch) to Aerial library and don't forget to modify cuda architecture in `cuBB/cuPHY/CMakeLists.txt`.


## Build

```bash
# inside the project folder
mkdir build && cd build
cmake ..
make -j
```

## Run

1. **Generate data**

For example, to generate 64x16 (antennas x users) uplink & downlink data,
```bash
# inside the project folder
python generate_data.py --build ./build/ --ants 64 --users 16 --ofdm 1200 --sg 16
```

2. **Run test**

We provide a set of unit tests to test whether our code is able to run on your machine.

```bash
# inside the project folder
cd build
ctest
```

3. **Run code**

To run the code,
```bash
# inside the build folder
./mega -ants 64 -users 16 -ofdm 1200 -sg 16 --up
```
Please make sure the generated data put in the same folder with the source code. Or use `-dir` to specify the data path.

To run benchmarks in the paper, take a look at [`benchmark` folder](./benchmark)

## Citation

<!---
TODO: add bibtex
-->