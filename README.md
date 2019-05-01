# voxRefactor

## Introduction
A CUDA accelerated C++ tool to dilate building voxels.

Input voxels to this tool should be of the format `.binvox`. This is a format used by [**binvox**](http://www.patrickmin.com/binvox/), which is a tool to read 3D models, rasterizes it into binary 3D voxel grid, and writes to voxel files.

Outputs of this tool are `.png` images. Shapes of the images depend on shape of the input voxels. For example, if a voxel grid is 256x256x256, then the output image will be 256x256 pixels.


## Usage

Build
```
git clone https://github.com/yuqli/vox2dem.git
cd build
cmake ../src
make
```
This will build target `refactor` from `main.cu`.  
Dependencies: CUDA

To use `refactor`:

`./refactor <cuda device> <binvox filename> <output root folder> <crop and scale >`

Options:
- `<cuda device>`: int, indicating which GPU to use, starting from 0.
- `<binvox filename>`: string, the full path to `.binvox` file.
- `<output root folder>`: string, the full path to the folder that stores output `.png` images.
- `<crop and scale>`: binary int. `0` means no crop or scale, `1` means crop and scale.   

Notes on output images:
1. Some voxels could be empty, i.e. none of the grids are occupied. In this case, the program will write paths to the empty voxels to `empty_log.txt` located in the `<output root folder>`.  



## Example
After building, run the following command from the `build` directory:

`./refactor 0 ../sample/test.binvox ../sample/result`


## Contact
For any questions please contact Yuqiong Li at yl5090 at nyu dot edu.
