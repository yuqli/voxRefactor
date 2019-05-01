// Date: 20190430
// Author: yuqiong Li
// read binvox and output digital elevation model (DEM) based on the voxel
// the DEM can also be interprted as a heightmap in which the z-axis is the heigh
//
// This example program reads a .binvox file and writes
// an ASCII version of the same file called "voxels.txt"
//
// 0 = empty voxel
// 1 = filled voxel
// A newline is output after every "dim" voxels (depth = height = width = dim)
//
// Note that this ASCII version is not supported by "viewvox" and "thinvox"
//
// The x-axis is the most significant axis, then the z-axis, then the y-axis.
// i.e. in binvox, read starts from y and then proceeds to z and finally
// Reference: http://www.patrickmin.com/binvox/binvox.html
//

#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <cuda.h>
using namespace std;


#define index2(x, y, W) (x)*(W) + (y)  // used to index 2D objects in C style arrays; x is row and y is column
#define index3(z, x, y, H, W)  ((z)*(H)*(W)) + (x)*(W) + (y)  // used to index 3D objects in C style arrays
#define binvoxIndex3(z, x, y, H, W)  ((x)*(H)*(W)) + (z)*(W) + (y)  // used to index 3D objects in binvox voxels. needed as the coordinates are different


using namespace std;

typedef unsigned char byte;

static int version;
static int D = 256;  // depth, also Z dim
const int W = 256;  // width, also X dim
const int H = 256;  // height, also Y dim
static int size;
static byte *voxels = 0;   // global variable to store voxels

const int smallH = 64;  // small size cube
static int smallSize;
static byte *smallV = 0;   // global variable to store small voxels
static float tx, ty, tz;
static float scale;
static float thres=0.5;   // threshold: what's the proportion of data in the small cube that's filled should we set it filled?

__global__ void fillKernel(byte * voxels, byte * smallV, int smallH, float thres_sum);


int read_binvox(const string & filespec)
{

    ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);
    if ((*input).fail()){
        cout << "Error: file does not exist at " << filespec << "!" << endl;
    }
    //
    // read header
    //
    string line;
    *input >> line;  // #binvox
    if (line.compare("#binvox") != 0) {
        cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
        delete input;
        return 0;
    }
    *input >> version;
    cout << "reading binvox version " << version << endl;

    int depth, height, width;  // values from file, compare if the same
    depth = -1;
    int done = 0;
    while(input->good() && !done) {
        *input >> line;
        if (line.compare("data") == 0) done = 1;
        else if (line.compare("dim") == 0) {
            *input >> depth >> height >> width;
        }
        else if (line.compare("translate") == 0) {
            *input >> tx >> ty >> tz;
        }
        else if (line.compare("scale") == 0) {
            *input >> scale;
        }
        else {
            cout << "  unrecognized keyword [" << line << "], skipping" << endl;
            char c;
            do {  // skip until end of line
                c = input->get();
            } while(input->good() && (c != '\n'));

        }
    }
    if (!done) {
        cout << "  error reading header" << endl;
        return 0;
    }
    if (D == -1) {
        cout << "  missing dimensions in header" << endl;
        return 0;
    }

    size = W * H * D;
    voxels = new byte[size];  // danger! not initialized!
    if (!voxels) {
        cout << "  error allocating memory" << endl;
        return 0;
    }

    //
    // read voxel data
    //
    byte value;
    byte count;
    int index = 0;
    int end_index = 0;
    int nr_voxels = 0;

    input->unsetf(ios::skipws);  // need to read every byte now (!)
    *input >> value;  // read the linefeed char

    while((end_index < size) && input->good()) {
        *input >> value >> count;

        if (input->good()) {
            end_index = index + count;
            if (end_index > size) return 0;
            for(int i=index; i < end_index; i++) voxels[i] = value;

            if (value) nr_voxels += count;
            index = end_index;
        }  // if file still ok

    }  // while

    input->close();
    cout << "  read " << nr_voxels << " voxels" << endl;

    return 1;

}


int get_index(int x, int y, int z) {
    // used to get correct index from the voxel data
    // http://www.patrickmin.com/binvox/binvox.html
    int index = x * H*W + z * W + y;
    return index;
}


vector<string> split(const char *phrase, string delimiter){
    // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
    // used to parse file name
    vector<string> list;
    string s = string(phrase);
    size_t pos = 0;
    string token;
    while ((pos = s.find(delimiter)) != string::npos) {
        token = s.substr(0, pos);
        list.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    list.push_back(s);
    return list;
}



int main(int argc, char **argv)
{
    if (argc != 4) {
        cout << "Usage: v2d <cuda device> <binvox filename> <output root folder>" << endl << endl;
        exit(1);
    }

    // check if exceeding number of GPUs
    int num_devices, device;
    device = *argv[1] - '0';
    cudaGetDeviceCount(&num_devices);

    if (device >= num_devices) {
        cout << "Error specifying GPU " << argv[1] << " that does not exist" << endl << endl;
        exit(1);
    }
    else
        cudaSetDevice(device);

    if (!read_binvox(argv[2])) {
        cout << "Error reading [" << argv[2] << "]" << endl << endl;
        exit(1);
    }

    vector<string> all_inputs = split(argv[2], "/");
    string bname = all_inputs.back();  // last element is file name "input.binvox"
    string bid = split(bname.c_str(), ".").front();    // first element is building id
    cout << "Building file name " << bname << endl;
    cout << "Building name " << bid << endl;


    smallSize = smallH * smallH * smallH;
    // https://stackoverflow.com/questions/2204176/how-to-initialise-memory-with-new-operator-in-c
    smallV = new byte[smallSize]();  // special syntax to initialize things to zero

    float thres_sum = thres * 64;   // hard code here as we know small cube size already!!


    //------------------------------------- CUDA code starts ----------------------------------------//
    byte * d_v, * d_sv;  // voxels and small voxels on device

    cudaMalloc(& d_v, size * sizeof(byte));
    cudaMalloc(& d_sv, smallSize * sizeof(byte));

    cudaMemcpy(d_v, voxels, size * sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sv, smallV, smallSize * sizeof(byte), cudaMemcpyHostToDevice);

    // assigning CUDA blocks and dimensions
    // a small voxel is 64**3, and needs 64**3 threads in total
    // every thread can only take care of a small cube because it needs to sum over. a small cube is 4**3 = 64
    // so the speed up is at least 64 times than a single thread, but GPUs are slower than CPU...
    dim3 blocksPerGrid(4, 4, 16);   //
    dim3 threadsPerBlock(16, 16, 4); // 1024 threads per block

    // start kernel
    fillKernel<<<blocksPerGrid, threadsPerBlock>>>(d_v, d_sv, smallH, thres_sum);

    // check errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
        exit(-1);
    }

    // copy results from device to host
    cudaMemcpy(smallV, d_sv, smallSize * sizeof(byte), cudaMemcpyDeviceToHost);

    //------------------------------------- CUDA code finish ----------------------------------------//
    string out_dir = argv[3];
    // https://stackoverflow.com/questions/5621944/how-to-find-out-if-a-folder-exists-and-how-to-create-a-folder
    if (!boost::filesystem::exists(out_dir))
        boost::filesystem::create_directories(out_dir);
    string outname = out_dir+ "/" + bid + ".bin";
    FILE* file = fopen(outname.c_str(), "wb" );
    fwrite( smallV, sizeof(byte), smallSize, file );

    delete smallV;

    return 0;
}


// GPU version of finding heights
__global__ void fillKernel(byte * voxels, byte * smallV, int smallH, float thres_sum){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // row
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // column
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth

    int local_sum = 0;   // sum of small cube valus
    unsigned int sx = x * 4;   //begin of small cube x values, inclusive
    unsigned int ex = sx + 3;   // end of small cube x values, inclusive
    unsigned int sy = y * 4;   //begin of small cube y values, inclusive
    unsigned int ey = sy + 3;   // end of small cube y values, inclusive
    unsigned int sz = z * 4;   //begin of small cube z values, inclusive
    unsigned int ez = sz + 3;   // end of small cube z values, inclusive

    // every thread is in charge of looping through all locations in the small slice of the large voxel
    // and summing over all values
    for (int i = sx; i <= ex; i++) {
        for (int j = sy; j <= ey; j++) {
            for (int k = sz; k <= ez; k++)
                local_sum += voxels[binvoxIndex3(k, i, j, H, W)];
        }
    }

    // fill the smaller voxel
    if (local_sum >= thres_sum){
        smallV[index3(z, x, y, smallH, smallH)] = 1;
    }
    else{
        smallV[index3(z, x, y, smallH, smallH)] = 0;
    }
}
