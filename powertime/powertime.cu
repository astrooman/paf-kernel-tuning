#include <stdio.h>
#include "thrust/device_vector.h"
#include "cuComplex.h"

#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48

#define NCHAN_COARSE 336
#define NCHAN_FINE_IN 32
#define NCHAN_FINE_OUT 27
#define NACCUMULATE 256
#define NPOL 2
#define NSAMPS 4


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

//Why isn't this a #define
__device__ float fftfactor = 1.0/32.0 * 1.0/32.0;

__global__ void powertime_original(cuComplex* __restrict__ in,
				   float* __restrict__ out,
				   unsigned int jump,
				   unsigned int factort,
				   unsigned int acc) {

  // 48 blocks and 27 threads
  //  336 1MHz channels * 32 finer channels * 4 time samples * 2 polarisations * 8 accumulates
  int idx1, idx2;
  int outidx;
  int skip1, skip2;
  float power1, power2;
  float avgfactor= 1.0f / factort;

  for (int ac = 0; ac < acc; ac++) {
    skip1 = ac * 336 * 128 * 2;
    skip2 = ac * 336 * 27;
    for (int ii = 0; ii < 7; ii++) {
      outidx = skip2 + 7 * 27 * blockIdx.x + ii * 27 + threadIdx.x;
      out[outidx] = (float)0.0;
      out[outidx + jump] = (float)0.0;
      out[outidx + 2 * jump] = (float)0.0;
      out[outidx + 3 * jump] = (float)0.0;
      idx1 = skip1 + 256 * (blockIdx.x * 7 + ii);
      for (int jj = 0; jj < factort; jj++) {
	idx2 = threadIdx.x + jj * 32;
	power1 = (in[idx1 + idx2].x * in[idx1 + idx2].x +
		  in[idx1 + idx2].y * in[idx1 + idx2].y) * fftfactor;
	power2 = (in[idx1 + 128 + idx2].x * in[idx1 + 128 + idx2].x +
		  in[idx1 + 128 + idx2].y * in[idx1 + 128 + idx2].y) * fftfactor;
	out[outidx] += (power1 + power2) * avgfactor;
	out[outidx + jump] += (power1 - power2) * avgfactor;
	out[outidx + 2 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].x
						    + in[idx1 + idx2].y * in[idx1 + 128 + idx2].y)) * avgfactor;
	out[outidx + 3 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].y
						    - in[idx1 + idx2].y * in[idx1 + 128 + idx2].x)) * avgfactor;
      }
    }
  }
}

__global__ void powertime_new(
  cuComplex* __restrict__ in,
  float* __restrict__ out,
  unsigned int nchan_coarse,
  unsigned int nchan_fine_in,
  unsigned int nchan_fine_out,
  unsigned int npol,
  unsigned int nsamps)
{
  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;

  //Need to know which chans are being dropped
  if (lane_idx >= nchan_fine_out)
    return;

  int offset = blockIdx.x * nchan_coarse * npol * nsamps * nchan_fine_in;
  int out_offset = blockIdx.x * nchan_coarse * nchan_fine_out;

  for (int coarse_chan_idx = warp_idx; coarse_chan_idx < nchan_coarse; coarse_chan_idx += warpSize)
    {

      float real = 0.0f;
      float imag = 0.0f;
      int coarse_chan_offset = offset + coarse_chan_idx * npol * nsamps * nchan_fine_in;

      for (int pol=0; pol<npol; ++pol)
      {
        int pol_offset = coarse_chan_offset + pol * nsamps * nchan_fine_in;
        for (int samp=0; samp<nsamps; ++samp)
        {
          int samp_offset = pol_offset + samp * nchan_fine_in;
          cuComplex val = in[samp_offset + lane_idx];
          real += val.x * val.x;
          imag += val.y * val.y;
        }
      }
      int output_idx = out_offset + coarse_chan_idx * nchan_fine_out + lane_idx;
      out[output_idx] = real+imag; //scaling goes here
    }
  return;
}

__global__ void powertime_new_hardcoded(
  cuComplex* __restrict__ in,
  float* __restrict__ out)
{
  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;

  if (lane_idx >= NCHAN_FINE_OUT)
    return;
  
  int offset = blockIdx.x * NCHAN_COARSE * NPOL * NSAMPS * NCHAN_FINE_IN;
  int out_offset = blockIdx.x * NCHAN_COARSE * NCHAN_FINE_OUT;

  for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += warpSize)
    {
      float real = 0.0f;
      float imag = 0.0f;
      int coarse_chan_offset = offset + coarse_chan_idx * NPOL * NSAMPS * NCHAN_FINE_IN;

      for (int pol=0; pol<NPOL; ++pol)
      {
        int pol_offset = coarse_chan_offset + pol * NSAMPS * NCHAN_FINE_IN;
        for (int samp=0; samp<NSAMPS; ++samp)
        {
          int samp_offset = pol_offset + samp * NCHAN_FINE_IN;
          cuComplex val = in[samp_offset + lane_idx];
          real += val.x * val.x;
          imag += val.y * val.y;
        }
      }
      int output_idx = out_offset + coarse_chan_idx * NCHAN_FINE_OUT + lane_idx;
      out[output_idx] = real+imag; //scaling goes here
    }
  return;
}

int main()
{
  thrust::device_vector<cuComplex> input(336*32*4*2*NACCUMULATE);
  thrust::device_vector<float> output(336*27*NACCUMULATE);
  for (int ii=0; ii<100; ++ii)
    {
      powertime_original<<<48, 27, 0>>>(thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()), 864, NSAMPS, NACCUMULATE);
      powertime_new_hardcoded<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()));
      powertime_new<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()),336,32,27,2,4);
      gpuErrchk(cudaDeviceSynchronize());
    }
  //gpuErrchk(cudaDeviceSynchronize());
}

