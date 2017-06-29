#include <stdio.h>
#include "thrust/device_vector.h"
#include "cuComplex.h"

#include "cufft.h"

#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48

#define NCHAN_COARSE 336
#define NCHAN_FINE_IN 32
#define NCHAN_FINE_OUT 27
#define NACCUMULATE 256
#define NPOL 2
#define NSAMPS 4
#define NCHAN_SUM 16


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

__global__ void unpack_original_tex(cudaTextureObject_t texObj, cufftComplex * __restrict__ out, unsigned int acc)
{

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * 128;
    int chanidx = threadIdx.x + blockIdx.y * 7;
    int skip;
    int2 word;

    for (int ac = 0; ac < acc; ac++) {
        skip = 336 * 128 * 2 * ac;
        for (int sample = 0; sample < YSIZE; sample++) {
            word = tex2D<int2>(texObj, xidx, yidx + ac * 48 * 128 + sample);
            out[skip + chanidx * YSIZE * 2 + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
        }
    }
}

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

__global__ void powertimefreq_new_hardcoded(
  cuComplex* __restrict__ in,
  float* __restrict__ out)
{

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE];

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;

  if (lane_idx >= NCHAN_FINE_OUT)
    return;

  int offset = blockIdx.x * NCHAN_COARSE * NPOL * NSAMPS * NCHAN_FINE_IN;
  int out_offset = blockIdx.x * NCHAN_COARSE * NCHAN_FINE_OUT / NCHAN_SUM;

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
      int output_idx = coarse_chan_idx * NCHAN_FINE_OUT + lane_idx;

      freq_sum_buffer[output_idx] = real+imag; //scaling goes here
      __syncthreads();

      for (int start_chan=threadIdx.x; start_chan<NCHAN_FINE_OUT*NCHAN_COARSE; start_chan*=blockDim.x)
      {
        if ((start_chan+NCHAN_SUM) > NCHAN_FINE_OUT*NCHAN_COARSE)
          return;
        float sum = freq_sum_buffer[start_chan];
        for (int ii=0; ii<4; ++ii)
        {
          sum += freq_sum_buffer[start_chan + (1<<ii)];
          __syncthreads();
        }
        out[out_offset+start_chan/NCHAN_SUM];
      }
    }
  return;
}

int main()
{

    unsigned char *rawbuffer = new unsigned char[7168 * 48 * NACCUMULATE];
    cudaArray *rawarray;
    cudaChannelFormatDesc cdesc;
    cdesc = cudaCreateChannelDesc<int2>();
    cudaMallocArray(&rawarray, &cdesc, 7,  48 * 128 * NACCUMULATE);

    cudaResourceDesc rdesc;
    memset(&rdesc, 0, sizeof(cudaResourceDesc));
    rdesc.resType = cudaResourceTypeArray;
    rdesc.res.array.array = rawarray;

    cudaTextureDesc tdesc;
    memset(&tdesc, 0, sizeof(cudaTextureDesc));
    tdesc.addressMode[0] = cudaAddressModeClamp;
    tdesc.filterMode = cudaFilterModePoint;
    tdesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &rdesc, &tdesc, NULL);

    thrust::device_vector<cuComplex> input(336*32*4*2*NACCUMULATE);
    thrust::device_vector<float> output(336*27*NACCUMULATE);

    cudaMemcpyToArray(rawarray, 0, 0, rawbuffer, 7168 * 48 * NACCUMULATE * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 rearrange_b(1,48,1);
    dim3 rearrange_t(7,1,1);

    for (int ii=0; ii<1; ++ii) {

        unpack_original_tex<<<rearrange_b, rearrange_t, 0>>>(texObj, thrust::raw_pointer_cast(input.data()), NACCUMULATE);
        // unpack_new<<<>>>()
        powertime_original<<<48, 27, 0>>>(thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()), 864, NSAMPS, NACCUMULATE);
        powertime_new_hardcoded<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()));
        powertime_new<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()),336,32,27,2,4);
        gpuErrchk(cudaDeviceSynchronize());
        powertimefreq_new_hardcoded<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()));
    }

  //gpuErrchk(cudaDeviceSynchronize());
}
