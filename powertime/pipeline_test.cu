#include <cstdlib>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cufft.h>

#include "errors.hpp"

#define INT_PER_LINE 2
#define NFPGAS 48
#define NCHAN_COARSE 336
#define NCHAN_FINE_IN 32
#define NCHAN_FINE_OUT 27
#define NACCUMULATE 128
#define NPOL 2
#define NSAMPS 4
#define NSAMPS_SUMMED 2
#define NCHAN_SUM 16
#define NSAMP_PER_PACKET 128
#define NCHAN_PER_PACKET 7

__global__ void UnpackKernel(int2 *__restrict__ in, cufftComplex *__restrict__ out) {

    int skip = 0;

    __shared__ int2 accblock[896];

    int chan = 0;
    int time = 0;
    int line = 0;

    cufftComplex cpol;
    int polint;

    int outskip = 0;

    for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
        // NOTE: This is skipping whole words as in will be cast to int2
        // skip = iacc * NCHAN_COARSE * NSAMP_PER_PACKET + blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET;

        skip = blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NCHAN_PER_PACKET * NSAMP_PER_PACKET;

        for (int ichunk = 0; ichunk < 7; ++ichunk) {
            line = ichunk * blockDim.x + threadIdx.x;
            chan = line % 7;
            time = line / 7;
            accblock[chan * NSAMP_PER_PACKET + time] = in[skip + line];
        }

        __syncthreads();

        skip = NCHAN_COARSE * NSAMP_PER_PACKET * NACCUMULATE;

        outskip = blockIdx.x * 7 * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET;

        for (chan = 0; chan < NCHAN_PER_PACKET; ++chan) {
            polint = accblock[chan * NSAMP_PER_PACKET + threadIdx.x].y;
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));
            out[outskip + threadIdx.x] = cpol;

            polint = accblock[chan * NSAMP_PER_PACKET + threadIdx.x].x;
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));

            out[skip + outskip + threadIdx.x] = cpol;

            outskip += NSAMP_PER_PACKET * NACCUMULATE;
        }
    }
}

__global__ void DetectScrunchKernel(
                                            cuComplex* __restrict__ in, // PFTF <-- FFT output order
                                            float* __restrict__ out  // TF <-- Filterbank order
                                            )
{

  /**
   * This block is going to do 2 timesamples for all coarse channels.
   * The fine channels are dealt with by the lanes
   */

  // gridDim.x should be Nacc * 128 / (32 * nsamps_to_add) == 256

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE]; // 9072 elements

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;

  int pol_step = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE;

  int nwarps_per_block = blockDim.x/warpSize;

  int offset_into_coarse_chan = blockIdx.x * NCHAN_FINE_IN * NSAMPS_SUMMED;

  //Drop first 3 fine channels and last two fine channels
  if ((lane_idx > 2) & (lane_idx < 30))
    {
      // This warp
      // first sample in inner dimension = (32 * 2 * blockIdx.x)

      // This warp will loop over coarse channels in steps of NWARPS per block
      // coarse_chan_idx (0,335)

      for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += nwarps_per_block)
        {
          float real = 0.0f;
          float imag = 0.0f;
          int coarse_chan_jump = NACCUMULATE * NCHAN_FINE_IN * NSAMPS * coarse_chan_idx + offset_into_coarse_chan + lane_idx;
          for (int pol_idx=0; pol_idx<NPOL; ++pol_idx)
            {
              int offset = pol_step * pol_idx + coarse_chan_jump;
              for (int sample_idx=0; sample_idx<NSAMPS_SUMMED; ++sample_idx)
                {
                  //Get first channel
                  int read_idx = offset + sample_idx * NCHAN_FINE_IN;
                  cuComplex val = in[read_idx];
                  real += val.x * val.x;
                  imag += val.y * val.y;
                }
              // 3 is the leading dead lane count
              // sketchy
              freq_sum_buffer[coarse_chan_idx*NCHAN_FINE_OUT + lane_idx - 3] = real + imag;
            }
        }
    }

  __syncthreads();

  for (int start_chan=warp_idx*warpSize; start_chan < (NCHAN_FINE_OUT * NCHAN_COARSE - NCHAN_SUM); start_chan+=blockDim.x) // blockDim.x is multiple of 32
    {
      //float sum = freq_sum_buffer[start_chan];
      // 4 because we are summing 16 channels in a warp reduce
      for (int ii=0; ii<4; ++ii)
        {
          if (lane_idx < warpSize-(1<<ii)-1)
            {
              freq_sum_buffer[start_chan+lane_idx] += freq_sum_buffer[start_chan + lane_idx + (1<<ii)];
            }
        }
      if (lane_idx & 0x0f)
        {
          int out_chan = (start_chan + lane_idx)/16;
          out[NCHAN_FINE_OUT * NCHAN_COARSE / NCHAN_SUM * blockIdx.x + out_chan] = freq_sum_buffer[start_chan+lane_idx];
        }
    }
  return;
}


int main(int argc, char *argv[])
{
//     unsigned short polai;
//     unsigned short polaq;
//
//     unsigned short polbi;
//     unsigned short polbq;

    size_t toread = 8 * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NFPGAS * NACCUMULATE;
    unsigned char *codifarray = new unsigned char[toread];

    for (int ifpga = 0; ifpga < 48; ++ifpga) {
        for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {

            for (int isamp = 0; isamp < 128; ++isamp) {

                for (int ichan = 0; ichan < 7; ++ichan) {

                    // polai = ((ifpga << 10) | (isamp << 2) | 0x0);
                    // polaq = ((ifpga << 10) | (isamp << 2) | 0x2);
                    // polbi = ((ifpga << 10) | (isamp << 2) | 0x1);
                    // polbq = ((ifpga << 10) | (isamp << 2) | 0x3);

                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 0] = 0;
                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 1] = 0;
                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 2] = 0;
                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 3] = 0;
                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 4] = 0;
                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 5] = 0;
                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 6] = 0;
                    codifarray[(ifpga * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET * NCHAN_PER_PACKET + isamp * NCHAN_PER_PACKET + ichan) * 8 + 7] = 0;

                }
            }
        }
    }

    unsigned char *devdata;
    cudaCheckError(cudaMalloc((void**)&devdata, toread * sizeof(unsigned char)));
    cudaCheckError(cudaMemcpy(devdata, codifarray, toread * sizeof(unsigned char), cudaMemcpyHostToDevice));

    cufftComplex *unpacked;
    cudaCheckError(cudaMalloc((void**)&unpacked, toread / 8 * sizeof(cufftComplex)));

    int sizes[] = {32};

    cufftHandle fftplan;
    cufftCheckError(cufftPlanMany(&fftplan, 1, sizes, NULL, 1, sizes[0], NULL, 1, sizes[0], CUFFT_C2C, 336 * NACCUMULATE * 4));

    float *detected;
    cudaCheckError(cudaMalloc((void**)&detected, toread / 8 * sizeof(float)));

    UnpackKernel<<<48, 128, 0>>>(reinterpret_cast<int2*>(devdata), unpacked);
    cufftCheckError(cufftExecC2C(fftplan, unpacked, unpacked, CUFFT_FORWARD));
    DetectScrunchKernel<<<2 * NACCUMULATE, 1024, 0>>>(unpacked, detected);

    cudaCheckError(cudaDeviceSynchronize());

    return 0;

}
