#ifndef CUIFFT_CALLBACK_H_
#define CUIFFT_CALLBACK_H_

#include <cufftXt.h>
#include <stdint.h>
#include <stddef.h>

extern __device__ cufftCallbackLoadC cufftLoadCallbackIPtr;
extern __device__ cufftCallbackStoreC cufftStoreCallbackIPtr;

__device__ static const float kShortFloatIFactor = 32768.0f;

struct bothInfo {
	size_t ofdmStart;
	size_t ofdmNum;
	size_t ofdmCAnum;
	size_t bsAnt;
};

#endif // CUIFFT_CALLBACK_H_
