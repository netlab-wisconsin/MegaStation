#ifndef CUFFT_CALLBACK_H_
#define CUFFT_CALLBACK_H_

#include <cufftXt.h>
#include <stdint.h>
#include <stddef.h>

extern __device__ cufftCallbackLoadC cufftLoadCallbackPtr;
extern __device__ cufftCallbackStoreC cufftStoreUplinkPtr;
extern __device__ cufftCallbackStoreC cufftStorePilotPtr;

__device__ static const float kShortFloatFactor = 32768.0f;

struct storeInfo {
	size_t ofdmStart;
	size_t ofdmNum;
	size_t ofdmCAnum;
	cufftComplex *pilotSign;
};

#endif // CUFFT_CALLBACK_H_
