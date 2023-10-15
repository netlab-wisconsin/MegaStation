#include "doFFTCallback.h"
#include <cstdio>

__device__ cufftComplex short_to_complex(
		void *dataIn,
		size_t offset,
		void *callerInfo,
		void *sharedPtr)
{
	size_t in_offset = offset << 1; // offset * 2
	short re_short = ((short *)dataIn)[in_offset];
	short im_short = ((short *)dataIn)[in_offset + 1];

	float re = float(re_short) / kShortFloatFactor;
	float im = float(im_short) / kShortFloatFactor;

	return {re, im};
}

__device__ cufftCallbackLoadC cufftLoadCallbackPtr = short_to_complex;


__device__ void shift_filter_uplink(
		void *dataOut,
		size_t offset,
		cufftComplex element,
		void *callerInfo,
		void *sharedPtr)
{
	struct storeInfo *storeInfo = (struct storeInfo *)callerInfo;
	int64_t block_offset = (offset + (storeInfo->ofdmCAnum / 2)) % storeInfo->ofdmCAnum
				- storeInfo->ofdmStart;

	if (block_offset < 0 || block_offset >= storeInfo->ofdmNum) {
		return;
	}

	size_t block_start = (offset / storeInfo->ofdmCAnum) * storeInfo->ofdmNum;

	((cufftComplex *)dataOut)[block_start + block_offset] = element;
}

__device__ cufftCallbackStoreC cufftStoreUplinkPtr = shift_filter_uplink; 


__device__ void shift_filter_pilot(
		void *dataOut,
		size_t offset,
		cufftComplex element,
		void *callerInfo,
		void *sharedPtr)
{
	struct storeInfo *storeInfo = (struct storeInfo *)callerInfo;
	int64_t block_offset = (offset + (storeInfo->ofdmCAnum / 2)) % storeInfo->ofdmCAnum
				- storeInfo->ofdmStart;

	if (block_offset < 0 || block_offset >= storeInfo->ofdmNum) {
		return;
	}

	/*if (offset == 0) {
		std::printf("First Input Symbol is %f, %f\n", element.x, element.y);
	}
	if (offset == 0) {
		std::printf("First Output Symbol is %f, %f\n", element.x, element.y);
	}*/

	cufftComplex pilot = storeInfo->pilotSign[block_offset];
	element = {
		element.x * pilot.x + element.y * pilot.y,
		element.y * pilot.x - element.x * pilot.y
	};

	size_t block_start = (offset / storeInfo->ofdmCAnum) * storeInfo->ofdmNum;
	/*if (block_start + block_offset == 11 * storeInfo->ofdmNum + 721) {
		std::printf("GPU Output Symbol is %f, %f\n", element.x, element.y);
	}*/

	((cufftComplex *)dataOut)[block_start + block_offset] = element;
}

__device__ cufftCallbackStoreC cufftStorePilotPtr = shift_filter_pilot;