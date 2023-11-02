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


// For one symbol
__device__ void shift_filter_uplink(
		void *dataOut,
		size_t offset,
		cufftComplex element,
		void *callerInfo,
		void *sharedPtr)
{
	struct storeInfo *storeInfo = (struct storeInfo *)callerInfo;
	int64_t carrier_offset = (offset + (storeInfo->ofdmCAnum / 2)) % storeInfo->ofdmCAnum
				- storeInfo->ofdmStart;

	if (carrier_offset < 0 || carrier_offset >= storeInfo->ofdmNum) {
		return;
	}

	size_t bsAnt_id = offset / storeInfo->ofdmCAnum;
	size_t block_start = carrier_offset * storeInfo->bsAnt;

	((cufftComplex *)dataOut)[block_start + bsAnt_id] = element;
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
	int64_t carrier_offset = (offset + (storeInfo->ofdmCAnum / 2)) % storeInfo->ofdmCAnum
				- storeInfo->ofdmStart;

	if (carrier_offset < 0 || carrier_offset >= storeInfo->ofdmNum) {
		return;
	}

	cufftComplex pilot = storeInfo->pilotSign[carrier_offset];
	element = {
		element.x * pilot.x + element.y * pilot.y,
		//element.y * pilot.x - element.x * pilot.y,
		element.x * pilot.y - element.y * pilot.x,
	};

	size_t bs_offset = offset / storeInfo->ofdmCAnum;
	size_t cg_offset = carrier_offset / storeInfo->scGroup;
	size_t ue_offset = storeInfo->ueStart + carrier_offset % storeInfo->scGroup;

	if (ue_offset >= storeInfo->ueAnt) {
		return;
	}

	size_t block_size = storeInfo->bsAnt * storeInfo->ueAnt;
	size_t block_start = cg_offset * block_size;
	size_t block_offset = bs_offset * storeInfo->ueAnt + ue_offset;

	((cufftComplex *)dataOut)[block_start + block_offset] = element;
}

__device__ cufftCallbackStoreC cufftStorePilotPtr = shift_filter_pilot;