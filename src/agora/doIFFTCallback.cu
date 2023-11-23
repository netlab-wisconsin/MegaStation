#include "doIFFTCallback.h"
#include <cstdio>

__device__ void complex_to_short(
		void *dataOut,
		size_t offset,
        cufftComplex element,
		void *callerInfo,
		void *sharedPtr)
{
	struct bothInfo *bothInfo = (struct bothInfo *)callerInfo;
    float re = element.x;
    float im = element.y;

    float re_scaled = re * (kShortFloatIFactor / bothInfo->ofdmCAnum);
    float im_scaled = im * (kShortFloatIFactor / bothInfo->ofdmCAnum);

    short re_short = __float2int_rn(re_scaled);
    short im_short = __float2int_rn(im_scaled);

    if (re_scaled > SHRT_MAX) {
        re_short = SHRT_MAX;
    } else if (re_scaled < SHRT_MIN) {
        re_short = SHRT_MIN;
    }
    if (im_scaled > SHRT_MAX) {
        im_short = SHRT_MAX;
    } else if (im_scaled < SHRT_MIN) {
        im_short = SHRT_MIN;
    }

    size_t out_offset = offset << 1; // offset * 2
	((short *)dataOut)[out_offset] = re_short;
	((short *)dataOut)[out_offset + 1] = im_short;
}

__device__ cufftCallbackStoreC cufftStoreCallbackIPtr = complex_to_short;


__device__ cufftComplex shift_expand(
		void *dataIn,
		size_t offset,
		void *callerInfo,
		void *sharedPtr)
{
    struct bothInfo *bothInfo = (struct bothInfo *)callerInfo;
    size_t block_start = (offset / bothInfo->ofdmCAnum) * bothInfo->ofdmCAnum;
	size_t carrier_offset = (offset + (bothInfo->ofdmCAnum / 2)) % bothInfo->ofdmCAnum;
	int64_t valid_offset = carrier_offset - bothInfo->ofdmStart;

	if (valid_offset < 0 || valid_offset >= bothInfo->ofdmNum) {
		return {0.f, 0.f};
	}

    return ((cufftComplex *)dataIn)[block_start + carrier_offset];
}

__device__ cufftCallbackLoadC cufftLoadCallbackIPtr = shift_expand; 
