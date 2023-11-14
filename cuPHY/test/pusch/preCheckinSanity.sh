#!/bin/bash
# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Script to batch execution of the PUSCH-multipipe example (cuphy_ex_pudsch_rx_multi_pipe) with test vectors of of the format: TV_cuphy_*.h5 
# See Usage for example run commands 

#########################
# Usage #
Usage() {
echo "Usage: $0 [-h] [-r] {<s|p> <build_dir> <TV_dir> <out_dir>}" >&2
echo
echo "   -h, --help           show this help text"
echo "   -r, --run            run executable from <build_dir> TVs from <TV_dir>, s|p selects serial|paralle TV execution, store results and logs in <out_dir>"
echo
echo "Suppose cuPHY dir contains the TVs then:"
echo "Example usage from build dir: ../test/pusch/preCheckinSanity.sh -r p \$PWD ../. results"
echo "Example usage from cuPHY dir: test/pusch/preCheckinSanity.sh -r p \$PWD/build \$PWD results"
exit 1
}

#########################
# Tests with failures #
DisplayResults() {
#export RESDIR="resultsSdkTvs201912210738"

# array to collect TC names of all failing TCs 
declare -a FAIL_TCNAME_ARR

# Report all tests with failures (note that some of the failures may be intentional)
echo "---------------------------------------------------------------"
echo "Build directory : $BUILDDIR"
echo "TV directory    : $TVDIR"
echo "Result directory: $RESDIR"

echo "Test summary    :" 
for TVFILE in "$RESDIR"/logs/*.log; do 
	if grep -q 'ERROR\|error\|EXCEPTION\|failed' $TVFILE; then
	    # Get file name without the path
	    TVFILENAME=$(basename -- "$TVFILE")
	    # Remove extension from file name
	    TCNAME="${TVFILENAME%.*}"
	    FAIL_TCNAME_ARR+=("$TCNAME")
	    #echo "$TCNAME"
	fi
done

if [ ${#FAIL_TCNAME_ARR[@]} -eq 0 ]; then
        echo "No failures detected" 
else
        echo "Tests with failures" 
        printf '%s\n' "${FAIL_TCNAME_ARR[@]}"
        #for each in "${FAIL_TCNAME_ARR[@]}"
        #do
	#    echo "$each"
        #done
fi
}

#########################
# Serial execution #
Execute() {
export RESDIR=$(date +${RESDIRPREFIX}'%G%m%d%H%M')
#export RESDIR="results201912090933"
 
echo "Build directory : $BUILDDIR"
echo "TV directory    : $TVDIR"
echo "Result directory: $RESDIR"

rm -fr $RESDIR
mkdir $RESDIR
mkdir $RESDIR/logs

#if [ 1 -eq 0 ]; then
# Find all files with specified TV prefix and extension and run them
find $TVDIR -maxdepth 1 -type f -name 'TV_cuphy_*.h5' -printf "%P\n" |
     while IFS= read -r -d $'\n' TVFILE; do 
	TCNAME="${TVFILE%.*}"
        echo "--------------------------------Running Testcase: $TCNAME-------------------------------"

	$BUILDDIR/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe -i $TVDIR/$TVFILE -o $RESDIR/gpu_out_$TCNAME.h5 > >(tee -a $RESDIR/logs/$TCNAME.log) 2> >(tee -a $RESDIR/logs/$TCNAME.log >&2)
     done
#fi

DisplayResults
}

#########################
# Parallel execution #
ExecuteParallel() {
export RESDIR=$(date +${RESDIRPREFIX}'%G%m%d%H%M')
#export RESDIR="resultsSdkTvs201912210738"
 
echo "Build directory : $BUILDDIR"
echo "TV directory    : $TVDIR"
echo "Result directory: $RESDIR"

rm -fr $RESDIR
mkdir $RESDIR
mkdir $RESDIR/logs

# array to collect pids of all background processes running test cases so that they may be waited upon
PID_ARR=""
# background processes run asynchronously, use a named pipe to collect outputs and display one they are all done
mkfifo OUTPIPE
RESULT=0

#if [ 1 -eq 0 ]; then
# Find all files with specified TV prefix and extension and run them
find $TVDIR -maxdepth 1 -type f -name 'TV_cuphy_*.h5' -printf "%P\n" |
     while IFS= read -r -d $'\n' TVFILE; do 
	(
	# get TV name only
	TCNAME="${TVFILE%.*}"
        echo "Running Testcase: $TCNAME"

	# Run test

	$BUILDDIR/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe -i $TVDIR/$TVFILE -o $RESDIR/gpu_out_$TCNAME.h5 > >(tee -a $RESDIR/logs/$TCNAME.log) 2> >(tee -a $RESDIR/logs/$TCNAME.log >&2) > OUTPIPE
        # $BUILDDIR/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe -i $TVDIR/$TVFILE -o $RESDIR/gpu_out_$TCNAME.h5 | tee $RESDIR/logs/$TCNAME.log > OUTPIPE
	) &
	PID_ARR="$PID_ARR $!"
     done

# wait for all background processes to finish/exit before moving forward 
for PID in $PID_ARR; do
    wait $PID || let "RESULT=1"
done

# display output from all background processes
cat OUTPIPE
# remove the named pipe
rm OUTPIPE

# check for error during test execution
if [ "$RESULT" == "1" ]; then
    echo "Failure during execution of one or more tests"
    exit 1
fi
#fi

DisplayResults
}


################################
# Check Options #
#echo "$0"
#echo "$1"
#echo "$2" 
#echo "$3"
#echo "$4"

while :
do
    case "$1" in
      -r | --run)
        shift 1
	if [ $# -eq 4 ]; then
           export BUILDDIR=$2 TVDIR=$3 RESDIRPREFIX=$4
           if [ "$1" == "s" ]; then
	      Execute 
           elif [ "$1" == "p" ]; then 
              ExecuteParallel 
	      #DisplayResults
           else
              echo "Error: Unknown option: $1" >&2
           fi
        else
           echo "Not enough input arguments" >&2
           Usage
        fi
        ;;
      -h | --help)
        Usage
        exit 0
	;;
      --) # End of all options
        shift
        break
        ;;
      -*)
        echo "Error: Unknown option: $1" >&2
        Usage
        exit 1 
        ;;
      *)  # No more options
        break
        ;;
    esac
done

