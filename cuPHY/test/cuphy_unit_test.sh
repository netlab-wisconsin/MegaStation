#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

sanitizer_tool_options=("N/A" "memcheck" "racecheck" "synccheck" "initcheck")

LOGFILE="run.log"
if [ -z "$ENABLE_STREAMS" ]; then
   ENABLE_STREAMS=1
fi

EXITCODE=0

function usage {
   echo './cuphy_unit_test.sh <Base_TV_path> <GPU> <PUSCH_CB_ERROR_CHECK> <RUN_COMPONENT_TESTS> <RUN_COMPUTE_SANITIZER> <PYTHON_BIN>'
   echo '   Parameters            Default Value    Description'
   echo '   Base_TV_path          ./               Path to test vectors'
   echo '   GPU:                   0               CUDA device to execute on'
   echo '   PUSCH_CB_ERROR_CHECK   1 (enabled)     flag to enable (1) or disable (0) PUSCH CB Error check'
   echo '   RUN_COMPONENT_TESTS    0 (disabled)    flag to enable (1) or disable (0) Component level tests'
   echo '   RUN_COMPUTE_SANITIZER  1 (memcheck)    bitmask for which compute sanitizers to run'
   echo '                                            memcheck = 1'
   echo '                                           racecheck = 2'
   echo '                                           synccheck = 4'
   echo '                                           initcheck = 8'
   echo '                                           e.g. 15 (1+2+4+8) enables all the sanitizers'
   echo '                                           numbers larger than 15 are invalid'
   echo '   PYTHON_BIN             python3         python call to execute scripts with'
   exit
}

function checkmem {
    # Check exit code of process that was the input to tee
    pid_errval=${PIPESTATUS[0]}
    if [ $pid_errval -ne 0 ]; then
       echo "Process exited with non-zero exit code $pid_errval"
       return 1
    fi

    # When sanitizer is run with  --error-exitcode 1  so it returns 1 on an error, this part of the code may not be reached.
    # FIXME some of the regex and other code is unreachable and needs to be checked
    if [ $sanitizer_idx -ne 0 ]; then
       if [ $sanitizer_idx -eq 2 ]; then # racecheck has differnet log format
           #========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 2 warnings)
           errval=$(grep 'RACECHECK SUMMARY' $1 | sed -e 's/========= RACECHECK SUMMARY: \(.*\)hazards\(.*\)/\1/')
       else
           errval=$(grep 'ERROR SUMMARY' $1 | sed -e 's/========= ERROR SUMMARY: \(.*\)error\(.*\)/\1/')
       fi
       if [ -z "$errval" ]; then
           echo "Exit on error found"
           return 1
       fi
       if [ $errval -ne 0 ]; then
           echo "${sanitizer_tool_options[$sanitizer_idx]} error count="$errval
           return $errval
       else
           echo ""
       fi
    fi
}

function checkPuschResults {
    # Check exit code of process that was the input to tee
    pid_errval=${PIPESTATUS[0]}
    if [ $pid_errval -ne 0 ]; then
       echo "Process exited with non-zero exit code $pid_errval"
       return 1
    fi
    # Check for non-zero BLER for all TBs in all cells and for UCI too
    errCBs_string=$(grep -Po 'Error CBs\s*\K\d+' $1)
    #errCBs_string=$(grep -Po '^.*TbIdx.*Error CBs\s*\K\d+' $1) # Use if you don't want to check UCI BLER too.
    readarray -t errCBs_array <<< "$errCBs_string"
    for errCBs in "${errCBs_array[@]}"; do
        if [ "$errCBs" -ne "0" ]; then
            echo "Non-zero BLER detected for at least one TB in a cell or UCI - Error Code Blocks = "$errCBs
            #return $errCBs
        fi
    done
    # Check for mismatched CBs based on cbErr
    mismatchedCBs_string=$(grep -Po 'Mismatched CBs\s*\K\d+' $1)
    readarray -t mismatchedCBs_array <<< "$mismatchedCBs_string"
    for mismatchedCBs in "${mismatchedCBs_array[@]}"; do
        if [ "$mismatchedCBs" -ne "0" ]; then
            echo "Mismatch detected for at least one TB in a cell or UCI - Mismatched Code Blocks = "$mismatchedCBs
            return $mismatchedCBs
        fi
    done
    # Check for MismatchedCRC CBs against cbErr
    mismatchedCrcCBs_string=$(grep -Po 'MismatchedCRC CBs\s*\K\d+' $1)
    readarray -t mismatchedCrcCBs_array <<< "$mismatchedCrcCBs_string"
    for mismatchedCrcCBs in "${mismatchedCrcCBs_array[@]}"; do
        if [ "$mismatchedCrcCBs" -ne "0" ]; then
            echo "MismatchedCRC detected for at least one CB in a cell or UCI - MismatchedCRC Code Blocks = "$mismatchedCrcCBs
            return $mismatchedCrcCBs
        fi
    done
     # Check for MismatchedCRC TBs against tbErr
    mismatchedCrcTBs_string=$(grep -Po 'MismatchedCRC TBs\s*\K\d+' $1)
    readarray -t mismatchedCrcTBs_array <<< "$mismatchedCrcTBs_string"
    for mismatchedCrcTBs in "${mismatchedCrcTBs_array[@]}"; do
        if [ "$mismatchedCrcTBs" -ne "0" ]; then
            echo "MismatchedCRC detected for at least one TB in a cell or UCI - MismatchedCRC Transport Blocks = "$mismatchedCrcTBs
            return $mismatchedCrcTBs
        fi
    done

    # Check for metric mismatches by searching for "ERROR:" and "mismatch" in the same line
    if [[ $(grep -Eo 'ERROR:.*mismatch' $1) ]]; then
        metricMismatch_string=$(grep -Eo 'ERROR:.*mismatch' $1)
        readarray -t metricMismatch_array <<< "$metricMismatch_string"
        echo "PUSCH metric mismatches detected, count ${#metricMismatch_array[@]}"
        return ${#metricMismatch_array[@]}
    fi
    
    # Check for metric mismatches by searching for "ERR" and "mismatch" in the same line
    if [[ $(grep -Eo 'ERR.*mismatch' $1) ]]; then
        metricMismatch_string=$(grep -Eo 'ERR.*mismatch' $1)
        readarray -t metricMismatch_array <<< "$metricMismatch_string"
        echo "PUSCH metric mismatches detected, count ${#metricMismatch_array[@]}"
        return ${#metricMismatch_array[@]}
    fi
}

function checkPucchMismatch {
    # Check for mismatches
    mismatches_string=$(grep -Po 'found\s*\K\d+' $1)
    readarray -t mismatches_array <<< "$mismatches_string"
    for mismatches in "${mismatches_array[@]}"; do
        if [ $mismatches -ne 0 ]; then
            echo "Mismatch detected for PUCCH = "$mismatches
            return $mismatches
        fi
    done
    echo ""
}

function checkSimplexMismatch {
   # Check for mismatches
   mismatches_string=$(grep -Po 'found\s*\K\d+' $1)
   readarray -t mismatches_array <<< "$mismatches_string"
    for mismatches in "${mismatches_array[@]}"; do
        if [ $mismatches -ne 0 ]; then
            echo "Mismatch detected for Simplex code = "$mismatches
            return $mismatches
        fi
    done
    echo ""
}

function checktest {
    # Check exit code of process that was the input to tee
    pid_errval=${PIPESTATUS[0]}
    if [ $pid_errval -ne 0 ]; then
       echo "Process exited with non-zero exit code $pid_errval"
       return 1
    fi
}

# argument 1: test command to run
# argument 2: mask of allowed sanitizers
# argument 3: prefix for test command (e.g. setting environment variables before test)
function runtest {
   test_cmd=$1
   test_prefix=''
   sanitizer_idx=0
   sanitizer_arg=$(($2 & $RUN_COMPUTE_SANITIZER))
   if [ $# -ge 3 ]; then
      test_prefix=$3
   fi
   while [ $sanitizer_arg -gt 0 ]; do
      sanitizer_idx=$((sanitizer_idx+1))
      if (( $sanitizer_arg & 1)); then
         case $sanitizer_idx in
            1)
               compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool memcheck --leak-check full"
               ;;
            2)
               compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool racecheck --racecheck-report all"
               ;;
            3)
               compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool synccheck"
               ;;
            4)
               compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool initcheck"
               ;;
         esac
         full_cmd=`echo -n $test_prefix $compute_sanitizer_cmd_prefix $test_cmd`
         echo "$full_cmd"
         eval $full_cmd | tee $LOGFILE
         checkmem $LOGFILE
         if [ -z $tv ]; then
            $PYTHON3 test/test_summary.py -t ${sanitizer_tool_options[$sanitizer_idx]} -r $errval
         else 
            $PYTHON3 test/test_summary.py -t ${sanitizer_tool_options[$sanitizer_idx]} -v $tv -r $errval
         fi
      fi
      sanitizer_arg=$(($sanitizer_arg >> 1))
   done
   # No compute-sanitizer selected
   if [ $sanitizer_idx -eq 0 ]; then
      full_cmd=`echo -n $test_prefix $test_cmd`
      echo "$full_cmd"
      eval $full_cmd | tee $LOGFILE
      checkmem $LOGFILE
   fi
}

if [ -z "$DEBUG" ]; then
   set -e # Exit immediately on non-zero return status.
fi

BASE_TV_PATH=${1:-'./'} #to enable easy standalone testing with a specified dir.
if [[ "$BASE_TV_PATH" == "-h" ]] || [[ "$BASE_TV_PATH" == *"-?" ]] || [ $# -gt 6 ]; then
   usage
fi

export CUDA_VISIBLE_DEVICES=${2:-0}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

PUSCH_CB_ERROR_CHECK=${3:-1} # flag to enable or disable PUSCH CB Error check
echo "PUSCH_CB_ERROR_CHECK=${PUSCH_CB_ERROR_CHECK}"

RUN_COMPONENT_TESTS=${4:-0} # flag to enable or disable Component level tests
echo "RUN_COMPONENT_TESTS=${RUN_COMPONENT_TESTS}"

RUN_COMPUTE_SANITIZER=${5:-1} # flag to enable or disable compute-sanitizer's tools; run only memcheck by default.
sanitizer_selected=''
sanitizer_idx=0
sanitizer_arg=$RUN_COMPUTE_SANITIZER
if [ $sanitizer_arg -gt 15 ]; then
   usage
fi
while [ $sanitizer_arg -ne 0 ]; do
   sanitizer_idx=$((sanitizer_idx+1))
   if (($sanitizer_arg & 1)); then
      sanitizer_selected=`echo -n $sanitizer_selected ${sanitizer_tool_options[$sanitizer_idx]}`
   fi
   sanitizer_arg=$(($sanitizer_arg >> 1))
done
echo "RUN_COMPUTE_SANITIZER=${RUN_COMPUTE_SANITIZER}, i.e., tools: $sanitizer_selected"

# Python3 with mypy, interrogate, pytest installed required
PYTHON3=${6:-'python3'}
$PYTHON3 --version

# pycuphy
echo Testing pycuphy ...
echo Static Type Analysis
$PYTHON3 -m mypy src/pycuphy/src/pycuphy --namespace-packages --explicit-package-bases --no-incremental --disallow-incomplete-defs --disallow-untyped-defs

echo Verify Docstring Coverage
pushd src/pycuphy
$PYTHON3 -m interrogate -vv --omit-covered-files
popd

echo Run the pycuphy unit tests
PYTHONPATH=./build/src/pycuphy $PYTHON3 -m pytest ./test/pycuphy

# SSB
bin_file=./build/examples/ss/testSS
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "*SSB_gNB_CUPHY*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv}`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      fi
      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -m 1`
      runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
	done
fi

# PDCCH
bin_file=./build/examples/pdcch/embed_pdcch_tf_signal
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "*PDCCH_gNB_CUPHY*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv}`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      fi
      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -m 1`
      runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
	done
fi


# PDSCH
#TODO Not all PDSCH + PDSCH component examples might be needed. Can comment some out.
#TODO Potentially include PDSCH multi-cell, but it needs a YAML file.
PDSCH_CMDS=(
            './build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 0' #PDSCH in non-AAS mode, streams
            #'./build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 0' #PDSCH in AAS mode, streams
            './build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 1' #PDSCH in non-AAS mode, graphs
            #'./build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 1' #PDSCH in AAS mode, graphs
            #'./build/examples/dl_rate_matching/dl_rate_matching TV_PLACEHOLDER 2'
            #'./build/examples/modulation_mapper/modulation_mapper TV_PLACEHOLDER 2'
            #'./build/examples/pdsch_dmrs/pdsch_dmrs TV_PLACEHOLDER'
            )

for ((i=0; i < "${#PDSCH_CMDS[@]}"; i++)); do
   pdsch_cmd=${PDSCH_CMDS[$i]} # Do not modify this as bin_file (see next line) should come first.
   bin_file=`echo -n $pdsch_cmd | sed -e "s/ .*$//g"`
   if [ -f "${bin_file}" ]; then
      for tv in $(find $BASE_TV_PATH \( -name "*PDSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-DS*.h5" -o -name "TV_cuphy_F01-DS*.h5" -o -name "TV_cuphy_V*-DS*.h5" \)); do
         current_pdsch_cmd=`echo -n "$pdsch_cmd" | sed -e "s|TV_PLACEHOLDER|$tv|g"`
         echo "$current_pdsch_cmd"
         runtest "$current_pdsch_cmd" 1
      done
   fi
done

# CSI-RS
bin_file=./build/examples/csi_rs/nzp_csi_rs_test
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "*CSIRS_gNB_CUPHY*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv}`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      fi
      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -m 1`
      runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
	done
fi

# PRACH
bin_file=./build/examples/prach_receiver_multi_cell/prach_receiver_multi_cell
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "*PRACH_gNB_CUPHY*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd="${bin_file} -i ${tv} -r 1 -k"
         echo "$test_cmd"
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER 'CUDA_MEMCHECK_PATCH_MODULE=1'
      fi
      # Graphs mode
		test_cmd="${bin_file} -i ${tv} -r 1 -k -m 1"
      echo "$test_cmd"
      runtest "$test_cmd" $RUN_COMPUTE_SANITIZER 'CUDA_MEMCHECK_PATCH_MODULE=1'
	done
fi

# PUCCH
bin_file=./build/examples/pucch_rx_pipeline/cuphy_ex_pucch_rx_pipeline
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "TVnr_*PUCCH_F?_gNB_CUPHY_s*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv}`
         runtest "$test_cmd" 15
         checkPucchMismatch $LOGFILE
      fi
      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -m 1`
      runtest "$test_cmd" 15
      checkPucchMismatch $LOGFILE
	done
fi

# PUSCH
bin_file=./build/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe
if [ -f "${bin_file}" ]; then
   # Streams mode
   if [ $ENABLE_STREAMS -eq 1 ]; then
      for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-US*.h5" -o -name "TV_cuphy_F01-US*.h5" -o -name "TV_cuphy_V*-US*.h5" \)); do
         test_cmd=`echo -n ${bin_file} -i ${tv} -r 1`
         runtest "$test_cmd" 7
         if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
            echo "do checkPuschResults $LOGFILE"
            checkPuschResults $LOGFILE 0
         fi
      done
   fi

   # Graphs mode
   for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-US*.h5" -o -name "TV_cuphy_F01-US*.h5" -o -name "TV_cuphy_V*-US*.h5" \)); do
      test_cmd=`echo -n ${bin_file} -i ${tv} -m 1 -r 1`
      runtest "$test_cmd" 7

      if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
         echo "do checkPuschResults $LOGFILE"
         checkPuschResults $LOGFILE 0
      fi
	done

   # HARQ2 cases
   for tv in $(find $BASE_TV_PATH -name "TVnrPUSCH_HARQ2_*_CUPHY_s0*.h5"); do
      
      # disable PUSCH TC7356 temporarily
      if [[ "$tv" == *"7356"* ]]; then
          continue
      fi
      
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv} -r 1 -H 2`
         runtest "$test_cmd" 7

         if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
            checkPuschResults $LOGFILE
         fi
      fi

      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -m 1 -r 1 -H 2`
      runtest "$test_cmd" 7

      if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
         checkPuschResults $LOGFILE
      fi
   done

   # HARQ4 cases
   for tv in $(find $BASE_TV_PATH -name "TVnrPUSCH_HARQ4_*_CUPHY_s0*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv} -r 1 -H 4`
         runtest "$test_cmd" 7

         if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
            checkPuschResults $LOGFILE
         fi
      fi

      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -m 1 -r 1 -H 4`
      runtest "$test_cmd" 7

      if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
         checkPuschResults $LOGFILE
      fi
   done
fi

# BWC
bin_file=./build/examples/bfc/cuphy_ex_bfc
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "*BFW_gNB_CUPHY*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv} -r 1 -m 0 -c`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      fi

      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -r 1 -m 1 -c`
      runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
	done
fi

# SRS
bin_file=./build/examples/srs_rx_pipeline/cuphy_ex_srs_rx_pipeline
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "TVnr_*_SRS_gNB_CUPHY_*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv}`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      fi
      # Graphs mode
      test_cmd=`echo -n ${bin_file} -i ${tv} -G`
      runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
	done
fi

# Simplex code
bin_file=./build/examples/simplex_decoder/cuphy_ex_simplex_decoder
if [ -f "${bin_file}" ]; then
	for tv in $(find $BASE_TV_PATH -name "TVnr_*SIMPLEX_gNB_CUPHY_s*.h5"); do
      # Streams mode
      if [ $ENABLE_STREAMS -eq 1 ]; then
         test_cmd=`echo -n ${bin_file} -i ${tv}`
         runtest "$test_cmd" 15
         checkSimplexMismatch $LOGFILE
      fi
	done
fi

# Component level tests
if [ $RUN_COMPONENT_TESTS -eq 1 ]; then

   # PDSCH dl_rate_matching, modulation_mapper, pdsch_dmrs, AAS mode
   PDSCH_CMDS=(
               #'./build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 0' #PDSCH in non-AAS mode, streams
               './build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 0' #PDSCH in AAS mode, streams
               #'./build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 1' #PDSCH in non-AAS mode, graphs
               './build/examples/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 1' #PDSCH in AAS mode, graphs
               './build/examples/dl_rate_matching/dl_rate_matching TV_PLACEHOLDER 2'
               './build/examples/modulation_mapper/modulation_mapper TV_PLACEHOLDER 2'
               './build/examples/pdsch_dmrs/pdsch_dmrs TV_PLACEHOLDER'
               )

   for ((i=0; i < "${#PDSCH_CMDS[@]}"; i++)); do
      pdsch_cmd=${PDSCH_CMDS[$i]} # Do not modify this as bin_file (see next line) should come first.
      bin_file=`echo -n $pdsch_cmd | sed -e "s/ .*$//g"`
      if [ -f "${bin_file}" ]; then
         for tv in $(find $BASE_TV_PATH \( -name "*PDSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-DS*.h5" -o -name "TV_cuphy_F01-DS*.h5" -o -name "TV_cuphy_V*-DS*.h5" \)); do
            current_pdsch_cmd=`echo -n "$pdsch_cmd" | sed -e "s|TV_PLACEHOLDER|$tv|g"`
            echo "$current_pdsch_cmd"
            runtest "$current_pdsch_cmd" 1
         done
      fi
   done

   # LDPC
   bin_file=./build/examples/error_correction/cuphy_ex_ldpc
   if [ -f "${bin_file}" ]; then
      for tv in $(find $BASE_TV_PATH -name "*ldpc_BG1*.h5"); do
         test_cmd=`echo -n ${bin_file} -i ${tv} -n 10 -p 8 -f`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      done

      for tv in $(find $BASE_TV_PATH -name "*ldpc_BG2*.h5"); do
         test_cmd=`echo -n ${bin_file} -i ${tv} -n 10 -p 8 -g 2`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      done

      # When no TV is provided, input data is generated. Uses the LDPC encoder to do so.
      test_cmd=`echo -n ${bin_file}`
      runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
   fi

   # Polar
   bin_file=./build/examples/polar_encoder/cuphy_ex_polar_encoder
   if [ -f "${bin_file}" ]; then
      for tv in $(find $BASE_TV_PATH -name "*TV_polarEnc*.h5"); do
         test_cmd=`echo -n ${bin_file} -i ${tv}`
         runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
      done
   fi

    # Tests for all cuPHY make test cases that do not require a special directory.
    # These tests do not require TV input
    COMPONENT_TESTS=('./build/test/crc/crcTest' #gtest for CRC
                     './build/test/descrambling/testDescrambling' #gtest for descrambling
                     './build/test/dl_rate_matching/testDlRateMatching' #gtest for rate matching
                     './build/test/modulation_mapper/testModulationMapper' #gtest for modulation mapper
                     './build/test/error_correction/test_ldpc_internal_app_addr' #gtest for LDPC
                     './build/test/error_correction/test_ldpc_internal_rc' #gtest for LDPC
                     './build/test/error_correction/test_ldpc_internal_loader' #gtest for LDPC
                     './build/test/soft_demapper/test_soft_demapper_internal' #gtest for soft demapper
                     './build/test/rng/test_rng' #gtest for random number generation
                     './build/test/fill/test_fill' #gtest for tensor fill operations
                     './build/test/tile/test_tile' #gtest for tensor tile/repmat operations
                     './build/test/elementwise/test_elementwise' #gtest for tensor elementwise operations
                     './build/test/reduction/test_reduction' #gtest for tensor reduction operations
                     './build/test/kernelDescr/testKernelDescr' #gtest for kernel descriptors
                    )

    for ((i=0; i < "${#COMPONENT_TESTS[@]}"; i++)); do
       bin_file="${COMPONENT_TESTS[$i]}"
       if [ -f "${bin_file}" ]; then
          echo "$bin_file"
          $bin_file | tee $LOGFILE
          checktest $LOGFILE
       fi
    done

    # Tests for the make test cases that require a special working directory
    HDF5_TESTS=('./build/test/hdf5/test_hdf5'
                './build/test/hdf5/test_hdf5_host'
               )
    cwd=`pwd` # save previous working directory
    HDF5_DIR="./test/hdf5" # HDF5 tests should be run from this directory
    if [ -d ${HDF5_DIR} ]; then
       cd ${HDF5_DIR}
    fi
    for ((i=0; i < "${#HDF5_TESTS[@]}"; i++)); do
       bin_file="${cwd}/${HDF5_TESTS[$i]}"
       if [ -f "${bin_file}" ]; then
          echo "$bin_file"
          $bin_file | tee $LOGFILE
          checktest $LOGFILE
       fi
    done
    cd ${cwd} # restore working directory
fi
