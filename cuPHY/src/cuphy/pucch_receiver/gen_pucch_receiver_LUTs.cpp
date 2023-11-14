/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "cuphy.h"

using namespace std;

/* Used in orthogonal sequences */
int8_t phi_values_for_tocc[7][49]  =
    { /* 1 symbol */
      {0, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1},

      /* 2 symbols */
      {0,  0, -1, -1, -1, -1, -1,
       0,  1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1},

      /* 3 symbols */
      {0,  0,  0, -1, -1, -1, -1,
       0,  1,  2, -1, -1, -1, -1,
       0,  2,  1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1},

      /* 4 symbols */
      {0,  0,  0,  0, -1, -1, -1,
       0,  2,  0,  2, -1, -1, -1,
       0,  0,  2,  2, -1, -1, -1,
       0,  2,  2,  0, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1},

      /* 5 symbols */
      {0,  0,  0,  0,  0, -1, -1,
       0,  1,  2,  3,  4, -1, -1,
       0,  2,  4,  1,  3, -1, -1,
       0,  3,  1,  4,  2, -1, -1,
       0,  4,  3,  2,  1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1},

      /* 6 symbols */
      {0,  0,  0,  0,  0,  0, -1,
       0,  1,  2,  3,  4,  5, -1,
       0,  2,  4,  0,  2,  4, -1,
       0,  3,  0,  3,  0,  3, -1,
       0,  4,  2,  0,  4,  2, -1,
       0,  5,  4,  3,  2,  1, -1,
      -1, -1, -1, -1, -1, -1, -1},

      /* 7 symbols */
      {0,  0,  0,  0,  0,  0,  0,
       0,  1,  2,  3,  4,  5,  6,
       0,  2,  4,  6,  1,  3,  5,
       0,  3,  6,  2,  5,  1,  4,
       0,  4,  1,  5,  2,  6,  3,
       0,  5,  3,  1,  6,  4,  2,
       0,  6,  5,  4,  3,  2,  1}};


int8_t phi_values_for_PAPR[NUM_PAPR_SEQUENCES][CUPHY_N_TONES_PER_PRB] =
    {{-3,  1, -3, -3, -3,  3, -3, -1,  1,  1,  1, -3},
     {-3,  3,  1, -3,  1,  3, -1, -1,  1,  3,  3,  3},
     {-3,  3,  3,  1, -3,  3, -1,  1,  3, -3,  3, -3},
     {-3, -3, -1,  3,  3,  3, -3,  3, -3,  1, -1, -3},
     {-3, -1, -1,  1,  3,  1,  1, -1,  1, -1, -3,  1},
     {-3, -3,  3,  1, -3, -3, -3, -1,  3, -1,  1,  3},
     { 1, -1,  3, -1, -1, -1, -3, -1,  1,  1,  1, -3},
     {-1, -3,  3, -1, -3, -3, -3, -1,  1, -1,  1, -3},
     {-3, -1,  3,  1, -3, -1, -3,  3,  1,  3,  3,  1},
     {-3, -1, -1, -3, -3, -1, -3,  3,  1,  3, -1, -3},
     {-3,  3, -3,  3,  3, -3, -1, -1,  3,  3,  1, -3},
     {-3, -1, -3, -1, -1, -3,  3,  3, -1, -1,  1, -3},
     {-3, -1,  3, -3, -3, -1, -3,  1, -1, -3,  3,  3},
     {-3,  1, -1, -1,  3,  3, -3, -1, -1, -3, -1, -3},
     { 1,  3, -3,  1,  3,  3,  3,  1, -1,  1, -1,  3},
     {-3,  1,  3, -1, -1, -3, -3, -1, -1,  3,  1, -3},
     {-1, -1, -1, -1,  1, -3, -1,  3,  3, -1, -3,  1},
     {-1,  1,  1, -1,  1,  3,  3, -1, -1, -3,  1, -3},
     {-3,  1,  3,  3, -1, -1, -3,  3,  3, -3,  3, -3},
     {-3, -3,  3, -3, -1,  3,  3,  3, -1, -3,  1, -3},
     { 3,  1,  3,  1,  3, -3, -1,  1,  3,  1, -1, -3},
     {-3,  3,  1,  3, -3,  1,  1,  1,  1,  3, -3,  3},
     {-3,  3,  3,  3, -1, -3, -3, -1, -3,  1,  3, -3},
     { 3, -1, -3,  3, -3, -1,  3,  3,  3, -3, -1, -3},
     {-3, -1,  1, -3,  1,  3,  3,  3, -1, -3,  3,  3},
     {-3,  3,  1, -1,  3,  3, -3,  1, -1,  1, -1,  1},
     {-1,  1,  3, -3,  1, -1,  1, -1, -1, -3,  1, -1},
     {-3, -3,  3,  3,  3, -3, -1,  1, -3,  3,  1, -3},
     { 1, -1,  3,  1,  1, -1, -1, -1,  1,  3, -3,  1},
     {-3,  3, -3,  3, -3, -3,  3, -1, -1,  1,  3, -3}};


template<typename Tscalar>
void computeAndStoreToccLUT(string LUT_directory,
                            string component_name,
                            string LUT_name,
                            string template_type,
                            unsigned int num_symbols,
                            unsigned int elements_per_symbol) {

    ofstream of(LUT_directory + "/" + component_name + "_" + LUT_name + "_LUT.h");

    of << "#ifndef _" << component_name << "_" << LUT_name << "_LUT_H_" << std::endl;
    of << "#define _" << component_name << "_" << LUT_name << "_LUT_H_" << std::endl;
    of << std::endl;

    of << "static __device__ __constant__ " << template_type << " " << LUT_name << "_LUT[" << num_symbols << "]["
       << 2*elements_per_symbol << "] = {" << std::endl;

    for (int symbol_id = 0; symbol_id < num_symbols; symbol_id += 1) {
        of << "    /*" << (symbol_id + 1) << " symbol(s) */" << std::endl;
        of << "    {";
        for (int element_id = 0; element_id < elements_per_symbol; element_id++) {
            Tscalar degrees = (Tscalar)(2.0f * M_PI * phi_values_for_tocc[symbol_id][element_id]/(symbol_id+1));
            of << cos(degrees);
            of << ", ";
            of << -sin(degrees);
            if (element_id != elements_per_symbol - 1) {
                of << "," << std::endl;
                of << "    ";
            }
        }
        of << "}";
        if (symbol_id != num_symbols - 1) {
            of << "," << std::endl << std::endl;
        }
    }
    of << "};" << std::endl;
    of << "#endif";
    of.close();

}

template<typename Tscalar>
void computeAndStoreTimeShiftSeqLUT(string LUT_directory,
                            string component_name,
                            string LUT_name,
                            string template_type,
                            unsigned int num_rows,
                            unsigned int num_cols) {

    ofstream of(LUT_directory + "/" + component_name + "_" + LUT_name + "_LUT.h");

    of << "#ifndef _" << component_name << "_" << LUT_name << "_LUT_H_" << std::endl;
    of << "#define _" << component_name << "_" << LUT_name << "_LUT_H_" << std::endl;
    of << std::endl;

    of << "static __device__ __constant__ " << template_type << " " << LUT_name << "_LUT[" << num_rows << "]["
       << 2 * num_cols << "] = {" << std::endl;

    for (int numerology = 0; numerology < num_rows; numerology += 1) {
        of << "    /* numerology: " << numerology << " */" << std::endl;
        of << "    {";
        for (int tone = 0; tone < num_cols; tone += 1) {
            // Compute tone of the CUPHY_N_TONES_PER_PRB elements of the time shift sequence for
            // the given numerology. Each element is a complex number.
            Tscalar degrees = (Tscalar)(2.0f * M_PI * (1 << numerology) * 15 * tone) / 1000;
            of << cos(degrees);
            of << ", ";
            of << sin(degrees);
            if (tone != num_cols - 1) {
                of << "," << std::endl;
                of << "    ";
            }
        }
        of << "}";
        if (numerology != num_rows - 1) {
            of << "," << std::endl << std::endl;
        }
    }
    of << "};" << std::endl;
    of << "#endif";
    of.close();
}

template<typename Tscalar>
void computeAndStorePaprLUT(string LUT_directory,
                            string component_name,
                            string LUT_name,
                            string template_type,
                            unsigned int num_rows,
                            unsigned int num_cols) {

    ofstream of(LUT_directory + "/" + component_name + "_" + LUT_name + "_LUT.h");

    of << "#ifndef _" << component_name << "_" << LUT_name << "_LUT_H_" << std::endl;
    of << "#define _" << component_name << "_" << LUT_name << "_LUT_H_" << std::endl;
    of << std::endl;

    of << "static __device__ __constant__ " << template_type << " " << LUT_name << "_LUT[" << num_rows << "]["
       << 2 * num_cols << "] = {" << std::endl;

    for (int seq_id = 0; seq_id < num_rows; seq_id += 1) {
        of << "    /* " << seq_id << " PAPR sequence */" << std::endl;
        of << "    {";
        for (int tone = 0; tone < num_cols; tone += 1) {
            Tscalar degrees = (Tscalar)(phi_values_for_PAPR[seq_id][tone] * M_PI ) / ((Tscalar) 4);
            of << cos(degrees);
            of << ", ";
            of << -sin(degrees);
            if (tone != num_cols - 1) {
                of << "," << std::endl;
                of << "    ";
            }
        }
        of << "}";
        if (seq_id != num_rows - 1) {
            of << "," << std::endl << std::endl;
        }
    }
    of << "};" << std::endl;
    of << "#endif";
    of.close();
}


int main(int argc, char** argv) {

    if(argc < 2) {
        std::cout << "Missing LUT directory parameter. Usage: ./genPucchLUT [LUT dirname]\n";
        return -1;
    }
    std::string LUT_DIR = std::string(argv[1]);

    std::cout << "Generate look-up tables for PUCCH Receiver - Format 1" << std::endl;
    std::cout << LUT_DIR << std::endl;

    std::string component_name = "PUCCH_RECEIVER_F1";

    computeAndStoreToccLUT<float>(LUT_DIR, component_name, "TOCC_VALUES", "float", 7, 49);
    computeAndStoreTimeShiftSeqLUT<float>(LUT_DIR, component_name, "TIME_SHIFT_SEQ_VALUES", "float", 5, CUPHY_N_TONES_PER_PRB);
    computeAndStorePaprLUT<float>(LUT_DIR, component_name, "PAPR_SEQ_VALUES", "float", NUM_PAPR_SEQUENCES, CUPHY_N_TONES_PER_PRB);
    return 0;
}
