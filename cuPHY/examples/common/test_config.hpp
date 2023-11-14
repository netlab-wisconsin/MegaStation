/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(TEST_CONFIG_HPP_INCLUDED_)
#define TEST_CONFIG_HPP_INCLUDED_

#include <vector>
#include <map>
#include "yaml.hpp"

namespace cuphy
{

////////////////////////////////////////////////////////////////////////
// test_config
// Class with a collection of channel strings and associated filenames
// for use with test programs that may process multiple slots.
//
// Example usage:
//     test_config cfg("test_config.yaml");
//     cfg.print(); // To show what was parsed
//
// Example input files:
//
// ---------------------------------------------------------------------
// test_config_example.yaml:
//
//cells: 3
//slots:
//    - PUSCH:
//        - UL_TV1.h5
//        - UL_TV2.h5
//        - UL_TV3.h5
//    - PUSCH:
//        - UL_TV1.h5
//        - UL_TV2.h5
//        - UL_TV3.h5
//    - PUSCH:
//        - UL_TV1.h5
//        - UL_TV2.h5
//        - UL_TV3.h5
//
// ---------------------------------------------------------------------
// test_config_example_fdd.yaml:
//cells: 3
//slots:
//    - PUSCH:
//        - UL_TV1.h5
//        - UL_TV2.h5
//        - UL_TV3.h5
//      PDSCH:
//        - DL_TV1.h5
//        - DL_TV2.h5
//        - DL_TV3.h5
//    - PUSCH:
//        - UL_TV1.h5
//        - UL_TV2.h5
//        - UL_TV3.h5
//      PDSCH:
//        - DL_TV1.h5
//        - DL_TV2.h5
//        - DL_TV3.h5
//    - PUSCH:
//        - UL_TV1.h5
//        - UL_TV2.h5
//        - UL_TV3.h5
//      PDSCH:
//        - DL_TV1.h5
//        - DL_TV2.h5
//        - DL_TV3.h5
// ---------------------------------------------------------------------
// test_config_example_tdd.yaml:
//cells: 3
//slots:
//    - PUSCH:
//        - UL_TV1.h5
//        - UL_TV2.h5
//        - UL_TV3.h5
//    - PDSCH:
//        - DL_TV1.h5
//        - DL_TV2.h5
//        - DL_TV3.h5
//    - PDSCH:
//        - DL_TV3.h5
//        - DL_TV1.h5
//        - DL_TV2.h5
//    - PDSCH:
//        - DL_TV2.h5
//        - DL_TV3.h5
//        - DL_TV1.h5
//    - PDSCH:
//        - DL_TV1.h5
//        - DL_TV3.h5
//        - DL_TV2.h5
class test_config
{
public:
    typedef std::vector<std::string>          file_vec_t;
    typedef std::map<std::string, file_vec_t> channel_map_t;
    typedef std::vector<channel_map_t>        slot_vec_t;
    //------------------------------------------------------------------
    // test_config()
    test_config(const char* filename)
    {
        yaml::file_parser fp(filename);
        yaml::document d = fp.next_document();
        yaml::node     r = d.root();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // number of cells (scalar)
        num_cells_ = r["cells"].as<unsigned int>();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // slots (sequence of mappings)
        yaml::node     slots = r["slots"];
        for(size_t slotIdx = 0; slotIdx < slots.length(); ++slotIdx)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // slot (mapping of channel names to sequence)
            yaml::node    slot = slots[slotIdx];
            channel_map_t cmap;
            for(size_t channelIdx = 0; channelIdx < slot.length(); ++channelIdx)
            {
                std::string  channelName = slot.key(channelIdx);
                file_vec_t   fileNames;
                yaml::node channel      = slot[channelName.c_str()];
                for(size_t fileIdx = 0; fileIdx < channel.length(); ++fileIdx)
                {
                    fileNames.push_back(channel[fileIdx].as<std::string>());
                }
                cmap[channelName] = fileNames;
            }
            slots_.push_back(cmap);
        }
    }
    //------------------------------------------------------------------
    // num_cells()
    unsigned int num_cells() const { return num_cells_; }
    //------------------------------------------------------------------
    // num_slots()
    unsigned int num_slots() const { return slots_.size(); }
    //------------------------------------------------------------------
    // slots()
    const slot_vec_t& slots() const { return slots_; }
    //------------------------------------------------------------------
    // print()
    void print() const
    {
        printf("number of cells: %u\n", num_cells_);
        for(size_t i = 0; i < slots_.size(); ++i)
        {
            const channel_map_t& m = slots_[i];
            printf("slot: %lu\n", i);
            for(auto it = m.begin(); it != m.end(); ++it)
            {
                printf("    channel: %s\n", it->first.c_str());
                for(auto& f : it->second)
                {
                    printf("        file: %s\n", f.c_str());
                }
            }
        }
    }    
    // print_channel()
    void print_channel(std::string channel) const
    {
        printf("number of cells: %u\n", num_cells_);
        for(size_t i = 0; i < slots_.size(); ++i)
        {
            const channel_map_t& m = slots_[i];
            printf("slot: %lu\n", i);
            auto it = m.find(channel);
            if(it != m.end())
            {
                printf("    channel: %s\n", it->first.c_str());
                for(auto& f : it->second)
                {
                    printf("        file: %s\n", f.c_str());
                }
            }
        }
    }
private:
    //------------------------------------------------------------------
    // Data
    unsigned int num_cells_;
    slot_vec_t   slots_;
};

} // namespace cuphy

#endif // !defined(TEST_CONFIG_HPP_INCLUDED_)
