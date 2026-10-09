#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 224*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer20_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer22_t;
typedef ap_fixed<37,17> features_0_block_0_accum_t;
typedef nnet::array<ap_fixed<37,17>, 16*1> features_0_block_0_result_t;
typedef ap_fixed<16,6> features_0_block_0_weight_t;
typedef ap_fixed<16,6> features_0_block_0_bias_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer4_t;
typedef ap_fixed<18,8> features_0_block_2_table_t;
typedef ap_fixed<16,6> features_0_block_3_accum_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer5_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer23_t;
typedef ap_fixed<41,21> features_1_block_0_accum_t;
typedef nnet::array<ap_fixed<41,21>, 32*1> features_1_block_0_result_t;
typedef ap_fixed<16,6> features_1_block_0_weight_t;
typedef ap_fixed<16,6> features_1_block_0_bias_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer8_t;
typedef ap_fixed<18,8> features_1_block_2_table_t;
typedef ap_fixed<16,6> features_1_block_3_accum_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer9_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer24_t;
typedef ap_fixed<42,22> features_2_block_0_accum_t;
typedef nnet::array<ap_fixed<42,22>, 64*1> features_2_block_0_result_t;
typedef ap_fixed<16,6> features_2_block_0_weight_t;
typedef ap_fixed<16,6> features_2_block_0_bias_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer12_t;
typedef ap_fixed<18,8> features_2_block_2_table_t;
typedef ap_fixed<16,6> features_2_block_3_accum_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer13_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer25_t;
typedef ap_fixed<43,23> features_3_block_0_accum_t;
typedef nnet::array<ap_fixed<43,23>, 64*1> features_3_block_0_result_t;
typedef ap_fixed<16,6> features_3_block_0_weight_t;
typedef ap_fixed<16,6> features_3_block_0_bias_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer16_t;
typedef ap_fixed<18,8> features_3_block_2_table_t;
typedef ap_fixed<36,16> pool_accum_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer17_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer21_t;
typedef ap_fixed<39,19> classifier_2_accum_t;
typedef nnet::array<ap_fixed<39,19>, 5*1> result_t;
typedef ap_fixed<16,6> classifier_2_weight_t;
typedef ap_fixed<16,6> classifier_2_bias_t;
typedef ap_uint<1> layer19_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
