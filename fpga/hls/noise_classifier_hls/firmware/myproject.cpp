#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer19_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x,layer19_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<features_0_block_0_weight_t, 144>(w2, "w2.txt");
        nnet::load_weights_from_txt<features_0_block_0_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<features_1_block_0_weight_t, 4608>(w6, "w6.txt");
        nnet::load_weights_from_txt<features_1_block_0_bias_t, 32>(b6, "b6.txt");
        nnet::load_weights_from_txt<features_2_block_0_weight_t, 18432>(w10, "w10.txt");
        nnet::load_weights_from_txt<features_2_block_0_bias_t, 64>(b10, "b10.txt");
        nnet::load_weights_from_txt<features_3_block_0_weight_t, 36864>(w14, "w14.txt");
        nnet::load_weights_from_txt<features_3_block_0_bias_t, 64>(b14, "b14.txt");
        nnet::load_weights_from_txt<classifier_2_weight_t, 320>(w19, "w19.txt");
        nnet::load_weights_from_txt<classifier_2_bias_t, 5>(b19, "b19.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=50176

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=51076

    hls::stream<features_0_block_0_result_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=50176

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=50176

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=12544

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=12996

    hls::stream<features_1_block_0_result_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=12544

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=12544

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=3136

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=3364

    hls::stream<features_2_block_0_result_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=3136

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=3136

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=784

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=900

    hls::stream<features_3_block_0_result_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=784

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=784

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=1

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=64

    auto& layer18_out = layer21_out;
    nnet::transpose<input_t, layer20_t, config20>(x, layer20_out); // transpose_input_for_x

    nnet::zeropad2d_cl<layer20_t, layer22_t, config22>(layer20_out, layer22_out); // zp2d_features_0_block_0

    nnet::conv_2d_cl<layer22_t, features_0_block_0_result_t, config2>(layer22_out, layer2_out, w2, b2); // features_0_block_0

    nnet::relu<features_0_block_0_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // features_0_block_2

    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // features_0_block_3

    nnet::zeropad2d_cl<layer5_t, layer23_t, config23>(layer5_out, layer23_out); // zp2d_features_1_block_0

    nnet::conv_2d_cl<layer23_t, features_1_block_0_result_t, config6>(layer23_out, layer6_out, w6, b6); // features_1_block_0

    nnet::relu<features_1_block_0_result_t, layer8_t, relu_config8>(layer6_out, layer8_out); // features_1_block_2

    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // features_1_block_3

    nnet::zeropad2d_cl<layer9_t, layer24_t, config24>(layer9_out, layer24_out); // zp2d_features_2_block_0

    nnet::conv_2d_cl<layer24_t, features_2_block_0_result_t, config10>(layer24_out, layer10_out, w10, b10); // features_2_block_0

    nnet::relu<features_2_block_0_result_t, layer12_t, relu_config12>(layer10_out, layer12_out); // features_2_block_2

    nnet::pooling2d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // features_2_block_3

    nnet::zeropad2d_cl<layer13_t, layer25_t, config25>(layer13_out, layer25_out); // zp2d_features_3_block_0

    nnet::conv_2d_cl<layer25_t, features_3_block_0_result_t, config14>(layer25_out, layer14_out, w14, b14); // features_3_block_0

    nnet::relu<features_3_block_0_result_t, layer16_t, relu_config16>(layer14_out, layer16_out); // features_3_block_2

    nnet::pooling2d_cl<layer16_t, layer17_t, config17>(layer16_out, layer17_out); // pool

    nnet::transpose<layer17_t, layer21_t, config21>(layer17_out, layer21_out); // transpose_input_for_classifier_0

    nnet::dense<layer21_t, result_t, config19>(layer18_out, layer19_out, w19, b19); // classifier_2

}

