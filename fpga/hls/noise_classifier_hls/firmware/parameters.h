#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_transpose.h"
#include "nnet_utils/nnet_transpose_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/w14.h"
#include "weights/b14.h"
#include "weights/w19.h"
#include "weights/b19.h"


// hls-fpga-machine-learning insert layer-config
// transpose_input_for_x
struct config20 {
    static const unsigned dims = 3;
    static const unsigned N = 50176;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config20_from_shape[3] = {1, 224, 224};
unsigned config20_to_shape[3] = {224, 224, 1};
unsigned config20_perm[3] = {1, 2, 0};
unsigned config20_perm_strides[3] = {224, 1, 50176};

const unsigned* const config20::from_shape = config20_from_shape;
const unsigned* const config20::to_shape = config20_to_shape;
const unsigned* const config20::perm = config20_perm;
const unsigned* const config20::perm_strides = config20_perm_strides;

// zp2d_features_0_block_0
struct config22 : nnet::padding2d_config {
    static const unsigned in_height = 224;
    static const unsigned in_width = 224;
    static const unsigned n_chan = 1;
    static const unsigned out_height = 226;
    static const unsigned out_width = 226;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// features_0_block_0
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 64;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef features_0_block_0_accum_t accum_t;
    typedef features_0_block_0_bias_t bias_t;
    typedef features_0_block_0_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 226;
    static const unsigned in_width = 226;
    static const unsigned n_chan = 1;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 224;
    static const unsigned out_width = 224;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 226;
    static const unsigned min_width = 226;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 50176;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef features_0_block_0_accum_t accum_t;
    typedef features_0_block_0_bias_t bias_t;
    typedef features_0_block_0_weight_t weight_t;
    typedef config2_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config2::filt_height * config2::filt_width> config2::pixels[] = {0};

// features_0_block_2
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 802816;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 64;
    typedef features_0_block_2_table_t table_t;
};

// features_0_block_3
struct config5 : nnet::pooling2d_config {
    static const unsigned in_height = 224;
    static const unsigned in_width = 224;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 112;
    static const unsigned out_width = 112;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = true;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 64;
    typedef features_0_block_3_accum_t accum_t;
};

// zp2d_features_1_block_0
struct config23 : nnet::padding2d_config {
    static const unsigned in_height = 112;
    static const unsigned in_width = 112;
    static const unsigned n_chan = 16;
    static const unsigned out_height = 114;
    static const unsigned out_width = 114;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// features_1_block_0
struct config6_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 64;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef features_1_block_0_accum_t accum_t;
    typedef features_1_block_0_bias_t bias_t;
    typedef features_1_block_0_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config6 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 114;
    static const unsigned in_width = 114;
    static const unsigned n_chan = 16;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 112;
    static const unsigned out_width = 112;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 114;
    static const unsigned min_width = 114;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 12544;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef features_1_block_0_accum_t accum_t;
    typedef features_1_block_0_bias_t bias_t;
    typedef features_1_block_0_weight_t weight_t;
    typedef config6_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config6::filt_height * config6::filt_width> config6::pixels[] = {0};

// features_1_block_2
struct relu_config8 : nnet::activ_config {
    static const unsigned n_in = 401408;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 64;
    typedef features_1_block_2_table_t table_t;
};

// features_1_block_3
struct config9 : nnet::pooling2d_config {
    static const unsigned in_height = 112;
    static const unsigned in_width = 112;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 56;
    static const unsigned out_width = 56;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = true;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 64;
    typedef features_1_block_3_accum_t accum_t;
};

// zp2d_features_2_block_0
struct config24 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_chan = 32;
    static const unsigned out_height = 58;
    static const unsigned out_width = 58;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// features_2_block_0
struct config10_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 64;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef features_2_block_0_accum_t accum_t;
    typedef features_2_block_0_bias_t bias_t;
    typedef features_2_block_0_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config10 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 58;
    static const unsigned in_width = 58;
    static const unsigned n_chan = 32;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 56;
    static const unsigned out_width = 56;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 58;
    static const unsigned min_width = 58;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 3136;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef features_2_block_0_accum_t accum_t;
    typedef features_2_block_0_bias_t bias_t;
    typedef features_2_block_0_weight_t weight_t;
    typedef config10_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config10::filt_height * config10::filt_width> config10::pixels[] = {0};

// features_2_block_2
struct relu_config12 : nnet::activ_config {
    static const unsigned n_in = 200704;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 64;
    typedef features_2_block_2_table_t table_t;
};

// features_2_block_3
struct config13 : nnet::pooling2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 56;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = true;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 64;
    typedef features_2_block_3_accum_t accum_t;
};

// zp2d_features_3_block_0
struct config25 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 30;
    static const unsigned out_width = 30;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// features_3_block_0
struct config14_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 64;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef features_3_block_0_accum_t accum_t;
    typedef features_3_block_0_bias_t bias_t;
    typedef features_3_block_0_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config14 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 30;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 28;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 30;
    static const unsigned min_width = 30;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 784;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef features_3_block_0_accum_t accum_t;
    typedef features_3_block_0_bias_t bias_t;
    typedef features_3_block_0_weight_t weight_t;
    typedef config14_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config14::filt_height * config14::filt_width> config14::pixels[] = {0};

// features_3_block_2
struct relu_config16 : nnet::activ_config {
    static const unsigned n_in = 50176;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 64;
    typedef features_3_block_2_table_t table_t;
};

// pool
struct config17 : nnet::pooling2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 28;
    static const unsigned stride_width = 28;
    static const unsigned pool_height = 28;
    static const unsigned pool_width = 28;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 1;
    static const unsigned out_width = 1;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = true;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 64;
    typedef pool_accum_t accum_t;
};

// transpose_input_for_classifier_0
struct config21 {
    static const unsigned dims = 3;
    static const unsigned N = 64;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config21_from_shape[3] = {1, 1, 64};
unsigned config21_to_shape[3] = {64, 1, 1};
unsigned config21_perm[3] = {2, 0, 1};
unsigned config21_perm_strides[3] = {1, 64, 64};

const unsigned* const config21::from_shape = config21_from_shape;
const unsigned* const config21::to_shape = config21_to_shape;
const unsigned* const config21::perm = config21_perm;
const unsigned* const config21::perm_strides = config21_perm_strides;

// classifier_2
struct config19 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 5;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 320;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef classifier_2_accum_t accum_t;
    typedef classifier_2_bias_t bias_t;
    typedef classifier_2_weight_t weight_t;
    typedef layer19_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
