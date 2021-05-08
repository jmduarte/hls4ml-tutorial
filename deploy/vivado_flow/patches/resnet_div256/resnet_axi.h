#ifndef RESNET_AXI_H_
#define RESNET_AXI_H_

#include "resnet.h"

static const unsigned N_IN = 3072;
static const unsigned N_OUT = 10;
typedef ap_uint<8> input_axi_t;
typedef ap_fixed<8,6> output_axi_t;

void resnet_axi(
    input_axi_t in[N_IN],
    output_axi_t out[N_OUT]
        );
#endif
