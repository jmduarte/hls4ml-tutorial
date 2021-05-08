#include "resnet_axi.h"

void resnet_axi(
    input_axi_t in[N_IN],
    output_axi_t out[N_OUT]
        ){

    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
    #pragma HLS INTERFACE m_axi depth=N_IN port=in offset=slave bundle=IN_BUS
    #pragma HLS INTERFACE m_axi depth=N_OUT port=out offset=slave bundle=OUT_BUS

    unsigned short in_size = 0;
    unsigned short out_size = 0;

    hls::stream<input_t> in_local("input_1");
    hls::stream<layer11_t> out_local("output_1");

    #pragma HLS STREAM variable=in_local depth=N_IN
    #pragma HLS STREAM variable=out_local depth=N_OUT

    for(unsigned i = 0; i < N_IN / input_t::size; ++i) {
        input_t ctype;
        #pragma HLS DATA_PACK variable=ctype
        for(unsigned j = 0; j < input_t::size; j++) {
        	ap_ufixed<16,8> tmp = in[i * input_t::size + j]; // store 8 bit input in a larger temp variable
        	ap_ufixed<8,0> tmp2 = tmp >> 8; // shift right by 8 (div by 256) and select only the decimal of the larger temp variable
            ctype[j] = typename input_t::value_type(tmp2);
        }
        in_local.write(ctype);
    }

    resnet(in_local, out_local, in_size, out_size);

    for(unsigned i = 0; i < N_OUT / layer11_t::size; ++i) {
        layer11_t ctype = out_local.read();
        for(unsigned j = 0; j < layer11_t::size; j++) {
            out[i * layer11_t::size + j] = output_axi_t(ctype[j]);
        }
    }
}
