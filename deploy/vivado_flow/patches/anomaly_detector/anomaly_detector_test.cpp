#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/anomaly_detector_axi.h"
#include "firmware/nnet_utils/nnet_helpers.h"

#define CHECKPOINT 1

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

#if 0
int main(int argc, char **argv)
{
    //load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    //load predictions from text file
    //  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    //std::string pline;
    int e = 0;

    if (fin.is_open() /*&& fpr.is_open()*/) {
        while ( std::getline(fin,iline) /* && std::getline (fpr,pline) */) {
            if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
            char* cstr=const_cast<char*>(iline.c_str());
            char* current;
            std::vector<float> in;
            current=strtok(cstr," ");
            while(current!=NULL) {
                in.push_back(atof(current));
                current=strtok(NULL," ");
            }
            /*
               cstr=const_cast<char*>(pline.c_str());
               std::vector<float> pr;
               current=strtok(cstr," ");
               while(current!=NULL) {
               pr.push_back(atof(current));
               current=strtok(NULL," ");
               }
               */

            //hls-fpga-machine-learning insert data
            input_t inputs[N_IN];
            nnet::copy_data<float, input_t, 0, N_IN>(in, inputs);
            result_t outputs[N_OUT];

            //hls-fpga-machine-learning insert top-level-function
            unsigned short size_in1,size_out1;
            anomaly_detector(inputs, outputs, size_in1, size_out1);

            if (e % CHECKPOINT == 0) {
                /*
                   std::cout << "Predictions" << std::endl;
                //hls-fpga-machine-learning insert predictions
                for(int i = 0; i < N_LAYER_10; i++) {
                std::cout << pr[i] << " ";
                }
                std::cout << std::endl;
                */
                std::cout << "Quantized predictions" << std::endl;
                //hls-fpga-machine-learning insert quantized
                nnet::print_result<result_t, N_OUT>(outputs, std::cout, true);
            }
            e++;

            //hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, N_OUT>(outputs, fout);

        }
        fin.close();
        /*fpr.close();*/
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
        //hls-fpga-machine-learning insert zero
        input_t input_1[N_INPUT_1_1];
        nnet::fill_zero<input_t, N_INPUT_1_1>(input_1);
        layer10_t layer10_out[N_LAYER_10];

        //hls-fpga-machine-learning insert top-level-function
        unsigned short size_in1,size_out1;
        anomaly_detector(input_1,layer10_out,size_in1,size_out1);

        //hls-fpga-machine-learning insert output
        nnet::print_result<layer10_t, N_LAYER_10>(layer10_out, std::cout, true);

        //hls-fpga-machine-learning insert tb-output
        nnet::print_result<layer10_t, N_LAYER_10>(layer10_out, fout);
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
#else
int main(int argc, char **argv)
{
    //load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);
    std::string iline;

    input_axi_t inputs[AXI_DEPTH_IN * MAX_BATCH_SIZE] ;
    output_axi_t outputs[AXI_DEPTH_OUT * MAX_BATCH_SIZE];

    unsigned batch = 1;

    if (fin.is_open()) {
        unsigned i = 0;

        std::cout << "Loading input from file ..." << std::endl;
        while ( std::getline(fin,iline) ) {
            if (i >= batch) break;
            char* cstr=const_cast<char*>(iline.c_str());
            char* current;
            std::vector<float> in;
            current=strtok(cstr," ");
            while(current!=NULL) {
                in.push_back(atof(current));
                current=strtok(NULL," ");
            }

            //hls-fpga-machine-learning insert data
            input_t input_feature[N_IN];
            nnet::copy_data<float, input_t, 0, N_IN>(in, input_feature);

            for (unsigned j = 0; j < AXI_DEPTH_IN; j++) {
                for (unsigned k = 0; k < W_COUNT_IN; k++) {
                    input_t data = input_feature[j * W_COUNT_IN + k];
                    inputs[i * AXI_DEPTH_IN +  j].range(((k+1)*W_IN)-1, k*W_IN) = data.range(W_IN-1, 0);
                }
            }
            i++;
        }

        std::cout << "Total inputs: " << i << std::endl;

       anomaly_detector_axi(inputs,outputs, batch);

       for (unsigned i = 0; i < MAX_BATCH_SIZE; i++) {
           if (i >= batch) break;
           ap_fixed<8,8> output_feature[N_OUT];
           for (unsigned j = 0; j < AXI_DEPTH_OUT; j++) {
               output_axi_t output_axi = outputs[i * AXI_DEPTH_OUT +  j];
               for (unsigned k = 0; k < W_COUNT_OUT; k++) {
                   output_feature[j * W_COUNT_IN + k].range(W_OUT-1, 0) = output_axi.range(((k+1)*W_OUT)-1, k*W_OUT);
               }
           }
           nnet::print_result<ap_fixed<8,8>, N_OUT>(output_feature, fout);
           nnet::print_result<ap_fixed<8,8>, N_OUT>(output_feature, std::cout, true);
       }

#if 0
            //hls-fpga-machine-learning insert top-level-function
            unsigned short size_in1,size_out1;


            if (e % CHECKPOINT == 0) {
                std::cout << "Quantized predictions" << std::endl;
                //hls-fpga-machine-learning insert quantized
                nnet::print_result<layer10_t, N_LAYER_10>(layer10_out, std::cout, true);
            }
            e++;

            //hls-fpga-machine-learning insert tb-output
            nnet::print_result<layer10_t, N_LAYER_10>(layer10_out, fout);
#endif

        fin.close();
        /*fpr.close();*/
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
#endif
