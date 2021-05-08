#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/resnet_axi.h"
#include "firmware/nnet_utils/nnet_helpers.h"

#define CHECKPOINT 1

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline(fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }


      //hls-fpga-machine-learning insert data
      input_axi_t inputs[N_IN];
      ap_ufixed<16,8> inputs_fxd[N_IN]; // Create a temp arrays with a "larger" fixed-point datatype (integer part, 8 bits, are initially unused)
      nnet::copy_data<float, ap_ufixed<16,8>, 0, N_IN>(in, inputs_fxd); // Copy floating point values in the temp array
      for (unsigned i = 0; i < N_IN; i++)  {
    	  inputs[i] = input_axi_t(inputs_fxd[i] << 8); // Shift left by 8 is multiply by 256 ("move 8 bits of decimal part into the integer part")
      }                                                // and select the integer part only as input

      output_axi_t outputs[N_OUT];

      //hls-fpga-machine-learning insert top-level-function
      resnet_axi(inputs,outputs);

      if (e % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_OUT; i++) {
          std::cout << pr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
        nnet::print_result<output_axi_t, N_OUT>(outputs, std::cout, true);
      }
      e++;

      //hls-fpga-machine-learning insert tb-output
      nnet::print_result<output_axi_t, N_OUT>(outputs, fout);

    }
    fin.close();
    fpr.close();
  } else {
;
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
