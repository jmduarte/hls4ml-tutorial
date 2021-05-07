

## Pynq-Z2
board_name='pynqz2'
fpga_part='xc7z020clg400-1'


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import hls4ml
import plotting
import keras_model
import sys
import matplotlib.pyplot as plt
import numpy as np
from plot_roc_curve import plot_roc

os.environ['PATH'] = '/opt/local/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

def is_tool(name):
    from distutils.spawn import find_executable
    return find_executable(name) is not None

print('-----------------------------------')
if not is_tool('vivado_hls'):
    # print('Xilinx Vivado HLS is NOT in the PATH')
    raise FileNotFoundError('Xilinx Vivado HLS is NOT in the PATH')
else:
    print('Xilinx Vivado HLS is in the PATH')

print('-----------------------------------')

# 


# list of models being synthesized
models = ["model/medium_model/train_config_bits_{}_frames_4_mels_64_encDims_8_hidDims_64_halfcode_4_fan_64_bn_False_qbatch_False_l1reg_0/model_ToyCar.h5".format(prec) for prec in [16,14,12,10,8,6]]

# number of bits for quantized layers
quantizations = [13,12,11,10,8,6,5,16,14,]
accum_t = 'ap_fixed<32,16>'
BN_quant =[13]

for indx, model_file in enumerate(models):
    if not os.path.exists(model_file):
        print("{} model not found at path ".format(model_file))
        sys.exit(-1)
    model = keras_model.load_model(model_file)
    model.summary()
    print('-----------------------------------')
    print('Model Sparsity Validation')
    print('-----------------------------------')
    w = model.layers[1].weights[0].numpy()
    h, b = np.histogram(w, bins=100)
    plt.figure(figsize=(7,7))
    plt.bar(b[:-1], h, width=b[1]-b[0])
    plt.semilogy()
    print('% of zeros = {}'.format(np.sum(w==0)/np.size(w)))


    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    if isinstance(quantizations[indx], int):
        INT_BITS = int(quantizations[indx]/2)+1
        BITS = quantizations[indx] +1
    reuse = [512]
    #Valid ReuseFactor(s): scales based on input feature size
    for REUSE in reuse:
        hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls_config['Model'] = {}
        hls_config['Model']['ReuseFactor'] = REUSE
        hls_config['Model']['Strategy'] = 'Resource'
        hls_config['Model']['Precision'] = 'ap_fixed<{},{}>'.format(32, 16)
        hls_config['LayerName']['input_1']['Precision'] = 'ap_fixed<{},{}>'.format(8, 8)

        if isinstance(quantizations[indx], int):
            for layer in hls_config['LayerName']:
                if 'dense' in layer:
                    hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<{},1>'.format(BITS)
                    hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<{},1>'.format(BITS)
                    hls_config['LayerName'][layer]['ReuseFactor'] = REUSE
                    hls_config['LayerName']['q_dense']['accum_t'] = accum_t


                if 'normalization' in layer:
                    hls_config['LayerName'][layer]['Precision']['scale'] = 'ap_fixed<{},{}>'.format(13, 6)
                    hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<{},{}>'.format(13, 6)
                    hls_config['LayerName'][layer]['ReuseFactor'] = REUSE
                    hls_config['LayerName']['batch_normalization']['accum_t'] = accum_t

                if 'activation' in layer:
                    hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<{},{}>'.format(BITS, INT_BITS)
                    hls_config['LayerName'][layer]['ReuseFactor'] = REUSE
                    hls_config['LayerName']['q_activation']['accum_t'] = accum_t

        plotting.print_dict(hls_config)

        project_name = 'anomaly_detector'
        interface = 'm_axi' # 's_axilite', 'm_axi', 'hls_stream'
        axi_width = 16 # 16, 32, 64
        implementation = 'serial' # 'serial', 'dataflow'

        sub_dir = 'medium_model/no_BN_dense_quantization'
        output_dir='hls/'+sub_dir+'/' + project_name + '_' + board_name + '_' + interface + '_' + str(axi_width) + '_' + implementation + '_dense_Q_'+str(quantizations[indx]) + '_reuse_' +str(REUSE) + '_prj' 

        # SET BACKEND
        # backend_config = hls4ml.converters.create_backend_config(fpga_part=fpga_part)
        # backend_config['ProjectName'] = 'anomaly_detector'
        # backend_config['KerasModel'] = model
        # backend_config['HLSConfig'] = hls_config
        # backend_config['OutputDir'] = output_dir
        # backend_config['Backend'] = 'Pynq'
        # backend_config['Interface'] = interface
        # backend_config['IOType'] = 'io_parallel'
        # backend_config['AxiWidth'] = str(axi_width)
        # backend_config['Implementation'] = implementation
        # backend_config['ClockPeriod'] = 10

        #print("-----------------------------------")
        #plotting.print_dict(backend_config)
        #print("-----------------------------------")

        #hls_model = hls4ml.converters.keras_to_hls(backend_config)
        hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                                hls_config=hls_config,
                                                                output_dir=output_dir,
                                                                fpga_part=fpga_part,
                                                                clock_period=10,
                                                                io_type='io_parallel',
                                                                project_name=project_name)

        print("-----------------------------------")
        plotting.print_dict(hls_config)
        print("-----------------------------------")

        _ = hls_model.compile()

        # PROFILING AND VALIDATION
        print("validating HLS model and Keras Model accuracy")
        X = np.load('./test_data/test_data_frames_4_hops_512_fft_1024_mels_64_power_2.0.npy', allow_pickle=True)
        y = np.load('./test_data/test_data_frames_4_hops_512_fft_1024_mels_64_power_2.0_ground_truths.npy', allow_pickle=True)
        plot_roc(hls_model=hls_model, keras_model=model, X=X, y=y, output_dir=output_dir)

        hls_model.build(csim=True,synth=True,export=False, vsynth=True)

        hls4ml.report.read_vivado_report(output_dir)
        plotting.print_dict(hls_config)