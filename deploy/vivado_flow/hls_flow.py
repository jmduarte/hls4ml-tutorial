#Boards

## ZCU106
#board_name='zcu106'
#fpga_part='xczu7ev-ffvc1156-2-e'
 
## Ultra96
#board_name='ultra96'
#fpga_part='xczu3eg-sbva484-1-e'

## Pynq-Z1
#board_name='pynqz1'
#fpga_part='xc7z020clg400-1'

## Pynq-Z2
board_name='pynqz2'
fpga_part='xc7z020clg400-1'

## MiniZed
# board_name='minized'
# fpga_part='xc7z007sclg225-1'

##Cmod A7-35t
#board_name='cmoda735t'
#fpga_part='xc7a35tcpg236-1'

import os
import hls4ml
import plotting
import keras_model
import sys
import matplotlib.pyplot as plt
import numpy as np



os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

def is_tool(name):
    from distutils.spawn import find_executable
    return find_executable(name) is not None

print('-----------------------------------')
if not is_tool('vivado_hls'):
    print('Xilinx Vivado HLS is NOT in the PATH')
else:
    print('Xilinx Vivado HLS is in the PATH')
print('-----------------------------------')



model_file = "{model}/model_{machine_type}.h5".format(model="./model/train_config_bits_6_frames_5_mels_128_encDims_8_bn_True_l1reg_0_expPower_3_beginSpar_0_finSpar_0.8",
                                                           machine_type="ToyCar")
# model_file = "model/KERAS_check_best_model.h5"
if not os.path.exists(model_file):
    print("{} model not found at path ".format(model_file))
    sys.exit(-1)
model = keras_model.load_model(model_file)
model.summary()


w = model.layers[1].weights[0].numpy()
h, b = np.histogram(w, bins=100)
plt.figure(figsize=(7,7))
plt.bar(b[:-1], h, width=b[1]-b[0])
plt.semilogy()
print('% of zeros = {}'.format(np.sum(w==0)/np.size(w)))


hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_config['Model'] = {}
hls_config['Model']['ReuseFactor'] = 1024
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['Precision'] = 'ap_fixed<7,4>'
hls_config['LayerName']['input_1']['Precision'] = 'ap_fixed<7,7>'

hls_config['LayerName']['q_dense']['Precision']['weight'] = 'ap_fixed<7,1>'
hls_config['LayerName']['q_dense']['Precision']['bias'] = 'ap_fixed<7,1>'
hls_config['LayerName']['q_dense']['ReuseFactor'] = 1024

hls_config['LayerName']['batch_normalization']['Precision']['scale'] = 'ap_fixed<7,4>'
hls_config['LayerName']['batch_normalization']['Precision']['bias'] = 'ap_fixed<7,4>'
hls_config['LayerName']['batch_normalization']['ReuseFactor'] = 1024

hls_config['LayerName']['q_activation']['Precision']['result'] = 'ap_fixed<7,4>'
hls_config['LayerName']['q_activation']['ReuseFactor'] = 1024

for i in range(1,9):
    
    hls_config['LayerName']['q_dense_{}'.format(i)]['Precision']['weight'] = 'ap_fixed<7,1>'
    hls_config['LayerName']['q_dense_{}'.format(i)]['Precision']['bias'] = 'ap_fixed<7,1>'
    hls_config['LayerName']['q_dense_{}'.format(i)]['ReuseFactor'] = 1024

    hls_config['LayerName']['batch_normalization_{}'.format(i)]['Precision']['scale'] = 'ap_fixed<7,4>'
    hls_config['LayerName']['batch_normalization_{}'.format(i)]['Precision']['bias'] = 'ap_fixed<7,4>'
    hls_config['LayerName']['batch_normalization_{}'.format(i)]['ReuseFactor'] = 1024

    hls_config['LayerName']['q_activation_{}'.format(i)]['Precision']['result'] = 'ap_fixed<7,4>'
    hls_config['LayerName']['q_activation_{}'.format(i)]['ReuseFactor'] = 1024
    
#final output
hls_config['LayerName']['q_dense_9']['Precision']['weight'] = 'ap_fixed<7,7>'
hls_config['LayerName']['q_dense_9']['Precision']['bias'] = 'ap_fixed<7,7>'
hls_config['LayerName']['q_dense_9']['ReuseFactor'] = 1024

print("-----------------------------------")
plotting.print_dict(hls_config)
print("-----------------------------------")



interface = 'm_axi' # 's_axilite', 'm_axi', 'hls_stream'
axi_width = 16 # 16, 32, 64
implementation = 'serial' # 'serial', 'dataflow'

output_dir='hls/' + board_name + '_' + interface + '_' + str(axi_width) + '_' + implementation + '_prj' 

backend_config = hls4ml.converters.create_backend_config(fpga_part=fpga_part)
backend_config['ProjectName'] = 'anomaly_detection'
backend_config['KerasModel'] = model
backend_config['HLSConfig'] = hls_config
backend_config['OutputDir'] = output_dir
backend_config['Backend'] = 'Pynq'
backend_config['Interface'] = interface
backend_config['IOType'] = 'io_parallel'
backend_config['AxiWidth'] = str(axi_width)
backend_config['Implementation'] = implementation
backend_config['ClockPeriod'] = 10

hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=hls_config,
                                                       output_dir=output_dir,
                                                       fpga_part=fpga_part)

print("-----------------------------------")
plotting.print_dict(backend_config)
plotting.print_dict(hls_config)
print("-----------------------------------")

hls_model = hls4ml.converters.keras_to_hls(backend_config)

_ = hls_model.compile()

hls_model.build(csim=False,synth=True,export=True)

hls4ml.report.read_vivado_report(output_dir)