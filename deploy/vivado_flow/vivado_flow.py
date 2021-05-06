#!/usr/bin/env python
# coding: utf-8

# 

# ## Setup
# 
# Choose the target board. For the time being, you can use `minized`, `pynqz1`, `pynqz2`, `cmoda735t`. You may need to install the proper board files for the chosen board.

# In[1]:


## ZCU106
#board_name='zcu106'
#fpga_part='xczu7ev-ffvc1156-2-e'
 
## Ultra96
#board_name='ultra96'
#fpga_part='xczu3eg-sbva484-1-e'

## Pynq-Z1
# board_name='pynqz1'
# fpga_part='xc7z020clg400-1'

## Pynq-Z2
board_name='pynqz2'
fpga_part='xc7z020clg400-1'

## MiniZed
#board_name='minized'
#fpga_part='xc7z007sclg225-1'

##Cmod A7-35t
#board_name='cmoda735t'
#fpga_part='xc7a35tcpg236-1'

## Arty A7-100t
#board_name='artya7100t'
#fpga_part='xc7a100t-csg324-1'

## Arty A7-35t
#board_name='artya735t'
#fpga_part='xc7a35ticsg324-1L'


# Add the project name. The notebook will create sub-directories for the Vivado projects with different models and configurations.

# In[2]:


PROJECT_NAME='anomaly_detector'
MODEL_FILE = "model/autoq_models/downsampled_skip/qmodel.h5"

# directory under hls
#SUB_DIR = "downsampled_skip/train_config_bits_8_frames_5_mels_128_encDims_8_hidDims_64_\
#halfcode_2_fan_64_bn_True_qbatch_False_l1reg_0"
SUB_DIR = 'downsampled_skip/qmodel'
# dataset load dir
DATASET_X = "/home/julesmuhizi/AE-Anomaly-Detection/anomaly_detection/v_1/test_data/downsampled_128_5_to_32_4_skip_method.npy"
DATASET_Y = "/home/julesmuhizi/AE-Anomaly-Detection/anomaly_detection/v_1/test_data/downsampled_128_5_to_32_4_ground_truths_skip_method.npy"
#BITS = int(MODEL_FILE.split('bits_')[-1].split('_frames')[0])+1
#INT_BITS = int(BITS/2)+1
BITS=8
INT_BITS=4
REUSE = 4096


# Let's import the libraries, call the magic functions, and setup the environment variables.

# In[3]:


import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from sklearn.metrics import accuracy_score
from sklearn import metrics

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects

import numpy as np

import hls4ml

from callbacks import all_callbacks
import plotting

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
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


# ## Load the dataset
# 
# This is a lot like the previous notebooks, so we will go through quickly.
# 
# First, we fetch the dataset from file, do the normalization and make a train and test split.
# 
# We save the test dataset to files so that we can use them later.

# In[4]:


#load processed test data
from sklearn.utils import shuffle
X = np.load(DATASET_X, allow_pickle=True)
y = np.load(DATASET_Y, allow_pickle=True)
y_keras = []
#use a quarter of the test_set to save time
for i in range(len(X)):
    quarter = int(len(X[i])/4)
    assert len(X) == len(y)
    #X[i], y[i] = shuffle(X[i], y[i])
    X[i], y[i] = X[i][0:quarter],  y[i][0:quarter]


# ## Train or Load Model

# In[5]:


import keras_model
train = False
#not os.path.exists('model/KERAS_check_best_model.h5')
if train:
    model.compile(loss="mean_squared_error", optimizer="adam")
        
    print("Shape of training data element is: {}".format(train_data[0].shape))
    history = model.fit(train_data,
                        train_data,
                        epochs=100,
                        batch_size=512,
                        shuffle=true,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=callbacks)
    

elif not os.path.exists(MODEL_FILE):
    print("{} model not found at path ".format(MODEL_FILE))
model = keras_model.load_model(MODEL_FILE)
    
model.summary()


# In[6]:


print(model.get_layer('dense_2').get_quantizers()[0])


# ## Check accuracy
# 
# Do not expect a good accuracy because of the low amount of neurons. I could have done better than this, but as long as it fits both Pynq-Z1 and MiniZed, it is fine with us.
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#%matplotlib inline
import plotting
import numpy

#load processed test data
# X = np.load('./test_data/test_data_frames_4_hops_512_fft_1024_mels_64_power_2.0.npy', allow_pickle=True)
# y = np.load('./test_data/test_data_frames_4_hops_512_fft_1024_mels_64_power_2.0_ground_truths.npy', allow_pickle=True)
y_keras = []
#use a quarter of the test_set to save time
for i in range(len(X)):
    quarter = int(len(X[i])/4)
    assert len(X) == len(y)
    X[i], y[i] = shuffle(X[i], y[i])
    X[i], y[i] = X[i][0:quarter],  y[i][0:quarter]

#perform inference
for index, X_data in enumerate(X):
    y_pred = [0. for ind in X_data]
    for file_idx, X_test in enumerate(X_data):
        predictions = model.predict(X_test)
        errors = np.mean(np.square(X_test-predictions), axis=1)
        y_pred[file_idx] = numpy.mean(errors)
        
    #generate auc and roc metrics
    y_test = y[index]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    y_keras.append(y_pred)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'AUC m_{} = {}'.format(index, round(roc_auc,2)), linewidth = 1.5)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--', linewidth=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plt.show()

# ## Make an hls4ml configuration (Step 2)
# 
# Notice we're using `Strategy: Resource` for every layer, and `ReuseFactor: 64`. The Programmable Logic (FPGA part) of the Pynq-Z1 SoC is not big compared to VU9P type of parts.
# 
# We also use some settings which are good for QKeras.
# 
# Notice the `fpga_part:'xc7z020clg400-1'`.

# In[7]:


hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

accum_t = 'ap_fixed<32,16>'

hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls4ml.utils.config.set_data_types_from_keras_model(hls_config, model, 16, test_inputs=X[0][0])
#hls4ml.utils.config.set_accum_from_keras_model(hls_config, model)
'''
hls_config['Model'] = {}
hls_config['Model']['ReuseFactor'] = REUSE
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['Precision'] = 'ap_fixed<32,16>'
hls_config['LayerName']['input_1']['Precision'] = 'ap_fixed<8,8>'


for layer in hls_config['LayerName']:
    if 'dense' in layer:
        hls_config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<{},1>'.format(BITS)
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<{},1>'.format(BITS)
        hls_config['LayerName'][layer]['ReuseFactor'] = REUSE
        hls_config['LayerName'][layer]['accum_t'] = accum_t


    if 'normalization' in layer:
        hls_config['LayerName'][layer]['Precision']['scale'] = 'ap_fixed<{},{}>'.format(16, 6)
        hls_config['LayerName'][layer]['Precision']['bias'] = 'ap_fixed<{},{}>'.format(16, 6)
        hls_config['LayerName'][layer]['ReuseFactor'] = REUSE
#         hls_config['LayerName'][layer]['accum_t'] = accum_t

    if 'activation' in layer:
        hls_config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<{},{}>'.format(BITS, INT_BITS)
        hls_config['LayerName'][layer]['ReuseFactor'] = REUSE
#         hls_config['LayerName'][layer]['accum_t'] = accum_t

    # set output dense quantization
    output_dense = [layer_name for layer_name in hls_config['LayerName'] if 'dense' in layer_name]
    output_dense.sort()
    output_dense = output_dense[-1]
    hls_config['LayerName'][output_dense]['Precision']['weight'] = 'ap_fixed<{},{}>'.format(BITS,BITS)
    hls_config['LayerName'][output_dense]['Precision']['bias'] = 'ap_fixed<{},{}>'.format(BITS,BITS)
    hls_config['LayerName'][output_dense]['ReuseFactor'] = REUSE
    hls_config['LayerName'][output_dense]['accum_t'] = accum_t
'''
# Enable tracing for all of the layers
for layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][layer]['Trace'] = True

# Reuse factor all of the layers
for layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][layer]['ReuseFactor'] = REUSE
    if 'dense' in layer:
        hls_config['LayerName'][layer]['accum_t'] = accum_t
        
# print("-----------------------------------")
plotting.print_dict(hls_config)
# print("-----------------------------------")


# ## Convert and Compile
# 
# You can set some target specific configurations:

# - Define the `interface`, which for our current setup should always be `m_axi`.
# - Define the  width of the AXI bus. For the time being, use `16` that is each clock cycle you transfer a single input or output value (`ap_fixed<16,*>`).
# - Define the implementation. For the time being, use `serial`.

# In[8]:


interface = 'm_axi' # 's_axilite', 'm_axi', 'hls_stream'
axi_width = 8 # 16, 32, 64
implementation = 'serial' # 'serial', 'dataflow'


# In[9]:


#output_dir='hls/' + '256x16x8x256_' + '_' + interface + '_' + str(axi_width) + '_' + implementation + '_prj'
output_dir="hls/{}/".format(SUB_DIR) + board_name + '_' + PROJECT_NAME + '_' + interface + '_' + str(axi_width) + '_' + implementation + '_prj' 

# backend_config = hls4ml.converters.create_backend_config(fpga_part=fpga_part)
# backend_config['ProjectName'] = project_name
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

# hls_model = hls4ml.converters.keras_to_hls(backend_config)
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                                hls_config=hls_config,
                                                                output_dir=output_dir,
                                                                fpga_part=fpga_part,
                                                                clock_period=10,
                                                                io_type='io_parallel',
                                                                project_name=PROJECT_NAME)

_ = hls_model.compile()


# # ## Profiling
a,b,c,d = hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=X[0][0])
# a.savefig('a.png')
# b.savefig('b.png')
# c.savefig('c.png')
# d.savefig('d.png')


# ## Prediction and Comparison
# 

error1 = hls4ml.model.profiling.compare(model, hls_model, np.ascontiguousarray(X[0][0]),'norm_diff')
error2 = hls4ml.model.profiling.compare(model, hls_model, np.ascontiguousarray(X[0][0]))
error1.savefig('error1.png')
error2.savefig('error2.png')
# In[ ]:
'''

#load processed test data
X = np.load(DATASET_X, allow_pickle=True)
y = np.load(DATASET_Y, allow_pickle=True)


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

#use a quarter of the test_set to save time
for i in range(len(X)):
    quarter = int(len(X[i])/4)
    assert len(X) == len(y)
    X[i], y[i] = shuffle(X[i], y[i])
    X[i], y[i] = X[i][0:quarter],  y[i][0:quarter]

#perform inference
for index, X_data in enumerate(X):
    keras_pred = [0. for ind in X_data]
    hls_pred = [0. for ind in X_data]
    for file_idx, X_test in enumerate(X_data):
        keras_predictions = model.predict(X_test)
        keras_errors = np.mean(np.square(X_test-keras_predictions), axis=1)
        keras_pred[file_idx] = np.mean(keras_errors)
        
        hls_predictions = hls_model.predict(X_test)
        hls_errors = np.mean(np.square(X_test-hls_predictions), axis=1)
        hls_pred[file_idx] = np.mean(hls_errors)
        
    #generate auc and roc metrics
    y_test = y[index]
    k_fpr, k_tpr, k_threshold = metrics.roc_curve(y_test, keras_pred)
    k_roc_auc = metrics.auc(k_fpr, k_tpr)
    h_fpr, h_tpr, h_threshold = metrics.roc_curve(y_test, hls_pred)
    h_roc_auc = metrics.auc(h_fpr, h_tpr)


    plt.title('Receiver Operating Characteristic')
    plt.plot(k_fpr, k_tpr, label = 'keras AUC m_{} = {}'.format(index, round(k_roc_auc,2)), linewidth = 1.5, color=colors[index])
    plt.plot(h_fpr, h_tpr, label = 'hls AUC m_{} = {}'.format(index, round(h_roc_auc,2)), linewidth = 1, linestyle='--', color=colors[index])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--', linewidth=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('roc_curve_autoq_model.png')


# 
# ## Synthesis

# In[ ]:


hls_model.build(csim=False,synth=True,export=True, vsynth=True)

hls4ml.report.read_vivado_report(output_dir)


# ## Resource Reference
# 
# See the resources availables on different boards.
# 
# ```
# +-----------------+---------+-------+--------+-------+-----+                    
# |                 |               Resource                 |
# +-----------------+---------+-------+--------+-------+-----+
# |      Board      | BRAM_18K| DSP48E|   FF   |  LUT  | URAM|
# +-----------------+---------+-------+--------+-------+-----+
# |   PYNQ-Z1/Z2    |      280|    220|  106400|  53200|    0|
# +-----------------+---------+-------+--------+-------+-----+
# |     MiniZed     |      100|     66|   28800|  14400|    0|
# +-----------------+---------+-------+--------+-------+-----+
# ``` 

# ## Generate .dat Files (Step 3)
# 
# The .dat files are used
# - during the following `csim` step
# - to generate the header files for SDK

# f = open(output_dir + '/tb_data/tb_input_features.dat', 'w')
# 
# # This is under the assumption that 
# # 1. all the machines have the same number of wave files
# # 2. all of the wave files have the same number of frames
# # 3. all of the frames have the same length
# 
# machine_count=len(X)
# wav_count=len(X[0])
# frame_count=len(X[0][0])
# frame_length=len(X[0][0][0])
# 
# # Save the first N frames in the first wave file of the first machine
# N=10
# for i in range(N):
#     for j in range(frame_length):
#         f.write('{} '.format(X[0][0][i][j]))
#     f.write('\n')
# f.close()
# 
# f = open(output_dir + '/tb_data/tb_output_predictions.dat', 'w')
# for i in range(N):
#     f.write('{} '.format(y[0][i]))
#     f.write('\n')
# f.close()

# ## Run Vivado HLS csim (Step 4)
# 
# At this step we generate simulation traces out from the hls4ml-model.
# 
# Run the following cell to run Vivado HLS GUI:
!cd $output_dir && vivado_hls -p anomaly_detector_prj
# **IMPORTANT** Click the button to `Run C Simulation`.
# 
# This will generate simulation traces with fixed-point arythmetic.
# 
# When completed close Vivado HLS GUI.

# ## Integrate IP in a Vivado Project and Generate Bitstream (Step 5)

# !cd sys/$board_name && make clean sys-gui

# **TODO** Tell the user how to visualize the `Block Diagram` to get a better understanding of the IP integration with both Zynq and MicroBlaze PS.

# ## Configure Software in Vivado SDK and Run HW/SW on the Board (Step 6)

# Create Vivado SDK project.
# 
# - `make sdk` to configure an application with register polling
# - `make sdk-irq` to configure an application with interrupts (default)

# !source /tools/Xilinx/Vivado/2019.1/settings64.sh && cd sdk/$board_name && make clean sdk

# !xterm -e "sleep 1 && source /tools/Xilinx/Vivado/2019.1/settings64.sh && cd sdk/$board_name && make gui && sleep infinity"
# 

# You can open a serial console, for example
# ```
# sudo minicom -D /dev/ttyUSB0
# ```
# and see something like
# 
# ![serial-console](doc/serial_console.png)
'''