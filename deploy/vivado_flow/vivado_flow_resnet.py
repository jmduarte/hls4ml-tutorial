import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects

import numpy as np

import hls4ml

from callbacks import all_callbacks
import plotting

import matplotlib.pyplot as plt

import os
os.environ['PATH'] = '/xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

# Choose the target board. You may need to install the proper board files for the chosen board.
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

acc_name='resnet'

def is_tool(name):
    from distutils.spawn import find_executable
    return find_executable(name) is not None

print('-----------------------------------')
if not is_tool('vivado_hls'):
    print('Xilinx Vivado HLS is NOT in the PATH')
else:
    print('Xilinx Vivado HLS is in the PATH')
print('-----------------------------------')

from tensorflow.keras.datasets import cifar10
from sklearn.utils import shuffle

_, (X_test, y_test) = cifar10.load_data()
X_test = np.ascontiguousarray(X_test)

num_classes = 10
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(X_test.shape)
print(y_test.shape)

import keras_model
    
# RN06
model_file = 'model/resnet/rn06/rushil2_nosoftmax.h5'
    
if not os.path.exists(model_file):
    print("{} model not found at path ".format(model_file))

model = keras_model.load_model(model_file)

model.summary()

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
inputs = Input((32,32,3,),name='input_3')
x = tf.keras.layers.experimental.preprocessing.Rescaling(1/256,name='rescaling_1')(inputs)
outputs = model(x)
model_rescale = Model(inputs=inputs, outputs=outputs)
model_rescale.summary()

tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plotting

y_keras = model_rescale.predict(X_test)

np.savetxt('input_features.dat', X_test[0:10].reshape(10, -1), fmt='%f', delimiter=' ' )       
np.savetxt('output_predictions.dat', y_keras[0:10].reshape(10, -1), fmt='%f', delimiter=' ')

classes = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

plt.figure(figsize=(9,9))
_ = plotting.plotMultiClassRoc(y_test, y_keras, classes)

import plotting # Import local package plotting.py
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y_keras, axis=1))
plt.figure(figsize=(9,9))
_ = plotting.plot_confusion_matrix(cm, classes)

# ## Make an hls4ml configuration

import yaml

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

# RN06
training_config = yaml_load('model/resnet/rn06/rushil2.yml')

hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
hls_config['Model'] = {}
hls_config['Model']['ReuseFactor'] = training_config['convert']['ReuseFactor']
hls_config['Model']['Strategy'] = training_config['convert']['Strategy']
hls_config['Model']['Precision'] = training_config['convert']['Precision']

# THE ORIGINAL CONFIGURATION
for name in hls_config['LayerName'].keys():
    hls_config['LayerName'][name]['Trace'] = bool(training_config['convert']['Trace'])
    hls_config['LayerName'][name]['ReuseFactor'] = training_config['convert']['ReuseFactor']
    hls_config['LayerName'][name]['Precision'] = training_config['convert']['Precision']
    if 'activation' in name:
        hls_config['LayerName'][name]['Precision'] = training_config['convert']['PrecisionActivation']
# custom configs
for name in training_config['convert']['Override'].keys():
    hls_config['LayerName'][name].update(training_config['convert']['Override'][name])

# Enable tracing for all of the layers
for layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][layer]['Trace'] = False
    
print("-----------------------------------")
plotting.print_dict(hls_config)
print("-----------------------------------")


# ## Convert and Compile
# 
# You can set some target specific configurations:

# - Define the `interface`, which for our current setup should always be `m_axi`.
# - Define the  width of the AXI bus. For the time being, use `16` that is each clock cycle you transfer a single input or output value (`ap_fixed<16,*>`).
# - Define the implementation. For the time being, use `serial`.

interface = 'm_axi' # 's_axilite', 'm_axi', 'io_stream'
axi_width = 8 # 16, 32, 64
implementation = 'serial' # 'serial', 'dataflow'

#output_dir='hls/' + '256x16x8x256_' + '_' + interface + '_' + str(axi_width) + '_' + implementation + '_prj'
output_dir='hls/' + board_name + '_' + acc_name + '_' + interface + '_' + str(axi_width) + '_' + implementation + '_prj' 

backend_config = hls4ml.converters.create_backend_config(fpga_part=fpga_part)
backend_config['ProjectName'] = acc_name
backend_config['KerasModel'] = model
backend_config['HLSConfig'] = hls_config
backend_config['OutputDir'] = output_dir
backend_config['Backend'] = 'Pynq'
backend_config['Interface'] = interface
backend_config['IOType'] = 'io_stream'
backend_config['AxiWidth'] = str(axi_width)
backend_config['Implementation'] = implementation
backend_config['ClockPeriod'] = 10
backend_config['InputData'] = 'input_features.dat'
backend_config['OutputPredictions'] = 'output_predictions.dat'

hls_model = hls4ml.converters.keras_to_hls(backend_config)

_ = hls_model.compile()

# ## Profiling

X_test = X_test[0:10].astype(float)
y_test = y_test[0:10]

# Run prediction on the test set for the hls model (fixed-point precision)\n

y_hls = hls_model.predict(X_test)
y_keras = model_rescale.predict(X_test)
print('-----------------------------------')
print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
print('-----------------------------------')

print(y_hls)
print(y_keras)

# Enable logarithmic scale on TPR and FPR axes 
logscale_tpr = False # Y axis
logscale_fpr = False # X axis

fig, ax = plt.subplots(figsize=(9, 9))
_ = plotting.plotMultiClassRoc(y_test, y_keras, classes, logscale_tpr=logscale_tpr, logscale_fpr=logscale_fpr)
plt.gca().set_prop_cycle(None) # reset the colors
_ = plotting.plotMultiClassRoc(y_test, y_hls, classes, logscale_tpr=logscale_tpr, logscale_fpr=logscale_fpr, linestyle='--')

from matplotlib.lines import Line2D
lines = [Line2D([0], [0], ls='-'),
         Line2D([0], [0], ls='--')]
from matplotlib.legend import Legend
leg = Legend(ax, lines, labels=['keras', 'hls4ml'],
            loc='center right', frameon=False)
_ = ax.add_artist(leg)


# ## Synthesis

hls_model.build(csim=True, synth=True, export=True)
hls4ml.report.read_vivado_report(output_dir)

# |                 |               Resource                 |
# +-----------------+---------+-------+--------+-------+-----+
# |      Board      | BRAM_18K| DSP48E|   FF   |  LUT  | URAM|
# +-----------------+---------+-------+--------+-------+-----+
# |   PYNQ-Z1/Z2    |      280|    220|  106400|  53200|    0|
# +-----------------+---------+-------+--------+-------+-----+
# |     MiniZed     |      100|     66|   28800|  14400|    0|
# +-----------------+---------+-------+--------+-------+-----+
# ``` 


os.system('source /xilinx/Vivado/2019.1/settings64.sh && cd sys/{board_name} && make clean sys ACC={acc_name} INTERFACE={interface}'.format(board_name=board_name,acc_name=acc_name,interface=interface))

# **TODO** Tell the user how to visualize the `Block Diagram` to get a better understanding of the IP integration with both Zynq and MicroBlaze PS.

# ## Configure Software in Vivado SDK and Run HW/SW on the Board (Step 6)

# Create Vivado SDK project.
# 
# - `make sdk` to configure an application with register polling
# - `make sdk-irq` to configure an application with interrupts (default)

os.system('source /xilinx/Vivado/2019.1/settings64.sh && cd sdk/{board_name} && make clean sdk ACC={acc_name} SAMPLE_COUNT=10'.format(board_name=board_name,acc_name=acc_name))

#os.system('source /xilinx/Vivado/2019.1/settings64.sh && cd sdk/{board_name} && make gui'.format(board_name=board_name))

# You can open a serial console, for example
# ```
# sudo minicom -D /dev/ttyUSB1
# ```
# and see something like
# 
# ![serial-console](doc/serial_console.png)
