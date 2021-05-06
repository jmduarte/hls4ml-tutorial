PROJECT_HLS=$(BOARD)_$(ACC)_m_axi_8_serial_prj

PROJECT = $(ACC)_standalone
PROJECT_I2C = i2c_test

help:
	@echo "INFO: make <TAB> to show targets"
.PHONY: help

setup:
	xsct script.tcl $(ACC)
.PHONY: setup

sdk: setup data
	rm -f $(PROJECT)/src/helloworld.c
	cd  $(PROJECT)/src && ln -s ../../../common/$(ACC)/main.c
	#rm -f $(PROJECT_I2C)/src/helloworld.c
	#cd  $(PROJECT_I2C)/src && ln -s ../../../main_i2c_test.c
.PHONY: sdk

#sdk-irq: setup data
#	rm -f $(PROJECT)/src/helloworld.c
#	cd  $(PROJECT)/src && ln -s ../../../common/main_irq.c main.c
#	#rm -f $(PROJECT_I2C)/src/helloworld.c
#	#cd  $(PROJECT_I2C)/src && ln -s ../../../main_i2c_test.c
#.PHONY: sdk-irq

gui:
	xsdk --workspace .
.PHONY: gui

clean:
	rm -rf $(PROJECT)
	rm -rf $(PROJECT)_bsp
	rm -rf $(PROJECT_I2C)
	rm -rf $(PROJECT_I2C)_bsp
	rm -rf $(ACC)_platform
	rm -rf RemoteSystemsTempFiles
	rm -rf SDK.log
	rm -rf webtalk
	rm -rf .sdk
	rm -rf .Xil
	rm -rf .metadata
.PHONY: clean
