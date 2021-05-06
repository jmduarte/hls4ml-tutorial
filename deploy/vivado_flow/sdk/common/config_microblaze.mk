PROJECT_HLS=$(BOARD)_$(ACC)_m_axi_8_serial_prj

PROJECT = $(ACC)_standalone

help:
	@echo "INFO: make <TAB> to show targets"
.PHONY: help

setup:
	xsct script.tcl $(ACC)
.PHONY: setup

sdk: setup data
	rm -f $(PROJECT)/src/helloworld.c
	cd  $(PROJECT)/src && ln -s ../../../common/$(ACC)/main_microblaze.c main.c
.PHONY: sdk

gui:
	xsdk --workspace .
.PHONY: gui

SAMPLE_COUNT=10
#SAMPLE_COUNT=166000

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
	rm -rf updatemem*
.PHONY: clean
