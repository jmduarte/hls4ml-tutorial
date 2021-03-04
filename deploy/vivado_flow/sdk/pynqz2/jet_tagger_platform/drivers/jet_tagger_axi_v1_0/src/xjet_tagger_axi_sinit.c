// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xjet_tagger_axi.h"

extern XJet_tagger_axi_Config XJet_tagger_axi_ConfigTable[];

XJet_tagger_axi_Config *XJet_tagger_axi_LookupConfig(u16 DeviceId) {
	XJet_tagger_axi_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XJET_TAGGER_AXI_NUM_INSTANCES; Index++) {
		if (XJet_tagger_axi_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XJet_tagger_axi_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XJet_tagger_axi_Initialize(XJet_tagger_axi *InstancePtr, u16 DeviceId) {
	XJet_tagger_axi_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XJet_tagger_axi_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XJet_tagger_axi_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

