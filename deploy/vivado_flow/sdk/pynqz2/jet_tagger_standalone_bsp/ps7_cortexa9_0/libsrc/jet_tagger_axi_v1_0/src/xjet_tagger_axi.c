// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xjet_tagger_axi.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XJet_tagger_axi_CfgInitialize(XJet_tagger_axi *InstancePtr, XJet_tagger_axi_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_bus_BaseAddress = ConfigPtr->Ctrl_bus_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XJet_tagger_axi_Start(XJet_tagger_axi *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_AP_CTRL) & 0x80;
    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_AP_CTRL, Data | 0x01);
}

u32 XJet_tagger_axi_IsDone(XJet_tagger_axi *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XJet_tagger_axi_IsIdle(XJet_tagger_axi *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XJet_tagger_axi_IsReady(XJet_tagger_axi *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XJet_tagger_axi_EnableAutoRestart(XJet_tagger_axi *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_AP_CTRL, 0x80);
}

void XJet_tagger_axi_DisableAutoRestart(XJet_tagger_axi *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_AP_CTRL, 0);
}

void XJet_tagger_axi_Set_in_V(XJet_tagger_axi *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_IN_V_DATA, Data);
}

u32 XJet_tagger_axi_Get_in_V(XJet_tagger_axi *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_IN_V_DATA);
    return Data;
}

void XJet_tagger_axi_Set_out_V(XJet_tagger_axi *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_OUT_V_DATA, Data);
}

u32 XJet_tagger_axi_Get_out_V(XJet_tagger_axi *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_OUT_V_DATA);
    return Data;
}

void XJet_tagger_axi_InterruptGlobalEnable(XJet_tagger_axi *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_GIE, 1);
}

void XJet_tagger_axi_InterruptGlobalDisable(XJet_tagger_axi *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_GIE, 0);
}

void XJet_tagger_axi_InterruptEnable(XJet_tagger_axi *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_IER);
    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_IER, Register | Mask);
}

void XJet_tagger_axi_InterruptDisable(XJet_tagger_axi *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_IER);
    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_IER, Register & (~Mask));
}

void XJet_tagger_axi_InterruptClear(XJet_tagger_axi *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XJet_tagger_axi_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_ISR, Mask);
}

u32 XJet_tagger_axi_InterruptGetEnabled(XJet_tagger_axi *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_IER);
}

u32 XJet_tagger_axi_InterruptGetStatus(XJet_tagger_axi *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XJet_tagger_axi_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XJET_TAGGER_AXI_CTRL_BUS_ADDR_ISR);
}

