// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XJET_TAGGER_AXI_H
#define XJET_TAGGER_AXI_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xjet_tagger_axi_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
#else
typedef struct {
    u16 DeviceId;
    u32 Ctrl_bus_BaseAddress;
} XJet_tagger_axi_Config;
#endif

typedef struct {
    u32 Ctrl_bus_BaseAddress;
    u32 IsReady;
} XJet_tagger_axi;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XJet_tagger_axi_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XJet_tagger_axi_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XJet_tagger_axi_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XJet_tagger_axi_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XJet_tagger_axi_Initialize(XJet_tagger_axi *InstancePtr, u16 DeviceId);
XJet_tagger_axi_Config* XJet_tagger_axi_LookupConfig(u16 DeviceId);
int XJet_tagger_axi_CfgInitialize(XJet_tagger_axi *InstancePtr, XJet_tagger_axi_Config *ConfigPtr);
#else
int XJet_tagger_axi_Initialize(XJet_tagger_axi *InstancePtr, const char* InstanceName);
int XJet_tagger_axi_Release(XJet_tagger_axi *InstancePtr);
#endif

void XJet_tagger_axi_Start(XJet_tagger_axi *InstancePtr);
u32 XJet_tagger_axi_IsDone(XJet_tagger_axi *InstancePtr);
u32 XJet_tagger_axi_IsIdle(XJet_tagger_axi *InstancePtr);
u32 XJet_tagger_axi_IsReady(XJet_tagger_axi *InstancePtr);
void XJet_tagger_axi_EnableAutoRestart(XJet_tagger_axi *InstancePtr);
void XJet_tagger_axi_DisableAutoRestart(XJet_tagger_axi *InstancePtr);

void XJet_tagger_axi_Set_in_V(XJet_tagger_axi *InstancePtr, u32 Data);
u32 XJet_tagger_axi_Get_in_V(XJet_tagger_axi *InstancePtr);
void XJet_tagger_axi_Set_out_V(XJet_tagger_axi *InstancePtr, u32 Data);
u32 XJet_tagger_axi_Get_out_V(XJet_tagger_axi *InstancePtr);

void XJet_tagger_axi_InterruptGlobalEnable(XJet_tagger_axi *InstancePtr);
void XJet_tagger_axi_InterruptGlobalDisable(XJet_tagger_axi *InstancePtr);
void XJet_tagger_axi_InterruptEnable(XJet_tagger_axi *InstancePtr, u32 Mask);
void XJet_tagger_axi_InterruptDisable(XJet_tagger_axi *InstancePtr, u32 Mask);
void XJet_tagger_axi_InterruptClear(XJet_tagger_axi *InstancePtr, u32 Mask);
u32 XJet_tagger_axi_InterruptGetEnabled(XJet_tagger_axi *InstancePtr);
u32 XJet_tagger_axi_InterruptGetStatus(XJet_tagger_axi *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
