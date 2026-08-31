# RK3588 Camera 驱动说明

## 一、驱动源码目录树

```txt
kernel/
├── arch/arm64/boot/dts/rockchip/
│   ├── rk3588s.dtsi                        # SoC 基础 DTS（csi2_dphy/dcphy、mipiN_csi2、rkcif、rkisp 节点定义）
│   └── topeet-camera-config.dtsi           # 板级相机配置（J1~J4 接口、sensor、endpoint 连接）
│
├── drivers/media/i2c/                      # Sensor 驱动
│   ├── ov5695.c                            # OV5695 5MP sensor
│   ├── ov13850.c                           # OV13850 13MP sensor
│   └── imx415.c                            # Sony IMX415 4K sensor
│
├── drivers/phy/rockchip/                   # MIPI D-PHY / C-PHY 驱动
│   ├── phy-rockchip-csi2-dphy.c            # CSI2 DPHY 框架层（csi2_dphy0~5）
│   ├── phy-rockchip-csi2-dphy-hw.c          # CSI2 DPHY 硬件层（csi2_dphy0_hw/1_hw）
│   ├── phy-rockchip-csi2-dphy-common.h      # DPHY 公共定义
│   └── phy-rockchip-inno-mipi-dphy.c        # Innosilicon MIPI DPHY
│
├── drivers/media/platform/rockchip/
│   ├── cif/                                # RKCIF —— MIPI/LVDS/DVP 输入捕获
│   │   ├── dev.c / dev.h                   # CIF 设备注册、v4l2/media 框架
│   │   ├── hw.c / hw.h                     # CIF 寄存器级硬件操作
│   │   ├── common.c / common.h             # CIF 公共接口
│   │   ├── capture.c                       # CIF 视频捕获节点（/dev/video）
│   │   ├── mipi-csi2.c / mipi-csi2.h       # mipiN_csi2 子设备（CSI2 接收）
│   │   ├── cif-scale.c                     # CIF scaler
│   │   ├── cif-luma.c / cif-luma.h          # Luma 统计
│   │   ├── subdev-itf.c / subdev-itf.h      # CIF 对外 subdev 接口（含 sditf）
│   │   ├── procfs.c / procfs.h             # procfs 调试
│   │   ├── regs.h                          # 寄存器定义
│   │   └── version.h
│   │
│   ├── isp/                                # RKISP —— ISP 图像处理
│   │   ├── rkisp.c / rkisp.h               # ISP 平台驱动入口
│   │   ├── dev.c / dev.h                   # ISP 设备注册
│   │   ├── hw.c / hw.h                     # ISP 硬件操作
│   │   ├── common.c / common.h             # ISP 公共
│   │   ├── csi.c / csi.h                   # ISP CSI 输入（rkisp_vir0~3）
│   │   ├── bridge.c / bridge.h             # ISP bridge（输入路径桥接）
│   │   ├── dmarx.c / dmarx.h               # DMA 接收
│   │   ├── capture.c / capture.h           # ISP 视频输出节点
│   │   ├── isp_params.c / isp_params.h      # 3A 参数下发
│   │   ├── isp_stats.c / isp_stats.h        # 3A 统计上报
│   │   ├── isp_ispp.h                      # ISP <-> ISPP 交互接口
│   │   ├── isp_external.h                 # ISP 对外接口
│   │   ├── regs.h / regs.c                 # 寄存器
│   │   └── *_v1x/*_v2x/*_v3x.*             # 各 ISP 版本实现
│   │
│   └── ispp/                               # RKISPP —— ISP 后处理
│       ├── stream.c / stream.h             # ISPP 视频流（输入/输出节点）
│       ├── dev.c / dev.h                   # ISPP 设备注册
│       ├── hw.c / hw.h                     # ISPP 硬件
│       ├── common.c / common.h             # ISPP 公共
│       ├── ispp.c / ispp.h                 # ISPP 核心
│       ├── params.c / params.h             # 后处理参数下发
│       ├── stats.c / stats.h               # 统计上报
│       ├── fec.c / fec.h                   # FEC（去畸变）
│       ├── procfs.c / procfs.h             # procfs 调试
│       └── regs.h
│
└── include/
    └── uapi/linux/rk-camera-module.h        # camera-module 用户态 ABI（module-index/facing/lens 等）
```

## 二、数据通路（Pipeline）

`topeet-camera-config.dtsi` 通过 `#define CAMERA_Jx` 选择接口，每个接口构成一条完整 pipeline：

```txt
Sensor ──► MIPI DPHY/DCPHY ──► mipiN_csi2 ──► rkcif_mipi_lvdsN ──► rkcif_..._sditf ──► rkispN_vir0 ──► (rkispp)
```

| 接口 | Sensor (i2c)        | PHY 节点        | CSI2 节点    | RKCIF 节点            | SDITF 节点                  | ISP 节点       | PHY 类型 |
|------|---------------------|-----------------|--------------|-----------------------|-----------------------------|----------------|----------|
| J1 ✅ | ov5695/ov13850/imx415 (i2c4) | csi2_dphy3      | mipi4_csi2   | rkcif_mipi_lvds4      | rkcif_mipi_lvds4_sditf      | rkisp0_vir0    | D-PHY    |
| J2   | ov5695/ov13850/imx415 (i2c7) | csi2_dcphy1     | mipi1_csi2   | rkcif_mipi_lvds1      | rkcif_mipi_lvds1_sditf      | rkisp1_vir0    | C-PHY    |
| J3   | ov5695/ov13850/imx415 (i2c3) | csi2_dphy0      | mipi2_csi2   | rkcif_mipi_lvds2      | rkcif_mipi_lvds2_sditf      | rkisp0_vir0    | D-PHY    |
| J4   | ov5695/ov13850/imx415 (i2c2) | csi2_dcphy0     | mipi0_csi2   | rkcif_mipi_lvds       | rkcif_mipi_lvds_sditf       | rkisp1_vir0    | C-PHY    |

> ✅ = 默认启用（`#define CAMERA_J1`）。J2/J3/J4 默认注释掉，按需打开。

### 各级职责

- **Sensor** (`drivers/media/i2c/`)：输出 RAW MIPI 数据。
- **DPHY/DCPHY** (`drivers/phy/rockchip/`)：MIPI 物理层，D-PHY（csi2_dphyN）走传统 MIPI CSI-2；C-PHY（csi2_dcphyN）走 D-PHY+ 组合。
- **mipiN_csi2** (`cif/mipi-csi2.c`)：CSI2 接收子设备，解析 MIPI 包。
- **rkcif_mipi_lvdsN** (`cif/`)：RKCIF 捕获，DMA 写入 DDR，输出 `/dev/videoN`，或经 SDITF 送往 ISP。
- **sditf** (`cif/subdev-itf.c`)：CIF → ISP 的同步数据接口（Self-Defined Interface）。
- **rkispN_vir0** (`isp/`)：ISP 虚拟通道输入，做去马赛克/3A/缩放/降噪，输出 `/dev/videoN`。
- **rkispp** (`ispp/stream.c`)：可选的后处理（FEC 去畸变、多帧降噪等）。


## 三、阅读顺序
1. [Sensor 驱动](drivers/media/i2c/ov5695.c)
2. [DTS 连接关系](arch/arm64/boot/dts/rockchip/topeet-camera-config.dtsi)
   phy、csi设备树:只看 csi2_dphy3、mipi4_csi2、rkcif_mipi_lvds、rkisp0_vir0 这几个节点的 compatible 和 reg，知道每级对应哪个驱动.[phy](arch/arm64/boot/dts/rockchip/rk3588s.dtsi)
3. MIPI D-PHY（物理层）:
   1. [DPHY 框架层](drivers/phy/rockchip/phy-rockchip-csi2-dphy.c)
   2. [硬件寄存器操作](drivers/phy/rockchip/phy-rockchip-csi2-dphy-hw.c)
   3. [公共结构体](drivers/phy/rockchip/phy-rockchip-csi2-dphy-common.h)
4. CSI2 接收:
   1. [CSI2 一个文件注册了两个驱动(逻辑层和硬件层)](drivers/media/platform/rockchip/cif/mipi-csi2.c)
   2. [结构体 csi2_device](drivers/media/platform/rockchip/cif/mipi-csi2.h)
5. RKCIF 捕获（DMA 入 DDR）:
   1. [CIF 平台驱动入口、media device 注册](drivers/media/platform/rockchip/cif/dev.c)
   2. [配置 CIF 输入格式、DMA 地址、帧中断](drivers/media/platform/rockchip/cif/hw.c)
   3. [视频捕获节点、vb2_ops、start_streaming、中断处理](drivers/media/platform/rockchip/cif/capture.c)
   4. [SDITF：CIF → ISP 的同步接口](drivers/media/platform/rockchip/cif/subdev-itf.c)
6. RKISP 图像处理：
   1. [ISP 平台驱动入口](drivers/media/platform/rockchip/isp/rkisp.c)
   2. [接收来自 sditf 的数据](drivers/media/platform/rockchip/isp/csi.c)
   3. [ISP 内部 bridge，输入路径选择](drivers/media/platform/rockchip/isp/bridge.c)
   4. [ISP 输出节点](drivers/media/platform/rockchip/isp/capture.c)
   5. [3A 参数下发接口（用户态 → ISP）](drivers/media/platform/rockchip/isp/isp_params.c)
   6. [3A 统计上报（ISP → 用户态）](drivers/media/platform/rockchip/isp/isp_stats.c)
7. RKISPP 后处理：
   1. [SPP 视频流：输入来自 ISP，输出经 FEC/降噪后再出 video 节点](drivers/media/platform/rockchip/ispp/stream.c)
   2. [ISPP 平台驱动入口](drivers/media/platform/rockchip/ispp/dev.c)
```txt
1. spec.md                          ← 整体拓扑
2. rk-camera-module.h               ← ABI 约定
3. ov5695.c                         ← sensor 怎么发 MIPI
4. topeet-camera-config.dtsi (J1)   ← 怎么连起来
5. phy-rockchip-csi2-dphy.c/hw.c    ← 物理层
6. cif/mipi-csi2.c                  ← CSI2 接收
7. cif/dev.c + capture.c            ← DMA 捕获
8. cif/subdev-itf.c                 ← CIF→ISP 接口
9. isp/rkisp.c + csi.c + capture.c  ← ISP 处理与输出
10. isp/isp_params.c + isp_stats.c   ← 3A 接口
11. ispp/stream.c                   ← 后处理（进阶）
```
