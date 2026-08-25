# Cross-Datacenter AI Training Testbed

> An end-to-end platform for understanding—and reducing—the communication bottleneck that appears when distributed GPU training crosses a dynamic wide-area network.

## Start with the basic problem

Large AI models are commonly trained across multiple GPUs. The GPUs compute locally, but repeatedly exchange parameters and gradients. When workers are split across datacenters, these collective transfers traverse a WAN whose bandwidth, delay, congestion, and available paths change over time.

This testbed connects the complete chain—from PyTorch FSDP tensors, through NCCL/MSCCL and GPUDirect RDMA, to a programmable multipath WAN—so communication decisions and network conditions can be controlled, observed, and evaluated together.

## 1. Why training depends on collective communication

**PyTorch Fully Sharded Data Parallel (FSDP)** divides model state across workers. Before computation, **All-Gather** reconstructs the required parameter on every worker. During backpropagation, **Reduce-Scatter** aggregates gradients and returns the appropriate reduced shard.

<p align="center"><img src="figures/fsdp-all-gather.png" width="680" alt="All-Gather reconstructs the complete parameter on every GPU"></p>

<p align="center"><img src="figures/fsdp-reduce-scatter.png" width="680" alt="Reduce-Scatter aggregates gradients and returns one shard to each GPU"></p>

```text
All-Gather → forward compute → backward compute → Reduce-Scatter → next step
```

These operations move large tensors on every step. Slow communication leaves GPUs waiting and directly increases training-step time.

## 2. Why crossing datacenters changes the problem

Across datacenters, collective transfers encounter lower bandwidth, longer delay, competing traffic, congestion, and heterogeneous paths. The testbed places four GPU workers across two logical datacenters and connects them through a programmable multipath WAN. Path capacity, propagation delay, queues, and background load can be controlled and replayed.

<p align="center"><img src="figures/cross-dc-testbed-overview.svg" width="720" alt="Four GPU workers connected through a programmable multipath WAN"></p>

## 3. Closing the loop across two layers

The collective layer decides which data is ready, where it moves, and which operations depend on earlier transfers. The WAN layer decides which path carries each flow and how much usable capacity that path provides.

<p align="center"><img src="figures/ccl-wan-control-timescales.png" width="720" alt="Collective schedules and WAN paths adapt at different timescales"></p>

- **Fast WAN adaptation** steers ready RDMA traffic around short-lived congestion.
- **Slower schedule adaptation** changes transfer ordering and channel assignment when network conditions persist.

Neither layer is sufficient alone: a good path cannot accelerate data that the schedule has not released, while a parallel schedule cannot overcome a persistently congested path.

Agents observe data-plane interfaces across the WAN and report synchronized per-link telemetry to a collector.

<p align="center"><img src="figures/wan-telemetry-architecture.png" width="620" alt="Network telemetry collection across the WAN data plane"></p>

```text
FSDP workload → collective schedule → RDMA transfers → WAN paths
       ↑                                             ↓
       └──── analysis and adaptation ← telemetry ───┘
```

## 4. From an FSDP tensor to WAN transfers

With four workers, each rank owns one quarter of the global payload. The collective schedule divides each shard into subchunks and determines when, where, and over which channel each subchunk moves.

<p align="center"><img src="figures/fsdp-workload-granularity.png" width="500" alt="How the global All-Gather payload becomes scheduled subchunks"></p>

- **FSDP** determines which distributed model data is required.
- **NCCL/MSCCL** organizes collective transfers and dependencies.
- **RoCEv2 RDMA** carries data between GPU nodes.
- The **WAN** supplies paths and time-varying capacity.

## 5. End-to-end GPU-to-GPU data path

GPU memory is registered for RDMA so the RNIC can move collective payloads directly. The WAN remains a transit network rather than a training endpoint.

<p align="center"><img src="figures/rocev2-gpudirect-data-path.png" width="760" alt="End-to-end RoCEv2 and GPUDirect RDMA data path"></p>

```text
GPU memory → NCCL/MSCCL → GPUDirect RDMA → local RNIC
           → programmable WAN → remote RNIC → remote GPU memory
```

Runtime evidence verifies that NCCL selected direct GPU-memory access instead of staging the payload through host memory.

<p align="center"><img src="figures/nccl-gpudirect-runtime.png" width="760" alt="Runtime evidence that NCCL selected GPUDirect RDMA"></p>

## 6. Why UDP destination port 4791 is special

RoCEv2 conventionally uses UDP destination port **4791**. A GPU endpoint should recognize and terminate the RDMA transport. A WAN node must instead forward the packet as transit IP traffic without consuming it as a local RDMA endpoint.

<p align="center"><img src="figures/udp4791-transit-forwarding.png" width="700" alt="Making RoCEv2 UDP 4791 forwardable through the WAN node"></p>

GPU-node RNICs therefore retain RoCEv2 endpoint behavior, while WAN-facing transit ports preserve and forward the original end-to-end packet.

## 7. Why collective dependencies matter

A collective schedule is a dependency graph, not merely a list of messages. Deep dependency chains hide parallelism and can leave WAN capacity idle. Restructuring All-Gather shortens the critical chain and releases useful transfers earlier.

<p align="center"><img src="figures/all-gather-dependency-levels.png" width="700" alt="Transfer release across collective dependency levels"></p>

The central insight is: **available WAN capacity helps only when the collective schedule has a ready transfer that can use it**.

## 8. Experimental results

A controlled interaction study separates two contributions: adaptive path steering and a low-dependency, channel-balanced All-Gather schedule. All configurations use the same workload and comparable WAN conditions.

<p align="center"><img src="figures/final-interaction-results.png" width="760" alt="All-Gather latency across path and schedule configurations"></p>

| Configuration | All-Gather latency | Training-step time | AG reduction | Step-time reduction |
|---|---:|---:|---:|---:|
| Current schedule + baseline paths | 3275.8 ms | 9575.6 ms | — | — |
| Current schedule + adaptive paths | 2785.4 ms | 8488.9 ms | 15.0% | 11.3% |
| Low-dependency schedule + baseline paths | 2351.4 ms | 8568.9 ms | 28.2% | 10.5% |
| Low-dependency schedule + adaptive paths | **2135.0 ms** | **7906.3 ms** | **34.8%** | **17.4%** |

### What the results mean

- **Path steering removes transient network waste:** with the original schedule unchanged, it reduces All-Gather latency by 15.0%.
- **Schedule redesign exposes communication parallelism:** with baseline paths unchanged, it reduces All-Gather latency by 28.2%.
- **The mechanisms are complementary:** combining both reduces All-Gather latency by 34.8% and complete training-step time by 17.4%.
- **The gain does not come from sending less model data:** the redesigned schedule preserves approximately the same cross-datacenter traffic volume.

The benefit persists as the global All-Gather payload grows, rather than appearing at only one tensor size.

<p align="center"><img src="figures/payload-sensitivity.png" width="620" alt="All-Gather latency across payload sizes"></p>

The system-level conclusion is that optimizing either the collective schedule or the WAN alone leaves performance on the table; coordinating both layers shortens the communication critical path and translates that gain into faster training steps.

## 9. Repeatable experimental workflow

1. Configure a distributed FSDP workload.
2. Define the WAN topology and replayable conditions.
3. Run the real NCCL/MSCCL and RoCEv2 path.
4. Collect synchronized training and network telemetry.
5. Verify executed paths, payload, and collective configuration.
6. Compare configurations under the same workload and WAN condition.
7. Trace the remaining bottleneck across dependencies and network behavior.

Controlled replay and end-to-end validation prevent a faster run from being mistaken for a better design when it merely encountered a different traffic window.

## Technical scope

| Area | Technologies and concepts |
|---|---|
| Distributed AI training | PyTorch FSDP, multi-node GPU workers |
| Collective communication | All-Gather, Reduce-Scatter, NCCL, MSCCL schedules |
| High-performance networking | RoCEv2, RDMA, GPUDirect RDMA, RNICs |
| WAN experimentation | Multipath forwarding, bandwidth, delay, queues, background traffic |
| Systems analysis | Cross-layer telemetry, dependency analysis, controlled validation |

## Repository scope

This repository is a **code-free technical presentation** of the testbed architecture, mechanisms, and validated findings. It excludes source code, private infrastructure details, raw logs, manuscripts, and organization-specific information.

---

**Keywords:** Distributed AI Training · PyTorch FSDP · NCCL · MSCCL · All-Gather · Reduce-Scatter · GPUDirect RDMA · RoCEv2 · UDP 4791 · Programmable WAN
