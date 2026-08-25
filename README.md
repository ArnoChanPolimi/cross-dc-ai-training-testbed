# Cross-Datacenter AI Training Testbed

> An end-to-end experimental platform for understanding how distributed GPU training behaves when collective communication crosses a dynamic wide-area network.

## Start with the basic problem

A large AI model may not fit, or train efficiently, on a single GPU. Distributed training solves this by splitting the work across multiple GPU workers.

The GPUs do not work independently. During every training step, they must repeatedly exchange model parameters and gradients. If that exchange is slow, the GPUs wait—even when their compute cores are otherwise ready.

This project studies that communication bottleneck when the GPU workers are separated across datacenters.

## 1. Why distributed training needs collective communication

The testbed uses **PyTorch Fully Sharded Data Parallel (FSDP)**. FSDP divides model state across workers so that each GPU stores only a shard instead of a complete copy.

Before a GPU can compute with a sharded parameter, the workers perform an **All-Gather**. Every worker contributes its local shard, and every worker receives the complete parameter needed for computation.

![All-Gather reconstructs the complete parameter on every GPU](figures/fsdp-all-gather.png)

During the backward pass, workers produce gradients for the same model parameters. **Reduce-Scatter** aggregates those gradients and returns the appropriate reduced shard to each worker.

![Reduce-Scatter aggregates gradients and returns one shard to each GPU](figures/fsdp-reduce-scatter.png)

These are not occasional control messages. They move large tensors and sit directly on the critical path of training:

```text
All-Gather → forward compute → backward compute → Reduce-Scatter → next step
```

The communication system therefore affects both GPU utilization and the time required to complete every training step.

## 2. Why crossing datacenters changes the problem

Inside one datacenter, GPU communication normally uses a fast and relatively predictable network fabric. Across datacenters, the same collective transfers encounter a WAN with:

- lower and time-varying available bandwidth;
- longer propagation delay;
- competing background traffic;
- queueing and congestion feedback;
- multiple paths with different conditions.

The testbed places four GPU workers across two logical datacenters and connects them through a programmable multipath WAN.

![Four GPU workers connected through a programmable multipath WAN](figures/cross-dc-testbed-overview.svg)

The WAN is not treated as a black box. Its paths, bandwidth, delay, queues, and background load can be controlled, allowing the same network condition to be replayed and compared across experiments.

## 3. From an FSDP tensor to WAN transfers

FSDP starts with a global model payload. With four workers, each rank owns one quarter of that payload. The collective schedule divides each shard into smaller subchunks and decides how those subchunks move between workers and across WAN paths.

![How the global All-Gather payload becomes scheduled subchunks](figures/fsdp-workload-granularity.png)

This distinction matters:

- **FSDP** determines what distributed model data is needed;
- **NCCL/MSCCL** determines how collective transfers are organized;
- **RoCEv2 RDMA** carries the data between GPU nodes;
- the **WAN** determines what network capacity and paths are available to those transfers.

The project connects these layers into one observable system instead of evaluating them in isolation.

## 4. The end-to-end GPU-to-GPU data path

The training workers use RDMA-capable network interfaces. GPU memory is registered for RDMA, allowing the RNIC to transfer collective payloads along the data path without treating the WAN emulator as a training endpoint.

![End-to-end RoCEv2 and GPUDirect RDMA data path](figures/rocev2-gpudirect-data-path.png)

The complete data path is:

```text
GPU memory
   ↓
NCCL / MSCCL collective channel
   ↓
GPUDirect RDMA and the local RNIC
   ↓
RoCEv2 over the IP network
   ↓
programmable OVS/Mininet WAN
   ↓
remote RNIC and remote GPU memory
```

This is why the project is an **end-to-end distributed AI systems testbed**, rather than only a network simulation or a standalone collective benchmark.

The data path is also checked at runtime. With GPUDirect RDMA enabled, NCCL selects direct GPU-memory access through the RNIC instead of staging the payload through host memory.

![Runtime evidence that NCCL selected GPUDirect RDMA](figures/nccl-gpudirect-runtime.png)

## 5. Why UDP destination port 4791 is special

RoCEv2 encapsulates RDMA transport in UDP/IP and conventionally uses **UDP destination port 4791**.

That port has special meaning to an RDMA-capable NIC. At a GPU endpoint, this is exactly what is needed: the RNIC recognizes the packet as RoCEv2 and processes the RDMA transport.

The WAN node is different. It must act as an IP transit device, not as the final RDMA endpoint. If a WAN-facing NIC consumes UDP/4791 locally as RoCEv2 traffic, the packet cannot traverse the emulated WAN correctly.

![Making RoCEv2 UDP 4791 forwardable through the WAN node](figures/udp4791-transit-forwarding.png)

The testbed therefore separates the two roles:

- GPU-node RNICs retain RoCEv2 endpoint behavior;
- WAN-facing transit ports treat the traffic as forwardable IP/UDP;
- the original end-to-end RoCEv2 packet remains intact between GPU workers.

This detail is essential: the middle of the network must forward RoCEv2 traffic without terminating the RDMA connection.

## 6. Two layers shape the same collective transfer

Once RoCEv2 traffic can cross the WAN, performance still depends on decisions made at two different layers.

![Collective schedules evolve more slowly than WAN path configurations](figures/ccl-wan-control-timescales.png)

### Collective-communication layer

An MSCCL schedule determines:

- which rank sends each subchunk;
- which rank receives it;
- which communication channel carries it;
- which transfers must wait for earlier transfers;
- when cross-datacenter traffic becomes ready for transmission.

### WAN layer

The WAN configuration determines:

- which physical path carries a communication channel;
- the capacity and delay of that path;
- how background traffic competes for the link;
- how queueing and congestion affect the RDMA flow.

WAN paths can change quickly, while generating and safely activating a different collective schedule is a slower operation. The testbed exposes both layers so their interaction can be observed under controlled conditions.

## 7. Why collective dependencies matter

A collective schedule is not merely a list of messages. Transfers are connected by ordering constraints. A later transfer may not begin until the data it needs has arrived through earlier operations.

A deep dependency chain can create two problems:

1. later WAN transfers remain unavailable even when the network has capacity;
2. a long tail of intra-datacenter forwarding can delay collective completion after cross-datacenter transfers have finished.

The following figure groups transfers by dependency level. It visualizes how changing the schedule structure can shorten the dependency chain and expose useful transfers earlier, without describing the schedule as a single opaque XML file.

![Transfer release across collective dependency levels](figures/all-gather-dependency-levels.png)

This is the key cross-layer insight: **available WAN capacity is useful only when the collective schedule has a ready transfer that can use it**.

## 8. What the testbed makes observable

The platform brings together measurements from training, collective communication, RDMA, and the WAN:

- training-step and collective timing;
- collective payload and subchunk configuration;
- active communication channels;
- selected WAN paths;
- per-interface transmitted and received rates;
- queue, drop, and congestion indicators;
- background load and available path capacity.

This allows an observed slowdown to be traced across layers instead of being attributed vaguely to “the network” or “the GPUs.”

![Network telemetry collection across the WAN data plane](figures/wan-telemetry-architecture.png)

## 9. A repeatable experimental workflow

The high-level workflow is:

1. configure a distributed FSDP workload;
2. define the WAN topology and replayable network conditions;
3. run the real NCCL/MSCCL and RoCEv2 communication path;
4. collect synchronized training and network telemetry;
5. verify that the intended path and communication configuration executed;
6. compare configurations under the same workload and WAN condition;
7. use the evidence to identify the next bottleneck.

Repeatability is a core requirement. Without controlled replay and end-to-end observation, a faster run could simply be the result of a different background-traffic window rather than a better system design.

## Technical scope

| Area | Technologies and concepts |
|---|---|
| Distributed AI training | PyTorch FSDP, multi-node GPU workers |
| Collective communication | All-Gather, Reduce-Scatter, NCCL, MSCCL schedules |
| High-performance networking | RoCEv2, RDMA, GPUDirect RDMA, RNICs |
| WAN experimentation | Multipath forwarding, bandwidth, delay, queues, background traffic |
| Systems analysis | Cross-layer telemetry, dependency analysis, repeatable validation |

## Repository scope

This repository is a **code-free technical presentation** of the testbed architecture and the systems problems it studies.

It intentionally excludes source code, private infrastructure details, raw experiment logs, unpublished manuscripts, and organization-specific information.

---

**Keywords:** Distributed AI Training · PyTorch FSDP · NCCL · MSCCL · All-Gather · Reduce-Scatter · GPUDirect RDMA · RoCEv2 · UDP 4791 · Programmable WAN
