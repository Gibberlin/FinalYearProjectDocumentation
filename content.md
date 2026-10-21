# Distributed Edge Artificial Intelligence

This file is the working content outline for the report. Draft project-specific
text here, then transfer approved material into
`distributed_edge_ai_report.tex`. Do not present placeholder values as
measured results.

## Details to confirm

- **Project title:** Distributed Edge Artificial Intelligence
- **Degree:** Bachelor of Technology
- **Department:** Computer Science & Engineering
- **Institution:** Barak Valley Engineering College
- **Supervisor:** Mr. Rana Singha
- **Supervisor designation:** [Add designation]
- **Academic session:** [Add session]
- **Submission date:** [Add month and year]
- **Head of department:** [Add name]
- **Recommendation:** [Add approval recommendation]

### Group members

| No. | Name | Roll number |
| --- | --- | --- |
| 1 | Anirbar Saikia | [Add roll number] |
| 2 | Avirup Roy | [Add roll number] |
| 3 | Hisbul Bahar Mazumdar | [Add roll number] |
| 4 | Ritesh Das | [Add roll number] |
| 5 | Syed Yashin Hussain | [Add roll number] |

## Abstract

Distributed Edge Artificial Intelligence (Edge AI) places artificial
intelligence workloads across devices near data sources, edge servers, and
cloud resources. This report presents a preliminary literature study of how
such workloads can be distributed while accounting for communication,
computing capacity, energy use, and privacy. A thematic review examines
end-edge-cloud architectures, federated learning, collaborative inference,
communication efficiency, resource heterogeneity, and security
considerations. The reviewed work indicates that distributing computation
closer to data can support responsive applications and reduce dependence on
centralised processing, but the benefits depend on task placement, network
conditions, and device capabilities. Federated training and partitioned
inference are complementary approaches, each with trade-offs in
communication, coordination, and resource requirements. This project remains
at the literature-review and feasibility-assessment stage; no implementation
or experimental measurements are claimed. Future work will select an
application and platform, then evaluate an appropriate distributed-AI design
using reproducible performance and energy measures.

## Acknowledgement

We would like to extend our sincere thanks to all those who helped us
prepare this report. Without their guidance, support, cooperation, and
encouragement, we would not have been able to complete it successfully. We
express our sincere gratitude to our project guide,
Mr. Rana Singha, for valuable guidance, support, and encouragement
throughout this work. We also thank the faculty members of the Department of
Computer Science & Engineering, Barak Valley Engineering College, for their
suggestions and academic support.

## Chapter 1: Introduction

### 1.1 Background

[Introduce distributed edge AI and the context for this project.]

### 1.2 Problem statement

[Describe the specific problem, application context, constraints, and gap.]

### 1.3 Objectives

1. [Objective 1]
2. [Objective 2]
3. [Objective 3]

### 1.4 Scope and report organization

[Define what the project includes and briefly describe each chapter.]

## Chapter 2: Literature Survey

### 2.1 From cloud AI to edge intelligence

The reviewed literature positions distributed edge AI as a response to the
limits of centralised cloud processing. Edge AI places AI computation near
users at the network edge rather than in a centralised data centre; the shift
is enabled by improvements in AI efficiency, the growth of IoT devices, and
the development of edge computing. Zhou et al. describe edge intelligence
as extending AI to the "last mile" for the many mobile and IoT devices
generating data at the edge. Deng et al. distinguish AI for edge from AI
on edge, the latter covering model training and inference on edge devices
([Deng et al., 2019](https://arxiv.org/abs/1909.00560)). The term "edge"
is not universally defined; its relation to cloud computing depends on
proximity, capacity, and device heterogeneity.

### 2.2 End-edge-cloud architectures

A common architecture distributes processing across end devices, edge
servers, and cloud resources. End devices perform lightweight local
computation, while edge servers offer nearby, more capable resources.
The literature distinguishes AI-for-edge and AI-on-edge and describes
orchestration of heterogeneous on-device, edge, and cloud resources for
distributed AI. Collaborative learning across cloud and edge resources is
also studied ([Deng et al., 2019](https://arxiv.org/abs/1909.00560)).
Hu et al. examine distributed edge-AI task placement across heterogeneous
resources ([Hu et al., 2023](https://doi.org/10.1145/3603166.3632546)).

### 2.3 Distributed training: federated learning

Federated learning (FL) trains local models on multiple clients without
collecting raw user data at a central data centre. Federated learning
and knowledge distillation are described as training approaches, while
model splitting and partitioning support inference. FL challenges
include statistical heterogeneity, communication cost, resource allocation,
and privacy/security; communication can dominate computation when local
datasets are small and devices are comparatively capable
([Deng et al., 2019](https://arxiv.org/abs/1909.00560)).

Communication-efficient methods include probabilistic client selection and
quantisation, with a cited study reporting up to 3.6% improvement in
identification accuracy and 87% improvement in convergence time over
standard FL. Other reviewed approaches address complexity reduction,
resource scheduling, and over-the-air computation, or combine low-rank
tensor models with over-the-air computation for resource-constrained devices.

### 2.4 Distributed inference and partitioning

Research surveys inference across end devices, edge servers, and cloud
resources, as well as training across multiple edge devices. Inference
placement is sensitive to bandwidth: one emulated setup reports more than
2 seconds on a constrained mobile device, and edge-server inference changing
from 0.123 seconds at 1 Mbps to 2.317 seconds at 50 kbps.

Model-partition methods split a deep neural network between a device and
server based on transmission latency and device energy. Early-exit methods
can reduce communication when intermediate features are sufficient.
DistrEdge applies deep reinforcement learning to distribute CNN inference
across heterogeneous IoT edge devices and reports a 1.1--3x speedup over
the compared methods ([Hou et al., 2022](https://arxiv.org/abs/2202.01699)).
Other work studies layer-based and horizontal
partitioning, joint multi-user partitioning and resource allocation, mathematical frameworks for partitioning, and classifications
of DNN partitioning methods.

### 2.5 Communication efficiency

Communication is a central bottleneck for both training and inference.
What is transmitted affects per-round overhead and the number of
communication rounds. Quantisation and pruning, along with in-network
computation, are discussed as ways to address communication cost.
Computation-intensive inference may require task splitting across edge
devices and servers, which makes communication efficiency especially
important.

### 2.6 Heterogeneity, non-IID data, and stragglers

FL over edge nodes must address differences between devices, non-independent
and identically distributed (non-IID) data, and limited communication
resources. Synchronous FL can wait for the slowest worker, while asynchronous
FL may cause frequent model transfers. The slow-worker or "straggler"
effect can limit synchronous training time. Non-IID and imbalanced data
can cause local updates to drift, instability, and slower convergence;
system heterogeneity can also lead to stragglers and dropouts.
Participant selection is one proposed mitigation, while edge-assisted
architectures, data augmentation, and mediator-based rescheduling are
reported as approaches to non-IID bias.

### 2.7 Privacy and security

FL does not eliminate privacy risks: shared models can reveal sensitive
attributes, and systems can be vulnerable to data or model poisoning.
One example in the reviewed literature reports inferring whether an
individual's input appeared in a gender-classifier training set with up to
90% accuracy.
Other surveyed threats include honest-but-curious servers, free-riding,
poisoning, and model inversion. Inference attacks may recover training
samples; differential privacy is one mitigation, with a possible trade-off
against model performance. FL threat surveys classify poisoning and
inference attacks, and identify communication blockage, poisoning, and
backdoor attacks among high-priority threats. Privacy-preserving
cloud-edge-end architectures include encrypted or anonymised protocols and
multi-centre hierarchical FL. Blockchain-based FL is also discussed as
a way to remove a central aggregator.

### 2.8 Energy and hardware constraints

Latency, energy efficiency, and computation or memory footprint are
important evaluation metrics; latency-sensitive applications may have
constraints around 100 ms. Energy-efficient edge training is a
motivation in the literature; one example uses AlphaGo's reported
requirement of 280 GPUs and a $3,000 electricity cost per game to illustrate
the cost of computation. Reported hardware results vary by
platform and workload: Raspberry Pi 5 MobileNetV3-Small measurements include
2.14--2.17 ms latency, about 2.55 W net power, and 5.5 mJ per inference;
reported ResNet-18 measurements include 12.9--14.1 ms, 2.4--2.8 W, and
31--39 mJ per inference. Other reported results include up to 10 W for a
Jetson Nano GPU configuration, about 5 W for a Google Coral TPU, and about
2 W for an Intel NCS2 accelerator. Reported values for Arduino Nano,
Raspberry Pi, and idle Jetson Nano are 0.17 W, 3.5 W, and 1.3--1.6 W,
respectively. Another comparison reports a Jetson Nano about three
times faster and 1.67 times lower in energy than a Raspberry Pi for its
network configuration.

These measurements use different platforms, workloads, and protocols, so
they are indicative rather than directly comparable.

### 2.9 Application domains

Reviewed applications include autonomous and connected vehicles, smart
homes, industrial automation, healthcare, surveillance, real-time gaming,
smart factories, and precision-agriculture pest detection.

## Chapter 3: Methodology

The project is currently at the learning and literature-review stage, and
the application has not yet been selected. The methods in this chapter are
therefore proposed, not completed implementation or experiments.

### 3.1 Review protocol

The preliminary study follows a structured literature-review approach.
Research questions are modelled on existing reviews of computing, storage,
and communication technologies in distributed edge AI and optimisation
along the edge-fog-cloud data path. The approach also draws on
systematic reviews that develop taxonomies of edge AI systems and
examine cloud-edge AI integration
([Deng et al., 2019](https://arxiv.org/abs/1909.00560);
[Hu et al., 2023](https://doi.org/10.1145/3603166.3632546)). Sources are selected from academic
databases for relevance to architecture, training, inference, communication,
heterogeneity, privacy/security, energy, and applications.

### 3.2 Analytical framework

Sources are synthesised across eight themes: definition and motivation;
architecture; training; inference; communication; heterogeneity;
privacy/security; and energy/hardware. This structure is intended to help
map a later application choice to the relevant literature.

### 3.3 Proposed feasibility-assessment framework

A multi-criteria assessment is proposed using latency, energy efficiency,
and computation/memory footprint, extended with bandwidth sensitivity, scalability, and cost. Candidate platforms could be compared with a
reproducible benchmarking approach that measures latency, throughput, power,
and energy.

### 3.4 Proposed technical methodology

Two possible implementation directions are proposed for a later phase:

1. **Partitioned collaborative inference:** split a model by layer or
   horizontally across devices and exchange intermediate outputs, following
   published partitioning schemes.
2. **Federated training:** participants train local models and share model
   updates through an aggregation server, as surveyed for mobile edge
   networks and edge training.

Neither direction has been implemented; both remain proposals for subsequent
work.

## Chapter 4: Results and Discussion

As this is a preliminary, literature-stage study, the results are the
outcomes of the review and feasibility synthesis, not experimental findings.

### 4.1 Literature synthesis

The reviewed literature converges on layered end-edge-cloud architectures, with federated training and partitioned or collaborative
inference as two major technical families. Communication
efficiency is a recurring bottleneck across both.

### 4.2 Feasibility indications

The literature suggests that distributed edge AI can support
latency-sensitive and privacy-sensitive workloads, while remaining
constrained by communication and energy. In one reported setup, edge-server
inference increased from 0.123 seconds at 1 Mbps to 2.317 seconds at
50 kbps, while a mobile device took more than 2 seconds. The hardware
energy figures above vary substantially by device and workload.

| Platform / workload | Reported metric | Reported value |
| --- | --- | --- |
| Raspberry Pi 5, MobileNetV3-Small | Latency / net power / energy | 2.14--2.17 ms / about 2.55 W / 5.5 mJ per inference |
| Raspberry Pi 5, ResNet-18 | Latency / net power / energy | 12.9--14.1 ms / 2.4--2.8 W / 31--39 mJ per inference |
| NVIDIA Jetson Nano | Peak power | Up to 10 W |
| Google Coral TPU | Power | About 5 W |
| Intel NCS2 plus Raspberry Pi 3B+ | Accelerator / host power | About 2 W / about 3 W |
| Arduino Nano | Power | 0.17 W |
| Raspberry Pi | Power | 3.5 W |
| NVIDIA Jetson Nano | Idle power | 1.3--1.6 W |

Partitioned collaborative inference is reported to achieve 1.1--3x speedups
over the methods compared in one study. A communication-efficient FL
method reports up to 87% improvement in convergence time and up to 3.6%
improvement in accuracy relative to standard FL. These results come
from different hardware, workloads, and measurement protocols and should
not be compared directly.

### 4.3 Discussion and limitations

The synthesis indicates a trade-off between latency and energy on one side
and communication cost on the other. Moving computation closer to the data
may reduce latency and bandwidth pressure, but lower edge computing
capacity and device heterogeneity can require model partitioning,
compression, or collaborative execution. Non-IID data and
slow or unreliable nodes can create stragglers and dropouts.
Privacy and security risks, including inference and poisoning attacks, also
constrain how data and model updates are handled.

No implementation or measurements have been carried out; this feasibility
assessment relies on published results. Those results use different
platforms, workloads, and measurement setups. Even power-measurement
protocols vary; one study reports reducing measurement uncertainty by
factors of 6 and 12 on two device families relative to an established
protocol. Direct cross-study comparison of energy values must
therefore be treated cautiously. The field is still developing, and the
definition of "edge" remains unsettled. Open challenges include
resource constraints, security threats, scalability, and communication
overhead.

## Chapter 5: Conclusion and Future Work

This preliminary report reviews distributed edge AI and assesses feasibility
at a conceptual level. The literature describes a shift from centralised
cloud AI toward layered end-edge-cloud architectures, with
federated learning and collaborative DNN partitioning as major technical
families. Potential suitability for latency- and
privacy-sensitive applications is bounded by communication cost,
device and data heterogeneity, energy constraints, and
privacy and security risks.

Because the application and implementation remain undecided, the next phase
is to select a specific application and platform using the proposed review
protocol and feasibility framework, then design an implementation for
future evaluation. No implementation or experimental result is claimed here.

## References

These are the source PDFs currently in `references/`; use the online links
for their publication records:

1. Fei Hu, Kunal Mehta, Shivakant Mishra, and Mohammad AlMutawa.
   *Distributed Edge AI Systems*. In Proceedings of the 2023 IEEE/ACM 16th
   International Conference on Utility and Cloud Computing (UCC '23), 2023.
   [DOI](https://doi.org/10.1145/3603166.3632546) ·
   Local file: [distributed-edge-ai-systems-hu-et-al-2023.pdf](references/distributed-edge-ai-systems-hu-et-al-2023.pdf)
2. Shuiguang Deng, Hailiang Zhao, Weijia Fang, Jianwei Yin, Schahram Dustdar,
   and Albert Y. Zomaya. *Edge Intelligence: The Confluence of Edge Computing
   and Artificial Intelligence*. arXiv:1909.00560, 2019.
   [arXiv record](https://arxiv.org/abs/1909.00560) ·
   Local file: [edge-intelligence-deng-et-al-2019.pdf](references/edge-intelligence-deng-et-al-2019.pdf)
3. Xueyu Hou, Yongjie Guan, Tao Han, and Ning Zhang. *DistrEdge: Speeding up
   Convolutional Neural Network Inference on Distributed Edge Devices*.
   arXiv:2202.01699, 2022.
   [arXiv record](https://arxiv.org/abs/2202.01699) ·
   Local file: [distr-edge-hou-et-al-2022.pdf](references/distr-edge-hou-et-al-2022.pdf)

Add verified entries and links here as further source documents are added
to `references/`.
