# Images

Store report figures, diagrams, and other image assets in this folder. Use
descriptive filenames and add captions and source/permission details in the
report when applicable.

The supplied college logo is `bvec_logo.png`; the LaTeX report uses it on the
cover and certificate pages.

The report uses a monochrome visual style. The vehicle-specific overview,
vehicle--edge--cloud collaboration diagram, sensor-to-perception pipeline,
feasibility workflow, and proposed system architecture are drawn inline with
TikZ in `distributed_edge_ai_report.tex`. The remaining image files used by
the report are `six-phase-research-plan.png` and
`distributed-video-object-detection.png`, with their sources identified in
the report.

Earlier generic architecture diagrams and feasibility artwork are retained
as historical assets but are no longer used in the report. The main
architecture remains a proposed design, not a depiction of an implemented
system.

The supplementary source review also uses `guo2023-edge-av-architecture.jpg`
and `guo2023-emergency-braking-speed.jpg`, reproducing Figures 3 and 11 from
Guo Ming, “Exploration of the Intelligent Control System of Autonomous
Vehicles Based on Edge Computing,” *PLOS ONE* 18(2), e0281294 (2023),
https://doi.org/10.1371/journal.pone.0281294. Both figures are reproduced
under the Creative Commons Attribution 4.0 International license (CC BY 4.0);
their report captions identify the source and license. The CoEdge energy
savings chart is redrawn inline in the report from reported values in Zeng
et al., “CoEdge: Cooperative DNN Inference With Adaptive Workload
Partitioning Over Heterogeneous Edge Devices,” https://doi.org/10.1109/TNET.2020.3042320;
the source figure itself is not reproduced.

The `carla_*.png` figures show project-generated CARLA outputs: ground-truth
label projections, demo frames, a recorded camera triplet, steering
diagnostics, detector timing, and steering-training results. They are
included in `distributed_edge_ai_report.tex` as our preliminary results,
not as figures reproduced from the cited research papers.

The `kitti_*.png` figures show project-generated KITTI vehicle motion analysis
and edge control results authored by Avirup Roy: real-world sensor profiles,
loss convergence, multi-target distributions, 2D perception dashboard, and 4-perspective
3D simulation with live HUD. They represent physical benchmark measurements and
closed-loop trajectory evaluations.
