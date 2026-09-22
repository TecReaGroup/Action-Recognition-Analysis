# Human Pose Estimation

**人体骨架关键点（Human Pose Estimation, HPE）**已经从「检测几个关节点」变成人体感知的基础设施：健身/康复、动作识别、虚拟人驱动、AIGC ControlNet、机器人、监控与运动分析都依赖它。下面按**范式 → 代表算法 → 业界落地 SOTA → 对比表**做深挖。数字来自论文/官方仓库，**不同论文的 AP 不能直接横比**（val vs test-dev、检测器 AP、输入分辨率、是否用额外数据都会差 1–3 个点）。

---

## 1. 输入输出

**输入**：RGB 图 / 视频帧（少数系统用深度或 IMU）。  
**输出**：每人一组关键点 \((x, y[, z], visibility)\)，再按骨架拓扑连成人体树。

常见骨架协议：

| 协议                             | 点数 | 覆盖                                    |
| -------------------------------- | ---- | --------------------------------------- |
| COCO                             | 17   | 躯干+四肢，业界 2D 标尺                 |
| MPII                             | 16   | 单人、动作多样                          |
| BODY_25（OpenPose）              | 25   | COCO + 脚                               |
| BlazePose                        | 33   | COCO 超集 + 面/手/脚细节，可出 3D world |
| Halpe / BODY_26                  | 26   | 加脚                                    |
| COCO-WholeBody                   | 133  | 17 身 + 6 脚 + 68 脸 + 42 手            |
| Sociopticon / Goliath（Sapiens） | 308  | 超密面部（约 274）+ 手脚                |

主流评测：

- **2D**：COCO AP（OKS，0.50:0.95）、MPII PCKh、CrowdPose / OCHuman（拥挤/遮挡）
- **3D**：Human3.6M MPJPE / PA-MPJPE，3DPW

---

## 2. 三条技术主线

**Top-down（先人后点）**  
先检测人体框，再对每个框估姿态。精度上限高，速度随人数线性变差。HRNet、ViTPose、RTMPose、Sapiens、AlphaPose 都走这条路。

**Bottom-up（先点后人）**  
全图预测所有关键点，再用 grouping（PAF、Associative Embedding 等）把点绑回人。人数多时更稳、时延相对恒定，但小人和遮挡更难。OpenPose、HigherHRNet、部分 Associative Embedding。

**One-stage / 端到端**  
检测与关键点一次前向完成：YOLO-Pose、RTMO、DETRPose。工程上最省事，多人场景延迟不随人数爆炸。

定位头也分三代：

1. **Heatmap**：把关键点当成高斯峰，精度高但上采样贵  
2. **直接回归**：轻、快，早期精度略差（DeepPose、部分 BlazePose）  
3. **SimCC / 分类式坐标**：把 \(x,y\) 当成离散分类再亚像素解码——RTMPose 系列能做到「去掉 heatmap 仍很准」

2022 之后还有两条增量：**ViT 当 backbone 直接吃尺度**、**大规模人体预训练（Sapiens 3 亿～10 亿人体图）**。

---

## 3. 代表性算法（按时间，原理不写实现细节）

### 3.1 奠基期（2016–2019）

**OpenPose（CMU，CVPR 2017 / TPAMI 2019）**  
- 原理：全图同时出关键点热图 + **Part Affinity Fields（肢体向量场）**，用场把属于同一个人的关节点连起来。速度几乎不随人数线性爆炸。  
- 精度：COCO test-dev 约 **61.8 AP**（BODY_18/25 时代数字，已被后来方法大幅超过）。  
- 性能：GPU 实时；人数多时相对 top-down 更稳。  
- 输入：整图 RGB。输出：18/25 身点，可加脸、手，最多约 135 点。  
- 链接：https://github.com/CMU-Perceptual-Computing-Lab/openpose  

**AlphaPose / RMPE（上海交大 MVIG，2017）**  
- 原理：典型 top-down。强调「检测框不准会毁掉姿态」，用对称空间变换等把框对齐后再估点，再加 PoseFlow 做跨帧跟踪。  
- 精度：早期 COCO test-dev **73.3 AP**，当时第一批突破 70+ 的开源系统。  
- 输入：图/视频 + 人体检测器。输出：COCO 17，后续支持 Halpe 26、WholeBody 133，以及 HybrIK 3D mesh。  
- 链接：https://github.com/MVIG-SJTU/AlphaPose  

**SimpleBaseline（ECCV 2018）**  
- 原理：ResNet + 几层反卷积把特征图拉回高分辨率再出 heatmap。证明「简单解码头 + 强 backbone」就够用。后续几乎所有 top-down 都站在这条基线上。  
- 精度：ResNet-152、384×288 大约 **73.7 AP**（COCO test-dev 量级）。  

**HRNet（微软，CVPR 2019）**  
- 原理：**全程保持高分辨率分支**，同时并行低分辨率语义分支并反复融合。不再「先降采样再硬恢复」，所以定位特别稳。  
- 精度：HRNet-W48 + 384×288，COCO 约 **75.5–76.3 AP**（视 val/test 与后处理）。一度长期霸榜。  
- 输入：人体框 crop（top-down）或整图（后来的 HigherHRNet）。输出：heatmap → 17 点。  
- 链接：官方实现已并入 MMPose；原始论文配套代码广泛 fork。https://github.com/open-mmlab/mmpose  

**HigherHRNet（CVPR 2020）**  
- 原理：把 HRNet 做成 bottom-up，多尺度热图解决「小人关键点糊掉」。  
- 精度：COCO val 无多尺度约 **71.4 AP**（W48，640）。拥挤场景常用基线。

### 3.2 端侧与工业实时（2019–2021）

**MediaPipe Pose / BlazePose（Google，约 2019–2020）**  
- 原理：检测器找人 → tracker 在 ROI 上回归 **33 个 landmark**；可出 2D 图像坐标 + 以米为单位的 3D world landmarks，可选分割 mask。为手机/浏览器设计。  
- 精度：官方在 Yoga/Dance/HIIT 私有集上，Heavy 对 17 个 COCO 点约 **68–74 mAP**（与 COCO 官方榜不可直接比）。  
- 性能：手机/浏览器 30+ FPS 是常态。  
- 输入：单人为主的 RGB。输出：33×(x,y,z,visibility)。  
- 链接：https://github.com/google-ai-edge/mediapipe  

**MoveNet Lightning / Thunder（Google，2021）**  
- 原理：MobileNetV2 风格、偏 bottom-up 的超轻量 17 点模型，智能裁剪稳住跟踪。  
- 精度：弱于学术 SOTA，但 Thunder 在端侧够用。  
- 性能：笔记本/手机 50+ FPS。  
- 链接：https://github.com/tensorflow/tfjs-models（pose-detection）

### 3.3 Transformer 与工程 SOTA（2022–2024）

**ViTPose / ViTPose++（NeurIPS 2022 / TPAMI 2024）**  
- 原理：**几乎不做姿态特化**——plain ViT 编码 + 轻量 decoder 出 heatmap。靠规模、MAE 预训练、多数据集联合训练取胜。ViTPose++ 用任务无关/任务相关 FFN 同时吃人、动物、全身点。  
- 精度：单模型 ViTPose-G **80.9 AP**，ensemble **81.1 AP**（COCO test-dev），长期是 2D 精度标杆之一。ViTPose-H val 约 79+。  
- 性能：大模型吃 GPU；小规格（S/B）可部署但不是端侧首选。  
- 输入：检测框 crop，常见 256×192。输出：heatmap → 17/全身/动物关键点。  
- 链接：https://github.com/ViTAE-Transformer/ViTPose  
  论文：https://arxiv.org/abs/2204.12484  

**RTMPose（OpenMMLab / 上海 AI Lab，2023）**  
- 原理：工程向 top-down。CSPNeXt 类轻量骨干 + **SimCC**（把坐标当分类，避开 heatmap 上采样）+ 强训练与 MMDeploy。目标是「COCO 上接近 HRNet，CPU/手机能跑实时」。  
- 精度 / 速度（官方，AIC+COCO 等设置下）：  
  - RTMPose-s：约 **72.2 AP**，骁龙 865 **70+ FPS**  
  - RTMPose-m：约 **75.8 AP**，i7-11700 **90+ FPS**，1660 Ti **430+ FPS**  
  - RTMPose-l 384：约 **77.3–78.3 AP**  
- 输入：人体框 256×192 或 384×288。输出：17 点；衍生 RTMW 出 133 全身点。  
- 链接：https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose  
  轻量推理：https://github.com/Tau-J/rtmlib  
  论文：https://arxiv.org/abs/2303.07399  

**RTMO（MMPose 系列，约 2023）**  
- 原理：把 RTMPose 做成 **one-stage**（无需先检测）。人多（>4）时往往比 top-down 更快，精度略降。  
- 精度：RTMO-l 在多数据集上约 **74.8 AP**（17 点）。  
- 链接：同样在 MMPose / rtmlib。

**DWPose（清华 & IDEA，ICCVW 2023）**  
- 原理：两阶段蒸馏——用大 RTMPose 当老师，把中间特征和「可见+不可见点」的 logits 蒸给学生，再自蒸馏头部。专打 **全身 133 点** 和 ControlNet 骨架条件。  
- 精度：DWPose-l 384×288，COCO-WholeBody **66.5 Whole AP**（当时超过老师 RTMPose-x 的 65.3）。  
- 性能：相对 RTMW 更小更快，生成式管线里极常用。  
- 输入：人体框。输出：133 点。  
- 链接：https://github.com/IDEA-Research/DWPose  
  论文：https://arxiv.org/abs/2307.15880  

**YOLO-Pose（Ultralytics YOLOv8 2023，后续 v11 / YOLO26）**  
- 原理：检测头旁路直接回归每人 17 点 \((x,y,vis)\)，单次前向出框+骨架。  
- 精度：YOLOv8x-pose 640 约 **68.9 AP**（pose 50-95）；YOLOv8x-p6 1280 约 **71.5**。YOLO26x-pose 官方表约 **71.6 / 91.6**（50-95 / 50）。低于 ViTPose/Sapiens，但部署极简单。  
- 性能：从 n 到 x，CPU 到 TensorRT 都有现成导出。  
- 输入：640 整图。输出：每人 box + 17×3。  
- 链接：https://github.com/ultralytics/ultralytics  

### 3.4 基础模型与 2025–2026 前沿

**Sapiens（Meta，ECCV 2024）**  
- 原理：在约 **3 亿野外人体图**上预训练高分辨率 ViT，再微调姿态 / 分割 / 深度 / 法向。姿态仍是 top-down heatmap，但表征极强，泛化（艺术、遮挡、罕见姿势）明显好于只在 COCO 上训的模型。  
- 精度（官方，输入常为 1024×768）：  
  - 17 点：0.3B **79.6** / 0.6B **81.2** / 1B **82.1** / 2B **82.2 AP**  
  - WholeBody 133：2B 约 **74.5 Whole AP**  
  - 308 点 Goliath：1B 约 **63.9 AP**  
- 性能：大、吃显存，适合云端/离线高精度，不是手机方案。  
- 链接：https://github.com/facebookresearch/sapiens  
  论文：https://arxiv.org/abs/2408.12569  

**Sapiens 2（Meta，ICLR 2026）**  
- 原理：预训练扩到约 **10 亿人体图**，分辨率可到 4K，任务扩展到 pointmap、matting。姿态仍为 308 点 top-down。官方称相对 v1 姿态约 **+4 mAP** 量级。规格 0.4B–5B。  
- 链接：https://github.com/facebookresearch/sapiens2  

**DETRPose（2025）**  
- 原理：改 Transformer decoder + 关键点去噪查询，**端到端多人、无 NMS、训练轮次少**。  
- 精度：DETRPose-S 约 **67.0 AP**，对标 YOLOv8x-pose，参数约 11.5M（约为 YOLO-x 的 1/6），延迟约 2.4 ms；L 在 COCO test-dev 约 **71.2 AP** / 4.66 ms。CrowdPose / OCHuman 上对遮挡更稳。  
- 论文：https://arxiv.org/abs/2506.13027  

**学术榜上更新的名字（2025–2026）**  
HAP、CLASP、Hulk、UniHCP、PATH、HRFormer、SOLIDER 等把姿态当成「人体中心多任务基础模型」的一个头，COCO 上报到 **77–78+ AP**。多数代码开放度、工程成熟度不如 ViTPose / MMPose / Sapiens。Wizwand 等聚合站最高看到 HAP（ViT-B）**78.2 mAP** 一档，需自行核对是否用了额外数据或多任务。

### 3.5 3D 骨架（和 2D 关键点是上下游）

2D 点估完后，常见第二条链路是 **lifting**：2D 序列 → 3D 关节。

| 方向             | 代表                    | 核心思路                   | 典型指标                         |
| ---------------- | ----------------------- | -------------------------- | -------------------------------- |
| 时序卷积 lifting | VideoPose3D（2019）     | 只吃 2D 轨迹               | H3.6M MPJPE ~46mm 量级（视协议） |
| Transformer 时序 | PoseFormer / MotionBERT | 时空注意力                 | 约 30–40mm                       |
| 概率/度量 3D     | MeTRAbs                 | 直接回归度量尺度 3D        | 临床/运动分析评测里常排前        |
| 人体网格         | HMR / 4DHumans / SMPL-X | 回归人体参数而不仅是 17 点 | 3DPW PA-MPJPE                    |

2026 综述里已出现 SSM（状态空间）长时序、扩散式 FinePOSE 等，H3.6M P1 报到 **30mm 出头**，但仍受单目深度歧义天花板限制。

---

## 4. 对比表（工程选型用）

AP 一律标明数据集/协议。速度是「量级」，硬件不同会差一倍。

| 模型         | 时间    | 范式              | 核心原理（一句话）        | 准确率（代表性数字）              | 性能量级                              | 输入 → 输出                 | 链接                                                         |
| ------------ | ------- | ----------------- | ------------------------- | --------------------------------- | ------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| OpenPose     | 2017    | Bottom-up         | 热图 + PAF 绑人           | COCO ~61.8 AP                     | GPU 实时，时延对人数不敏感            | 整图 → 18/25/最多135点      | [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |
| AlphaPose    | 2017    | Top-down          | 对齐检测框再估点 + 跟踪   | COCO test-dev ~73.3               | 约 20 FPS 级（当年 Titan，4.6 人/图） | 图+检测器 → 17/26/133       | [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)          |
| HRNet        | 2019    | Top-down          | 始终保持高分辨率表征      | COCO ~75.5–76.3                   | GPU 中速                              | crop → 17 点 heatmap        | [MMPose](https://github.com/open-mmlab/mmpose)               |
| HigherHRNet  | 2020    | Bottom-up         | 多尺度高分辨率热图        | COCO val ~71.4                    | 中等，人多友好                        | 整图 → 多人 17 点           | MMPose                                                       |
| BlazePose    | 2020    | 检测+跟踪         | 端侧回归 33 landmark + 3D | 私有健身集约 68–74 mAP（17点）    | 手机/浏览器 30–200 FPS                | 单人 RGB → 33×(x,y,z)       | [MediaPipe](https://github.com/google-ai-edge/mediapipe)     |
| MoveNet      | 2021    | 轻量 BU           | 超小网络 + 智能裁剪       | 端侧够用，低于学术 SOTA           | 50+ FPS 手机/网页                     | RGB → 17 点                 | [tfjs-models](https://github.com/tensorflow/tfjs-models)     |
| ViTPose(+ +) | 2022    | Top-down          | 朴素 ViT + 可扩展预训练   | test-dev **80.9 / 81.1**          | 大模型需 GPU                          | crop 256×192 → 17/全身/动物 | [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)      |
| RTMPose      | 2023    | Top-down          | 轻骨干 + SimCC 分类定位   | s 72.2 / m 75.8 / l~77+           | **CPU 90+、1660Ti 430+、手机 70+**    | crop → 17 点                | [rtmpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) |
| RTMO         | 2023    | One-stage         | 无检测器的实时多人        | l ~74.8                           | 人多时优于 top-down                   | 整图 → 多人 17 点           | [rtmlib](https://github.com/Tau-J/rtmlib)                    |
| DWPose       | 2023    | Top-down 蒸馏     | 两阶段蒸馏全身点          | WholeBody **66.5**                | 快于同精度 RTMW                       | crop → 133 点               | [DWPose](https://github.com/IDEA-Research/DWPose)            |
| YOLOv8-Pose  | 2023    | One-stage         | 框和点同一头              | n 49.7 / x 68.9 / x-p6 71.5       | 极快，导出友好                        | 640 整图 → box+17×3         | [ultralytics](https://github.com/ultralytics/ultralytics)    |
| Sapiens      | 2024    | Top-down 基础模型 | 3 亿人体图预训练 ViT      | 17点最高 **82.2**；133点 **74.5** | 重，1024 级输入                       | 高分辨率 crop → 17/133/308  | [sapiens](https://github.com/facebookresearch/sapiens)       |
| DETRPose     | 2025    | 端到端 DETR       | 查询解码 + 关键点去噪     | S 67.0；L test-dev 71.2           | S ~2.4 ms，参数很少                   | 整图 → 多人点               | [arXiv](https://arxiv.org/abs/2506.13027)                    |
| Sapiens 2    | 2026    | 基础模型          | 10 亿图 + 更高分辨率      | 相对 v1 姿态约 +4 mAP             | 0.4B–5B，偏云端                       | 高分辨率 → 308 点           | [sapiens2](https://github.com/facebookresearch/sapiens2)     |
| YOLO26-Pose  | 2025–26 | One-stage         | YOLO 家族最新姿态头       | x 约 71.6 / 91.6（50-95/50）      | 实时检测级                            | 640 整图 → 17 点            | Ultralytics                                                  |

统一工具箱（不是单一算法，但业界默认入口）：

- **MMPose**：几乎所有学术模型的实现与训练标准  
  https://github.com/open-mmlab/mmpose  
- **rtmlib**：去掉 mmcv 依赖的 RTM/DW/RTMO/ViTPose 推理  
  https://github.com/Tau-J/rtmlib  

---

## 5. 2026 年怎么选（比追榜更有用）

**要精度、有 GPU、可离线**  
Sapiens / Sapiens2 或 ViTPose-H/G。17 点 COCO 已到 **80–82 AP** 区间；要脸手细节用 133/308。

**要上线、CPU/边缘/多人实时**  
第一选择 **RTMPose + RTMDet/YOLOX**（精度）或 **RTMO / YOLO-Pose**（人多、管线简单）。手机 App 优先 MediaPipe 或 MoveNet。

**要给 ControlNet / 数字人 / 全身驱动**  
**DWPose 或 RTMW**（133 点）几乎是事实标准；更高保真面部用 Sapiens 308。

**拥挤、遮挡、检测器会漏人**  
Bottom-up（HigherHRNet）或端到端（DETRPose、RTMO）；并在 CrowdPose / OCHuman 上复测，不要只看 COCO。

**要 3D 关节角、临床/运动分析**  
先用高质量 2D（RTMPose-l / ViTPose / Sapiens），再 lifting（MotionBERT、MeTRAbs）；需要体型就走 SMPL 系，不要指望 17 个 2D 点直接变准确骨骼长度。

---

## 6. 输入输出在工程里长什么样

典型 top-down 推理：

1. 检测器 → `bbox`  
2. 按长宽比 pad + resize 到 256×192 或 384×288 或 1024×768  
3. 网络出 heatmap 或 SimCC logits  
4. 峰值 / 软argmax → 原图坐标  
5. 每人 `N × 3`：`x, y, confidence`（3D 再加 `z` 或相机坐标）

COCO 17 点顺序固定：鼻、眼、耳、肩、肘、腕、髋、膝、踝。  
OKS 用关节标准差 σ 加权，所以手腕/脚踝比髋更「难得分」——这也是为什么全身模型手部 AP 总是最低。

---

## 7. 领域判断（到 2026）

1. **纯 COCO 17 点精度已经很卷**，单点提升主要来自更大预训练和更强检测器，而不是再发明一种卷积块。  
2. **真正拉开差距的是**：全身点、遮挡、域外（婴幼儿、艺术人体、极端运动）、时序稳定、端侧延迟。Sapiens 证明「人体专用基础模型」比任务特化结构更有后劲。  
3. **工业界收敛到四类栈**：OpenPose（遗产/ROS）、MediaPipe（端侧单人）、RTMPose 家族（服务器/边缘精度–速度甜点）、ViTPose/Sapiens（精度上限）。  
4. **下一步**更可能是：视频原生（不要逐帧抖）、3D/mesh 与 2D 联合、与语言模型/动作语义对齐，而不是再刷 0.3 个 COCO AP。

如果你下一步要落地（例如「手机端 30FPS 全身点」或「服务器批处理最高精度 17 点」），可以缩小场景，我可以按延迟预算给一条具体的检测器 + 姿态模型 + 导出格式组合。