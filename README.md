# AwesomePaper-for-AI
Awesome system papers for AI

## Arbitrage 投机加速
Arbitrage: Efficient Reasoning via Advantage-Aware Speculation
https://arxiv.org/abs/2512.05033 伯克利 apple等，2025.12.4

1. 🤔 现有的步级 Speculative Decoding (SD) 方法，例如 RSD，在推理任务中**由于频繁且无意义地重新生成步骤**，导致了大量的计算浪费。
2. 💡 ARBITRAGE 提出了一种**优势感知型步级推测生成框架**，它使用轻量级 ARBITRAGE ROUTER 来**预测目标模型何时能提供显著更好的步骤**，从而最大限度地减少冗余计算。
3. ✨ 在多个数学推理基准测试中，ARBITRAGE 始终优于先前的基线，在匹配精度下可将推理延迟降低高达约 2 倍。
基于草稿模型和目标模型在特定推理步骤上的预期质量差异。它动态地**在草稿模型和目标模型之间进行路由**，仅在预期目标模型**能够提供“有意义的改进”时才调用目标模型**。

本文提出了一种名为 ARBITRAGE 的新型分步式推测生成框架，旨在解决大型语言模型（LLMs）在长篇链式思考（Chain of Thought, CoT）推理任务中推理效率低下的问题。尽管传统的推测解码（Speculative Decoding, SD）通过利用一个快速但不太精确的草稿模型（draft model）和一个更强大、更精确的目标模型（target model）来加速推理，但其在推理任务中面临挑战，因为微小的 token 级不匹配就可能导致语义上等效的步骤被不必要地拒绝。近期工作转向了分步式语义验证，例如奖励引导推测解码（Reward-guided Speculative Decoding, RSD），通过使用进程奖励模型（Process Reward Model, PRM）评估整个推理步骤，但这仍然导致大量被拒绝步骤的无谓重新生成，浪费了宝贵的计算资源。

**现有方法（RSD）的局限性：**
RSD 通过将草稿步骤的 PRM 分数 $s_d$ 与一个固定阈值 $\tau$ 进行比较来做出决策：如果 $s_d > \tau$，则接受草稿步骤；否则，拒绝并由目标模型重新生成。这种基于绝对分数的方法存在固有缺陷：即使目标模型预期不会产生显著更好的结果，它也会触发昂贵的目标模型调用。作者通过量化“浪费的计算”（wasted computation）来证明这一点，即目标模型重新生成步骤但并未比草稿步骤获得更高或同等 PRM 分数的情况。
$W_{RSD} \triangleq E[C_t \cdot I\{A=0\} \cdot I\{s_t \le s_d\}]$
其中 $C_t$ 是目标模型的计算成本，$A$ 是接受指标，$s_t$ 是目标模型生成的步骤的 PRM 分数。实证分析表明，在拒绝率较高时，高达约40%的总步骤重新生成并没有带来质量提升。

**ARBITRAGE 方法论：**
ARBITRAGE 的核心思想是，决策不应仅仅基于草稿输出的绝对质量，而应基于草稿模型和目标模型在特定推理步骤上的预期质量差异。它动态地在草稿模型和目标模型之间进行路由，仅在预期目标模型能够提供“有意义的改进”时才调用目标模型。

1.  **ARBITRAGE ORACLE（理想路由策略）：**
    理论上，ARBITRAGE ORACLE 代表了路由效率的上限。对于给定上下文 $x$，以及草稿步骤 $z_d$（PRM 分数 $s_d = h_{\theta_{PRM}}(x, z_d)$）和目标步骤 $z_t$（PRM 分数 $s_t = h_{\theta_{PRM}}(x, z_t)$），ORACLE 选择 PRM 分数较高的步骤：
    $z^* = \underset{z \in \{z_d, z_t\}}{\arg\max} h_{\theta_{PRM}}(x, z)$
    这引入了“步级优势”（step-level advantage）的概念：
    $\Delta = s_t - s_d$
    其中 $\Delta > 0$ 表示目标模型优于草稿模型。
    ORACLE 的升级决策是一个二元变量 $a \in \{0, 1\}$，其中 $a=1$ 表示升级到目标模型，$a=0$ 表示接受草稿步骤。在控制升级率的情况下，ORACLE 使用阈值策略：
    $a^*_\tau = I\{\Delta > \tau\}$
    在预算受限下（即限制目标模型调用的次数），该阈值规则被证明是局部最优的，它在给定升级率下最大化了预期 PRM 分数 $S(a) = s_d + a\Delta$。

2.  **ARBITRAGE ROUTER（实用路由策略）：**
    由于 ORACLE 在实践中需要运行昂贵的目标模型才能计算 $s_t$，因此无法直接使用。ARBITRAGE ROUTER 是一个轻量级的预测模型，它近似于 ORACLE 的决策，**仅利用草稿侧信息 $(x, z_d)$ 来预测升级是否会提高质量。**具体而言，路由器的输出 $\hat{y} = h_{\theta_{router}}(x, z_d)$ 被解释为目标模型优于草稿模型的可能性。在推理过程中，如果 $\hat{y} \le \tau$，则接受草稿步骤；否则，由目标模型重新生成。

    **路由器训练流程：**
    *   **数据集构建：** 通过并行解码草稿模型和目标模型，获取配对的步骤 $(z_d, z_t)$。然后使用固定 PRM $h_{\theta_{PRM}}$ 计算它们的 PRM 分数 $s_d$ 和 $s_t$，从而得到优势 $\Delta = s_t - s_d$。ORACLE 标签 $y = I[\Delta > 0]$ 表示是否需要升级。
    *   **数据预处理：** 数据集存在类别不平衡问题（$y=0$ 占多数），通过对多数类别进行随机下采样来平衡训练集。同时，标准化序列长度并强制使用一致的步骤分隔符。
    *   **路由器训练：** 路由器初始化自一个紧凑的 PRM 检查点，以便获得评估中间推理质量的归纳偏置。使用 AdamW 优化器进行微调，并采用线性 warmup 和 decay。分类头应用于最终 token 嵌入，通过标准交叉熵损失进行优化。
    *   **路由质量评估：** 采用 Spearman 秩相关系数 $\rho = \text{corr}_S(\hat{y}, \Delta)$ 作为阈值不变的代理指标，评估预测概率 $\hat{y}$ 与真实优势 $\Delta$ 之间的对齐程度，从而反映路由器预测排序与 ORACLE 真实排序的一致性。

**实验评估：**
ARBITRAGE 在多个数学推理基准（MATH500 和 OlympiadBench）和不同模型配置（LLaMA3 和 Qwen2.5-Math 系列，包括量化草稿模型）上进行了评估。

*   **计算-质量权衡：** 实验结果表明，在相同的接受率下，ARBITRAGE ROUTER 曲线始终严格优于 RSD，表明其在单位目标模型使用量下提取了更高的准确率。特别是当草稿模型显著弱于目标模型时，ARBITRAGE ROUTER 的优势更为明显。
*   **端到端延迟加速：** ARBITRAGE 在准确率-延迟的 Pareto 曲线上严格优于 RSD。在 MATH500 上，使用量化草稿模型时，ARBITRAGE 在可比准确率下实现了高达 1.62 倍的延迟降低。在 OlympiadBench 上，使用小草稿模型时，它在匹配准确率下实现了高达 1.97 倍的加速。这归因于 ARBITRAGE 更具选择性的升级策略，仅将昂贵的计算集中在预期目标模型能带来最大收益的实例上。
*   **案例研究：** 质性分析显示，RSD 因绝对 PRM 分数截止而拒绝的某些步骤，ARBITRAGE 能够接受，因为这些步骤尽管 PRM 分数不高，但目标模型也无法带来显著改进。这避免了冗余的重新生成，并保持了推理轨迹的连贯性。

**消融实验：**
*   **模型架构：** 2-分类设置（接受 vs. 升级）在 Spearman 相关性和两类准确率之间取得了最佳平衡，优于 4-分类或 10-分类以及序数回归变体，证实二元分类是最鲁棒的选择。
*   **数据相关设计：**
    *   **步级注释：** 在每个推理步骤的输入前添加模型选择历史（如“Model 0”或“Model 1”），提升了标签 1（拒绝草稿）的准确率和 Spearman 相关性，表明利用路由历史有助于提升性能。
    *   **数据下采样：** 对多数类别（接受草稿）进行类平衡下采样，避免了路由器过度自信地预测接受草稿，从而改善了校准和预测平衡性，提升了路由质量。

**结论：**
ARBITRAGE 通过将决策从绝对、仅基于草稿的接受规则转变为基于预期优势的估计，显著提高了分步式推测解码的效率-准确率权衡。它减少了推理延迟，并在固定准确率目标下，实现了高达约 2 倍的加速。这为推理密集型任务中的 LLM 推理加速树立了新的基准。

## SOSP Copier
How to copy memory
https://ipads.se.sjtu.edu.cn/_media/pub/sosp25-copier-preprint.pdf 陈海波团队 SOSP25 best paper
1. 💡 该论文提出 Copier，一个将内存拷贝 (memory copy) 提升为 OS first-class service 的新方法，旨在通过协调异步拷贝来解决其在 syscalls、IPC 和用户应用中普遍存在的性能瓶颈。
2. ⚙️ Copier 通过利用 Copy-Use windows 抽象异步拷贝、充分发挥 SIMD 和 DMA 等硬件能力，以及通过全局视图实现 holistic optimization (如 copy absorption) 来实现高效拷贝。
3. 🚀 实验结果表明，Copier 在 Redis 中实现了高达 1.8 倍的加速，在 TinyProxy 等实际应用中相较于 state-of-the-art 解决方案有显著提升，并成功集成到 HarmonyOS 等商业 OS 中。


内存拷贝在现代系统中是一个关键的性能瓶颈，广泛存在于系统调用（syscalls）、进程间通信（IPC）和用户态应用中。现有的优化方案，如零拷贝（zero-copy）和硬件加速拷贝，通常针对特定场景，且存在诸多局限。例如，Linux 的零拷贝 `send()` 仅对 ≥10KB 的消息有效，而 `zIO` 需要 ≥16KB 的拷贝大小才能带来收益。

本文提出将内存拷贝作为一个**一等公民的操作系统服务（first-class OS service）**，并引入了名为 **Copier** 的协调异步拷贝服务。Copier 的核心理念在于利用普遍存在的**拷贝-使用窗口**（Copy-Use window）来隐藏拷贝延迟，充分发挥硬件潜力，并通过全局视角实现整体优化。

Copier 的设计解决了以下挑战：
1.  **从同步到异步**：如何在保持同步拷贝语义的同时，充分利用异步拷贝的性能优势？
2.  **从函数到OS服务**：如何充分利用硬件特性和全局视图进行优化，同时将额外开销降至最低？
3.  **从单客户端到多客户端**：如何确保资源隔离、公平性和正确性？

为此，Copier 采用了以下技术：

**核心方法论：**

1.  **管道化拷贝-使用（Pipelined Copy-Use）与 Copier 抽象**：
    *   **接口**：Copier 提供 `amemcpy()`（异步 `memcpy`）和 `csync()`（拷贝同步）两个基本原语。`amemcpy()` 用于提交异步拷贝任务，`csync()` 用于在数据使用前确保数据已准备就绪。
    *   **CSH 队列（CSH Queues）**：客户端通过内存映射的（per-client）队列与 Copier 交互，包括 `Copy Queue (QCopy)`、`Sync Queue (QSync)` 和 `Handler Queue`。
        *   `Copy Queue`：用于提交 `Copy Task`（包含源、目标、长度）。为了提高异步效率，Copier 支持**分段拷贝（segment-based copy）**，将一个拷贝任务划分为多个固定大小的 `segments`，并提供一个 `descriptor`（位图）来细粒度地跟踪每个 `segment` 的完成状态，允许应用在整个拷贝完成前就使用已拷贝的部分数据，实现拷贝-使用管道化。
        *   `Sync Queue`：用于提交 `Sync Task`，当客户端需要未就绪的数据时，通过 `csync()` 提交 `Sync Task`，提升对应 `segments` 的优先级，实现**任务提升（task promotion）**，解决传统 FIFO 队列可能导致的**队头阻塞（head-of-line blocking）**问题。
        *   **委托式处理器（Delegation-based Handler）**：`Copy Task` 可附带一个 `FUNC`（函数指针和参数），在拷贝完成后由 Copier 或 libCopier 异步执行，解决了零拷贝中内存所有权管理复杂和 `TOCTTOU` 等问题。

2.  **依赖追踪（Dependency Tracking）**：
    *   **顺序依赖（Order Dependency）**：通过引入**跨队列屏障（Cross-Queue Barrier）**，并利用系统事件（如 syscall trap 和 return）作为屏障指示器，Copier 能够追踪不同特权级（用户态和内核态）队列间拷贝任务的提交顺序，确保即使在异步和乱序执行下也能维持正确性。
    *   **数据依赖（Data Dependency）**：Copier 追踪拷贝任务之间是否存在重叠内存区域，从而建立数据依赖关系。这对于处理 `Sync Task` 和实现拷贝吸收至关重要。

3.  **异构拷贝单元协调（Harmonizing Copy Units）**：
    *   Copier 作为 OS 服务，可以充分利用 CPU 的 SIMD (如 AVX2) 和 DMA 引擎。
    *   **CPU-DMA 混合子任务（CPU-DMA Hybrid Subtasks）**：将拷贝任务划分为子任务，针对不同大小的子任务选择合适的硬件单元。小于特定阈值（如 1.4KB）的子任务优先使用 CPU（AVX2），大于该阈值的考虑 DMA。
    *   **搭便车式调度器（Piggyback-based Dispatcher）**：通过将 DMA 任务“搭便车”到 AVX 任务上并行执行，实现 DMA 和 AVX 拷贝的重叠，避免 CPU 等待 DMA 完成造成的周期浪费。
        *   **内部搭便车（i-piggyback）**：针对单个大型任务，将其中适合 DMA 的部分分配给 DMA，其余部分给 AVX2。
        *   **外部搭便车（e-piggyback）**：针对多个小型任务，在它们之间没有数据依赖时，从多个任务中选取 DMA 候选子任务进行批量 DMA 处理。
    *   **地址转换缓存（ATCache）**：利用拷贝地址的局部性，缓存虚拟地址到物理地址的转换，减少 DMA 任务的翻译开销。

4.  **拷贝吸收（Copy Absorption）**：
    *   **分层拷贝吸收（Layered Copy Absorption）**：Copier 能够识别并消除冗余的中间拷贝。当存在 `A→B` 和 `B→C` 这样的拷贝链时，如果 `B` 的部分数据未被访问（或已访问但未修改），Copier 可以直接从 `A` 拷贝到 `C`，或者从 `B` 的最新数据源拷贝。
    *   **懒拷贝任务（Lazy Copy Task）**：应用程序可以将一个 `Copy Task` 标记为 `lazy`，表示它优先级最低，只在被依赖或特定时间后才处理，为拷贝吸收提供更多机会。
    *   `abort` 任务：允许显式放弃不再需要的队列中拷贝任务。

5.  **公平与隔离的多客户端服务（Fair and Isolated Multi-client Serving）**：
    *   **Copier 线程（Copier Threads）**：Copier 拥有独立的内核线程来处理请求，通过**情景驱动式轮询（Scenario-driven polling）**和**线程自适应扩缩容（auto-scaling）**来平衡性能和能耗。
    *   **cgroup 扩展**：将拷贝长度（copied length）作为资源单位，扩展 Linux `cgroup` 机制，通过 `copier.shares` 实现不同客户端（进程或 OS 服务）之间的资源隔离和公平调度，类似于 `CFS`。
    *   **主动故障处理（Proactive Fault Handling）**：Copier 在拷贝前主动触发并处理页面错误，确保虚拟地址到物理地址的映射，并通过锁定映射来避免拷贝过程中的页面失效问题，同时执行安全检查。

**应用与评估：**

Copier-Linux 在 Linux 5.15.131 上实现，并实验性集成到 HarmonyOS 5.0 商用手机操作系统中。
*   **微基准测试**：Copier 在吞吐量上比 Linux 内核默认的 `ERMS` 拷贝方法提升高达 158%，比用户态的 `AVX2` 提升高达 38%。
*   **OS 服务优化**：
    *   `send()` 和 `recv()` 系统调用：延迟降低 7%-59% 不等，优于 `UB`、`io_uring` 和零拷贝方案在中小数据量时的表现。
    *   `Binder IPC`：端到端延迟降低 9.6%-35.5%。
    *   **CoW（写时复制）**故障处理：页故障处理的线程阻塞时间降低 8%-71.8%。
*   **真实应用优化**：
    *   Redis：`SET` 和 `GET` 操作的端到端平均延迟降低 2.7%-43.4%，吞吐量提升 2.4%-50.0%，显著优于 `zIO` 和 `UB`。
    *   TinyProxy：吞吐量提升 7.2%-32.3%，并能有效利用拷贝吸收。
    *   Protobuf：消息接收和反序列化延迟降低 4%-33%。
    *   OpenSSL：`SSL_read()` 延迟降低 1.4%-8.4%。
    *   zlib：压缩性能提升 18.8%。
    *   HarmonyOS 上的 `Avcodec`：视频解码延迟降低 3%-10%，帧丢失减少 22%。

**开发与讨论：**

*   **开发工作量**：由于 libCopier 封装了大部分复杂性，移植现有应用的工作量适中。CopierSanitizer 工具辅助检测 `csync()` 遗漏，CopierGen 探索自动化移植。
*   **系统资源利用**：在有空闲核心时，Copier 可显著提升性能。在核心完全利用时，虽然可能引入少量 `polling` 开销导致吞吐量略有下降，但对于延迟敏感的应用仍有益；对于有拷贝链或大型拷贝的应用，由于硬件加速和拷贝吸收节省的 CPU 周期，Copier 仍能提升整体吞吐量。
*   **微架构影响**：Copier 通过解耦大拷贝和应用执行，减少缓存污染，降低 `CPI`。

**未来展望**：Copier 的概念可以进一步整合为 CPU 硬件原语，或应用于更广泛的 OS 服务和场景（如文件 I/O、分层内存管理）。其安全性也优于零拷贝方案，因为它确保数据被拷贝到私有缓冲区后才被检查，避免了所有权共享带来的信任问题。

## SOSP Mercury
Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling
https://dl.acm.org/doi/pdf/10.1145/3731569.3764798
  - https://github.com/ChandlerGuan/mercury_artifact

1. 💡 大型语言模型 (LLMs) 对计算和内存的需求已超出单 GPU 的能力，导致多 GPU 算子优化成为关键挑战，而现有方法未能有效利用远程 GPU 内存。
2. 🛠️ 为此，本文提出了 Mercury，一个**基于 CommIR 的多 GPU 算子编译器，CommIR 将远程 GPU 内存视为内存层级中的一等公民**，通过引入 p**arallelize、shift 等变换原语来统一计算、内存和通信调度**。
3. 🚀 实验表明，Mercury **能够自动生成高效的多 GPU 算子**，在各种 LLM 算子和硬件平台上，其性能一致优于现有手调基线和自动优化方法，平均加速比达 1.56 倍。

Mercury是一篇关于大模型（LLMs）多GPU算子优化的论文，提出了一个名为Mercury的编译器。该编译器核心思想是将远程GPU内存视为内存层次结构中显式管理的一部分，**扩展了可用存储和通信资源，超越了本地HBM的限制**。**通过这种统一的视角，编译器能够整体性地推理数据放置和设备间通信，从而解锁了一个远超现有手动策略的巨大设计空间**。
<img width="884" height="232" alt="image" src="https://github.com/user-attachments/assets/b8dbe386-a93d-4d06-a99e-ab750a88a2c1" />
<img width="512" height="320" alt="image" src="https://github.com/user-attachments/assets/2c792f9f-2074-4f3c-8533-e66d07f99144" />
<img width="1028" height="398" alt="image" src="https://github.com/user-attachments/assets/ccc2d40d-11f8-4a6f-bbbc-ecdb49219eaa" />
<img width="1041" height="572" alt="image" src="https://github.com/user-attachments/assets/ddab11b0-2660-40d5-b060-26eccaf05581" />

**1. 问题背景与动机**
随着LLMs模型大小和输入序列长度的增长，单个算子（特别是Attention和GEMM）的计算和内存需求已超出单个GPU的容量。例如，Llama-3 70B的KV缓存需要282GB内存，远超NVIDIA H100 GPU的80GB HBM。因此，多GPU算子设计不仅是性能优化，更是训练和推理大型模型的根本要求。然而，**优化LLMs的多GPU算子是一个高度手动且劳动密集的过程**，现有方法难**以适应新的硬件和模型配置**。当前**多GPU编译器（如torch.compile）的性能瓶颈在于其“本地内存中心”的执行模型假设**，即所有输入数据必须在计算前完全存在于每个GPU的本地内存中。这种**限制导致数据重复、浪费HBM容量**，并阻碍了更深层次的平铺（tiling）优化，**使得编译器无法探索更灵活的执行和通信模式，特别**是利用远程内存作为共享数据存储资源**以减少本地内存占用和通过计算-通信重叠**（compute-communication overlap）降低整体延迟。

**2. 核心方法：CommIR与远程内存调度**
Mercury的核心是CommIR，一个**基于循环（loop-based）的中间表示（IR）**。CommIR通过将远程GPU内存视为内存层次结构中一等公民，统一了计算、内存和通信。它引入了**结构化的转换原语（transformation primitives）**，以支持标准的**GPU内平铺和高级的GPU间调度模式，如异步的shifted patterns和**集体通信原语。

**2.1 CommIR的定义与原语**
CommIR扩展了传统的循环IR，引入了四种用于远程内存访问语义的转换原语：
*   **Parallelize**: 将一个循环的迭代分配到**并行工作器**（parallel workers）上，指定网络层级（如inter-node或intra-node）。它默认初始化，将由并行循环索引的缓冲区分片（shard），未索引的则复制（replicate）。
*   **Shift**: **偏移本地循环的索引**，使其相对于并行循环的索引，从而引入异步访问模式，使不同工作器错开数据访问。这显式地在不同时间步引入远程内存访问，并自动对移位循环相关的缓冲区进行分片。
*   **Shard**: 将缓冲区**跨工作器分片**，每个并行rank拥有缓冲区的一个不相交部分。
*   **Replicate**: 在参与并行循环的所有工作器之间**复制缓冲区**，使每个rank都拥有一个完整的副本。

**2.2 远程内存访问模式的表示**
这些原语使得CommIR能够表示广泛的远程内存访问模式：
*   **并行语义（Parallel Semantic）**: `Parallelize`原语通过在指定网络层级（如层级0为inter-node，层级1为intra-node）并行化循环，来实现工作负载分配。例如，在图4(c), (d)中，循环$I_0$和$J_0$分别映射到层级0和1，这导致了缓冲区$B$在inter-node层级被共享，在intra-node层级被复制。
*   **异步访问（Asynchronous Access）**: `Shift`原语通过偏移循环索引在工作器之间引入异步性。例如，在图5中，对循环$J$进行`Shift`操作，使其索引依赖于并行循环$I$的索引，即`(j+i)%J`。这导致了错开的数据访问，从而降低了存储需求，并通过允许计算和通信重叠（如Ring-style传递）来提高效率。
*   **集体访问（Collective Access）**: `Shard`和`Replicate`原语在降低阶段（lowering phase）被解释为集体操作，如AllGather、Broadcast、AllReduce和ReduceScatter。例如，分片读取缓冲区会触发集体收集操作（AllGather或Broadcast），而分片写入缓冲区则触发归约操作（AllReduce或ReduceScatter）。CommIR在降低阶段通过基于规则的模式匹配来决定插入哪种集体通信。

**2.3 CommIR的表达能力**
CommIR可以表示广泛的并行策略，包括：
*   **同步并行算子**：如上下文并行（Context Parallel）通过并行化注意力操作的查询维度（I轴），以及DeepSpeed-Ulysses的头部并行（Head Parallel）通过并行化头部维度。
*   **异步算子**：如RingAttention和LoongTrain/USP，它们通过在特定轴（如KV激活的J轴）上应用`Shift`原语来实现数据在工作器之间的异步传递和通信-计算重叠。
*   **集体归约**：如TreeAttention，通过重排循环并并行化归约轴（J轴），然后使用`AllReduce`来实现。
*   **新型模式**：CommIR能自动探索和发现手动设计难以企及的新策略。例如，将`Shift`应用于归约轴（如图7中的“Shift Reduce”），允许部分和在工作器之间并行传递。更复杂的混合模式（如图7中的“Hybrid Pattern”）则结合了多层级的`Parallelize`和`Shift`，实现了复杂的组内和组间通信。

**3. 自动调优（Auto-Tuner）**
Mercury采用自动调优系统来搜索最佳的分布式算子调度。
*   **设计空间生成**：生成过程分为两个顺序阶段：
    *   **计算调度（Computation Schedule）**：应用`Tile`、`Reorder`和`Join`转换来定义本地循环结构和缓冲区布局，探索不同的计算划分方式。
    *   **通信调度（Communication Schedule）**：应用`Parallelize`、`Shift`、`Shard`和`Replicate`等通信原语，分配计算并管理远程内存访问。
    *   为了减少搜索空间，生成过程显式结合了硬件拓扑（hardware mesh configuration），只根据网格大小进行平铺和并行化。
*   **搜索目标**：调优器旨在最小化生成分布式算子的端到端延迟，同时满足内存约束。
    *   **延迟评估**：每个候选调度都被完全降低（lowered）并在真实硬件上进行性能分析以获取运行时测量数据。
    *   **内存约束**：通过CommIR表示静态分析每个候选的存储布局，并计算每个工作器的内存占用，超出容量的候选被提前剪枝。

**4. 实现细节**
*   **DSL**：Mercury设计了一个Python-like的领域特定语言（DSL），它扩展了PyTorch的前端，使其支持循环级别的分发语义。DSL基于`Axis`（循环变量）、`Buffer`（带轴形状的张量）和`Grid`（计算迭代空间）三个核心抽象。
*   **调度原语实现**：
    *   **计算原语**：通过重写IR的循环树来修改循环嵌套表示。
    *   **通信原语**：以IR中循环和缓冲区对象的注解形式实现，不直接插入通信代码，而是在代码生成阶段进行实例化。
*   **代码生成**：
    *   **通信核生成**：通过静态分析CommIR中的循环索引转换来推断P2P通信（如`Shift`引入的错开发送/接收模式）的发送方和接收方rank。集体通信则通过分析带有`shard`或`replicate`注解的缓冲区的访问模式和聚合语义来推断。
    *   **本地计算降低**：将IR的计算部分降低到后端特定的代码，主要使用TorchInductor。对于计算密集型区域，可选地修补（Patch）高性能库（如FlashAttention）。
*   **算子间重新分片（Inter-Operator Resharding）**：为了将算子级优化的好处扩展到整个模型，Mercury支持图级推理，考虑单个算子的执行时间以及它们之间重新分片所需的通信开销。通过一个图级搜索算法（如Algorithm 1所示），在算子配置和重新分片成本之间找到最优组合，最小化总成本。

**5. 评估**
Mercury在多种GPU设备和互连设置上进行了评估，包括H100、A100和L4。
*   **算子基准测试**：在Attention（MHA, GQA）和GEMM（AllGather-GEMM, GEMM-ReduceScatter）算子上，Mercury在所有设置中始终优于现有解决方案，平均加速比在H100上高达4倍（MHA），在A100上达到1.9倍（GEMM）。它能自适应地为算子特征和硬件定制优化策略。
*   **网络拓扑适应性**：在不同的多GPU配置下（如1x4、2x2、4x1等），Mercury的延迟始终最低，平均加速比达2.91倍。其优势在复杂的混合拓扑中更为显著，而静态方法则表现不佳。
*   **上下文长度可扩展性**：在MHA算子中，随着序列长度从32K扩展到2M tokens，Mercury始终优于所有基线。在2M tokens下，其他基线出现OOM错误，而Mercury通过积极分片KV缓存和输出张量，以增加通信换取内存减少，从而生成可行的执行计划。
*   **模型级基准测试**：在Llama3-8B和Llama3-70B模型上，Mercury的计算图级搜索算法（图12），相比3D并行策略显著降低了延迟。这得益于对算子调度和重新分片决策的协同考虑，消除了冗余布局转换，并简化了层间数据流。
*   **设计空间分析**：如图13所示，Mercury通过CommIR提供的广泛和富有表现力的搜索空间，能够探索现有策略以及手动设计难以企及的新颖调度。例如，在最佳延迟的调度中，它结合了多层级的HP与shifted CP，并对归约维度进行shift以实现细粒度的计算-通信重叠。而最佳内存的调度则使用上下文维度的intra-node并行和shifted的本地Q维度，显著减少了峰值内存使用。USP等手动设计仅探索了整体设计空间中的一小部分。

**6. 相关工作**
*   **张量编译器**：Halide和AutoTVM/Ansor/MetaSchedule等早期张量编译器主要关注GPU内调度。随着模型规模增大，它们加入了基础的多GPU支持，但通常将设备间数据移动视为外部机制，限制了远程内存复用和异步共享的机会。Mercury通过将远程GPU内存视为一等公民，并统一计算、内存和通信与显式原语，在此基础上取得了进展。
*   **融合分布式算子**：Flux、Comet、Triton-Distributed和TileLink等系统探索了将通信与计算紧密结合的融合设计，以最大化重叠。Mercury的`Shift`原语概念可以应用于这些融合设计中，但其代码生成和自动调优仍是未来研究挑战。

**7. 结论**
Mercury是一个用于多GPU张量程序的自动化编译器框架，建立在创新的循环IR CommIR之上。通过结合自定义DSL和先进的调度与通信原语，Mercury协同优化计算和通信，发现了新颖的并行策略，在Attention和GEMM算子上超越了现有最先进的技术。它简化了复杂的多GPU算子设计，并能适应多样化的硬件，为大规模模型的扩展性、高效执行铺平了道路，并为未来的调优、图级集成和异构设备支持开辟了途径。

## Coruscant
Coruscant: Co-Designing GPU Kernel and Sparse Tensor Core to Advocate Unstructured Sparsity in Efficient LLM Inference
- Micro25：软件+硬件。
- https://github.com/dhjoo98/coruscant 稀疏加速 基于Flash-LLM。M

1. 🚀 Coruscant 提出了一种 Bitmap-based 的稀疏格式，显著提高了 LLM 剪枝在 30% 到 70% 稀疏度范围内的内存压缩效率，弥补了现有稀疏格式在该范围内的不足。
2. ⚡️ 基于此稀疏格式，Coruscant GPU Kernel 通过减少 GPU 全局内存到处理器的数据传输，并进行高效解压缩，在 GPU 上加速了 SpMM 操作，实现了相对于** cuBLAS 高达 2.02 倍、Flash-LLM 高达 1.48 倍的加速。**
3. ✨ Coruscant Sparse Tensor Core 通过在 Tensor Core 中集成 Bitmap Decoder，实现了对压缩格式的直接操作，消除了软件解压缩开销，进一步将 SpMM 速度提升至 cuBLAS 的 2.75 倍，并在 Llama 2 推理中显著提高了 token 生成吞吐量。
<img width="1461" height="351" alt="image" src="https://github.com/user-attachments/assets/b9e6b756-4cda-4baa-a31b-94984bc9cfcf" />

Coruscant 论文提出了一种针对大型语言模型（LLMs）推理中非结构化稀疏性（unstructured sparsity）的高效解决方案，通过协同设计GPU kernel和稀疏Tensor Core来应对当前硬件在处理非结构化稀疏性时的挑战。

**1. 背景与动机**
LLMs的庞大尺寸和长上下文生成对内存资源提出了高要求。剪枝（pruning）、量化（quantization）和蒸馏（distillation）是常用的模型压缩技术。其中，剪枝在非结构化稀疏性方面潜力巨大，能更好地保持精度，但在现代硬件上利用效率低下。当前LLM剪枝常受限于结构化模式（如2:4半结构化稀疏性，NVIDIA Tensor Core支持），尽管这限制了剪枝的灵活性和精度。Coruscant的目标是弥合非结构化稀疏性潜力与硬件效率之间的鸿沟，专注于加速稀疏矩阵-密集矩阵乘法（SpMM），这是LLM推理decode阶段的主要瓶颈，且通常是内存密集型（memory-bound）操作。研究发现，LLM剪枝的最佳稀疏度范围通常在30%到70%之间，而现有稀疏格式（如CSR, COO, Tiled-CSL）在此范围内的压缩效率不佳，甚至可能导致稀疏表示大于原始密集矩阵。

**2. Coruscant核心方法**
Coruscant包含三个主要组件：Coruscant Sparse Format、Coruscant GPU Kernel和Coruscant Sparse Tensor Core。

**2.1 Coruscant Sparse Format**
该格式采用基于位图（bitmap-based）的方法来表示非零元素的位置，显著提高了30%到70%稀疏度范围内的压缩比（70%稀疏度时可达36.25%，50%稀疏度时可达56.25%）。它将矩阵逻辑上划分为列式瓦片（column-wise tiles），每个瓦片（例如1x64）由其非零值和对应的64位位图表示。位图的优势在于避免了存储每个非零元素的显式坐标数组，从而在目标稀疏度范围内实现了比传统格式（如CSR, COO, Flash-LLM的Tiled-CSL）更高的压缩效率。

**2.2 Coruscant GPU Kernel**
该 kernel利用Coruscant Sparse Format来减少GPU global memory与处理器之间的数据传输，从而加速内存受限的SpMM操作。
*   **SpMM公式（Formulation）**：它沿用了Flash-LLM的流水线机制，将压缩的稀疏矩阵瓦片加载到GPU处理器寄存器中，然后解压缩到shared memory中的密集矩阵瓦片，最后Tensor Core在这些密集瓦片上执行矩阵乘法。寄存器和shared memory充当乒乓缓冲区（ping-pong buffer），实现加载-压缩和计算-密集迭代的流水线化。
*   **解压缩算法（Decompression Algorithm）**：核心在于高效地将位图编码的稀疏瓦片解压缩为shared memory中的密集瓦片。每个线程负责解压缩两个长度为64的列，并将其写入shared memory。算法使用 `clz` (count leading zeros) 指令快速找到位图中非零位的位置，避免了逐位迭代，提高了效率。关键优化在于确保CUDA编译器将非零值放在线程寄存器中，并通过将每个瓦片的非零元素填充到8的倍数，确保内存访问合并（coalesced memory access）并充分利用GPU内存带宽。
*   **避免Shared Memory Bank Conflict**：Coruscant采用列式瓦片划分（column-wise tiling），确保每个线程写入不同的shared memory bank，从而避免了解压缩过程中常见的bank conflict，进一步提升了性能。

**2.3 Coruscant Sparse Tensor Core**
该组件旨在通过硬件层面的修改，消除Coruscant GPU kernel中解压缩到shared memory的开销（这部分在GPU kernel中可占总执行时间的36%）。

**3. 实验评估与结果**
Coruscant在NVIDIA RTX 6000 Ada GPU上进行了评估，并将其内核和稀疏Tensor Core模拟集成到Huggingface Transformers库中，用于Llama 2 7B和13B的端到端推理。

*   **端到端LLM评估**：
    *   在Llama 2 7B上，Coruscant GPU kernel将token吞吐量提高103-135 tokens/sec（26%提升），Coruscant Sparse Tensor Core进一步提高到206 tokens/sec（40%提升）。
    *   在Llama 2 13B上，Coruscant的权重压缩能力使得在32 batch size下能够成功运行（cuBLAS OOM），从而提高了76 tokens/sec（kernel）和98 tokens/sec（STC）的吞吐量。
*   **Kernel性能与分析**：
    *   **延迟**：Coruscant kernel在30%-70%稀疏度范围内始终优于cuBLAS。相比Flash-LLM，Coruscant在30%-60%稀疏度下表现更优，在70%稀疏度时性能接近。这表明Coruscant在目标稀疏度范围内达到了平衡点。
    *   **内存占用**：Coruscant通过高效压缩，将内存占用比密集cuBLAS减少14%（30%稀疏度）到55%（70%稀疏度）。相比Flash-LLM，Coruscant在所有30%-70%稀疏度下都显示出更低的内存占用。这不仅加速了SpMM，还为KV cache等腾出了更多GPU VRAM。
    *   **内存开销细分**：Coruscant的压缩特性导致更低的全局内存stall cycles，且内存利用率更高，验证了压缩有效降低了GPU全局内存压力。
*   **稀疏Tensor Core评估**：
    *   **性能提升**：Coruscant Sparse Tensor Core将Tensor Core利用率平均提升2.5倍，并将共享内存stall cycles降低到Coruscant kernel的5%，平均实现1.3倍的额外加速。
    *   **面积和功耗**：Bitmap Decoder的硬件开销极小，对Volta、Ampere和Hopper架构的GPU面积增加不超过0.018%，功耗增加在26.24mW到74.79mW之间。
    *   **与现有稀疏Tensor Core比较**：
        *   **DSTC [55]和RM-STC [21]**：这些设计支持任意稀疏度，但主要针对计算密集型（compute-bound）的大型方阵乘法。它们的复杂之处在于需要散集（scatter-gather）逻辑和中间累加缓冲区，导致硬件复杂度和面积开销巨大（比Coruscant-STC高95%-96%）。此外，它们的瓦片尺寸较小，导致非零指针开销相对较高。Coruscant通过专注于SpMM和利用现有Tensor Core的每通道累加能力，避免了这些复杂性。
        *   **半结构化稀疏性（Semi-Structured Sparsity）**：例如NVIDIA的2:4稀疏性（cuSPARSELt）。半结构化稀疏性通过固定非零模式来消除存储开销和实现静态计算跳过，从而在硬件效率上通常优于非结构化稀疏性。Coruscant-STC在50%稀疏度时性能与2:4 kernel接近，在70%稀疏度时甚至超越，这得益于硬件解压缩和更高的压缩比。然而，半结构化剪枝（如Wanda [49]在2:4模式下）会导致更高的模型精度损失（更高困惑度 Perplexity，更低TriviaQA分数），因为其结构限制了剪枝的灵活性。Coruscant则在硬件效率和模型精度之间取得了更好的平衡。

*   **对通信开销的影响**：Coruscant的压缩格式不仅减少了GPU内部数据传输，也显著降低了模型权重在CPU/SSD与GPU之间传输的延迟，这对于offloading和多GPU并行场景非常有利。
*   **可扩展性与普适性**：
    *   **稀疏度范围**：在10-20%稀疏度下，Coruscant kernel的解压缩开销使其不如cuBLAS。在80-90%稀疏度下，Flash-LLM因其简单的坐标存储而表现更好，但高稀疏度通常会导致严重的模型精度损失。Coruscant在30-70%的目标稀疏度范围内表现最佳。
    *   **大N维度（Batch Size）**：随着N增大，计算量增加，cuBLAS的低寄存器使用和高Occupancy使其在N较大时更具优势。但Coruscant Sparse Tensor Core即使在N=256时仍能实现加速。考虑到当前LLM推理的KV Cache限制了Batch Size，Coruscant在常见Batch Size下仍能保持优势。
    *   **任务通用性**：在预填充（prefill）阶段，由于SpMM的计算特性，Coruscant可能比cuBLAS慢。但在解码（decode）阶段的累积加速可以抵消预填充的开销，对于长生成（如对话和推理任务）场景，Coruscant能带来显著的端到端加速。

**4. 结论**
Coruscant通过引入高效的位图式稀疏格式和协同设计的GPU kernel与Tensor Core，成功解决了LLMs中非结构化稀疏性在硬件上利用效率低下的问题。Coruscant GPU kernel在30%-70%稀疏度范围内实现了相对于cuBLAS高达2.02倍的加速。Coruscant Sparse Tensor Core通过将位图解码器集成到Tensor Core中，消除了软件解压缩开销，进一步将加速提升至2.75倍。这些改进显著减少了内存占用，提高了LLM推理的token生成吞吐量，同时保留了非结构化剪枝带来的模型精度优势，为LLM的部署提供了更高效的解决方案。

## SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs 
- https://dl.acm.org/doi/abs/10.1145/3689031.3717481 范瑞波 港科大（广）。EuroSys25 best paper

1. ✨ SpInfer提出了一种名为Tensor-Core-Aware Bitmap Encoding (TCA-BME)的新型稀疏矩阵存储格式，该格式通过高效的位图索引显著减少了索引开销，即使在低稀疏度（如30%）下也能实现有效的内存压缩。
2. 🚀 该框架还集成了一个高度优化的SpMM内核，该内核利用Shared Memory Bitmap Decoding (SMBD)和异步流水线设计，最大化GPU Tensor Core的利用率，并有效重叠内存传输与计算。
3. ⚡️ 实验结果表明，SpInfer在**30%至70%的稀疏度范围内，其SpMM内核性能显著优于现有的Flash-LLM和SparTA，并在端到端LLM推理中实现了高达1.58倍**的速度提升和优异的内存效率。更稀疏时（>80%），加速效果不如Flash-LLM。
<img width="511" height="351" alt="image" src="https://github.com/user-attachments/assets/f344570d-c2e7-4883-872a-9de4c90714b8" />

SpInfer是一项旨在通过利用非结构化稀疏性来加速GPU上大型语言模型（LLM）推理的高性能框架。该论文解决了非结构化剪枝（unstructured pruning）在LLM推理中难以实现其理论优势的关键挑战，主要原因是索引非零元素的存储开销和在低稀疏度（约50%）下稀疏矩阵乘法（SpMM）内核效率低下。

**背景与挑战**
LLM因其巨大的参数规模而面临内存和计算成本的挑战。权重剪枝（weight pruning）作为一种模型压缩技术，通过引入稀疏性来减少资源需求。非结构化剪枝因其灵活性和在保持模型精度方面通常优于结构化剪枝而备受关注。然而，对于LLM而言，其稀疏度通常只能达到50%左右，这带来了两个主要挑战：
1.  **索引开销：** 在低稀疏度下，传统稀疏格式（如CSR、Tiled-CSL）存储非零元素索引的开销可能抵消剪枝带来的内存节省。例如，Flash-LLM和cuSPARSE在50%稀疏度下压缩率（CR）低于1，意味着它们的存储量甚至可能高于原始稠密矩阵。
2.  **计算效率：** GPU上的SpMM内核在低稀疏度下难以超越其稠密对应物（cuBLAS）。现有的SpMM实现，即使是针对LLM剪枝的Flash-LLM，在50%或更低稀疏度时也难以实现加速，这使得非结构化剪枝的理论性能增益难以转化为实际效益。

**核心方法：SpInfer的设计**
SpInfer旨在通过其核心组件——Tensor-Core-Aware Bitmap Encoding (TCA-BME) 和高度优化的SpMM内核——来弥补这些差距。

**1. Tensor-Core-Aware Bitmap Encoding (TCA-BME)**
TCA-BME是一种新颖的稀疏矩阵存储格式，旨在最小化索引开销并最大化压缩率，同时与GPU Tensor Core架构对齐。
*   **分块设计 (Tiling Design)：** TCA-BME采用多级分块设计，将权重矩阵划分为不同粒度的Tile以匹配GPU硬件的不同层次：
    *   **BitmapTile (BT)：** 尺寸为 $8 \times 8$，是TCA-BME格式中最小的粒度单元，直接对应于Tensor Core的最小计算单元。它使用一个 $64$ 位位图（`uint64_t`）来表示其中的稀疏模式，每个位指示相应元素是否为非零。
    *   **TCTile (TT)：** 尺寸为 $16 \times 16$，由 $2 \times 2$ 个BitmapTile组成，与Tensor Core的`mma`指令（例如用于FP16精度的`mma.m16n8k16`）的矩阵形状对齐。BitmapTile在TCTile内以列主序排列，以简化解码过程。
    *   **GroupTile (GT)：** 包含多个TCTile，对应于线程块级别。TCTile在GroupTile内也以列主序存储，而GroupTile本身以行主序存储。
*   **存储结构：** TCA-BME使用三个数组来高效表示稀疏权重矩阵：
    *   `GTileOffset`：记录每个GroupTile在稀疏矩阵中的起始偏移位置（ $4$ 字节整型）。
    *   `Values`：存储所有非零元素（FP16精度， $2$ 字节），按GroupTile、TCTile、BitmapTile的嵌套顺序排列。
    *   `Bitmap`：存储所有BitmapTile的位图（ $8$ 字节整型），每个BitmapTile由一个 $64$ 位整数表示。
    这种格式的总存储开销计算为：
    $S_{TCA-BME} = 4B \times (N_{GT} + 1) + 8B \times N_{BT} + 2B \times N_{NZ}$
    其中 $N_{GT}$ 是GroupTile的数量，$N_{BT}$ 是BitmapTile的数量，$N_{NZ}$ 是非零元素的数量。与CSR和Tiled-CSL相比，TCA-BME在低至中等稀疏度（30%-70%）下实现了更高的压缩率（CR > 1），显著降低了索引开销。

**2. 高性能SpInfer-SpMM内核设计**
SpInfer的SpMM内核包含了以下关键优化：
*   **高效数据移动 (Efficient Data Movement)：**
    *   利用 `LDGSTS.128` 异步矢量化内存访问指令，将稀疏权重数据 (GTile) 从全局内存直接加载到共享内存，绕过L1缓存和寄存器文件，以提高全局内存带宽利用率。通过预处理在GTile内对`Value`数组进行填充，确保8字节对齐以实现128位矢量化。
    *   利用 `LDSM.M88` (PTX `ldmatrix.x4`) 指令将输入矩阵 $X$ 的数据从共享内存加载到寄存器，并自动调整数据布局以适应Tensor Core计算。
    这种数据移动路径近似于cuBLAS的理想情况，减少了通过寄存器文件进行不必要往返的开销。
*   **共享内存位图解码 (Shared Memory Bitmap Decoding, SMBD)：**
    SMBD是SpInfer-SpMM内核的关键优化，它在共享内存中高效地解压缩位图编码的WTile到寄存器文件，确保为Tensor Core计算准备好正确的布局。
    *   **寄存器分布：** 在Warp级别的Tensor Core操作中，一个Warp（32个线程）共同处理操作数矩阵的片段。每个线程持有部分操作数矩阵，其分布必须与Tensor Core `mma`指令的要求对齐。对于FP16计算，`mma.m16n8k16`指令在 $16 \times 16$ 矩阵片段上操作。每个线程的寄存器（如`Ra0-Ra3`）需要填充解码后的非零值。
    *   **两阶段解码过程：** SMBD将解码过程分为两个阶段，以高效处理半精度值：
        *   **阶段I (Decoding a0)：** 每个线程解码其32位寄存器中的第一个半精度值（a0）。线程 $i$ 检查位图的第 $(2i)$ 位。如果该位为1，线程使用 `MaskedPopCount` （基于NVIDIA GPU内置函数 `__popcll` 计算 $64$ 位位图中1的个数）来计算在其位置之前有多少个非零值，并从压缩的`Values`数组中加载相应的值。如果该位为0，则加载零值。
        *   **阶段II (Decoding a1)：** 每个线程解码同一32位寄存器中的第二个半精度值（a1）。线程 $i$ 检查位图的第 $(2i+1)$ 位。此时无需额外的 `MaskedPopCount`，直接复用阶段I的结果，若a0是非零值，则偏移量加1以加载a1。这种重用减少了位计数操作的数量。
    通过这些操作，SpInfer并行高效地解码压缩的矩阵片段，避免了全局内存中显式存储偏移量的需求。
*   **异步流水线设计 (Asynchronous Pipeline Design)：**
    SpInfer采用细粒度的异步流水线来最大化内存传输和Tensor Core计算之间的重叠。
    *   **双缓冲机制：** 为GTile和XTile实现双缓冲，使得可以在当前迭代数据计算的同时，异步预取下一迭代的数据到共享内存，从而隐藏内存加载延迟。
    *   **细粒度异步组管理：** 使用两个独立的 `cp.async` 组来独立管理GTile和XTile的加载，实现更高的并发性。
        *   一旦GTile加载完成，SMBD立即开始，与XTile加载并发执行，有效隐藏SMBD的延迟。
        *   在当前Tile的Tensor Core计算指令发出后，下一Tile的SMBD立即开始。SMBD的位操作和计数在CUDA Core上运行，与Tensor Core指令独立，从而增加了指令级并行（ILP），优化了硬件资源利用率。

**性能评估**
SpInfer在RTX4090和A6000 GPU上进行了广泛评估，包括内核级和端到端LLM推理。
*   **内核性能对比：**
    SpInfer在40%到70%的稀疏度范围内，始终优于cuSPARSE、Sputnik、SparTA和Flash-LLM等主流SpMM实现，并且在低稀疏度（如40%）下能够显著超越cuBLAS。在RTX4090上，SpInfer平均比cuBLAS快1.79倍，比Flash-LLM快1.56倍。在50%稀疏度时，SpInfer比cuBLAS快1.66倍，而Flash-LLM和SparTA仅有微小提升（1.00倍和1.01倍）。微观分析显示，SpInfer消耗更少的寄存器，显著减少了DRAM访问，最小化了共享内存Bank冲突，并实现了更高的Tensor Core流水线利用率。
*   **端到端LLM推理：**
    SpInfer在OPT-13B、OPT-30B和OPT-66B模型上的端到端推理中，显著降低了延迟并提高了内存效率。在RTX4090上，SpInfer平均比Flash-LLM、FasterTransformer和DeepSpeed分别快1.35倍、1.42倍和1.49倍。最大的加速比为1.58倍。
    内存效率方面，SpInfer通过TCA-BME格式实现了模型权重与稀疏度几乎线性的内存缩减。例如，在60%稀疏度下，OPT-13B的内存占用仅为14.4 GB，比稠密基线减少了47.5%。SpInfer在单张RTX4090 GPU上能支持更长的输出序列和更大的Batch Size，而Flash-LLM等在相同条件下可能遇到OOM错误，显示出SpInfer在资源受限环境下的部署优势。

**局限性与讨论**
*   **Prefill阶段性能：** 在Prefill阶段，当批处理大小和序列长度较大时（操作变得计算密集型），SpInfer可能比cuBLAS_TC慢11.8%，因为其内存访问优化优势减弱，且位图解码引入了开销。
*   **稀疏度限制：** 在极高稀疏度（>90%）下，位图索引效率下降，可能不如CSR格式。
*   **动态激活稀疏性：** SpInfer目前不支持动态激活稀疏性，这需要更自适应的稀疏编码技术。
*   **通用性：** 尽管SpInfer为NVIDIA Tensor Cores优化，其核心技术（TCA-BME的分块策略、SMBD的位操作）可推广到其他硬件架构，如Google TPU、AMD Matrix Cores和Intel AMX，通过调整Tile配置和利用通用位操作实现跨平台效率。

**总结**
SpInfer是首个能够有效加速LLM在低稀疏度（低于50%）推理的框架，同时保持计算效率和内存节省。它通过创新的TCA-BME格式和高度优化的SpMM内核（包括SMBD和异步流水线）解决了现有稀疏推理技术的关键瓶颈。SpInfer显著超越了现有最先进的SpMM内核和推理框架，弥补了LLM剪枝理论优势与实际性能之间的差距。
  
## GPAS pre-LN


https://arxiv.org/abs/2506.22049 港科大等 2025.7
https://github.com/dandingsky/GPAS


<img width="1199" height="269" alt="image" src="https://github.com/user-attachments/assets/5d4d28a8-8b23-4669-97c3-31ee73b6e9d2" />
1. 🧮 论文指出，Pre-LayerNorm (Pre-LN) Transformer在预训练过程中存在激活方差指数级增长的问题，这限制了深层网络的学习能力并导致收敛速度变慢。
2. 💡 为解决此问题，作者提出了**Gradient-Preserving Activation Scaling** (GPAS)，该方法通过缩放中间激活来降低方差，同时利用stop gradient机制保留梯度幅值以避免梯度消失。
3. 🚀 实验结果表明，GPAS能显著加速LLM预训练的收敛并提升下游任务性能，其在从71M到1B的多种模型尺寸和包括Pre-LN、DeepNorm等不同架构上均展现出一致的有效性。
可以用在西湖大学等LNScaling（LNS）上。实验结果中LNS表现已经非常不错。

<img width="857" height="323" alt="image" src="https://github.com/user-attachments/assets/16b00113-ffd4-46f3-ac85-aed6df7942e6" />
<img width="857" height="288" alt="image" src="https://github.com/user-attachments/assets/c8958696-4643-4ef9-a6b7-999f59e84631" />

## Multi-token attn
https://arxiv.org/abs/2504.00927 Facebook Research, 2025.7

https://github.com/facebookresearch/RAM/tree/main/projects/mta 

1. 现有的LLM注意力机制受限于仅依据单个查询和键token的相似性来决定权重，这限制了其利用丰富上下文信息的能力，尤其在长文本处理中易受瓶颈影响。
2. 🏆 为解决此问题，论文提出多token注意力（MTA），它通过对查询、键和注意力头应用卷积操作，使得注意力权重能够同时考虑多个向量信息，从而实现更精细的上下文定位。
3. 🚀 广泛的实验表明，MTA在标准语言建模任务和长上下文信息检索任务上均显著优于Transformer基线模型，证明了其利用更丰富信息提升性能的有效性。

该论文提出了一种名为Multi-Token Attention (MTA) 的新型注意力机制，旨在解决现有LLM中“单Token注意力”的瓶颈问题。传统的注意力机制（如Transformer中使用的）通过计算单个查询Token和单个键Token向量之间的相似性来确定注意力权重。这种方法在需要模型基于多个Token的信息来识别相关上下文时，会限制其捕捉更丰富、更细微信息的能力。例如，若需定位同时提及“Alice”和“rabbit”的句子，单个查询Token难以有效编码这两种信息，即使使用不同的注意力头也无法直接组合这些注意力图。
<img width="982" height="430" alt="image" src="https://github.com/user-attachments/assets/385deffb-f01c-47b0-a54f-fae33d19b8e1" />

MTA通过对查询、键和注意力头应用卷积操作，允许注意力权重同时依赖于多个相邻的查询和键向量，以及不同注意力头之间的信息。这使得MTA能够利用更丰富的上下文信息，从而实现更精确的注意力聚焦。

**核心方法 (Core Methodology)**

MTA是在标准多头注意力机制的基础上进行修改的。标准多头注意力（Multi-head Attention）的计算如下：
给定隐藏状态 $H \in \mathbb{R}^{T \times D}$，
<img width="905" height="356" alt="image" src="https://github.com/user-attachments/assets/76b53b5d-ce0c-4bd8-9fa3-4cbb84b7a071" />

其中，Softmax在键维度上操作，并且应用了因果掩码。

MTA引入了三个主要组件：

1.  **键-查询卷积 (Key-query convolution)**
    MTA在注意力对数（$\hat{A}$）或注意力权重（$A$）上应用卷积操作，以整合来自多个查询和键Token的信息。
    *   **Pre-softmax 卷积：** 在Softmax之前对注意力对数进行卷积。
        $$ A = \text{Softmax}(\text{Conv2d}_\theta(\text{Mask}_0(\hat{A}))) $$
        其中 $\text{Conv2d}_\theta$ 是一个2D卷积操作，其核权重为 $\theta$，核大小为 $(c_q, c_k)$。卷积在键和查询的长度维度上进行。为了保持因果性，只使用过去的查询Token，并在键维度上通过掩码1$_{i \ge j-j'}$（或简化为两次因果掩码）来避免信息泄露。
    *   **Post-softmax 卷积：** 在Softmax之后对注意力权重进行卷积。
        $$ A = \text{Mask}_0(\text{Conv2d}_\theta(\text{Softmax}(\text{Mask}_{-\infty}(\hat{A})))) $$
    每个注意力头都拥有独立的 $\theta$ 参数，使得它们可以执行不同的卷积操作。核维度 $c_q$ 和 $c_k$ 决定了可以组合的Token的距离范围。

2.  **头部混合卷积 (Head mixing convolution)**
    为了在不同注意力头之间共享知识并混合注意力权重，MTA引入了头部卷积。它将所有注意力头分为 $M/c_h$ 组，并在每组内应用一个非重叠的卷积操作（实际上是一个全连接操作，因为核大小 $c_h$ 与组大小相同）。
    *   **Post-softmax 头部混合：** 假设 $A^1$ 和 $A^2$ 是两个头的注意力权重，新的注意力权重可以表示为：
        $$ A^1_{\text{new}} = w_{11}A^1 + w_{12}A^2, \quad A^2_{\text{new}} = w_{21}A^1 + w_{22}A^2 $$
        其中 $w$ 是核权重。
    *   **Pre-softmax 头部混合：** 同样，可以在Softmax之前混合对数：
        $$ \hat{A}^1_{\text{new}} = w_{11}\hat{A}^1 + w_{12}\hat{A}^2, \quad \hat{A}^2_{\text{new}} = w_{21}\hat{A}^1 + w_{22}\hat{A}^2 $$
    当键-查询卷积和头部混合卷积都采用 pre-softmax 形式时，它们可以合并为一个3D卷积操作，其中两个维度是键和查询维度，第三个维度是注意力头维度。

3.  **带门控机制的组归一化 (Group normalization with gating mechanism)**
    为了对抗残差连接并改善梯度流，MTA在注意力输出之后应用了带标量门控（sigmoid gating）的组归一化。这允许模型在不同层之间开关注意力头，更好地适应不同任务。

**实验结果 (Experimental Results)**
<img width="982" height="289" alt="image" src="https://github.com/user-attachments/assets/f3f7cc24-c4a4-4244-ad71-8daecb8a260a" />
<img width="980" height="230" alt="image" src="https://github.com/user-attachments/assets/455e3989-dd45-47a3-84b8-da6eb372a512" />
<img width="980" height="366" alt="image" src="https://github.com/user-attachments/assets/e93c31cf-20b5-496a-8654-b0fd3bac5f46" />

*   **动机性玩具任务 (Motivating Toy Task):** 在一个需要模型定位包含多个给定字母块的任务中，标准Transformer难以学习（错误率高达51.6%），而MTA几乎完美解决了任务（错误率0.1%）。这验证了MTA能够利用卷积聚合多Token信息的能力。
*   **大规模语言建模 (Large Language Modeling):**
    *   在SlimPajama数据集上对8.8亿参数模型进行预训练，MTA在所有验证集上均优于Transformer基线、Differential Transformer和Talking Heads attention，验证困惑度（PPL）更低（MTA: 10.91 vs Transformer: 11.25）。
    *   在标准基准测试（如BoolQ, PIQA, MMLU等）的零样本设置中，MTA也取得了更高的平均分数。
*   **长上下文微调 (Long Context Finetuning):**
    *   将预训练模型在更长上下文（4096 Token）上进行微调后，MTA模型依然在困惑度评估中表现优异。
*   **长程依赖任务 (Long-range Dependency Tasks):**
    *   **LAMBADA:** 在需要长程依赖的词预测任务中，MTA模型的困惑度显著低于基线（MTA: 13.2 vs Transformer: 17.6）。
    *   **Needle-in-the-Haystack:** 在2k和4k上下文窗口中插入多根“针”并要求模型检索时，MTA在不同“针”数量和插入深度下都显示出显著的提取能力提升。
    *   **BabiLong:** 在需要从长文档中推理事实的问答任务中，MTA在存在大量干扰文本时，表现明显优于其他模型。
*   **核模式分析 (Kernel Patterns):** 论文分析了键-查询和头部卷积核的学习模式。发现部分键-查询核接近单位矩阵，但也有许多复杂的核，例如斜对角线核（用于序列匹配）或用于“启动”（priming）和“边缘检测”（edge detecting）的核。头部核模式则相对简单，常表现为身份缩放或对比（相减）模式。
*   **消融研究 (Ablation Studies):**
    *   键-查询卷积层的数量：即使仅在少量层（如2层）中应用MTA，模型也能超越强基线，而6层MTA能在性能与复杂度之间取得平衡。
    *   头部卷积核大小：增大核大小（增加头部间通信）能提高模型性能。
    *   核初始化：身份初始化（即初始时行为类似于普通Transformer）能带来更好的收敛和最终性能。
    *   组件组合：组归一化和指数深度缩放是MTA性能提升的重要因素。
*   **缩放定律 (Scaling Laws):** 在3亿、5.5亿和10亿参数模型上，MTA始终表现出优于基线的性能，展现了其在不同模型规模下的一致优势。
*   **集成到现有模型 (Finetuning with MTA):** MTA可以作为附加层集成到已训练好的模型中，并通过持续训练更新权重。实验表明，这种方法不仅能够融合新的卷积核，还能使模型在困惑度方面超越基线。

**结论 (Conclusion)**

MTA通过引入键-查询卷积和头部混合卷积，解决了传统注意力机制中单Token瓶颈的问题。它允许模型利用更细粒度的信息进行注意力聚焦。通过在从玩具任务到大规模LLM的广泛评估，论文证明了MTA在语言建模和长上下文信息检索任务中的优越性，尤其在需要精确识别相关信息的情况下，MTA表现出显著的性能提升。尽管目前MTA的实现未针对优化内核进行优化，导致运行时性能较低，但其在模型能力上的提升预示了其在未来LLM架构中的潜力。

## CAT:稀疏压缩token
Attention and Compression is all you need for Controllably Efficient Language Models

https://arxiv.org/abs/2511.05313 纽约大学 2025.12.1

https://github.com/rajesh-lab/cat-transformer

1. 🎯 针对现有高效Transformer模型在提高效率时常牺牲上下文召回性能、且缺乏灵活性的问题，本文提出了一种名为Compress & Attend Transformer (CAT) 的新架构，它**结合了密集注意力机制和序列压缩技术**。
2. 💡 CAT通过**并行压缩历史令牌块**并让解码器关注这些压缩表示来生成当前令牌块，其独特之处在于**可在训练时支持多种块大小，从而实现测试阶段在不重新训练的情况下动态调整质量与计算效率的权衡**。
3. 🚀 最大1B模型，单个自适应CAT模型在语言建模、常识推理和长上下文理解等任务上全面超越了许多现有高效基线，并能以1.4-3倍的速度和2-9倍的内存效率，达到与密集Transformer相当甚至更优的性能。

这篇论文提出了一种名为 Compress & Attend Transformer (CAT) 的语言模型架构，旨在解决现有Transformer模型中注意力机制的二次计算成本和线性内存成本问题，同时避免现有高效方法在质量、计算-内存权衡灵活性和架构设计复杂性方面的缺陷。
<img width="659" height="269" alt="image" src="https://github.com/user-attachments/assets/1dc54b5b-a4cf-44d5-9d19-f9c9794d5ada" />

<img width="611" height="587" alt="image" src="https://github.com/user-attachments/assets/9c29df7b-0f4d-446e-9abd-764e63e34009" />
<img width="997" height="551" alt="image" src="https://github.com/user-attachments/assets/e12a1dd9-49db-422f-990b-7ff424099622" />

**核心问题与现有方法的局限性：**
标准Transformer模型中的自注意力机制导致其计算成本与序列长度呈二次关系，内存成本呈线性关系，这使得大型语言模型（LLMs）的部署变得昂贵。为了提高效率，出现了多种方法：
1.  **稀疏注意力与滑动窗口注意力（Sparse and Sliding Window Attention）**：通过启发式地限制注意力范围来减少计算量，但通常会牺牲模型质量，尤其是在上下文召回（in-context recall）性能上。
2.  **线性注意力（Linear Attention）**：依赖固定大小的循环状态，实现了恒定的计算和内存成本，但在管理长序列信息时表现不佳，同样损害上下文召回性能。
3.  **递归压缩（Recursive Compression）**：虽然可以避免固定内存瓶颈和启发式限制，但其序列计算特性导致训练缓慢且优化困难。
4.  **混合架构（Hybrid Architectures）**：为了提升性能，上述方法常需要与密集注意力层精心组合，这使得架构设计过程复杂，难以规模化。
此外，现有方法在训练前就固定了计算-内存预算，无法在测试时根据不同任务需求灵活调整，导致次优性能或需要训练多个模型，成本高昂。

**CAT 架构的核心思想与方法：**
CAT 架构概念简单，仅包含两个基本组成部分：密集注意力和压缩。其核心思想是通过对序列的过去部分进行压缩表示，然后解码当前token块。

**1. 压缩与解码（Compression and Decoding）：**
*   **分块（Chunking）**：给定一个包含 $N$ 个token的序列 $x = (x_1, x_2, \dots, x_N)$，CAT 首先将其分割成 $N/C$ 个块，每个块 $c_i = (x_{C \cdot i+1}, \dots, x_{C \cdot i+C})$ 包含 $C$ 个token。
*   **压缩（Compression）**：CAT 使用一个“压缩器”（compressor）$f_\theta$ 对每个token块 $c_i$ 进行并行压缩，生成一个固定维度的块表示 $f_\theta(c_i) \in \mathbb{R}^{D_g}$。压缩器 $f_\theta$ 本身是一个密集双向Transformer，具有隐藏层大小 $D_f$，并通过线性投影输出 $D_g$ 维度的表示。
*   **解码（Decoding）**：压缩后，CAT 使用一个“解码器”（decoder）$g_\theta$ 从这些压缩的块表示 $\{f_\theta(c_1), \dots, f_\theta(c_{N/C})\}$ 中解码原始序列 $x$。解码器 $g_\theta$ 是一个因果密集Transformer，其隐藏层大小为 $D_g$。解码器以自回归方式解码每个块，即解码块 $c_i$ 中的每个token $x_{i,j}$ 时，它会将当前块中已解码的token $\{x_{i,j-1}, \dots, x_{i,1}\}$ 以及所有过去的压缩块表示 $\{f_\theta(c_1), \dots, f_\theta(c_{i-1})\}$ 作为输入。形式上，token $x_{i,j}$ 的预测分布定义为：
    $p_\theta (c_i | c_{i-1} \dots c_1) = \prod_{j=1}^{C} g_\theta (x_{i,j} | x_{i,j-1}, \dots, x_{i,1}, f_\theta(c_{i-1}), \dots, f_\theta(c_1))$
    这种通过压缩块表示进行解码的方式，有效减少了所需的计算和内存。块大小 $C$ 越大，效率提升越显著。

**2. 训练与测试时控制（Training for Test-Time Control）：**
*   **块大小作为控制旋钮**：CAT 中的块大小 $C$ 是一个关键参数，它直接权衡了模型质量与计算效率。
*   **自适应训练**：为了实现测试时的灵活控制，CAT 可以同时使用多个块大小进行训练。在每个训练迭代中，模型会均匀采样一个块大小 $C$，并通过一个可学习的指示token告知CAT当前操作的块大小。训练完成后，同一个CAT模型无需再训练，只需改变指示token，即可在测试时以不同的计算/内存预算运行，从而实现质量-效率的自适应调整。

**3. 高效与可扩展的实现（Fast and Scalable Implementation）：**
*   **并行压缩**：块的压缩是高效且并行的，例如可以使用 `torch.vmap` 对所有块进行操作，其自注意力计算成本为 $O(\frac{N}{C} \cdot C^2) = O(NC)$，远低于 $O(N^2)$。
*   **可扩展训练**：为了解决解码器训练中的并行化难题（每个块需要关注的过去压缩块数量不同导致形状可变），CAT 提出了一种巧妙的解决方案：在原始序列中，每个块 $c_i$ 之后插入其压缩表示 $f_\theta(c_i)$，形成序列 $\{c_1, f_\theta(c_1), c_2, f_\theta(c_2), \dots\}$。然后将此序列输入解码器，并使用一个自定义注意力掩码（custom attention mask）。这个掩码允许块 $c_i$ 中的token只关注当前块内的先前token以及所有过去的压缩块表示 $\{f_\theta(c_1), \dots, f_\theta(c_{i-1})\}$，但不关注当前块之外的原始token。这种设计使得解码器能够并行计算所有块的 logits，并复用已计算的压缩块表示的 Key-Value 向量，从而将训练时的自注意力操作成本降低到 $O(\frac{N^2}{C})$，比标准Transformer的 $O(N^2)$ 提高了 $C$ 倍。
*   **高效推理（Generation）**：推理时，CAT 只保留过去压缩块的表示在 KV 缓存中，而丢弃原始token块。这使得 KV 缓存大小减少了 $C$ 倍，显著降低了内存占用和内存访问成本。解码器在生成时最多关注 $\frac{N}{C} + C$ 个token，大大减少了自注意力所需的计算量。CAT 的纯 PyTorch 实现就能与使用自定义 CUDA/Triton 核的高效架构媲美。生成过程是：并行计算所有 $f_\theta(c_i)$ 并预填充 KV 缓存，然后逐块自回归生成，每生成完一个块就将其压缩表示加入 KV 缓存。
<img width="905" height="633" alt="image" src="https://github.com/user-attachments/assets/71ae95f5-762a-42bf-a820-0ab5c4805758" />

**关键优势：**
*   **无需启发式选择或复杂规则**：CAT 直接基于密集Transformer抽象构建，不依赖于启发式注意力掩码、复杂的循环状态更新规则或层间混合。
*   **灵活且高效的内存使用**：通过压缩，CAT 的内存使用随序列长度线性增长，但增长速度显著放缓，解决了固定内存瓶颈，实现了优越的上下文召回性能。
*   **可扩展训练**：压缩和解码可以在训练过程中并行进行。
*   **测试时自适应**：训练单个模型即可在测试时根据需求调整质量-效率权衡，无需重新训练，降低了部署成本和复杂性。
*   **性能优越**：在语言建模、常识推理、上下文召回和长文本理解等任务上，单个自适应CAT模型在不同计算-内存预算下均优于多种现有高效基线（包括混合架构），在某些设置下甚至超越了密集Transformer，同时显著更快且内存效率更高。例如，CAT在语言建模方面与密集Transformer性能匹配，但速度快 1.4-3 倍，总内存使用量低 2-9 倍。

**模型实例化细节：**
为了与深度为 $L$、隐藏层大小为 $D$ 的密集Transformer竞争，CAT 采用了一个深度为 $L$、隐藏层大小为 $2D$ 的解码器，以及一个深度为 $L/4$、隐藏层大小为 $D$ 的压缩器。尽管这增加了参数数量（CAT模型参数量接近 1B，而基线约 300M），但由于压缩机制，CAT 在推理时依然能实现显著的速度和内存效率提升。论文训练的CAT模型同时支持 $C = \{4, 8, 16, 32\}$ 四种块大小。

**未来工作展望：**
CAT架构具有通用性，可以整合其他序列混合器（如线性注意力作为压缩器）。未来可以探索数据依赖的自适应性，例如通过强化学习让CAT模型根据上下文和任务自动分配计算预算。此外，CAT也可以被实例化为层级组件，与其他注意力层混合构成新的混合架构。

## ZIP-RC
ZIP-RC: Zero-overhead Inference-time Prediction of Reward and Cost for Adaptive and Interpretable Generation

https://arxiv.org/abs/2512.01457 伯克利 MIT LiquidAI等 2025.12.1

1. 💡 本文提出了ZIP-RC，一个零开销（zero-overhead）的自省推理框架，它通过重用大型语言模型（LLM）的**输出logits来实时预测最终奖励和剩余生成长度的联合分布**，解决了LL**M缺乏内省能力导致的推理效率**低下问题。
2. ⚙️ ZIP-RC利用预测的联合分布计算采样效用，并通过优化元动作（meta-actions）来最大化预期最大奖励、总计算量和延迟的线性组合，从而在**推理过程中自适应地分配计算资源**。
3. 📈 在**混合难度的数学基准测试中，ZIP-RC在相同或更低的平均成本下将准确率提高了多达12%**，并展示了在质量、计算和延迟之间平滑的Pareto frontiers，**显著优于Best-of-N**等现有基线方法。

<img width="736" height="642" alt="image" src="https://github.com/user-attachments/assets/53ec7f43-1670-4b0b-9198-a4a21129cba6" />

本文提出了一种名为 ZIP-RC（Zero-overhead Inference-time Prediction of Reward and Cost）的自适应推理方法，旨在赋予大型语言模型（LLMs）在推理时预测自身成功（奖励）和所需计算（成本）的能力，且不引入额外的推理开销。传统方法如 Best-of-N (BoN) 采样虽然能提升性能，**但其固定采样预算导致计算效率低下，尤其是在处理难度不一的任务时**；而现有的一些早期停止或剪枝方法通常**依赖于简单的标量信号或启发式规则，无法充分捕捉**奖励-成本之间的权衡，也无法量化增加样本的边际效益。ZIP-RC 通过在每次生成过程中预测一个完整的奖励和剩余长度联合分布，有效解决了这些局限性。

### 核心方法论：ZIP-RC

#### 1. Zero-overhead Inference-time Prediction (ZIP)
ZIP 是 ZIP-RC 的基础机制，其核心思想是**重用**语言模型输出头中**保留或未使用的 logits**，以在与下一个 token 预测相同的正向传播中实现辅助预测，而无需引入额外的模型、修改模型架构或增加推理开销。
<img width="1020" height="264" alt="image" src="https://github.com/user-attachments/assets/b4b9f817-ba6c-436e-afbc-4f8a71a9d74a" />


**为何预测联合分布而非标量**：标量值难以捕捉奖励-成本之间的权衡。例如，低置信度的轨迹可能因为接近完成而有价值，而高置信度的轨迹可能因预示着高昂且漫长的继续生成而变得不切实际。联合分布可以提供更丰富的信息，例如允许进行序统计计算 (order-statistic calculations) 来量化继续部分样本或生成额外样本的边际效益。

**为何预测期望奖励而非实际奖励**：预测期望值 $\hat{V}(s_T)$（通常由评估器给出）而非实际奖励 $R(s_T)$，能够更好地与 BoN 采样中的实际选择目标对齐，并允许在计算序统计期望时进行闭式（closed-form）推导，因为期望值 $V(s_T)$ 可以假定是独立的。

#### 3. ZIP-RC 采样（Test-time Compute）
ZIP-RC 采样将测试时间搜索建模为一个**元 MDP**（Meta-MDP）上的决策问题。
*   **元状态 (Meta-states)**：时间步 $t$ 的元状态是一个以提示 $x$ 为根的**前缀树（trie）** $S_t$，它编码了所有已处理或生成的token前缀。
*   **元动作 (Meta-actions)**：时间步 $t$ 的元动作是选择一个有限的多重集的前缀（节点） $A_t \subseteq \mathcal{M}(N_t)$ 来继续采样。多重性表示分支，例如，如果前缀 $s$ 在 $A_t$ 中出现 $r$ 次，则 $s$ 会独立地采样 $r$ 次。
*   **元奖励 (Meta-reward)**：在每个步骤中，会产生一个成本 $C(S_t, A_t) = \beta \alpha |\text{supp}(A_t)| + (1-\alpha) \max_{s \in A_t} L^\pi(s)$，其中 $|\text{supp}(A_t)|$ 是 $A_t$ 中不同前缀的数量（每次独立前向传播的成本），第二项是最大延迟成本（最长轨迹的步数）。$ \alpha \in [0, 1]$ 平衡计算和延迟，$\beta > 0$ 平衡奖励和成本。在最终时间步 $T$ 选定一个完成的生成 $s^*_T$，奖励为基础 MDP 奖励 $R(s^*_T)$。

ZIP-RC 采样的目标是选择能最大化采样效用（sampling utility）的元动作 $A_t$。采样效用 $U(S_t, A_t)$ 被定义为在预定义策略类别 $M_t$ 中，某个策略的元 Q 函数的最大值，这个 Q 函数近似了一个“带剪枝的 rollouts”策略的价值。这个 Q 函数的计算利用了 ZIP-RC 预测的联合分布：
$$ Q_{Rollouts}(S_t, A_t) = E[\max_{s \in A_t} Z^\pi(s)] - \beta \left( \alpha \sum_{s \in A_t} E[L^\pi(s)] + (1-\alpha) E[\max_{s \in A_t} L^\pi(s)] \right) $$
关键在于，为了实现基于剪枝的自适应，ZIP-RC 定义了一个**截断联合分布** $p_\theta(b, \ell | s; h_s)$。它将超过给定 horizon $h_s$ 的所有概率质量折叠到一个指定的“截断”状态中，同时将奖励成分折叠到一个指定的基础 bin $b_0$ 中，以反映因剪枝而放弃的奖励。这使得能够在形式上计算出在不同剪枝策略下，价值和剩余长度的期望值。
最终，ZIP-RC 采样定义为在每个元状态 $S_t$ 下，选择最大化采样效用 $U(S_t, A_t)$ 的元动作 $A_t$ 的策略。由于 ZIP-RC 采样在每个时间步重新优化 $A_t$，它能根据前缀树的随机演化进行在线自适应：如果当前轨迹预计成本高昂或价值低下，它可以立即将计算重定向到其他地方。

### 实验与结果
作者在混合难度的数学基准测试（AIME 2024, AMC 2023, MATH-500, GSM8K）上对 ZIP-RC 进行了评估，使用了 Qwen3-1.7B、LFM2-1.2B Math 和 LFM2-350M Math 三个不同规模的模型。
<img width="820" height="269" alt="image" src="https://github.com/user-attachments/assets/47e07e9d-0b23-41ad-9949-99a51c52fce2" />

主要发现包括：
1.  **ZIP-RC 预测准确性高**：ZIP-RC 预测的奖励-成本联合分布能够很好地校准并准确预测其自身 rollouts 的结果（表1，图2）。
2.  **平滑的质量-计算-延迟 Pareto 前沿**：通过调整效用函数中的参数 $\alpha$（平衡计算与延迟）和 $\beta$（平衡奖励与成本），ZIP-RC 采样能够描绘出平滑的 Pareto 前沿，在质量、计算和延迟之间实现可控的权衡。在计算限制（$\alpha=1.0$）和延迟限制（$\alpha=0.0$）两种设置下，ZIP-RC 均显著优于多数投票（Majority Voting, MV）等基线方法（图3）。
3.  **自适应推理能力**：在匹配生成成本的配置下，ZIP-RC 采样在所有模型和基准测试上均提升了准确性。在 AIME 2024 等更难的数据集上，准确率提升高达 12%（表2）。ZIP-RC 能够根据难度自适应地分配计算资源，对更困难的实例或能力较弱的模型分配更多样本，而在容易的问题上则积极剪枝。

### 结论
ZIP-RC 提供了一种零开销的内省推理框架，通过重用现有 logits 来预测未来的奖励和成本，从而实现了原则性的实时解码搜索。它能够在更低的平均成本下，比强大的 Best-of-N 基线模型提升高达 12% 的绝对准确率，并在质量、计算和延迟之间描绘出平滑的 Pareto 前沿。ZIP-RC 标志着从僵化、基于启发式的扩展到基于效用的自适应推理的转变，使得模型能够预测自身的成功和计算成本，是构建更自主、可靠和高效 LLM 的关键一步。
   
## Countdown分析RL能力
How Does RL Post-training Induce Skill Composition? A Case Study on Countdown

https://arxiv.org/abs/2512.01775 普林斯顿等 2025.12.1
中文解读：https://mp.weixin.qq.com/s/YVnHN9o7LZnLPBNwEPIWwQ

任务很简单：给定个数字和一个目标值，你需要用这些数字和四则运算（）构造一个表达式，使其结果等于目标值。

这个任务的巧妙之处在于，任何一个解（比如 ）都可以被唯一地解析成一棵“表达式树”。通过规范化处理，这棵树可以对应到一个唯一的、抽象的计算模式（Canonical Pattern），如下图所示。

<img width="847" height="188" alt="image" src="https://github.com/user-attachments/assets/ab07b3bb-20a1-44a4-bf3f-c60f518d5a31" />

图1：从模型生成的表达式到唯一的“计算模式”。这使得研究者可以精确分析模型采用的推理结构。

有了这个框架，我们就能清晰地区分两种泛化能力：

• 长度泛化（Length Generalization）：模型能解决比训练数据更长的问题（例如，用更多的数字）。
• 组合泛化（Compositional Generalization）：模型能用已知技能，组合出训练中从未见过的全新“模式树”。

1. 结构为王：决定推理难度的不是问题长度，而是其内在的“计算模式”结构。
2. 前瞻瓶颈：“先易后难”的右重结构是自回归模型的一个根本性弱点，限制了其组合能力。
3. RL激发创造：RL能够引导模型超越模仿，学会“无中生有”地组合技能，实现真正的组合泛化。
   
## SparseSpec
Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding

https://arxiv.org/pdf/2512.01278 MIT 伯克利 NV 韩松，清华等，2025.12.1

http://github.com/sspec-project/SparseSpec

1.  ✨ SparseSpec 提出了一种无损且**无需训练的**自推测解码框架，用于加速推理语言模型 (RLMs) 的推理，旨在解决**长生成过程**中 KV-Cache 内存访问带来的性能瓶颈。
2.  💡 该框架的核心是 **PillarAttn 稀疏注意力机制**，它通过**重用验证阶段的注意力分数来动态选择关键 token 进行草稿生成 加速attn部分**，并结合了**统一批调度器、延迟验证和动态 KV-Cache **管理等系统级优化。
3.  🚀 实验结果表明，SparseSpec 在各种模型和数据集上均表现出色，吞**吐量相比现有最先进的解决方案提升高达 2.13** 倍，同时保持了**高接受率和高效的资源利用**。
<img width="760" height="155" alt="image" src="https://github.com/user-attachments/assets/ac07d2fc-cad8-4ef0-9382-e88f958d6271" />

`SparseSpec` 是一种用于加速大规模推理语言模型 (RLM) 推理的无损且免训练的框架。该研究指出，由于RLM会生成冗长的 `chain-of-thought` 解决方案，推理瓶颈已从 `compute-bound` 转移到 `memory-bound`。具体而言，每个令牌的生成都需要对所有先前生成的 `Key-Value` (KV) 向量应用完整的注意力机制，导致对 `KV-Cache` 的**内存访问需求随生成长度的增加而急剧上升**，从而**对内存带宽造成巨大压力**。例如，在 NVIDIA H100 GPU 上，对于 Qwen3-8B 模型，在生成 8192 个令牌时，`KV-Cache` 加载平均占用端到端延迟的 70% 以上。
<img width="913" height="278" alt="image" src="https://github.com/user-attachments/assets/684380ae-3154-4fe5-9ccf-53fe7480b640" />

提出了一种名为 SparseSpec 的加速推理框架，专门针对大规模推理语言模型（RLMs）在长序列生成场景下的内存瓶颈。RLMs（如 OpenAI-o1）在复杂任务中通过生成详细的思维链（chain-of-thought, CoT）解决方案展现了强大的能力。然而，这种冗长的生成过程将推理瓶颈从计算密集型（compute-bound）转移到内存密集型（memory-bound），因为模型在生成每个 token 时都需要对所有先前生成的 token 应用完全注意力机制，这要求访问越来越大的 KV-Cache。随着生成长度的增加，每一步的内存访问需求也随之增加，对内存带宽造成巨大压力。
<img width="920" height="332" alt="image" src="https://github.com/user-attachments/assets/4fc97824-4ff2-4266-afc1-c6d5dcf5ea43" />

为了解决这一问题，SparseSpec 引入了一种自推测解码（self-speculative decoding）框架，它将同一个模型同时用作草稿模型（draft model）和目标模型（target model）。SparseSpec 的核心是一个新颖的稀疏注意力机制——PillarAttn，它通过巧妙地重用验证阶段的信息来准确选择关键 token 作为草稿模型。此外，SparseSpec 还与三项系统优化共同设计：
1.  **统一调度器（Unified Scheduler）**：批处理草稿阶段和验证阶段，以最大化并行性并缓解工作负载波动。
2.  **延迟验证（Delayed Verification）**：将验证操作延迟一个迭代，实现 CPU/GPU 重叠，将 CPU 密集型任务从关键路径中移除。
3.  **动态 KV-Cache 管理（Dynamic KV-Cache Management）**：支持将 KV-Cache 卸载到主机内存，以最大限度地提高 GPU 内存利用率。

通过这些优化，SparseSpec 在各种模型和数据集上都优于现有解决方案，吞吐量最高可提升 2.13 倍。

### 核心方法与技术细节

**1. PillarAttn：专为自推测解码设计的动态稀疏注意力**

RLMs 的推理生成通常涉及动态变化的上下文，导致关键 token 的注意力模式随时间显著变化。现有的稀疏注意力方法（如滑动窗口或基于规则的启发式方法）无法适应这种动态性，导致草稿 token 预测不准确，接受率较低。PillarAttn 通过以下方式解决此问题：

*   **动态稀疏模式（Dynamic Sparsity Pattern）**：PillarAttn 周期性地重新识别和更新草稿阶段使用的稀疏模式。它假设上下文语义具有空间局部性，因此在一个小步长（stride）内保持稀疏模式固定。这个步长与推测步长 $k$（即每次推测生成的候选 token 数量）相同。
*   **零开销识别（Overhead-free Identification）**：每次进行 $k$ 个草稿步后，会执行一个带有完全注意力机制的验证步。在此验证步中，模型会计算所有 KV-Cache 中 token 的注意力分数。PillarAttn 利用定制的注意力核在验证阶段“即时（on-the-fly）”地导出这些注意力分数。为了避免额外的计算和存储开销，PillarAttn 直接重用这些已导出的注意力分数。具体而言，它对这些分数应用 Top-K 采样，以识别出最重要的 token 作为稀疏模式。如果采用了分组查询注意力（Group Query Attention），这些分数会先在 $k$ 个草稿 token 和查询头（在同一组内）之间进行平均。这种方法确保了稀疏模式的识别几乎没有额外开销，并且能够根据推理过程中的实际注意力分布自适应地调整。

**2. 统一批调度器（Unified Batch Scheduler）**
<img width="1058" height="432" alt="image" src="https://github.com/user-attachments/assets/7c33b063-5541-475b-bc5a-8d5d117ee300" />

传统的推测解码调度器倾向于将草稿阶段和验证阶段分开处理，导致硬件利用率低下和工作负载波动，因为这两个阶段的资源需求（特别是 GEMM 操作的输入大小）存在显著差异。SparseSpec 的统一批调度器通过以下方式解决这个问题：

*   **统一抽象（Uniform Abstraction）**：由于自推测解码中草稿模型和目标模型共享相同的权重，GPU 上的数据和控制流是相同的，仅在注意力类型（稀疏或完全）上有所不同。PagedAttention 本质上是一种按页粒度的稀疏注意力实现。通过将页大小设置为 1，PillarAttn 能够将稀疏注意力（用于草稿）和完全注意力（用于验证）统一到一个流水线中，从而简化系统设计并提高调度灵活性。
*   **工作负载感知调度（Workload-aware Scheduling）**：SparseSpec 在每个生成步中均匀地混合来自草稿阶段和验证阶段的请求。它维护 $k$ 个桶来跟踪每个草稿阶段的请求数量，并将新的传入请求分配到负载最低的桶中，从而实现工作负载的均匀分布，避免了批处理大小的剧烈波动，确保了 GEMM 操作的输入批大小保持稳定，从而提高了硬件利用率。
*   **融合稀疏与完全注意力核（Fused Sparse and Full Attention Kernel）**：为了进一步提高硬件效率，SparseSpec 引入了一个融合注意力核。该核采用持久化核（persistent-kernel style）的方式，能够根据草稿和验证阶段的不同注意力类型（稀疏与完全）在芯片上动态调度其最佳的核配置（如 tile size 和 MMA 指令），而不是启动两个独立的核。这使得在稀疏和完全注意力之间实现最佳性能，避免了由于注意力操作异构性导致的低硬件利用率。

**3. 延迟验证（Delayed Verification）**

推测解码范式要求 CPU 在 GPU 完成验证后进行同步，以处理被拒绝的 token 并更新元数据。这种显式同步会使 CPU 操作成为关键路径上的瓶颈。SparseSpec 的延迟验证机制通过以下方式解决了这一问题：

*   **异步 CPU-GPU 执行**：SparseSpec 观察到，这种同步主要影响处于验证阶段的请求，这些请求只占整个批处理的一小部分（约 $1/(k+1)$）。因此，SparseSpec 允许非验证请求的元数据准备在 CPU 上直接进行，无需等待 GPU 的验证结果。只有验证请求（前一个迭代的）会被暂停，并从当前迭代中取出。在从 GPU 获得验证结果后，这些请求会在下一个迭代（例如 $i+1$ 迭代）中被重新提交到 GPU 执行。这样，CPU 的验证工作可以与 GPU 的计算重叠，从而减少了关键路径上的延迟。

**4. 动态 KV-Cache 管理（Dynamic KV-Cache Management）**

RLMs 输出长度的巨大方差使得 KV-Cache 管理变得复杂，难以在不发生重新计算（recomputation）的情况下实现高利用率。SparseSpec 采取了以下策略：

*   **激进的 CPU 卸载（Aggressive CPU Offloading）**：SparseSpec 倾向于激进地增加请求并发度以充分利用 KV-Cache 容量。当 GPU 内存接近不足时，KV-Cache 会被异步、分块（chunk-by-chunk）地卸载到主机内存，以避免重新计算。这种卸载操作的开销可忽略不计，因为每步生成的 KV-Cache 数据量相对较小，所需的带宽远低于 PCIe 带宽限制。
*   **FIFO 调度与内存优先（FIFO with Memory Prioritization）**：卸载和加载操作遵循 FIFO 顺序，确保公平性并避免饥饿。SparseSpec 会优先调度已被卸载的请求，一旦 GPU 内存可用，这些请求就会被加载回来。

### 实验结果
- Qwen3 1.5b/7b/14b：分别H100 TP1/2/4。最大长度17k（AIME23？），**2048个batch**（有点太大）只测吞吐

<img width="1061" height="646" alt="image" src="https://github.com/user-attachments/assets/f90e2763-6226-433c-82b5-dcda1f852571" />

SparseSpec 在 Qwen3-1.7B/8B/14B 等模型上，并在 AIME、OlympiadBench 和 LiveCodeBench 等真实世界推理工作负载上进行了评估。结果显示：
*   相较于最先进的服务框架 vLLM，SparseSpec 实现了高达 **2.13 倍** 的吞吐量提升。
*   相较于现有无训练方法（vLLM-NGram、MagicDec 和 TriForce），SparseSpec 分别实现了高达 **1.56 倍、1.36 倍和 1.76 倍** 的吞吐量提升。
*   即使与需要额外训练的基于草稿模型的方法 EAGLE3 相比，SparseSpec 仍能实现更好或相似的吞吐量，且无需额外训练成本。
*   **接受率**：PillarAttn 在草稿 8 个 token 时，平均接受 token 长度为 6.16，远超其他草稿方法（NGram 和 EAGLE3 接受率低于 2）。
*   **消融研究**：各项设计组件的贡献显著，统一批调度器、动态 KV-Cache 管理和延迟验证分别带来了 1.23 倍、1.61 倍和 1.12 倍的性能提升，总计提升了 2.22 倍。
*   **GEMM 批大小和计算利用率**：统一批处理使得 GEMM 输入批大小在迭代中保持稳定，避免了传统调度的波动和硬件利用率不足。
*   **融合注意力核性能**：融合核相较于顺序执行和简单批处理分别实现了 1.3 倍和 1.8 倍的加速。
*   **内存利用率**：SparseSpec 几乎完全利用了可用的 GPU 内存，且没有引入重新计算的开销，卸载操作仅导致平均 0.5% 的周期时间延长。

### 总结

SparseSpec 针对推理语言模型长序列生成中的内存瓶颈，提出了一种创新性的无损、无训练加速框架。它通过 PillarAttn 动态稀疏注意力机制，结合统一批调度、延迟验证和动态 KV-Cache 管理等系统级优化，显著提升了 RLMs 的推理吞吐量。其核心在于算法与系统协同设计，既提升了推测解码的准确性（高接受率），又解决了实际部署中的系统效率问题，为大规模 RLMs 推理提供了高效的解决方案。


## layerNorm scaling
The Curse of Depth in Large Language Models 西湖大学等，NIPS2025 

https://arxiv.org/pdf/2502.05795 

https://github.com/lmsdss/LayerNorm-Scaling 

1. 🎯 该研究提出了大语言模型（LLM）中存在的“深度诅咒”现象，即广泛使用的Pre-Layer Normalization (Pre-LN) **导致深层网络输出方差呈指数级增长**，使得**深层Transformer模块对训练的贡献降低**。
2. 💡 为解决此问题，作者提出了LayerNorm Scaling (LNS) 方法，通过将**层归一化输出按其深度平方根的倒数进行缩放**，有效抑制了深层方差的爆炸，从而增强了深层模型的贡献。
3. 🏆 实验证明，LNS在多种LLM模型尺寸上显著提升**了预训练性能并改善了下游任务的微调效果**，同时在Vision Transformer中也展现出稳定方差和提升性能的潜力，验证了其普适性和有效性。

该论文引入了“深度诅咒”（Curse of Depth, **CoD**）这一概念，旨在揭示、解释并解决现代大型语言模型（LLMs）中**近半数深层**（Transformer blocks）不如预期有效的问题。作者首先在Llama、Mistral、**DeepSeek和Qwen等主流LLM家族中**广泛证实了CoD现象的存在，通过层剪枝实验（layer pruning experiments）和角度距离（angular distance）分析发现，深层模块对模型性能贡献甚微，且其表征（representations）高度相似，这表明它们未能执行有意义的转换。
<img width="832" height="382" alt="image" src="https://github.com/user-attachments/assets/f312881c-739e-41d6-9a61-996caea408a7" />

论文从理论和实证两方面指出，CoD的根本原因在于广泛使用的预层归一化（Pre-Layer Normalization, Pre-LN）。Pre-LN虽然稳定了Transformer LLMs的训练，但其输出方差（output variance）会随模型深度呈指数级增长。具体而言，对于第\(\ell\)层的输入\(x_\ell\)和中间输出\(x'_\ell\)，其方差增长趋势可表示为：
<img width="1354" height="382" alt="image" src="https://github.com/user-attachments/assets/ab981c35-77b3-4b01-94e5-d7d80440ff21" />
<img width="967" height="271" alt="image" src="https://github.com/user-attachments/assets/f082d129-af34-4abc-8b87-3fd85e833fb8" />

<img width="1019" height="375" alt="image" src="https://github.com/user-attachments/assets/28b6d3e1-1ba0-4bbc-9a5f-96b6cbf76856" />

<img width="615" height="375" alt="image" src="https://github.com/user-attachments/assets/a0c3dcfc-eea5-49f4-b671-fa1ee0439b2d" />


实验结果表明，LNS在130M到**7B等多种模型规模上**，始终**优于以往的归一化和缩放技术**，显著提升了LLM的预训练性能（降低了困惑度Perplexity），并将这种改进无缝地传递到监督微调（Supervised Fine-tuning,** SFT）阶段**。具体表现为：
1.  在不同LLaMA模型尺寸上，LNS均取得了最低的困惑度，且训练过程稳定，而DeepNorm和Mix-LN等方法在较大模型上表现出不稳定性。
2.  在SFT任务中，LNS模型在Commonsense170K数据集的**八个下游任务上性能一致超越其他归一化技术**，提升了深层特征表示的质量。
3.  在OLMo和**Qwen2.5等先进架构上进行大规模训练**时，LNS同样展现出强大的可扩展性和有效性，甚至超越了OLMo的Scaled Initialization策略。
4.  机制分析证实，LNS有**效控制了输出方差的增长**，并**促使深层模块学习到更具多样性的特征**（通过更高的角度距离和更均匀的性能下降来体现）。
5.  初步实验还表明，LNS机制（在不同插入位置）对Vision Transformer（ViT）也有效，展现了其在不同架构上的泛化能力。
6.  消融研究（Ablation Study）表明，LNS优于其他缩放方法和LayerNorm变体，且最佳的插入位置是LayerNorm之后。

总之，LNS作为一种简单、无需超参数调优且不引入额外参数的方法，有效解决了LLM中的深度诅咒问题，提升了模型的训练效率和最终性能。

## Stabilizing Reinforcement Learning with LLMs:Formulation and Practices

https://arxiv.org/pdf/2512.01374 Qwen团队 2025.12.1

基于Qwen3-30b-a3b 数学类，FP8推理 BF16训练。数十万卡时
结论：
on policy 1200step：TIS类的比较重要，route replay能稳定 但模型上限不高；
off olicy时（>=2400 step），TIS + 必须route replay

解读：https://mp.weixin.qq.com/s/rUIN-oaBdX91BnqwKSFANg 

## gLLM
gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling

https://arxiv.org/pdf/2504.14775 中山大学 卢宇彤团队，SC2025
https://github.com/gty111/gLLM


1.  ✨ 针对分布式LLM服务中管道并行（pipeline parallelism）因预填充（prefill）和解码（decode）阶段计算不平衡导致管道**气泡（pipeline bubbles）和GPU利用率低**的问题，现有方法未能有效解决。
2.  💡 本文提出了gLLM系统，引入了Token Throttling机制，该机制基于**全局系统信息（如待处理token和KV缓存利用率）独立调控预填充和解码token的数量**，结合异步运行时架构，实现了批次间计算的全局平衡。
3.  🚀 实验结果表明，gLLM相比现有最先进的TP/TPP并行系统，在吞吐量上提升了11%至398%，同时保持了更低的延迟。

本文介绍了gLLM，一个用于分布式LLM服务、采用Token Throttling机制的全局均衡流水线并行系统。流水线并行（pipeline parallelism）因其较低的通信开销，已成为在分布式节点上部署大型语言模型（LLM）的主流方法。然而，尽管在请求服务中展现出高吞吐量，流水线并行通常会因流水线气泡（pipeline bubbles）而受到性能限制，这些气泡主要源于批次间计算延迟的不平衡。

**研究问题与现有方法不足：**
现有方法，如Sarathi-Serve，试图通过混合调度分块（chunked）的预填充（prefill）和解码（decode）令牌，并使用固定的令牌预算来解决此问题。然而，此类方法可能因预填充令牌不足或解码令牌分布不均而导致显著波动，最终造成计算不平衡。此外，预填充与解码阶段的计算特性截然不同：预填充通常能充分利用GPU容量，而解码操作的GPU利用率显著较低。一些研究提出了预填充-解码分离架构来缓解干扰，但仍面临各阶段内部批次计算不平衡以及GPU资源分配动态调整的挑战。
<img width="576" height="494" alt="image" src="https://github.com/user-attachments/assets/3b5bb8e1-ed13-4127-a4c2-540b2476c4d1" />
<img width="576" height="364" alt="image" src="https://github.com/user-attachments/assets/97c10884-2e00-46f4-8cee-acbba744764e" />
<img width="573" height="263" alt="image" src="https://github.com/user-attachments/assets/f2697bac-30e3-49cd-aed1-14372f595bf1" />

**gLLM的核心贡献与方法：**
为了克服这些低效率问题，gLLM提出了一种全局均衡的流水线并行系统，其核心是**Token Throttling**机制，以有效缓解流水线气泡。
1.  **Token Throttling机制**：这是一种细粒度的调度策略，能够独立调节预填充和解码令牌的数量，从而通过利用推理系统的全局信息实现均衡计算。
<img width="962" height="311" alt="image" src="https://github.com/user-attachments/assets/0ec8c32f-9379-467a-b830-1054f9d4c2eb" />

2.  **异步执行和消息传递架构**：gLLM运行时采用针对流水线并行特性优化的异步执行和消息传递架构，以降低数据依赖性和CPU开销。
    *   系统采用多进程架构，每个流水线阶段分配一个专用工作进程，并有一个独立的后端（frontend）进程处理用户交互。工作进程分为驱动工作进程和普通工作进程。
    *   驱动工作进程负责接收请求、调度micro-batch、广播元数据以及将输出流式传输回前端。
    *   所有工作进程专注于模型执行，接收上一阶段的激活、执行前向计算并发送激活到下一阶段。
    *   异步特性通过三项设计原则实现：非阻塞的流水线操作、解耦的前后端处理、以及抢占式元数据调度（元数据和激活数据分离传输，实现数据准备与计算的重叠）。
<img width="570" height="562" alt="image" src="https://github.com/user-attachments/assets/27728589-f21c-43a2-a73a-77aec06824a4" />

<img width="1129" height="562" alt="image" src="https://github.com/user-attachments/assets/ba9601d6-0251-4da8-a7e7-dff978b6eba3" />

<img width="571" height="519" alt="image" src="https://github.com/user-attachments/assets/6b18e638-eb2e-49c2-b595-aa4746daffd0" />
<img width="1133" height="573" alt="image" src="https://github.com/user-attachments/assets/d2f2bbcc-1017-48ad-a4fe-0dcca2dd5d6b" />


**实验评估：**
gLLM在Qwen2.5系列（14B和32B）和Llama-3.1-100B等代表性LLM上进行了实验评估。实验结果表明，gLLM相比最先进的流水线或张量并行系统（如vLLM和SGLang）取得了显著的性能提升，最大吞吐量提高了11%至398%，同时保持了更低的延迟。gLLM在不同请求速率下通常比其他系统能支持2-6倍更高的请求速率转折点，并具有更低的端到端延迟（E2EL）增长斜率。在扩展性研究中，gLLM随着GPU数量增加表现出近乎线性的扩展效率，尤其在跨节点部署中，其性能优势更为显著。在服务水平目标（SLO）达成率方面，gLLM比vLLM高出64%，能在80%SLO达成率下支持79%更高的请求速率。消融研究（ablation study）证实了Token Throttling中各组件（WT和UT）以及gLLM运行时本身对性能提升的关键作用。系统代码已开源。


## HyTiS
HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality

https://dl.acm.org/doi/pdf/10.1145/3712285.3759771 武汉大学 程大钊；NV等，SC2025
https://zenodo.org/records/16674739 开源

1. 💥 针对GPU通用矩阵乘法（GEMM）中因波量化导致的硬件利用率低下和性能下降问题，本文提出了HyTiS混合瓦片调度框架。
2. 🚀 HyTiS结合了两级瓦片调度（全波优化吞吐量，部分波降低延迟）和自适应瓦片布局选择（通过分析模型优化L2缓存），并利用离线性能画像生成高效的微核以减小运行时搜索空间。
3. 📈 在NVIDIA H100和A100 GPU上的广泛评估表明，HyTiS相较于cuBLAS等基线实现了显著加速，最高可达**1.95倍和2.08倍**，并有效缓解了波量化问题并提升了L2缓存亲和性。
m需要>=1000 ?

<img width="981" height="720" alt="image" src="https://github.com/user-attachments/assets/83676a1b-401b-4b72-9487-f5143ec24f1d" />

本文提出了HyTiS（Hybrid Tile Scheduling），一个针对GPU通用矩阵乘法（GEMM）的混合瓦片调度框架，旨在增强波（wave）利用率和缓存局部性。GEMM作为深度学习和科学计算中的基础操作，在GPU上进行加速已成为常态。然而，随着现代GPU核心数量的增加和更大瓦片尺寸的采用，由部分填充波引起的波量化（wave quantization）问题日益严重，导致硬件利用率降低和性能显著下降。现有解决方案往往执行效率低下或引入额外的同步开销。

为解决这些挑战，HyTiS提出了一个结合了两级瓦片调度和自适应瓦片布局选择的框架。
1.  **两级瓦片调度（Two-level Tile Scheduling）**：
    *   **第一级调度**：旨在最大化满波（full waves）的吞吐量，通常采用较大的瓦片尺寸以提高计算效率，充分利用硬件资源。
    *   **第二级调度**：针对部分波（partial waves），通过细粒度瓦片（fine-grained tiling）来最小化延迟，以提高硬件利用率。
    *   **离线微核生成与搜索空间优化**：为避免在两个调度级别上进行详尽搜索（可能导致 $10^4$ 级别的巨大设计空间），HyTiS引入了离线分析阶段。在此阶段，系统会识别出代表性的吞吐量优化型（throughput-oriented）和延迟优化型（latency-oriented）微核（micro-kernels），从而构建一个显著缩小且更易处理的运行时搜索空间。
        *   **吞吐量优化型微核（Throughput-Oriented Micro-Kernels, $S_{TO}$）**：通过经验吞吐量分析构建。候选微核需满足资源限制（如共享内存和寄存器不溢出）、指令集限制（如H100上$b_M$需是64的倍数），并采用“每个SM一个瓦片”的调度策略以最大化共享内存利用率。最终选出吞吐量最佳的微核$K_{TO}^{opt}$，并保留吞吐量下降在阈值$l_1$内的其他候选微核。
            *   吞吐量定义：$T(K_i) = (M_i \times N_i) / (n_0 \times t(K_i))$
            *   资源约束：$SMEM(K_{TO}) \le SMEM_0$, $REG_{spill}(K_{TO}) == 0$
            *   利用率约束：$\forall K' | K'.x \ge K.x, x \in \{b_M, b_N, b_K\}, K' \ne K_{TO}, SMEM(K') \le SMEM_0$
            *   最优选择：$K_{TO}^{opt} = \arg\max_{K_i}(T(K_i))$
            *   候选集：$S_{TO} = \{K_{TO}^i | diff(T(K_{TO}^i), T(K_{TO}^{opt})) < l_1\}$
        *   **延迟优化型微核（Latency-Oriented Micro-Kernels, $S_{LO}$）**：从$S_{TO}$中采样更小的瓦片配置来构建。选择每波延迟最低的微核$K_{LO}^{opt}$，并保留延迟偏差在阈值$l_2$内的其他候选微核。
            *   最优选择：$K_{LO}^{opt} = \arg\min_{K_i}(t(K_i)/n_0)$
            *   候选集：$S_{LO} = \{K_{LO}^i | diff(t(K_{LO}^i), t(K_{LO}^{opt})) < l_2\}$
        *   **分层瓦片调度**：第一级调度使用$S_{TO}$中的微核处理满波，计算剩余负载。第二级调度使用$S_{LO}$中的微核处理剩余负载。如果第二级所需的瓦片数超过可用SM数，则该调度方案无效。这种分层方法避免了贪婪选择可能导致的次优解。
        *   **自适应自动调优**：HyTiS的搜索空间是根据GPU架构和工作负载特性自适应确定的，而不是固定预定义的。超参数$l_1$和$l_2$允许根据问题规模动态调整搜索空间。

2.  **自适应瓦片布局选择（Adaptive Tile Layout Selection）**：
    *   **问题观察**：瓦片布局显著影响GEMM性能，不同布局会导致L2缓存数据局部性差异，进而影响从全局内存到L2缓存的数据传输量。
    *   **分析模型**：HyTiS引入了一个分析模型，旨在最小化波粒度下从全局内存到L2缓存的数据传输量。模型考虑了单波内部和多波之间的数据重用。
    *   **布局策略**：支持Group-M ($G_M$) 和Group-N ($G_N$) 两种瓦片布局模式，其行主序和列主序是特例。通过最小化第一波的数据传输量$V_1$来确定最优组大小$s_{opt}$，然后使用所有波的总数据传输量$V_{tol} = \sum_i V_i$作为度量标准，选择$G_M$和$G_N$中$V_{tol}$较小者作为最优瓦片布局$(tl_{opt}, s_{opt})$。该优化仅在第一级调度（多波）中应用，第二级调度（单波）采用固定的列主序。

HyTiS基于Triton实现，利用Triton的编译框架进行瓦片内优化，HyTiS则专注于瓦片调度层面。内核设计上，实现了两级调度的GEMM内核，由$L_oad$、$C_ompute$、$S_tore$和地址偏移函数$offset\_fn$构成。在NVIDIA Hopper架构上，利用持久化内核（persistent kernel）和TMA指令；在Ampere架构上，采用传统数据并行启动策略。

在NVIDIA H100和A100 GPU上的广泛评估表明，HyTiS实现了显著的加速：
*   相较于cuBLAS，在H100上最高加速1.95倍，在A100上最高加速2.08倍。
*   相较于Split-K、Stream-K和Inductor-Triton，也有显著性能提升。
*   详细的系统级分析证实了HyTiS在缓解波量化问题、改善SM工作负载平衡和L2缓存亲和性方面的有效性。

HyTiS的开销包括离线分析（设备和数据布局一次性成本，H100约19分钟，A100约36分钟）和自动调优（H100平均搜索空间14，最大66；A100平均16，最大77），这些开销在显著的性能提升面前被认为是可接受的。

## MXBlas
MXBLAS: Accelerating 8-bit Deep Learning with a Unified Micro-Scaled GEMM Library

https://dl.acm.org/doi/pdf/10.1145/3712285.3759809 武汉大学 程大钊；NV等，SC2025


https://github.com/yatorho/MXBLAS 
m需要 >= 128??
<img width="617" height="311" alt="image" src="https://github.com/user-attachments/assets/f5085d7a-a5ed-4675-995d-fb719ddacb68" />

1. 🌐 现有的8位微缩放通用矩阵乘法（MX-GEMM）实现因其模型导向性，在处理多样的MX格式时面临紧耦合、推广操作效率低下和量化开销被忽视等局限。
2. 🚀 MXBLAS提出了一个统一的MX-GEMM库，通过模板化设计支持不同推广模式、自适应运行时内核生成和计算-存储协同优化（将量化融合到内核尾声），以克服这些限制。
3. ⚡ 实验证明，MXBLAS的性能比现有库平均提高33%，并且首次全面实现了8位广义计算在所有MX格式变体中的性能优势，展现了其卓越的效率、稳定性和通用性。

MXBLAS是一项旨在加速8位深度学习的统一微尺度通用矩阵乘法（MX-GEMM）库。MX-GEMM利用8位微尺度格式（MX-format）输入，是深度学习工作负载加速的重要进展。MX-format空间多样，包含多种缩放模式和粒度。然而，当前的MX-GEMM实现通常采用面向模型的方法，即格式定制是针对单个模型量身定制的。这导致了三个关键限制：刚性的问题-内核耦合、低效的Promotion操作以及被忽视的量化开销。

现有的MX-GEMM实现存在以下问题。首先是**问题-内核刚性**：内核通常针对特定MX-format进行专门化，降低了可扩展性；支持新的MX-format往往需要完全重新设计，导致高昂的内核开发成本。其次是**次优的Promotion操作**：这些实现通常考虑缩放模式但忽视缩放粒度的变化。这种不匹配可能导致GEMM内核的Tile大小与缩放因子（SFs）的步幅之间对齐不佳。为了弥补这一点，当前实践强制缩放因子符合内核结构，导致Promotion阶段（在缩放之后）效率低下。最后是**被忽视的量化开销**：量化通常被视为一个外部步骤，使MX-GEMM与下游操作（例如FlashAttention-3）隔离，后者直接消耗FP8输入。这种分离需要昂贵的类型转换并增加了不可忽略的量化开销。现有的MX-GEMM库（如DeepGEMM、COAT、FB_GEMM、SGLang、CUTLASS）对MX-format的支持是碎片化的，仅支持有限的缩放模式和粒度，且对Procedural Arguments $P_r$和Parametric Arguments $P_a$存在限制，无法实现MX-GEMM的全部潜力。

<img width="521" height="412" alt="image" src="https://github.com/user-attachments/assets/33e6c75a-85f9-4a87-948b-ec88ac7e229a" />

MXBLAS通过三项关键创新解决了上述限制。首先，它采用**模板化设计**，在统一框架下支持多样化的Promotion模式。MXBLAS将内核设计抽象为两个维度：“何时执行Promotion”和“如何执行Promotion”。
1.  **何时执行Promotion？** 基于Promotion操作发生的阶段，将内核分为Main-loop M类型和Epilogue E类型，以适应沿$K$维度的缩放粒度$S_K$。M类型内核在主循环中执行Promotion，需要额外的寄存器文件进行累积，从而限制了Tile大小；而E类型内核仅在Epilogue中应用缩放，实现in-place操作并减少寄存器文件使用，允许更大的Tile。
2.  **如何执行Promotion？** 基于累加器$M$或$N$维度是否共享单个A或B的SF，分为Full-broadcast F类型和Partial-broadcast P类型。F类型表示CTA内所有线程仅使用CTA级别信息即可确定SF地址；P类型则需要额外的线程ID和累加器布局信息。
这两个语义维度是正交的，因此可以将所有MX-GEMM内核抽象为四种模板$T = \langle P, O \rangle$，其中$P \in \{M, E\}$，$O \in \{F, P\}$。这种解耦设计将原始$O(N^6)$的参数空间复杂度降低到$O(2^2 \cdot N^3)$。在此基础上，MXBLAS提出了两种与$P_a$无关的模板内核优化器：
*   **K-loop splitting**：针对M类型内核，将K-loop拆分为$k_s$和$k$两层循环，将冗余的SF加载操作提升到$k_s$循环中，消除了对动态循环变量$k$依赖的SF地址计算，减少了冗余内存访问和CUDA Core同步开销。
*   **SF block loading**：利用GPU共享内存的未利用空间，将多轮SF加载合并为一次更大的SF块加载，通过矢量化内存操作提高带宽效率，并考虑了阶段数$Q$和SF块大小$V$来扩展$P_r$的搜索空间。

<img width="521" height="592" alt="image" src="https://github.com/user-attachments/assets/e697a70c-f45c-4034-9015-4e0624946fe3" />

<img width="724" height="349" alt="image" src="https://github.com/user-attachments/assets/a599e49b-d2d2-4f70-8b67-680559e96d72" />

<img width="555" height="334" alt="image" src="https://github.com/user-attachments/assets/f3163e50-0681-4d8a-b056-ade747b5c9cd" />


其次，MXBLAS利用**自适应运行时内核生成**——结合模板匹配、引导式搜索空间剪枝和自动调优——动态选择最佳内核配置。
1.  **多模板匹配机制**：与现有库直接在固定模板内调优$P_r$不同，MXBLAS首先将$P_a$与多个模板类型匹配。这扩大了内核类型和搜索空间的范围，从而发现更优的性能潜力。
2.  **快速启发式Procedural Parameters自动调优**：MXBLAS采用全JIT编译框架，通过定制策略定义可调优的$P_r$搜索空间。为提高调优效率，MXBLAS引入了三条指导规则来构建搜索空间：优先与高性能指令对齐、最大化硬件资源利用、以及扩展搜索空间维度以释放潜在性能。它通过经验剪枝提前过滤次优$P_r$候选，并根据指导规则对候选进行评分，优先搜索高分内核，结合early-stop策略显著减少调优时间。

最后，MXBLAS引入**计算-存储协同优化策略**，将量化融合到内核的Epilogue中，以减少开销并提高执行效率。
1.  **Post-reduce scaling策略**：在Epilogue阶段，传统的Pre-reduce scaling策略需要$N$次浮点乘法和$N+1$次浮点除法操作。MXBLAS引入Post-reduce scaling，利用$\sum_i (ISF \times \text{Accum}_i) = ISF \times \sum_i \text{Accum}_i$的数学性质，将缩放操作推迟到amax-reduction之后，从而将浮点乘法次数从$N$减少到1次。此外，静态逆模块在编译时预计算$ISF$的倒数，用乘法替代昂贵的浮点除法，将量化成本从$C_{pre-reduce} = N \times C_{fmul} + (N+1) \times C_{fdiv}$优化为$C_{post-reduce} = (N+2) \times C_{fmul} + 1 \times C_{fdiv}$，显著减少Epilogue延迟。
2.  **Thread-wise register rearrangement**：针对8位输出场景中共享内存bank冲突和带宽利用率低的问题，MXBLAS提出线程级寄存器重排策略。通过在相邻线程$T_{2i}$和$T_{2i+1}$之间交换寄存器，使每个线程能够聚合互补数据，一次操作写入整个bank，消除bank冲突。同时，偶数线程写入当前阶段的bank，奇数线程写入下一阶段的bank，并利用`__shfl_xor_sync`指令进行16位聚合操作，从而将共享内存效率提高一倍并减少所需指令。


<img width="1080" height="474" alt="image" src="https://github.com/user-attachments/assets/61b6debb-edf3-4bc2-8eab-3e226a39c8a6" />

<img width="1080" height="474" alt="image" src="https://github.com/user-attachments/assets/521bf82c-f49d-48d2-a3df-e9ac1af15ab3" />

实验结果表明，MXBLAS在H100 PCIe GPU上平均性能超越现有MX-GEMM库33%，具体而言，比COAT快19%，比FB_GEMM快29%，比SGLang快35%，比DeepGEMM快26%，比CUTLASS快42%。MXBLAS是首个全面支持所有MX-format变体的库，能够充分发挥广义8位计算的性能优势。其在M类型内核（如$B \times B$和$G \times B$）上表现尤为突出，相较于CUTLASS、FB_GEMM、DeepGEMM和SGLang，吞吐量分别提升38.9%、37.3%、26.3%和49.6%。在E类型内核（如$C \times C$和$T \times T$）上，相较于FB_GEMM、SGLang、COAT和CUTLASS，性能分别提升21.2%、21.1%、19.3%和47.5%，并能与cuBLASLt持平。在量化性能方面，MXBLAS比COAT平均提升11.8%，且随着量化粒度$Q_N$的增加，性能提升更为显著。在LLaMA2-70B模型单层解码器上的评估显示，MXBLAS相较于FlashAttention、Transformer Engine和COAT，分别实现了40.3%、4.4%和12.3%的加速。

## KAMI
KAMI: Communication-Avoiding General Matrix Multiplication within a Single GPU

https://www.ssslab.cn/assets/papers/2025-wang-KAMI.pdf 北邮 SC2025

https://zenodo.org/records/16947669/files/KAMI.zip

<img width="936" height="253" alt="image" src="https://github.com/user-attachments/assets/066bebe7-be7c-4ab0-b6a3-2acc35fc7294" />


1. 🚀 KAMI 提出了一套新颖的 1D、2D 和 3D 通信规避（CA）GEMM 算法，旨在通过利用 GPU 内部的寄存器作为本地存储、共享内存作为通信介质以及张量核作为计算单元，来优化单个 GPU 上的小规模和批处理矩阵乘法。
2. 💡 该研究首次将分布式 CA 理论应用于单个 GPU 内存层次结构，并引入了一种基于 GPU 时钟周期的理论分析模型，以更精确地评估算法的计算与通信开销。
3. 📈 实验结果显示，KAMI 在 NVIDIA、AMD 和 Intel GPU 上对通用、低秩、批处理和稀疏矩阵乘法均实现了显著的性能提升，相较于现有库获得了最高达数百倍的加速。

KAMI论文提出了一套名为KAMI的1D、2D和3D通用矩阵乘法（GEMM）算法，旨在将通信规避（communication-avoiding, CA）理论扩展到单个GPU内部，以解决小规模和批处理GEMM操作效率低下的问题。该研究指出，虽然大规模GEMM已接近GPU的浮点峰值性能，但小规模和批处理GEMM往往受限于内存访问，未能充分利用现代处理器的计算能力。

**核心问题与动机**
传统上，CA算法主要应用于分布式计算环境，旨在减少节点间的数据传输。本文作者观察到，GPU内部的内存层次结构（寄存器、共享内存、全局内存）与分布式系统的内存层次结构（本地DRAM、网络通信）在延迟和带宽差异上具有相似性。例如，NVIDIA Hopper GPU上寄存器的访问速度比片上共享内存快约20倍，带宽快4倍。这促使作者思考，是否可以将分布式CA算法的原理应用于单个GPU内部，通过优化GPU片上内存层次（尤其是寄存器和共享内存）的使用来加速小规模GEMM。

**KAMI方法学**
KAMI将GPU的片上组件重新组织为：
*   **计算单元（Computational Units）**: Tensor Cores。
*   **本地存储（Local Memory）**: 低延迟的线程寄存器，用于存储矩阵A、B和C的数据。
*   **通信介质（Communication Medium）**: 高延迟的片上共享内存，用于计算单元（即warp之间）传递子矩阵。

KAMI实现了三种通信规避算法：1D、2D和3D。这些算法在warp级别并行执行，每个warp负责处理矩阵的一部分。与传统方法不同，KAMI不使用执行时间，而是采用GPU时钟周期作为理论分析的单位，以更精细地评估计算和通信开销。

1.  **数据布局（Data Layout）**:
    *   **1D算法**: 矩阵A、B和C沿行方向划分为$p$个子矩阵，每个warp处理对应部分。矩阵B的子矩阵在warp之间通过共享内存进行通信。
    *   **2D算法**: 矩阵A、B和C划分为$\sqrt{p} \times \sqrt{p}$的二维子矩阵网格。每个warp处理其对应的子矩阵。A的子矩阵在同一行warps之间通信，B的子矩阵在同一列warps之间通信。
    *   **3D算法**: 矩阵A和B被进一步细分为$\sqrt[3]{p} \times \sqrt[3]{p} \times \sqrt[3]{p}$的三维子矩阵，C矩阵保持$\sqrt[3]{p} \times \sqrt[3]{p}$的二维划分。通信模式类似于2D算法，但发生在三维warp立方体中。

2.  **算法流程与成本分析**: 每种算法都包含通信阶段和计算阶段，并在多个阶段迭代执行。

    *   **1D算法**:
        *   **通信（Communication）**: 每个阶段，一个warp将其持有的B子矩阵（BSend）写入共享内存（SmB），然后所有其他warp从共享内存读取（BRecv）。该warp也会将其BSend复制到自己的寄存器作为BRecv。
        *   **通信量 $V_{cm}$**: $k n \times s_e$。
        *   **通信成本 $T_{cm}$**: $L_{sm} + \frac{kn \times s_e}{\theta_w p B_{sm}} + \frac{(p-1)kn \times s_e}{\theta_r p B_{sm}}$。
        *   **计算成本 $T_{cp}$**: $\frac{2 \times \frac{m}{p} \times \frac{k}{p} \times n}{O_{tc}} = \frac{2mnk}{p^2 O_{tc}}$。
        *   **总成本 $T_{all}$**: $p \times (T_{cm} + \frac{p}{n_{tc}} T_{cp}) = L_{sm}p + \frac{kn \times s_e}{\theta_w B_{sm}} + \frac{(p-1)kn \times s_e}{\theta_r B_{sm}} + \frac{2mnk}{n_{tc}O_{tc}}$。

    *   **2D算法**:
        *   **通信**: 每个阶段，位于当前列的warp将A子矩阵（ASend）写入共享内存，位于当前行的warp将B子矩阵（BSend）写入共享内存。同行的其他warp读取ASend，同列的其他warp读取BSend。
        *   **通信量 $V_{cm}$**: $(mk + kn) \times s_e$。
        *   **通信成本 $T_{cm}$**: $L_{sm} + \frac{(mk+nk) \times s_e}{\theta_w \sqrt{p} B_{sm}} + \frac{(\sqrt{p}-1)(mk+nk) \times s_e}{\theta_r \sqrt{p} B_{sm}}$。
        *   **计算成本 $T_{cp}$**: $\frac{2 \times \frac{m}{\sqrt{p}} \times \frac{k}{\sqrt{p}} \times \frac{n}{\sqrt{p}}}{O_{tc}} = \frac{2mnk}{p^{\frac{3}{2}} O_{tc}}$。
        *   **总成本 $T_{all}$**: $\sqrt{p} \times (T_{cm} + \frac{p}{n_{tc}} T_{cp}) = L_{sm}\sqrt{p} + \frac{(mk+nk) \times s_e}{\theta_w B_{sm}} + \frac{(\sqrt{p}-1)(mk+nk) \times s_e}{\theta_r B_{sm}} + \frac{2mnk}{n_{tc}O_{tc}}$。

    *   **3D算法**:
        *   **通信**: 每个阶段，位于当前“层”的特定列和行的warp分别广播A和B子矩阵。
        *   **通信量 $V_{cm}$**: $(mk + kn) \times s_e$。
        *   **通信成本 $T_{cm}$**: $L_{sm} + \frac{(mk+nk) \times s_e}{\theta_w \sqrt[3]{p} B_{sm}} + \frac{(\sqrt[3]{p}-1)(mk+nk) \times s_e}{\theta_r \sqrt[3]{p} B_{sm}}$。
        *   **计算成本 $T_{cp}$**: $\frac{2 \times \frac{m}{\sqrt[3]{p}} \times \frac{k}{\sqrt[3]{p^2}} \times \frac{n}{\sqrt[3]{p}}}{O_{tc}} = \frac{2mnk}{p^{\frac{4}{3}} O_{tc}}$。
        *   **总成本 $T_{all}$**: $\sqrt[3]{p} \times (T_{cm} + \frac{p}{n_{tc}} T_{cp}) = L_{sm}\sqrt[3]{p} + \frac{(mk+nk) \times s_e}{\theta_w B_{sm}} + \frac{(\sqrt[3]{p}-1)(mk+nk) \times s_e}{\theta_r B_{sm}} + \frac{2mnk}{n_{tc}O_{tc}}$。

    其中，$s_e$ 为单个矩阵元素的字节大小，$L_{sm}$ 为寄存器到共享内存的延迟，$B_{sm}$ 为共享内存带宽，$\theta_r, \theta_w$ 为读写银行冲突因子，$O_{tc}$ 为每个Tensor Core每周期算术操作数，$n_{tc}$ 为每个SM的Tensor Core数量，$p$ 为并行执行的warps数量。

3.  **稀疏扩展（Sparse Extension）**: KAMI支持稀疏-稠密矩阵乘法（SpMM）和稀疏通用矩阵乘法（SpGEMM）。稀疏矩阵以用户可配置大小（默认16x16）的稠密块形式存储，并采用Z-Morton顺序进行多级索引。通信时，除了数值数组（Val），还需要传输索引数组（RowPtr和ColBlkIdx）。SpGEMM需要一个符号阶段来预计算非零块数量和分配内存。

4.  **实现细节（Implementation Details）**: 针对寄存器和共享内存容量限制，KAMI采用沿$k$维度切片（slicing）的策略，将不活跃的子矩阵卸载到共享内存。切片比例是一个可调参数。作者指出，CUDA warp调度和底层GPU硬件可以有效地交错数据传输和计算，因此未强制执行显式的重叠策略。

**实验评估**
KAMI在NVIDIA GH200、RTX 5090、AMD 7900 XTX和Intel Max 1100四种GPU上进行了广泛评估，并与cuBLASDx、CUTLASS、cuBLAS、MAGMA和SYCL-Bench等现有库进行比较。

*   **块级方阵GEMM**:
    *   在NVIDIA GPU上，KAMI-1D/2D/3D相较于cuBLASDx和CUTLASS展现出显著加速，例如在GH200上FP64 GEMM平均加速比最高达4.02x（KAMI-1D）和3.65x（KAMI-1D）。在RTX 5090上FP16 GEMM，KAMI-1D相对于cuBLASDx和CUTLASS平均加速比最高达2.46x和19.98x，峰值可达3.38x和74.36x。
    *   在AMD GPU上，KAMI在矩阵阶数超过48时性能下降。
    *   在Intel GPU上，KAMI-1D/2D/3D相较于SYCL-Bench平均加速比最高达4.97x，峰值可达14.48x。
    *   KAMI-1D通常优于KAMI-2D/3D，原因在于后两者控制流更复杂，增加了更多的nop指令。

*   **块大小影响**: KAMI-1D在不同块大小下性能稳定，而KAMI-2D/3D在块大小较大时（如超过256）表现更佳，表明KAMI-1D更适合严格的块大小限制。

*   **共享内存与寄存器协作**: 对于小矩阵（32-64），仅使用寄存器即可；对于中等大小矩阵，适度使用共享内存（例如50%的数据临时存储在共享内存中）可将性能提高1.34倍；过度使用共享内存会导致性能下降。

*   **低秩GEMM**: KAMI在低秩GEMM中表现出更显著的优势，相较于cuBLASDx和CUTLASS，平均加速比最高达3.66x和4.89x，峰值可达6.11x和11.61x。这归因于KAMI直接将数据加载到寄存器，并使用共享内存进行通信，更符合低秩GEMM的模式。

*   **批处理GEMM**: KAMI在批处理GEMM中实现了极高的加速比，相对于MAGMA和cuBLAS，批处理大小为1000时平均加速31.60x和340.37x，批处理大小为10000时平均加速10.23x和96.17x。

*   **SpMM和SpGEMM**: SpMM的性能趋势与稠密GEMM相似，因为B和C是稠密矩阵。而SpGEMM由于两输入矩阵的稀疏结构不同，导致索引复杂性增加和内存访问模式不可预测，吞吐量较低。

*   **理论分析**:
    *   **寄存器分配**: 实际寄存器使用量低于理论预测（KAMI-1D为76.86%，KAMI-2D为73.14%，KAMI-3D为65.67%），这归因于编译器优化。KAMI的片上内存使用（2-8 KB共享内存）远低于cuBLASDx和CUTLASS。
    *   **周期分解**: 实验测量的通信和计算周期与理论模型基本一致，计算周期上的差异可能源于Hopper架构上MMA指令执行效率的限制（约62%）。

**贡献总结**
1.  提出了KAMI，将CA算法扩展到单个GPU内部以加速小规模矩阵乘法。
2.  提出了一种新的基于GPU时钟周期的通信和计算理论分析方案。
3.  利用块状Z-Morton存储格式，在CA方法中支持SpMM和SpGEMM。
4.  在NVIDIA、AMD和Intel GPU上实现了KAMI，并展示了优于现有SOTA工作的性能。

## AdvancedIF
AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following 

https://arxiv.org/abs/2511.10507 Meta 超级智能lab等 2025.11.26

https://huggingface.co/datasets/meta-llama/AdvancedIF
https://github.com/facebookresearch/AdvancedIF

1. 📚 本文推出了AdvancedIF，一个包含1,600多个提示和专家策划的rubrics的**综合基准**，旨在**评估大型语言模型（LLMs）遵循复杂、多轮和系统级指令的能力**。
2. 🛠️ 为解决训练挑战，研究者提出了RIFL（**Rubric-based Instruction-Following Learning**），一个新颖的后训练流程，通过**rubric生成、微调的rubric验证器和奖励塑形**（reward shaping）实现指令遵循的有效强化学习。
3. 🚀 广泛实验表明，RIFL显著提**升了LLMs的指令遵循能力**，在AdvancedIF基准上取得了6.7%的绝对增益，并在公共基准上表现出色，确立了rubrics在LLMs高级指令遵循训练和评估中的强大作用。

大型语言模型（LLMs）在多项任务中展现了卓越的性能，然而，对于复杂、多轮和系统提示的指令遵循（Instruction Following, IF）能力仍是一个重大挑战。当前缺乏高质量的人工标注基准和可靠、可解释的奖励信号，阻碍了对这些能力的严格评估和有效训练。
<img width="761" height="392" alt="image" src="https://github.com/user-attachments/assets/7104a67c-b5ef-4184-9c6d-d8913c7c15dd" />

<img width="749" height="404" alt="image" src="https://github.com/user-attachments/assets/e5ad9c49-213b-420e-a2f2-c500dc468efc" />

为解决这些问题，本文引入了 **AdvancedIF**，一个全面的基准测试，包含超过1,600个提示和由专家精心策划的评估准则（rubrics），旨在评估LLMs遵循复杂、多轮和系统级指令的能力。AdvancedIF还开源了其评估脚本。该基准的独特性在于，其所有提示和评估准则均由人类专家编写，涵盖了三个核心方面：**显式和复杂用户指令遵循 (Explicit and Complex User Instruction Following)**（单个提示包含6条以上指令，涉及语气、格式、风格、结构、长度、负面约束、拼写和条件指令等）、**多轮承载上下文指令遵循 (Multi-Turn Carried Context Instruction Following)**（模型需遵循对话历史中延续的指令）以及 **系统提示可控性 (System Prompt Steerability)**（模型遵循系统提示中关于响应风格、安全性、产品上下文设置等约束）。AdvancedIF的数据集统计显示，其平均每个对话包含7.44（复杂IF）、6.08（多轮IF）和9.81（系统提示）个评估准则，平均轮次分别为1.00、7.69和11.21。对现有最先进LLMs（如GPT-5、Gemini 3 Pro）的基准测试表明，其表现最佳也仅约75%，显示AdvancedIF是一个极具挑战性的基准。
<img width="676" height="305" alt="image" src="https://github.com/user-attachments/assets/a23e5c0a-115a-40ab-9709-bae736b395d7" />

此外，本文提出 **RIFL (Rubric-based Instruction-Following Learning)**，一个新颖的后训练（post-training）流程，通过利用评估准则生成、微调评估准则验证器和奖励塑形（reward shaping）技术，实现对指令遵循能力的有效强化学习。RIFL将指令遵循问题建模为一个强化学习问题，其目标是最大化以下期望回报：
$$J (\pi_\theta) = E_{(q,r)\sim D}\left[E_{o\sim\pi_\theta (\cdot|q)}[R(q, o, r)] - \beta D_{KL}[\pi_\theta (\cdot|q)\|\pi_{ref}(\cdot|q)]\right]$$
其中，$\pi_\theta$ 是训练中的LLM策略，$\pi_{ref}$ 是参考策略，$q$ 是提示，$o$ 是模型响应，$r = \{r_i\}_{i=1}^d$ 是与提示 $q$ 对应的评估准则集合，$R(q, o, r)$ 是基于评估准则的奖励信号。RIFL主要包含三个核心组件：

1.  **评估准则生成器 (Rubric Generator)**：为了大规模生成高质量的提示和评估准则，作者基于少量专家编写的数据训练了一个评估准则生成器。该生成器是一个经过SFT微调的Llama 4 Maverick模型，使用数千条人工标注的提示-评估准则对进行训练，F1分数从0.639显著提升至0.790。

2.  **评估准则验证器 (Rubric Verifier)**：为了构建一个可靠的验证器，RIFL采用了一个两阶段微调流程来训练LLM作为评估准则验证器。首先进行 **SFT阶段**，使用人类专家标注的评估数据Dgolden来冷启动模型，使其能够像人类专家一样评估响应。接着进行 **RL阶段**，在更广泛的数据集上进一步提升验证器的泛化能力。在此阶段，验证器对每个评估准则进行判断并提供理由，然后将其判断结果与人类专家的二元标签进行比较，以两者之间的一致性比例作为奖励信号进行强化学习。实验结果表明，经过两阶段训练的评估准则验证器（F1分数0.728）相比于原始LLM（F1分数0.515）和仅SFT的模型（F1分数0.656）实现了显著更高的与人类判断的一致性，且与强大的o3-mini模型（F1分数0.723）相当。

3.  **奖励设计与塑形 (Reward Design and Shaping)**：对于每个提示-响应-评估准则对 $(q, o, r = \{r_i\}_{i=1}^d)$，评估准则验证器 $V$ 输出一个 $d$ 维二元标签 $v = \{v_i\}_{i=1}^d$，其中 $v_i$ 表示响应 $o$ 是否满足评估准则 $r_i$。本文采用最直接的奖励函数 $R(q, o, r) = I [V (q, o, r) = 1]$，即只有当模型满足所有评估准则时才获得奖励1，否则为0（all-or-nothing reward）。为了解决早期实验中观察到的奖励作弊（reward hacking）问题，RIFL引入了额外的评估准则作为奖励塑形技术，例如检查模型响应是否包含误导性伪影或是否完整，以促使模型生成更干净、更完整的响应。
<img width="573" height="326" alt="image" src="https://github.com/user-attachments/assets/b879279d-4bf4-42ed-b368-f5632e22fd42" />

广泛的实验表明，RIFL显著提升了LLMs的指令遵循能力，在AdvancedIF基准上取得了6.7%的绝对增益，并在MultiChallenge和IFEval等公共基准上取得了强劲结果。具体而言，RIFL使Llama 4 Maverick在AdvancedIF上的平均得分从51.4%提升至58.1%。消融研究证实了RIFL中每个组件的有效性，包括微调评估准则验证器优于普通LLM判官，以及“all-or-nothing”奖励设计和奖励塑形技术在缓解奖励作弊问题上的有效性。这项工作确立了评估准则在LLMs高级IF训练和评估中的强大作用，为开发更强大、更可靠的AI系统铺平了道路。

## RGR-GRPO
Reward and Guidance through Rubrics: Promoting Exploration to Improve Multi-Domain Reasoning
https://arxiv.org/abs/2511.12344 中科院计算所等，2025.11.18

1. 🌐 针对现有大型语言模型（LLM）强化学习在单领域可验证奖励和受限探索空间上的局限性，本文提出了RGR-GRPO框架，旨在通过**评价标准（rubrics）提供细粒度奖励和离线指导**以提升多领域推理能力。
2. 💡 RGR-GRPO通过构**建跨领域的问题特定评价标准**，提供包含**事实性和过程性**两类指标的密集奖励，并利用这些评价标准指导模型对**次优轨迹进行自修正**，以实现离策略探索。
3. 📈 实验证明，RGR-GRPO在14个多领域基准测试上持续优于现有RL方法，在数学、物理、化学和通用推理任务中**平均性能提升显著**，并展现出**稳定的探索动态和优越的pass@k**表现。
<img width="525" height="353" alt="image" src="https://github.com/user-attachments/assets/83a88c95-7859-4b7f-81d7-8154c9c12071" />

大型语言模型（LLMs）在强化学习（RL）的推动下，在复杂推理能力方面取得了显著进步。然而，**现有方法主要集中于单一领域（如数学）且奖励可验证（RLVR）的任务**，并且其纯在线RL框架限制了探索空间，从而制约了推理性能。为解决这些局限性，本文提出了RGR-GRPO（Reward and Guidance through Rubrics）框架，一个由评分标准（rubrics）驱动的RL框架，用于多领域推理。RGR-GRPO利用评分标准提供细粒度的奖励信号和离线指导，使LLMs能够接收密集且信息丰富的奖励，并在GRPO训练期间探索更大的解决方案空间。

**核心方法论**

RGR-GRPO框架包含两个关键组件：Rubric-based fine-grained rewards（**基于评分标准的细粒度奖励**）和Rubric-guided offline exploration（**基于评分标准的离线探索**）。
<img width="1048" height="575" alt="image" src="https://github.com/user-attachments/assets/4f77785b-fa2d-45be-aa5e-78da3b752560" />

**1. 基于评分标准的细粒度奖励 (Rubric-based Fine-Grained Rewards)**
为了克服单一领域训练和稀疏奖励信号的限制，RGR-GRPO构建了跨多个领域的、针对特定问题的评分标准。每个评分标准包含两种互补的评估类型：Factual Criteria（事实性标准）和Process Criteria（过程性标准），每种类型都分配一个反映其相对重要性的自适应权重$w_k$。
*   **Factual Criteria**：用于验证中间和最终结果的准确性。
*   **Process Criteria**：衡量推理路径的逻辑健全性。

评分标准的生成采用两阶段过程：首先，对于每个问题$q$，使用专家LLM（如OpenAI O3）生成一个高质量的参考答案$a_{ref}$。然后，在$q$和$a_{ref}$的条件下，指示专家LLM生成细粒度的评分标准集$C = \{c_k\}_{k=1}^{|C|}$，其中每个标准$c_k = (d_k, w_k)$包含描述性规范$d_k$和自适应权重$w_k$。

对于模型生成的输出$o_i$和每个标准$c_k \in C$，采用一个判断模型（LLM-as-Judge）来生成二元验证分数$s_k(q, o_i) \in \{0, 1\}$，表示输出是否满足该标准：
$s_k(q,o_i) = \begin{cases} 1, & \text{if response } o_i \text{ satisfies } d_k \text{ given prompt } q \\ 0, & \text{otherwise} \end{cases}$
然后，所有评分标准分数及其对应权重被聚合成一个归一化的标量奖励：
$R(q,o_i) = \frac{\sum_{k=1}^{|C|} w_k \cdot s_k(q,o_i)}{\sum_{k=1}^{|C|} w_k}$
为了允许模型在推理过程偏离预定义评分标准时仍能生成正确答案，最终奖励$r_i$的计算如下：如果所有事实性标准$C_{fact_i}$都被验证通过，则奖励为1；否则，奖励为$R(q,o_i)$。
$r_i = \begin{cases} 1, & \text{if } \sum_{k=1}^{|C_{fact_i}|} s_k(q,o_i) = |C_{fact_i}| \text{ where } c_k \in C_{fact_i} \\ R(q,o_i), & \text{otherwise} \end{cases}$
这种设计提供了可靠且密集的奖励信号，平衡了缓解奖励欺骗和减少组内方差的需求。

**2. 基于评分标准的离线探索 (Rubric-Guided Offline Exploration)**
为了打破纯在线方法的探索瓶颈，RGR-GRPO将基于评分标准的信号集成到离策略指导机制中。该过程分为三个步骤：

*   **步骤1：探索评估 (Exploration Assessment, EA)**
    在每个GRPO训练迭代中，首先从旧策略$\pi_{\theta_{old}}$采样$G-1$个初始响应$\{o_i\}_{i=1}^{G-1}$。每个响应通过基于评分标准的奖励函数（第2.2节）进行评估，得到聚合奖励和详细的标准判断。然后确定性能最佳的响应$o_{best} = \arg\max_{o_i \in \{o_1,...,o_{G-1}\}} r_i$。如果$o_{best}$满足所有评分标准（即$\sum_{k=1}^{|C_{best}|} s_k(q,o_{best}) = |C_{best}|$），则认为当前在策略探索已足够，生成最后一个响应$o_G$并使用在线GRPO更新策略。否则，即策略未能达到完美解决方案时，将应用后续的混合策略细化。探索评估机制通过判断当前探索上限是否足够来决定是否需要离策略指导，有效避免了不必要的离策略更新和熵爆炸的风险。

*   **步骤2：基于评分标准的自细化 (Rubric-Based Self-Refinement)**
    为了提高当前探索组的上限，通过明确以$o_{best}$未满足的评分项$C_{failed} = \{c_k | s_k(q,o_{best}) = 0\}$为条件，对$o_{best}$进行细化。策略模型以三元组$(q, o_{best}, C_{failed})$为提示，生成一个细化后的响应$o_G \sim \pi_{\theta_{old}}(\cdot | q, o_{best}, C_{failed})$，并计算其奖励$r_G$。

*   **步骤3：混合策略GRPO (Mix-Policy GRPO)**
    最后，将离策略细化的轨迹$o_G$与初始的在策略轨迹$\{o_i\}_{i=1}^{G-1}$合并，共同更新策略。优势估计仍遵循GRPO的公式（1）。模型在以下混合策略目标下进行优化：
    $J_{\text{RGR-GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^{G-1} \sim \pi_{old}, o_G \sim \pi_{old}(\cdot|q,o_{best},C_{failed})} \frac{1}{G} \left[ \sum_{i=1}^{G-1} \sum_{t=1}^{|o_i|} r^{(i)}_t(\theta) \hat{A}_i + \sum_{t=1}^{|o_G|} f_{shaper}(G)_t(\theta) \hat{A}_G \right]$
    其中，$r^{(i)}_t(\theta) = \frac{\pi_{\theta}(o_i|q,o_{(<t)}^{(i)})}{\pi_{\theta_{old}}(o_i|q,o_{(<t)}^{(i)})}$是正常的重要性采样比率。离策略细化项通过一个塑形函数$f(\cdot)$进行调制，以调整离策略细化响应$o_G$中每个token的贡献：
    $f_{shaper}(G)_t(\theta) = \frac{\pi_{\theta}(o_G | q,o_{(<t)}^{(G)})}{\pi_{\theta_{old}}(o_G | q,o_{(<t)}^{(G)}) + \gamma}$
    其中$0 < \gamma < 1$。塑形函数通过赋予细化轨迹中低概率token更高的重要性来重新加权梯度，鼓励模型从成功但分布外（out-of-distribution）的行为中学习，同时减轻失败细化的影响。

**实验设置**

*   **数据集**：训练集基于WebInstruct-Verify，一个大规模、多领域、可验证的数据集，包含约1万个样本。评估数据集涵盖特定主题（数学：Sci-Math, MATH, MATH500；物理：PIQA, Sci-Physics；化学：ChemBench, Sci-Chemistry）和通用推理（MMLU, MMLU-Pro, GPQA, GPQA*, OlympicArena）任务。
*   **模型**：Qwen2.5-3B和Qwen2.5-7B。
*   **基线**：
    *   **在线GRPO方法**：Outcome-GRPO（基于最终答案二元验证奖励）、Likert-GRPO（通过与参考答案比较提供密集奖励）、Rubric-GRPO（基于聚合评分标准验证结果的奖励）。
    *   **离线指导方法**：LUFFY（直接混合离线监督响应）、Critique-GRPO（利用基于真值的critique指导策略细化）。
*   **RL实现**：使用Verl框架，训练400步，批大小96，学习率$1 \times 10^{-6}$，采样温度1.0，每次提示采样8个轨迹。引入长度惩罚。移除GRPO中的裁剪函数和KL散度约束（设置$\beta=0$），以允许更灵活的策略更新。
![Uploading image.png…]()

**实验结果**

*   **主要结果**：RGR-GRPO在所有基准测试中始终优于所有基线。在7B模型上，RGR-GRPO比Outcome-GRPO在数学、物理、化学和通用推理任务上分别平均提高了+7.0%、+5.4%、+8.4%和+6.6%。与最佳在线方法Rubric-GRPO相比，RGR-GRPO也取得了3.7%的平均提升。
*   **分布外（OOD）性能**：在MedMCQA和CS-Bench数据集上，RGR-GRPO展示了最佳的OOD性能，表明其强大的泛化潜力。
*   **Pass@k曲线分析**：在SciBench的物理、化学和数学子集上，RGR-GRPO在不同$k$值下持续表现出优越的Pass@k性能，并随着$k$的增加，其性能提升更稳定，表明其在促进有效策略探索和推理多样性方面的强大能力。
*   **策略探索分析**：RGR-GRPO在训练过程中保持了更平滑、更渐进的熵下降，表明持续的探索和持续学习。零奖励响应的比例稳步下降，离策略数据混合比例逐渐下降，同时重要性采样概率保持稳定，表明策略混合策略平衡且稳定。
*   **消融研究**：消融实验证实了不同评分标准类别（Factual, Process）以及离策略塑形和探索评估（EA）配置对整体性能的关键贡献。
![Uploading image.png…]()

**评分标准在奖励丰富和自细化中的作用**

*   **基于评分标准的密集有效奖励**：与稀疏的Outcome奖励和嘈杂的Likert奖励相比，基于评分标准的奖励将评估分解为明确的标准，使得判断模型可以对每个方面进行可靠的二元决策并聚合为密集且可解释的奖励，从而实现稳定且有效的强化学习。
*   **基于评分标准的测试时自细化**：在不同RL训练阶段，基于评分标准的自细化持续带来显著改进。即使模型内在推理能力提高，评分标准指导带来的收益也保持稳定，凸显了评分标准作为离线指导在整个强化学习过程中持续的益处。

**结论**

RGR-GRPO通过结合细粒度的评分标准奖励和基于评分标准的离线自细化，有效解决了多领域推理中奖励稀疏和探索受限的问题。在多个基准测试上的广泛实验证明，RGR-GRPO持续超越现有RL基线，并在训练过程中保持稳定的熵波动，实现卓越的pass@k性能，展示了持续的探索和有效突破现有性能瓶颈的能力。

## RuscaRL: Breaking the exploration bottleneck
https://arxiv.org/abs/2508.16949 浙大 理想等 2025.10.22

https://github.com/IANNXANG/RuscaRL

1. ✨ 针对大型语言模型（LLM）强化学习中探索**高质量样本的瓶颈**，该研究提出了Rubric-Scaffolded Reinforcement Learning (**RuscaRL**) 框架，旨在打破LLM通用推理的探索限制。
2. 💡 RuscaRL通过在**生成响应时提供清单式评分标准作为显式引导**，并**逐步衰减此引导以促使模型内化推理模式**；同时，它利用这些**评分标准**作为可验证奖励来指导模型训练，即使在开放式任务中也能有效学习。
3. 🚀 实验结果显示，RuscaRL在多项基准测试中显著优于现有方法，有效拓宽了LLM的推理边界，并在HealthBench-500等任务上超越了领先的LLM，证明了其优越性和广泛适用性。

<img width="837" height="402" alt="image" src="https://github.com/user-attachments/assets/ceae50f2-59f2-4bb6-997b-0b4fcee09ae3" />

该论文提出了一种名为 Rubric-Scaffolded Reinforcement Learning (RuscaRL) 的新型教学支架（instructional scaffolding）框架，旨在解决大型语言模型（LLM）在通用推理任务中面临的“探索瓶颈”（exploration bottleneck）问题。

**背景与问题**
尽管可验证奖励强化学习（RLVR）在推动 LLM 推理能力方面取得了显著进展，但其有效性主要局限于具有客观可验证答案的领域（如数学证明和代码生成）。对于医疗咨询、创意写作等开放式、需要多维度评估且缺乏单一事实真相的任务，传统的 RLVR 难以提供稳定可靠的奖励信号。此外，LLM 的强化学习过程普遍存在探索瓶颈：模型倾向于收敛到有限的推理轨迹，导致策略熵（policy entropy）坍塌，难以发现多样化和高质量的解决方案，从而限制了其推理能力的拓展。
<img width="837" height="473" alt="image" src="https://github.com/user-attachments/assets/e645f021-78a5-4976-90fe-101cfb794d6f" />

**RuscaRL 方法论**
RuscaRL 通过将清单式（checklist-style）的评估标准（即 rubrics）融入强化学习过程，以两种互补的方式解决了上述挑战：

1.  **用于探索的显式支架（Explicit Scaffolding for Exploration）**
    在策略模型生成响应（rollout generation）阶段，RuscaRL **利用 rubrics 作为外部指导**。具体机制包括：
    *   **组内支架差异化（Intra-Group Scaffolding Differentiation）**：为了促进生成响应的多样性，RuscaRL 在每个采样组（group）内**为不同的候选响应提供不同程度的 rubric 支架。**通过一个线性的组级比率向量 $\lambda_{\text{group}} = [\lambda_1, \lambda_2, \ldots, \lambda_G]$，其中 $\lambda_i = \frac{G-i}{G-1}$，确保部分样本获得更强的指导，而另一些则暴露于较弱的指导下，从而增强组内多样性。
    *   **跨步支架衰减（Inter-Step Scaffolding Decay）**：受到教学心理学中支架理论的启发，RuscaRL 采用一个 sigmoid 函数 $\lambda_{\text{step}}(t) = \frac{1}{1+e^{\alpha(t-t_0)}}$ 随着训练进程 $t$（$t \in [0, 1]$）逐渐减少支架强度。其中，$t_0$ 是衰减中点，$\alpha$ 控制衰减的陡峭程度。这鼓励模型逐步内化潜在的推理模式，避免对外部指导的过度依赖。
    *   **集成支架机制**：最终的支架比率 $\lambda_S$ 由组内差异化和跨步衰减共同决定，即 $\lambda_S = \lambda_{\text{step}}(t) \times \lambda_{\text{group}} = \left[\frac{1}{1+e^{\alpha(t-t_0)}} \times \frac{G-1}{G-1}, \ldots, \frac{1}{1+e^{\alpha(t-t_0)}} \times \frac{G-G}{G-1}\right]$。

2.  **用于利用的可验证奖励（Verifiable Rewards for Exploitation）**
    在模型训练阶段，RuscaRL 使用 rubrics 获取鲁棒的奖励信号。
    *   **Rubric-Based Evaluation System**：一个 rubric $R = \{c_1, c_2, \ldots, c_N\}$ 定义为一组 $N$ 个可验证的标准，每个标准 $c_i$ 都有描述和相应的点数 $p_i$。对于给定的指令 $q$、响应 $o$ 和 rubric $R$，一个作为“评分员”（Grader）的 LLM 对每个标准 $c_i$ 进行二元评估 $b_i = G(q, o, c_i) \in \{0, 1\}$（满足或不满足）。
    *   **加权聚合奖励**：所有标准的评估结果形成二元指示向量 $b = [b_1, b_2, \ldots, b_N]$。通过元素乘法 $s = b \odot p = [b_1p_1, b_2p_2, \ldots, b_Np_N]$ 获得精细的得分向量。最终的标量奖励 $r_i = S = \frac{\sum_{j=1}^N s_j}{S_{\text{total}}}$ 通过所有标准得分的总和除以所有正向标准可能获得的总分 $S_{\text{total}}$ 得到。

RuscaRL 采用 Group Relative Policy Optimization (GRPO) 作为核心 RL 算法。GRPO 通过组内优势估计（group-based advantage estimation）替代了价值模型。组相对优势（group-relative advantage）计算为 $\hat{A}_i = \frac{r_i - \text{mean}\{r_j\}_{j=1}^G}{\text{std}\{r_j\}_{j=1}^G}$，其中 $r_i$ 是响应 $o_i$ 的奖励。在计算对数概率时，模型基于 $\pi_\theta(o_{i,t}|q, o_{i,<t})$ 而非 $\pi_\theta(o_{i,t}|q, R_S, o_{i,<t})$ 进行，以鼓励模型内化推理模式，减少对支架的显式依赖。

**实验结果**
广泛的实验表明，RuscaRL 在医疗、写作、指令遵循和 STEM 等多个基准测试中优于现有 SOTA 方法。
*   在 HealthBench-500 上，RuscaRL 将 Qwen2.5-7B-Instruct 的性能从 23.6 提升至 50.3，超越了 GPT-4.1。
*   使用 Qwen3-30B-A3B-Instruct 进行微调后，在 HealthBench-500 上达到了 61.1 分，超越了包括 OpenAI-o3 在内的领先 LLM。
*   RuscaRL 显著提高了模型的采样效率（Best-of-N 曲线更陡峭）和推理边界，生成了初始模型几乎无法生成的高度新颖的响应。
*   在训练动态分析中，RuscaRL 展现了良好的探索-利用平衡，策略熵先上升后下降，验证准确率在整个训练过程中保持最佳。
*   消融实验验证了组内支架差异化和跨步支架衰减机制的有效性，其中线性差异化策略和 Sigmoid 衰减函数表现最佳。
<img width="854" height="409" alt="image" src="https://github.com/user-attachments/assets/f099e78e-e280-49ba-a7c0-f7084bd53dba" />
<img width="825" height="343" alt="image" src="https://github.com/user-attachments/assets/c86a749a-f144-44ba-9e6e-6f70c1071e2d" />
<img width="825" height="343" alt="image" src="https://github.com/user-attachments/assets/d2b40331-896b-4043-9275-f52caf7c8243" />
![Uploading image.png…]()


**结论与展望**
RuscaRL 成功地将教学支架理论应用于 LLM 的强化学习，通过提供逐步衰减的外部指导和鲁棒的奖励函数，打破了通用 LLM 推理的探索瓶颈。未来研究方向包括开发高质量 rubric 数据生成管道、探索基于 rubric 的自然语言反馈策略，以及将方法应用于多模态任务和 Agent 系统。

## FAPO: FLAWED-AWARE POLICY OPTIMIZATION FOR EFFICIENT AND RELIABLE REASONING

https://arxiv.org/pdf/2510.22543 
https://fapo-rl.github.io 字节等 2025.10.26
1. 💡 大型语言模型 (LLM) 在可验证奖励强化学习 (RLVR) 中，存在“**有缺陷的正面输出”（flawed-positive rollouts），即模型通过不可靠的推理路径达到正确答案**，这在**早期训练中能加速能力提升，但后期却阻碍了**可靠推理能力的建立。
2. 🚀 为解决此问题，本文提出了**缺陷感知策略优化** (FAPO) 算法，通过**对有缺陷的正面输出施加无参数的惩罚**，使其在初期作为有用捷径，**后期则逐步引导模型进行可靠推理**，并引入生成式奖励模型 (GenRM) 以精确检测并定位推理错误。
3. ✅ 实验结果表明，FAPO 在不增加token预算的情况下，显著提高了LLM的最终答案正确性、推理过程可靠性及训练稳定性。

提出了一种名为 Flawed-Aware Policy Optimization (FAPO) 的算法，旨在提高大型语言模型 (LLM) 在可验证奖励强化学习 (RLVR) 中的**推理效率和可靠性。**

**1. 背景与问题**
在 RLVR 范式中，LLMs 通过探索推理轨迹并利用最终答案正确的 rollout 作为策略优化的积极信号。然而，这些被标记为“正确”的 rollout 可能包含缺陷模式，例如“**答案猜测** (answer-guessing)”或“**跳步推理** (jump-in-reasoning)”。这些被称为“缺陷正向 (flawed-positive)”的 rollout 获得了与完全正确 rollout 相同的奖励，导致策略模型内化这些不可靠的推理模式，最终**限制了模型的性能上限**。
<img width="869" height="301" alt="image" src="https://github.com/user-attachments/assets/2127fed2-c08e-4405-99b9-700ec5cdbfbb" />

**2. 缺陷正向的初步分析**
论文首先对 RL 训练过程中缺陷正向 rollout 的分布和影响进行了系统研究。
- **普遍性：** 在初始阶段，缺陷正向在各种 LLM 中**普遍存在，占正确 rollout 的 20%-40%。**
- **学习阶段的作用：** 在模型学习的早期阶段，当模型尚不能生成完全正确的 rollout 时，缺陷正向充当达到正确答案的“捷径”，**加速了能力提升**。
- **持续性和双重影响：** 缺陷正向在整个训练过程中持续存在。虽然它们在早期是“垫脚石”，但一旦模型能够生成完全正确的 rollout，这些缺陷正向可能会通过强化不可靠的推理模式来阻碍进一步的学习。对缺陷正向进行惩罚的初步实验显示出显著的性能提升，尤其是在后期训练中。

**3. FAPO 核心方法**
基于以上洞察，FAPO 旨在利用缺陷正向在热身阶段作为有用的捷径，同时在后期细化阶段逐步将优化重心转向可靠推理。FAPO 包含两个主要组成部分：缺陷正向检测和缺陷正向惩罚。

**3.1. 缺陷正向检测**
为了准确且全面地检测缺陷正向，论文引入了一个生成式奖励模型 (Generative Reward Model, GenRM)。
- **现有模型不足：** 现有的 LLM 在检测缺陷正向方面存在问题，例如“过度批评 (over-critic)”现象（高召回但低精度）或推理效率低下。
- **逐步 RL 优化：** 为了增强检测能力，FAPO 采用逐步 RL 奖励公式来训练 GenRM：
    $R_{\text{FAPO-GenRM}} = R_{\text{Outcome}} + R_{\text{Process}}$
    其中，$R_{\text{Outcome}} = \begin{cases} 1, & \text{If } \hat{y}_\theta = y^* \\ -1, & \text{Otherwise} \end{cases}$。
    $R_{\text{Process}} = \begin{cases} - \frac{|\hat{t}_\theta - t^*|}{n}, & \text{If } \hat{y}_\theta = y^* = \text{FP} \\ 0, & \text{Otherwise} \end{cases}$
    这里，$\hat{y}_\theta$ 是模型预测是否为缺陷正向，$y^*$ 是真实标签。$\hat{t}_\theta$ 和 $t^*$ 分别是预测和真实的错误步骤索引，$n$ 是总步骤数。$R_{\text{Process}}$ 在缺陷正向情况下提供距离敏感的惩罚，引导模型精确地定位错误，而非仅仅猜测。这种奖励设计自然地实现了早期强调预测正确性，后期转向过程优化的学习转移。
- **训练数据：** 构建了一个名为 FAPO-Critic-85K 的缺陷正向数据集，通过多个模型生成响应，并使用强大的 LLM (Qwen3-32B) 识别步骤级错误位置。

**3.2. 缺陷正向惩罚**
检测到缺陷正向后，FAPO 通过对缺陷正向 rollout 施加无参数的奖励惩罚来调整其在最终 RL 优化中的作用。
- **奖励函数：** FAPO 修改了标准 RLVR 奖励 $R_{\text{RLVR}}$，引入了一个惩罚项 $R_\Delta$：
    $R_{\text{FAPO}}(o, a^*|\theta) = R_{\text{RLVR}}(o, a^*) + R_\Delta(o, a^*|\theta)$
    其中，$R_\Delta(o, a^*|\theta) = \begin{cases} -\lambda, & \text{If } I(o, a^*) \text{ and } \hat{y}_\theta(o, a^*) = \text{FP} \\ 0, & \text{Otherwise} \end{cases}$
    $I(o, a^*)$ 是指示函数，表示最终答案是否正确。$\lambda$ 控制惩罚强度。
- **优势估计：** 采用 group-relative 优势估计：$\hat{A}_{i,t} = \left[ r_i - \text{mean}(\{R_i\}^G_{i=1}) \right] / \text{std}(\{R_i\}^G_{i=1})$。
- **理论分析：** 论文提供了理论分析，解释了 FAPO 如何实现优化方向的自然转移和训练过程的稳定性。当当前 rollout 包含 $\alpha$ 比例的正样本和 $\beta$ 比例的负样本时，学习进程 $\rho = \alpha / \beta$ 达到 $2\lambda - 1$ 时，优化从热身阶段转向细化阶段。当 $\rho > 4\lambda - 1$ 时，正样本的优势估计会被下调，使优化更稳定。
- **$\lambda$ 的确定：** 采用“多数指导策略 (majority-guided strategy)”，即优化方向由正样本或负样本的主导决定。这导致 $\rho_{\text{shift}} = 1$，从而确定 $\lambda = 1$ 作为默认设置。这使得在负样本占多数的早期阶段，缺陷正向仍能提供积极信号；当完全正确 rollout 成为多数时，优化自然转向强化可靠性。
<img width="867" height="528" alt="image" src="https://github.com/user-attachments/assets/7fb19be8-b42c-44f5-8d07-a7c1f4e559af" />

<img width="867" height="528" alt="image" src="https://github.com/user-attachments/assets/aa38483b-7e45-4346-8745-a329f5fc6769" />

<img width="427" height="403" alt="image" src="https://github.com/user-attachments/assets/dd24e3e9-8ef5-4f94-8871-be6497ab6b4d" />

**4. 实验结果**
- **FAPO-GenRM 性能：** 训练的 FAPO-GenRM-4B 在 FlawedPositiveBench 和 ProcessBench 上取得了显著提升，甚至优于其教师模型 Qwen3-32B 和判别式 SOTA 模型 Qwen2.5-Math-PRM-72B。
- **FAPO-Reasoning 性能：** Qwen2.57b/32b模型 500steps
    - **结果正确性：** 在 AIME24、AIME25 和 GPQA-Diamond 等数学和通用领域任务上，FAPO 始终优于基线模型。
    - **过程可靠性：** FAPO 生成的响应显著降低了缺陷正向比例，经过 LLM-as-a-judge 和人工验证均得到证实。
    - **训练稳定性：** FAPO 缓解了缺陷正向的影响，使学习曲线更平滑，且在训练后期没有出现显著的性能下降。
    - **Token 预算：** FAPO 的改进不依赖于增加响应长度，实现了更高效的推理。
- **消融研究：** 验证了 FAPO-GenRM 方法的有效性，更强的检测能力能转化为最终 RL 性能的提升。还分析了自校正能力的影响，FAPO 在早期利用自校正，后期转向更短、更高效的完全正确 rollout。
- **讨论：**
    - **算法挑战 (奖励劫持)：** 细粒度奖励信号容易导致策略利用信号缺陷而非真正执行任务。FAPO 通过其可解释的框架和理论分析，在对抗奖励劫持方面表现出鲁棒性。
    - **基础设施挑战 (长尾问题)：** GenRM 引入了额外的生成阶段。为解决效率问题，FAPO 采用异步设计，将 GenRM 从 rollout 推理和 Actor 训练中解耦，减少 GPU 空闲时间，使训练时间增加不到 20%。
<img width="1013" height="474" alt="image" src="https://github.com/user-attachments/assets/810839ab-1b1c-47e8-af29-b02faf9af5f8" />

**5. 结论**
FAPO 算法通过揭示缺陷正向 rollout 的双重作用（早期加速能力提升，后期限制推理质量），并提出了一种参数无关的奖励调整机制来调和这一权衡。它利用 GenRM 准确检测和定位推理错误。实验和理论分析均证明了 FAPO 在提高 LLM RL 效率和可靠性方面的有效性。

## Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training
NVIDIA, 2025.7.16

1.  📚 本研究深入探讨了对小型语言模型进行长时间强化学习以解锁多样化推理能力的方法，并强调了可验证奖励、改进的GRPO算法以及训练稳定性技术的重要性。
2.  ⚙️ 论文提出了多项关键技术，包括**解耦裁剪、动态采样、受控KL正则化和周期性参考策略重置**，以有效应对熵崩溃和训练停滞问题，从而实现性能的持续提升。
3.  🚀 实验结果表明，该方法在数学、编码、逻辑谜题等多个推理领域取得了显著进展，相比基线模型分别提升了14.7%、13.9%和54.8%，并且研究团队已公开发布了Nemotron-Research-Reasoning-Qwen-**1.5B模型**。

本论文深入探讨了如何通过延长强化学习（RL）训练来提升小型语言模型（LLMs）在多样化推理任务上的能力。通过可验证的奖励任务、改进的Group Relative Policy Optimization (GRPO)算法以及提升训练稳定性和泛化性的实用技术，即使是小规模模型也能在不依赖更大计算架构的情况下实现显著的推理能力提升。
<img width="1114" height="413" alt="image" src="https://github.com/user-attachments/assets/32854abb-5e3e-4da4-8fbe-f23794765884" />

**1. 引言与背景**

**2. 多样化训练数据**
为了促使模型泛化并学习鲁棒的决策策略，本研究在多种具有可验证奖励信号的任务上进行训练，涵盖了：
*   **数学 (Math):** 使用来自DeepScaleR数据集的4万个数学问题，采用二元奖励（1表示正确，0表示错误）。
*   **代码 (Code):** 使用Eurus-2-RL数据集的2.4万个编程问题，采用连续奖励，基于通过测试用例的比例。
*   **STEM:** 使用SCP-116K数据集（经过GPT-4o筛选）的2.5万个科学问题，采用二元奖励。
*   **逻辑谜题 (Logical Puzzles):** 使用Reasoning Gym项目生成的3.7万个合成训练样本，采用连续奖励。
*   **指令遵循 (Instruction Following):** 使用Llama-Nemotron的1万个合成生成数据，采用连续奖励。
为了应对复杂性和多样性，研究采用了沙盒奖励服务器架构，以隔离执行环境、确保安全性和容错性，并通过多进程和分布式服务器高效地扩展奖励评估。

**3. 核心方法**
研究基于Group Relative Policy Optimization (GRPO)算法，并整合了DAPO中的技术，同时引入了KL散度惩罚和周期性参考策略重置，以实现长时间的稳定训练。

**3.1. GRPO背景**
**3.2. 缓解熵坍缩**
为解决长时间策略优化中的熵坍缩问题（模型输出分布过早集中，限制探索），本研究采用了以下策略：
*   **高采样温度 (High Sampling Temperature):** 在rollout阶段使用较高的采样温度（例如1.2）以促进探索，但仅能延迟熵坍缩。

**3.3. 解耦裁剪和动态采样策略优化 (DAPO)**
为了进一步解决熵坍缩并维持探索和输出多样性，本研究采纳了DAPO [4] 的组件：
*   **解耦裁剪 (Decoupled Clipping):** 将PPO目标中的上下裁剪边界 \(\epsilon_{low}\) 和 \(\epsilon_{high}\) 作为独立超参数：
    \[ \text{clip}(r_\theta(\tau), 1 - \epsilon_{low}, 1 + \epsilon_{high}) \]
    通过设置更高的 \(\epsilon_{high}\) 值（例如0.4），鼓励“向上裁剪”，提升之前不太可能出现的token的概率，从而促进更广泛的探索。
*   **动态采样 (Dynamic Sampling):** 过滤掉模型持续成功或失败（准确率1或0）的prompt，因为这些prompt无法提供学习信号，从而使训练更聚焦于中等难度的样本，维持多样化和稳定的学习信号。

**3.4. KL正则化和参考策略重置**
虽然DAPO和温度调整有助于减缓熵坍缩，但显式的KL散度惩罚提供了更强大、更稳定的解决方案。
*   **KL惩罚 (KL Penalty):** 在GRPO损失中加入当前策略 \(\pi_\theta\) 和参考策略 \(\pi_{ref}\) 之间的KL散度惩罚：
    \[ \mathcal{L}_{\text{KL-RL}}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta) - \beta D_{\text{KL}}(\pi_\theta || \pi_{ref}) \]
    其中，无偏估计量通常为：
    \[ D_{\text{KL}}(\pi_\theta || \pi_{ref}) = \mathbb{E}_{\tau \sim \pi_{\text{ref}}} \left[ \frac{\pi_\theta(\tau)}{\pi_{\text{ref}}(\tau)} - \log \frac{\pi_\theta(\tau)}{\pi_{\text{ref}}(\tau)} - 1 \right] \]
    这个惩罚不仅有助于维持熵，还作为正则化项防止在线策略漂离稳定参考，从而稳定学习并减轻对虚假奖励信号的过拟合。与近期一些工作主张移除KL惩罚不同，本研究认为从一个已具备CoT输出能力的强预训练模型（如DeepSeek-R1-Distill-Qwen-1.5B）开始训练时，保留KL惩罚对稳定性和熵维持仍有益。
*   **参考策略重置 (Reference Policy Reset):** 周期性地将参考策略 \(\pi_{ref}\) 硬重置为在线策略 \(\pi_\theta\) 的最新快照，并重新初始化优化器状态。这使得模型能在保持KL正则化益处的同时持续改进，避免过早收敛并鼓励长时间训练。

<img width="975" height="684" alt="image" src="https://github.com/user-attachments/assets/27c376ec-9d21-4ac0-9a27-68c3e8248efb" />

**4. 实验结果**
研究采用分阶段训练策略，包括多个顺序运行，以迭代优化模型行为、整合额外数据、调整超参数和重置训练动态。
*   **训练设置:** 使用开源框架verl [16] 进行RL训练，DAPO增强（\(\epsilon_{low}=0.2, \epsilon_{high}=0.4\)），KL散度惩罚系数 \(\beta=0.0001\)。rollout采样16个响应，上下文窗口8096，采样温度1.2。批大小256，mini-batch大小64。AdamW优化器，学习率 \(2 \times 10^{-6}\)。初始模型为DeepSeek-R1-Distill-Qwen-1.5B。总训练时长约1.6万GPU小时。
*   **训练过程:** 经历了多个运行阶段，包括初始训练、硬重置、引入指令遵循数据、奖励塑形（惩罚未正确终止的响应）、增加rollout计数以及扩展上下文窗口至16k等。
*   **评估基准:** 在数学（AIME2024/2025, AMC, MATH, Minerva Math, Olympiad Bench）、编码（PRIME, HumanevalPlus, LiveCodeBench）、逻辑谜题（Reasoning Gym）、STEM推理（GPQA Diamond）和指令遵循（IFEval）等多样化任务上进行评估。
*   **主要发现:**
    *   与DeepSeek-R1-Distill-Qwen-1.5B相比，本模型Nemotron-Research-Reasoning-Qwen-1.5B在数学上平均提升15.7%，编码上表现优异，STEM推理和指令遵循分别提升25.9%和22.0%。在逻辑谜题上也取得了显著进步，尤其是在克服初始格式错误后。
    *   与领域专用模型（DeepScaleR-1.5B、DeepCoder-1.5B）相比，本模型在广泛领域数据上训练后，仍能在数学和代码基准上达到竞争性性能，展现出强大的泛化能力。

**5. 消融研究**
*   **Rollout采样温度:** **在早期和后期训练中，较高的采样温度（例如1.2**）都能带来更稳定的训练和更好的性能，能防止模式坍缩并支持持续进步。
*   **解耦裁剪和动态采样:** 设置 \(\epsilon_{low}=0.2\) 和 \(\epsilon_{high}=0.4\) 取得了最佳验证性能，缓解了熵坍缩。动态采样通过过滤无优势的prompt，提高了每批次的奖励信号密度，从而提升了样本效率。
*   **重置参考策略:** 当训练性能下降或停滞时，**硬重置参考策略和优化器状态能恢复稳定性**并实现进一步的有效训练。
*   **缓解熵坍缩:** 比较了多种策略（GRPO-KL=0, GRPO-KL=1e-4, DAPO, AdaptiveEntropy），发现DAPO和KL惩罚的组合提供了一种保守而鲁棒的解决方案，既有助于熵保持，又通过防止模型偏离参考策略来提高训练稳定性。
<img width="1114" height="413" alt="image" src="https://github.com/user-attachments/assets/e5a4bdc4-1a7e-4bdf-a83b-9e791c0df466" />
<img width="1110" height="540" alt="image" src="https://github.com/user-attachments/assets/4e5ff27c-5fc9-4123-b6c8-e3a24d4c4782" />


## Metis-HOME
Metis-HOME: Hybrid Optimized Mixture-of-Experts for Multimodal Reasoning

https://arxiv.org/abs/2510.20519 美团等 2025.11.25

https://github.com/MM-Thinking/Metis-HOME

1. ✨ 本文提出了 Metis-HOME 框架，这是一种**混合优化专家混合**（MoE）模型，旨在解决多模态推理模型中**复杂推理能力与通用能力之**间的效率低下和能力退化问题。
2. 🧠 Metis-HOME 引入**“混合思维**”范式，通过**结构化为专门处理复杂多步推理的“思考专家”分支**和**优化快速直接推理的“非思考专家”分支，**并由**轻量级路由器动态**分配查询。
3. 🚀 实验证明，Metis-HOME 不仅大幅提升了复杂推理能力，还逆转了推理专业化模型普遍存在的通用能力下降趋势并有所提升，有效解决了推理能力与泛化能力之间的固有矛盾。

<img width="780" height="345" alt="image" src="https://github.com/user-attachments/assets/e89b6eb0-794c-4587-864c-ce6b18acfb25" />

本文提出了 Metis-HOME，一个混合优化专家混合 (Hybrid Optimized Mixture-of-Experts, MoE) 框架，旨在解决多模态大推理模型 (MLLMs) 中复杂推理能力与通用能力之间的权衡问题。当前 MLLMs 在复杂任务（如数学问题求解）上表现出色，但存在两大局限：一是即使对于简单查询也倾向于使用计算昂贵的推理（“过度思考”），导致效率低下；二是对专业推理的过度关注往往会损害其更广泛的通用理解能力，从而在复杂推理性能提升的同时，通用视觉问答 (VQA) 和光学字符识别 (OCR) 等基础能力下降。

Metis-HOME 通过构建“混合思维”范式来解决这一困境。其核心方法是将原始的密集模型结构化为两个不同的专家分支：一个“思考分支 (thinking branch)”，专门用于复杂的多步推理任务；另一个“非思考分支 (non-thinking branch)”，则针对通用 VQA 和 OCR 等任务进行优化，以实现快速直接的推理。一个轻量级、可训练的路由器 (router) 动态地将查询分配给最合适的专家。

**核心方法论**

1.  **架构扩展 (Architectural Expansion)**：
    *   Metis-HOME 基于 Qwen2.5-VL-7B 模型进行实例化。对于原始密集模型中的每个 Transformer 块，其 Feed-Forward Network (FFN) 模块被复制并扩展为两个独立的专家：一个专用于“思考”，另一个专用于“非思考”处理。
    *   自注意力机制 (self-attention) 和其他组件在专家之间共享，这符合 MoE 架构的常见实践，并认识到思考与非思考模式在上下文关系学习上仍有大量共通之处。
    *   路由器采用多层感知机 (MLPs) 实现，负责根据输入类型和复杂性动态分配输入到相应的专家。
    *   MoE 权重初始化时使用经过 RL 训练的“思考”模型，理由是推理能力的形成需要更密集的专业训练，而通用能力可通过少量数据有效恢复。

2.  **训练策略 (Training Strategy)**：
    训练分为两个主要阶段：
    *   **第一阶段：强化学习 (Stage-RL)**：
        *   此阶段旨在显著增强基础密集模型的内在推理能力。
        *   采用 Metis-RISE [Qiu et al., 2025] 的策略，适配 Group Relative Policy Optimization (GRPO) 算法，并结合 DAPO [Yu et al., 2025] 和 VAPO [Yue et al., 2025] 中的先进优化技术。
        *   给定从数据池 $D$ 中采样的查询-答案对 $(q, a)$，行为策略模型 $\pi_{\theta_{old}}$ 生成一组 $G$ 个候选轨迹 $\{\tau_i\}_{i=1}^G$。RL 目标函数定义为：
            $$J_{RL}(\theta) = E_{(q,a)\sim D,\{\tau_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}\min\left(r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}\left(r_{i,t}(\theta), 1 - \varepsilon_{low}, 1 + \varepsilon_{high}\right) \hat{A}_{i,t}\right)\right]$$
            其中，重要性采样比率 (importance ratio) 和优势函数 (advantage) 计算为：
            $$r_{i,t}(\theta) = \frac{\pi_{\theta} (\tau_{i,t} | q, \tau_{i,<t})}{\pi_{\theta_{old}} (\tau_{i,t} | q, \tau_{i,<t})} , \hat{A}_{i,t} = R_i - \text{mean}(\{R_i\}_{i=1}^G)/\text{std}(\{R_i\}_{i=1}^G)$$
        *   采用混合奖励机制，结合了格式奖励 (format reward) 和准确性奖励 (accuracy reward)。格式奖励强制模型输出遵循 `<think>` 和 `</think>` 包含推理过程，`<answer>` 和 `</answer>` 包含最终答案的结构。准确性奖励是二元的 (0 或 1)，仅当提取的答案被基于规则的验证器验证为正确时才授予。
    *   **第二阶段：监督微调 (Stage-SFT)**：
        *   在此阶段，模型被扩展为 MoE 架构，并使用精心构建的思考型和非思考型混合数据集进行 SFT。
        *   **思考型 SFT 数据**：收集数学及其他复杂推理领域的提示。对于每个提示，模型生成 $N$ 个响应，计算通过率。通过率为 1 的提示被丢弃；通过率在 0 到 1 之间的，使用模型自身的正确推理轨迹作为监督信号；通过率为 0 的，利用外部专家模型提供参考推理轨迹。数据格式为：“`<think>thinking_content</think><answer>answer_content</answer>`”。
        *   **非思考型 SFT 数据**：汇编高质量的通用 VQA、OCR 和图像字幕示例，数据量与思考型数据相当。数据格式为：“`<think></think><answer>answer_content</answer>`”。
        *   训练目标结合了两个交叉熵损失：一个用于最终答案预测 ($L_{prediction}$)，另一个用于路由器的分配决策 ($L_{router}$)。总损失为：
            $$L_{total} = L_{prediction} + L_{router}$$
            这两个损失以 1:1 的比例结合。

**实验与结果**

Metis-HOME 在 Qwen2.5-VL-7B 模型上实现。RL 阶段使用约 40K 多模态推理样本训练。SFT 阶段使用约 16K 训练样本（8K 思考型，8K 非思考型）。

*   **推理能力提升**：在六个推理基准测试（MathVista, MathVision, MathVerse, DynaMath, WeMath, LogicVista）上，Metis-HOME 平均得分 46.1%，相比基线 (Qwen2.5-VL-7B) 提升了 6.9%。它超越了专用的推理模型 VL-Rethinker-7B，并与 Metis-RISE-7B 性能相当。
*   **通用能力保持与提升**：与其他推理专业化模型（如 VL-Rethinker-7B 和 Metis-RISE-7B）在通用能力上表现出显著下降不同，Metis-HOME 不仅避免了性能下降，反而在八个通用基准测试上取得了 0.9% 的整体提升，平均得分 71.2%。这表明 Metis-HOME 成功解决了推理与泛化之间的困境。
*   **思维比率分析 (Thinking Ratio Analysis)**：
    *   在推理密集型基准测试（如 WeMath, MathVision）上，“思考”比率显著偏高（78%至98%），表明路由器能有效识别复杂查询并导向思考专家。
    *   在通用基准测试（如 MMBench, OCRBench）上，“思考”比率显著降低（低至2%–5%），显示出对非思考专家的强烈偏好。
    *   MMMU 基准测试展现出 50.1% 的综合思考比率，进一步细致分析显示，对于分析性科目（如“图论”、“微积分”）思考比率高达 100%，而对于非推理型科目（如“美国文学”）则降至 0%，验证了路由器在多学科数据集内识别查询复杂度的能力。
    *   训练过程中，在 MathVerse 上，“思考”比率随训练步数增加而提高，并伴随准确率的同步提升，验证了“在适当问题上更多思考”能显著提升推理能力的假设。
    *   SFT 训练早期，模型倾向于过度拟合非思考响应模式导致思考比率下降；随着训练进行，模型逐渐学会区分任务，通用基准的思考比率保持低位，而复杂数学推理基准的思考比率则稳步回升并稳定在高水平。

定性分析进一步证实，Metis-HOME 的路由器能成功区分不同认知负荷的查询，动态激活最合适的专家分支，例如将 OCR 任务和图像字幕任务路由到非思考分支，而将复杂的平面几何问题路由到思考分支，生成详细的多步推理过程。

综上所述，Metis-HOME 提出了一个有效的混合 MoE 架构，通过专用的专家分支和动态路由器实现了“混合思维”范式，成功解决了 MLLMs 中复杂推理能力与通用能力之间的固有矛盾，为构建强大而通用的 MLLMs 提供了新范式。

## CDLM
CDLM: Consistency Diffusion Language Models For Faster Sampling 

https://arxiv.org/abs/2511.19269 2025.11.24 伯克利 首尔 togetherAI等

https://github.com/SqueezeAILab/CDLM

1.  ✨ 本文提出了CDLM（Consistency Diffusion Language Models），一种基于训练的加速方法，旨在解决**扩散语言模型（DLMs）采样速度慢和无法使用标准KV缓存**两大瓶颈。
2.  🚀 CDLM通过整合一致性建模实现**多令牌并行确认**，显著减少了所需的采样步骤，并通过在微调时强制使用**块级因果注意力掩码**，使模型完全兼容KV缓存。
3.  📊 实验结果表明，CDLM在数学和编程任务上实现了**3.6至14.5倍的延迟降低**，并减少了**3.4至7.9倍的精炼步数**，同时保持了有竞争力的准确性，甚至在每秒令牌数上超越了同等大小的自回归大型语言模型。

CDLM (Consistency Diffusion Language Models) 论文提出了一种基于训练的加速方法，旨在解决扩散语言模型 (DLMs) 在推理过程中面临的两个主要瓶颈：迭代步骤过多导致推理缓慢，以及双向注意力机制导致无法使用标准的 KV 缓存。CDLM 通过整合一致性建模 (consistency modeling) 来显著减少所需的采样步骤，并引入多令牌终结化 (multi-token finalization)；同时，在微调过程中强制采用块级因果注意力掩码 (block-wise causal attention mask)，使模型与 KV 缓存完全兼容。

**1. 引言与背景**
传统的自回归语言模型 (autoregressive LLMs) 虽然训练可并行化，但推理本质上是序列化的，且无法利用双向上下文。DLMs 作为一种有前景的替代方案，通过迭代去噪随机或掩码令牌序列来生成文本，每一步并行更新所有令牌位置，消除了令牌级别的序列依赖。然而，现有开源 DLMs 存在两大挑战：(1) 双向注意力阻碍了 KV 缓存的使用；(2) 高质量生成通常需要大量精炼步骤，导致推理效率低下。

**2. CDLM 方法论**
CDLM 通过联合优化三个目标来解决上述问题：
*   **一致性引导蒸馏 (Consistency-guided Distillation)**：定义了一个令牌级解码轨迹，并结合从一个全双向教师模型 (fully bidirectional teacher) 进行蒸馏。
*   **缓存友好的块级因果性 (Cache-friendly Block Causality)**：通过块级因果注意力掩码进行微调，使模型支持块级 KV 缓存和块边界的提前停止。
*   **端到端加速 (End-to-end Speedups)**：综合上述两点，实现了推理速度的显著提升。

**2.1. 轨迹收集 (Trajectory Collection)**
为了训练 CDLM，首先需要从教师 DLM $p_\theta$ 生成解码轨迹。教师模型采用标准 DLM 中的双向注意力掩码，而学生模型 $q_\phi$ 则使用块级因果掩码进行训练。轨迹收集过程如下：
1.  对于给定的提示 $x$ 和真实文本 $\hat{y}$，教师模型 $p_\theta$ 运行块级解码，步数 $N$ 等于生成长度 $L_g$，每步在当前块内终结一个令牌（置信度最高的令牌 $y_i$）。
2.  记录令牌级轨迹 $T_x$ 以及在令牌终结时教师模型最后一层的隐藏状态 $H_x \in \mathbb{R}^{L_g \times d}$，以用于后续的 logit 重构。
3.  为增加数据多样性，每个提示会生成多个不同温度（如 $\{0.0, 0.5\}$）下的轨迹。

**2.2. 训练目标 (Training Objectives)**
CDLM 的训练目标是联合最小化以下三个损失函数：
1.  **蒸馏损失 (Distillation Loss) $L_{Distillation}$**：
    该损失将教师模型的多令牌终结化信号传递给块级因果学生模型。对于轨迹中的中间状态 $y$ 和其块完成状态 $y^\star$，定义 $U_y$ 为在 $y$ 和 $y^\star$ 之间新解除掩码的令牌索引集。蒸馏损失计算学生模型在状态 $y$ 时对这些新解除掩码位置的预测与教师模型对应位置预测之间的 KL 散度：
    $$L_{Distillation} = \mathbb{E}_{(x,T_x,H_x)\sim\mathcal{D}} \mathbb{E}_{y\sim T_x}\left[\frac{1}{|U_y|}\sum_{i\in U_y} D_{KL}\left(p^{(T)}_i \| q_\phi(\cdot | y, x)_i\right)\right]$$
    其中 $p^{(T)}_i$ 是通过教师模型在 $H_x$ 中存储的隐藏状态 $h_{x,i}$ 经 $lm\_head$ 重构的分布。

2.  **一致性损失 (Consistency Loss) $L_{Consistency}$**：
    该损失强制学生模型在同一块内两个不同状态（$y$ 和 $y^\star$）下的预测在未被解除掩码的位置保持一致。定义 $S_y$ 为在 $y^\star$ 时仍被掩码的令牌索引集。一致性损失计算学生模型在状态 $y$ 对 $S_y$ 中令牌的预测与在状态 $y^\star$ 对 $S_y$ 中令牌的预测之间的 KL 散度，其中 $q_\phi^-$ 表示停止梯度的目标：
    $$L_{Consistency} = \mathbb{E}_{(x,T_x)\sim\mathcal{D}} \mathbb{E}_{y\sim T_x}\left[\frac{1}{|S_y|}\sum_{i\in S_y} D_{KL}\left(q_\phi^-(\cdot | y^\star, x)_i \| q_\phi(\cdot | y, x)_i\right)\right]$$
    这使得模型能够稳定地跳过多个步骤。

3.  **DLM 损失 (DLM Loss) $L_{DLM}$**：
    这是一个辅助损失，与标准 DLM 预训练中使用的掩码去噪目标相同，用于保持模型的掩码预测能力。对于输入 $x$ 和真实响应 $\hat{y}$，随机采样掩码比率 $t \sim U[0, 1]$ 生成掩码序列 $\hat{y}_t$，然后计算模型预测与真实标签的交叉熵：
    $$L_{DLM} = -\mathbb{E}_{(x,\hat{y})\sim\mathcal{D}} \mathbb{E}_t\left[\frac{1}{tL_g}\sum_{i=1}^{L_g}\mathbf{1}_{[\hat{y}_{t,i} = \text{[MASK]}]} \log q_\phi(\hat{y}_i | \hat{y}_t, x)\right]$$

总训练目标为：
$$L(\phi) = w_{distill} L_{Distillation} + w_{cons} L_{Consistency} + w_{dlm} L_{DLM}$$
其中 $w_{distill}, w_{cons}, w_{dlm}$ 是对应损失的权重。

**2.3. 推理 (Inference)**
在推理阶段，CDLM 学生模型在块级因果掩码下进行块级解码，并复用提示和已完成块的 KV 缓存。在每个块内部，采用置信度阈值并行终结化 (confidence-thresholded parallel finalization)：在每一步中，当前块内所有置信度超过阈值 $\tau_{conf}$ 的掩码令牌都会被揭示。此外，模型支持提前停止，一旦当前块内生成 `<endoftext>` 令牌则停止解码。

**3. 实验结果**
CDLM 在数学推理和代码生成任务（GSM8K, MATH, HumanEval, MBPP）上进行了评估，基准模型包括 Dream-7B-Instruct 和 LLaDA-8B-Instruct。

*   **加速效果**：CDLM 实现了 3.6 倍至 14.5 倍的更低延迟，并减少了 3.4 倍至 7.9 倍的精炼步骤。
*   **准确性**：在保持有竞争力的准确性的同时，在 MBPP-Instruct 和 HumanEval-Instruct 等任务上甚至有所提升。虽然在某些任务（如 MATH）上可能略有下降，但总体性能优于或媲美现有加速方法。
*   **吞吐量 (TPS)**：CDLM 在大多数任务上实现了最高的每秒令牌数 (TPS)，与原始 DLMs 相比提升了 3 倍至 21 倍，并且超越了同等规模的自回归 LLMs。

**4. 总结**
CDLM 是一种将一致性建模引入扩散语言模型的训练方法，通过强制块内时间一致性和微调块级因果学生模型，有效地减少了精炼步骤并实现了自然的 KV 缓存。这使得 DLMs 在数学和编码任务上实现更快的推理速度、更少的步骤、更低的延迟和更高的吞吐量，同时保持了竞争力强的准确性。未来的工作方向包括扩展蒸馏语料库、拓宽领域覆盖以及从更强的教师模型进行蒸馏。

   
## Adaptive 投机 TLT
Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter

https://arxiv.org/pdf/2511.16665 MIT NVIDIA 韩松团队等 2025.11.21 ASPLOS26

https://github.com/mit-han-lab/fastrl 基于verl（fsdp+sglang）
 
https://mp.weixin.qq.com/s/yJ5C9w3Hdc5g91QCUaID3Q 

1. 📖 大语言模型推理强化学习训练因响应长度的长尾分布而面临严重效率瓶颈，即少数极长响应大幅增加了训练时间和资源消耗。
2. 💡 为此，本文提出TLT系统，通过集成自适应推测解码来无损加速训练，该系统包含：利用空闲GPU持续训练**轻量级草稿模型的“自适应草稿器”，** 以及**根据输入批次动态选择推测解码策略（基于蚂蚁等lookAhead）**的“自适应Rollout引擎”。
3. 🚀 实验表明，TLT系统相比现有SOTA verl实现了超过1.7倍的端到端RL训练加速，同时保持了模型精度，并免费生成了高质量的草稿模型。
<img width="428" height="321" alt="image" src="https://github.com/user-attachments/assets/b68a3031-f648-4cba-b4b8-a4b9a4438b53" />

本文提出了一种名为 TLT（Taming the Long-Tail）的系统，旨在通过集成自适应推测解码（adaptive speculative decoding）无损地加速推理强化学习（Reasoning Reinforcement Learning, RL）训练。推理 LLM 的训练（通常采用 RL）面临关键效率瓶颈：在 RL 训练过程中，响应生成呈现持久的长尾分布，即少数极长的响应占据了大部分执行时间，导致资源浪费和成本增加。

<img width="511" height="337" alt="image" src="https://github.com/user-attachments/assets/c4954f09-ab31-4020-8f9d-f8a3f0cde02d" />

TLT 旨在解决这一问题，其核心设计基于两个洞察：首先，利用 RL rollout 阶段特有的长时间特性，以及随着长尾效应逐渐释放的 GPU 资源（即“rollout bubbles”），将这些资源用于其他任务，如 draft model 训练，而无需额外成本。其次，draft model 的训练可以与完整的 rollout 完成解耦，通过异步处理部分或可用数据，使其计算可以与正在进行的 rollout 有效重叠。

<img width="647" height="310" alt="image" src="https://github.com/user-attachments/assets/6784da40-d3b1-4ee6-8626-32066ff4f3a6" />


<img width="847" height="605" alt="image" src="https://github.com/user-attachments/assets/1e87106c-d939-4c83-a91a-d483f29c5594" />

TLT 包含两个协同组件来克服 SD 在 RL 中应用的挑战（动态工作负载、不断演进的目标模型和 draft model 训练开销）：

<img width="1503" height="594" alt="image" src="https://github.com/user-attachments/assets/2f4bfc75-1a13-426e-af72-d30d0fce93e2" />

1.  **Adaptive Drafter（自适应草稿模型）**：
    *   **Draft Model 架构**：TLT 采用轻量级、单层模型作为 drafter，它复用目标模型（target model）的 Embedding 和 LM Head 层，仅训练一个 Transformer 解码器层。这种设计显著减少了训练和推理开销，并被证明能够紧密匹配目标模型的自回归分布。
    *   **Spot Trainer（即时训练器）**：为解决目标模型在 RL 训练中不断更新导致的 draft model 过时问题（C1），Spot Trainer 利用 rollout 过程中空闲的 GPU 资源进行机会性（opportunistic）、可抢占的 drafter 更新，从而确保其与不断演进的目标模型保持一致，且不增加额外开销（C2）。
        *   **Worker Coordinator**：负责协调资源分配，当 rollout 过程中出现空闲 GPU 资源时，将其重新用于 drafter 训练。它通过监控 worker 状态（BUSY, IDLE, TRAINING）来决定何时启动或停止训练。
        *   **Spot Training with DataBuffer**：drafter 训练不必等待所有 rollout 响应完成。Spot Trainer 利用部分早期完成的响应进行训练，并通过 DataBuffer 缓存推理阶段的 hidden states 和 tokens。DataBuffer 支持“一步偏移采样”（one-step offset sampling），即保留前一步骤的长序列数据，以弥补当前部分数据中长尾数据的稀缺性，从而解决分布不匹配问题。
        *   **Selective Asynchronous Checkpointing（选择性异步检查点）**：为确保 drafter 训练的可抢占性而不丢失大量训练进度，TLT 采用异步检查点机制，将保存 draft model 状态的任务卸载到后台线程，并仅保存可训练参数，显著减少了检查点延迟。
        *   **Sequence Packing（序列打包）**：为了最大化 GPU 利用率，Spot Trainer 使用序列打包技术将多个变长序列连接成一个打包序列，消除填充（padding）的计算浪费。

<img width="720" height="674" alt="image" src="https://github.com/user-attachments/assets/72592630-4cfb-4a38-8fb2-12961a765123" />

2.  **Adaptive Rollout Engine（自适应 Rollout 引擎）**：
    *   **Tree-based Drafting（基于树的草稿生成）**：该引擎采用基于树的推测解码，通过在每个步骤探索 topK 个最可能的选项并分支，构建候选序列树，然后选择最高置信度的 tokens 提交给目标模型进行并行验证，以增加每次验证接受的 tokens 数量。
    *   **Memory-Efficient CUDAGraph Capture（内存高效的 CUDA 图捕获）**：为了应对动态变化的 batch size（C3）并支持多种 SD 策略，同时避免 CUDA Graph 带来的高内存开销，TLT 提出了分桶式 CUDA Graph 捕获（Bucketed CUDAGraph Capture）。它通过将 batch size 分组到不同桶中、解耦捕获目标模型和草稿模型的 CUDA Graph、以及合并共享相同参数设置的策略的捕获，显著减少了内存占用。
    *   **Auto-Tune Algorithm (BEG-MAB)**：TLT 使用 Bucketed-Epsilon-Greedy (BEG) MAB 选择器算法来自动化 SD 策略的选择。该算法根据 `Tokens_to_Verify` 将策略分组，并将其动态匹配到 batch size 范围。它采用 $\epsilon$-greedy 策略，以小概率进行探索，大概率利用（选择中位数奖励最高的策略），从而平衡探索和利用，并适应 RL 训练中的非稳态动态。
    *   **Model-free Drafter（无模型草稿生成器）**：除了基于学习的 drafter，TLT 还集成了一个互补的、非参数化的无模型草稿生成策略。它通过为每个 prompt 动态构建 n-gram 检索数据库，利用序列相似性和重复模式来预测后续 token 序列。该 drafter 在基于学习的 drafter 不可用（例如初始训练步骤）或效率低下时作为备用机制。


<img width="601" height="448" alt="image" src="https://github.com/user-attachments/assets/75bb2f3e-a7d7-414d-89d8-0f9260dc539b" />


<img width="702" height="424" alt="image" src="https://github.com/user-attachments/assets/9dd1411e-ff6b-47a1-9b97-8ba9e72a0400" />


TLT 通过这些协同优化，有效缓解了 RL 训练中固有的长尾问题。实验结果表明，TLT 比最先进的系统实现了超过 1.7 倍的端到端 RL 训练加速，同时保持了模型精度，并免费生成了一个高质量的 draft model，适用于高效部署。

关键技术细节：
*   **Adaptive Drafter 训练流程**：利用 RL 推理阶段产生的 hidden states 和 token embedding，通过轻量级线性层降维后作为 drafter 单解码器层的输入。训练目标可以是目标模型与 draft model hidden states 之间的 L1 损失，或输出 logits 上的交叉熵损失，或两者结合。
*   **BEG-MAB 选择器**：
    *   **输入**：策略集 $S$、Batch 阈值 $T = \{t_1, \dots, t_m\}$、探索率 $\epsilon$、窗口大小 $w$。
    *   **初始化**：根据 `Tokens_to_Verify` 将策略分组 $S_i$，定义 batch size 桶 $B_i$，并将桶映射到策略组。为每个策略初始化双端队列 $R_s$ (奖励) 和 $A_s$ (接受长度) 记录，大小为 $w$。
    *   **记录**：在每次生成步骤后，计算接受率 $a = (\sum \text{accept\_lens}) / \text{batch\_size} + 1$，以及奖励 $r_s = a \times \text{batch\_size} / \text{elapsed\_time}$，并将其添加到 $R_s$ 和 $A_s$ 中。
    *   **选择策略**：根据当前 `batch_size` 确定候选策略集 $V$。如果 $|V|=1$，直接返回该策略。否则，以概率 $\epsilon$ 从 $V$ 中随机选择（探索），以概率 $1-\epsilon$ 选择使 $R_s$ 中位数最大的策略（利用）。

**实现**：基于VeRL，其中Adaptive Drafter基于EAGLE FSDP2；异步检查点，基于PyTorch DCP。采用 SGLang作为 rollout 后端，并按照 [55] 实现 SD 的自适应启用、drafter 权重更新和 BEG-MAB 调优器。使用 [68] 作为model-free drafter。
**评测**：8台DGX H100=合计64个 GPU，节点间通信NVIDIA Mellanox 400Gb/s InfiniBand 实现。
**模型**：4个模型，最大70b：Qwen2.5-7b（base），DS-R1-蒸馏-7b，Qwen-32b-base，LLama-3.1-70b-instruct
**基线**：3个：Open-R1（vllm deepseed，分离式），verl，verl+基本lookahead投机（TLTbase）


<img width="1071" height="286" alt="image" src="https://github.com/user-attachments/assets/0e447ebb-45d1-4d8f-a279-3212aea1458f" />

<img width="1074" height="285" alt="image" src="https://github.com/user-attachments/assets/ef42eb45-bf65-4a50-97e5-d680ae08146b" />

<img width="1073" height="291" alt="image" src="https://github.com/user-attachments/assets/4fe933cc-3d64-45ae-8994-14881835a342" />

<img width="1083" height="644" alt="image" src="https://github.com/user-attachments/assets/6af1846f-70ce-4172-a6d1-6e10c0b99d9c" />


<img width="1091" height="499" alt="image" src="https://github.com/user-attachments/assets/312e9d1b-0cc7-4288-a6dc-2b5cf0efd116" />



<img width="716" height="536" alt="image" src="https://github.com/user-attachments/assets/8b99fdc7-51e0-4a80-a734-dc54a60d641b" />

## C3 压缩
Context Cascade Compression: Exploring the Upper Limits of Text Compression

https://arxiv.org/pdf/2511.15244 2025.11.19
https://github.com/liufanfanlff/C3-Context-Cascade-Compression

对context输入小模型压缩到latent token（5x～20x），然后大模型decode还原准确率98%！
https://mp.weixin.qq.com/s/0TfoM48EPlLfceZ6xCyL5Q

1.  📚 本文提出了Context Cascade Compression (C3)方法，旨在探索文本压缩的上限，以应对大型语言模型在处理长上下文时面临的计算和内存挑战。
2.  💡 C3采用级联的双LLM架构，其中一个小型LLM负责将长文本压缩为一组潜在token，另一个大型LLM则根据这些压缩的上下文执行解码任务。Qwen2.5-1.5b作encoder；Qwen2.5-3b作decoder。
3.  🚀 实验证明，在20倍压缩比下，C3的解码准确率高达98%，远超DeepSeek-OCR的60%，在40倍压缩比下仍能保持93%的准确率，展示了其在纯文本压缩方面的卓越性能和潜力。

<img width="830" height="319" alt="image" src="https://github.com/user-attachments/assets/e80905be-e1be-4aed-958c-c470e0ff2760" />

大型语言模型（LLMs）在处理百万级别token的长文本输入时面临巨大的计算和内存挑战。受DeepSeek-OCR在“Contexts Optical Compression”方面初步成果的启发，本文提出了一种名为Context Cascade Compression (C3) 的方法，旨在探索文本压缩的极限。C3方法采用两阶段级联架构，利用大小不同的LLM分别执行压缩和解码任务。

<img width="683" height="455" alt="image" src="https://github.com/user-attachments/assets/d4e2d1ee-deec-4f2b-bb5f-77c9bfc41b2b" />

**核心方法论（Core Methodology）：**
C3架构的核心是一个级联的双LLM设计，包括一个上下文压缩编码器LLM和一个解码器LLM。

<img width="678" height="434" alt="image" src="https://github.com/user-attachments/assets/1b7c099d-a95e-469b-867c-3874fb66d0fa" />

1.  **上下文压缩编码器LLM (Context Compression Encoder LLM):**
    *   **目标：** 将变长的文本序列压缩成定长的潜入（latent）表示。
    *   **架构：** 该编码器以一个预训练的Qwen2.5 1.5B模型作为骨干。作者认为**预训练LLM本身就具备高级的信息提取、语义理解和总结能力**，因此直接适配现有LLM而不是从头设计压缩模块。
    *   **压缩机制：** 引入一组可学习的嵌入，称为“context query”。这个查询是一个可训练的张量 $\mathbf{Q}$，其维度为 $N \times D$，其中 $N$ 是期望输出潜入上下文的固定token数量（例如32、64或100），$D$ 是Qwen2.5 1.5B模型嵌入层的隐藏维度。
    *   **输入处理：** 编码器的输入序列是原始长文本（text tokens）与context query嵌入的拼接。模型将**这种混合序列统一处理，context query嵌入在模型的自注意力机制中被视为标准文本token。**无需引入跨注意力层等架构修改，整个前向传播完全依赖模型原生的因果注意力机制。
    *   **输出：** 在前向传播完成后，**提取对应于context query token位置的最后一层输出的隐藏状态。这个形状为 $N \times D$ 的张量构成了高效且密集的原始文本表示，**即“latent context”，并被传递给下游的解码器。

2.  **解码器LLM (Decoder LLM):**
    *   **目标：** 解释编码器提供的密集、压缩的latent context，并生成满足特定下游任务的连贯文本输出。
    *   **架构：** 本文使用一个更大的LLM，即Qwen2.5 3B，作为解码器。
    *   **任务：** 在本文的研究范围内，解码器主要执行文本重建任务，以此作为评估C3压缩架构信息保真度的直接且严格的基准。通过让模型从其压缩表示中完美重建原始输入文本，可以量化压缩-解压缩周期中保留的信息量。
    *   **输入：** 解码器的输入是latent context与任务特定提示（prompt）的拼接。对于重建任务，使用的显式指令是"repeat the text: "。
    *   **训练：** 解码器被训练以自回归方式生成与原始真实文本相同的token序列，以此证明输入文本的语义完整性已在latent context中成功维持。

与DeepSeek-OCR等光学压缩方法不同，C3采用更简洁的纯文本管道，忽略了布局、颜色和视觉编码器造成的信息损失等因素。这种直接的文本到潜入（text-to-latent）压缩范式避免了视觉模态固有的信息瓶颈和潜在伪影（例如图像分辨率限制、布局复杂性）。

**主要贡献：**
*   提出了Context Cascade Compression (C3)，一种实现长上下文高效压缩的新颖架构。
*   实现了远超光学字符压缩的性能。在40倍压缩比（文本token数量是潜入token数量的40倍）下，模型仍保持93%的解码准确率，而DeepSeek-OCR约为10倍压缩比。
*   分析了C3的遗忘模式，发现其表现出序列性信息损失，错误倾向于集中在文本末尾，这与人类记忆衰减过程高度相似。

**实验设置与结果：**
*   **数据：** 使用从互联网收集的100万页OCR数据进行训练，主要包含英文和中文文档。
*   **训练：** 在8块NVIDIA H800 GPU上进行，全局批次大小为256，使用AdamW优化器，峰值学习率为1e-5，采用余弦学习率调度器，总训练步数为40,000步。
*   **评估：** 采用Fox基准测试的英文文档部分，选择token数量在600到1300之间的文档。在32、64和100个latent token的不同压缩级别下进行性能评估，任务是文本重建。

<img width="692" height="699" alt="image" src="https://github.com/user-attachments/assets/71d954d4-09c1-4c68-ac18-404b553b371e" />

**定量结果：**
*   **在64个latent token的设置下，当压缩比接近20倍（1200-1300个文本token压缩为64个latent token）时，C3仍保持98.4%的重建精度，** 而DeepSeek-OCR在类似条件下精度骤降至59.1%。
*   在更极端的32个latent token设置下，即使压缩比接近**40倍**（1200-1300个文本token压缩为32个latent token），C3仍能保持93.3%的惊人精度。
*   总体而言，C3在所有测试条件下都显著优于光学压缩方法，在相似压缩率下展现出卓越的信息保存能力，尤其在高压缩比下性能差距更为显著。

**定性结果：**
*   模型在标准英文散文、古典中文、掺杂随机字符的英文段落以**及结构混乱的中文段落上，即使在极高的压缩级别（32个latent token）下，也能实现近乎完美的重建**。
*   C3的错误**模式表现为序列性信息损失，即文本开头部分的信息保真度通常完美，而错误逐渐出现在文本末尾。这**与光学压缩中信息在整个文本上均匀模糊的降级方式形成对比，更类似于人类记忆衰减的过程。

<img width="773" height="611" alt="image" src="https://github.com/user-attachments/assets/3160d1d5-e54c-4b9d-9ae2-a06f9a062d5b" />

**结论与未来工作：**
C3通过提出一种更直接的纯文本到潜入路径，并采用级联双LLM架构，实现了高效长上下文压缩。实验证明其在各种压缩比下均显著优于现有的光学压缩方法，并在高压缩比下展示了卓越的信息保真度。C3的成功为LLM生态系统内的未来研究和实际应用开辟了多条途径，包括作为LLM的强大前端压缩器以处理超长上下文（如百万级token），实现多模态级联架构，以及作为下一代生成模型（如Diffusion Language Models和Latent Auto-regressive Models）的基础组件。

## 多轮迷失
LLMs Get Lost In Multi-Turn Conversation
https://arxiv.org/abs/2505.06120 2025.5.9 
https://mp.weixin.qq.com/s/mWqtMjgHtXqI_ho4qQcHOQ

## ToolOrchestra
ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration

NVIDIA 2025.11.26

1.  ⚙️ 针对大型语言模型在处理复杂任务时面临的效率和成本挑战，本文引入ToolOrchestra方法，通过训练一个小型编排模型来高效协调多种智能工具和模型。
2.  🚀 该方法通过强化学习端到端地训练这个8B参数的编排模型，其奖励设计综合考虑了任务结果的准确性、资源使用效率和用户工具偏好。
3.  💡 实验证明，其训练出的Orchestrator模型在HLE、𝜏2-Bench和FRAMES等基准测试上超越了GPT-5等前沿模型，以显著更低的成本实现了更高的准确率，并能稳健泛化到未见工具。

<img width="1007" height="306" alt="image" src="https://github.com/user-attachments/assets/c1e18111-1de8-493e-8918-bf88b9437167" />

本文介绍了ToolOrchestra，一种通过训练小型编排模型（orchestrator model）来高效协调多样化模型和工具，以解决复杂agentic任务的方法。尽管大型语言模型（LLMs）能力强大，但在处理如“人类终极考试”（Humanity's Last Exam, HLE）等深层次复杂问题时，仍面临概念性和计算成本高的挑战。ToolOrchestra旨在通过一个轻量级编排器来管理其他智能工具和模型，从而提高智能上限并提升效率。
<img width="1016" height="463" alt="image" src="https://github.com/user-attachments/assets/361f2569-a5a0-435e-bedb-a8dabdefac4c" />

**核心方法（ToolOrchestra）**
<img width="1006" height="360" alt="image" src="https://github.com/user-attachments/assets/b8d044f3-1664-4a54-ae9e-1a7aac2c7b91" />

ToolOrchestra通过强化学习（RL）端到端地训练一个小型语言模型（例如8B参数），使其作为异构工具使用agent的“大脑”，动态地选择和利用各种外部工具。
![Uploading image.png…]()

1.  **统一工具调用（Unified Tool Calling）**：
    与现有工具使用agent不同，ToolOrchestra扩展了工具集，不仅包含传统工具（如网页搜索、代码解释器），还包括领域专用模型（specialized LLMs）和通用大模型（generalist LLMs）。所有工具都通过一个统一的JSON接口暴露，包含工具名称、描述和类型化参数schema。当LLM作为工具使用时，其描述通过以下步骤自动生成：随机抽取10个训练任务，获取LLM完成这些任务的轨迹，然后由另一个LLM根据任务指令、LLM轨迹以及LLM是否完成任务来编写描述。

2.  **端到端Agentic强化学习（End-to-End Agentic Reinforcement Learning）**：
    将多轮工具使用agentic任务形式化为一个马尔可夫决策过程（MDP）$\mathcal{M} = (\mathcal{U}, \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{Z}, r, \rho, \gamma)$。Orchestrator通过迭代rollout生成解决方案，交替进行工具使用和环境反馈。每次交互遵循“思维链（chain-of-thought）-行动（tool call）-观察（tool response）”循环，直到任务解决或达到最大轮次。

    *   **奖励设计（Reward Design）**：为了平衡任务解决的正确性、资源使用效率和用户偏好，ToolOrchestra引入了三类奖励：
        1.  **结果奖励（Outcome Reward）**：对于每个rollout轨迹$\tau$，如果成功解决任务，则获得二元奖励$r_{outcome}(\tau) \in \{0, 1\}$。任务正确性由GPT-5作为裁判进行评估。
        2.  **效率奖励（Efficiency Reward）**：为鼓励高效解决方案，模型会因过度的计算或延迟受到惩罚，奖励表示为$r_{compute}(\tau) = -\$(\tau)$和$r_{latency}(\tau) = -\text{Clock}(\tau)$，其中$\$(\tau)$是轨迹的货币成本，$\text{Clock}(\tau)$是消耗的实际时间。为了统一计算开源模型和专有模型的成本，输入和输出token都转换为货币成本。
        3.  **偏好奖励（Preference Reward）**：鼓励模型在每一步选择工具时考虑用户偏好。对于一个轨迹$\tau$，构建一个向量$M^\tau = [m^\tau_{t_1}, m^\tau_{t_2}, \dots, m^\tau_{t_n}, r_{outcome}(\tau), r_{compute}(\tau), r_{latency}(\tau)]$，其中$m^\tau_{t_\cdot}$是工具$t_\cdot$被调用的次数。在RL训练中，$M^\tau$的每个元素在rollout批次T中进行归一化：$M^\tau_{normalized}[k] = (M^\tau[k] - M^{\text{T}}_{\text{min}}[k])/(M^{\text{T}}_{\text{max}}[k] - M^{\text{T}}_{\text{min}}[k])$。最终奖励计算为：
            $$R(\tau) = M^\tau_{normalized} \cdot P \quad \text{if } r_{outcome}(\tau)=1 \quad \text{else } 0$$
            其中$P = [p_{t_1}, p_{t_2}, \dots, p_{t_n}, p_{outcome}, p_{compute}, p_{latency}]$是用户偏好向量。

    *   **训练过程（Training Procedure）**：Orchestrator使用策略梯度强化学习算法Group Relative Policy Optimization (GRPO) [21]进行微调。对于批次中的每个任务，策略$\pi_\theta$生成一批轨迹T，每个轨迹$\tau \in \text{T}$获得一个标量奖励$R(\tau)$。GRPO在组内对奖励进行归一化以计算优势函数：
        $$A(\tau) = \frac{R(\tau) - \text{mean}_{\tau \in \text{T}} R(\tau)}{\text{std}_{\tau \in \text{T}} R(\tau)}$$
        策略通过最大化裁剪的代理目标函数进行更新：
        $$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \min \left( \text{ratio}_\theta(\tau)A(\tau), \text{clip}(\text{ratio}_\theta(\tau), 1-\epsilon, 1+\epsilon)A(\tau) \right) \right]$$
        其中$\text{ratio}_\theta(\tau) = \frac{\pi_\theta(\tau)}{\pi_{old}(\tau)}$是当前策略和旧策略的似然比。

    *   **训练技巧（Training Techniques）**：为了稳定RL训练，采用了同质性过滤（homogeneity filtering）、格式一致性过滤（format consistency filtering）和无效输出过滤（invalid output filtering）。

3.  **数据合成（Data Synthesis）- ToolScale**：
    由于可验证的agentic工具调用数据稀缺，ToolOrchestra开发了一个两步数据合成流程ToolScale：
    1.  **模拟丰富环境**：选择一个领域D，LLM生成数据库schema、主要关注点和数据库条目，并基于领域D提出常用工具。
    2.  **生成多样化任务**：LLM首先提出领域D中常见的多种意图，然后根据详细数据库信息将其转换为具体任务，包括任务指令I、黄金函数调用序列$A = a_1, \dots, a_l$以及任务解决过程中必须提及的简短信息o。为增加任务难度，利用额外的LLM添加复杂性。通过执行黄金函数调用、LLM解决任务通过率和是否需要实际操作来过滤数据，确保质量。

**实验结果**

在HLE、$\tau^2$-Bench和FRAMES三个挑战性基准测试中，ToolOrchestra训练的Orchestrator（8B模型）表现出卓越的性能和成本效率。

*   **性能优越性**：在HLE上，Orchestrator达到37.1%，超越GPT-5（35.1%）和Claude Opus 4.1（34.6%）。在$\tau^2$-Bench和FRAMES上，Orchestrator也以显著优势超越GPT-5。
*   **成本效率**：Orchestrator在HLE上比GPT-5效率高2.5倍，在$\tau^2$-Bench和FRAMES上仅使用约30%的成本，却取得了更高的性能。
*   **工具使用分析**：Orchestrator-8B学会了更策略性地协调工具，而不是过度调用强大的或昂贵的工具，展现出平衡的工具调用模式，避免了GPT-5和Claude Opus 4.1等模型中出现的偏见（如过度依赖自身变体或最强模型）。
*   **泛化能力**：Orchestrator对训练中未见的工具配置和定价策略展现出强大的泛化能力，能够理解新工具的描述并有效利用它们。
*   **用户偏好适应性**：Orchestrator-8B在测试时能更好地遵循用户偏好指令，表现出比其他强大多模型系统更优的偏好依从性。

**结论**

ToolOrchestra提供了一种有效训练小型编排模型的方法，使其能够统一和协调各种工具和专用模型。通过强化学习，Orchestrator学习了自适应的工具使用策略，平衡了任务结果质量、效率和人类偏好。该方法证明了由轻量级编排模型组合多样化工具比现有方法更高效和有效，为实用和可扩展的工具增强推理系统铺平了道路。


## Gated Attn for LLM
https://openreview.net/pdf?id=1b7whO4SfY ali Qwen, NIPS25
https://mp.weixin.qq.com/s/AYuUHghkLEQezMp5Oq1uGw 
**门控机制的有效性已经被广泛证实，但其在注意力机制中的有效性及扩展（scaling up）的能力**并未被充分讨论。论文系统性地分析了门控机制对大语言模型的有效性，并通过一系列控制实验**证实了门控机制的有效性来源于增强了注意力机制中的非线性与提供输入相关的稀疏性**。
此外团队还进一步发现了**门控机制能消除注意力池（Attention Sink）和巨量激活（Massive Activation）等现象，提高了模型的训练稳定性**，**极大程度减少了训练过程中的损失波动（loss spike）。**
得益于门控机制对注意力的精细控制，模型在**长度外推上相比基线得到了显著的提升**。团队在各个尺寸、架构、训练数据规模上验证了方法的有效性，并最终成功运用到了 Qwen3-Next 模型中。

1. 🕵️‍♀️ 本文系统性地研究了softmax attention中的门控机制，通过对超过30种门控变体的综合实验比较，发现将头特异性Sigmoid门控应用于Scaled Dot-Product Attention (SDPA) 输出能持续提升性能、增强训练稳定性并改善模型的可扩展性。
2. 💡 研究揭示，该门控机制的有效性主要归因于其在softmax attention的低秩映射中引入了非线性，并应用了查询依赖的稀疏门控分数来调节SDPA输出。
3. 🚀 这种稀疏门控设计还能有效缓解“massive activation”和“attention sink”现象，并显著增强模型在长上下文场景下的外推表现。

这篇论文深入研究了大型语言模型（LLMs）中门控机制（Gating Mechanisms）在softmax注意力机制中的具体作用和效果。尽管门控机制在从LSTM到现代状态空间模型和注意力机制等多种架构中被广泛采用，但对其特定效果的系统性探究却相对缺乏。

**核心发现与贡献：**
作者通过在3.5万亿（3.5T）tokens数据集上训练15B MoE模型和1.7B密集模型，对超过30种门控注意力变体进行了全面对比实验。研究核心发现是一个简单但有效的改进：在Scaled Dot-Product Attention (SDPA) 输出后应用一个头部特异性（head-specific）的sigmoid门控，能够持续提升模型性能，增强训练稳定性，允许使用更大的学习率，并改善模型扩展性。

**方法论与技术细节：**
论文在标准softmax注意力机制中系统性地探索了门控机制的五大方面：
1.  **门控位置 (Positions)**：如图1所示，作者在自注意力层的不同阶段引入门控：
    *   G1：在SDPA输出之后。
    *   G2：在Value投影 $V$ 之后。
    *   G3：在Key投影 $K$ 之后。
    *   G4：在Query投影 $Q$ 之后。
    *   G5：在最终的密集输出层 $O$ 之后。
    核心结论是G1位置的门控效果最佳。
2.  **粒度 (Granularity)**：
    *   **Headwise (头部级别)**：一个单一的标量门控分数调制整个注意力头的输出。
    *   **Elementwise (元素级别)**：门控分数是与输入 $Y$ 维度相同的向量，实现更细粒度的逐维度调制。
3.  **头部特异性或共享 (Head Specific or Shared)**：
    *   **Head-Specific (头部特异性)**：每个注意力头都有其独立的门控分数。
    *   **Head-Shared (头部共享)**：门控参数 $W_\theta$ 和门控分数在所有注意力头之间共享。研究表明，头部特异性门控至关重要。
4.  **乘法或加法 (Multiplicative or Additive)**：
    *   **Multiplicative Gating (乘法门控)**：$Y' = Y \odot \sigma(XW_\theta)$。
    *   **Additive Gating (加法门控)**：$Y' = Y + \sigma(XW_\theta)$。实验发现乘法门控效果更好。
5.  **激活函数 (Activation Function)**：主要考察SiLU和sigmoid。默认和最优选择是sigmoid激活函数。

门控机制的通用形式定义为：
$$ Y' = g(Y, X, W_\theta, \sigma) = Y \odot \sigma(XW_\theta) $$
其中 $Y$ 是被调制的输入，$X$ 是用于计算门控分数的另一输入（通常是预归一化后的隐藏状态），$W_\theta$ 是门控的可学习参数，$\sigma$ 是激活函数（如sigmoid），$Y'$ 是门控后的输出。

**实验结果总结：**
*   **MoE模型**：SDPA输出（G1）和Value输出（G2）门控最有效，其中G1效果最佳，PPL可降低0.2，MMLU提升2点。头部特异性门控显著优于头部共享门控。乘法门控优于加法门控，sigmoid激活函数优于SiLU。
*   **密集模型**：门控在不同模型配置、训练数据量和超参数下均能持续带来收益。它显著提升了训练稳定性，几乎消除了训练过程中的损失尖峰（loss spikes），从而支持使用更大的学习率和批处理大小，促进了模型的可扩展性。

**分析与机制解释：**
论文将门控的有效性归因于两个关键因素：
1.  **引入非线性 (Non-Linearity)**：
    标准softmax注意力中，Value投影 $W_V$ 和最终输出层 $W_O$ 连续的线性变换可以合并为一个低秩线性投影。在G1或G2位置引入非线性门控，能有效增加这种低秩线性变换的表达能力。例如，G2位置的门控对应于 $\text{Non-Linearity-Map}(X_j W_V)$，而G1位置的门控对应于 $\text{Non-Linearity-Map}(\sum_{j=0}^{i} S_{ij}^k \cdot X_j W_V)$。引入RMSNorm或简单的SiLU激活函数也能带来性能提升，进一步支持了非线性假设。
2.  **引入输入依赖的稀疏性 (Input-Dependent Sparsity)**：
    有效的门控分数（尤其是在SDPA输出处的门控）表现出强烈的稀疏性，即大量门控分数接近于0。这种查询依赖的稀疏性对SDPA输出进行调制，过滤掉与当前查询token不相关的上下文信息。
    *   **减少“大规模激活”（Massive Activation）和“注意力汇聚”（Attention Sink）**：研究发现，头部特异性的查询依赖型SDPA输出门控能够显著降低分配给第一个token的注意力分数（“注意力汇聚”现象），并减少模型隐藏状态中的“大规模激活”。这种稀疏性通过降低激活值，可能提高了BF16训练的数值稳定性，从而增强了训练稳定性。
    *   **促进长上下文外推性能**：无注意力汇聚模式使模型在长上下文场景下表现更优。在上下文长度扩展至128k时，门控模型相比基线模型表现出显著优势。这表明门控可能帮助模型更好地适应上下文长度扩展，因为基线模型可能依赖注意力汇聚来调整注意力分数分布，而门控模型则依赖输入依赖的门控分数控制信息流，使其对RoPE等位置编码的修改更具鲁棒性。

**实际建议：**
为获得最佳效果，建议在SDPA输出（G1）后应用元素级别（elementwise）门控，并使用适度增加的学习率进行训练。这一机制已应用于Qwen3-Next模型中。

## 长序列推理压缩技术评测
easoning-Focused Evaluation of Efficient Long-Context Inference Techniques
https://openreview.net/pdf?id=uCOMb0EsPq  普林斯顿 chendaiqi等 NIPS25

1.  📝 本研究系统评估了多种高效长上下文推理技术，包括权重量子化（NF4、Int8）和KV缓存token逐出方法，发现在内存和性能的帕累托最优性方面，量子化技术显著优于基线和token逐出方法。
2.  😮 实验表明，现有高效推理技术在处理需要长输出和高信息分散的任务（对推理至关重要）时表现不佳，特别是token逐出方法因倾向于移除关键token而难以可靠地执行精确字符串检索。
3.  💡 研究进一步发现，使用推理模型可以部分缓解这些性能下降，即使在较小的缓存尺寸下也能使性能接近完整缓存基线，从而提升了内存效率和长上下文性能的帕累托前沿。
   
**核心方法论**

1.  **评估任务维度扩展**：
    *   现有评估主要关注输入上下文长度。本文在此基础上增加了两个关键维度：**信息分散度 (information dispersion)** 和 **输出生成长度 (output generation length)**。
    *   任务被分为三类：
        *   **Easy (Short Output, Low Dispersion)**：例如 NIAH (从文章中检索数字)、JSON KV (在JSON字典中检索键)，主要测试召回能力。
        *   **Medium (Short Output, High Dispersion)**：例如 HotpotQA (多跳问答)、Natural Questions (事实性问答)、ASQA Cite (引用生成)、MS MARCO (文档重排序)、BANKING77/CLINC150 (上下文学习)、Multi-LexSum (多文档摘要)，测试信息检索、引用和复杂上下文理解。
        *   **Hard (Long Output, High Dispersion)**：例如 HTML-to-TSV (从HTML提取信息到TSV)、Pseudocode-to-Code (伪代码转C++代码)、Travel Planning (基于约束生成旅行计划)，这些任务需要分散检索、程序生成和长程推理。
    *   共使用9个HELMET任务（16K和32K上下文长度）和3个LongProc任务（0.5K和2K输出长度）。

2.  **高效推理技术选择**：
    *   **输入无关 (Input-independent) 方法**：权重量化。
        *   **NF4 (4-bit NormalFloat)**：一种信息论最优的量化数据类型，适用于正态分布的权重。
        *   **Int8 (8-bit Integer)**：标准的8位整数权重。
    *   **输入依赖 (Input-dependent) 方法**：KV cache 压缩/token 驱逐。
        *   **DuoAttention**：当前最先进的token驱逐方法。
        *   **SnapKV**：基于注意力机制的KV cache压缩。
        *   **PyramidKV**：基于金字塔信息漏斗的动态KV cache压缩。
        *   **StreamingLLM**：通过注意力汇聚 (attention sinks) 实现高效流式处理。

3.  **模型选择**：
    *   参数规模为8B，以在学术计算资源限制下研究小型推理模型。
    *   **非推理Instruction模型**：Llama-3.1-8B-Instruct 和 Qwen2.5-7B-Instruct (上下文长度128K)。
    *   **推理模型**：DeepSeek-R1-Distill-Llama-8B (从Llama-3.1-8B蒸馏) 和 DeepSeek-R1-Distill-Qwen-7B (从Qwen2.5-Math-7B蒸馏)，旨在隔离推理能力对内存和性能的影响。

**实验结果**

1.  **整体性能与内存效率的Pareto最优性**：
    *   **NF4** 和 **Int8 权重量化** 在内存和性能的Pareto最优性上显著优于基线模型和KV cache token驱逐方法。
        *   NF4 实现了$-48.46\% \pm 0.91\%$ 的内存节省，性能下降仅为$-2.46\% \pm 5.99\%$。
        *   Int8 实现了$-23.49\% \pm 3.73\%$ 的内存节省，性能下降为$-4.08\% \pm 6.13\%$。
    *   **DuoAttention** 接近基线性能 ($-0.71\% \pm 2.82\%$)，但内存节省微乎其微 ($-1.92\% \pm 0.04\%$)。
    *   **Token驱逐方法 (SnapKV, PyramidKV, StreamingLLM)** 在多种cache配置下，平均表现出显著的内存开销 ($+29.53\% \pm 12.08\%$) 和严重的性能下降 ($-22.85\% \pm 13.80\%$)。小cache (w256, c2048) 导致性能严重下降，而大cache (w2048, c8192) 则大幅增加内存消耗。因此，token驱逐方法未能达到Pareto最优。

    *   基线模型随着任务难度增加 (从Easy到Hard) 性能自然下降。
    *   **NF4** 和 **Int8** 表现出极小的性能下降，在所有难度级别上保持接近基线的性能。
    *   **DuoAttention** 作为一个例外，在所有难度级别上都实现了性能提升 ($+10.10\% \pm 4.74\%$)，尤其在Easy任务上表现出色 ($+18.6\%$)。
    *   **Token驱逐方法** 平均表现出显著的性能下降 ($-25.62\% \pm 3.56\%$)，且随着难度增加，其内部性能下降更加剧烈。

3.  **任务级故障模式分析**：
    *   **Token驱逐方法** 在需要精确字符串检索的任务 (如 NIAH, Recall, Re-rank) 上表现出最大的性能下降。这表明它们可能在解码过程中丢弃了关键token，导致无法可靠地执行精确检索。例如，在Re-rank任务中，模型输出的文档ID从7位变为3位。

4.  **任务间相关性**：
    *   低分散性短输出任务 (NIAH, Recall, RAG) 与高分散性短输出任务 (ICL, Cite, Re-rank, Summ) 之间存在一定相关性。
    *   然而，低分散性任务与长输出高分散性任务 (Pseudocode to Code, HTML to TSV, Travel Planning) 之间没有强相关性。这强调了评估时同时考虑输出长度和信息分散度的重要性。

5.  **案例研究：Re-rank任务中Token驱逐的失败**：
    *   SnapKV在Re-rank任务中表现出多种失败模式：生成顺序数字、重复ID、异常短ID或被单个ID主导的输出。这表明关键token被驱逐，模型内部的文档表示被破坏，无法正确进行排序。

6.  **案例研究：推理模型缓解Token驱逐的性能下降**：
    *   在In-Context Learning (ICL) 任务 (BANKING77, CLINC150) 上，**推理模型 DeepSeek-R1-Distill-Llama-8B** 的基线性能优于 Llama-3.1-8B-Instruct。
    *   **推理模型对Token驱逐技术表现出更强的鲁棒性**。在激进的token驱逐设置下 (SnapKV/PyramidKV)，Llama-Instruct 的性能下降幅度远大于 R1-Distill-Llama。推理模型性能下降的幅度比instruct模型平均少5.9个百分点。
    *   这表明推理模型可能以更冗余的方式组织上下文信息，使其对token丢失更具抵抗力，从而在较小cache尺寸下也能实现接近全性能，推动了Pareto前沿。


## Seer
Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning

https://arxiv.org/abs/2511.14617 Moonshot AI, 2025.11.18

1.  ⚙️ 同步LLM强化学习（RL）的rollout阶段因长尾延迟和资源利用率低下而面临严重的性能瓶颈，主要源于工作负载不均衡和长生成需求。
2.  🚀 Seer系统提出了一种新颖的在线上下文学习方法，通过利用共享相同prompt的请求在输出长度和生成模式上的相似性，引入了划分式rollout进行动态负载均衡、上下文感知调度和自适应分组推测解码。
3.  📈 实验证明，Seer相比现有同步RL系统，能将端到端rollout吞吐量提高74%至97%，并将长尾延迟减少75%至93%，显著加速了RL训练迭代。

Seer是一项新颖的在线上下文学习系统，旨在加速大型语言模型（LLM）强化学习（RL）的同步Rollout阶段。现有同步RL系统面临严重的性能瓶颈，其中Rollout阶段占据了迭代总时间的80%以上。这些瓶颈主要源于长时间生成任务（例如Chain-of-Thought推理）导致的工作负载不平衡：请求的输出长度和KVCache内存占用具有高度不可预测性，从而导致硬件利用率低下、频繁的抢占以及严重的“长尾”延迟问题。虽然异步RL可以缓解部分问题，但其以牺牲算法保真度为代价，引入了off-policy学习和数据分布倾斜。Seer通过利用具有相同prompt的请求之间（如GRPO算法中为每个prompt生成G个响应）输出长度和生成模式上的相似性，解决了这些挑战。

Seer的核心贡献在于其引入的三项关键技术：
1.  **Divided Rollout with Global KVCache（分段Rollout与全局KVCache）**：
    传统的Rollout系统将整个请求组（即共享相同prompt的请求集合）作为单一单元进行调度，导致实例间和实例内的负载严重不平衡。Seer打破了这一限制，将每个请求组不仅分解为G个独立的请求，还进一步将每个请求细分为多个更小的“chunk”进行增量调度和分派。这种细粒度调度由Request Buffer管理，每个chunk完成后，请求会被重新排队并迭代提交，直到生成`<eos>`标记或达到其原始`max_tokens`。
    *   **精细内存管理（Fine-grained Memory Management）**：通过将请求分解为小chunk，每个请求在生成过程中保持相对恒定的KVCache占用。Seer的调度器可以动态计算最大并发级别以避免抢占，从而最大化资源利用率。
    *   **动态负载均衡（Dynamic Load Balancing）**：分段Rollout将调度粒度从每个请求组一次实例选择扩展到$G \times \text{num\_chunks}$次选择。当一个子请求从Request Buffer重新提交时，Seer会根据实时监控的并发请求数和内存占用动态选择负载最小的推理实例。这显著减少了由实例间负载不平衡引起的长期延迟。
    为了支持高效的分段Rollout并避免重新分派时昂贵的KVCache重复计算，Seer基于Mooncake构建了一个跨推理节点分布的全局共享KVCache池，实现了KVCache的按需迁移和复用。

2.  **Context-Aware Scheduling（上下文感知调度）**：
    Seer利用请求组内的长度上下文来预测输出长度，并实现近似的“最长作业优先”（Longest-Job-First, LFS）调度策略。
    *   **推测请求（Speculative Request）机制**：Seer将每个组的第一个请求指定为“推测请求”，赋予其更高的调度优先级，并采用“最短作业优先”（Shortest-First Scheduling, SFS）策略进行调度。推测请求作为在线探测，用于估计该组剩余请求的预期工作负载（如生成长度和KVCache占用）。这种长度过滤方法能迅速完成短请求，并尽早识别潜在的长尾候选。
    *   **长度估计更新（Length Estimation Update）**：Context Manager为每个prompt组$g$维护一个估计输出长度$\hat{L}_g$，该值会随着组内已完成请求的最大生成长度动态更新。对于尚未完成任何请求的组，其估计长度保守地设置为原始`max_tokens`限制。
    *   **调度策略**：调度器优先执行推测请求以获取长度估计。一旦所有推测请求处于in-flight或已完成状态，Seer会切换到近似LFS策略，优先调度具有更长预测生成长度$\hat{L}_g$的组。
    算法伪代码如下：
    ```
    Algorithm 1 Context-Aware Scheduling based on Divided Rollout
    Require: Active requests R = {rg,i} grouped by prompt g; group-level length estimates {bLg}; inference instances I with KV-usage telemetry.
    Ensure: A scheduling decision (r⋆, i⋆) with r⋆ ∈ R and i⋆ ∈ I .
    1: for all rg,i ∈ R do
    2:    if rg,i is finished then
    3:        bLg ← UPDATEESTIMATE(bLg, Lg,i)
    4:        remove rg,i from R
    5:    else if rg,i is the group’s speculative request then
    6:        keep in high-priority queue Qspec
    7:    else
    8:        add to low-priority candidate set Crest
    9:    end if
    10: end for
    11: r⋆ ← None
    12: if ¬ISEMPTY(Qspec) then
    13:    r⋆ ← PICKSFS(Qspec) ▷ SFS: smallest generated length first
    14: else if ¬ISEMPTY(Crest) then
    15:    r⋆ ← PICKLFS(Crest) ▷ LFS: largest bLg first
    16: else
    17:    return all requests are finished
    18: end if
    19: r⋆.max_tokens ← min(chunk_size, r⋆.ori_max_tokens − r⋆.generated_tokens)
    20: i⋆ ← SELECTINSTANCE(I , r⋆.max_chunk_tokens, KV-usage)
    21: if i⋆ = None then
    22:    return (r⋆, i⋆)
    23: end if
    24: return no available instance for this cycle
    ```
    这种调度策略利用全局KVCache池的灵活性，使得请求可以在chunk执行之间暂时存储在Request Buffer中，等待基于持续更新的长度上下文进行调度。

3.  **Adaptive Grouped Speculative Decoding（自适应分组推测解码）**：
    为进一步加速Rollout阶段，特别是长尾阶段，Seer引入了自适应分组推测解码。该技术利用分组模式上下文（同一组内请求的生成内容相似性）来提高推测解码的接受率，并根据计算强度自适应调整推测长度。
    *   **分布式分组Draft服务器（Distributed Grouped Draft Server, DGDS）**：DGDS是一个分布式框架，用于跨请求和实例共享推测上下文。其核心数据结构是压缩后缀树（Compressed Suffix Tree, CST），它能高效聚合来自多个序列的上下文统计信息，并以$O(p+s)$的复杂度提供Draft tokens（$p$为匹配模式长度，$s$为推测token数量）。DGDS将同一组内所有请求的序列上下文聚合到一个统一的CST中，为所有推理实例提供高质量的推测tokens。
    *   **工作流程**：
        1.  **异步追加（Asynchronous Append）**：每个推理实例独立处理输出tokens，并将新生成的tokens连同`group_id`发送给DGDS。
        2.  **全局聚合（Global Aggregation）**：DGDS聚合属于同一组的token更新，通过`request_id`隔离更新，确保每个新token仅映射到CST中对应的本地路径。
        3.  **周期性获取（Periodic Fetch）**：每个推理实例内嵌一个Draft客户端库，定期从DGDS同步最新的CST。为减少通信开销，客户端只获取其当前正在处理的请求对应的CST，并支持增量同步。
        4.  **本地推测（Local Speculation）**：推理实例基于其本地CST执行推测，这些CST聚合了同一组内所有请求的路径，从而共享上下文统计信息并获得更高质量的Draft tokens。
    *   **自适应推测范围机制（Adaptive Speculation Range Mechanism）**：DGDS动态调整`max_draft_length`和候选路径数$k$。系统根据当前并发级别、模型架构（如dense或MoE）预先计算的推测token阈值，动态计算最大Draft长度。生成Draft tokens时，还会根据token出现频率和CST提供的置信度分数过滤候选，以提高接受率。这种自适应控制能够在保持解码准确性的同时，最大化分布式Rollout实例的整体吞吐量和推测解码效率。

**实验结果**：

<img width="2263" height="859" alt="image" src="https://github.com/user-attachments/assets/3ccda96b-3de1-4377-a887-2bc392a300fe" />

Seer在生产级RL工作负载上进行了评估，结果表明，与最先进的同步RL系统相比，Seer的端到端Rollout吞吐量提高了74%至97%，长尾延迟降低了75%至93%，显著加速了RL训练迭代。消融实验也验证了分段Rollout、上下文感知调度和自适应分组推测解码各自的有效性。例如，上下文感知调度实现了接近Oracle LFS调度器95%的吞吐性能，而自适应分组推测解码相比无SD基线提高了30%的端到端吞吐量。

## Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation

https://arxiv.org/abs/2511.11500 

2024.11.14 丰田研究院（芝加哥），UCSD等

1. LLMs包括GPT等SOTA模型在内即使在不确定或错误答案后果严重时也极少拒绝回答。这源于现有训练范式（如RLVR）奖励任何答案而非沉默，导致它们无法建立可信赖智能所需的基本犹豫能力。
2. 为此提出“强化犹豫”（Reinforced Hesitation, RH）机制，通过将强化学习中可验证奖励（RLVR）的二元奖励（对错）扩展为三元奖励（正确+1，拒绝回答0，错误-λ），在训练阶段明确赋予犹豫以价值。
3. Qwen3-1.7b模型RH训练形成了一个帕累托前沿，在不同风险偏好下表现最佳，并且其学会的“拒绝回答”信号可用于级联推理（cascading）和自级联（self-cascading）等高效推理策略，显著提升了模型在准确性和信任度上的表现。3种正面效果：1）拒绝回答率显著提升；2）回答更精简；3）通过自级联（提高多样性）设计的多种采样，提高了准确率（从不做拒绝回答的基线 85% -> 92%）

**核心问题与现有范式缺陷**
LLMs在医学诊断、金融咨询、法律研究等高风险领域中的应用日益广泛，但其在这些场景中犯错的成本是非线性的，一个自信的错误可能会永久性地损害用户信任。然而，当前的评估标准仍主要关注准确率最大化，对错误的类型和后果不加区分，这导致模型在追求排行榜名次的同时，丧失了“知之谓知之 不知谓不知”这一可信智能的基本要求。特别是RLVR范式，其二进制奖励机制（答对+1，答错0）鼓励模型进行猜测，即使推理过程是虚假的，只要答案正确也会得到奖励，从而忽略了灾难性风险。

作者首先通过实验评估了当前前沿模型（如GPT-4o、Gemini 2.5 Pro等）在GSM8K、MedQA和GPQA等基准测试上的表现。即使在提示中明确告知了严重的错误惩罚和拒绝回答选项，这些模型也几乎从不选择拒绝回答（通常低于1%），且错误率依然很高（超过10%）。这表明，仅仅依靠提示词无法纠正模型在数千次训练步骤中形成的“有回答胜于无回答”的内在偏好，模型缺乏有效的拒绝回答能力并非能力缺失，而是训练中形成的固有行为。

提出的解决方案：Reinforced Hesitation (RH)
为了解决这一根本性问题，本文提出了Reinforced Hesitation (RH)，这是对RLVR奖励机制的一个最小化修改。RH将RLVR的二元奖励信号（+1，0）转换为三元结构（+1，0，-$\lambda$），分别对应正确答案、拒绝回答和错误答案。惩罚参数 $\lambda \geq 0$ 编码了领域特定的错误后果和验证成本：
$\text{reward} = \begin{cases} +1 & \text{if the answer is correct,} \\ 0 & \text{if the model says ‘I don’t know’,} \\ -\lambda & \text{If the answer is wrong.} \end{cases}$
在一个理性Agent的视角下，当回答的预期效用低于零时，Agent会选择拒绝回答，这在置信度阈值 $\frac{\lambda}{1+\lambda}$ 处形成了一个自然的决策边界。因此，$\lambda$ 不仅仅是一个超参数，它是一个可解释的领域控制参数，用于权衡错误与拒绝回答：例如，在错误代价巨大的医学诊断中，可设置 $\lambda=100$（要求 $>99\%$ 的置信度）；在错误可容忍的创意任务中，可设置 $\lambda=1$（要求 $>50\%$ 的置信度）。
核心方法论（技术细节）
RH作为LLM后训练标准RLVR阶段的修改，无需对模型架构或早期训练阶段进行改动。总奖励 $R_{\text{total}}(y, y^*)$ 被分解为内容奖励 $R_{\text{content}}(y, y^*)$ 和格式奖励 $R_{\text{format}}(y)$：
$R_{\text{total}}(y, y^*) = R_{\text{content}}(y, y^*) + R_{\text{format}}(y)$
其中，内容奖励 $R_{\text{content}}(y, y^*)$ 即上述三元奖励结构：
$R_{\text{content}}(y, y^*) = \begin{cases} +1 & \text{if } y = y^* \text{ (correct answer)} \\ 0 & \text{if } y = \text{“I don’t know”} \\ -\lambda & \text{if } y \neq y^* \text{ (incorrect answer)} \end{cases}$
格式惩罚 $R_{\text{format}}(y)$ 用于确保输出结构正确，例如：
$R_{\text{format}}(y) = \begin{cases} 0 & \text{if format is valid} \\ -0.5\lambda & \text{if format is violated (missing tags, truncation, etc.)} \end{cases}$
这种分解可以防止模型通过破坏输出格式来规避惩罚。RH的训练过程（算法1）是在标准的RLVR框架内，通过在提示中明确允许模型拒绝回答（例如：“If you don’t know the answer with sufficient confidence, you must say ‘I don’t know’.”），并用三元奖励计算替代二元奖励。
实验设计与结果
作者使用Qwen3-1.7B模型，在Knights & Knaves逻辑谜题数据集上进行了一系列受控RLVR实验，数据集包含5、6、7人谜题，并根据逻辑复杂性分为简单和困难两类。实验通过改变 $\lambda \in \{0, 1, 2, 5, 10, 20\}$ 来观察模型行为。

实验结果表明，不同 $\lambda$ 值训练出的模型展现出截然不同的行为模式：
1. 激进回答者 ($\lambda=0$)：模型总是回答，错误率保持在15%左右，几乎不拒绝回答。
2. 平衡型 ($\lambda \in \{1, 2, 5\}$)：模型学会权衡覆盖率与安全性，错误率降至2%以下。它们能进行校准式拒绝回答，对简单问题拒绝率仅5-10%，而对复杂问题拒绝率高达60-95%。
3. 保守拒绝者 ($\lambda \ge 10$)：模型学会通过保守拒绝回答来最大化预期奖励，错误率低于1%（$\lambda=20$ 时几乎普遍拒绝回答）。

训练动态显示，当 $\lambda=10$ 时，模型经历了“瞬时危机”：拒绝回答率一度飙升至90%（在简单问题上更是达到97%），随后恢复到稳定的状态，这表明模型不是丧失了能力，而是在学习新的决策边界。所有 $\lambda>0$ 的模型都表现出难度区分能力，对难题的拒绝率显著高于简单题。此外，高 $\lambda$ 值还会促使模型生成更简洁的响应，因为超出最大Token长度会受到惩罚，从而意外地提高了推理效率。

模型表示“我不知道”时，为什么再次提问可能会有所帮助？答案在于 LLM 推理的本质。每次生成都涉及两种形式的非确定性：算法上的（影响 token 选择的抽样策略，如 temperature 和 top-p）和计算上的（硬件级别的数值不稳定性，在自回归步骤中以不同的方式累积）。对于模型学会弃权的问题，这些变化有时会产生不同的结果。一个导致弃权的推理链，由于生成过程中早期不同的随机选择，可能会发展出提供答案的信心。 自我级联利用了这一点，将每次弃权视为探索解决方案空间中不同轨迹的机会，而不是永久性的失败。

## MoC
Mixture-of-Channels: Exploiting Sparse FFNs for Efficient LLMs Pre-Training and Inference

https://arxiv.org/abs/2511.09323v1 北大yuan kun 2025.11.12

https://mp.weixin.qq.com/s/IHeGDK4SNP3C__L1wDjL7g

降低激活内存和计算量：利用SwiLU激活接近稀疏 选重要的通道（50%）理论内存降低4x（实测E2E 25%）；并借助A100的硬件稀疏（2:8??），FFN层加速38%~56%。最大1B的模型。精度号称持平。

1. 👉 大型语言模型（LLMs）训练面临巨大的内存开销，特别是在FlashAttention优化注意力机制后，前馈网络（FFNs）的激活内存已成为预训练和推理的关键瓶颈。
2. 💡 本文提出了一种新颖的FFN架构，Mixture-of-Channels (MoC)，它利用SwiGLU固有的门控机制，为每个token选择性地激活Top-K个最相关的通道。
3. 🚀 MoC显著减少了预训练期间的激活内存，并通过仅加载所需权重到GPU SRAM来提高推理效率，实验证明其在保持模型性能的同时，实现了显著的内存节省和吞吐量提升。

## Virtual Width Networks

http://arxiv.org/abs/2511.11238v1 2025.11.17 字节

https://mp.weixin.qq.com/s/sYflxeZSInStAvTdxaow6g 

1. 💡 本文提出了Virtual Width Networks (VWN)框架，它通过解耦表示宽度和骨干网络宽度，能够在不增加隐藏层二次计算成本的情况下扩展嵌入空间，主要通过Generalized Hyper-Connections (GHC)实现。
2. 🚀 大规模实验显示，VWN的8倍虚拟宽度扩展可将next-token预测的优化速度提升超过2倍，next-2-token预测提升3倍，且这种效率优势会随训练的进行而放大，并与Multi-Token Prediction (MTP)展现出协同效应。
3. 📈 研究还发现虚拟宽度与损失降低之间存在近似对数线性关系，这为将虚拟宽度扩展作为一个新的可预测维度来提升大型模型效率提供了重要的经验基础和探索方向。

## Optimizing Mixture of Block Attention

https://arxiv.org/abs/2511.11571 2025.11.14, MIT 韩松团队

https://github.com/mit-han-lab/flash-moba

1. ✨ 本文针对Mixture of Block Attention (MoBA)在长上下文LLM中的应用，提出了一个统计模型，并通过信噪比（SNR ∝ √(d/B)）揭示了路由准确性与架构参数（d/B比值、关键键卷积）的关键联系，从而提供了MoBA设计的理论指导。提高信噪比关键: 更大的dimension；更小的Block块
2. 🌟 针对理论上最优但GPU效率低下的小block问题，开发了硬件感知的FlashMoBA CUDA内核，通过融合tiled Top-K和“gather-and-densify”(固定Key，gather Q)策略，显著提升了MoBA的执行效率。
3. 🚀 1B模型，精度可匹敌或略超dense，>=128K时 attn加速10~14x

MoBA 机制回顾
MoBA 通过将 Key 和 Value 分割成大小为$B$的块，并通过一个学习到的路由器让每个 Query 仅稀疏地关注一小部分 Key-Value 块，从而将计算复杂度从 $O(N^2)$ 降低到近乎线性的 $O(N \cdot kB)$。具体而言，对于每个 Query $q$，MoBA 会计算其与每个 Key 块质心（centroid）$\tilde{k}_i = \frac{1}{B}\sum_{k \in K_i} k$ 的相似度 $s_i = q^\top \cdot \tilde{k}_i$。随后，选择得分最高的 $k$ 个块进行注意力计算。为了保持因果性，对未来块进行掩码，并且每个 Query 总是关注其自身所在的块。
这篇论文深入探讨了 Mixture of Block Attention (MoBA) 这一高效处理长上下文的 LLM 机制，旨在解决其性能机制理解不足和 GPU 实现效率低下的两大核心问题。论文首先通过建立一个统计模型来分析 MoBA 的底层工作原理，揭示了其性能关键在于路由器准确区分相关与非相关块的能力。在此基础上，论文提出了 FlashMoBA，一个硬件感知的 CUDA kernel，使得理论上最优的小块尺寸在实践中变得高效可行。
MoBA 的统计模型与信号-噪声比 (SNR) 分析
论文构建了一个统计模型来理解 MoBA 路由器如何成功选择正确块的挑战。
1. 块选择挑战建模：路由器通过块的质心来评分，这可能导致单个相关 Token 的信号被淹没。论文将 Query $q$ 与 Key 之间的点积视为随机变量，假设“信号”Key $k^*$ 的期望点积 $\mu_{signal} = E[q^\top k^*]$ 高于“噪声”Key $k$ 的期望点积 $\mu_{noise} = E[q^\top k]$。两者的基本分离是 $\Delta\mu = \mu_{signal} - \mu_{noise}$。
2. SNR 分析：为量化块选择的成功条件，论文分析了路由器得分的信号-噪声比 (SNR)。考虑信号块 $j^*$ 和纯噪声块 $j$ 之间得分差异 $D = s_{j^*} - s_j$。此差异的期望值代表“信号”，其标准差代表“噪声”。
通过统计分析（详见附录 A），得到：
  ○ 期望差异：$E[D] = \frac{\Delta\mu_{eff}}{B}$
其中，$\Delta\mu_{eff} = \Delta\mu + (m-1)(\mu_{cluster} - \mu_{noise})$ 是有效信号分离，如果目标块中聚集了 $m$ 个相关信号 Token，则该分离被放大。
  ○ 差异方差：$Var(D) \approx \frac{2}{dB}$ (对于归一化向量，假设点积方差 $\sigma^2 \approx 1/d$)
由此，核心发现是 SNR：
$\text{SNR} = \frac{E[D]}{\sqrt{Var(D)}} = \Delta\mu_{eff} \sqrt{\frac{d}{2B}}$
检索失败（噪声块排名高于信号块）的概率 $p_{fail} = \Phi(-\text{SNR})$ 随 SNR 增加呈指数下降。
3. 架构洞察：SNR 公式提供了两个关键设计原则：
  ○ $d/B$ 比是关键：SNR 与 $\sqrt{d/B}$ 成正比。这意味着增加头维度 $d$ 或减小块大小 $B$ 都能提高路由器的检索能力。为了控制模型容量，论文在实验中固定 $d$ 并系统性地改变 $B$。
  ○ 块内聚类是性能倍增器：当语义相关的 Token 在块内聚集时（可以通过对 Key 进行 Token 级别的卷积来促进），有效信号 $\Delta\mu_{eff}$ 会通过更大的 $m$ 和 $\mu_{cluster}$ 而增加，从而显著提高 SNR。
FlashMoBA：小块 MoBA 的优化核
尽管理论上小块尺寸能带来质量提升，但朴素的 GPU 实现效率低下。原始 MoBA 在小块配置下，其性能瓶颈会抵消稀疏性带来的计算节省。FlashMoBA 旨在解决这些挑战，使理论最优配置在实践中可行。
1. 小块尺寸带来的性能挑战：
  ○ 内存访问效率低下：为每个 Query 收集稀疏、非连续的 Key-Value 块导致 HBM 非 coalesced 内存读取。
  ○ Top-K 和门控开销：小块尺寸 $B$ 增加路由器需要评分的块数量 $n=N/B$，原始实现会具体化一个大的$N \times n$分数矩阵，造成巨大的内存开销。
  ○ GPU 占用率低：每个块的工作量减少以及启动大量独立核的开销导致并行度差和硬件利用率低。
2. FlashMoBA 核设计：FlashMoBA 采用三个融合核（fused kernels），以最小化 HBM 往返次数并与 GPU 架构对齐：
  ○ Tiled Top-K Selection (Flash TopK)：
    ⅰ. 计算块质心：一个融合的 Triton 核首先计算 Key-块质心，生成一个比原始 Key 矩阵小$B$倍的 $\tilde{K}$ 矩阵。
    ⅱ. 融合 Top-K 选择：第二个融合核采用 FlashAttention-2 的瓦片（tiling）策略，识别每个 Query 的 Top-K 块，通过计算 $Q$ 和 $\tilde{K}$ 之间的分数，而无需将完整的得分矩阵具体化到 HBM。它在片上维护 Top-K 索引和分数列表，使用冒泡排序进行高效更新。
    ⅲ. 索引重格式化为 Varlen 格式：为了后续注意力计算的密集化，将以 Query 为中心的 Top-K 索引重格式化为以 Key-块为中心的变长 (varlen) 布局。这通过两阶段的 epilogue 实现：首先通过前缀和计算每个 Key-块的内存偏移，然后将 Query-centric 索引散射到 Varlen 数组中。
  ○ 带有 Gather-and-Densify 的前向传播：
核心策略是“gather-and-densify”，允许在稀疏上下文中利用 FlashAttention-2 的高效密集计算模式。区分两种块：
    ■ 逻辑块 (Logical Blocks)：Query ($Q_i$) 和 Key ($K_j$) 的大型连续块，核在外层循环中遍历。逻辑 Key 块等同于 MoBA Key 块。
    ■ 物理块 (Physical Blocks)：加载到 SRAM 中进行实际矩阵乘法的小瓦片（例如 64x64 或 128x128）。
核将一个逻辑 Query 块分配给一个线程块，并遍历所有逻辑 Key 块。对于每对 $(Q_i, K_j)$，它使用 varlen 索引识别 $Q_i$ 中相关的 Query 子集，并将此稀疏子集批处理成密集的物理块。一个物理 Query 块从 HBM 收集到 SRAM 中进行计算。这种两级阻塞方法通过在 SRAM 中缓存 Query，使得它们可以在逻辑 Key 块的所有物理瓦片中重复使用，从而摊销代价高昂的不规则内存访问，并进行高效的密集 GEMM 计算。
  ○ 带有重计算的后向传播：
后向传播沿用 FlashAttention-2 的内存高效设计，实现为三个融合核的序列。它并行化计算，每个线程块处理一个 Key 块。为处理稀疏性，它模仿前向传播的“gather-and-densify”策略，使用 varlen 索引将 Query 和输出梯度子集收集到片上瓦片中。它在后向传播期间重新计算注意力分数，以避免存储完整的注意力矩阵。Key 和 Value 梯度直接写入 HBM，而部分 Query 梯度（dQ）通过原子操作累积到高精度全局缓冲区中。

## UltraAttn 
UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling

https://dl.acm.org/doi/pdf/10.1145/3712285.3759894 SC25 翟季冬团队

https://github.com/oliverYoung2001/UltraAttn

//挑的是llama2-7b模型 这么小的模型做CP并行，不够典型

1. 📚 本文提出了UltraAttn，一种针对不规则注意力机制（irregular attention）的新型上下文并行（context parallelism）解决方案，旨在解决现有方法在长上下文LLM训练和推理中存在的通信开销大、内核粒度不灵活及带宽浪费等问题。
2. 🚀 UltraAttn通过在节点、设备和内核层级进行分层上下文切片（hierarchical context-tiling），并结合基于整数线性规划（ILP）的运行时调度器，显著减少通信量、优化硬件利用率并实现计算与通信的平衡。
3. ⚡ 在64个GPU上，UltraAttn在多种不规则注意力类型中，相对于现有最先进的上下文并行方法实现了平均5.5倍的加速，并展现出卓越的强扩展性和性能预测准确性。
UltraAttn 提出了一种针对不规则稀疏注意力（irregular block-sparse attention）的高效并行化解决方案，旨在加速大型语言模型（LLMs）的长上下文训练和推理。现有上下文并行化方法存在可扩展性差的问题，主要表现为：1) 条带状（striped-like）划分模式导致高通信流量；2) 基于环（ring-based）的通信模式限制了内核粒度，降低了设备利用率，并引入冗余通信。UltraAttn 通过分层上下文切片（hierarchical context-tiling）和基于整数线性规划（ILP）的运行时优化来解决这些问题，显著提高了分布式注意力的性能、适应性和可扩展性。
核心方法学
UltraAttn 的核心在于在三个层面（节点级、设备级、内核级）进行上下文切片，并结合 ILP 优化的运行时调度

## SwiReasoing切换
SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs

https://arxiv.org/abs/2510.05069 2025.10.9 佐治亚 微软等

https://github.com/sdc17/SwiReasoning

1. 📄 针对现有大型语言模型（LLMs）在显式推理中信息损失及隐式推理中稳定性与效率不足的问题，本文提出了SWIREASONING，一个免训练的动态推理框架。
2. 🔄 该框架基于次token分布的熵趋势评估置信度，智能地在显式和隐式推理模式间切换，以平衡探索和收敛；同时，通过限制模式切换次数来有效抑制过度思考，提升词元效率。
3. 📈 在多项数学和STEM推理基准测试中，SWIREASONING在不同LLM模型上均持续提高了平均1.5%至2.8%的准确性，并在有限词元预算下实现了56%至79%的平均词元效率提升。
SWIREASONING 是一种免训练（training-free）的框架，旨在提升大型语言模型（LLMs）在推理任务中的表现，同时提高 token 效率。该研究指出，传统的显式思维链（explicit chain-of-thought, CoT）推理受限于自然语言的离散性，在每一步都会选择一个 token 并丢弃其他可能性，从而限制了信息带宽和潜在推理路径。另一方面，虽然通过在连续的隐空间（latent space）中进行推理（latent reasoning）可以保留更丰富的信息并隐含地探索多条假设路径，但在免训练设置下，纯粹的隐式推理容易扩散概率质量、引入噪声，阻碍模型收敛到高置信度的解决方案，从而影响准确性。此外，即使在隐空间中，模型也可能出现“过度思考”（overthinking）现象，浪费 token 并降低效率。

## FastGRO 4+
FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning

https://arxiv.org/abs/2509.21792 兰州大学 2025.9.26 ICLR 2026投稿

https://github.com/yedaotian9/GRPO_speculative

针对RL 投机优化，基于EAGLE-2实现。
● batch变化 影响加速效果：-> 动态调整投机的参数(深度 广度)，使得投机+验证|原batch 打满算力 
● RL的更新后分布漂移 : -> 小模型也要更新，否则影响接受率（3.7->2.5）。训练有开销 说不多
7/8b模型单卡（torch），H800。消融实验里展示 更新draft模型的帮助更大。

## IMO bench 
Towards Robust Mathematical Reasoning
https://arxiv.org/abs/2511.01846 2025.11.3 Google

https://aclanthology.org/2025.emnlp-main.1794/

https://imobench.github.io

1. 🎉 本文提出了IMO-Bench，一套旨在评估和提升大模型国际数学奥林匹克（IMO）级别数学推理能力的基准套件。
2. 🏆 该套件包含IMO-AnswerBench（短答案）、IMO-Proof Bench（完整证明书写）和IMO-GradingBench（证明评分）三个部分，其中Gemini Deep Think模型在IMO 2025中实现了金牌级别的表现。
3. 🤖 研究还开发并验证了与人类评估高度相关的自动化评分系统，旨在推动社区从单纯追求答案转向发展深入、可验证的数学推理过程。
本文介绍了 IMO-Bench，一个旨在评估基础模型数学推理能力的基准套件，尤其关注国际数学奥林匹克 (IMO) 级别的鲁棒推理，以应对现有评估过于简单或仅侧重于最终答案匹配的局限性。IMO 竞赛题目以其严谨的多步骤推理和高度的创新性而闻名，使其成为评估 AI 推理能力的理想测试平台。

## AlphaEvolve
Mathematical exploration and discovery at scale

https://arxiv.org/abs/2511.02864 2025.11.3 Google DeepMind

AlphaEvolve: A coding agent for scientific and algorithmic discovery. Technical report, Google DeepMind, May 2025.
1. 🚀 AlphaEvolve 是一种通用进化coding agent，它结合了大型语言模型 (LLM) 的生成能力和自动化评估的迭代进化框架，旨在自主发现新颖的数学构造并推进对长期开放问题的理解。
2. 💡 该系统在数学分析、组合学、几何学和数论等67个问题中，不仅重新发现了最佳已知解决方案，还在多个案例中找到改进方案，并能将有限输入结果推广为普遍有效的公式。
3. 🛠️ AlphaEvolve 通过“搜索模式”和“泛化模式”实现大规模数学探索，可与 Deep Think 和 AlphaProof 等AI工具集成以实现自动化证明和形式化验证，从而匹配甚至超越人类已知的最佳结果，但对于需要深层全新见解的问题仍有局限。

核心方法论：
AlphaEvolve的核心是一个复杂的搜索算法，其设计灵感来源于局部搜索，但将其应用在更高抽象层次的空间中。
1. 演化计算框架：
AlphaEvolve维护一个程序群体（population），每个程序都编码了一个给定问题的潜在解决方案。这个群体通过一个模拟自然选择的循环进行迭代改进。该过程包含两个主要组成部分：
  ○ 生成器（Generator，LLM）： 负责引入变异。它选取当前群体中表现较好的程序，并利用LLM对它们进行“变异”，生成新的候选解决方案。这些变异是智能的、语法感知的代码修改，灵感来源于父程序的逻辑和人类用户提供的专家建议。这个过程可以在多个CPU上并行化。
  ○ 评估器（Evaluator，通常由用户提供）： 这是“适应度函数”。它是一段确定性的代码，运行群体中的一个程序，并根据其性能为其分配一个数值分数。对于数学构造问题，这个分数可以是构造满足特定属性的程度（例如，图中的边数或填充的密度）。
演化过程从几个简单的初始程序开始。在每一代中，选择一些得分较高的程序，将其输入LLM以生成新的、可能更好的后代。这些后代随后被评估、评分，其中得分较高的将构成未来程序的基础。这种生成和选择的循环使得程序群体能够随着时间演化，从而产生质量越来越高的解决方案。
2. 在程序空间中搜索：
AlphaEvolve的关键思想之一是，其局部搜索不是在数学对象（例如，图）的空间中进行，而是在生成这些对象的程序（例如，Python程序）的空间中进行。这种方法利用了LLM生成类似但略有不同的程序（“变异”）的能力。尽管每次LLM调用通常比修改一个数学对象本身昂贵得多，但通过在程序空间中搜索，可以探索比传统局部搜索方法少成千上万甚至数百万倍的候选对象。许多“优美”的数学对象，如最优的Hoffman-Singleton图，通常具有简洁、优雅的代码描述。因此，在程序空间中搜索可以作为一种强大的先验，偏好简洁性和结构性，帮助系统避开杂乱的局部最优解，找到优雅且通常是最优的解决方案。
3. 搜索模式（Search Mode）——演化搜索启发式算法：
AlphaEvolve的核心创新在于其“搜索模式”。它不是演化直接生成数学构造的程序，而是演化搜索数学构造的程序。
  ○ 在AlphaEvolve的群体中，每个程序都是一个“搜索启发式算法”（search heuristic）。
  ○ 每个启发式算法都会获得一个固定的时间预算（例如100秒），任务是在此时间内找到最佳的数学构造。
  ○ 该启发式算法的得分即为其所能找到的最佳构造的得分。
这种方法解决了LLM调用速度慢与数学对象评估速度快的矛盾：一次缓慢的LLM调用来生成一个新的搜索启发式算法，可以触发由该启发式算法自身进行的、大规模且廉价的数百万次候选构造的探索。
搜索过程并非每次都从零开始，而是新的启发式算法在其改进迄今为止发现的最佳构造的能力上进行评估。这创建了一个动态、自适应的搜索过程：初期，侧重于广泛探索的启发式算法可能受到青睐；随着接近好的解决方案，执行巧妙、问题特定优化的启发式算法可能占据主导。最终结果通常是一系列专业的启发式算法，它们串联起来能够产生最先进的构造。
4. 泛化模式（Generalizer Mode）——从实例到公式：
除了为固定问题规模（例如，针对𝑛=11的填充问题）找到构造外，AlphaEvolve还引入了更具雄心的“泛化模式”。在该模式下，AlphaEvolve被要求编写一个能够解决任意给定𝑛值的问题的程序。程序的评估基于其在一定范围的𝑛值上的表现。期望是AlphaEvolve通过观察自身针对小𝑛值（通常是最优的）的解决方案，能够发现模式并将其推广为适用于所有𝑛值的构造。这种模式在某些情况下产生了激动人心的结果，例如在Nikodym问题上启发了新的数学论文。

## Kosmos
Kosmos: An AI Scientist for Autonomous Discovery

https://arxiv.org/abs/2511.02824 爱迪生科学，牛津等

1. 🔬 Kosmos是一款AI科学家，它利用结构化“世界模型”协调并行数据分析和文献检索代理，以实现对开放性科学目标的自主、连贯探索。
2. 🚀 该系统能够进行数百次代理运行，处理数万行代码并阅读数千篇论文，单次运行平均可完成相当于人类研究者六个月的工作量，并产出引用清晰、可追溯的科学报告。
3. 💡 Kosmos已在多领域取得七项发现（其中四项为全新贡献），其报告中79.4%的陈述被独立专家验证准确，证明了其在加速和规模化数据驱动型科学发现方面的强大潜力。

## 渐进训练 5
Deep Progressive Training: scaling up depth capacity of zero/one-layer models

https://arxiv.org/abs/2511.04981 Meta Bu Zhiqi(唯一作者)

1. 🚀该研究提出了一种“零/一层渐进式训练”方法，旨在通过在训练过程中从极浅模型逐步扩展深度，高效提升大型深度模型的训练效率。
2. ⚙️该方法基于优化理论和特征学习，强调了新层的随机或复制初始化、通过μP理论实现的超参数迁移以及WSD（Warmup-Stable-Decay）学习率调度在确保收敛和训练稳定性方面的关键作用。
3. ⚡️实验证明，这种渐进式训练策略在GPT2等模型上可节省约80%的计算量或实现5倍加速，同时能达到与全尺寸模型几乎相同的性能，并适用于ResNet、GPT2和MoE等多种架构。
该论文深入研究了深度学习模型中的渐进式训练（progressive training）策略，旨在通过在训练过程中逐步扩展模型容量来提高深度模型的训练效率，同时保持或接近固有的模型性能。作者提出并倡导“零/一层渐进式训练”（zero/one-layer progressive training）方法，并通过优化理论和特征学习视角提供了关于新层初始化、超参数迁移、学习率调度和模型扩展时机的深刻见解。例如，在GPT2模型上，零/一层渐进式训练可以节省约80%的计算资源，相当于将训练速度提升约5倍，同时达到与完全训练的60层、70亿参数模型几乎相同的损失。

## CAT KV压缩 5？
Attention and Compression is all you need for Controllably Efficient Language Models

https://arxiv.org/abs/2511.05313 纽约大学等2025.11.7 压缩token，加速效果不错（基线recompute?）；精度比dense持平，比混合线性高，延迟和混合线性持平；

https://github.com/rajesh-lab/cat-transformer

1. 🎯 针对现有高效Transformer模型在提高效率时常牺牲上下文召回性能、且缺乏灵活性的问题，本文提出了一种名为Compress & Attend Transformer (CAT) 的新架构，它结合了密集注意力机制和序列压缩技术。
2. 💡 CAT通过并行压缩历史令牌块并让解码器关注这些压缩表示来生成当前令牌块，其独特之处在于可在训练时支持多种块大小，从而实现测试阶段在不重新训练的情况下动态调整质量与计算效率的权衡。
3. 🚀 实验结果表明，单个自适应CAT模型在语言建模、常识推理和长上下文理解等任务上全面超越了许多现有高效基线，并能以1.4-3倍的速度和2-9倍的内存效率，达到与密集Transformer相当甚至更优的性能。

## AlphaResearch 5
AlphaResearch: Accelerating New Algorithm Discovery with Language Models

https://arxiv.org/abs/2511.08522 清华 纽约 字节等 2025.11.11

https://github.com/answers111/alpha-research
1. 🤖 AlphaResearch是一个自主研究智能体，旨在利用大语言模型（LLM）发现开放式问题的新算法，其创新之处在于结合了基于程序执行的验证与模拟真实世界同行评审的双重研究环境。
2. 🏆 在包含8个开放式算法问题的AlphaResearchComp基准测试中，AlphaResearch取得了2/8的胜率，其中在“Packing Circles”问题上发现的算法性能超越了人类研究者及现有SOTA。
3. 💡 这项研究不仅展示了LLM加速算法发现的潜力，还通过对6/8失败案例的分析，为未来自主算法发现的挑战与发展方向提供了宝贵见解。
AlphaResearch 是一项旨在通过大型语言模型（LLMs）加速新算法发现的研究。该论文提出了一种名为 AlphaResearch 的自主研究智能体，其目标是在开放式问题上发现新算法，以弥补当前 LLMs 在处理复杂但易于验证的问题上取得显著进展，但在独立发现未知知识方面的不足。
为协同发现过程的可行性和创新性，AlphaResearch 构建了一个新颖的双重研究环境 (dual research environment)。该环境结合了基于执行的验证 (execution-based verify) 和模拟的真实世界同行评审环境 (simulated real-world peer review environment)。AlphaResearch 通过迭代运行以下步骤来发现新算法：
1. 提出新想法 (propose new ideas)
2. 在双重研究环境中验证想法 (verify the ideas)
3. 优化研究提案 (optimize the research proposals) 以获得更好的性能。

## Tree剪枝 5
Chopping Trees: Semantic Similarity Based Dynamic Pruning for Tree-of-Thought Reasoning

https://arxiv.org/abs/2511.08595 MIT等 2025.10.30  NIPS25

https://github.com/kimjoonghokim/SSDP
1. 🧠 针对Tree-of-Thought (ToT) 推理中因语义冗余导致的高昂计算成本，本文提出了一种名为语义相似度动态剪枝（SSDP）的轻量级方法。
2. ✂️ SSDP通过将在线语义合并模块整合到并行树搜索框架中，能够实时识别并修剪语义上等效的推理路径，大幅减少搜索空间。
3. ⚡️ 在GSM8K和MATH500等基准测试中，SSDP在保持竞争性准确率的同时，相比最先进的树搜索基线实现了高达2.3倍的速度提升，并减少了85%-90%的探索节点，为LLM推理提供了一种高效且可扩展的方案。
7b/8b模型相比best-of-N 精度更高；速度~4x。基于陶大成的DPTS

## 连续推理百万步 5
Solving a Million-Step LLM Task with Zero Errors

https://arxiv.org/abs/2511.09030 Cognizant AI Lab，2025.11.12

https://mp.weixin.qq.com/s/7JpXtjcXOa3LFNVXNvoGtg

意义：
模型规模不是唯一路径：MAKER证明，通过正确的架构设计，即使是较小的模型也能完成极其复杂的任务。这为AI发展提供了一个正交方向：不是一味追求更大、更"智能"的基础模型，而是通过更好的系统设计来提升能力。
错误纠正至关重要：
在计算机科学的许多领域（如数字电路、通信系统、生物系统），错误纠正都是实现可靠性的关键。

MAKER将这一原则应用到LLM系统中，取得了显著成功。
极致分解的力量：通过将任务分解到极致，每个子任务变得足够简单，使得错误率大大降低。这种"分而治之"的策略在软件工程中已被广泛应用，现在也证明对LLM系统有效。
MAKER的成功表明，大规模分解的agent过程（MDAPs）可能为高效解决组织和社会层面的问题提供了一条途径。这不仅是技术突破，更是AI发展范式的重要转变。

## EBM（能量校准）4
Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought

https://arxiv.org/abs/2511.07124v1 剑桥 清华 腾讯等，2025.11.10

https://mp.weixin.qq.com/s/twLIi_Mw-NPu5126bcMdxQ
1. 传统的显式CoT推理方法存在错误传播和表达限制，而现有隐式CoT模型则缺乏明确机制来确保推理步骤的一致性，常导致推理路径发散和结果不稳定。
2. 针对此问题，本文提出了基于能量的思维链校准（EBM-CoT）框架，利用能量模型（EBM）动态调整LLM的潜在思维表征，将其引导至嵌入空间中能量更低、一致性更高的区域。
3. 🌍 实验证明，EBM-CoT显著提升了数学、常识和符号推理任务中的准确性和一致性，以单链推理实现了接近多链自洽的效果，从而提高了推理效率。

## SSR
SSR: Socratic Self-Refine for Large Language Model Reasoning

https://arxiv.org/abs/2511.10621 Salesforce等 2025.11.13

https://github.com/SalesforceAIResearch/socratic-self-refine-reasoning

## TiDR 5
TiDAR: Think in Diffusion, Talk in Autoregression

https://arxiv.org/abs/2511.08923 NV 2025.11.12

1. 提出一种创新的序列级混合架构，旨在解决自回归（AR）模型的高质量低效率与扩散语言模型（dLMs）并行性强但质量受损的矛盾。
2. 该架构通过在单次前向传播中，利用特殊设计的注意力掩码，结合扩散机制并行生成草稿（Thinking）和自回归机制进行高质量采样（Talking）。
3. TiDAR在保持与AR模型相当的生成质量下，实现了4.71倍至5.91倍的吞吐量加速，并在效率和质量上超越了投机解码及其他扩散模型。

## KV Cache Transform Coding for Compact Storage in LLM Inference

https://arxiv.org/abs/2511.01815
1. 🧠 针对大型语言模型（LLM）推理中KV缓存管理面临的显存消耗和传输开销挑战，本文提出了kvtc，一种轻量级变换编码器，旨在实现KV缓存的紧凑存储。
2. 🌟 kvtc借鉴经典媒体压缩技术，通过结合基于PCA的特征去相关、利用动态规划实现自适应量化以及熵编码（DEFLATE），仅需一次简短的初始校准，且不改变模型参数。
3. 🚀 实验结果表明，kvtc能够在保持推理和长上下文准确性的前提下，实现高达20倍（特定用例下可达40倍或更高）的压缩率，并一致优于现有的推理时基线方法，显著降低了LLM服务的内存和传输成本。

## NVFP4训练 5
TetraJet-v2: Accurate NVFP4 Training for Large Language Models with Oscillation Suppression and Outlier Control

https://arxiv.org/abs/2510.27527 2025.10.31 清华大学陈剑飞 
1. 🚀 本文提出了TetraJet-v2，一个针对大型语言模型（LLM）的端到端NVFP4全量化训练方法，旨在解决低精度训练中普遍存在的权重震荡和异常值问题。
2. 💡 该方法引入了无偏双块量化设计，并开发了OsciReset算法以抑制权重震荡，同时提出了OutControl算法来保留激活值和梯度中的异常值精度。
3. ✅ 实验结果表明，TetraJet-v2在高达370M模型和200B tokens数据规模的LLM预训练上，持续优于现有FP4训练方法，平均将与全精度训练的性能差距缩小了51.3%。
该论文提出了TetraJet-v2，一种针对大型语言模型（LLMs）的端到端NVFP4全量化训练（FQT）方法，旨在解决低精度训练中普遍存在的权重振荡和离群值问题，从而实现接近无损的训练性能。

核心问题与贡献：
LLMs训练成本高昂，促使研究人员探索低精度FQT。NVFP4等4比特格式能显著提升效率，但在如此低的精度下实现接近无损的训练仍具挑战。论文识别出阻碍低精度LLM训练的两个关键问题：
1. 权重振荡（Weight Oscillation）：量化权重在高精度主权重（master weights）只有微小变化时，却在两个量化桶之间频繁跳变，损害模型性能。
2. 离群特征（Outlier Features）：激活中少数通道的幅度巨大，导致动态范围过高，FP4难以精确表示，这与模型最终性能高度相关。
为解决这些问题，TetraJet-v2提出了三项关键技术：
1. NVFP4线性层的无偏双块量化方法（Unbiased Double-Block Quantization for NVFP4 Linear Layers）：设计了满足NVFP4数值需求的双块缩放机制，并提供无偏梯度估计的全FP4线性层。
  a. 在32block之外 在增加一层128的block 先行缩放到 448*6
2. OsciReset算法：抑制训练过程中的权重振荡。
3. OutControl算法：保持离群值的精度

## Vibe Thinking 5(TBD)
Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B

https://arxiv.org/abs/2511.06221 2025.11.9 weibo

1. 🧠 VibeThinker-1.5B模型挑战了小型模型缺乏强大推理能力的普遍观点，以仅1.5亿参数和不到8000美元的总训练成本，实现了与领先大型模型相当甚至更优的推理性能。
2. 🛠️ 该模型采用创新的“Spectrum-to-Signal Principle (SSP)”后训练方法，通过SFT阶段的“Two-Stage Diversity-Exploring Distillation”生成多样化解决方案，并在RL阶段利用“MaxEnt-Guided Policy Optimization (MGPO)”策略放大正确的信号。
3. 🚀 VibeThinker-1.5B在AIME24、AIME25和HMMT25等挑战性数学基准上超越了参数量大400倍的DeepSeek R1，并在LiveCodeBench V6编码基准上表现出色，证明了小型模型实现高级推理能力的可行性并大幅降低了相关成本。

VibeThinker-1.5B是Sina Weibo Inc.发布的一个1.5B参数的紧凑型密集模型，挑战了当前大型模型领域中“小模型缺乏强大推理能力，必须通过扩展参数量来增强性能”的普遍共识。该研究报告提出，这一假设可能是不准确的。
该模型通过一种创新的后训练方法，即“频谱到信号原则（Spectrum-to-Signal Principle, SSP）”，在多样性驱动的优化下，激发了小型模型的强大推理能力。SSP框架系统地增强了模型输出的多样性，并将其分为两个协同阶段：
SFT阶段像“撒网”一样广泛探索解决方案，RL阶段则“收网”聚焦最优路径，这种分工协同是模型成功的关键。

## 稀疏更新 
田渊栋等

这篇论文深入探讨了RLVR（Reinforcement Learning with Verifiable Rewards）如何有效地提升大型语言模型（LLM）的推理能力，同时却只修改了极小部分参数的“稀疏更新”这一反直觉现象。作者认为，这种表面上的稀疏性是模型条件优化偏差（model-conditioned optimization bias）的体现，即对于一个固定的预训练模型，参数更新会一致地集中在模型偏好的参数区域，并且这种模式在不同的训练运行中高度一致，对数据集和RL算法的变化具有很强的鲁棒性。
为了从机制上解释这些动态，论文提出了一个“三门理论”（Three-Gate Theory）

## 反思：多数是确认
First Try Matters: Revisiting the Role of Reflection in Reasoning Models

https://arxiv.org/abs/2510.08308 MicroMind AI等 2025.10.9 

https://github.com/Olafyii/first-try-matters 推理基于SGlang

1. 本研究系统分析了大型语言模型推理过程中的“反思”行为，发现这些反思步骤大多是确认性的（超过90%），而非纠正性的，即很少改变模型最初得出的答案。
2. 💡 尽管反思在推理过程中消耗大量tokens但对最终性能提升有限，但训练数据中丰富的反思能显著提高模型首次尝试的正确性，而非增强其自我纠错能力。
3. 🚀 基于此，研究提出了一种问题感知型早期停止推理方法，通过动态截断不必要的反思步骤，可在降低24.5% token消耗的同时，将准确率降幅控制在2.9%以内。

## OWL
OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs

https://arxiv.org/abs/2510.07535 Snowflake AI Research, CMU, 2025.10.8

https://anonymous.4open.science/r/owl-BFB8

这篇论文提出了一种名为OWL（Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs）的新型推测解码方法，旨在解决现有方法在处理长上下文输入时性能严重下降的问题。论文首先指出，尽管推测解码能加速大型语言模型（LLM）的推理，但现有基准测试（如最大2K tokens）未能反映真实世界中长上下文（如多轮对话、智能体系统）的场景。例如，EAGLE3在长上下文输入下甚至导致生成速度降低0.81倍。
为解决这些问题，论文做出了三项主要贡献：
1. LongSpecBench新基准测试： 引入了一个专门用于评估长上下文推测解码性能的新基准测试LongSpecBench。该基准从WildChat-4.8M对话数据集中采样200个输入长度介于4K到64K tokens的例子。实验表明，现有方法如EAGLE3在此基准上表现不佳，其接受长度仅为1.28。
2. OWL模型： 提出了一种创新的推测解码模型OWL，它在长上下文输入上的接受长度比EAGLE3高出近5倍。OWL的核心创新包括：
  ○ 长度泛化Drafter： 针对现有基于Transformer的Drafter（如EAGLE3）受限于训练上下文窗口（如2K tokens）的问题，OWL设计了一个基于LSTM的Drafter。该Drafter仅依赖目标LLM的最后一个token的隐藏状态来预测后续token，从而避免了对完整输入序列的依赖，使其与上下文长度无关，实现了长度泛化。具体地，给定下一个预测token

## SuffixDecoding
SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications

https://arxiv.org/pdf/2411.04975 NIPS25

https://github.com/snowflakedb/ArcticInference 
1. 🧐 针对大型语言模型（LLM）代理应用中常见的重复且可预测的长令牌序列推理请求，现有推测解码方法未能有效利用这些模式，导致延迟瓶颈。
2. 🚀 SuffixDecoding提出一种新颖的无模型推测解码方法，利用高效的后缀树缓存提示和历史输出中的长令牌序列，并根据接受可能性自适应地调整推测长度，从而优化计算效率。
3. ⚡ 评估结果显示，在SWE-Bench和Text-to-SQL等代理基准测试中，SuffixDecoding实现了高达5.3倍的速度提升，显著优于EAGLE-2/3等基于模型和Token Recycling等无模型的现有先进方法。

SuffixDecoding是一种新颖的无模型推测解码（speculative decoding）方法，旨在解决新兴AI应用（尤其是基于LLM的Agent应用）中存在的推理延迟问题。这类Agent工作负载（如多Agent管道或自我修正循环）通常会提交重复且高度可预测的推理请求，生成长而重复的token序列，而现有推测解码方法未能有效利用这些模式。
该研究的核心方法是利用高效的后缀树（suffix trees）来缓存提示（prompts）和先前输出中的长token序列。具体而言，SuffixDecoding维护两个后缀树：一个全局后缀树（global suffix tree）用于存储历史生成的输出，另一个per-request后缀树用于存储当前正在进行的推理请求的提示和已生成部分。后缀树的每个节点代表一个token，从根节点到任意节点的路径编码了之前观察到的子序列，从而能够快速匹配模式并识别基于先前出现次数的可能延续。这种无模型的方法使得草稿token的生成速度极快（约20微秒/token），且无需额外的GPU开销。

## 机会专家
Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

https://arxiv.org/abs/2511.02237 2025.11.4 Tri Dao等

1. 💡 针对MoE大型语言模型在自回归生成（解码阶段）中常见的内存瓶颈问题，研究发现其延迟主要受限于批次内激活专家总数，而非单个专家的计算负载。
2. 🚀 本文提出了一种名为机遇性专家激活（OEA）的批次感知路由框架，该框架在不重新训练模型的情况下，通过两阶段策略动态优化专家路由：首先为每个令牌确定一组基线核心专家，然后允许令牌“顺带利用”批次中其他令牌已激活的专家。
3. ⚡ 实验结果表明，在Qwen3-30B和Qwen3-235B模型上，OEA在保持模型质量不显著下降的前提下，分别实现了MoE层解码延迟39%和15%的显著降低。

这篇论文介绍了一种名为“机会主义专家激活”（Opportunistic Expert Activation, OEA）的批次感知（batch-aware）路由框架，旨在通过动态调整令牌到专家的映射，以减少在大型语言模型（LLMs）自回归生成（decode stage）阶段激活的专家总数，从而降低推理延迟，同时保持模型质量。

## bigbang
BigBang-Proton: Next-Word-Prediction is Scientific Multitask Learner

大爆炸-光子: 基于自回归序列的跨尺度、跨结构、跨学科通用科学预训练大模型（~1.5B，20层，序列10^30）

https://arxiv.org/abs/2510.00129 2025.9.30 上海 超对称公司（SuperSymmetry Tech）

https://github.com/supersymmetry-technologies/BigBang-Proton 

https://huggingface.co/SuperSymmetryTechnologies/BigBang-Proton
Hengkui Wu (hkwu@ssymmetry.com)
Liujiang Liu (liuliujiang@ssymmetry.com)
地址: SuperSymmetry Technologies, 1528 Gumei Road, Xuhui District, Shanghai
● 早期的模型：BigBang-Neutron Scaling Particle Collision Data Analysis
● https://arxiv.org/abs/2412.00129 2024
1. ✨ BigBang-Proton 是一种统一的基于自回归语言序列的架构，通过在跨尺度、跨结构、跨学科的真实世界科学任务上进行预训练，旨在构建一个科学多任务学习器，并引入了理论-实验学习范式、二进制补丁编码和蒙特卡洛注意力机制。
2. 🎯 该模型在高达50位数的算术加法运算中实现了100%的准确率，在粒子物理喷注标记和原子间势模拟中表现与领先的专用模型相当，显著超越了主流通用大型语言模型。数学、物理、材料、基因序列等。
3. 🚀 这些结果表明，语言引导的科学计算能够匹配甚至超越任务专用科学模型的性能，同时保持多任务学习能力，为开发普适的物质世界基础模型奠定了关键基础。

## FP16
Defeating the Training-Inference Mismatch via FP16

https://arxiv.org/abs/2510.26788 SeaAI 2025.10.30

https://github.com/sail-sg/Precision-RL 

## 采样温度
On the Role of Temperature Sampling in Test-Time Scaling

https://arxiv.org/abs/2510.02611 2025.10.2 斯坦福

1. 🤔 传统上，大语言模型（LLMs）通过测试时扩展（TTS）增加采样数量K来提高推理能力，但研究发现，当K值较大时性能提升趋于停滞，部分难题仍无法解决。
2. 💡 本文发现不同采样温度能解决不同子集的难题，这表明单一温度采样未能充分发掘模型潜力，因此提出沿温度维度进行扩展以拓宽LLMs的推理边界。
3. 🚀 该多温度扩展策略在多个模型和基准测试上平均带来7.3点的额外性能提升，使得基础模型无需额外训练即可达到与强化学习（RL）训练模型相当的水平，并设计了高效的多温度投票方法以降低计算开销。

## 分子生物学ReaSyn
Rethinking Molecule Synthesizability with Chain-of-Reaction

https://arxiv.org/abs/2509.16084

https://mp.weixin.qq.com/s/lDWO5MdH_ywKYVBg4gVx-g 

基于https://github.com/wenhao-gao/synformer （MIT 2024）修改 pytorch，pytorch-lighting，A100

1. 🎯 针对分子生成模型在可合成性方面表现不佳，以及现有方法在化学空间覆盖和优化性能上的局限，本文提出了 ReaSyn 框架，旨在通过生成可合成途径来探索给定分子的可合成类似物。
2. 💡 ReaSyn 引入了“Chain-of-Reaction (CoR)”符号，将其合成途径类比于大语言模型 (LLM) 的“Chain-of-Thought (CoT)”推理路径，显式地包含反应物、反应类型和中间产物以实现密集的逐步监督和推理，并通过强化学习微调和目标导向的测试时间计算扩展进一步增强其推理能力。
3. 🏆 实验证明，ReaSyn 在可合成分子重建、目标导向分子优化和命中扩展等任务中均显著超越现有方法，实现了最高的重建率、途径多样性和优化性能，展现了其在导航庞大可合成化学空间中的卓越能力。

## RSPO
Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts

https://arxiv.org/abs/2510.23027 2025.10.27 微软

1. 🎯 针对专家混合模型（MoE）在强化学习（RL）训练中因Router波动和方差不匹配而导致的稳定性问题，本文提出了一种新型路由器感知策略优化算法——Router-Shift Policy Optimization (RSPO)。
2. 💡 RSPO 通过引入一个路由器偏移比率来量化新旧策略间路由分布的偏差，并以此对 token 级重要性采样（IS）权重进行软性重缩放，同时采用序列级重要性比率的几何平均以减少梯度方差。
3. 🚀 实验证明，RSPO 在数学推理基准测试中显著提升了 MoE 模型的收敛稳定性和最终性能，优于现有基线方法，并有助于保持更高的 token 熵。

MoE模型通过稀疏激活专家来提高模型容量和计算效率，使其在大规模RL训练中具有吸引力。然而，将RL应用于MoE模型面临稳定性挑战，主要源于路由器波动（router fluctuation）：相同输入 token 选择的专家集合在策略更新后可能发生显著变化，导致重要性采样（IS）权重方差增大，优化不稳定，甚至奖励崩溃。此外，传统方法常采用 token-level 的 IS 比率，这与RLVR（Reinforcement Learning with Verifiable Rewards）通常使用的 sequence-level 奖励不匹配，进一步加剧了不稳定性。
为解决这些问题，本文提出了一种名为 Router-Shift Policy Optimization (RSPO) 的RL算法，专为MoE架构设计，以实现稳定高效的训练。RSPO 不像 router freezing 或 routing replay 那样对路由器施加严格限制，而是引入了一个 路由器偏移比率（router shift ratio），用于量化当前策略与旧策略之间路由决策的偏离程度。此比率基于路由器 logit 计算，并用于软性地重新缩放 IS 权重。
RSPO 的核心思想和技术细节如下：
1. 问题动机：
  ○ 路由器波动（Routing fluctuations）： 策略更新后，相同 token 激活的专家集合或其路由概率可能发生变化。这导致 IS 比率大幅波动，频繁触发裁剪机制，增加训练方差。
  ○ 方差不匹配（Variance mismatch）： 大多数 GRPO 实现将 sequence-level 的优势估计与 token-level 的 IS 比率对齐，这在 MoE 环境下被放大，加剧了不稳定性。
2. RSPO 算法：
RSPO 保留了 sequence-level 的重要性比率，但采用 token-level 裁剪以减少信息损失。同时，引入路由器偏移比率来度量路由器分布在当前策略和旧策略之间的偏差，并用其对 token-level 的重要性比率进行重新加权，并软性裁剪表现出严重路由漂移的 token。这在不冻结路由器的情况下稳定了训练。

## AEPO
Agentic Entropy-Balanced Policy Optimization
https://www.arxiv.org/abs/2510.14545 快手 人大
https://github.com/dongguanting/ARPO
1. 🔍 本文提出了Agentic Entropy-Balanced Policy Optimization (AEPO) 算法，旨在解决Agentic强化学习中，过度依赖熵信号导致的轨迹回溯崩溃和高熵Token梯度裁剪问题。
2. 💡 AEPO包含两部分：一是动态熵平衡 回溯机制，通过熵预监控自适应分配采样预算并惩罚连续高熵分支；二是熵平衡策略优化，通过停止梯度操作保留并重新缩放高熵Token梯度，并结合熵感知优势估计。
3. 🚀 在14个挑战性数据集上的实验表明，AEPO持续优于7种主流强化学习算法，显著提升了回溯采样多样性和策略熵稳定性，从而促进了可扩展网络Agent的训练。

本文提出了一种名为Agentic Entropy-Balanced Policy Optimization (AEPO) 的强化学习算法，旨在解决Agentic强化学习（Agentic RL）中，过度依赖熵信号进行探索可能导致的训练崩溃问题。该算法通过在Rollout和策略更新阶段平衡熵来提升多轮、长周期工具使用（multi-turn, long-horizon tool-use）的Web Agent性能。
文章首先揭示了Agentic RL中两个由高熵引发的核心挑战：
1. High-entropy Rollout Collapse (高熵Rollout崩溃)：在Rollout阶段，连续的高熵工具调用步骤（tool-call steps）会导致模型在一个单一轨迹（single trajectory）上过度分支（over-branching），从而耗尽其他潜在正确分支的采样预算，限制了Rollout采样的多样性和覆盖范围。
2. High-entropy Token Gradient Clipping (高熵Token梯度裁剪)：在策略更新阶段，传统的强化学习算法会积极裁剪高熵Token的梯度，这阻碍了模型学习有效的探索性行为，尤其是在涉及外部工具调用时。

## RollFlash
ROLL Flash – Accelerating RLVR and Agentic Training with Asynchrony

https://arxiv.org/abs/2510.11345 2025.10.13 阿里
1. 🚀 ROLL Flash 是一个为大语言模型（LLM）强化学习（RL）后训练设计的异步系统，通过细粒度并行和Rollout-训练解耦两大核心原则，有效解决了传统同步方法中资源利用率低和可扩展性差的问题。
2. ⚙️ 该系统采用队列调度、提示复制、环境级异步执行等机制，实现Rollout与训练阶段的并行化，理论和实践均证明其能显著缓解长尾Rollout造成的GPU空闲，从而提升资源利用率和吞吐量。
3. ⚡️ 实验结果显示，ROLL Flash在RLVR任务中实现了高达2.24倍的加速，在Agentic任务中达到2.72倍的加速，并且在最优异步比配置下，通过支持Off-policy算法，能保持与同步训练相当的性能和稳定性。

## RLBoost
RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs

https://arxiv.org/abs/2510.19225 2025.10.24 UCB, CMU, Goolge etc

https://github.com/Terra-Flux/PolyRL 基于SGLang Verl
1. 💡 RLBoost提出了一种混合架构，旨在解决大型语言模型(LLM)强化学习(RL)中rollout和训练阶段资源需求不匹配的问题，通过有效利用抢占式GPU资源实现成本效益。
2. 🤖 该系统通过自适应rollout卸载 动态调整预留训练集群的工作负载、拉取式权重传输快速为新实例提供最新模型，以及token级响应收集与迁移来高效处理抢占并实现持续负载均衡。
3. 🚀 实验结果表明，与仅使用按需GPU资源相比，RLBoost在保持同步RL算法的同时，将训练吞吐量提高了1.51至1.97倍，并将成本效率提升了28%至49%。
RLBoost 提出了一种系统性解决方案，旨在通过利用可抢占式 GPU 资源，实现大语言模型（LLM）强化学习（RL）训练的高效且经济。

##开源模型 IOI金牌
Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models

https://arxiv.org/abs/2510.14232v1 2025.10.16

1. 🏆 本文提出了GENCLUSTER，一个可扩展的测试时计算框架，旨在通过大规模代码生成、行为聚类、排名和轮询提交策略，帮助大型语言模型（LLMs）在有限验证预算下高效探索解决方案空间。
2. 🥇 实验首次证明，该框架结合开源模型gpt-oss-120b，能在国际信息学奥林匹克竞赛（IOI）中达到金牌水平，得分446.75。OpenAI的分数比这个高7%~20%；分数比DeepSeek高25+%。
  a. OSS加大K分数提升scale 47%；DeepSeekscale 37%；Qwen3-235B-A22B scale 15%

## EPPO 采样遗忘
https://arxiv.org/abs/2510.05837 2025.10.7 微软等

1. 🤖 针对大型语言模型（LLMs）中可验证奖励强化学习（RLVR）存在的探索-利用平衡挑战，现有方法常过度侧重利用，导致熵坍缩和泛化能力受限。
2. 💡 为此，本文提出了EEPO（Exploration-Enhanced Policy Optimization）框架，通过“先采样后遗忘”机制促进探索，该机制将rollout过程分为两阶段，在第一阶段采样后执行轻量级遗忘步骤，以暂时抑制已采样响应，强制第二阶段探索输出空间的不同区域。
3. 🚀 在五个数学推理基准上，EEPO始终优于GRPO及其他基线方法，在Qwen2.5-3B上实现平均相对收益24.3%，在Llama3.2-3B-Instruct上实现33.0%，在Qwen3-8B-Base上实现10.4%，同时保持了训练效率。
4. EEPO (Exploration-Enhanced Policy Optimization) 是一种旨在解决大型语言模型 (LLM) 可验证奖励强化学习 (RLVR) 中探索-利用困境的框架。现有 RLVR 方法（如 GRPO）往往过度强调利用高奖励轨迹，导致策略熵急剧下降（即熵坍塌），模型对训练分布过拟合，泛化能力受限，并陷入“自我强化循环”：模型反复采样并奖励主导行为模式，进一步抑制了对其他潜在优质推理策略的探索。
为了解决这一问题，EEPO 引入了一种“先采样后遗忘 (sample-then-forget)”机制，通过自适应遗忘在 rollout 过程中促进探索。该机制将 GRPO 的 rollout 过程分为两个阶段：
1. 第一阶段采样 (Stage 1 sampling)：Rollout 模型 $\pi_{rollout}$ 生成一半轨迹 (例如 $G/2$ 条)。
2. 遗忘 (Unlearning)：Rollout 模型对第一阶段采样的轨迹执行一个轻量级的遗忘步骤，暂时抑制这些已采样的响应。
3. 第二阶段采样 (Stage 2 sampling)：模型从经过遗忘更新后的 $\pi_{rollout}$ 中采样剩余的轨迹
   
## MoE R3 

小米
1. 💡 Mixture-of-Experts (MoE) 强化学习 (RL) 训练存在路由机制不稳定性问题，主要源于推理和训练阶段路由行为的显著不一致性，甚至在相同条件下重复前向传播也会产生专家选择差异。
2. 🛠️ 为解决此根本性问题，本文提出了 Rollout Routing Replay (R3) 方法，通过记录推理引擎的路由分布并在训练时复现，以对齐两阶段的路由决策并保留梯度流。
3. 📈 实验结果表明，R3 显著降低了训练-推理策略的 KL 散度，将极端差异令牌的数量减少了一个数量级，从而有效稳定了 MoE RL 训练，防止了崩溃并优于现有方法。

## PEAR阶段熵
PEAR: Phase Entropy Aware Reward for Efficient Reasoning

https://arxiv.org/abs/2510.08026 2025.10.10 新加坡南洋理工

https://github.com/iNLP-Lab/PEAR
1. 🧐大型推理模型(LRMs)生成的思维链(CoT)解释常过长，增加了推理成本并降低可用性；通过实证分析发现，模型熵与推理响应长度在不同阶段呈正相关，其中“思考阶段”熵高而“最终答案阶段”熵低。
2. 💡基于此洞察，本文提出Phase Entropy Aware Reward (PEAR)奖励机制，它在奖励设计中融入阶段性熵值，惩罚思考阶段的过度探索性熵，同时允许最终答案阶段适度探索，以生成更简洁的推理路径。// 还应该有转折token 鼓励
3. 📊PEAR在四个基准测试中显著减少了37.8%至59.4%的响应长度，同时保持了竞争力甚至更高的准确率，并展示出强大的域外泛化能力，无需预设长度目标或硬性截断规则。

## LLMs as Improvement Operators
Rethinking Thinking Tokens: LLMs as Improvement Operators  
  
https://arxiv.org/abs/2510.01123 2025.10.1 Meta Superintelligence Lpdf & Anthropic

优化纯CoT推理 和 RL
1. 💡 该研究将大型语言模型（LLM）视为改进操作符，提出了Parallel-Distill-Refine (PDR) 和 Sequential Refinement (SR) 等迭代推理方法，旨在解决长链思维 (Long CoT) 推理导致的上下文长度、计算成本和延迟过高问题。
2. 🚀 PDR通过并行生成多样化的草稿，将其提炼为有界的文本工作区，再基于此工作区进行迭代细化，有效解耦了上下文长度与总生成Token数，从而在固定延迟下实现对计算成本的精确控制。
3. ✅ 实验证明，PDR在相同顺序预算（Bseq）下，其准确性优于Long CoT和SR，尤其在数学任务上获得最大增益（AIME 2024/2025分别提升11%和9%）；与PDR推理方法一致的RL训练能进一步提升模型性能。

## 2-GRPO
It Takes Two: Your GRPO Is Secretly DPO
https://arxiv.org/abs/2510.00977 2025.10.1

1. ✨ GRPO（Group Relative Policy Optimization）是LLM后训练中的一种重要强化学习算法，但普遍认为其需要较大的分组规模以确保统计估计的稳定，从而带来高昂的计算成本。
2. 💡 本研究将GRPO重新诠释为对比学习的一种形式，揭示了其与DPO（Direct Preference Optimization）的内在联系，并在此启发下提出了极小分组尺寸（G=2）的2-GRPO。
3. ⚡ 理论分析与实证结果表明，2-GRPO在保持与16-GRPO相当性能的同时，能将rollout使用量减少八分之一，并显著缩短70%以上的训练时间，挑战了GRPO需要大分组规模的传统认知。

## LoopLM
Scaling Latent Reasoning via Looped Language Models 
https://arxiv.org/abs/2510.25741v1 2025.10.29

http://ouro-llm.github.io/

1. 论文提出了Ouro（循环语言模型，LoopLM），通过在预训练阶段集成迭代潜在计算和熵正则化深度分配目标，在7.7T tokens上训练，旨在将推理能力内置于模型中而非依赖后训练。
2. 实验结果表明，1.4B和2.6B的Ouro模型在多项基准测试中，性能可媲美甚至超越4B和8B的SOTA LLM，其卓越优势主要源于其知识操纵能力的提升，而非知识存储容量的增加。
3. LoopLM通过自适应计算和高效KV缓存共享机制提高了推理效率和安全性，并提供了一个比显式Chain-of-Thought（CoT）更忠实的内部推理过程，确立了循环深度作为模型扩展的第三个关键维度。

## 混合架构
Hybrid Architectures for Language Models: Systematic Analysis and Design Insights

https://arxiv.org/abs/2510.04800 Meta 2025.10.6
1. 📚 该研究系统地评估了语言模型中层间（inter-layer）和层内（intra-layer）混合架构的设计策略，旨在深入分析其建模质量与计算效率。
2. ✨ 实验结果表明，混合模型在语言建模性能、长上下文能力和MoE兼容性方面均优于同质架构及SWA模型，其中层内混合架构在质量-效率的Pareto曲线上表现最佳。
3. 💡 这些混合架构通过利用Mamba的线性复杂度，显著提升了训练和推理效率，并提供了关于最优模块比例、排列顺序及融合策略的实用设计指导。
这篇论文深入探讨了语言模型的“混合架构”（Hybrid Architectures），旨在平衡建模质量与计算效率，尤其是在处理长上下文时。论文对两种主要的混合策略——“层间混合”（inter-layer hybrid）和“层内混合”（intra-layer hybrid）进行了系统性的分析和比较。

## Attn与规则
Extracting Rule-based Descriptions of Attention Features in Transformers

https://arxiv.org/abs/2510.18148 2025.10.20

1. 🤔 现有Transformer特征的机械可解释性常依赖于主观的示例检查，本研究提出通过提取注意力层稀疏自编码器（SAE）特征的规则描述，以提供更具解释性的理解。
2. ⚙️ 论文将注意力头计算分解为基于输入特征 对交互的加权和，并提出了三种规则类型：跳跃n-gram规则（skip-gram rules）、缺失规则（absence rules）和计数规则（counting rules），同时开发了一种自动提取这些规则的实证方法并应用于GPT-2 small。
3. 📊 结果表明，大多数特征可由约100条跳跃n-gram规则良好描述，早期层中也广泛存在缺失规则，并发现了计数规则，这为未来通过规则描述和理解大型语言模型行为奠定了基础。
这篇论文深入探讨了 Transformer 语言模型中注意力特征的机制可解释性（mechanistic interpretability），旨在用基于规则的描述（rule-based descriptions）来解释模型行为，而非传统上依赖对高激活范例（exemplars）的主观检查。虽然现有的主流方法，如稀疏自编码器（Sparse Autoencoders, SAEs），能提取特征向量（feature vectors），但对这些特征的解释通常需要耗时的人工检查，且可能存在主观性、不完整或不准确的问题。
论文提出了一种不同的解决方案：用符号化、人类可读的规则来描述 SAE 特征。具体来说，该研究关注注意力层（attention layers）的输出特征，并将其计算表达为输入特征之间交互的加权和。这些交互可以匹配输入中的 token 模式，并相应地增加或减少特定输出 token 的可能性。
论文的核心方法论是将注意力特征的计算进行分解（Decomposing Attention Features），使其可以被解释为特征对（feature pairs）之间交互的加权和，这些交互代表了输出 token 的促进或抑制。

## 样本级梯度更新
Per-example gradients: a new frontier for understanding and improving optimizers  
https://arxiv.org/abs/2510.00236 Google Deepmind, 2025.9.30

1. 💡 深度学习优化器通常仅使用mini-batch平均梯度，限制了对逐样本梯度统计信息的访问；本文证明了通过计算图修改或JAX的vmap转换，高效获取此类逐样本梯度统计信息在计算上是可行的。
2. 🛠️ 研究表明，这些泛型梯度统计信息可通过自动微分图的“手术式”修改实现，在某些情况下，其计算和内存开销与传统mini-batch梯度计算几乎相同，尤其在Transformer等序列模型中表现出高效性。
3. 📈 基于逐样本梯度分析，作者发现SignSGD中符号操作的最佳位置应尽可能晚地应用，且Adam优化器中的预处理器若由梯度均方项而非方差项主导，则训练速度更快、稳定性更强，这与传统观点相悖。

这篇论文探讨了在深度学习优化器中访问并利用单样本梯度（per-example gradients）信息的新方法和可能性。传统上，深度学习训练算法通常将 mini-batch 中的样本视为一个整体，通过对 mini-batch 梯度求平均值进行处理。由于自动微分（AD）框架在计算除平均值之外的其他统计量时资源开销巨大，研究人员通常避免使用这些信息。这篇论文指出，情况并非总是如此。
核心方法学：高效计算单样本梯度统计量

论文的核心贡献之一是提出并验证了两种高效获取单样本梯度统计量的方法：
1. 计算图重构（Computational Graph Surgery）：
该方法通过对自动微分（AD）的计算图进行“重构”来实现。在 mini-batch 梯度计算中，大多数操作保留了单样本信息，直到最后一个聚合步骤。通常，梯度在进行求和约简（sum reduction）之前，每个样本的计算路径是独立的。通过在求和约简之前“注入”所需的非线性函数 $\phi$，可以直接计算单样本梯度统计量。
具体来说，对于一个权重为 $W_k \in \mathbb{R}^{D \times D}$ 的全连接层，单样本的梯度为 $\nabla_{W_k} f(\theta; x_i) = s_i r_i^\top$（一个秩一矩阵），其中 $s_i$ 是中间输入， $r_i$ 是反向传播的残差。 mini-batch 梯度是 $\nabla_{W_k} \sum_{i=1}^B f(\theta; x_i) = \sum_{i=1}^B s_i r_i^\top = SR$，其中 $S=(s_1, \dots, s_B)^\top$ 和 $R=(r_1, \dots, r_B)$ 分别堆叠了中间输入和残差。
对于“可因式分解（factorable）”的操作 $\phi$（例如 $\text{sign}$ 函数或任意幂函数，如平方），如果 $\phi(sr) = \phi(s)\phi(r)$，则可以通过简单地对 $S$ 和 $R$ 进行元素级操作来计算相应的单样本梯度统计量。例如，元素级平方和可以表示为：
$\sum_{i=1}^B [\nabla_{W_k} f(\theta; x_i)]^2 = \sum_{i=1}^B (s_i r_i^\top)^2 = \sum_{i=1}^B (s_i)^2 (r_i^\top)^2 = S^2 R^2$
2. JAX vmap 向量化变换：
对于序列模型（如 Transformer），激活值的内存成本为 $BDL$（$B$ 为 batch size， $D$ 为维度， $L$ 为序列长度），而参数的内存成本为 $D^2$。如果序列长度 $L$ 大于等于维度 $D$，那么存储 $B$ 个独立梯度的成本可能小于存储激活值的成本。在这种情况下，使用 JAX 的 vmap 向量化变换来实现单样本梯度统计量是可行的，并仅带来适度的计算开销（例如，在 1.2B 参数的 Transformer 模型上，仅增加约 17% 的计算开销），同时不增加峰值内存使用。这种方法为快速原型开发和实验提供了途径

## Jet-Nemotron
Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
https://arxiv.org/abs508.15884 NVIDIA

https://github.com/NVlabs/Jet-Nemotron

1. 🧠 Jet-Nemotron 是一系列新型混合架构语言模型，其在保持或超越领先全注意力模型准确性的同时，显著提升了生成吞吐量。
2. 🚀 该模型通过Post Neural Architecture Search (PostNAS)这一高效神经架构探索流程，以及新颖的JetBlock线性注意力机制设计，实现了优化的模型构建。
3. 📈 Jet-Nemotron-2B 在多项基准测试中表现出色，与Qwen3等模型相比，在长上下文生成任务上实现了高达53.6倍的吞吐量加速和6.1倍的预填充加速，并显著降低了LLM架构探索成本。
本文提出了一种名为 Jet-Nemotron 的新型混合架构语言模型家族，其目标是在显著提高生成吞吐量的同时，达到或超越现有最先进的全注意力（full-attention）模型的准确性。Jet-Nemotron 的开发核心是一种新颖的神经架构探索流程，称为 Post Neural Architecture Search (PostNAS)，它能有效进行模型设计。

核心方法学：PostNAS
PostNAS 与以往方法不同，它首先从一个预训练的 full-attention 模型开始，并冻结其多层感知机（MLP）权重，从而实现对注意力块设计的有效探索。这一策略显著降低了训练成本和数据需求。PostNAS 流程包含四个关键步骤：
1. Full Attention Layer Placement and Elimination（全注意力层放置与消除）：
  ○ 动机：保留少量 full-attention 层对于在 MMLU、数学推理和检索等复杂任务上保持高精度至关重要，但最佳放置位置尚不明确。
  ○ 方法：作者引入了一种新颖的方法，通过训练一个 once-for-all super network 来自动学习 full-attention 层的最佳放置位置。在训练过程中，每次迭代随机采样一个子网络，并使用特征蒸馏损失（feature distillation loss）进行训练。训练完成后，通过 beam search 确定在给定约束（例如，两个 full-attention 层）下的最优放置。搜索目标取决于任务，例如 MMLU 最小化损失，数学和检索任务最大化准确率。
  ○ 发现：不同注意力层对不同能力（如 MMLU 和检索）的贡献不同，且并非所有层都同等重要。学习到的放置策略显著优于均匀放置策略。
2. Linear Attention Block Selection（线性注意力块选择）：
  ○ 动机：确定 full-attention 层的放置后，需要选择最适合该设置的线性注意力块。
  ○ 方法：作者评估了包括 RWKV7、RetNet、Mamba2、GLA、Deltanet 和 Gated DeltaNet 在内的六种现有最先进的线性注意力块，从准确性、训练效率和推理速度等方面进行系统评估。
  ○ 发现：Gated DeltaNet 表现出最佳的综合准确性，这归因于其结合了 Data-Dependent Gating Mechanism 和 Delta Rule，前者动态控制模型是更关注当前 token 还是历史状态，后者使用当前 token 的信息增量更新历史状态以节省有限的状态记忆。
3. New Attention Block Design (JetBlock)（新注意力块设计）：
  ○ 动机：现有线性注意力块通常使用静态卷积核，缺乏动态适应输入特征的能力。
  ○ 方法：作者提出了一个新的线性注意力块 JetBlock，它将 dynamic convolution 集成到线性注意力中。JetBlock 使用一个 kernel generator 模块根据输入特征动态生成 causal convolution kernels，并将其应用于 value (V) tokens。同时，它移除了 query (Q) 和 key (K) 上冗余的静态卷积，简化了计算。
  ○ 发现：JetBlock 相比以往的线性注意力块（如 Gated DeltaNet）在数学推理和检索任务上显示出更高的准确性，同时保持了相似的效率。
4. Hardware-Aware Architecture Search（硬件感知架构搜索）：
  ○ 动机：传统上，模型参数量被用作效率指标，但这与实际硬件上的生成效率并不直接相关。
  ○ 方法：作者将生成吞吐量作为选择架构超参数（如 key/value dimension 和 number of attention heads）的直接目标。在固定 KV cache 大小（这是影响长上下文和长生成吞吐量最关键的因素）的前提下，通过小范围的网格搜索来寻找最优配置。
  ○ 发现：KV cache 大小是影响长上下文和长生成吞吐量最关键的因素。当 KV cache 大小不变时，具有不同参数量的模型表现出相似的生成吞吐量。通过这种搜索，最终配置在保持与原始设计相当的生成吞吐量的同时，通过增加参数量提高了准确性。
关键成果：Jet-Nemotron 模型家族
Jet-Nemotron-2B 模型在综合基准测试（包括 MMLU(-Pro)、常识推理、数学推理、检索、编码和长上下文任务）中，与 Qwen3、Qwen2.5、Gemma3 和 Llama3.2 等领先的 full-attention 模型相比，表现出相当或更优的准确性。
● 效率提升：在 NVIDIA H100 GPU 上，当上下文长度为 64K tokens 时，Jet-Nemotron-2B 实现了高达 53.6 倍的生成吞吐量加速和 6.1 倍的预填充（prefilling）加速。在 256K 上下文长度下，预填充速度提升 6.14 倍，解码速度提升 53.6 倍。
● 准确性超越：Jet-Nemotron-2B 在 MMLU 和 MMLU-Pro 上的准确性甚至超过了最近先进的 MoE full-attention 模型，如 DeepSeek-V3-Small 和 Moonlight（尽管它们总参数高达 15B，激活参数为 2.2B）。
● 架构优势：Jet-Nemotron 家族通过显著减少 full-attention 层的数量和更小的 KV cache，实现了卓越的推理效率。例如，Jet-Nemotron-2B 采用两个 full-attention 层和两个 sliding window attention (SWA) 层，其余层替换为 JetBlock，而 Jet-Nemotron-4B 采用三个 full-attention 层和七个 SWA 层。
● 训练成本降低：PostNAS 通过重用预训练 LLM，降低了 LLM 架构探索的成本和风险，加速了语言模型架构设计的创新

## CAD(CoreAttnDisaggr)
Efficient Long-context Language Model Training by Core Attention Disaggregation

https://arxiv.org/abs/2510.18121 2025.10.20 CMU, UCSD, 阶跃

1. 📖 针对长上下文大型语言模型（LLM）训练中，核心注意力（core attention, CA）计算与模型其他组件间的负载不平衡问题，本文提出了一种核心注意力解耦（Core Attention Disaggregation, CAD）技术。
2. 💡 CAD将无参数的CA计算从其他组件中分离，并调度到独立的注意力服务器资源池，通过利用CA的无状态性和可组合性，实现了token划分与动态重批处理，从而达到近乎完美的负载均衡。
3. 🚀 该技术在名为DistCA的系统中实现，结合了就地AttnServer、乒乓式执行机制以隐藏通信开销和通信感知调度器，在多达512块H200 GPU和512K上下文长度下，将端到端训练吞吐量提高了1.35倍。

**核心问题与挑战** 
现有系统通常将核心注意力（即参数无关的 softmax(QK⊤)V 计算）与其他组件共置。然而，在长上下文场景下，核心注意力计算量呈二次方增长，而模型其余部分的计算量呈近线性增长。这种计算复杂度的不匹配导致负载不平衡，进而引发数据并行（Data Parallelism, DP）和流水线并行（Pipeline Parallelism, PP）训练中的“掉队者”（straggler）问题。例如，文档打包（document packing）策略虽然提高了吞吐量，但不同长度文档的组合会导致注意力计算量差异显著，即使总token数相同，也会产生高达1.34-1.44倍的减速。
现有的解决方案，如可变长度数据块（variable-length data chunk）尝试平衡计算量但牺牲了内存平衡，导致部分GPU激活内存膨胀；按文档上下文并行（per-document context parallelism, CP）虽然能平衡计算和内存，但会引入显著的all-gather通信开销，并可能因小分片而降低核函数效率，且无法缓解PP掉队者问题。
核心注意力解耦 (CAD) 方法
CAD基于两个关键观察：
1. 无状态性（Statelessness）：核心注意力不包含可训练参数，只存储少量每行softmax统计信息，因此其平衡问题可简化为计算密集型任务的调度。
2. 可组合性（Composability）：核心注意力计算可以在token粒度上被分解为任意长度的分片。每个分片可以独立计算，并且来自不同文档（或DP/PP副本）的分片可以被重新批处理（re-batched）以形成单个高利用率的注意力核函数调用（例如Flash Attention）。核函数的吞吐量主要取决于聚合的token总数，而非其原始文档来源。
CAD将核心注意力从模型其余部分中解耦，并在一个专用于CA计算的资源池（称为“注意力服务器” attention servers）上独立调度。注意力服务器接受任意划分的文档分片的核心注意力任务（CA-tasks）作为计算请求。运行时动态地将这些CA-tasks批处理成大型融合核函数调用，并调度到任何注意力服务器上，从而实现近乎完美的负载均衡，同时避免内存不平衡。与CP不同，CAD完全解耦了核心注意力与模型其余部分的并行化方式，消除了DP和PP中的掉队者。

基于Megatron-LM修改，实现了AlltoAll；对比了WLB-LLM

**局限性**
● 虽然DistCA采用就地注意力服务器来提高内存利用率，但如果内存需求已满足，专用GPU池可能进一步减少计算时间，并提供更好的容错和性能隔离。
● 调度器目前限制CA-task为具有完整K、V上下文的Q分片，未来可考虑更灵活的上下文范围。
● 通信成本估算模型可能过于悲观，未来可考虑已驻留在目标设备上的K、V状态。
● 在34B模型的4D并行实验中，内存碎片化和频繁的PyTorch垃圾回收导致了运行时开销，限制了DistCA的性能，未来计划通过静态内存分配和CUDA Graphs解决。

## 训练表征变化
Tracing the Representation Geometry of Language Models from Pretraining to Post-training  
https://arxiv.org/abs/2509.23024 McGill University & UC Berkeley, 2025.9.27

这项研究通过测量语言模型表示的有效秩（RankMe）和特征谱衰减（𝛼ReQ），揭示了预训练过程中一致的非单调三阶段几何演化：先是快速表示崩溃的“warmup”阶段，随后是维度扩展并伴随n-gram记忆的“entropy-seeking”阶段，最后是各向异性整合并显著提升下游任务性能的“compression-seeking”阶段。
🧠 研究发现，这些几何阶段源于交叉熵优化在偏斜token频率和表示瓶颈（d ≪ |𝒱|）下的基本相互作用，其中“entropy-seeking”阶段与短上下文记忆相关，而“compression-seeking”阶段则促进了长上下文理解和泛化能力。
🔄 后训练阶段进一步改变了几何结构：SFT和DPO驱动“entropy-seeking”动态以整合特定指令或偏好数据，提高内部分布性能但降低了分布外（OOD）鲁棒性；而RLVR则诱导“compression-seeking”动态，增强了奖励对齐但减少了生成多样性。

<img width="1436" height="851" alt="image" src="https://github.com/user-attachments/assets/2a1faaa2-b962-47e2-bba4-08568e249d09" />

## HedgeSpec
Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs

https://arxiv.org/abs/2510.20064v1 2025.10.22 Rice, AWS，UCSD

1. 🧐 该研究提出了HedgeSpec，一个用于大语言模型推测解码中在线选择草稿模型的框架，其核心创新在于能够通过一次目标模型验证获取所有候选草稿模型的“全面信息反馈”，从而将问题从Multi-armed Bandit转化为全信息在线学习。
2. 🚀 HedgeSpec基于在线学习算法，提供了可证明的“无悔”理论保证，其遗憾值与草稿模型数量呈对数关系，实现了相对于现有基于Bandit方法的指数级性能提升。
3. ✨ 实验结果表明，HedgeSpec在LLaMA-3.1-8B和Qwen-3-8B/32B等模型及多样化数据集上显著优于EAGLE3和BanditSpec等现有基线，尤其在长推理链场景下表现出色，并展现了对领域分布变化的强大鲁棒性。

## TALE
TALE: Token-Adaptive Low-Rank KVCache Approximation with Reconstruction Elimination
https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.39/133612 2025.10.9

https://github.com/thnkinbtfly/TALE

1. ✨ 本文提出了一种名为TALE的创新型KVCache低秩近似压缩策略，旨在解决大型语言模型在处理长输入序列时KVCache内存需求日益增长的瓶颈。
2. 💡 TALE通过token自适应地应用不同秩进行压缩、采用惰性近似以保留完整信息，并引入无重建设计以避免代价高昂的重新计算，同时可与多级量化结合使用。
3. 🚀 实验结果显示，TALE在Llama-3.1-8B模型上实现了9.1倍的KVCache大小缩减，对GSM8K等复杂任务的性能影响微乎其微，并在长上下文场景中将延迟降低了高达2倍。

## CAGE: QAT梯度优化
CAGE: Curvature-Aware Gradient Estimation For Accurate Quantization-Aware Training

https://arxiv.org/abs/2510.18784
1. 🌟 本文提出了CAGE（Curvature-Aware Gradient Estimation）方法，通过引入曲率感知校正项增强了直通估计器（STE）的梯度，旨在弥补量化感知训练（QAT）与原生训练之间的精度差距。
2. 💡 CAGE基于QAT的多目标优化视角，平衡损失最小化与量化约束，定义了量化优化的Pareto最优解，并在平滑非凸设置下提供了强收敛保证。
3. 🚀 实验表明，CAGE在W4A4等低比特量化场景下，能为Llama模型恢复超过10%的量化损失增量，显著提升了模型有效容量和性能，验证了曲率感知梯度校正在弥合性能差距方面的关键作用。
CAGE (Curvature-Aware Gradient Estimation) 是一种新颖的量化感知训练 (QAT) 方法，旨在弥补低比特量化训练与全精度训练之间的精度差距。该方法通过引入一个曲率感知校正项来增强标准的 Straight-Through Estimator (STE) 梯度，以抵消量化引起的损失增加。

## Not bits equal
Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models
https://arxiv.org/abs/2510.10964 2025.10.13 KraftOn 威斯康辛等

https://github.com/krafton-ai/not-all-bits-are-equal 尚未开源 基于vLLM
1. 📝 本研究表明，推理模型的内存优化策略并非一概而论，而是高度依赖于模型的有效尺寸（effective size）。
2. 🚀 对于有效尺寸小于8-bit 4B的模型，将内存优先分配给模型权重以提升容量更为高效，而更大的模型则通过延长生成长度和启用并行推理获得更显著的收益。
3. 💾 KV Cache压缩对于推理模型至关重要，其中逐出（eviction）策略在有效尺寸小于8-bit 8B的模型上优于量化（quantization），而大模型上两者则表现相当。
  a. vLLM A100
本文深入探讨了推理模型在固定内存预算下，如何优化内存分配以最大化准确率的问题。研究发现，传统的4比特量化作为非推理模型和零样本任务的内存最优选择，在推理模型中不再普遍适用，因为其Key-Value (KV) 缓存可能成为主要的内存瓶颈。例如，一个4比特量化的Qwen3-4B模型，其权重占用2.49 GB，但生成32k令牌所需的KV缓存高达4.42 GB，约为权重的1.8倍。本文通过在AIME25和GPQA-Diamond两个基准上进行超过1,700种推理场景的系统实验，揭示了内存优化策略的高度依赖于模型规模（或称为“有效尺寸”，即参数量乘以每权重比特数）。

## QeRL
QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs
https://arxiv.org/abs/2510.11696 2025.10.13 NVIDIA, MIT等

https://github.com/NVlabs/QeRL

1. 🚀 QeRL 是一种量化增强的 LLM 强化学习框架，它通过结合 NVFP4 量化与 LoRA 微调，旨在解决 LLM 强化学习中高资源消耗和训练效率低的问题。
2. 💡 该框架的核心发现是，量化噪声能有效增加策略熵以增强探索，并且通过引入自适应量化噪声 (AQN) 机制，能够动态调整噪声水平以进一步优化探索过程。
3. ⚡ 实验结果表明，QeRL 在 rollout 阶段实现了超过 1.5 倍的加速，使得在单个 H100 80GB GPU 上训练 32B LLM 成为可能，并获得了比 16-bit LoRA 和 QLoRA 更快的奖励增长及更高的最终准确性，甚至在数学基准测试上匹配了全参数微调的性能。

## ThinKV
ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models
https://arxiv.org/abs/2510.01290 2025.10.1 NVIDIA

1. 🤔 大型推理模型（LRMs）在生成长输出时，其Key-Value (KV) 缓存会迅速增长并大量消耗GPU内存。
2. 😮 为解决此问题，ThinKV提出了一个思绪自适应的KV缓存压缩框架，它利用注意力稀疏性将链式思绪分解为推理、执行和转换等不同类型，据重要性进行混合量化与驱逐。
3. 🚀 ThinKV还设计了Continuous Thinking (CT) 内核以高效复用被驱逐token的内存槽，最终在保留不足原始KV缓存5%内存的同时，实现了近乎无损的准确性，并将推理吞吐量提升高达5.8倍。
为了实现 ThinKV，该研究设计了一个扩展 PagedAttention 的 kernel，以实现对被驱逐 token 内存槽的有效重用，从而消除了内存整理（compaction）的开销。在 DeepSeek-R1-Distill、GPT-OSS 和 NVIDIA AceReason 等模型上的数学和编程基准测试中，ThinKV 在使用不到原始 KV 缓存 5% 的情况下实现了接近无损的准确性，并相对于 SoTA 基线提高了高达 5.8 倍的推理吞吐量。

## vAttention
vAttention: Verified Sparse Attention

https://arxiv.org/abs/2510.05688  2025.10.7 UCB等针对LLM 长序列生成的动态稀疏attn，核心是动态组合两大类稀疏方法Top-k类 + 均匀采样，达到1+1>2的精度效果。
https://github.com/xAlg-ai/sparse-attention-hub  和HF集成

里面有AIME25 但论文里只有AIME24的结果：https://github.com/xAlg-ai/sparse-attention-hub/tree/main/benchmark/AIME2025 
现有稀疏注意力方法（如近似 top-k、top-p 和基于采样的估计）在近似全注意力方面存在根本局限性：它们无法在不同 attention head 和查询向量之间提供一致的近似效果，最关键的是，它们缺乏对近似质量的理论保证，这限制了其在实际应用中的部署。
vAttention 的核心洞察在于 Top-k 和随机采样是互补的：当注意力分数由少数几个 token 主导时，top-k 表现良好；而当注意力分数相对均匀时，随机采样能提供更好的估计。
基于这一观察，并利用采样的统计学保证，vAttention 引入了首个实用的稀疏注意力机制，能够提供用户指定 $(\epsilon, \delta)$ 的近似精度保证（因此得名“verified”，可验证）。通过组合 top-k 和采样，vAttention 的表现优于单独使用两者，实现了卓越的质量-效率权衡。

## FSA

FSA通过“选中注意力（selected attention）”GPU kernel的循环次序互换与配套的内核分解（在线 softmax 统计预计算 + 独立归约 kernel）以及非连续批处理优化，消除了NSA在小 GQA 组时必须做的填充（padding）带来的额外加载与计算，显著降低内存访问与FLOPs，使NSA在主流小 GQA 组配置（g∈{1,2,4}）和现代GPU上获得稳定核级与端到端加速。

## MR-GPTQ W4A4 FP4
Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization
https://arxiv.org/abs/2509.23202 

https://github.com/IST-DASLab/FP-Quant

https://github.com/IST-DASLab/qutlass 

● vLLM PR: https://github.com/vllm-project/vllm/pull/24440 dense-only
nvfp4的精度最好，mxfp4不如好的int4
hadmard变换会影响nvfp4，但可以显著提升mxfp4精度(仍然比不上nvfp4)
5090量化后kernel算力能提升6x，B200能提升3.6x。
这篇论文深入研究了最近推出的微缩放（microsaling）4比特浮点数格式MXFP4和NVFP4在大型语言模型（LLM）推理后训练量化（Post-Training Quantization, PTQ）中的应用，揭示了其理论潜力与实际性能之间的差距。作者们发现，现有的最先进量化方法在处理FP4格式时面临两个主要问题：(1) NVFP4的小分组大小（group size）使得传统的外点（outlier）缓解技术失效；(2) MXFP4的2次幂尺度量化（power-of-two scale quantization）由于引入了高误差而严重降低了精度。
为了弥合这一差距，论文提出了一种名为Micro-Rotated-GPTQ（MR-GPTQ）的GPTQ量化算法变体。MR-GPTQ通过采用块级（block-wise）Hadamard变换和针对FP4格式的特定优化，将量化过程调整以适应FP4的独特属性。

## 清华SLA
SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

https://arxiv.org/abs/2509.24006

https://github.com/thu-ml/SLA

新观察：
- 注意力权重可以分解为高秩的大权重和极低秩的小权重！DiT的attn 分3部分;
● 少量重要的 高秩-> FA
● 多数低秩 -> linear attn。计算量很小
● 其余的不重要的 忽略
- 仅依靠稀疏注意力或线性注意力单独处理 无法在高质量生成的同时实现高计算效率，而SLA组合可以。
- 通过少量微调步骤，SLA可以显著加速模型而不损失性能，使其特别适合大规模视频生成任务。Wan2.1-1.3b，5090, attn kernel加速13x(vs. full attn FA2, 精度持平); E2E 加速2.2x

## Expected Attention
Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution

https://arxiv.org/abs/2510.00636 NVIDIA

https://github.com/NVIDIA/kvpress 

https://huggingface.co/spaces/nvidia/kvpress-leaderboard
🎯大型语言模型（LLMs）的KV缓存内存消耗是长上下文推理的关键瓶颈，现有压缩方案因无法获取未来注意力分数或完整的注意力矩阵而受限。
💡为解决此问题，本文提出“Expected Attention”方法，通过预测未来查询将如何关注KV对来估算其重要性，并利用LLM激活的分布特性以封闭形式计算期望注意力分数，实现无需训练的缓存压缩。
🚀“Expected Attention”在预填充和解码阶段均表现出优于现有基线的效果，可在高达50%的缓存压缩率下保持性能，甚至在高压缩场景中依然出色；作者还发布了KVPress库以支持相关研究。

现有基于注意力分数的方法面临实际限制，因为未来的注意力分数在压缩时不可用，并且像 Flash Attention 这样的现代实现不具体化完整的注意力矩阵，导致无法访问过去的注意力分数。Expected Attention 旨在克服这些挑战，它利用 LLM 激活的分布特性来为每个 KV 对计算闭式形式的预期注意力分数。
