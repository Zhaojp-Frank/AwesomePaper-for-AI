# AwesomePaper-for-AI
Awesome system papers for AI

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
