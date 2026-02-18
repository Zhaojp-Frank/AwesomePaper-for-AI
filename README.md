# AwesomePaper-for-AI
Awesome or inspiring paper for AI

## TreeTraining
Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse

https://arxiv.org/pdf/2511.00413 快手 首次更新2025.11.1；最后更新2026.2.9

1. 针对Agentic负载（如Terminus，Claude code）LLM训练中多分支、共享前缀的轨迹导致大量冗余计算，提出了Tree Training框架来消除这种冗余。共享的比例大约28%～88%
2. Tree Training的核心组件是Gradient Restoration，它允许每个共享前缀只计算一次，同时通过Tree Packing和引擎重新设计来高效处理大型轨迹树。
3. Qwen30B-A3/32B，Tree Training在真实Agentic轨迹上实现了6.2x的训练加速，且不影响模型质量。
<img width="809" height="525" alt="image" src="https://github.com/user-attachments/assets/68608330-1170-40f3-add0-b808f435c582" />

本文介绍了Tree Training，一个用于加速Agentic LLMs训练的框架。Agentic LLMs的训练通常涉及多轮交互轨迹，这些轨迹因并发工具使用、思考模式（think-mode）、子agent、上下文管理等运行时设计而分支，形成带有共享前缀的树状token轨迹。现有训练方法通常将这些轨迹线性化，并独立处理每个分支，导致前向和后向传播中存在大量冗余计算。

**核心问题与挑战：**
传统的prefix caching方法在推理和训练的前向传播中非常有效，因为它利用了因果attention mask使得相同前缀产生相同的key/value states。然而，在反向传播中，attention操作的转置特性导致梯度具有反因果性。具体来说，对于attention操作 $O = P \times V$，反向传播中 $dV = P^T \times dO$。这里的 $P^T$ 是上三角矩阵（anti-causal），意味着某个token的梯度 $dV_i$ 不仅依赖于其自身的输出梯度 $dO_i$，还聚合了所有后续token的梯度 $dO_j$ ($j \ge i$)。因此，即使两个序列共享相同的前缀，它们不同的后缀也会导致不同的输出梯度 $dO_{suffix}$，从而使得共享前缀部分的梯度 $dV_{prefix}$ 产生分歧，传统的缓存方法无法直接应用于反向传播，因为需要存储所有后缀的信息，这将导致GPU内存爆炸。

**Tree Training框架：**
为了解决这一挑战，Tree Training引入了**Gradient Restoration**机制，以消除反向传播中的冗余计算。该框架还包括**Tree Packing**内存优化策略和重新设计的训练引擎，使其能够原生处理树状结构数据。

**核心方法论详细解释：**

1.  **Gradient Restoration（梯度恢复）**
    *   **基线方法与冗余：** 当前标准做法是将树状轨迹的每个叶节点（即每个完整执行路径）线性化为一个单独的序列。对于具有共享前缀的多个序列，这意味着共享前缀部分会在每个序列的反向传播中被重复计算，造成效率低下。
        *   基线序列化：$X_{base}(i) = \text{Concat}[\text{token}(i), X_{base}(\text{child}(0, i)), \text{token}(i), X_{base}(\text{child}(1, i)), \dots, \text{token}(i), X_{base}(\text{child}(m, i))]$
    *   **Tree Training序列化：** 我们的方法仅将共享前缀 $\text{token}(i)$ 连接一次，并在所有子节点中重用。
        *   我们的序列化：$X_{ours}(i) = \text{Concat}[\text{token}(i), X_{ours}(\text{child}(0, i)), X_{ours}(\text{child}(1, i)), \dots, X_{ours}(\text{child}(m, i))]$
    *   **梯度等价性：** 目标是确保我们的方法在单次计算共享前缀时，产生的梯度与基线方法（重复计算）所累积的梯度在数学上是等价的。对于任意线性变换 $Y = X \times \text{weight}$，其梯度 $d\text{weight} = X^T \times dY$。为了保持等价性，需要满足以下条件：
        *   $P^T \times dY_{ours_P} = P^T \times (\sum_{i=1}^n dY_{base_{P_i}})$
        *   $\sum_{i=1}^n S_i^T \times dY_{ours_{S_i}} = \sum_{i=1}^n S_i^T \times dY_{base_{S_i}}, \forall i \in [1, n]$
        其中，$P$ 是前缀token，$S_i$ 是第 $i$ 个后缀token，$dY_{ours_P}$ 和 $dY_{ours_{S_i}}$ 分别是我们方法中前缀和后缀的输出梯度，$dY_{base_{P_i}}$ 和 $dY_{base_{S_i}}$ 分别是基线方法中第 $i$ 个序列的前缀和后缀的输出梯度。
    *   **核心梯度补偿条件：** 通过分析，上述条件可以简化为对输出梯度 $dY$ 的要求：
        *   $dY_{ours_P} = dY_{base_{p1}} + dY_{base_{p2}} + \dots + dY_{base_{pn}}$
        *   $dY_{ours_{S_i}} = dY_{base_{S_i}}, \forall i \in [1, n]$
    *   **跨操作传递性：**
        *   **线性操作：** 由于线性操作是逐点计算的 ($dX_i = dY_i \times \text{weight}^T$)，如果上述梯度补偿条件成立，那么输入梯度 $dX$ 也能保持等价性。这意味着梯度修正具有传递性。
        *   **Attention操作：** 类似地，对于Attention操作中的 $dV$ 计算，如果输出梯度 $dO$ 满足补偿条件，那么 $dV$ 也能保持等价性。
        *   **其他逐点操作（如RoPE）：** 对于 $Y_i = f(X_i)$ 形式的逐点操作，反向传播中的梯度 $dX_i = dY_i \cdot \frac{\partial Y_i}{\partial X_i}$。为了保持传递性，必须确保 $\frac{\partial Y_{ours_i}}{\partial X_{ours_i}} = \frac{\partial Y_{base_i}}{\partial X_{base_i}}$。这意味着在Tree Training中，必须恢复token的原始位置ID，以确保位置编码（如RoPE）的计算与基线方法一致。
    *   **实现细节：**
        *   **Shared Prefix Attention Mask：** 在前向传播中，引入修改后的因果掩码，以确保在packed的树状数据中，不同轨迹的token之间不会相互泄露信息，同时允许共享前缀的正确attention计算。这基于FlashAttention V3实现。
        *   **Position Embedding：** 将packed序列中的token位置ID映射回其在原始轨迹中的位置ID，以保持位置信息的一致性。
        *   **Gradient Scaler（梯度缩放器）：** 这是梯度恢复的关键。在反向传播中，计算出每个共享节点被多少个下游序列重用（即其“tree-scale”系数）。然后，将该节点的梯度乘以其对应的tree-scale系数。例如，如果一个前缀节点被5个轨迹重用，其梯度将乘以5。这补偿了因跳过冗余计算而“丢失”的梯度贡献，从而确保总梯度累积与独立处理每个序列时在数学上是等价的。

2.  **Tree Packing（树状打包）**
    *   **必要性：** 单个agentic轨迹树可能过大，无法完全载入GPU内存。
    *   **目标：** 在内存限制 $C$ 下，将大型计算树分割成多个内存可行的子树，同时最大化共享前缀的重用，从而最小化总训练成本。
    *   **复杂性：** 理论上的最优分割是一个NP-hard问题（结合了动态规划和bin packing），计算成本过高。
    *   **启发式DFS算法：** 本文采用了一种贪婪的启发式深度优先搜索（DFS）算法。该算法线性扩展，并遵循三个原则：
        1.  优先分配最深的叶子节点（它们对总轨迹长度贡献最大）。
        2.  在每个子树内，将深度相似的叶子节点分组，以提高打包同质性。
        3.  以深度优先顺序遍历树，当累积长度超过容量 $C$ 时，启动新的遍历以创建新的子树。
    *   **单路径与多路径打包（附录A）：**
        *   **单路径：** 每次训练步选择一个内部节点 $u$ 作为共享前缀，其 $L(u)$（从根到 $u$ 的长度）和 $R(u)$（$u$ 下所有叶子的总残余长度）的总和必须小于 $C$。这种方法可能无法达到最大重用。
        *   **多路径：** 泛化到可以同时激活多个共享路径，共享子轨迹可以分支并重叠。这通过一个动态规划框架实现，其中每个子树生成一组候选状态，父状态通过“Lift”（传播子状态）和“Bin packing”（组合 lifted demands）操作构建。尽管理论最优，但其复杂性高（状态空间指数增长，bin packing是NP-hard）。因此，实践中采用启发式算法。

**性能与正确性：**
*   **无偏性：** Tree Training不会引入训练偏差，因为每个全局batch都是一个自包含的树，数据混洗仅发生在完整的树样本之间，不破坏树内部结构。树状结构源自单个rollout，并在一个梯度累积步骤内处理。
*   **指标：**
    *   **Potential Overlap Ratio (POR)：** 衡量共享前缀固有的冗余度，即 $1 - \frac{N_{tree}}{N_{X_{base}(\text{root})}}$，表示理论上的计算重用上限。
    *   **Effective Reuse Ratio (ERR)：** 衡量在内存约束下实际实现的计算重用效率，即 $1 - \frac{N_{pack}}{N_{X_{base}(\text{root})}}$。
*   **实验结果：**
    *   在包含思维模式（think mode）等真实agentic轨迹数据集上，Tree Training在Qwen3-32B（密集模型）上实现了平均6.3倍的训练加速，在Qwen3-30B（MoE模型）上实现了平均6.2倍的训练加速，捕获了理论潜力（POR约6.5倍）的95%以上。
    *   训练损失曲线与基线完美重叠，数值偏差小于1%，严格验证了数学等价性。
    *   合成数据集实验表明，加速效果随POR的增加而单调增强，在理想情况下（完整树能放入GPU内存）可达8.7倍加速。
    *   额外引入的内存开销（attention masks、original position IDs、gradient scalers）与最小激活内存占用相比可以忽略不计。
    *   在Terminal Bench 2.0上，使用Tree Training在完整轨迹树上进行训练（开启think-mode）的模型性能显著优于仅在单一最长轨迹上训练的基线模型，得分从20.9提升到28.8。


## AREAL-DTA
AREAL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning

https://arxiv.org/pdf/2602.00482 2026.1.31 港科 清华大学  蚂蚁等

1. 为优化LLM RL的**训练环节中**，因大量**rollout 序列共享长令牌前缀**而导致的计算效率低下问题。
2. 通过采用深度优先搜索 (DFS) 策略动态遍历 rollout 前缀树，一次只具现化一个根到叶的路径来高效复用计算并大幅减少内存占用，同时结合负载均衡的分布式批处理机制实现多 GPU 扩展。
3. Qwen3-8b模型RL训练任务（Taubench）8.3x训练吞吐提升（RL E2E 2.3x），峰值显存用量-50%以上，同时保持了训练稳定性。

<img width="739" height="328" alt="image" src="https://github.com/user-attachments/assets/1dd07ded-ada5-4754-a116-edf0800c6233" />

REAL-DTA通过引入一种深度优先搜索（DFS）的动态计算策略，有效利用了prefix共享，并克服了扩展性挑战。它还包含一个load-balanced的分布式batching机制，以在多GPU环境中高效扩展。

**核心方法论——动态DFS遍历：**

AREAL-DTA将rollout序列集合视为一个prefix tree $$T$$。其核心在于一个DFS遍历算法，它在Transformer-based policy model训练的forward和backward passes中动态执行：
1. 栈维护： AREAL-DTA维护一个栈，代表从prefix tree根节点到当前访问节点的路径（即当前prefix）。栈中包含：(i) 当前prefix的token序列；(ii) 由当前policy model为该prefix生成的相应中间状态（即Transformer的Key/Value Cache，KV cache）。
2. Push中间节点： 沿着DFS路径下行时，当访问到prefix tree的中间节点，将其push到栈中。这意味着扩展当前prefix并为新token执行forward computation，从前一个prefix的model state继续。具体来说，它利用前缀的cached KV state，将新token输入policy model，计算其log-probabilities，并更新扩展prefix的KV cache。新的KV cache被append到栈中。这一过程有效重用了prefix token的计算。
3. 访问叶子节点： 当DFS遍历到达对应于完整序列$$s_i$$的叶子节点时，栈中包含完整的token序列$s_i$及其forward pass结果。此时，立即计算损失$$L(s_i)$$（例如，通过对正确token的负对数似然求和或应用RL reward），并将损失梯度注入到该序列的输出端，从而启动针对branch $$s_i$$的backward pass。这种“即时回传”策略确保一旦序列的forward pass完成，其计算图就不需要保留在内存中。
4. Pop中间节点： 处理完叶子节点后，DFS遍历沿着该branch进行backward propagation，然后从prefix tree中pop出节点。通过pop操作，梯度从先前的$$L(s_i)$$传播到当前branch的policy model计算中。关键在于，当backpropagate通过一个在prefix tree中是分支节点（对应多个序列共享的token segments）时，来自所有这些序列的损失梯度将累积在该prefix节点和模型的参数中。AREAL-DTA通过按DFS遍历顺序依次执行每个branch的backward pass来正确处理这一点：共享prefix节点将接收多次梯度贡献（每个对应原始序列$s_i$的后代叶子节点一次），并在DFS遍历时对这些贡献进行求和。一旦完成当前prefix tree叶子节点计算的梯度回传，其对应的token和相关activations会从栈中pop出，回退到父级prefix state。此时，对应于已pop token的节点在计算图中变得不必要，其activations和任何临时梯度可以安全删除。然后继续DFS到下一个同级branch，使用仍然保留的prefix state。
空间复杂度分析： 这种DFS遍历机制使得AREAL-DTA仅存储从根节点到叶子节点的当前路径的KV cache和少量栈状态。这意味着峰值内存使用量与最长序列的长度（即prefix tree中最长路径的长度）成比例，而非所有序列的总token数量。这显著优于现有方法的二次方内存增长。

**系统优化：**
为了进一步减少内存使用和提高计算效率，AREAL-DTA实施了一系列系统优化：
1. 内存高效梯度计算： 通过交错进行forward和backward steps，动态构建和增量释放部分计算图。一旦中间节点被pop，表示其所有子孙叶子节点都已遍历，该节点可立即回传并释放。
2. 长rollout序列的Chunked Backpropagation： 对于非常长的序列（数万个token），AREAL-DTA将backpropagation沿序列分成多个chunks（例如2048个token）。在每个chunk之后进行部分backpropagation以释放activations。实现方式是使用存储的prefix KV cache和outputs，重新计算该chunk的forward activations以重建计算图，然后立即backpropagate并释放该chunk的图。这引入了额外的forward计算成本，但通过合理选择chunk长度，可以摊销开销。
3. 避免叶子节点KV cache计算： 在DFS遍历中，AREAL-DTA不对prefix tree的叶子节点存储KV cache。由于只有prefix节点才能作为chunked backpropagation执行期间的计算图重建锚点，叶子节点的KV cache永远不会被其他计算重用。
4. 确定最优DFS遍历顺序： 采用贪婪算法优化DFS序列，目标是：(i) 最大化共享prefix的重用以最小化forward passes次数；(ii) 平衡backward pass segment长度以避免频繁的短步。这提高了内存效率和运行时稳定性。
  
**负载均衡并行化：**
为实现大规模RL训练，AREAL-DTA引入了负载均衡的分布式batching策略。在异步RL框架中，rollouts不断生成并分发到多个trainer GPU。
1. 问题定义： 将N个rollout序列分发给K个trainer GPU，每个GPU处理其group内的prefix tree $T_j$。目标是最小化各group最大成本$$C = \min \max_{j=1}^K C(T_j)$$，其中$$C(T_j)$$定义为$$T_j$$中所有节点token长度之和。
2. 算法： AREAL-DTA采用词典顺序（实际上是DFS顺序）对N个序列进行排序，形成一个单prefix tree。然后将这个有序列表划分为K个连续的segments。这种划分方式最大程度地保留了prefix共享，因为连续的DFS segments会将具有共同prefix的序列聚类在一起。分割只会引入少量额外的token复制。通过对最大允许成本进行二分查找（binary search），结合贪婪检查，AREAL-DTA可以在$$O(N \log C(T))$$时间内找到近似最优的连续划分。
  
**实验评估：**
AREAL-DTA在τ^2-bench工作负载和PPO算法上进行了评估，使用QWEN-3模型（1.7B、8B）作为RL training workflows，并与AREAL基线系统进行比较。
1. 端到端RL训练性能（RQ1）：
  - 奖励曲线： 在训练步数维度上，AREAL-DTA和AREAL展现出相似的奖励曲线，表明AREAL-DTA的设计不会影响异步RL训练的稳定性。
  - 训练吞吐量： 在累计实际RL训练时间维度上，AREAL-DTA显著提高了RL训练效率。对于1.7B模型，AREAL-DTA实现1.28倍的端到端吞吐量提升；对于8B模型，提升高达2.28x。这主要归因于AREAL-DTA卓越的policy model训练性能，使其能将更多GPU资源分配给rollout生成阶段，从而实现更好的整体加速。
2. 组件消融研究（RQ2）：
  - Backward pass优化： 对比标准训练（带activation checkpointing的“Dense+CKPT”），AREAL-DTA的“Tree”方法实现了6.59x的平均加速。进一步应用避免叶子节点KV cache计算（“Cut Tail”或“CT”）和最优DFS遍历顺序（“DFS”）两项优化，加速分别提升至7.53倍和7.74倍。在充足显存预算下，增加backward chunk size（“LB”）可将加速提升至8.31x。AREAL-DTA展现出显著的内存效率（峰值GPU内存减少超过50%），避免了Activation Recomputation等辅助内存优化技术的需求，从而避免了其引入的计算开销，也解决了更大模型（4B/8B/14B）中的OOM问题。
  - 数据并行均衡算法： 相较于传统的贪婪数据并行均衡算法，AREAL-DTA精心设计的负载均衡算法（基于DFS顺序和$$C(T)$$的连续划分）将系统性能退化降低了11.93%。


## SWE-World
SWE-World: Building Software Engineering Agents in Docker-Free Environments

https://arxiv.org/pdf/2602.03419 人大 2026.2.3

https://github.com/RUCAIBox/SWE-World

1. 为解决软件工程 (SWE) agents 训练中**基于 Docker 环境的可伸缩性限制**， 提出 SWE-World，一个无 Docker 的框架，旨在**通过用学习到的替代模型**取代物理执行环境，。
2. SWE-World 由一个**轻量级 Sandbox**、预测执行结果的 SWE-World **Transition Model** (SWT) 和模拟测试报告与奖励的 SWE-World **Reward Model**(SWR) 组成，实现了**完全无 Docker 的 SFT 和 RL 优化**。
3. SWE-World 显著提升了 agents 在 SWE-bench Verified 上的性能，例如将 Qwen2.5-Coder-32B 的**解析率从 6.2% 提高到 55.0%** (RL)，并在结合测试时扩展 (TTS) 后达到 68.2%，从而大大降低了 SWE 研究的资源和基础设施成本。
传统上，SWE 智能体严重依赖于基于 Docker 的物理执行环境来获取反馈，例如程序执行结果和测试报告。然而，这种**范式资源密集、维护困难**，并极大地限制了智能体训练的可扩展性。

<img width="737" height="408" alt="image" src="https://github.com/user-attachments/assets/983829f3-5592-4c13-a216-3c2519a79b1e" />

## SWE-Master
SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training

https://arxiv.org/pdf/2602.03411 人大 2026.2.3 

https://github.com/RUCAIBox/SWE-Master

1. 🌟 SWE-Master是一个**开源且完全可复现的软件工程Agent后训练框架**，旨在通过系统化的优化方法，从初始能力有限的基座模型中挖掘并释放强大的**长周期SWE任**务解决能力。
2. 🚀 该框架系统性地探索了Agent开发的完整流程，包括教师轨迹合成、数据精炼、长序列SFT、基于真实执行反馈的RL，并创新性地引入了**基于语言服务器协议（LSP）的IDE级代码导航工具**，显著提升了Agent的效率和代码理解能力。
3. 🎯 在SWE-bench Verified基准测试中，SWE-Master使用Qwen2.5-Coder-32B**实现了61.4%的解决率，通过Test-Time Scaling（TTS@8）进一步提升至70.8%** 优于现有开源基线。
   
<img width="716" height="305" alt="image" src="https://github.com/user-attachments/assets/90360f05-9991-4139-b383-61a2a40aec0b" />
<img width="737" height="231" alt="image" src="https://github.com/user-attachments/assets/1a72cc7b-067d-476e-8add-0e1d62318fd6" />


## SkyRL-Agent
SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent

https://arxiv.org/pdf/2511.16108 伯克利 AnyScale等 2025.11.20

GitHub: https://github.com/NovaSky-AI/SkyRL

HuggingFace: https://huggingface.co/NovaSky-AI/SA-SWE-32B

1. SKYRL-AGENT 是一个高效的框架，旨在用于多轮LLM代理的强化学习训练与评估，提供高效的异步调度、轻量级工具集成和灵活的后端互操作性。
2. 框架引入了优化的异步管道调度器，实现1.55倍加速，并结合增强工具的训练方案（如基于AST的搜索工具），使SA-SWE-32B在SWE-Bench Verified上达到39.4% Pass@1，同时将成本降低两倍以上。
3. SA-SWE-32B 在Terminal-Bench、WebArena等通用Agent任务中表现出良好的泛化能力，且SKYRL-AGENT通过支持深度研究、内存和计算机使用等不同类型的Agent，展示了其出色的可扩展性。
   
## SpeContext retrieval稀疏加速decode
SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs

https://arxiv.org/pdf/2512.00722 清华大学 王钰；上海交大 戴国浩等 ASPLOS26

1. 目标：长序列推理decode阶段的KV cache问题，如耗时layer-wise retrieval、新生成KV的完全保留以及序列长度增加导致的性能下降。
2. SpeContext提出了一种算法与系统协同设计，利用Distilled Language Model (DLM，基于EAGLE-3)的轻量级retieval head信息筛选Q，并通过async prefetch的弹性加载和adaptive memory magt优化效率。
3. 最大8b dense模型，A100上比FlashInfer加速1.3x~1.6x (32K->2K) 或 1.6x～2.2x（2K->32K）。没有测试CoT 例如AIME25（完全忽视背景里强调的TTS）；没有测MoE模型。

<img width="948" height="475" alt="image" src="https://github.com/user-attachments/assets/0f612763-76f9-4a06-9feb-99067b0441ee" />

1. **引言与背景问题**
LLM的测试时缩放（test-time scaling）在通过逐步生成（step-by-step generation）增强模型性能方面被证明是有效的。尽管现有的KV缓存优化在长上下文输入场景中表现良好，但在长上下文decode场景中仍面临以下挑战：
1. 耗时的layer-wise retrieval: 现有方法在推理阶段的每一层都需要进行KV对的检索和加载，由于数据依赖性，这引入了显著的同步开销，且其开销随模型深度线性增长，导致高达60%的延迟开销。
2. 新生成KV缓存的完整保留: 现有为长上下文输入设计的工作通常在Prefill阶段对KV缓存进行预处理，但在解码阶段完全保留新生成的KV对以避免重复昂贵的处理，这在KV缓存持续增长的长上下文推理场景中效率低下。
3. 序列长度微小增加导致的性能显著下降: 现有的卸载策略在推理前确定，无法适应自回归解码过程中序列长度的动态增长，导致性能在序列长度轻微增加时下降超过80%。
  
2. **核心方法**
本研究的核心洞察是：检索算法的目标在于高效地与LLM在信息焦点（information focus）上对齐。受LLM知识蒸馏中输出分布对齐目标的启发，作者提出了一种新颖的范式，即利用蒸馏语言模型（Distilled Language Model, DLM）作为检索算法。

从信息论的角度来看，通过互信息（Mutual Information, $$I(X;Y)$$）和数据处理不等式（Data Processing Inequality, DPI）进行分析，可以证明：如果DLM（学生模型）的输出概率分布$$P_S$$能够很好地近似原始LLM（教师模型）的输出概率分布$$P_T$$，即最小化KL散度$$D_{KL}(P_T||P_S)$$，那么DLM的内部表示$$R_S$$也必须捕获与教师模型相似的、对上下文$$C$$重要的信息。这意味着DLM在给定相同输入时，其信息焦点（即对结果贡献最大的Token）与原始LLM高度相似。因此，DLM可以有效识别重要的上下文信息。

3. **SpeContext 的算法与系统协同设计**
基于上述洞察，SpeContext 在算法、系统和编译三个层面进行了设计：

3.1. 算法层面：**轻量级检索头** (Lightweight Retrieval Head)
- 挑战: 直接使用完整的DLM进行检索会引入约20%的额外开销。
- 洞察: DLM的主要作用是识别重要Token，这主要由注意力权重决定。实验发现，DLM与原始LLM在head-level注意力权重上的相似度更高，且存在大量冗余操作。
- 方法: 设计轻量级retrieval head，通过修剪DLM中的冗余部分（如FFN和LM_Head），仅保留Embedding模块和QK投影权重。该检索头在LLM推理前处理相同的输入，并利用其attn weight的计算结果进行head-level的稀疏Token选择。
- 实现细节:
  - 该检索头支持MHA、GQA， MQA和MLA（Multi-Head Latent Attention）等主流注意力机制。
  - MHA: 直接根据每个注意力头的权重进行选择，并通过torch.gather加载对应KV缓存。
  - GQA/MQA: 由于KV头数量< Query头，且组内共享KV，因此在组内进行element-wise最大值操作，生成group-level的注意力权重，然后基于此进行选择，从而适配物理KV缓存的结构。
  - MLA: 与MHA类似，但选择的是低维的潜在表示$$c$$缓存，并在计算时将其映射到高维空间。
- 优势: 实现超过90%的参数量削减，同时准确捕捉重要信息。
  
3.2. **系统层面：**异步预取数据流与弹性加载 (Asynchronous Prefetch Dataflow via Elastic Loading)
- 挑战: 现有KV检索方法因数据依赖性导致同步开销，且GPU内存带宽有限，KV传输延迟远超LLM推理延迟。
- 洞察: 检索头在LLM推理前就已识别重要KV对，消除了推理过程中的数据依赖性。同时，研究发现相邻Token生成所选的重要Token存在高度重叠（>80%）。
- 方法:
  - 异步预取: 利用多个CUDA Stream，将KV缓存的预取与LLM的计算并行化，消除数据依赖性导致的同步瓶颈。
  - 弹性加载策略（Elastic Loading）: 利用相邻生成中KV选择的上下文相似性。只加载与上一次生成相比发生变化的KV对（即$$ S_{now} - S_{last}$$），从而大幅减少数据传输量（90%），最大限度地重用已在GPU上的KV缓存。
    
3.3. **编译层面**：自适应内存管理 (Adaptive Memory Management)
- 挑战: 长上下文推理中序列长度动态增长，固定（全上GPU或全下CPU）的KV缓存卸载策略效率低下。
- 方法:
  - 理论内存模型: 构建一个综合考虑LLM模型大小、硬件规格和推理工作负载的理论内存开销模型。
    - 总内存需求公式（当所有KV缓存都在GPU上时）：
    $$M_{all} = M_{Model} + M_{KV} = 1.3(M_O + M_D) + 4R(L + 1 + \alpha)SHD$$
    其中，$$M_O$$是原始LLM模型大小，$$M_D$$是DLM模型大小，$$R$$是请求数量，$$L$$是LLM层数，$$\alpha$$是与GQA/MQA相关的额外内存因子（这里代表了Repeat KV操作导致的额外层级内存需求），$$S$$是当前序列长度，$$H$$是KV头维度，$$D$$是每个头的维度。系数1.3代表模型大小的130%用于考虑运行时内存。
    - 部分KV缓存卸载到CPU时的内存需求公式：
    $$M_{part} = 1.3(M_O + M_D) + 4R [(L_{GPU} + 1 + \alpha)S + L_{CPU}B]HD$$
    其中，$$L_{GPU}$$和$$L_{CPU}$$分别是保留在GPU和卸载到CPU的层数，$$B$$是加载到GPU进行计算的KV缓存预算。
  - 自适应内存管理系统: 在编译阶段预先计算一系列序列长度阈值（Algorithm 1）。在推理阶段，系统根据当前序列长度动态调整KV缓存的存储位置（Algorithm 2）。当序列长度增加并超过预设阈值时，系统会逐步将LLM层中的KV缓存卸载到CPU，从而释放GPU内存，以确保GPU HBM的最大化利用，实现最佳性能。
    
4. **系统架构 (Architecture)**
SpeContext的架构设计如下：接收推理请求后，在编译阶段，自适应内存管理系统依据理论模型计算序列长度阈值并初始化KV缓存内存。在自回归推理过程中，retrieval head识别关键KV对并获取其index。这些索引立即传递给异步prefetch，进行差异计算和KV预取，并通过弹性加载机制与LLM的原始推理并行执行，从而实现GPU计算和CPU-GPU数据传输的重叠。

6. 实验评估
- 硬件平台: 云端NVIDIA A100-80GB，边缘RTX 4060 Laptop (8GB)。
- 模型: Llama3.1-8B, DeepSeek-R1-Distill-Llama-8B, Qwen3-8B (云端)，Reasoning-Llama-3.2-1B (边缘)。
- 基准: LongBench（长上下文输入场景），LongWriter（长上下文推理场景）。
- 基线: Huggingface (Eager), FlashInfer (Full Attention), Quest, ClusterKV, ShadowKV (Sparse Attention)。
- 准确性: 在KV预算达到1K时，SpeContext在LongBench上的准确性与Full Attention相当，甚至超越部分基线。在LongWriter基准测试中，SpeContext在多个维度上的平均得分与Full Attention接近，而其他稀疏KV方法由于其预处理机制的限制，在长上下文推理中表现不佳。
- 性能:
  - 云端环境（多请求）: 相比HF (Eager)，吞吐量提升24x；相比FlashInfer，吞吐量提升2.2x。
  - 边缘环境（单请求）: 相比Huggingface (Eager)，速度提升10x；相比ShadowKV，速度提升1.17倍。
- 开销: 检索头的内存占用极小（Llama3-8B或Qwen3-8B仅约60MB），DLM的训练时间可控（EAGLE-3提供的DLM仅需RTX 3090 GPU训练24小时）。
- 消融研究: 证实了轻量级检索头、异步预取数据流和自适应内存管理三大贡献的独立加速效果。
   
## QuoKA
QUOKA: Query-Oriented KV Selection For Efficient LLM Prefill

https://arxiv.org/pdf/2602.08722 2026.2.9 高通 

1. 无需训练且与硬件无关的稀疏注意力算法，特别是**结合chunked prefill**（同时附带加速decode）。
2. 通过优先选择与平均查询（mean query）余弦相似度较低的查询来近似全注意力行为，并通过查询子选择（Query Subselection）、余弦相似度评分（Cosine-Similarity Scoring）和组感知聚合（Group-Aware Aggregation）来高效地子选择键值对（KV pairs）。兼容GQA attn kernel。
3. 在Needle-In-A-Haystack、LongBench等多个基准测试中实现了接近基线的准确性，A100上60k长度attn kernel加速最高5x、TTFT减少3x（decode加速2.2x：缩小了Q/Key），实际参与attn的KV cache减少88%。
  1. 但KV cache物理所需显存容量没有减少，仍保留全量

<img width="888" height="378" alt="image" src="https://github.com/user-attachments/assets/d956f565-e47f-422e-8fb5-d621bb6c2e31" />

**1. 问题背景与挑战**
近期部署越来越多地采用 **chunked prefill**，将输入划分为块以改善调度和利用率。然而，由于底层注意力机制的二次复杂度，chunked prefill 仍然计算成本高昂。稀疏注意力算法通过识别和利用注意力中的稀疏性来降低复杂度。现有的稀疏注意力方法主要分为两类：
*   **Pattern-based approaches (基于模式的方法)**：对 $QK^\top$ 施加固定的稀疏模式。这些方法通常通过 kernel-level 优化实现加速，但由于动态计算图和 chunked prefill 下 KV cache 内存带宽开销，其益处有限，且依赖自定义 kernel 限制了跨异构硬件的可移植性。
*   **Query-dependent approaches (依赖查询的方法)**：直接在 KV cache 上操作，自适应地为给定查询子选择最相关的 KV 对。这些方法与优化过的 kernel 兼容并提供强大的可移植性。然而，它们主要为decode阶段单query设计。在 prefill 阶段，需要为许多Q同时选择相关的 KV 对(chunk prefill)，这会导致显著的性能下降。

**2. QUOKA 核心思想与方法**
其核心观察是：**“与平均查询 (mean query) 的余弦相似度较低的查询会与更多的键 (keys) 进行更强的交互，并且对最终注意力 logits 的贡献最大。”** QUOKA 利用这一观察，保留一小部分有代表性的查询，并子选择它们强烈交互的 KV 对。
<img width="902" height="486" alt="image" src="https://github.com/user-attachments/assets/b9830446-8583-4a90-bcd4-45ced4e93506" />

QUOKA 的实现分为三个阶段（如 Algorithm 1 所示），并集成到 chunked prefill 过程中（如 Algorithm 2 所示）：

**2.1. Query Subselection (查询子选择)**
*   **目的**：减少冗余，只保留信息量最大的查询。
*   **原理**：观察到与平均查询 $M_Q$ 余弦相似度较低的查询倾向于与大多数键广泛对齐，而接近平均查询的查询则集中在少量共享的键组上。
*   **方法**： QUOKA 通过计算每个查询 $q$ 与平均查询向量 $M_Q$ 之间的角度距离来识别这些查询。具体而言，对每个查询 $q$，根据 $-CosSim(M_Q, q)$ 进行排名，并保留排名前 $N_Q$ 的查询。较高的 $-CosSim(M_Q, q)$ 值（即较低或更负的 $CosSim(M_Q, q)$ 值）表示查询与 $M_Q$ 的角度距离越大。
*   **理论支撑**：Theorem 1 形式化了这一直觉：
    $$CosSim(M_Q, q^*) \leq 1 + \alpha_q \beta_q - 0.5\alpha_q^2 - 0.5\beta_q^2$$
    其中 $\beta_q = CosSim(k, q^*)$ 且 $\alpha_q = CosSim(M_Q, k)$。该定理表明，如果一个查询 $q^*$ 对键 $k$ 有很强的注意力 ($\beta_q$ 大)，同时 $M_Q$ 与 $k$ 的相似度较低 ($\alpha_q$ 负值大)，那么 $q^*$ 与 $M_Q$ 的余弦相似度将很低，从而 $S_q = -CosSim(M_Q, q^*)$ 较高。这表明选择与平均查询“相距较远”的查询，能够捕获那些对注意力分布贡献最大的查询。
*   **效果**：通过子选择，QUOKA 保留了在几何上与键对齐且在注意力中占主导地位的查询。

**2.2. Cosine-Similarity Scoring (余弦相似度评分)**
*   **目的**：评估保留查询与键之间的相关性。
*   **原理**：现有方法常使用点积 $QK^\top$ 来评分，但其依赖于尺度且在聚合下不稳定。
*   **方法**：QUOKA 计算 $S = CosSim(Q, K)$。余弦相似度将向量归一化为单位长度，提供了一个有界、几何感知且近似 Softmax 注意力权重的代理。
*   **效果**：消融实验表明，余弦相似度比点积显著提高了子选择质量。

**2.3. Score Aggregation (分数聚合)**
*   **目的**：将分数聚合以选择最终的 KV 对。
*   **跨查询聚合**：对查询轴，QUOKA 采用最大值 (maximum) 聚合，而非平均值。这是因为平均值可能掩盖稀有但重要的查询-键交互，而最大值能保留这些异常值。图 3 的重尾分布支持了这一选择。
*   **跨 KV 组聚合**：对于 GQA (grouped-query attention) 轴，由于头级别的重要性通常是相关的，QUOKA 简单地取平均值。此外，在计算分数之前对 K 和 Q 进行归一化，可以通过预聚合 (pre-aggregation) 来降低计算和内存成本，从而提高了与现代架构的兼容性和效率。

**2.4. 集成到 Chunked Prefill 中**
*   **过程**：对于每个传入的块 $X_i$，QUOKA 使用 Algorithm 1 中的步骤子选择活动 KV token。得到的键和值子集随后被传递给该块的注意力计算。
*   **公式**：
    $$Attn(X_i) = Softmax\left(Q_i\left[ K_i | K_{<i} \right]^\top/\sqrt{d} + M_i\right)\left[ V_i | V_{<i} \right]$$
    其中 $K_{<i}$ 和 $V_{<i}$ 是来自所有前序块的键和值。QUOKA 通过减少每个块的 KV 预算来降低计算和内存传输成本。
*   **稀疏化选择**：
    $$I = \text{topk}(f(Q, K), BSA), \hat{K} = \text{gather}(K, I), \hat{V} = \text{gather}(V, I)$$
    这里 $f(Q, K)$ 是 QUOKA 的打分和聚合过程。

**3. 实验结果**
QUOKA 在多项长上下文基准测试和模型上进行了广泛验证，包括 Needle-In-A-Haystack (NIAH)、LongBench、RULER 和 Math500，以及 Llama3、Qwen3、SmolLM、GPT-OSS 等模型。
*   **准确性**：在所有基准测试中，QUOKA 都实现了接近基线 (dense attention) 的准确性，并且显著优于现有的稀疏注意力方法。在 LongBench 上，即便在小预算下，QUOKA 也能保持最小的准确性下降，平均性能优于其他方法 10-20%。在 RULER 上，分数比基线高 10-20%。
*   **延迟减少**：
    *   attention 模块级别：在 Nvidia GPU 上实现 5 倍加速，在 Intel Xeon CPU 上实现近 7 倍加速。
    *   端到端 time-to-first-token (TTFT)：实现 3 倍的降低。
    *   KV 对数量：每个注意力评估使用的 KV 对减少 88%。
*   **泛化能力**： QUOKA 成功泛化到多种解码器-only LLM (Llama3, Qwen3, SmolLM, GPT-OSS) 和 RoPE/NoPE、MoE-based LLMs。
*   **超参数鲁棒性**：在 $BSA$ (selective attention budget)、$BCP$ (prefill chunk size) 和 $N_Q$ (number of queries) 等关键超参数变化时，准确性缓慢下降，表明其在不同硬件约束下仍能保持高效性。

**4. 与相关工作的比较**
*   **动态查询依赖稀疏注意力**：现有方法如 SampleAttention、LOKI 等主要为生成阶段设计，在 prefill 阶段对多查询进行朴素聚合会导致性能下降。QUOKA 则通过查询子选择和几何感知机制，更有效地处理多查询 prefill。
*   **KV cache 淘汰**：KV cache 淘汰通过移除低显著性 KV 对来减少内存占用，但多数为单查询生成设计。KV cache 淘汰与 QUOKA 互补，未来可考虑结合。
*   **Kernel-level 稀疏注意力**：这类方法依赖于 CUDA 等专用实现，缺乏可移植性，且在 chunked prefill 下可能因重复的 kernel 调用和内存传输而效率降低。QUOKA 则兼容标准密集 kernel，避免了硬件和运行时依赖。

## HySparse
HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing

https://arxiv.org/pdf/2602.03560 2026.2.3 小米 罗福莉团队

1. HySparse提出一种**混合稀疏Atte**架构，通过将全Attn层与多个稀疏Attention**层交错**（1:11），并从前一个全Attention层中**选择重要Token并跨层共享KV cache**，解决稀疏Attention的**代理选择和内存限制**问题。
2. 💡 该架构利用修改后的**FlashAttention来精确识别重要Token**，并允许稀疏Attention分支重用全Attention的KV cache，同时为滑动窗口Attention（SWA）分支维护一个独立的本地KV cache，以平衡全局和局部信息处理。
3. 7bdense/80B-A3b实验GQA，表现优异，显著超越Full Attention和混合SWA基线，并在**80B MoE模型中实现了近10倍的KV cache存储减少**，同时保持甚至增强了长上下文建模能力。

该研究提出了一种名为 HySparse 的混合稀疏注意力架构，旨在解决当前大型语言模型 (LLM) 中自注意力机制随序列长度二次方增长所导致的**计算和 KV cache 成本**高昂问题。
<img width="757" height="489" alt="image" src="https://github.com/user-attachments/assets/d150823b-b35d-4824-b429-72faa9559b9a" />
<img width="695" height="490" alt="image" src="https://github.com/user-attachments/assets/e25e9a7d-0632-4f72-b1d0-6f32795475ae" />

现有稀疏注意力方法面临两大挑战：一是**依赖代理进行 token 选择**，引入额外复杂性且精度次优；二是虽然能**减少计算量，但通常无法节省 KV cache 内存**。

HySparse通过在每**个full attention 层后交错多个 sparse attention**层来克服这些限制。其核心思想是，**sparse attention 层的 token 选择和 KV caches 直接来源于前一个 full attention 层。**

**核心方法 (Methodology)：**

1.  **HySparse 架构总览 (HySparse Overview)**
    *   HySparse 架构用重复的混合块替代了标准的 Transformer 主干，每个混合块包含一个 full attention 层和 N 个连续的 sparse attention 层。
    *   在这些 sparse attention 层中，重要的 token 索引和 KV caches 直接从同一混合块中前一个 full attention 层派生。
    *   每个 sparse attention 层还带有一个额外的 Sliding Window Attention (SWA) 分支，该分支维护一个小的局部 KV cache，以增强短程建模能力。
    *   **两个分支（稀疏全局和 SWA 局部）的输出通过 sigmoid 门控进行融合**。

2.  **Full Attention 层 (Full Attention Layers)**
    *   Full attention 层计算标准的 scaled dot product self-attention。
    *   为了识别后续 sparse attention 层的重要 token，模型需要获取注意力分数。由于完全实例化注意力矩阵成本过高，HySparse 修改了 FlashAttention 算法。
    *   具体而言，FlashAttention 已经在其在线 softmax 过程中计算了注意力 logits **的行最大值。HySparse 利用这一中间结果**，通过存储和适当重新缩放，导出**块级最大注意力分数** $S \in \mathbb{R}^{t \times \lceil t/B \rceil}$。
    *   块级最大注意力分数 $S_i^t$ 定义为：
        $$S_i^t = \max_{i' \in \text{B}_i}\left(\frac{\exp(\mathbf{q}^{\top}_t \mathbf{k}_{i'}/\sqrt{d})}{\sum_{j=1}^t \exp(\mathbf{q}^{\top}_t \mathbf{k}_j/\sqrt{d})}\right)$$
        其中 $B$ 是注意力分数输出的块大小，$B_i = \{(i-1)B+1, \dots, \min(iB, N)\}$ 是块索引 $i$ 处的列 token 索引集合。
    *   通过对 $S$ **应用TopK 操作**，选择 key-block 索引 $I$，这些索引将被后续 sparse attention 层重用。
    *   在 Grouped-Query Attention (GQA) 下，模型会在每个 query 组内聚合 $S$ (通过组级最大值)，使得同一组内的所有头共享相同的稀疏索引，从而提高稀疏注意力 kernel 的效率并减少索引开销。

3.  **Sparse Attention 层 (Sparse Attention Layers)**
    *   每个 sparse attention 层包含两个注意力分支，它们作用于相同的 query，但使用不同的 KV 源。
    *   **Block Sparse Attention 分支**：
        *   使用由输入 $\mathbf{x}_t$ 经过线性投影 $W_{q'}$ 得到的 query $\mathbf{q}'_t$。
        *   通过连接从前一个 full attention 层共享的 KV cache 中根据索引 $I$ 选择的 key 和 value 块来形成 $\tilde{K}, \tilde{V}$：
            $$\tilde{K}, \tilde{V} = \text{concat}(\{\text{K/V}[ (j-1)B+1: jB] \}_{j \in I})$$
        *   计算注意力输出 $\tilde{\mathbf{o}}_t$：
            $$\tilde{\mathbf{o}}_t = \sum_{i=1}^k \frac{\exp(\mathbf{q}'^{\top}_t \tilde{\mathbf{k}}_i/\sqrt{d})}{\sum_{j=1}^k \exp(\mathbf{q}'^{\top}_t \tilde{\mathbf{k}}_j/\sqrt{d})} \tilde{\mathbf{v}}_i$$
    *   **SWA 分支**：
        *   同样使用 query $\mathbf{q}'_t$ (与 Block Sparse Attention 分支共享相同的 query 投影 $W_{q'}$，但有独立的 $W_{k'}$ 和 $W_{v'}$ 投影)。
        *   维护其自身的轻量级 KV cache，用于处理大小为 $w$ 的局部滑动窗口。
        *   计算注意力输出 $\mathbf{o}'_t$：
            $$\mathbf{o}'_t = \sum_{i=t-w+1}^t \frac{\exp(\mathbf{q}'^{\top}_t \mathbf{k}'_i/\sqrt{d})}{\sum_{j=t-w+1}^t \exp(\mathbf{q}'^{\top}_t \mathbf{k}'_j/\sqrt{d})} \mathbf{v}'_i$$
    *   **分支融合**：最后，两个分支的输出通过 sigmoid 门控融合：
        $$\tilde{g}_t, g'_t = \sigma(W_{\tilde{g}/g'} \mathbf{x}_t)$$
        $$\mathbf{o}_t = \tilde{g}_t \odot \tilde{\mathbf{o}}_t + g'_t \odot \mathbf{o}'_t$$
        其中 $\sigma$ 是 sigmoid 函数，$W_{\tilde{g}/g'}$ 是门控权重。
    *   研究发现，SWA 分支拥有独立的 KV cache 对于保持模型表达能力至关重要，它可能作为局部信息路径，捕获短程一致性，而 full attention 共享的 KV cache 则更优化于全局检索。

**实验结果与贡献 (Experimental Results and Contributions)：**

*   **性能提升**：HySparse 在 7B dense 和 80B Mixture-of-Experts (MoE) 模型上进行了评估，在所有设置下均持续优于 full attention 和 hybrid SWA 基线模型。
*   **KV Cache 效率**：在 80B MoE 模型中，尽管 49 层中只有 5 层使用了 full attention（即 KV cache 减少了近 10 倍），HySparse 仍实现了显著的性能提升。HySparse 在 KV cache 成本方面并未比 hybrid SWA 基线增加额外开销。
*   **长文本能力**：HySparse 在 RULER 基准测试中展现出与 full attention 相当甚至超越的长文本能力，尤其是在多 key/value 和推理密集型子集上表现出色。
*   **消融研究 (Ablation Study)**：
    *   **Intra-layer Hybridization with SWA**：移除 sparse attention 层内的 SWA 分支会导致准确率显著下降，表明即使有高质量的稀疏选择，专用的滑动窗口路径对于建模短程一致性仍很重要。
    *   **Cross-layer KV Cache Sharing Configuration**：对 SA 和 SWA 两个分支都共享 KV cache 会严重降低准确率。只有 SA 分支共享来自 full attention 的 KV cache，而 SWA 分支维护自己的独立 KV cache 时，性能才得以恢复和提升。这证实了 SA 可以安全地重用跨层 KV cache 以节省 GPU 内存，而 SWA 需要自己的 KV cache 以保留强大的局部信息。


## CoMeT 长段记忆周期压缩转换
CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling

https://arxiv.org/pdf/2602.01766 2026.2.3 阿里未来生活

https://anonymous.4open.science/r/comet-B00B/

1. CoMeT (Collaborative Memory Transformer) 提出了一种创新的**即插即用架构和微调方法**，优化prefill和decode计算（可常数），以KV cache显存。
2. 采用双内存系统：基于**FIFO队列的temporary memory用于近期事件**，以及带有**门控更新规则的global memory用于长期依赖**，并通过layer-level pipeline parallelism实现高效**长上下文训练**。
3. Qwen4b/8b模型微调？32k上下文**训练后**，能从1M token序列中准确检索passkey，相比full-attention基线实现**21倍推理加速和10倍内存优化**，并在SCROLLS基准测试及真实世界任务中表现出色。
<img width="759" height="291" alt="image" src="https://github.com/user-attachments/assets/acd7bf3d-b985-41e2-83df-7e2a6ff19f49" />


CoMeT的核心创新在于其协同记忆系统，该系统基于对序列数据块的处理。它采用双记忆系统来管理上下文：
1.  **全局记忆（Global Memory）**：用于管理长程依赖。全局记忆 $G^i_\tau$ 由持久的全局状态 $S^i_\tau$ 派生。为确保参数效率和训练稳定性，状态到记忆的转换通过Residual Low-Rank Adapter（**RLA）模块实现**，定义为 $\text{RLA}(X) = X + W_{\text{up}}(W_{\text{down}}X)$，其中 $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times r}$，$W_{\text{down}} \in \mathbb{R}^{r \times d_{\text{model}}}$，通常 $r=8$。

因此，$G^i_\tau = \text{RLA}(S^i_\tau)$。全局状态 $S^i_\tau$ 的更新机制**引入了一个门控单元，以选择性地整合来自输出读出标记**（readout tokens）$R^{i+1}_\tau$ 的信息，同时**保护关键历史信息不被覆盖**。$R^{i+1}_\tau$ 首先被 RMSNorm 归一化为候选状态 $\tilde{S}^{i+1}_\tau = \text{RMSNorm}(R^{i+1}_\tau)$。然后，全局状态通过以下门控规则更新：$S^{i+1}_\tau = g \odot S^i_\tau + (1 - g) \odot \tilde{S}^{i+1}_\tau$，其中门控值 $g = \sigma(W_g([S^i_\tau; \tilde{S}^{i+1}_\tau]))$。这里 $[;]$ 表示沿特征维度进行拼接，$W_g \in \mathbb{R}^{2d_{\text{model}} \times 1}$ 是一个可学习的权重矩阵，$\sigma$ 是 sigmoid 函数。这种加性更新结构为跨数据块的梯度流提供了更直接的路径。
   
4.  **临时记忆（Temporary Memory）**：用于捕捉近期事件的细粒度信息。临时记忆 $T^i_\tau$ 由固定容量的First-In-First-Out（FIFO）队列管理。新的记忆条目源自输出压缩标记（compression tokens）$C^{i+1}_\tau$。
这些标记在入队前，会先**经过 RMSNorm 归一化，然后使用与全局记忆相同的 RLA 模块**进行转换。FIFO 队列的特性保留了近期数据块信息的时间连续性，当新条目添加时，最旧的条目被丢弃。

在每一层 $i$ 处理第 $\tau$ 个输入数据块时，CoMeT 将全局记忆 $G^i_\tau$ 和临时记忆 $T^i_\tau$ 前置于数据块的隐藏状态 $H^i_\tau$。通过因果自注意力机制，$H^i_\tau$ 可以从两种记忆中检索相关信息以辅助下一个标记的预测。同时，一系列压缩标记 $C^i_\tau$ 被交错在 $H^i_\tau$ 之中，用于捕获细粒度的局部信息。最后，$m$ 个读出标记 $R^i_\tau$ 被附加到序列末尾，以总结数据块最显著的内容。单个Transformer层的整体计算公式为：$H^{i+1}_\tau, C^{i+1}_\tau, R^{i+1}_\tau = TL(G^i_\tau, T^i_\tau, H^i_\tau, C^i_\tau, R^i_\tau)$，其中 $TL$ 表示Transformer层计算。

为了实现高效的超长上下文微调，CoMeT引入了一种新颖的**层级管道并行（layer-level pipeline parallelism）**策略。与传统的上下文并行不同，该策略在层级交错计算和通信。工作节点 $j+1$ 在收到工作节点 $j$ 必要的中间状态后，即可立即开始处理层 $i$，而工作节点 $j$ 同时推进到层 $i+1$，从而显著减少管道气泡并最大化硬件利用率。

实验结果表明，CoMeT在多个方面表现出色：
-   **长上下文外推能力**：在一个32k上下文上进行微调的CoMeT模型，能够从1M标记序列的任意位置准确检索出passkey。
-   **效率**：在1M标记上下文长度下，CoMeT相比全注意力（Full Attention）基线实现了2**1倍的推理加速和10倍**的内存占用减少。其推理内存消耗保持恒定（约10GB），预填充（prefill）延迟与上下文长度呈线性关系，而解码（decode）阶段的每标记延迟稳定在约22ms。
-   **基准测试表现**：在SCROLLS基准测试中，CoMeT超越了其他高效方法，并在总结任务上表现与经过微调的全注意力基线相当。在较短上下文的QA任务（2WikiMQA和HotpotQA）上，CoMeT也匹配了全注意力性能。
-   **实际应用**：在用户行为QA（UQA）任务中，CoMeT超越了xRAG基线2.7个百分点，并显著优于4k截断的全注意力基线。在长上下文Agent任务（Terminal-Bench）中，CoMeT的训练速度比朴素上下文并行快2.7倍，并取得了与全注意力模型相当的性能。
-   **记忆作用分析**：临时记忆对训练域内序列长度的表现至关重要，而带门控的全局记忆是处理超出训练数据长度序列的关键，门控机制对于保护关键信息免受覆盖至关重要。

## TSA
Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection

https://arxiv.org/pdf/2602.03216 2026.2.3 韩国

https://github.com/dongwonjo/Token-Sparse-Attention  待开源

1. 针对**prefill 动态、可逆的token级别稀疏化**机制，通过在压缩空间内执行注意力操作并在之后恢复原始序列维度，有效解决了长上下文LLM中attention的二次复杂性问题。
2. prefill阶段 每层：先选少量token（例如最后5%）作快速attn计算和打分，从而压缩token + 原始残差输入合并。实现每head独立和层级的token选择，并基于Inter-Layer Representation Drift智能地选择进行稀疏化的层，从而适应token重要性的动态变化：。
3. TSA与FlashAttention、FlexPrefill兼容（叠加使用），llama3-8b/mixtral-12b，A100, 128K下最多3.23x提升（在FlexAttn）, 精度通常<1个点，显著改善了精度-延迟权衡。

<img width="759" height="291" alt="image" src="https://github.com/user-attachments/assets/23dd7d2a-abc3-463f-bcca-2a50359c1241" />

现有加速方法通常采用结构化注意力图稀疏化或在特定层永久驱逐token，但这些方法可能保留不相关的token，或因未能考虑token重要性在层间/头间的动态变化而做出不可逆的早期决策。

该论文提出的Token Sparse Attention通过在注意力计算期间将每个头的查询、键和值 (Q, K, V) 压缩到一组减少的token集合 ($L' \ll L$)，然后将输出解压回原始序列维度 ($L$)，从而实现高效的长上下文推理。这种“压缩再解压” (Compress and then Decompress) 的设计允许token信息在后续层中被重新考虑，弥补了永久性token驱逐方法的不足。TSA还在token选择和稀疏注意力之间引入了一个新的设计点。

Token Sparse Attention主要包含两个阶段：压缩QKV和解压注意力输出，并辅以动态token覆盖和稀疏层选择策略。

1.  **动机 (Motivation)**
    *   **层间Token重要性动态变化：** 研究发现，token的重要性在不同层之间显著漂移。图2(a)显示，即使相邻层共享一定比例的重要token，但随着层距离的增加，重叠部分迅速减少。这表明，基于早期层token重要性进行永久性移除，可能会过早地排除在后续层中变得相关的token。
    *   **头内Token重要性异质性：** 图2(b)展示了在同一层内，不同注意力头对token重要性的排名存在差异。多头注意力固有的特性是每个头专注于捕捉不同的上下文关系。因此，统一的、层级的驱逐策略会强制所有头共享相同的token集合，可能丢弃对某些头至关重要的token。
    受这些观察启发，TSA旨在实现跨层和跨头的灵活性，而不是永久性地依赖早期层的决策。

2.  **Token Sparse Attention机制 (Token Sparse Attention Mechanism)**
    TSA旨在选择性地跳过不相关的token计算，同时不永久地将其从序列中移除。
    *   **阶段1：QKV压缩 (Compression for QKV)**
        每个注意力头 $h$ 独立地选择一个token索引子集 $S_H=h$，从而得到一个缩减的序列长度 $L' \ll L$。利用这些索引，从原始的 Q、K 和 V 张量中收集相应的行，以构建压缩后的张量 $\hat{Q}$、$\hat{K}$ 和 $\hat{V}$。这种每个头独立选择的设计直接解决了头间的异质性问题。$\hat{Q}$、$\hat{K}$、$\hat{V}$ 在内存中保持密集和连续，这使得它们能兼容 FlashAttention 等高度优化的硬件感知核或其它专用稀疏注意力实现而无需修改。注意力操作在这些压缩张量上执行，产生一个缩减的输出张量 $\hat{O}$，其中包含所选token的上下文感知表示。这一步将二次注意力成本从 $O(L^2d)$ 降低到 $O(L'^2d)$。
    *   **阶段2：注意力输出解压 (Decompression for Attention Output)**
        在注意力计算之后，压缩后的输出 $\hat{O}$ 使用特定于头的索引集 $S_H=h$ 被散射回一个形状为 $R^{L \times d}$ 的零初始化张量中。这确保了输出维度与原始输入匹配，避免了后续层中的维度不匹配问题。输出张量中未选择的位置保持为零，这在功能上等同于对这些token在注意力图上应用了一个硬掩码 (hard mask)。最后，恢复的注意力输出被添加到残差连接中。**这一步至关重要，因为残差连接保留了来自前一层未选择token的信息**。
    这种可逆设计（“压缩再解压”）在实现计算稀疏化益处的同时，保持了原始序列的结构完整性，允许模型选择性地忽略不相关的上下文以减少计算，同时不断重新评估跨层和跨头的token重要性。

3.  **动态Token覆盖 (Dynamic Token Coverage)**
    TSA自适应地选择推理时的稀疏预算，涉及两个关键决策：(1) 保留多少token；(2) 为每个注意力头保留哪些token。
    *   **计算Token重要性分数：** 首先，为每个注意力头独立估计token重要性。对于给定头 $h$，通过将少量最新查询 (recent queries) 与所有键 (all keys) 进行注意力计算，得到注意力图 $\hat{A}$ 的轻量级代理：
        $$ \hat{A} \leftarrow \text{softmax}(Q[-\text{last q}:]K^T/\sqrt{d}) $$
        token $t$ 的重要性分数 $s_h[t]$ 通过沿垂直轴（即查询的序列长度维度）对注意力权重求和得到，这一步使用 Triton 定制核实现。
    *   **确定保留token数量：** 将头级token分数聚合为层级重要性分布，通过跨头求和并在序列维度上归一化得到。这捕捉了当前层每个token的整体贡献，作为确定稀疏预算的基础。
        我们不是按重要性降序选择token，而是按估计重要性升序对token进行排序，并识别最不重要token的最小集合 $k_{sparse}$，其累积质量超过预定义的覆盖阈值 $\tau$。
        $$ k_{sparse} \leftarrow \text{arg min}_{k \in \{0,...,L\}} \left\{ \sum_{j=1}^k s_l[I[j]] \ge \tau \right\} $$
        其中 $s_l = (\sum_H s_h) / (\sum_L \sum_H s_h[t])$，I是按 $s_l$ 升序排序后的索引。然后计算要保留的token数量 $k_{keep} = L - k_{sparse}$。
    *   **选择保留哪些token：** 一旦确定了层级预算，再根据 $s_h$ 为每个注意力头独立地执行最终的token选择，即选择Top-$k_{keep}$ 的token作为 $S_h$。

4.  **稀疏层选择 (Sparse Layer Selection)**
    并非所有层都适合稀疏化。为了识别那些token表示足够稳定以进行稀疏化的层，引入了一个名为“层间表示漂移” (Inter-Layer Representation Drift) 的指标 $R_\ell$，它通过比较token输入和输出隐藏状态的 $L_2$ 范数来衡量其表示的相对变化：
    $$ R_\ell = E_t \left[ \frac{\lVert h_{\ell+1,t} - h_{\ell,t} \rVert_2}{\lVert h_{\ell,t} \rVert_2 + \epsilon} \right] $$
    其中 $h_{\ell,t}$ 表示层 $\ell$ 中token $t$ 的隐藏状态（即层输入）。较低的漂移值表示层间表示变化较小，暗示token表示更稳定。根据这一观察，我们使用表示漂移作为选择进行token级稀疏化层的标准。定义每个层的归一化漂移排名 $\hat{R}_\ell = \frac{1}{L} \sum_{k=1}^L \mathbf{1}[R_k \le R_\ell]$，并选择 $L_{sparse} = \{\ell \mid \hat{R}_\ell \le \delta\}$ 的层进行TSA。在所有实验中，$\delta=0.5$，即仅对表示最稳定的层应用该方法。此层选择作为每个模型的预处理步骤执行一次。

**实验结果 (Experimental Results)**

LLaMA-3.1-8B-Instruct 和 Mistral-Nemo-12B-Instruct，使用 RULER 和 InfiniteBench 作为主要基准，并在附录中提供 LongBench 和 Needle-in-a-Haystack 的结果。基线包括 FlashAttention、Minference、FlexPrefill 以及token驱逐方法 FastKV 和 GemFilter。

1.  **精度结果 (Accuracy Results)**
    *   在 RULER 基准上，与 FlashAttention、Minference 和 FlexPrefill 结合时，Token Sparse Attention在所有上下文长度下基本保持了底层注意力核的精度，同时提高了注意力效率。例如，在LLaMA-3.1-8B-Instruct上应用于 FlexPrefill 时，平均精度保持不变，而128K上下文的注意力加速从 ×2.44 提高到 ×2.76。
    *   在 InfiniteBench 上也观察到类似趋势，TSA 与所有基线方法的结合只导致微小的精度差异，表明其作为通用加速机制的兼容性。

2.  **效率结果 (Efficiency Results)**
    *   **精度-速度权衡 (Accuracy-Speedup Trade-offs)：** 通过调整 FlexPrefill 的超参数 $\gamma$ 得到的帕累托前沿 (Pareto frontier) 显示，TSA（在 $\tau=0.005$ 时）始终将帕累托前沿向外推，在可比精度水平下实现更高的加速。即使在更高的稀疏度下，精度下降也保持在1%以内，表明TSA能有效识别和移除不相关的token。
    *   **跨序列长度的稀疏度和速度 (Sparsity and Speedup across Sequence Lengths)：** 随着上下文长度的增加，TSA实现的注意力加速持续增加。在128K和256K等注意力计算占据主导地位的长上下文下，速度提升尤为显著。这是因为在长上下文下，平均注意力稀疏度会随之增加。
    *   **延迟和开销分解 (Latency and Overhead Breakdown)：** 额外的开销（包括token评分、索引、QKV压缩和注意力输出解压）在128K上下文长度下，即使在最高稀疏度下，也仅占总注意力延迟的不到11%。
    *   **动态稀疏度与固定稀疏度 (Dynamic Sparsity vs. Fixed Sparsity)：** 在相似的速度下，动态稀疏度（TSA）始终比固定稀疏度实现更高的 RULER 平均精度。在更高的稀疏度下，动态稀疏度在保持精度的同时表现出显著优势。

3.  **与Token驱逐方法的比较 (Comparison with Token Eviction)**
    在相近的效率预算下（FastKV、GemFilter和TSA在128K上下文下达到相似的加速），Token Sparse Attention实现了最高的平均 RULER 精度。这归因于TSA的层级动态预算分配、可逆的交错处理（通过残差路径保留被跳过的token），以及细粒度的、头特定的token集合选择，避免了驱逐方法中严格的统一token约束


## RRAttention
RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference 

https://arxiv.org/pdf/2602.05853 2026.2.5 北大 百度

中文解读： https://mp.weixin.qq.com/s/iXhEwrbgsecmOw88Bs2qUg

1. 现有动态稀疏注意力方法在预处理、全局评估和查询独立性等方面存在固有限制。
2. 提出RRAttention，一种新型prefill动态稀疏attn，通过**head round-robin采样策略**，在**不同的注意力头**（Attention Heads）之间旋转查询（Query）的采样位置(stride S=8/16)同时实现高效**全局模式发现和查询独立性**。
3. llama/qwen 7b/8b模型最大30b-A3b，~50%稀疏，HELMET LLM和Video-MME, **绝对掉点通常>1，128K prefill 加速2.4x**。

<img width="639" height="192" alt="image" src="https://github.com/user-attachments/assets/60f0502e-fafb-4185-ad23-b253e78c5f34" />

<img width="745" height="690" alt="image" src="https://github.com/user-attachments/assets/fbf3e802-9f41-4712-940d-9e856d5ef579" />
<img width="658" height="367" alt="image" src="https://github.com/user-attachments/assets/262bd0b0-c8a9-472a-9eeb-a5c3e24dd871" />


## DeepContext
DeepContext: A Context-aware, Cross-platform, Cross-framework Tool for Performance Profiling and Analysis of Deep Learning Workloads

https://dl.acm.org/doi/pdf/10.1145/3676642.3736127 北卡 AWS等 ASPLOS26

https://zenodo.org/doi/10.5281/zenodo.15589616

1. DeepContext 是一款新型的性能分析工具，旨在解决现有深度学习剖析工具在异构计算环境中缺乏跨栈和跨框架上下文的痛点。
2. 该工具通过其独特的 DLMonitor “shim” 层统一了 Python、深度学习框架、C/C++ 库和 GPU 执行的调用路径，并提供了一个自动性能分析器来给出可操作的优化建议。
3. A100 PyTorch, DeepContext 能有效识别真实世界应用中的性能瓶颈，指导优化实现了 1.06 倍至 1.66 倍的加速，且具有较低的内存开销，展现了其在复杂深度学习工作负载分析中的实用性。

<img width="780" height="274" alt="image" src="https://github.com/user-attachments/assets/01767c94-0847-492b-80de-c827d7ebb769" />
<img width="345" height="328" alt="image" src="https://github.com/user-attachments/assets/03687b85-7dba-43dd-8e50-27a094730390" />

**时间开销**：

- 轻量级模式下（仅 Python 和框架调用路径），DeepContext 在 Nvidia 和 AMD GPU 上的 PyTorch 工作负载中位数开销分别为 1.12\times1.12×1.12\times1.12× 和 1.50\times1.50×1.50\times1.50×；在 JAX 工作负载中位数开销分别为 1.33\times1.33×1.33\times1.33× 和 1.28\times1.28×1.28\times1.28×。

- 综合模式下（包含原生 C/C++ 调用路径），PyTorch 中位数开销分别为 1.50\times1.50×1.50\times1.50× 和 1.90\times1.90×1.90\times1.90×；JAX 中位数开销分别为 1.60\times1.60×1.60\times1.60× 和 1.46\times1.46×1.46\times1.46×。

- 相比之下，PyTorch Profiler 在 Nvidia 和 AMD GPU 上的中位数开销分别为 1.06\times1.06×1.06\times1.06× 和 1.01\times1.01×1.01\times1.01×；JAX Profiler 分别为 1.17\times1.17×1.17\times1.17× 和 1.10\times1.10×1.10\times1.10×。DeepContext 在轻量级模式下的开销与框架原生剖析器相当。对 Llama3 和 Gemma-7B 等工作负载，由于帧统一系统和指标聚合传播机制，DeepContext 的时间开销较高，尤其是在启动大量小内核时。


**内存开销**：DeepContext 的中位数内存开销为 1.00\times \text{-} 2.44\times1.00×-2.44×1.00\times \text{-} 2.44\times1.00×-2.44×，远低于 PyTorch Profiler (1.29\times \text{-} 27.28\times1.29×-27.28×1.29\times \text{-} 27.28\times1.29×-27.28×) 和 JAX Profiler (1.27\times \text{-} 6.98\times1.27×-6.98×1.27\times \text{-} 6.98\times1.27×-6.98×)。DeepContext 通过运行时聚合指标，显著降低了内存开销，使其更适用于长时间运行的工作负载。




## PAT
PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

https://arxiv.org/abs/2511.22333 天津大学 ASPLOS26

https://github.com/flashserve/PAT

1. 为优化decode阶段重复KV缓存加载和GPU效率问题，PAT提出了一种前缀感知的attn kernel，通过pack-forward-merge范式来加速LLM decodeing。
2. 采用启发式pack共享前缀的查询分组以减少冗余内存访问，并设计了资源高效的Multi-Tile kernel、多流转发和长KV分割策略来提升GPU利用率。
3. Cutlass/CuTe实现，vLLM集成，在一些长序列+tool calling场景，PAT vs.SOTA注意力内核，平均可将注意力延迟降低53.5%，并将TPOT降低17.0-93.1%，显著提升了LLM解码性能。

decode阶段通常是内存密集型操作，瓶颈在于从全局内存加载大规模的 KV cache。真实世界的 LLM 工作负载展现出大量分层的共享前缀 (shared prefixes)，例如系统提示、工具/模板和 RAG 文档。现有注意力实现未能充分利用前缀共享：基于 (one-query-per-CTA)”模式会重复加载共享前缀的 KV cache；而“一刀切 (one-size-fits-all)”的 tiling 设计则导致片上资源闲置并因 KV 长度不均而加剧execution bubble。这些问题共同放大了内存带宽压力，并导致解码注意力操作停滞。

PAT 提出了一种前缀感知的注意力内核实现，其核心范式是“pack-forward-merge”。该方法旨在**减少重复的内存访问，并提高资源效率**。

**一、背景与动机**

1.  **LLM 推理与注意力瓶颈：** LLM 推理分为预填充 (prefill) 和解码 (decode) 两个阶段。解码阶段迭代生成 token，每次解码都需要查询 (query) 当前 token 的信息并关注所有先前的键 (key) 和值 (value)。KV cache 的引入减少了重复计算，但随着上下文和输出长度的增加，每次解码步骤都必须从全局内存中获取不断增长的 KV cache 数据到片上内存，使得解码注意力成为内存瓶颈，占总延迟高达68%。
2.  **GPU 执行模型：** GPU 内存层级结构显示，全局内存访问速度远慢于片上内存 (shared memory / L1 Cache 和 Registers)。现有的注意力内核，如 FlashAttention，通过将 KV cache 分割成小 tile 并以流水线方式处理 (一边计算一边异步预取)，试图利用内存层级结构。然而，受限于全局到片上内存带宽不足以及 KV cache 加载导致的低算术强度，优化仍需遵循两项原则：(1) 减少 KV cache 从全局内存的传输量；(2) 充分利用可用内存带宽。
3.  **共享前缀与现有实现缺陷：**
    *   **冗余内存访问：** query-centric attn如FlashAttention、FlashInfer采用“每个查询一个block”策略，导致共享 KV 前缀例如，**多请求共用的系统提示被重复从慢速全局内存加载**，引入 4.3-8.7 倍的冗余 KV cache 流量。
    *   **资源利用率低下：** 现有内核采用固定 tile size 的“一刀切”设计（例如，**m=64, n=32**），忽略了 LLM 工作负载的动态性，导致双重资源低效：
        *   内存浪费 ($I_{mem}$): 当共享前缀的查询数量少于 tile size 时，**CTA 必须填充输入**，浪费共享内存和寄存器。
        *   执行气泡 ($I_{exe}$): 不同 CTA 的 KV 长度差异导致工作负载不平衡，**使SM在执行后期阶段利用率不足**。

**二、PAT 的核心方法：Pack-Forward-Merge 范式**

(1) CTA 内部KV ache 共享：将具有共享 KV prefix的Q打包到同一个CTA 中，以避免冗余全局内存访问。(2) 资源高效内核设计：根据 GPU 架构和 CTA **配置定制内核实现**，以维持高内存带宽利用率并最小化资源浪费。

**1. Pack 阶段 (Pack Scheduler)**
目标是最小化给定解码批次的总全局内存访问。

*   **树状结构块表 (Tree Structure Block Table)：** 将解码批次的 block table 转换为树状结构，每个内部节点代表一个共享 KV 块的前缀，包含属性 $l$ (共享前缀的 KV cache 长度) 和 $s$ (共享该前缀的查询数量)。叶节点对应一个查询。
*   **利润模型 (Profit Model)：**
    *   **节点内利润 (Intra-node profit)：** 将一个非叶节点 $u$ 内的 $s_u$ 个查询打包到一个 CTA 中，相比“每个查询一个 CTA”范式，可减少 $(s_u - 1) l_u d$ 的全局内存访问量（$d$ 为头维度）。开销为 $8 s_u d$（两次 FP32 中间结果读写）。打包一个节点的利润与开销比率 $r = \frac{(s_u - 1) l_u d}{8 s_u d} \ge \frac{l_u}{16}$，通常为正。
    *   **节点间利润 (Inter-node profit)：** 比较两种方案：
        *   **方案 1 (Split)：** 将父节点和子节点拆分到不同的 CTA，利润为 $(s_u - 1) l_u d - 4 s_u d + \sum_i (s_i - 1) l_i d$。
        *   **方案 2 (Merge)：** 将特定子节点 $v_i$ 与父节点 $u$ 合并到一个 CTA，消除其间中间结果。利润为 $(s_u - s_i - 1) l_u d - 4(s_u - s_i)d + \sum_{k \neq i} (s_k - 1) l_k d + (s_i - 1) (l_u + l_i) d$。
        当 $4s_j > l_u$ 时，方案 2 更优，即短共享前缀和足够大的特定子节点查询数量时，合并可获得更高利润。
*   **启发式打包调度器 (TreeHeuristic)：** PAT 使用线性复杂度的启发式算法 (Algorithm 1)，根据利润分析将解码批次打包成 CTAs。它将每个叶节点作为一个独立的 CTA，扫描内部节点的子节点，应用节点间利润模型选择方案，并递归打包子节点。
*   **懒更新 (Lazy Update)：** 为降低调度开销，PAT 采用懒更新策略：(1) 在 block table 不变时重用调度结果；(2) 将调度器移至服务系统并异步运行，与预注意力任务 (如 LayerNorm, QKV projection) 重叠。

**2. Multi-tile Kernel**
针对打包后的 CTAs 选择最优的 Q-tile ($m$) 和 KV-tile ($n$)。

*   **多 tile 内核套件 (Multi-tile kernel suite)：** 通过离线硬件分析和 CTA 约束，预先计算可行的 $(m, n)$ 配置。选择 $(m, n)$ 需满足以下约束：
    *   **寄存器和共享内存约束：** CTA 的共享内存使用量 ($m h b + n h b + m h b'$) 不得超过单 SM 共享内存 ($S_{smem}$)。每个线程的寄存器使用量 ($R_{thr}(m,n)$) 不得超过 $S_{reg\_thr}$，同时并发 CTA 的总寄存器量 ($C \cdot R_{CTA}(m,n)$) 不得超过 $S_{register}$。
    *   **高带宽利用率约束：** 为使全局内存带宽饱和，在途数据量 ($D_{flight} = S C n h b$) 必须大于固有内存延迟 ($L$) 乘以可持续带宽 ($B$)，即 $n \ge \lceil \frac{L B}{S C h b} \rceil$。
    *   **CUTLASS 约束：** 为高效使用 CUTLASS/CuTe MMA，tile size 必须是 2 的幂且至少为 16（或 32 for int8），即 $m, n \in \{2^k | k \in N, 2^n \ge 16\}$。
*   **Tile 选择器 (Tile Selector)：** 在运行时为每个 CTA 分配 $(m, n)$。
    *   **Q-tile ($m$) 选择：** 采用向上取整规则 (round-up rule)，选择满足 $m \ge q$ 的最小 $m$（$q$ 为 CTA 的查询大小），以避免因拆分查询而导致共享 KV cache 的冗余访问。
    *   **KV-tile ($n$) 选择：** 对于长 KV，倾向选择较大的 $n$ 来减少每个 SM 的并发 CTA 数量 ($C$)，从而增加每个 CTA 可用的带宽，并减小执行气泡。对于短 KV，选择较小的 $n$ 可以缩短最后一个 tile 的计算时间，避免计算气泡。

**3. Forward 阶段 (Multi Kernel Forward)**
缓解内核执行气泡 ($I_{exe}$)。
*   **多流转发 (Multi-Stream Forward)：** 为每种独特的 $(m, n)$ 配置创建独立的 CUDA 流。调度器将相同配置的 CTAs 分组并放入对应流，使得不同流并行运行，从而重叠内核启动开销，并通过内核并行化减轻执行气泡。
*   **长 KV 分割 (Long KV Split)：** 将 KV 长度超过批次平均 KV 长度的 CTA 分割成等份，以缩短最晚完成 CTA 的时间，提高整体 SM 利用率。

**4. Merge 阶段 (Output Merge)**
轻量级内核使用 “online softmax” 结合每个查询的中间结果，生成最终输出。它从全局内存加载中间结果（最大分数、log-sum-exp 累加器、部分值加权和），通过 online softmax 进行归约和归一化，然后将所有头连接起来，将最终查询输出写回全局内存。

**三、实现与评估**

PAT 作为 vLLM 的后端插件实现，利用 Cutlass/CuTe 编写内核。KV cache 管理依赖 vLLM 的 PagedAttention。
基线：FlashAttention, FlashInfer；FastTree, RelayAttention, RelayAttention++ (扩展 RelayAttention 以利用 vLLM 风格的 KV cache 重用)

**工作负载：**
    *   **内核性能：** 合成解码批次，模拟不同共享前缀结构和批量设置。
    *   **端到端：** 真实世界痕迹 (toolagent, conversation)，使用 Qwen3-8B 和 Llama3-8B 模型。
**主要结果：**
    *   **内核性能：** 在有共享前缀的配置下，PAT 比 FlashAttention、FlashInfer、FastTree、RelayAttention 和 RelayAttention++ 分别提速高达 21.5 倍、11.7 倍、3.2 倍、11.9 倍和 5.7 倍。平均而言，PAT 比查询中心内核的注意力延迟减少 67.8% 和 52.1%。即使没有共享前缀，PAT 仍能获得 1.6% 的延迟降低。
    *   **端到端性能：** 在相同请求速率下，PAT 的平均 TPOT (Time Per Output Token) 比 RelayAttention++ 降低 17.2–68.1%，比 FlashAttention 降低 17.0–89.5%，比 FlashInfer 降低 32.2–93.1%。TTFT (Time To First Token) 也有显著降低。
    *   **消融研究 (Ablation Study)：** 验证了 PAT 各组成部分的有效性：内存导向的打包调度器优于计算导向或简单打包；多 tile 内核相比固定 tile size 显著提升性能；多流转发有效缓解了执行气泡。
    *   **开销分析：** 由于懒更新机制和异步执行，打包调度器的开销相对于预处理延迟可忽略不计。


## TPLA
https://arxiv.org/pdf/2508.15881 北大 腾讯

https://github.com/fxmeng/TransMLA

1.  为解决 MLA 在 TP 下 KV 缓存全量复制导致的内存效率低下问题，提出了 TPLA (Tensor-Parallel Latent Attention)，一种结合 MLA (Multi-Head Latent Attention) 高效 KV 缓存压缩与 Tensor Parallelism (TP) 的方法，
2.  TPLA核心方法是在多个设备之间对latent representation和每个attention head的输入维度进行分区，每个分片独立执行attention计算，然后**通过一个AllReduce操作合并结果**；应用基于 Hadamard 或** PCA 的重参数化技术来减轻跨分片干扰**，使其能直接加载 MLA 预训练模型，并通过PD感知策略（主要是decode）。
3.  TPLA比MLA精度有2-5点**下降**！在DeepSeek-V3 和 Kimi-K2（1T）模型上 故意转换成BF16凸显显存压力，故意去掉MoE部分凸显速度，最终说实现了一个纯attn的decode吞吐加速（1.79x 和 1.93x）。

<img width="690" height="306" alt="image" src="https://github.com/user-attachments/assets/3140206d-c7f6-4810-b017-10fe2728b37a" />


## M2XFP 软硬协同设计
M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization

https://arxiv.org/pdf/2601.19213 上交 冷静文团队 2026.1.28 ASPLOS26

https://github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26

1. 提出M2XFP，一种**元数据增强的Microscaling数据格式**，旨在解决现有MXFP4等低位量化格式在大型语言模型（LLMs）中**因共享缩放因子导致的显著精度下降问题**。
2. M2XFP采用**算法-硬件协同设计**，针对激活值和权重分别引入混合元数据策略：对**动态激活值使用Element-level Extra Mantissa**，对静态权重使用Subgroup-level Extra Mantissa结合自适应共享缩放，并通过**轻量级硬件单元**实现高效支持。
3. 实验结果表明，M2XFP在LLM基准测试中将精度损失相较于MXFP4平均降低70.63%，**相较于NVFP4降低37.30%**，同时实现了高达1.91倍的速度提升和1.75倍的**能效节省**。


论文的核心在于一个算法-硬件协同设计，该设计基于灵活的元数据，并采用在线量化和简易编码。M2XFP的提出源于对现有MX量化误差的深入分析。研究发现，MXFP4等格式精度下降的主要原因是其**使用的Power-of-Two（E8M0）共享缩放因子**无法精确匹配块内的最大值，导致关键数据点的量化误差。现有的解决方案，如采用更精细的FP8缩放因子（NVFP4）或定制数据类型，要么**存在动态范围受限问题**，要么引入过高的硬件开销。鉴于缩放因子和数据类型的设计空间已趋于收敛且存在固有局限，论文提出元数据作为解决这一挑战的关键。

论文系统地探索了元数据分配策略，将其归纳为基于子组（subgroup）的框架。元数据可以沿两个正交轴进行分类：
1.  **精度 vs. 范围增强**：**元数据可以作为额外的尾数位**（mantissa bits）来提升精度，或作为**指数位**（exponent bits）**来扩展动态范围**。
2.  **元素级 vs. 子组级应用**：元数据可以应用于子组中最关键的元素，或应用于子组的共享缩放因子。
基于此，论文定义了四种代表性策略：
*   Elem-EM (Element-level Extra Mantissa)：将元数据作为单个元素内的额外尾数位。
*   Elem-EE (Element-level Extra Exponent)：将元数据作为单个元素的指数偏移。
*   Sg-EM (Subgroup-level Extra Mantissa)：增强子组缩放因子的精度。
*   Sg-EE (Subgroup-level Extra Exponent)：编码子组的指数以提升局部动态范围。

此外，还考虑了两种分配模式：
*   **固定共享缩放（Fixed Shared Scale）**：元数据仅局部优化元素或子组，不改变整体共享缩放因子。
*   **自适应共享缩放（Adaptive Shared Scale）**：元数据影响共享缩放因子的选择，通过基于均方误差（Mean Squared Error, MSE）的搜索来共同优化缩放和元数据。

通过详细的帕累托（Pareto）分析，论文揭示了关键的不对称性：在固定共享缩放模式下，元素级元数据（Elem-EM）表现最优，因为它能够直接捕获关键异常值。然而，**当引入自适应共享缩放模式时，子组级元数据（Sg-EM）变得更优**，因为它能通过优化共享缩放因子来重新平衡整个块的量化误差。

基于这一发现，M2XFP采用了**混合策略**来同时优化权重和激活值：
*   **权重（Weights）**：采用Sg-EM-2bit格式。由于权重是静态的且可离线量化，有充足时间进行自适应优化以确定最佳的子组级细化。这通过最小化MSE来选择最优的尾数细化因子$k$和指数偏置$b$。对于一个子组内的权重$W_i$，最优参数$(b^*, \{k_i^*\})$通过以下分层MSE最小化确定：
    $$b^*, \{k_i^*\} = \underset{b \in \{-1,0,1\}}{\operatorname{argmin}} \sum_{i \in \text{sg}} \left\| \hat{W}_{k_i^*,b} - W_i \right\|_2^2$$
    其中$k_i^* = \underset{k \in \{0,1,2,3\}}{\operatorname{argmin}} \left\| \hat{W}_{k,b} - W_i \right\|_2^2$，而$\hat{W}_{k,b}$是使用缩放因子$(1 + \frac{k}{4}) \cdot 2^{E+b}$量化并反量化后的权重。
*   **激活值（Activations）**：采用Elem-EM-top1格式。激活值是在推理时动态生成的，对延迟有严格要求，因此必须采用轻量级、确定性的量化策略。Elem-EM-top1能以最小的路由开销实时捕获每个子组内的异常值。

量化与编码过程：
1.  **激活量化（Elem-EM）**：
    *   **步骤1：计算共享缩放因子**：确定输入张量块中的最大绝对值$x_{max}$，并计算共享缩放因子$S = 2^{\lfloor \log_2 (x_{max}/\text{FP4\_max\_pow2}) \rfloor}$。
    *   **步骤2：量化为FP4 (E2M1)**：将所有元素量化为基线4位E2M1格式。
    *   **步骤3 & 4：识别子组中的top-1值并解决重复**：在每个子组中，识别绝对值最大的元素。如果存在多个相同最大值，则选择内存地址最低的元素作为top-1。
    *   **步骤5：将top-1量化为FP6 (E2M3)**：将识别出的top-1元素的原始高精度值量化为FP6 (E2M3)格式。
    *   **步骤6 & 7：编码偏差并钳制**：为了确保top-1元素在FP6量化后其高4位与原始FP4值保持一致，并允许其在不改变FP4值的基础上细化，论文采用了一种独特的编码策略。首先，将FP6二进制值加上一个偏差（例如+1），然后钳制（Clamp）结果，以保证其高4位（即FP4的表示）与原始FP4值相同。低2位作为元数据，代表额外的尾数精度。例如，如果一个值在FP4中被量化为4.0，那么它在FP6中可能对应3.5, 3.75, 4.0, 4.5, 5.0。通过偏差和钳制，可以利用额外的2位来表示这5种可能性中的4种，而高4位维持FP4表示。
    *   **步骤8：打包量化数据和元数据**：最终的4位FP4数据和2位额外尾数元数据被打包，并按照硬件友好的内存布局组织（元数据集中存储，随后是FP4数据元素）。

2.  **权重量化（Sg-EM）**：
    *   每个子组使用2位额外尾数来细化共享缩放因子$S=2^E$，提供候选值$\{(1 + \frac{k}{4}) \cdot S \mid k \in \{0, 1, 2, 3\}\}$。
    *   当自适应共享缩放启用时，整个组的缩放因子可以通过对指数添加一个偏置$b \in \{-1, 0, 1\}$来调整，该偏置不占用额外存储位。
    *   通过层级MSE最小化来选择最优参数，首先对给定指数偏置$b$寻找每个子组的最优尾数细化$k$，然后选择最佳的组级指数偏置$b$。

**硬件架构**：
为了高效支持M2XFP，论文**对现有脉动阵列加速器**进行了轻量级扩展：
1.  **Top-1解码单元（Top-1 Decode Unit）**：在计算前预处理输入子组。它包含一个紧凑的16条目查找表（LUT），将FP4元素映射到无符号整数，以便进行单调比较。一个三级比较器树识别每个子组中唯一的top-1元素，并在出现平局时选择索引最低的元素。选定的索引和元数据随后发送给PE阵列。
2.  **增强型处理单元（Augmented PE Tile）**：集成了基线FP4-FP4 MAC数据路径，并增加了元数据细化逻辑。
3.  **量化引擎（Quantization Engine）**：负责激活值的在线编码。它是一个两阶段流水线：第一阶段计算组级缩放并生成FP4/FP6候选值；第二阶段识别每个子组的top-1元素，应用偏差-钳制编码，并将结果FP4数据与2位元数据打包。

M2XFP在LLaMA-2/3 (7B-70B)、OPT、Mistral和Falcon等多个LLMs上进行了评估。
*   **精度**：M2XFP平均降低了MXFP4的精度损失70.63%，相对于最新的NVFP4也减少了37.30%的损失。在复杂推理任务（如DeepSeek-R1-Distill-Qwen）上，M2XFP能有效恢复MXFP4导致的精度下降。
*   **硬件开销**：M2XFP的PE单元面积比MXFP4增加4.0%，相比NVFP4增加2.3%，但总面积和功耗开销（0.26%面积，0.36%功耗）相对于整个加速器而言微不足道，这得益于其E8M0格式的低开销。
*   **性能和能效**：M2XFP比现有最先进的MX加速器MicroScopiQ平均实现1.91倍的加速和1.75倍的能效提升。

## NVFP4 QAD 量化感知蒸馏
Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery

https://arxiv.org/pdf/2601.20088 2026.1.27 NVIDIA

https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qad

<img width="749" height="343" alt="image" src="https://github.com/user-attachments/assets/58040798-2b3b-4ae3-8e63-9b93f6526145" />

1.  提出了量化感知蒸馏（QAD），这是一种有效恢复NVFP4量化大型语言模型（LLMs）和视觉语言模型（VLMs）推理准确性的方法。
2.  QAD通过使用**KL divergence损失**将全精度教师模型蒸馏到量化学生模型中，解决了传统量化感知训练（QAT）在复杂多阶段后训练（如SFT和RL）管线中遇到的挑战。
3.  QAD**对数据质量和覆盖范围**具有鲁棒性，能够使用部分或合成数据将模型精度恢复到接近BF16水平，特别适用于RL-heavy模型。
更大模型如671b 即时GPTQ-NVFP4精度也基本持平（包括AIME24）；另外一种手段是QAT融入到RL训练中。

本技术报告详细阐述了量化感知蒸馏 (Quantization-Aware Distillation, QAD) 及其在恢复NVFP4量化大语言模型 (LLM) 和视觉-语言模型 (VLM) 推理精度方面的最佳实践。

**1. 引言与背景**
传统的量化感知训练 (Quantization-Aware Training, QAT) 虽然有效，但对现代LLM面临挑战：一是LLM训练流程复杂，通常包含多阶段后训练（如监督微调 SFT、强化学习 RL、模型合并），难以复制原始训练过程；二是数据可用性和质量问题。
为此，本报告提出QAD方法。QAD通过知识蒸馏将全精度教师模型的能力转移到量化学生模型。与QAT不同，QAD利用KL散度作为损失函数，而非任务特定的目标。其核心优势在于：1) 对通过SFT、RL和模型合并等多阶段后训练的模型展现出卓越的有效性和稳定性，而QAT在此类场景下常遇到工程复杂性和训练不稳定性；2) 对数据质量和覆盖范围具有鲁棒性，无需完整的训练数据即可实现精度恢复。

**2. NVFP4格式与量化方法**
QAT通过微调量化模型来恢复推理精度，量化权重和激活但保留高精度梯度以确保收敛稳定性。知识蒸馏 (Knowledge Distillation, KD) 从教师模型向学生模型传递知识，通常使用KL散度衡量软标签（概率分布）之间的差异。KD能够提供隐式正则化并加速收敛。

**3. 量化感知蒸馏 (QAD) 核心方法论**
QAD的核心在于其损失函数与QAT的根本区别。对于给定输入 $x$ 和词汇表 $V$，设 $p_{\text{teacher}}(y|x)$ 为全精度教师模型的输出概率分布， $p_{\text{student}}(y|x)$ 为量化学生模型的输出概率分布。QAD的损失函数定义为教师和学生分布之间的KL散度：
$$ \mathcal{L}_{\text{QAD}} = D_{\text{KL}}(p_{\text{teacher}} \| p_{\text{student}}) = \sum_{y \in V} p_{\text{teacher}}(y|x) \log \frac{p_{\text{teacher}}(y|x)}{p_{\text{student}}(y|x)} $$
而QAT则使用与原始模型训练相同的任务特定损失，例如语言建模的下一词交叉熵损失。实验表明，QAD能够实现与BF16教师模型近乎零的KL散度，忠实地保留了原始模型的输出分布。相比之下，QAT虽然能匹配验证损失，但会显著改变模型的输出分布，这表明QAT实际上充当了一个额外的后训练阶段。

**4. QAD在后训练模型上的有效性**
*   **SFT-Heavy模型：** 在Llama Nemotron Super V1和Nemotron Nano V2等SFT-Heavy模型上，QAD在挑战性推理基准测试（如AIME25和GPQA-D）上始终优于QAT，恢复至接近BF16的精度。
*   **RL-Heavy模型：** 在Nemotron 3 Nano和AceReason Nemotron等RL-Heavy模型上，QAT显著降低了性能，因为它可能破坏RL训练阶段学习到的能力。而QAD通过匹配教师的输出分布，成功恢复了接近BF16的性能，避免了从头重新学习数据分布的风险，证明了蒸馏对于RL训练模型的关键作用。
*   **对不完整领域数据的鲁棒性：** AceReason Nemotron的实验表明，即使只使用部分领域数据（如仅数学或仅代码数据）进行训练，QAD也能达到接近使用完整数据时的性能。通过蒸馏，教师模型的输出分布编码了所有领域的隐式知识，使得学生模型能实现跨领域知识迁移。
<img width="755" height="703" alt="image" src="https://github.com/user-attachments/assets/3483b51b-13bc-440d-9715-705a2d7aa15c" />

<img width="755" height="219" alt="image" src="https://github.com/user-attachments/assets/26e95b5b-25e1-455d-9d7a-daf496e59406" />

**5. 消融研究**
*   **训练数据质量：** QAD对训练数据的来源和质量表现出显著的鲁棒性。在AceReason Nemotron上的实验显示，无论是原始SFT数据、由RL提示生成的BF16数据、甚至完全随机的token序列，QAD都能保持可比的性能，且不会破坏模型，这表明合成数据对于QAD非常有效。
*   **学习率敏感性：** QAD的学习率选择因原始训练类型而异。对于SFT-trained模型，最佳学习率通常低于或等于原始后训练学习率（如1e-6或2e-6）。对于RL-trained模型，由于RL阶段将模型从初始SFT数据分布转移，QAD可能受益于更高的学习率（如1e-5）。
*   **其他选择：** KL散度作为蒸馏损失函数优于MSE，因为它更适合衡量概率分布差异并提供更好的概率匹配梯度。使用原始BF16模型作为教师模型的效果优于使用更大的同系列教师模型，这可能因为适应不同的分布需要更多的训练数据。



## MatGPTQ
MatGPTQ: Accurate and Efficient Post-Training Matryoshka Quantization

https://arxiv.org/pdf/2602.03537 2026.2.3 Dan Alistarh

https://github.com/IST-DASLab/MatGPTQ
1. 后训练Matryoshka量化（PTQ INT）方法，可生成一个**多精度 最高有效位 推理时动态切换**”模型，旨在解决先前Matryoshka量化（MatQuant）方法在效率和开源支持方面的局限性。
2. ⚙️ 该方法通过适应GPTQ算法并引入多精度优化目标和跨比特错误补偿，同时结合进化搜索（EvoPress）实现异构层级量化，从而有效提升了量化精度。
3. llama3.1-8b，Qwen3-8b/14b等模型，A6000 GPU等评测：**高比特精度上与基线相当**，在3比特设置下平均提升1.34%；单卡vLLM 小batch下相比BF16加速1.5x～3x。

<img width="820" height="286" alt="image" src="https://github.com/user-attachments/assets/d19f69fc-0a8d-4f09-9cbd-3dad87c32c4a" />

## NVFP4预训练
Dissecting Outlier Dynamics in LLM NVFP4 Pretraining

https://arxiv.org/pdf/2602.02047 港科广，阿里等 2026.2.2

1. 对**LLM NVFP4在预训练**中的异常值动态分析，旨在缩小其与BF16基线之间的性能差距。
2. 研究发现，Softmax Attn中的softmax和attn.V对精度更敏感；Linear Attention的outprojct敏感。LA比SA表现出更轻的重尾现象，异常值从训练早期的瞬态尖峰演变为后期更持久的“热通道”，且“**post-QK**”**操作对量化高度敏感**。
3. 引入了Hot-Channel Patch (**HCP**)在线补偿机制和CHON训练配方，将GLA-1.3B模型的NVFP4损失差距从0.94%降低到0.58%，更高的下游任务的准确性。只增加9%～13%的训练效率开销。并对Qwen3-8b进行了SFT和RL训练，和基线基本持平。
   
<img width="821" height="238" alt="image" src="https://github.com/user-attachments/assets/8a57c27d-fe58-43d3-bc20-77f81e49f7ff" />

现有研究多集中于训练后的量化（post-training quantization），而本研究首次对NVFP4预训练期间异常值在不同架构中的定位、产生原因及时间演化进行了纵向分析。

1.  **异常值的定位**：与 SoftmaxAttn (SA) 相比，Linear Attention (LA) 在 per-tensor 层面展现出更轻的重尾分布（heavy tails），减少了系统性异常值的数量，如GLA模型较低的激活值 Kurtosis 所示。然而，在 block-level 量化下，LA 仍表现出持续的局部尖峰。
<img width="480" height="273" alt="image" src="https://github.com/user-attachments/assets/58a391d9-40a6-412b-bd4d-b4c142e1c002" />

3.  **异常值产生原因**：SA中**Softmax是异常值产生的根本原因，这归因于其归一化约束**。具体来说，**求和为一的要求迫使模型采用极端的动态范围来有效抑制信息量不足的 token**。**为了使这些token的贡献接近于零，模型被迫生成幅度较大的 Pre-Softmax** 值。
图 7 中的三个指标来实证验证这一机制：Post-Softmax Entropy、Pre-Softmax Kurtosis 和 Pre-Softmax Max。
<img width="527" height="133" alt="image" src="https://github.com/user-attachments/assets/d78953fd-769d-48e9-b012-9f8c03e637c9" />
<img width="587" height="273" alt="image" src="https://github.com/user-attachments/assets/bd240252-810c-4486-9b72-55d0128a88bc" />

① Post-Softmax Entropy 的下降反映了注意力**权重的集中度不断增加**。
② Pre-Softmax Kurtosis 的上升**表明logit空间向重尾分布转变，这是异常值形成的标志**。
③ Pre-Softmax Max（从 4 增长到 10）的增长证实了使 Softmax 函数饱和所需的 logit 差异不断增大

**LA中的门控机制中的逐元素指数函数** \phi(x) = \exp(t \cdot x)(t > 2)ϕ(x)=exp⁡(t⋅x)(t>2)\phi(x) = \exp(t \cdot x)(t > 2)ϕ(x)=exp(t⋅x)(t>2)，特别是 gk proj 层，是 GLA 中极端异常值的主要来源。衰减因子 \lambda_tλt\lambda_tλt​ 通过 log-sigmoid 函数从 gk proj 导出。为了实现状态重置 (\lambda_t \approx 0λt≈0\lambda_t \approx 0λt​≈0)，预激活输入必须是极负值 (例如，\approx -120≈−120\approx -120≈−120)，而长期保持 (\lambda_t \approx 1λt≈1\lambda_t \approx 1λt​≈1) 需要正向饱和。这需要巨大的动态范围 (例如，[-120, 80][−120,80][-120, 80][−120,80])，对均匀 FP4 量化构成了严峻挑战。如图 6b 所示，gk proj 的平均 Top-1 幅度显著超过其他组件

5.  **异常值的演化**：异常值在训练早期（例如，步骤400-5,400）表现为瞬态的、漂移的尖峰，但在训练后期（例如，步骤15k）则演变为一小部分持续存在的“热通道”（hot channels），即具有持续高幅度的通道。这种从随机漂移到结构性固定的转变，为在线选择性缓解策略提供了依据。此外，激活值的 Flush-to-Zero (FTZ) 率始终高于权重，且 CHON 方法能有效降低激活值的 FTZ。异常值也倾向于在网络的更深层（特别是最后四层）累积。

本文提出了 Hot-Channel Patch (HCP) 机制。HCP 是一种轻量级的在线补偿机制，它周期性地识别并跟踪这些持续存在的“热通道”，并使用硬件高效的核（kernel）对量化残差进行补偿。
<img width="488" height="435" alt="image" src="https://github.com/user-attachments/assets/7f4746d0-b478-42d6-a455-3ca852544531" />

HCP 的核心思想是利用量化后的乘积分解：对于线性变换 $Y = W^T X$，其量化近似为 $ \tilde{W}^T \tilde{X} = W^T X + W^T \Delta X + \Delta W^T X + \Delta W^T \Delta X $，其中 $ \Delta X $ 和 $ \Delta W $ 分别是激活值和权重的量化误差。HCP 旨在通过补偿误差项来使 $ \tilde{W}^T \tilde{X} $ 更接近 $W^T X$。
HCP 通过定义一个复合误差得分 $s_j = \frac{1}{k} \| \Delta X_{j,:} \|_1 + \frac{1}{m} \| \Delta W_{:,j} \|_1$ 来识别前 $k$ 个热通道 $I$。在多种补偿配置中，HCP 选择了 Single-Kernel, Second-Order, Both (S-O2-B) 方案，它通过构造增强矩阵 $ \tilde{X} = [ \tilde{X}^T, \Delta X_I^T ]^T $ 和 $ \tilde{W} = [ \tilde{W}^T, \Delta W_I^T ]^T $，将补偿融入到单次 GEMM 操作中，使得近似后的乘积 $Y_{HCP}^I$ 的误差仅限于二阶项 $ - \Delta W_I^T \Delta X_I $。这在理论上（如引理 A.9 和定理 A.12 所示）显著减小了量化误差，且由于补偿通道数 $k$ 远小于总通道数 $D$，额外计算开销与 $O(k)$ 成线性关系，并通过融合 Triton kernels 保持了高效性。

HCP 被集成到 NVFP4 预训练流程中，并结合了对 `post-QK` 操作的额外保护（例如，将LA的 `Wo` 和 GLA的 `gk proj` 保持在 BF16 精度），形成了新的训练配方：CHON (Compensated Hot-channel Optimization for NVFP4)。
<img width="973" height="539" alt="image" src="https://github.com/user-attachments/assets/61fdd09f-7f74-4166-acf8-812dd05725b6" />

**SFT**：Qwen3-8b, FSDP2. 所有线形层包括Q/K/V/Out,MLP都用NVFP4，attn/LM_head/emd不做量化。
<img width="773" height="311" alt="image" src="https://github.com/user-attachments/assets/817d1812-e139-4885-82e6-39ecfb64aa78" />

**RL**: Qwen3-8b,veRL/FSDP/vLLM, GRPO. E2E的NVFP4目前还有问题。

<img width="829" height="449" alt="image" src="https://github.com/user-attachments/assets/687b62ae-9659-4eac-ac0e-f421aaaa2565" />

## MX量化评估

https://arxiv.org/pdf/2601.09555 华为 2026.1.14

1. 系统评估了MXFP格式下大型语言模型（LLMs）的Post-Training Quantization（PTQ）方法，7b/8b模型 vLLM。
2. 推理(AIME24/25)/非推理任务下，**MXFP8能持续实现近乎无损**，**MXFP4引入了显著的精度下降且仍具挑战性**，其中误差补偿和仿射变换方法与MXFP量化更兼容，而**旋转变换方法则会损害MXFP4的性能**。
3. PTQ性能趋势在不同模型家族和模态间高度一致，且MXFP4中**量化缩放因子是关键误差源**，可通过简单的**预缩放优化策略**显著缓解其影响。
基于 https://github.com/microsoft/microxcaling

<img width="500" height="260" alt="image" src="https://github.com/user-attachments/assets/153ffdd1-72c4-4e8f-b6a5-886da890dd83" />
<img width="945" height="626" alt="image" src="https://github.com/user-attachments/assets/8280e49a-86cb-4625-9645-522f81a80c87" />

**2. 预备知识**
**2.1 低位整数（INT）与浮点（FP）量化**
整数量化通常通过裁剪和缩放操作将高精度张量$W$映射到低位整数范围。其定义为：
$W_q := \text{clip} (\lfloor W/s \rceil , Q_{\text{min}}, Q_{\text{max}}) \cdot s$
其中，$\text{clip}(\cdot)$用于截断值，而$s$是缩放因子。
浮点量化则更复杂，涉及符号位（S）、指数（E）和尾数（M）。一个FP格式通常表示为$\text{E}a\text{M}b$。FP量化定义为：
$W_q := \text{nearest} (\lfloor W/s \rceil , C_{\text{FP}}) \cdot s$
其中，$C_{\text{FP}}$是可表示的低位浮点值集合，$\text{nearest}(\cdot)$将归一化值映射到$C_{\text{FP}}$中最近的元素。

**2.2 微缩浮点（MXFP）量化**
MXFP是OCP提出的一系列量化浮点格式，采用块量化（block quantization）机制，块大小为32，并为每个块使用共享的UE8M0数据类型。本文主要关注MXFP8和MXFP4，其中MXFP8采用E4M3变体（因其更大的尾数宽度对细粒度量化性能更关键），MXFP4采用E2M1。

**2.3 后训练量化（PTQ）方法分类**
本文将现有PTQ方法分为四类：
*   **通道级变换（Channel-Wise Transformation）**：通过自适应调整激活和权重的数值范围来减少量化误差。代表算法包括SmoothQuant（通过每通道缩放将量化难度从激活迁移到权重）和AWQ（通过识别关键权重来高效量化LLMs）。
*   **误差补偿（Error Compensation）**：显式建模和补偿量化引起的差异。代表算法包括GPTQ（逐层量化并利用逆Hessian信息更新权重）和MR-GPTQ（GPTQ的扩展，针对FP4特性融入块级Hadamard变换和格式特定优化）。
*   **旋转变换（Rotational Transformation）**：利用预量化正交变换来重构数据分布，以减轻极端离群值的影响。代表算法包括QuaRot（随机正交旋转）和SpinQuant（校准过程中学习可训练的旋转）。
*   **仿射变换（Affine Transformation）**：通过应用可学习的重缩放变换来改善低位模型压缩，从而重新分配跨维度数值大小。代表算法FlatQuant（通过轻量级、块级训练策略在校准阶段识别逐层最优仿射变换）。

**3. 实验设置**
*   **量化配置**：
    *   **仅权重量化（Weight-Only Quantization）**：仅量化线性层权重，激活保持全精度。
    *   **权重-激活量化（Weight-Activation Quantization）**：权重和输入激活均量化，实现全量化矩阵乘法。
    *   **KV缓存量化（KV Cache Quantization）**：量化attention块中的Key和Value张量。
    *   表示方式：W{bits}A{bits}[KV{bits}]，如W4A8表示权重4位，激活8位。
*   **评估基准**：
    *   语言模型质量：WikiText2上的困惑度（PPL）。
    *   非推理任务（零样本）：PIQA、Winogrande、HellaSwag、ARC-Easy、ARC-Challenge。
    *   推理基准：MATH-500、AIME24、AIME25。
    *   多模态基准：OCRBench、MMBench、MMBenchCN、TextVQA、ChartQA、MME、MMMU。
*   **模型**：Llama-3.1-8B-Instruct、openPangu-Embedded-7B-V1.1、Qwen2.5-VL-7B、openPangu-VL-7B。
*   **工具**：所有实验均使用**microxcaling库模拟MXFP格式，评估后端使用vLLM**。

**4. 核心发现与实验分析**
**4.1 不同MXFP量化设置的性能（RQ1）**
本文将后量化性能下降分为三个区域（相对于BF16恢复率）：无损（≤1%）、良好（1%-3%）和高风险（≥3%）。
*   **W8A8**：**在所有任务和模态上基本实现无损性能**，表明8位权重和激活量化对于当前LLMs和MLLMs是安全的，可直接部署。
*   **W4A8**：在RTN设置下出现显著精度下降，**但通过PTQ算法可有效缓解**。对于非推理任务，性能损失可控，**但推理任务仍处于高风险区域**。
*   **W4A4**：是最具挑战性的设置，精度下降严重且普遍（恢复率降至86.37%–97.36%），几乎所有方法都进入高风险区域。量化噪声超过临界阈值时，不同方法处理离群值、动态范围对齐或激活敏感度的差异变得至关重要。
**结论1**：W8A8在不同模型和基准上均能保持无损性能。但对于4位权重或激活量化（如W4A8和W4A4），仍是MXFP下的开放挑战。
<img width="717" height="726" alt="image" src="https://github.com/user-attachments/assets/0a18e1e2-2f39-4a7c-a21f-ff335496659e" />

**4.2 PTQ方法对比（RQ2）**
*   **误差补偿方法**（如GPTQ和MR-GPTQ）在大多数情况下**优于通道级变换方法**（如SmoothQuant和AWQ）。通道级缩放的粒度较粗，难以完全捕获MXFP分组量化下的组内幅度变化。误差补偿方法则在校准过程中显式最小化量化误差，提供更强的性能保证。
*   **旋转变换**（如QuaRot、SpinQuant）反而**损害MXFP4量化精度**，表现差于RTN基线。MXFP4依赖于局部统计特性（如各组内的分布形状）来保留信息。全局旋转会混合所有维度的信息，使离群值结构扁平化，减少峰度，从而使分布不适合有效的组级缩放。
*   **仿射变换**（FlatQuant）**在4位量化下表现出最强的鲁棒性**。它通过可学习的仿射变换调制绝对数值大小，而非像正交旋转那样保留L2范数，因此更适合低位MXFP量化。
*   **RTN**（Round-to-Nearest）在所有位宽下仍是强劲的基线。现有PTQ方法大多为INT格式设计，直接应用于MXFP时，往往只带来微弱增益甚至退化，这表明MXFP格式需要为其特定量化方案量身定制的量化方法。
**结论2**：误差补偿和仿射变换方法与MXFP量化兼容性更强，尤其是在低位宽下。RTN作为强基线表明MXFP需要为其特定方案设计的量化方法。

**4.3 模型家族和模态的影响（RQ3）**
*   **PTQ在MXFP下的有效性在模型和模态间高度一致**。不同模型在MXFP量化设置下的性能恢复率曲线表现出高度一致性，平均皮尔逊相关系数为0.917。这表明**PTQ方法的有效性不强烈依赖于模型架构或模态**。
*   在**多模态中，量化敏感性主要由LLM组件主导**，而非视觉Transformer（ViT）。将LLM从W8A8量化到W4A4会导致显著的精度下降（Qwen2.5-VL-7B下降3%），而对ViT进行相同的W4A4量化仅导致约1%的下降。这建议了一种实用有效的量化策略：**LLM保持较高精度（如W8A8），而ViT**可进行激进量化（如W4A4）。
*   **视觉tokens在MXFP量化下比INT格式更鲁棒**。先前研究表明，量化MLLMs到INT格式面临挑战，因为视觉tokens通常具有更大的离群值和更宽的激活范围。然而，在**MXFP量化下，降低视觉tokens的位宽并未导致显著的精度损失**。这可能与MXFP的指数-尾数解耦特性有关，使其能够更灵活地处理宽激活范围，同时保持足够的精度。
**结论3**：MXFP下的PTQ方法在不同模型和模态间表现出一致的有效性。在MLLMs中，量化敏感性由LLM而非ViT主导，这有利于采用LLM保持更高精度的混合精度设计。视觉tokens在MXFP下比INT更鲁棒，降低位宽未导致显著精度损失。

**4.4 MXFP4量化组件分析（RQ4）**
*   **缩放因子引入的量化误差不可忽略**。MXFP格式中**FP8块级缩放因子$s$必须满足E8M0**数据类型约束，这意味着**缩放因子必须是2的幂**。这种粗糙的量化常导致最优缩放与允许缩放之间存在较大不匹配，从而影响块内所有值。实验表明，**恢复高精度缩放因子可显著降低困惑度**。
*   **预缩放（Pre-Scale）优化策略**被推荐。本文采用了**无偏MXFP4量化策略，通过在量化前将输入按3/4缩放**，有效**防止了截断，同时保留了相对幅度**。实验结果显示，启用预缩放操作显著提升了性能并降低了PPL。
**结论4**：缩放因子导致的量化误差不可忽略。推荐使用预缩放优化策略。

   
## NVFP4 Four Over Six
https://arxiv.org/pdf/2512.02010 MIT 韩松团队 2025.12.22

https://github.com/mit-han-lab/fouroversix

1. 提出了Four Over Six (4/6)，一种改进NVFP4 QAT量化算法的方法，通过自适应地将数据块缩放到4或6，显著减少了浮点格式中对接近最大值的量化误差。
2. 方法解决了NVFP4非均匀步长在接近最大值（尤其约5附近）处引入的性能下降问题，使得模型在预训练时能将训练损失带到更接近BF16的水平，并提升了后训练量化（PTQ）的准确性。
3. 在NVIDIA Blackwell GPU实现，具有~15%的计算开销，并能与现有的PTQ方法（如GPTQ、AWQ和SmoothQuant）结合，在多种LLM模型和下游任务上展示出广泛的性能提升。
<img width="756" height="438" alt="image" src="https://github.com/user-attachments/assets/aedd46f0-d4ac-4589-a8e3-e572dfe7623d" />


## Nightjar动态投机
Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving

https://arxiv.org/pdf/2512.22420 国防科大 2025.12.27

1. 推测解码在动态请求负载下因**固定推测长度而导致的性能瓶颈**，本文提出了Nightjar，一种基于学习的自适应推测解码算法。
2. Nightjar利用上下文多臂老虎机（contextual bandit）方法，能够根据实时批次大小**动态选择最优的推测长度**，并首次纳入了从禁用状态重新启用推测解码时的KV缓存重建成本。
3. 最大7b/4090/vLLM0.8.2，Nightjar相比传统推测解码，吞吐量最高提升14.8%，延迟降低20.2%，并在不同模型和数据集上均持续优于现有SOTA方法，显著提升了LLM服务的效率和鲁棒性。
   
## entquant ANS
Float8@2bits: Entropy Coding Enables Data-Free Model Compression

https://arxiv.org/pdf/2601.22787 德国 2026.2.2

https://github.com/merantix-momentum/entquant

1. EntQuant 提出了一种创新的后训练**量化+压缩**，通过ANS熵编码将数值**精度维持Float8 或 Int8，但存储为2～4bit**，首次实现了data-free、模型无关的极端模型压缩。
2. 70B参数压缩时间不到30分钟; 使用nvComp.ANS即时解压缩(**计算密集**)，在标准基准测试和指令微调模型上均展现出领先的性能。
3. HF,H100. llama2/3/Qwen3, Oloma等模型，比 BFloat16 慢1.5-2倍, 显著降低了内存占用。
精度总体和GPTQ,QuIP差不多，不需要校准，速度上欠缺优化。～=NVFP4 （推广到A/H？）

<img width="500" height="269" alt="image" src="https://github.com/user-attachments/assets/f1a8cb51-20a2-4084-8d4c-87a3b329dbe8" />
<img width="497" height="381" alt="image" src="https://github.com/user-attachments/assets/af208325-7467-4f5e-8ca9-f347434601b5" />
<img width="876" height="493" alt="image" src="https://github.com/user-attachments/assets/d49ed9ad-99cc-4e3b-9de7-3ac8d0b628ad" />
<img width="1015" height="568" alt="image" src="https://github.com/user-attachments/assets/2c20aaa8-8bec-4687-9d4d-dbc2d820f120" />


## ZipMoE
https://arxiv.org/pdf/2601.21198 南京大学 周志华团队 2026.1.30

1. 💡 ZipMoE 提出了一种高效且语义**无损的**片上 MoE 服务系统，旨在解决大型 MoE 模型在资源受限边缘设备上部署时面临的内存和 I/O 瓶颈。
2. 对BF16参数中指数位的统计冗余进行无损压缩(lz4HC, zstd, -30%)，并结合 CPU 并行解压与 GPU 内存合并的张量恢复，将推理重心从 I/O 转移至计算。
3. 基于HF，Orin（64GB），DeepSeek-Lite, Qwen-1.5b引入了针对共享内存 SoC 架构优化的缓存亲和调度与分层缓存管理机制， 72.77% 的推理延迟降低和6x吞吐。
   
<img width="543" height="499" alt="image" src="https://github.com/user-attachments/assets/bcd2a435-ec76-4d1d-9717-1760919d79ce" />

## ODC
https://arxiv.org/pdf/2601.19362 SeaAI，2026.1.27

https://github.com/sail-sg/odc

1. 针对LLM后训练中因序列**长度差异导致的负载不平衡问题**，本文提出了一种名为On-Demand Communication (ODC)的新范式，旨在**重新引入参数服务器（PS）的容错优势**。
2. ODC通过**将FSDP中逐层的集体通信替换为点对点通信**，从而将同步barrier从逐层**放松到逐minibatch**，有效缓解了设备空闲和“慢者拖累”效应。
3. 基于Triton-Distributed实现，最大dense-32b，ODC显著提高了设备利用率和训练吞吐量，在多种LLM后训练任务中**+36%的速度提升**，证明了其在处理不平衡工作负载方面的优越性。
   
<img width="677" height="274" alt="image" src="https://github.com/user-attachments/assets/3c544ffa-aef9-4edf-9e9a-798aeb6563a6" />
<img width="676" height="371" alt="image" src="https://github.com/user-attachments/assets/7dad84c4-8b90-4b6f-bb27-15b82b134cdd" />
<img width="748" height="637" alt="image" src="https://github.com/user-attachments/assets/7aad10f2-4867-47e6-8702-6c70a55f4507" />

LLM后训练（post-training）中，由于**sequence length高度不一致性**。这种不平衡导致工作负载较小的设备出现空闲（idle），形成所谓的“**拖慢效应**”（straggler effects），进而造成设备利用率低下。FSDP通过在**每一层（per-layer）进行`all-gather`（用于前向传播的参数重建）和`reduce-scatter`（用于后向传播的梯度聚合）操作**，引入了细粒度同步障碍使得设备必须**等待最慢的设备**完成当前层操作，从而在**工作负载不平衡时显著放大了效率低下问题**，导致设备空闲时间可达50%。

本文重新审视了Parameter Server (PS) 范式，并提出了On-Demand Communication (ODC) 方案，将其**适应到现代分片数据并行**（sharded DP）中，以应对LLM后训练中的工作负载不平衡问题。ODC的核心思想是将FSDP中基于集体通信的每层同步操作替换为直接的点对点（point-to-point）通信。具体而言：
1.  **分散式Parameter Server架构：** ODC将FSDP框架重构为一个分散式（decentralized）的Parameter Server。每个设备**既充当“服务器”（拥有并管理模型参数和优化器状态的某个分片），又充当“工作器”**。这种设计保留了FSDP的内存效率和可扩展性，同时避免了传统集中式PS的网络瓶颈。
2.  **细粒度通信替代：**
    *   `all-gather`操作被一系列有针对性的`gather`请求所取代。在计算特定层之前，每个设备只从其对等（peer）设备请求并获取其所需参数的分片（shard）。
    *   `reduce-scatter`操作被分解为一系列`scatter-accumulate`操作。每个设备在计算完梯度后，直接将其计算出的梯度推送到拥有相应梯度分片的设备上。
3.  **放松同步粒度：** 通过这种点对点通信，ODC将同步障碍从每一层放松到每个minibatch的末尾。这意味着设备可以在minibatch内独立推进其计算，无需等待其他设备完成当前层的计算，从而显著缓解了拖慢效应。
4.  **非侵入式通信：** ODC的通信操作是**非侵入式的**（non-intrusive）。当一个设备向另一个设备发起`gather`或`scatter-accumulate`请求时，不会中断目标设备上正在进行的计算。这主要通过利用远程直接内存访问（RDMA）技术实现：节点内（intra-node）通信使用CUDA IPC，节点间（inter-node）通信使用NVSHMEM，通信内核基于Triton-Distributed构建，在Python Triton kernels中直接暴露RDMA功能。

ODC的引入也简化并增强了负载均衡策略。传统的序列打包（sequence packing）方法（如Krell et al., 2021）通常在microbatch层面进行负载均衡，但受限于设备内存和计算（$O(s^2)$）与内存（$O(s)$）扩展不匹配的固有矛盾。ODC消除了每个设备必须处理相同数量microbatch的隐含要求，使得负载均衡可以从细粒度的microbatch层面提升到粗粒度的minibatch层面。新的LB-Mini策略首先在全局样本集合中平衡总计算负载到各个设备，然后每个设备独立地将其本地样本子集打包成microbatch，仅受本地内存约束。这通过Karmarkar-Karp算法实现，并迭代验证分区是否会导致out-of-memory (OOM)。

在实验评估中，ODC在SFT（LongAlign, SWE-Smith）和RL（GRPO on AIME prompts）等多样化的LLM后训练任务上，一致性地提高了设备利用率和端到端吞吐量。与标准的FSDP相比，ODC实现了高达36%的加速。特别是在打包场景下，ODC的收益更为显著。当minibatch大小为1时，ODC与集体通信的性能相似，因为在这种情况下，两者都几乎每个样本后进行同步。参数研究表明，ODC的加速效果随序列长度和设备数量的增加而增强，随打包比率（packing ratio）的增加而减弱，并在中等minibatch大小下达到峰值。

尽管节点间通信基准测试显示ODC的点对点RDMA相比NCCL优化后的集体通信（利用层次化互联拓扑）存在带宽劣势，但论文指出，这种劣势可以通过两种方式缓解：
1.  **通信与计算重叠（Overlapping Communication with Computation）：** 对于长序列，计算复杂度（$O(s^2)$）远高于通信量（$O(s)$），使得大规模计算能够有效掩盖通信延迟。
2.  **混合分片（Hybrid Sharding）：** 类似于ZeRO++，在节点内分片参数和梯度，而在节点间分片优化器状态。这消除了跨节点的参数`gather`和梯度`scatter-accumulate`，虽然会增加单节点内存使用，但在许多情况下是可接受的权衡。

未来的工作方向包括：开发ODC特有的通信优化（如拓扑感知取回），探索放松同步保证以支持异步SGD方案，以及集成Parameter Server固有的弹性（elasticity）和容错（fault tolerance）能力，以提升大规模LLM训练的韧性和灵活性。


## SNIP
SNIP: An Adaptive Mixed Precision Framework for Subbyte Large Language Model Training
https://www.arxiv.org/pdf/2602.01410 ASPLOS26
1. 🚀 SNIP 是一个针对（LLM）训练的自适应混合多种精度训练框架，旨在通过细粒度层级量化解决模型质量与训练效率之间的挑战。
2. 💡 该框架通过引入前向损失散度和后向权重散度这两个新颖指标来量化精度损失，并利用整数线性规划（ILP）问题优化层级精度配置以最小化质量损失。
3. Hopper/A100等SNIP 在 1B 至 **70B 规模的 Llama dense** 类模型上，能在保持接近全精度模型质量的同时，将浮点运算（FLOPs）降低高达 80%，并持续优于现有基线方法。


## PAT
PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

https://arxiv.org/pdf/2511.22333 天津大学 ASPLOS26

https://github.com/flashserve/PAT.git

1. PAT提出了一种前缀感知的注意力内核实现，通过打包-转发-合并范式来加速LLM解码，旨在解决现有方法中重复的KV缓存加载和资源利用效率低下问题。
2. 系统采用启发式打包调度器将共享前缀的查询分组以减少冗余内存访问，并设计了资源高效的多瓦片内核、多流转发和长KV分割策略来提升GPU利用率。
3. 3K代码 集成到vLLM。8b模型/A100，PAT与现有FlashAttn/FlashInfer相比，平均可将注意力延迟降低**53.5%，并将TPOT降低17.0-93.1%**，显著提升了LLM解码性能。
<img width="606" height="304" alt="image" src="https://github.com/user-attachments/assets/13acac61-b859-49a8-9355-2154a158e3ae" />

<img width="748" height="426" alt="image" src="https://github.com/user-attachments/assets/a3368ea9-5c2a-429c-b782-d6854981da6e"
<img width="1087" height="481" alt="image" src="https://github.com/user-attachments/assets/cb2ad33b-e062-46b8-b78d-e79b36802eec" />
<img width="1155" height="443" alt="image" src="https://github.com/user-attachments/assets/b5afc0ee-c111-4107-9a0a-2a2f8b25ec89" />

**共享前缀与现有实现缺陷**：
**冗余内存访问**： 现有查询中心 (query-centric) 的注意力内核（如 FlashAttention、FlashInfer）采用“每个查询一个 CTA”的打包策略，导致共享 KV 前缀（例如，多请求共用的系统提示）被重复从慢速全局内存加载，引入 4.3-8.7 倍的冗余 KV cache 流量。
**资源利用率低下**： 现有内核采用固定 tile size 的“一刀切”设计（例如，m=64, n=32），忽略了 LLM 工作负载的动态性，导致双重资源低效：
内存浪费 (I_{mem}ImemI_{mem}Imem​): 当共享前缀的查询数量少于 tile size 时，CTA 必须填充输入，浪费共享内存和寄存器。
执行气泡 (I_{exe}IexeI_{exe}Iexe​): 不同 CTA 的 KV 长度差异导致工作负载不平衡，使 SM 在执行后期阶段利用率不足。

## ZipServ
ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

https://cse.hkust.edu.hk/~weiwa/papers/zipserv-asplos26.pdf ASPLOS26 港科广 范瑞波等

https://github.com/HPMLL/ZipServ_ASPLOS26.git
 
1. 提出了一种硬件感知的**无损压缩框架**，通过 Tensor-Core-Aware **Triple Bitmap Encoding** (TCA-TBE：固定长度、基于位图的编码格式而抛弃变长熵编码器如Huffman编码）将BF16权重压缩**～30%（平均11.3b）**，同时**实现恒定时间，高效并行解码**。
2. 创新的融合式**ZipGEMM内核**，加载压缩（到shm），计算即时解压缩（到寄存器），消除了中间缓冲区和冗余数据传输，4090GPU上实现了高达2.21x内核级加速（相比 cuBLAS）。
3. 2.5K LoC CUDA/C++; 1K python集成到vLLM。推理中PD阶段感知策略。prefill阶段由高效的解压缩内核将压缩权重解压缩到全局内存，然后执行高吞吐量的GEMM**分摊解压缩开销，而decode切换到融合的ZipGEMM内核降低延迟。
4. GPU 4090/L40S，ZipGEMM比cuBLAS_TC平均加速1.3x; 比DietGPU、nvCOMP和DFloat11平均加速2.14x～1.10x。8b～70b模型E2E比vLLM、Transformers和DFloat11平均延迟分别降低 17.60%、60.79% 和 82.13%，吞吐量分别提高1.22x、3.18x 和 8.52x。5090上加速更大些。

<img width="614" height="406" alt="image" src="https://github.com/user-attachments/assets/a0956ce1-1ba7-43d1-9fa4-9d4435c5e222" />
<img width="1025" height="331" alt="image" src="https://github.com/user-attachments/assets/0f5bd5b4-d41f-4334-8265-4efe671d3f7b" />
<img width="888" height="403" alt="image" src="https://github.com/user-attachments/assets/52928c7a-cf76-4a18-9b56-17d0d64d7cc2" />
<img width="508" height="298" alt="image" src="https://github.com/user-attachments/assets/99662b13-dd8d-44a0-90bd-f213ffb49475" />
<img width="1028" height="489" alt="image" src="https://github.com/user-attachments/assets/1cac0954-c33e-4cc9-b67d-a17334022cad" />
<img width="710" height="470" alt="image" src="https://github.com/user-attachments/assets/842c2305-6d0f-4027-9473-a6e9b411b9d5" />
<img width="687" height="663" alt="image" src="https://github.com/user-attachments/assets/85f5ecd6-9c2d-4481-b3d5-63d36f395b2d" />

ZipServ 是一种用于实现高效 LLM 推理的无损压缩框架，旨在解决现有无损压缩方法在 GPU 上推理时，由于与 GPU 架构不匹配而导致的显著性能下降问题。传统方法在内核级别产生可变长度位流，这会破坏 SIMT 并行性；在系统级别，解耦的流水线导致冗余的内存流量。ZipServ通过硬件感知设计，提供了存储节省和 LLM 推理加速。

尽管无损模型压缩对于缓解内存和带宽瓶颈具有巨大潜力，但现有方法通常会导致推理速度大幅减慢。这源于传统压缩算法与现代 GPU 架构之间的根本不匹配。
在GPU SIMT模式下，**可变长度位流的解码效率低下**，导致控制流发散和计算资源利用不足。此外，大多数框架权重在计算前完全解压缩到全局内存中，这引入了**冗余的数据传输**和低算术强度。

ZipServ 致力于通过以下两个核心技术来解决这些问题：

1.  **Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE)**：
    *   **核心观察**：BF16 权重中的指数位在当代 LLM 中呈现出高度偏斜、低熵的分布。**前7个最常见的指数值覆盖了超过95%的权重**，且这些指数通常构成数值上连续的序列（例如 $e^*, \dots, e^*+6$）。
    *   **设计理念**：利用这一统计冗余和连续性，TCA-TBE 采用**固定长度、基于位图的编码格式**，**摒弃了传统可变长度的熵编码器**（如 Huffman 编码），从而实现**恒定时间、并行解码**，避免了控制流发散。
    *   **编码细节**：
        *   **3 位码字**：选择 3 位码字，因为它在利用高度偏斜的指数分布方面实现了近乎最优的压缩比。它将前 7 个最频繁的指数值映射到码字 001-111。特殊的码字 000 用于表示指数落在前 7 个范围之外的权重（以全精度存储）。
        *   **平均比特成本**：每元素的平均存储成本为 $\text{AverageBits}(n) = r_n \cdot (n + 8) + (1 - r_n) \cdot (n + 16)$，其中 $n$ 是码字长度，$r_n$ 是被前 $2^n - 1$ 个指数值覆盖的权重比例。对于 $n=3$，平均每元素 11.3 比特。
        *   **解耦三位图布局**：为最大化 SIMT 架构上的解码效率，TCA-TBE 将每个 8x8 权重瓦片的 3 位码字分解为三个独立的 64 位位图，每个位图代表 3 位码字中的一个位平面。这确保了合并的内存访问和无分支解码，与 GPU 的 SIMT 模型对齐。
        *   **分层瓦片设计**：采用三级分层瓦片方案，与现代 GPU 的架构粒度对齐：
            *   **FragTile (FT)**：基本单元为 8x8 瓦片，与 Tensor Core 指令的最小操作数片段匹配。
            *   **TensorCoreTile (TT)**：每个 16x16 瓦片由 2x2 的 FragTile 网格组成，与 PTX 级别 Tensor Core `mma.sync.m16n8k16` 指令的操作数维度匹配。
            *   **BlockTile (BT)**：最粗粒度为 64x64 瓦片，由一个线程块协同处理。
        *   **数据存储**：每个 8x8 FragTile 使用五个缓冲区进行编码：三个 64 位位图（表示 3 位码字的位平面）、一个 PackedSignMantissa 缓冲区（存储高频权重的符号和尾数）以及一个 FullValue 缓冲区（存储全精度回退值）。

2.  **ZipGEMM (Fused Decompression-GEMM Kernel)**：
    *   **设计目标**：实现“加载压缩，计算解压缩”（load-compressed, compute-decompressed）的执行模型，将解压缩和矩阵乘法融合到一个内核中，直接将压缩权重从 DRAM 获取并实时解压缩到 Tensor Core 寄存器中，从而消除冗余数据传输，最大化算术强度。
    *   **内核工作流**：
        *   **瓦片加载**：线程协同将压缩权重瓦片和对应激活瓦片从全局内存加载到共享内存。
        *   **Warp 级解码**：每个 warp 独立地从共享内存中解压缩压缩权重，将其重建为兼容 Tensor Core 消耗的 BF16 值。
        *   **激活寄存器传输**：激活瓦片从共享内存移动到寄存器。
        *   **Tensor Core 计算**：当解压缩的权重和激活都位于寄存器中时，warp 执行 Tensor Core `mma` 指令。
    *   **高效解压缩器**：
        *   **空间位图指示器 (Spatial Bitmap Indicator)**：通过对三个位图进行位或操作生成一个 64 位指示掩码，每个线程根据该掩码中对应比特的值（1 为压缩，0 为回退）来确定其分配元素的存储模式。该过程完全在寄存器中进行，且恒定时间完成。
        *   **动态寻址 (Dynamic Addressing)**：每个线程通过对其分配元素前的空间指示器进行 warp 局部前缀和（`__popc()` 和 `__shfl_sync()` 指令）来实时计算其在值缓冲区中的读取偏移量，无需显式索引。
        *   **通过隐式查找的快速指数重组 (Fast Exponent Reassembly via Implicit Lookup)**：利用指数的数值连续性，离线压缩时识别出前 7 个最常见的连续指数值，并将其映射到 3 位码字（001-111），同时记录一个全局基准指数 (`base_exp = min(top_exponents)-1`)。运行时，每个线程通过将 3 位码字与 `base_exp` 相加来算术重构原始指数，避免了共享内存查找表，仅需一次整数 ALU 操作。
        *   **重打包为 Tensor Core Fragment**：将两个重构的 BF16 元素重新打包到单个 `bfloat162` 寄存器中，以匹配 Tensor Core `mma.sync` 指令所需的操作数布局。
    *   **细粒度软件流水线**：ZipGEMM 采用两级分层流水线：瓦片级双缓冲（覆盖全局到共享内存传输与计算）和切片级交错（覆盖共享到寄存器移动、解压缩与 Tensor Core 操作），有效隐藏了内存和解压缩延迟，确保了稳定的计算流。

ZipServ 采用**阶段感知推理策略 (Stage-Aware Inference Strategy)**：
*   在**预填充 (prefill) 阶段**（计算密集型，大 $N$），ZipServ 采用解耦流水线：首先由高效的解压缩内核将压缩权重解压缩到全局内存，然后执行高吞吐量的 GEMM 操作以分摊解压缩开销。
*   在**解码 (decode) 阶段**（内存密集型，小 $N$），ZipServ 切换到融合的 ZipGEMM 内核，实现实时解压缩以加速 token 生成。

实验评估表明，ZipServ 显著优于现有基线。在 NVIDIA RTX4090 和 L40S GPU 上，ZipGEMM 比 NVIDIA 的 cuBLAS_TC 平均加速 1.31x 和 1.36x，最高加速可达 2.21x，而其他解耦解压缩方法（DietGPU, nvCOMP, DFloat11）均引入了显著的运行时开销。ZipServ 的独立解压缩内核 ZipServ-Decomp 也比 DietGPU、nvCOMP 和 DFloat11 平均加速 2.14x、1.83x 和 1.10x。在端到端推理性能方面，ZipServ 相较于 vLLM、Transformers 和 DFloat11，平均延迟分别降低 17.60%、60.79% 和 82.13%，吞吐量分别提高 1.22x、3.18x 和 8.52x。内存方面，ZipServ 将 LLaMA3.1-8B、Mistral-24B 和 LLaMA3.1-70B 的权重占用空间分别减少到原始的 72.4%、71.3% 和 71.1%，释放的内存可用于 KV Cache，从而支持更大的批次和更长的上下文。ZipServ 的设计也展现出良好的向前兼容性，在最新的 RTX5090 上仍能提供显著加速，并能缩小消费级 GPU 与数据中心级 GPU 的性能差距。

**局限性**：ZipServ 主要针对消费级和推理优化型 GPU，在内存带宽充足的训练型数据中心 GPU（如 A100、H800）上，其性能可能不总是超越高度优化的 cuBLAS 基线，这反映了硬件-软件的某些不匹配。然而，即使在这种情况下，ZipServ 仍能提供最佳的压缩推理支持。

该研究首次证明，当与硬件协同设计时，无损压缩可以为 LLM 推理提供存储节省和实质性的加速。

## GFS
GFS: A Preemption-aware Scheduling Framework for GPU Clusters with Predictive Spot Instance Management

https://arxiv.org/pdf/2509.11134 ASPLOS26 阿里巴巴 上海交大等

1.   针对大型语言模型（LLMs）对 GPU 资源需求的激增导致低优先级（LP）任务抢占率高和排队时间长的问题，本文提出了 GFS，一个**抢占感知调度框架**，旨在优化高优先级（HP）任务的 SLO 合规性并最小化 LP 任务的抢占。
2.  GFS 框架包含三大核心模块：GPU Demand Estimator (GDE) 提供精确的 GPU 需求预测，Spot **Quota Allocator (SQA) 动态调整 spot 实例配额**，以及 **Preemptive Task Scheduler** (PTS) 执行抢占式调度策略以最小化预估成本。
3.  实际生产环境和仿真测试中，GFS 将 LP 任务的抢占率降低了 33.0%，排队时间缩短了 44.1%，GPU 分配率提升了高达 22.8%，每月为集群节省了约 459,715 美元。

## Niyama
Niyama: Breaking the Silos of LLM Inference Serving

https://arxiv.org/pdf/2503.22562 ASPLOS26 微软印度
1. Niyama旨在解决LLM推理服务中筒仓式部署的低效问题，通过引入一种新型的QoS驱动系统，实现在共享基础设施上高效协同调度多样化工作负载。
2. 通过动态分块（Dynamic Chunking）优化吞吐量、混合优先级策略（Hybrid Prioritization）平衡公平性与效率，以及主动降级（Eager Relegation）机制，实现在过载条件下优雅地降低服务质量。
3. Niyama在Llama3-8B和Qwen-7B模型上，使用ShareGPT、Azure Conv和Azure Code等数据集进行评估，并对比了Sarathi-Silo（SOTA竖井部署）、Sarathi-FCFS、Sarathi-EDF和Sarathi-SRPF等基线。Niyama将服务能力提高了高达32%，并在极端负载下显著减少了SLO违规，优于SOTA筒仓式部署。
   
包括三个队列（prefill queue、decode queue、relegated queue），一个prefill selector（实现混合优先级），一个violation checker（实现主动降级），以及一个轻量级预测器（用于动态分块）。Niyama基于Sarathi调度器实现，并兼容vLLM的PagedAttention机制。


## DuetServe
DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing

https://arxiv.org/pdf/2511.04791 USC加州大学 2025.11

1. DuetServe提出了一种统一的LLM服务框架，旨在解决LLM预填充(prefill)和解码(decode)阶段不同资源需求导致的效率问题，通过在预测到TBT(Time-Between-Tokens)延迟恶化时，自适应地在单个GPU内进行SM级别空间复用，按需隔离这两个阶段的执行。
2. DuetServe集成了注意力感知roofline模型以预测延迟、一个动态GPU分区优化器以平衡SM（libsmctrl），以及一个无中断执行引擎以消除CPU-GPU同步开销并确保并发执行。
3. H100/CUDA13.0 python实现。与现有最先进的LLM服务框架相比，DuetServe在保持低生成延迟的同时，将总吞吐量提高了1.3倍，有效融合了聚合和分离执行的优势。

DuetServe是一种统一的LLM（大型语言模型）服务框架，旨在在维持高吞吐量的同时，满足计算密集型Prefill阶段和内存受限型Decode阶段严格的延迟SLO（服务等级目标）。现有的LLM服务方法面临挑战：聚合模式下，Prefill和Decode共享GPU资源会导致阶段间干扰，从而降低TBT（Time-Between-Tokens）；而分离模式（Disaggregation）虽然提高了延迟，但却因模型和KV Cache的重复造成资源浪费。DuetServe的核心思想是在单个GPU内实现Disaggregation级别的隔离，它默认以聚合模式运行，并在预测到TBT劣化时动态激活SM（Streaming Multiprocessor）级别的GPU空间多路复用（Spatial Multiplexing）。该系统仅在需要时通过细粒度、自适应的SM分区来解耦Prefill和Decode的执行，从而在拥塞威胁到延迟SLO时提供阶段隔离。

DuetServe由三个紧密耦合的组件构成：

1.  **Attention-Aware Roofline Analytical Modeling（注意力感知型Roofline分析模型）**：
    该模型用于预测迭代延迟，从而帮助调度器提前检测潜在的TBT违规。它基于操作符、计算和内存特性估算模型前向传播的延迟。模型将操作符分为三类：
    *   **Token-Level Operators（Token级操作符）**：这类操作符的成本仅取决于批次中处理的Token总数（Prefill和Decode Token之和），例如线性投影（Linear Projections）、层归一化（Layer Normalization）和激活函数（Activation Functions）。对于线性操作符，给定$n$个总Token、嵌入维度$d$、线性输入维度$d_i$、输出维度$d_o$和元素大小$s$，计算量（FLOPs）和内存开销（Bytes）分别为：
        $F_{lin} = 2nd_i d_o$
        $B_{lin} = nd_i s + d_i d_o s + nd_o s$
        Token级操作符的延迟通常估计为：$t_{tok} = \max(F_{tok}/\Pi_{SM}, B_{tok}/B_{HBM})$，其中$\Pi_{SM}$是活跃SM的计算吞吐量，$B_{HBM}$是HBM（高带宽内存）带宽。
    *   **Sequence-Level Operators (Attention)（序列级操作符，即注意力）**：注意力操作的成本取决于批次中每个请求的查询Token和KV Cache序列长度。对于注意力头数$h_q$、键值头数$h_{kv}$和头维度$d_h = d/h_q$，每个请求的FLOPs和内存字节数分别为：
        $F_{attn/req}(q, c) = 4h_q q(q + c)d_h + 2h_q q(q + c)$
        $B_{attn/req}(q, c) = 2h_q qd_h s + 2h_{kv} (q + c)d_h s$
        其中$q$表示调度的查询Token数量，$c$是缓存的键值Token数量。模型会遍历批次中的每个请求，计算其注意力延迟并进行聚合：$t_{attn} = \sum_{r=1}^{|R|} \max(\frac{F_{attn/req}(q_r, c_r)}{\Pi_{SM}}, \frac{B_{attn/req}(q_r, c_r)}{B_{HBM}})$。
    *   **Communication Operators（通信操作符）**：当LLM服务跨多个GPU进行时，如Tensor Parallelism，会引入GPU间通信开销。例如，Ring AllReduce的通信成本建模为：
        $t_{allreduce} = 2(N-1)\alpha + \frac{2(N-1)B_{lin\_o}}{NB_{NVLink}} + \frac{N(N-1)B_{lin\_o}}{\Pi_{SM}}$
        其中$N$是GPU数量，$\alpha$是启动延迟，$B_{lin\_o}$是线性操作符输出张量的大小，$B_{NVLink}$是所有NVLink连接的聚合单向带宽。
    总模型延迟估算为：$t_{total} = L \cdot t_{block} + t_{cls}$，其中$L$是层数，$t_{block}$是每个Transformer Block的延迟，$t_{cls}$是最终线性分类器的延迟。

2.  **GPU Partitioning Configuration Optimization（GPU分区配置优化）**：
    一旦Roofline模型预测到TBT违规，DuetServe会决定如何分配GPU资源以维持延迟保证并最大化整体吞吐量。系统初始化时，会分析每个可能的SM分区大小下可实现的计算吞吐量$\Pi_{SM}(S)$和内存带宽$B_{HBM}(S)$。对于一个候选分割，将$S_d$个SM分配给Decode，$S_p = S - S_d$个SM分配给Prefill，预测延迟为：
    $t_p(S_p) = f_{roofline}(\text{R}_{prefill}, \Pi_{SM}(S_p), B_{HBM}(S_p))$
    $t_d(S_d) = f_{roofline}(\text{R}_{decode}, \Pi_{SM}(S_d), B_{HBM}(S_d))$
    当激活SM分区时，系统目标是找到一个配置$(S_p, S_d, k)$，使得总Token吞吐量最大化，同时满足延迟约束：
    $\max_{S_p, S_d, k} \frac{k \cdot T_{decode} + T_{prefill}}{\max(k \cdot t_d(S_d), t_p(S_p))}$ s.t. $t_d(S_d) \le \tau_{TBT}$
    其中$T_{decode}$是每次Decode步长生成的Token数，$T_{prefill}$是Prefill批次中的Token数，$\tau_{TBT}$是预定义的TBT延迟边界。该优化倾向于将更多SM分配给Prefill任务以降低其延迟，同时为Decode分配满足TBT约束的最小SM数量。

3.  **Interruption-Free Kernel Dispatching and Look-Ahead Decode Execution（无中断内核分派和预先Decode执行）**：
    为了实现Prefill和Decode的并发执行，DuetServe为两者各初始化一个专用的CUDA Stream。确定最佳SM分区配置后，调度器利用`libsmctrl`将每个Stream绑定到其指定的SM区域，确保两个工作负载独立执行且互不干扰。Prefill和Decode的GPU Kernel由CPU在其各自的Stream上下文中并发调度。
    为最小化CPU端的调度开销，DuetServe利用CUDA Graph捕获Decode执行。在初始化阶段，系统将Decode Kernel序列记录为可重用的CUDA Graph，从而实现高效的图重放（Graph Replay），启动延迟可忽略不计。Prefill执行无法被捕获为CUDA Graph，因为其Attention Kernel通常表现出动态张量形状和可变控制流。因此，Prefill Kernel由CPU单独启动，而Decode Kernel通过缓存的CUDA Graph集体启动。由于启动CUDA Graph的开销小于0.5ms，而Prefill Kernel的调度开销可能达到几十毫秒，调度器总是首先启动Decode执行以防止CPU引发的停顿。
    为进一步减少连续Decode步长之间的同步开销，DuetServe引入了预先Decode执行（Look-Ahead Decode Execution）机制。通过为每个请求预分配多个KV Cache槽位，并提前准备未来$k$个Decode步长的所有元数据，CPU可以连续启动$k$个预记录的CUDA Graph，无需等待中间同步，从而实现跨多个Decode迭代的连续GPU执行，消除频繁的CPU-GPU同步停顿。

DuetServe在Qwen3-8B和Qwen3-14B模型上，使用Azure Code、Azure Conversation和Mooncake三种真实工作负载进行了评估。实验结果表明，DuetServe在保持低生成延迟（TBT）的同时，总吞吐量比最先进的LLM服务框架提高了高达1.3倍。在多GPU环境下，DuetServe的自适应空间多路复用也能有效扩展，维持低解码延迟，同时避免了完全分离式Prefill-Decode配置固有的低效率和不平衡问题。消融研究（Ablation Study）验证了Roofline模型的准确性，并证明了自适应SM分区相对于静态分区的显著优势。通过在CPU和GPU活动上的分析，DuetServe展示了其在动态工作负载下通过在空间共享和时间共享之间切换，维持平衡利用率和高并发性的能力。


## Bullet
Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

https://arxiv.org/abs/2504.19516 中山大学 ASPLOS26

https://github.com/zejia-lin/BulletServe

https://github.com/zejia-lin/Bullet.git

1. 针对LLM服务中计算密集型预填充和内存密集型解码阶段不匹配导致的GPU利用率低下问题，提出了Bullet系统，通过时空GPU资源共享实现阶段并发执行。
2. Bullet通过结合精准的性能估算器、面向SLO的任务调度器进行细粒度阶段协调，以及轻量级的计算资源管理器动态分配GPU流多处理器（SMs）来实现。
3. H100/H20/A100 Bullet通过最大限度提高GPU利用率，在吞吐量方面平均提升1.26倍（最高达1.55倍），并持续满足延迟约束，显著优于现有先进方法。
基于MPS, libsmctrl Hopper, cuda12.6 SGLang

LLM（Large Language Model）服务中，由于计算密集型的Prefill阶段与内存受限的Decode阶段计算特性差异，导致GPU利用率低下。现有方法，如混合批处理（hybrid batching），试图通过将这两个阶段组织在一起以解决此问题，但这往往以牺牲吞吐量或延迟为代价，使得大量GPU资源未能充分利用。

本文识别了导致GPU利用率低下的两个关键根本原因：1) Prefill阶段由于波量化（wave quantization）和注意力（attention）瓶颈，导致计算利用率不佳；2) 混合批处理（hybrid batching）过度优先考虑延迟而非吞吐量，从而浪费了计算资源和内存带宽。

为解决这些问题，本文提出了Bullet，一个新颖的时空编排系统（spatial-temporal orchestration system），旨在通过细粒度的阶段协调来消除这些低效率。Bullet支持Prefill和Decode请求的并发执行，并根据实时性能建模动态配置GPU资源。通过整合SLO（Service Level Objective）感知的调度和自适应资源分配，Bullet在不牺牲延迟目标的情况下最大化了GPU利用率。

Bullet的核心方法围绕其四大关键组件：性能评估器、SLO感知任务调度器、计算资源管理器和并发执行引擎。

1.  **性能评估器 (Performance Estimator)**
    Bullet使用一个准确且低开销的性能评估器，它基于SM-scaling Roofline Model (SRM) 来预测Prefill和Decode阶段在不同SM（Streaming Multiprocessor）分配下的延迟。
    *   **SM-scaling Roofline Model (SRM):** SRM通过分析GPU的计算性能和内存带宽随SM数量变化的关系来估算内核延迟。给定某个核的FLOPs ($f_{lopk}$) 和内存事务量 ($mem_k$)，其在 $N_p$ 个SMs上的理论延迟 $T'_{k,p}$ 可估算为：
        $$ T'_{k,p} = \left(f_{lopk} \cdot \min\left(\frac{f_{lopk}}{mem_k} \cdot D_p, C_p\right)\right)^{-1} $$
        其中，$C_p = C_{peak} \cdot N_p / N$ 表示在 $N_p$ 个SMs上的计算性能，$D_p = D_{peak} \cdot \min(1, N_p / N_d)$ 表示在 $N_p$ 个SMs上可达的内存带宽（$N_d$ 是内存带宽饱和的拐点）。
    *   **争用建模 (Modeling Contention):** Bullet进一步考虑了Prefill和Decode核在不同SMs上并发执行时，内存子系统和网络带宽的争用。即使在并发执行时，内核延迟也保持稳定。通过量化“内存拷贝”和大型GEMM（如LLM中的up-gated layer）并发执行时的性能下降来模拟最坏情况下的内存争用。
    *   **离线分析与在线校准 (Profiling and Online Calibration):** 在离线阶段，Bullet一次性收集SM-scaling Roofline Model所需数据，并对稀疏的Execution State (ES) 值进行测量，用于初始校准因子 $\alpha_{p,ES} = T^{measured}_{p,ES} / T'_{p,ES}$。在运行时，模型通过连续收集在线数据进行微调，以适应动态负载，预测开销在微秒级别。

2.  **SLO感知任务调度器 (SLO-aware Task Scheduler)**
    作为系统的核心协调者，SLO感知任务调度器负责Prefill和Decode任务的时空调度。
    *   **动态调度策略:** 调度器持续监控系统状态（包括执行状态ES、Prefill进度PS和每请求延迟RS），并使用性能评估器预测潜在的SLO违规。Prefill调度器按层（layer-wise）启动内核，而Decode调度器则将内核打包为单个CUDA Graph以减少启动开销。
    *   **资源动态配置:** 当检测到SLO违规风险时，调度器贪婪地搜索最优的资源配置和调度决策，通过调用计算资源管理器来重新分配SMs。它优先为Prefill分配SMs，以缩短TTFT（Time-To-First-Token），但同时确保Decode的TPOT（Time-Per-Output-Token）满足SLO。在请求高负载期间，Prefill阶段可以暂时获得更多SMs，甚至可能暂时暂停Decode，以快速处理排队请求，避免雪崩效应。

3.  **计算资源管理器 (Computational Resource Manager)**
    为了实现细粒度且低开销的SM资源配置，Bullet采用了基于SM屏蔽（SM masking）的技术。
    *   **SM屏蔽 (SM Masking):** 与NVIDIA Multi-Process Service (MPS) 高达700MB的内存开销和静态策略不同，Bullet利用`libsmctrl_set_stream_mask` API来设置CUDA Stream的元数据，从而将该Stream中启动的所有内核限制在指定的SM子集上。这种方法提供了微秒级的运行时开销和零额外的内存占用，支持GPU资源的即时重新配置，以快速适应动态系统状态。

4.  **并发执行引擎 (Concurrent Execution Engine)**
    Bullet的并发执行引擎由独立的Prefill和Decode进程组成，通过高效的通信和内存共享机制实现无缝协作。
    *   **共享内存架构:** 引擎通过OS管理的共享内存（如`/dev/shm`）交换全局系统状态和请求元数据，实现低延迟通信。
    *   **统一GPU内存池:** 模型权重和KV Cache在引擎启动前被分配在一个统一的GPU内存池中。通过`cudaIpcGet/OpenMemHandle` API，不同引擎可以共享访问这些内存区域，无需在Prefill完成后进行KV Cache的数据传输，仅需异步发送元数据。原子锁机制确保了并发内存分配和释放的正确性。
    *   **异步控制流:** Bullet的控制平面独立于CPU和GPU的执行流程，通过共享缓冲区进行主动通信，从而减少了频繁同步的需求，实现了并发内核提交。

在真实世界工作负载上的实验评估表明，Bullet在吞吐量和延迟方面均显著优于现有SOTA系统。Bullet实现了平均1.26倍（最高1.55倍）的吞吐量提升，同时持续满足延迟约束。相较于SGLang-1024，Bullet的TTFT平均缩短13.5倍，端到端加速1.86倍。Bullet在P90尾延迟方面表现出色，将TTFT尾延迟从SGLang-1024的显著高值降低至0.31s，从而大幅提升了SLO达标率。此外，Bullet将GPU的SM激活周期利用率平均提升至86.2%，Tensor Core利用率提升11.8%，内存带宽利用率提升19.3%。其控制平面（元数据传输、性能预测、资源重配置）的平均开销均在微秒级别，对整体性能影响微乎其微。


## DASH
DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training

https://arxiv.org/abs/2601.21824 字节 上海交大冷静文团队，ICLR2026

https://github.com/SJTU-Liquid/deterministic-FA3

1. 提出 DASH 框架，旨在通过将确定性反向传播建模为有向无环图（DAG）上的调度问题来优化关键路径长度，从而解决大型语言模型（LLM）训练中**确定性 FlashAttention 带来的显著性能损失**。
2. DASH 包含两种互补的调度策略：Descendig Q-Tile Iteration（一种通过反向查询块遍历减少因果注意力（causal attention）流水线停滞的启发式方法）以及 Shift Scheduling（一种在 DAG 模型中理论最优的调度算法）。
3. 在 NVIDIA H800 GPU 上，DASH 将**确定性注意力的吞吐量提高了高达 1.28 倍**。

DASH（Deterministic Attention Scheduling for High-Throughput）的调度框架，旨在解决大型语言模型（LLM）训练中可重现性所必需的确定性反向传播操作所带来的显著性能损失。在广泛使用的 FlashAttention-3 实现中，**确定性反向传播的吞吐量可能比非确定性版本低 37.9%** 归因于梯度累积操作必须序列化以保证数值一致性。这种性能损失源于计算和梯度规约阶段调度次优，导致硬件利用率低下。
<img width="810" height="383" alt="image" src="https://github.com/user-attachments/assets/104f7fe8-f3fa-46ce-b14b-1fbe4dc97648" />
<img width="816" height="303" alt="image" src="https://github.com/user-attachments/assets/30a1b962-985e-4ff0-8ec1-dc5558ee6126" />
<img width="988" height="297" alt="image" src="https://github.com/user-attachments/assets/84ac6d36-abe6-4351-99de-208f6eb78791" />

为了应对这一挑战，本文将确定性注意力机制的反向传播形式化为一个在有向无环图（DAG）上的调度问题，并推导出最小化关键路径长度的调度方案。基于此，DASH 框架封装了两种互补的调度策略：

1.  **Descending Q-Tile Iteration（降序 Q-Tile 迭代）**：这是一种针对因果注意力（causal attention）的启发式方法。它通过反转查询块（Q-block）的处理顺序来加速依赖解析并减少流水线停顿（pipeline stalls）。对于因果掩码，传统的调度方式会导致明显的流水线气泡（pipeline bubbles），因为后续的计算依赖于先前的规约完成。通过倒序处理 Q-block，更短的任务首先完成，从而更快地释放 SM（Streaming Multiprocessor）资源，使得后续的注意力头能够更紧密地衔接，显著提高流水线效率。其总执行时间约为 $m \cdot \frac{(n+1)(c+r)}{2} + (n-1) \cdot r$，其中 $m$ 是注意力头数量，$n$ 是 SM 数量，$c$ 是计算成本，$r$ 是规约成本。

2.  **Shift Scheduling（移位调度）**：这是一种在作者的 DAG 模型中理论最优的调度算法，旨在减少全掩码（full mask）和因果掩码（causal mask）下的流水线停顿。该策略通过一种相移的计算任务到 GPU SM 的分配方式，创建一个完美交错的执行模式。其核心思想基于一个引理：在包含并行同构链的 DAG 中，如果通过添加零权重依赖边来保持其原始关键路径长度，则每条添加的边 $(u, v)$ 必须满足 $\text{depth}(u) \le \text{depth}(v)$。

    *   **全掩码下的最优调度**：对于全掩码，每个 KV-tile 的工作负载是均匀的。移位调度通过循环分配 KV 块给 SM 实现。具体而言，SM$_i$ 按照 $(i, i+1, \ldots, n-1, 0, \ldots, i-1)$ 的顺序处理 KV 块。这种循环分配自然地为任何给定的 dQ 块创建了无冲突的、顺序的规约次序，直接满足了上述引理的条件，从而达到理论最优。其总执行时间为 $T_{full\ opt} = m \cdot n \cdot (c+r)$。
    *   **因果掩码下的对称移位调度**：因果掩码会导致严重不平衡的工作负载。对称移位调度通过对称配对原则来解决：SM 共同处理 KV 块 $i$ 和 $n-1-i$，将最长的任务与最短的任务配对，以此类推，从而平衡每个 SM 的任务链长度。它采用两阶段调度：第一阶段对稠密的左下矩形应用循环移位；第二阶段通过“工作负载折叠”（workload folding）处理剩余的三角形区域，将其逻辑地映射到概念上的方形区域，并从主对角线开始遍历。这种等价性确保了工作负载平衡，每个 KV 块的计算连续性，以及满足引理的深度单调累积，最终消除了所有流水线气泡。其总执行时间为 $T_{causal\ opt} = m \cdot \frac{(n+1)(c+r)}{2}$。

在 NVIDIA H800 GPU 上的实证评估表明，DASH 显著缩小了确定性注意力与基线之间的性能差距。所提出的策略将注意力反向传播的吞吐量提高了最高 1.28 倍。然而，实验也揭示了理论最优性与实际应用之间的权衡：在极高的序列长度下（例如 16384），**全掩码的移位调度由于需要频繁的跨 SM 通信，导致 L2 缓存访问延迟**（特别是在访问远程 L2 缓存段时）成为瓶颈，其复杂的依赖图对这种通信开销更敏感，从而略微降低了性能。在因果掩码下，当 `headdim` 较大时（例如 128），对称移位调度因为更高的寄存器使用量，可能触发寄存器溢出（register spilling）到较慢的本地内存，导致性能下降，此时更简单的降序 Q-Tile 迭代反而表现更好。这表明硬件限制（如寄存器压力和跨 SM 通信延迟）会影响算法的实际效益。

在端到端性能评估中，DASH 在 transformer 块级别实现了显著加速。对于因果模型，整体性能提升了 2% 到 10%；对于全掩码模型，提升了约 4%，平均加速约 5%。确定性内核确保了梯度比特级一致，而传统非确定性内核导致约 $O(10^{-4})$ 的梯度偏差。本文强调，尽管确定性会带来额外的性能开销，但对于 LLM 训练的可重现性至关重要。


## MetaAttention
MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

https://dl.acm.org/doi/pdf/10.1145/3774934.3786444 微软 上海交大陈海波等 PPoPP2026

https://github.com/SJTU-IPADS/MetaAttention

1. 现有Attention优化方法难以应对多样化的Attention算法和不断演进的硬件平台，因其手写内核代码量大、灵活性差且移植困难。
2. MetaAttention通过将Attention**抽象为相关性评分和聚合两大核心操作**，并引入**可定制函数和基于IntermediateTensor的两层调度策略**，实现了统一的Attention机制编程和自动化性能优化。
3. **NVIDIA H100**和AMD MI250等硬件上，性能可**媲美手工优化库**，对未支持配置提供高达10.4倍的加速，同时显著降低了开发代码量。

现有的高性能 `Attention` 实现，如 `FlashAttention` 和 `Mamba2`，通常是为特定算法或硬件平台手工优化的，这导致它们难以泛化到其他算法变体或不同硬件后端，并且开发成本高昂。
<img width="499" height="509" alt="image" src="https://github.com/user-attachments/assets/3faaf82e-7585-4e36-92e1-d0760ea9230b" />
<img width="1066" height="220" alt="image" src="https://github.com/user-attachments/assets/fda3bd0a-4ff5-4c9f-b017-94ca6dfcee11" />
<img width="446" height="480" alt="image" src="https://github.com/user-attachments/assets/dbb87a56-79a2-4345-8421-a93e689c1590" />
<img width="428" height="290" alt="image" src="https://github.com/user-attachments/assets/70aa8ca6-76ec-484c-ab7e-752ab59a533d" />

### 论文深入解析

**1. 问题背景与挑战**
`Attention` 机制是 `LLMs` 的核心，其计算量在模型训练和推理中占据主导地位，尤其随着序列长度的增加，这一比例持续上升（例如，在 `Llama-3.2-3B` 中，当序列长度从 2048 增加到 8192 时，`Attention` 的计算时间占比从 55% 上升到 82%）。然而，优化 `Attention` 面临多重挑战：
*   **计算与内存需求高昂**：需要精细的内存管理和并行计算策略。
*   **手工优化成本高**：现有高性能库（如 `FlashAttention` 系列）依赖于大量人工编写的 `CUDA` 或 `Triton` 内核，这些内核针对特定硬件和配置硬编码了执行策略（如融合、并行和流水线），导致其灵活性和可移植性差。例如，`FlashMLA` 的 `Multi-head Latent Attention` 实现需要超过1000行 `CUDA` 代码。
*   **算法多样性激增**：研究人员不断提出新的 `Attention` 变体，如 `Sigmoid Attention`、`Linear Attention`（如 `Mamba2`）、`Sparse Attention`（如 `Seer Attention`），以及需要非标准张量维度（如 `DeepSeek MLA` 和 `RetNet`）的变体。这些变体与传统 `Attention` 模式的微小偏差，都可能导致现有优化库失效或性能急剧下降。
*   **硬件平台差异大**：`NVIDIA` `A100`、`H100` 与 `AMD` `MI300X` 等不同 `GPU` 架构在 `tile` 大小、内存层级和流水线策略上存在差异，使得为不同平台进行优化变得更加复杂和耗时。

**2. MetaAttention 的核心洞察与统一抽象**
`MetaAttention` 的关键洞察在于，尽管 `Attention` 变体繁多，但它们都可以被抽象为两个核心操作：
*   **Relevance Scoring (相关性评分)**：计算输入 `token` 之间的两两相似度或交互。这通常通过点积或其他相似度度量来实现。
*   **Aggregation (聚合)**：利用相关性评分将上下文信息整合到每个 `token` 的表示中。

基于此，`MetaAttention` 提出了一个统一的 `Attention` 模板，该模板固定了 `Relevance Scoring` 和 `Aggregation` 的核心操作，并通过可定制的函数来扩展。该模板捕获了 `Attention` 机制的本质，同时提供了灵活性，使其能够适应各种 `Attention` 变体。
标准 `Attention` 机制可表示为：
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
`MetaAttention` 将这一过程分解为：
*   `relevance = relevance_scoring(Q[i], K, state)`
*   `state = aggregate(relevance, V, state)`

**3. 编程接口与可定制性**
`MetaAttention` 的编程接口允许用户通过定义输入张量形状、`Attention` 模式和可定制函数来设计 `Attention` 机制。
*   **`Attention` 模式**：
    *   **Parallel Pattern (并行模式)**：适用于需要全局上下文信息（如传统 `softmax Attention`、`DeepSeek MLA`、`RetNet` 并行模式）。 `relevance scoring` 和 `aggregation` 实现为并行矩阵乘法，形式为 $scores = matmul(Query, Key)$ 和 $state = matmul(scores, Value)$。
    *   **Recurrent Pattern (循环模式)**：适用于迭代遍历序列，将上下文信息存储在固定大小的隐藏状态中（如 `Mamba2`、`RetNet` 循环模式）。`relevance scoring` 和 `aggregation` 迭代计算，形式为 $output = matmul(Query, h)$ 和 $h = h + matmul(Key[i], Value[i])$。
*   **可定制函数 (`Customizable Functions`)**：这些函数应用于中间张量，支持元素级转换（如缩放、掩码）或全局张量调整（如归一化）。
    *   **Modification (Mod)**：支持细粒度的元素级转换，例如在标准 `softmax Attention` 中将 `Query` 张量按 $1/\sqrt{d_k}$ 缩放，或在稀疏 `Attention` 中应用稀疏掩码。
    *   **Row-wise Normalization (RowNorm)**：支持跨张量行的全局调整，如行级 `softmax` 归一化或数值稳定技术。
    *   **RowNorm Online Interface**：为支持在线处理（如 `FlashAttention` 中的在线 `softmax`）而设计，分块处理数据，避免中间结果写入全局内存。它包含 `online_prologue`（初始化状态）、`online_forward`（每块内部计算，更新状态）和 `online_epilogue`（最终化计算）三个组件。

**4. 核心方法论：MetaAttention Runtime**
`MetaAttention` 的核心创新在于其 `Attention Runtime` 和 **两层调度策略**，实现了在高性能下的自动化优化。
*   **调度空间 (`Scheduling Space`)**：由两个关键组件定义：
    *   **`IntermediateTensor` (中间张量)**：代表 `Attention` 计算过程中设备内存中的所有瞬态张量。其属性包括：
        *   `tile`：张量 `tile` 形状，影响计算并行度与片上内存消耗的平衡。
        *   `mem`：内存位置（如全局内存、共享内存、寄存器），影响延迟、带宽和资源可用性。
        *   `pipelineStage`：流水线阶段（如内存复制、计算），决定了缓冲需求和调度灵活性。
    *   **`DeviceConfig` (设备配置)**：提供硬件特定的约束，如：
        *   `basetile`：目标硬件上计算的最佳 `tile` 形状，与硬件的矩阵乘法指令和内存事务对齐。
        *   `memoryInfo`：可用的内存层级及其容量信息。
*   **两层调度策略 (`Two-Layer Scheduling Policy`)**：
    `MetaAttention` 采用两层调度策略来生成最优执行计划，即 `IntermediateTensor` 的属性配置。
    *   **Tile Config Scheduling (外层)**：负责探索 `IntermediateTensor` 的 `tile` 大小属性。
        1.  以 `Attention` 计算图和 `DeviceConfig` 作为输入。
        2.  枚举输出张量的所有可能 `tile` 大小。
        3.  通过计算图将这些 `tile` 大小传播到所有中间张量，生成一系列 `tile_graphs`。
        4.  对于每个 `tile_graph`，调用内层的 `Tile Resource Scheduling`，并通过性能分析进行评估，最终选择最佳 `tile` 配置。
    *   **Tile Resource Scheduling (内层)**：负责确定给定 `tile` 配置下 `IntermediateTensor` 的内存位置和流水线阶段属性。
        1.  初始化所有中间张量到最高的可用内存层级（例如，寄存器），以最小化内存 `I/O` 开销。
        2.  通过枚举未配置属性（如流水线阶段）来生成候选计划。
        3.  根据 `DeviceConfig` 中的硬件约束（如内存容量）检查计划的可行性。
        4.  如果找不到有效的计划，则迭代地将张量降级到较低的内存层级（例如，从寄存器到共享内存，或从共享内存到全局内存），并重新尝试计划生成。这一过程平衡了延迟和片上资源利用。

*   **代码生成与执行 (`Code Generation & Execution`)**：
    *   **可定制函数的编译**：用户定义的可定制函数被追踪成一个张量 `DAG`。 `DAG` 中的每个节点代表一个计算原语（如元素级操作或行归约操作）。 `MetaAttention` 将这些原语映射到优化的硬件特定实现：元素级操作采用 `SIMT` 风格并融合到寄存器或片上内存中，行归约操作利用 `intra-warp` 并行归约。
    *   **`Attention Runtime` 实现**：`Attention Runtime` 是一个编排层，将优化的调度计划转化为完整的内核。它包含一套用于并行和循环模式的内核模板，包括在不同内存层级之间移动中间张量的操作和矩阵乘法。
    *   **代码内联 (`Code Inlining`)**：根据调度计划，`Attention Runtime` 选择合适的模板，并将硬件映射的可定制函数直接融合到高性能 `Attention` 执行循环中。这消除了额外的内核启动开销，并受益于相同的内存高效流水线和硬件原生优化。
    *   **硬件后端映射**： `MetaAttention` 利用 `TileLang` 和 `CUTE` 等框架，将内核模板映射到 `NVIDIA` `GPU`（利用 `Tensor Memory Accelerator` (`TMA`) 和 `Tensor Cores`）和 `AMD` `GPU`（利用 `AMD Matrix Cores` 和异步复制单元）等不同硬件后端，以实现峰值性能。

**5. 实验评估与成果**
`MetaAttention` 在 `NVIDIA H100` 和 `AMD MI250` `GPU` 上进行了广泛评估，测试了包括 `Softmax Attention`、`Sigmoid Attention`、`ReLU Attention`、`RetNet`、`Mamba2` 和 `DeepSeek MLA` 在内的10种 `Attention` 机制。
*   **性能提升**：
    *   在 `NVIDIA H100` 上，`MetaAttention` 对于之前不支持的 `Attention` 配置（如 `ReLU Attention`）实现了最高达 **10.4 倍** 的加速，对 `Sigmoid Attention` 实现了平均 **3.6 倍** 的加速。
    *   与高度优化的 `FlashAttention-3` 相比，在 `Diff-Transformers-3B` 的前向计算中实现了 **1.61 倍** 的平均加速，并在其他 `Softmax Attention` 任务上保持了可比性能。
    *   在循环模式 `Attention`（如 `Mamba2`、`RetNet`、`Gated Retention`）上，相比 `Flash-Linear-Attention` 实现了前向 **1.66 倍** 和反向 **1.78 倍** 的平均加速。
    *   在 `DeepSeek MLA` 上，性能与手工优化的 `FlashMLA` 库相当，比 `Triton` 提升 **4.6 倍**。
    *   在 `AMD MI250` `GPU` 上，相比基线实现了前向 **3.3 倍** 和反向 **2.0 倍** 的平均加速，证明了其多后端支持能力。
*   **开发效率提升**：`MetaAttention` 显著降低了开发工作量，例如，实现 `Softmax Attention` 仅需 **87 行代码**，而手工优化的 `CUDA` 库需要 **2.7k 行**。
*   **编译时间**：受益于高效的调度策略，框架的编译时间控制在分钟级别（例如，在 `H100` 上编译 `Softmax Attention` 为 46 秒，`Mamba2` 为 82 秒），远低于传统深度学习编译器。
*   **端到端性能**：在 `H100` 上的 `LLM` 推理和训练中，`MetaAttention` 带来了平均 **1.4 倍** 的端到端加速。


## ECO 
ECO: Quantized Training without Full-Precision Master Weights

https://arxiv.org/pdf/2601.22101 Google Research等 2026.1.29

1. 提出了Error-Compensating Optimizer (ECO)，这是一种**无需全精度master weights即可进行量化LLM训练的方法**，通过将**量化误差注入优化器momentum缓冲区**来显著降低内存开销。
2. ECO在不增加额外内存的情况下形成了一个误差反馈循环，使得更新能够直接应用于量化参数，理论分析表明，在衰减学习率下，ECO能收敛到最优点的常数半径邻域，而朴素的master weight移除会导致误差发散。
3. 在30M–800M Transformers、Gemma-3 1B和2.1B SMoE模型的**FP8量化预训练以及DeepSeek-MoE-16B的INT4微调中**，ECO达到了**与有master weights基线模型近乎无损的精度**，并将静态内存使用量降低了高达25%。
<img width="447" height="414" alt="image" src="https://github.com/user-attachments/assets/57b1d52f-4a50-4ed9-b2ff-63d2fa4f1d24" />
<img width="874" height="571" alt="image" src="https://github.com/user-attachments/assets/8f948183-606f-4d24-aad0-69d6e6e197b5" />

传统上，quantized training 方法仍依赖高精度的 master weights 来累积 gradient updates，因为低精度格式难以精确捕获微小更新，易导致更新消失或引入巨大 quantization noise。

master weights 尤其在 Sparse Mixture of Experts (SMoE) 模型中造成显著内存负担。ECO 通过**将updates直接应用于 quantized parameters 来消除 master weights**，并在每个训练 step 后，**将产生的 quantization error 精心注入到 optimizer 的 momentum buffer 中**，从而形成一个无需额外内存的 error-feedback loop。

<img width="798" height="316" alt="image" src="https://github.com/user-attachments/assets/933efcb7-7fd9-4352-b324-aab5d1a61e52" />

## ContextMATH
From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics

https://arxiv.org/pdf/2601.23048 港中文 微软等，2026.2.2 ICLR26

1. 引入了ContextMATH基准测试，旨在评估大型语言模型在**情境化数学推理方面**的能力，该基准测试包含**情景落地**（SG）和**复杂性分级**（CS）两种问题类型。
2. 对61个专有和开源模型的评估显示，在ContextMATH上它们的准确率大幅下降，错误分析表明**不正确的数学问题表述是主要瓶颈**。
3. 问题表述和推理是限制LLM情境化数学问题解决能力的两个互补瓶颈，即使通过场景数据微调，性能差距也仅部分缓解，表明这是一项尚未解决的核心挑战。

<img width="712" height="391" alt="image" src="https://github.com/user-attachments/assets/0ab54168-f4e2-4666-98a6-732938c65b4c" />

尽管LLMs在如AIME等基准测试中已达到近乎专家水平，甚至触及国际数学奥林匹克（IMO）金牌标准，但这种进步未能完全转化为其在真实世界应用中可靠的表现。研究人员将这一差距归因于LLMs在“**情境数学推理**”（contextual mathematical reasoning）能力上的不足，即从**描述性场景中准确地建立数学核心问题的能力**。
<img width="379" height="311" alt="image" src="https://github.com/user-attachments/assets/19b17d52-87a4-4185-92a2-ae43e306fbb7" />

**核心方法学**
为了系统性地探究LLMs的情境数学推理能力，论文引入了名为ContextMATH的基准测试。ContextMATH通过将AIME 2024、AIME 2025和MATH-500（难度级别$\ge 3$）中的抽象问题，转换为两种受控叙述变体来构建：

1.  **场景落地（Scenario Grounding, SG）**：
    *   **目标**：评估模型在数学问题以叙述形式呈现时，能否应用其数学知识，而不增加核心推理难度。
    *   **生成过程**：采用多步骤提示（multi-step prompt）指导LLM（o1-mini）进行迭代生成、自验证和修订。
        *   **步骤1：数学元素映射（Map All Mathematical Components）**：将原问题中的每个数学元素（数字、变量、形状、运算）明确映射到具体的真实世界对应物。例如，抽象的“变量$x$”映射为“油桶的初始数量”。
        *   **步骤2：定义真实世界交互规则（Define the Real-World Rules of Interaction）**：结合步骤1中的对应物，构建连贯的叙述上下文（如工程、物理、物流），将数学运算转换为故事中的行动或规则。例如，数学函数$f(x) = 2x + 3$可转换为“机器的输出通过将输入值翻倍后加上固定的校准偏移量3来计算”。
        *   **步骤3：编写最终问题（Write the Final Problem）**：将上述元素整合成一个简洁的故事。严格要求新问题在数学上与原问题完全等价，避免使用数学术语和符号（如“三角形”、“变量$x$”、“$\cos$”），并确保最终问题清晰且与原问题要求相同的结果形式。

2.  **复杂度量化（Complexity Scaling, CS）**：
    *   **目标**：模拟现实世界中量化条件通常通过间接描述（例如简单计算、相对位置或日常事实）来推断的情况。
    *   **生成过程**：通过提示引导LLM将原问题中的显式条件隐藏在简单、自洽的子问题中。
    *   此外，CS问题还引入了看似合理但与数学解无关的额外信息，以增加认知负荷，同时避免直接使用数学变量，而是赋予它们具体的真实世界含义。

**质量控制**
ContextMATH的构建非常重视数据质量。通过广泛的先导研究优化了生成指令和流程。三位拥有计算机科学高级学位和数学竞赛背景的专家独立审查每个项目，确保叙述的可信度、清晰度、数学等价性、可解性，并通过GPT-5和Gemini进行测试。只有通过所有检查的问题才会被采纳。

**评估与发现**
研究评估了46个开源模型和15个专有模型（包括GPT-5和DeepSeek-R1）。核心发现如下：

1.  **情境复杂性构成普遍瓶颈**：无论开源还是专有模型，在情境化场景中性能均出现显著下降。例如，DeepSeek-R1-0528-Qwen3-8B在AIME 2024上的准确率从75.0%降至CS变体的39.6%；即使是参数量庞大的GPT-5，在AIME 2025-CS上也下降了26%。这表明情境数学推理是LLMs尚未解决的根本性挑战。

2.  **模型规模缓解但未解决问题**：虽然更大的模型通常表现出更强的鲁棒性，但模型规模的扩大并不能完全消除情境化推理的失败。例如，OpenMath-Nemotron系列在AIME 2024-CS上的性能下降：1.5B模型下降77%，而14B和32B模型分别下降43%和51%。

3.  **初步SFT提升鲁棒性，但进一步专业化调整无效**：初期的Supervised Fine-tuning (SFT)能够提升模型的原始和情境准确率，但后续的SFT或Reinforcement Learning (RL)等更专业化的训练（例如R1-Distil-Qwen-7B $\to$ AceMath-RL-Nemotron-7B），尽管可能提升在抽象基准上的表现，却往往未能改善在SG或CS任务上的性能，有时甚至导致更大的下降。这表明当前的后训练方法可能过度专业化于规范格式，强化了模式识别而非情境推理能力。

**错误模式分析：公式化瓶颈**
研究进一步通过定性和定量分析揭示了性能下降的根本原因：
1.  **定性分析**：通过GPT-5对模型错误进行分类，**发现在情境化问题（SG和CS）的失败案例中，约80%的错误源于“公式化错误”**（incorrect mathematical interpretation），即从叙述中错误地解释了问题的数学结构，而非计算或逻辑错误。这证实了**从叙述中正确提取数学公式**是主要弱点。

**提升情境数学推理能力**
论文探索了两种训练策略：
1.  **端到端微调（End-to-End Fine-tuning）**：
    *   使用DeepMath-103K的原始数据和5万个经过验证的合成情境数据对Qwen3-Base系列模型进行微调。
    *   **训练方案**：仅原始数据（+SFTOri）、仅合成情境数据（+SFTSyn）、以及两者的平衡混合（+SFTMix）。
    *   **结果**：**SFT显著提升了情境推理能力**。情境数据监督（SFTSyn）在情境问题上比原始问题（SFTOri）更有效。混合训练（SFTMix）在所有规模模型上提供了最佳整体平衡，表明抽象和情境数据具有互补性。同时，这些改进也泛化到AMC23和Math-Perturb等其他基准，表明有针对性的情境训练并未损害抽象推理能力，反而增强了对分布变化的鲁棒性。尽管取得显著进展，但即使是经过SFTMix训练的大模型，在AIME 2024/25-CS问题上的解决率仍低于40%，表明仍有巨大的提升空间。

2.  **训练专用公式化模型（Training a Dedicated Formulation Model）**：
    *   探讨是否可以通过训练一个专门负责公式化的模型，并将其与现有求解器结合来改善情境推理。
    *   **实验设置**：使用Qwen3-8B和Qwen3-14B作为公式化和推理模型。比较了直接求解、未经微调的公式化模型与推理模型结合、以及经过微调的公式化模型与推理模型结合三种情况。
    *   **结果**：**直接求解情境问题（不经过单独的公式化阶段）表现最佳**。增加未经微调的公式化阶段导致性能轻微下降，而经过微调的公式化模型则导致性能大幅崩溃。研究认为，单独从情境-原始问题对中有效学习公式化能力是困难的。


## Qrita TopP
Qrita: High-performance Top-k and Top-p Algorithm for GPUs

https://arxiv.org/abs/2602.01518 伯克利 2026.2.2

1. 提出了一种用于 GPU 的高性能 **Top-k 和 Top-p** 算法，旨在克服大型语言模型 (LLM) 采样中现有截断方法的**效率**和确定性挑战。
2. 引入 Gaussian σ-truncation 大幅缩小搜索空间，并结合 Quaternary pivot search with duplication handling，从而实现更快的迭代和确定性结果。
3. 与 vLLM、SGLang 和 FlashInfer 等现有 LLM 执行引擎相比，Qrita 在提供相同输出的同时，**吞吐量最高可提高 2 倍**，内存使用量减少一半。

该算法解决了传统 Top-k 和 Top-p 实现中存在的效率问题，即基于排序的方法在 GPU 上计算和内存开销大，而随机方法则会改变算法输出的确定性。
在**H100 和 RTX4090**上与 PyTorch、vLLM、SGLang 和 FlashInfer 等基线进行了广泛评估。
<img width="889" height="617" alt="image" src="https://github.com/user-attachments/assets/94a2eca2-954c-4611-a3ff-25b19aeb90eb" />

吞吐量： 大部分测试中持续超越所有基线，RTX4090 上甚至实现了高达 2x的吞吐量提升。即使与 FlashInfer 和 SGLang 等功能受限（只返回单个 token）的解决方案相比，Qrita 也能保持竞争力或表现更优。
内存使用： Qrita 的内存使用与所有基线持平或更低，除了 vLLM 的 Top-k (它使用就地 RadixSelect)。Qrita 的截断缓冲区是其主要内存开销，但其带来的性能提升足以抵消。
高斯 \sigmaσ\sigmaσ 截断： 该技术提供了最大的加速，使延迟降低了 74%。即使存在约 0.4ms 的开销，它仍是 Qrita 整体性能提升的关键。
四分位枢轴搜索： 尽管在截断命中（搜索空间已小）时略慢于二分搜索，但在回退到全词汇表搜索时，它带来了 2.3ms 的显著绝对延迟降低，证明了其在处理大规模数据时的优势。
重复值处理： 引入的开销微乎其微（约 3.5%），但对于保证输出的确定性至关重要。
分级自适应调优： 这项优化至关重要，将执行延迟降低了近 30%。

## Quartet2
Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation

https://arxiv.org/pdf/2601.22813 2026.2.2 Dan Alistarh团队

code: https://github.com/IST-DASLab/Quartet-II

1. 提出了一种名为MS-EDEN的新型无偏量化：基于NVFP4微缩放格式，相较于现有**随机舍入（SR）方法，其量化误差降低超过2倍**。
2. 基于MS-EDEN构建了Quartet II：一种**全NVFP4线性层量化方案**，fwd中引入更精细的4/6尺度动态选择，bwd用MS-EDEN替代了SR量化。
3. 1.9B参数、38B token的**LLM预训练中持续提高了准确性**，并提供专为NVIDIA Blackwell GPU优化的CUDA内核，实现了**相较于BF16高达4.2倍的速度提升**。

现有的量化训练方法，特别是那些依赖 SR (Stochastic Rounding) 进行无偏梯度估计的方法，在 NVFP4 精度下会牺牲模型表示能力，并导致相对于标准 FP16 和 FP8 训练显著的精度损失。

**核心方法：MS-EDEN (MicroScaling EDEN) 无偏量化**

本文提出了一种名为 MS-EDEN 的新型无偏量化例程，专门为微尺度 (micro-scaled) 格式设计。MS-EDEN 的量化误差比 SR 低两倍以上，其核心创新在于将**随机性从单个 FP4 值转移到微尺度因子** (microscale factors)，同时保持预期上的可证明无偏性。

为了实现无偏性，梯度估计必须满足：$\forall x_d \in \mathbb{R}^d : \mathbb{E}_{\omega}[\text{Q}(x_d, \omega)] = x_d$，其中 $\text{Q}$ 是量化操作符，$\omega$ 是随机性来源。

NVFP4 格式使用 E2M1 浮点编码表示值，并结合两级尺度：每 16 个值一个 FP8 E4M3 组尺度，以及每个张量一个 FP32 全局尺度。传统的 SR 量化器 QSR 定义为：
<img width="557" height="647" alt="image" src="https://github.com/user-attachments/assets/50578038-d6aa-43dc-a7c5-c01faa92eeb2" />

传统的 EDEN 方法通过引入一个偏置校正因子 $S = \frac{\langle x, x \rangle}{\langle \text{RHT}(x, \omega), \text{Q}(\text{RHT}(x, \omega)) \rangle}$ 来确保无偏性，即 $Q_{\text{EDEN}}(x, \omega) = S \cdot Q(\text{RHT}(x, \omega))$。然而，该 $S$ 因子需要高精度表示（精细表示0.94～1.06），与 NVFP4支持的最小scale FP8不兼容（仅能表示精度1.0625）。

<img width="748" height="292" alt="image" src="https://github.com/user-attachments/assets/f366f03f-24e2-44b9-a3d1-9d20f051bb8e" />


MS-EDEN 实现了无偏性，并显著降低了 MSE (9.8e-3)，远低于传统 SR 的 MSE (23.5e-3)。理论上，MS-EDEN 量化器 $Q_{\text{MS-EDEN}}$ 满足 $\mathbb{E}_{\omega_{\text{RHT}},\omega_{\text{SR}}}[\text{RHT}^{-1} (b_x, \omega_{\text{RHT}})] = x$，其中 $b_x = Q_{\text{MS-EDEN}}(x, \omega_{\text{RHT}}, \omega_{\text{SR}}, s)$。

Quartet II 是一个全 NVFP4 线性层计算方案，它结合了：
*   **前向传播 (Forward Pass)：** 采用原生 NVFP4 尺度 (每 16 个元素一个 FP8 E4M3 尺度，以及一个每个张量的 FP32 尺度) 的 RTN FP4 量化，并辅以 "Four-over-Six" (4/6) 局部尺度选择启发式算法。4/6 算法通过评估两个潜在的尺度因子（4.0 和 6.0）来选择 MSE 较低的那个。研究表明，4/6 与原生 NVFP4 尺度（1x16gs）协同效应更好，能带来显著的精度提升。前向传播中量化后的权重和激活值会被保存用于反向传播。
*   **反向传播 (Backward Pass)：** 首先生成一个 RHT 旋转矩阵。然后，保存的量化权重和激活值被反量化、转置，并与张量 E 和 ET 一起使用 MS-EDEN 进行重新量化，以获得无偏估计。这些量化后的张量在 NVFP4 Tensor Cores 中进行乘法运算。乘积输出无需进一步处理，因为旋转在 GEMM 的内维上相互抵消。


<img width="541" height="399" alt="image" src="https://github.com/user-attachments/assets/c777a679-612e-4a7e-9203-f648db4251ab" />
<img width="936" height="406" alt="image" src="https://github.com/user-attachments/assets/1cd9a7b2-8c21-45a9-be3c-d4eb99127940" />
<img width="400" height="338" alt="image" src="https://github.com/user-attachments/assets/08d6a15e-4cfa-4d30-bd5e-f74fd1e0f05c" />

**实验验证与扩展**

*   **LLM 预训练：** 在 Llama 2 架构模型上（参数量从 30M 到 200M，C4 数据集）进行了验证。结果显示，MS-EDEN 在所有适用场景下都持续优于 SR。即使 MS-EDEN 需要在反向传播中重新量化权重，其全量化版本（图 1(e)）仍优于无需重新量化权重的全量化 SR 版本（图 1(d)）。
*   **Nanochat 预训练：** 在更大规模（560M 和 1.9B 参数，38B tokens，FineWeb-Edu 数据集）上进行了验证。Quartet II 稳定，并使预训练损失相对于 BF16 的差距减少了 15-25%，优于现有 NVFP4 方法。零样本基准测试显示，与 BF16 相比，不同 QAT 方法之间的性能差异不显著。

**内核支持和性能**

*   **融合重向量化内核优化：** 传统的 NVFP4 量化中的全局最大值归约操作构成全局屏障，阻止了操作的完全融合，导致需要两个内核传递，加倍了内存带宽和矩阵乘法成本。
*   **事后范围对齐 (Post Hoc Range Alignment)：** 为 MS-EDEN 引入了一种硬件感知的实现启发式方法。在第一个内核中，尺度被对齐到 E8M3（一种在 BF16 中表示的扩展范围 FP8 代理），而不是预先计算的 AbsMax。张量值被尺度除后舍入到 FP4，形成 ER-NVFP4。同时计算 EDEN 校正因子。在第二个内核中，加载 E8M3 伪尺度和缩减后的 FP32 全局最大值，将伪尺度移入 FP8 可表示范围，应用 EDEN 校正，并通过 SR 量化为 FP8。该优化显著减少了内存移动量（理论带宽节省约 20%），第二个内核的延迟也远低于第一个。
*   **加速效果：** Quartet II 在线性层操作上实现了相对于 BF16 超过 4 倍的训练加速，并比现有 FP4 训练内核（如 Quartet）提升了约 70%。5090单卡1B LLM 预训练中，实现了相对于 BF16 超过 2.4 倍的总吞吐量提升，这主要归因于内存节省使得可以使用更大的 micro-batch size。


## Hot Mess
The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity? 

https://arxiv.org/abs/2601.23045 ICLR26 Anthropic等 2026.1.30

https://github.com/haeggee/hot-mess-of-ai 

中文解读：https://mp.weixin.qq.com/s/_5P7SYkZEZ8_-m5AUS9U-w
本研究的期望（expectation）不是针对训练随机性（\varepsilonε\varepsilonε）进行，而是针对固定模型在同一任务上的输入（如few-shot上下文）和输出（采样）随机性进行。这意味着通过多次运行模型，观察其在相同输入下的行为变化来估计方差。
<img width="951" height="487" alt="image" src="https://github.com/user-attachments/assets/69f4a1b2-ed0b-4baa-8b40-961f3263ae3f" />
<img width="965" height="638" alt="image" src="https://github.com/user-attachments/assets/ead807d0-b695-473a-bc73-4f3f625f80fe" />
<img width="1132" height="683" alt="image" src="https://github.com/user-attachments/assets/a841a8ea-e89c-4557-97d5-522cd9e0bc87" />

**论文在以下任务上进行实验**：

Multiple Choice Tasks (**GPQA, MMLU**): 模型需选择正确答案，目标明确。
Agentic Coding (**SWE-BENCH**): 评估AI解决GitHub问题的能力，通过单元测试衡量成功率。
**Safety and Alignment**(Model-Written Evals - MWE): 评估模型在AI风险方面的自我报告行为，包括多项选择和开放式格式。
**Synthetic Settings**: 训练Transformer模型模拟优化器在病态二次损失函数上的轨迹，以控制设定验证理论。
**Survey**: 引用了Sohl-Dickstein (2023) 的人类主观调查，评估AI、人类、组织等的智能与不连贯性。

模型： 探索了Anthropic的**SONNET 4、OpenAI的O3-MINI、O4-MINI**等前沿模型，以及QWEN3系列（1.7B至**32B**）模型。
采样策略： 除非另有说明，每道题至少**进行30次采样以估计偏差和方差**。对于GPQA和MMLU，采样除了生成随机种子外，还使用了不同的few-shot上下文。
主要发现：

**推理长度与不连贯性**：
模型推理时间**越长、采取的行动越多，其失败就越不连贯**。这在多项选择、Agentic Coding和安全任务中均观察到。
对于**推理时间更长的任务，其错误主要由方差主导**。
即使任务固定且推理预算固定，**自然变异（即模型在某些问题上“过度思考”）也导致更高的不连贯性**。

**模型规模、智能与不连贯性**：
**更大、能力更强的模型有时反而更不连贯**。
任务**复杂性依赖**： 随着模型规模的扩大，简单任务上的不连贯性会降低（更连贯），而**困难任务上的不连贯性会增加（更不连贯）。这表明虽然整体错误率下降，但对于难题，方差成为限制因素**。
**合成优化器实验**： 在模拟优化器轨迹的受控合成任务中，随着模型**规模的增加，模型能更快地降低偏差，但方差降低得较慢**，**使得最终的性能更多地由方差主导，即模型变得更不连贯**。这支持了AI系统在追求目标时可能变得更混乱的观点。
人类调查结果： 人类受试者主观评价更智能的实体（AI模型、非人类生物、人类组织）也往往被评价为更不连贯，与实验结果一致。

**推理预算和集成**：
推理预算： 增加推理预算（模型API提供的“thought”时间）可以轻微降低不连贯性，并提升性能。但这种效果远不如模型自然思考长度的变动对不连贯性的影响大。
**集成（Ensembling）： 对模型多次尝试的结果进行集成，能显著降低方差**（接近1/E1/E1/E1/E，其中EEEE是集成大小），从而降低不连贯性。这表明错误纠正机制可以提升连贯性。

讨论与启示：
论文指出，不连贯性增加的可能原因包括：LLMs作为动态系统，难以被约束为固定损失的有效优化器，因为在状态空间中，作为优化器的动态系统是零测度的；以及方差在轨迹中通常会累积，除非有主动的纠正机制。
研究结果表明，当先进的AI系统执行复杂任务时，其失败更可能是以不一致的方式发生，而不是因为持续追求一个稳定的错误目标。这意味着在AI风险研究中，相比于奖励劫持或目标错配等“系统性失准”，可能需要更多关注由不可预测的“工业事故”造成的风险。

## Kascade
Kascade: A Practical **Sparse Attention Method for Long-Context LLM Inference**

https://arxiv.org/pdf/2512.16391 2025.12 微软老印
https://github.com/microsoft/kascade 待开源

1. Kascade是一种**无需训练的稀疏注意力方法**，它利用了后**softmax注意力固有的稀疏性**以及高权重**key在相邻层间结构的稳定性**来加速长上下文LLM推理。
2. 该方法通过动态规划算法选择“**锚点层**”以计算**精确的Top-k索引**，并在“**复用层”中重用这些索引**，同时结合了头部重映射和tile-level池化等机制以确保高效和准确性。
3. Tile-Lang中实现了FlashAttention内核的修改版本，支持prefill和decode阶段; H100 GPU了最高**4.1倍的decode注意力加速**和2.2倍的prefill注意力加速，并在LongBench和AIME-24等长上下文基准测试中保持了与密集注意力相近的精度。
- 主要优化attn 计算，不减少内存容量需求，KV caches仍可能很大
  
Kascade是一种无需训练的稀疏注意力方法，旨在解决长上下文LLM推理中注意力操作带来的延迟瓶颈。该方法基于两个核心观察：
1) post-softmax 注意力天然稀疏；
2) high-weight keys 的身份在相邻层之间保持稳定。Kascade通过在**少量anchor layers中计算精确的Top-k索引**，并在**中间的reuse layers中重用这些索引**，从而实现显著的性能提升，同时保持与稠密注意力相当的准确性。

<img width="1163" height="630" alt="image" src="https://github.com/user-attachments/assets/a5a350d5-7cf4-459f-a598-f5ebd4cc28b2" />

<img width="797" height="475" alt="image" src="https://github.com/user-attachments/assets/bde4d6eb-4e19-449b-bcd3-eb3490516425" />

<img width="771" height="551" alt="image" src="https://github.com/user-attachments/assets/22fe02fa-727c-4703-bd0f-47be16cbce1b" />

## FinDEP
Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

https://arxiv.org/pdf/2512.21487 2025.12.25 港科广楚晓文团队

1. 🛠️ MoE模型推理因其内存密集型特性和现有AF分离 Disaggregated Expert Parallelism (DEP)方法（如PPPipe）的粗粒度调度，导致GPU空闲时间长、性能次优。
2. 💡 为解决这些问题，FinDEP提出了一种细粒度任务调度框架，通过**将计算和通信任务划分为更小的段，并将其表述为一个优化问题**，以最大化任务重叠和资源利用。
3. 🚀 该框架利用性能模型和高效的多项式时间求解器，在PCIe单机多卡 和H20多机，DeepSeekv2-236b Qwen2356b-A22b模型；pytorch推理吞吐量比现有最先进方法提高了高达1.61倍，并在不到一秒内实现配置自适应。
   
MoE (Mixture-of-Experts)架构因其在LLM (Large Language Models)中能够以亚线性（sublinear）计算开销扩展模型规模的优势而被广泛采用。然而，MoE模型的推理需要大量的内存，特别是在注意力层（attention layers）需要访问KV (Key-Value)缓存，以及在专家层（expert layers）仅利用有限数量的专家时，内存需求尤为显著。现有研究尝试通过DEP (Disaggregated Expert Parallelism)将注意力层和专家层分配到两个专用的GPU组——AG (Attention Group)和EG (Expert Group)——以提高推理效率。但现有DEP方法对包含共享专家（shared experts）的现代MoE模型支持有限，并且对GPU组中复杂的通信和计算任务的调度探索不足，导致推理性能次优。
为解决这些问题，本文提出了FinDEP，一种用于DEP的细粒度（fine-grained）任务调度算法，旨在通过最大化任务重叠（maximal task overlap）来提高MoE模型的推理吞吐量（throughput）。FinDEP整合了三项关键创新：

细粒度任务划分（Fine-Grained Task Partitioning）：将AG和EG中计算密集型任务（computation-intensive tasks）和通信任务（communication tasks）划分为多个更小的任务。具体而言，通过将每个任务的输入张量（input tensor）分割成多个片段（segments），创建了rrrr个小任务。这种划分允许进行动态调度，无论MoE模型是否包含共享专家，都能提高吞吐量。它将时间消耗任务，包括EG中的计算、A2E（Attention-to-Expert）和E2A（Expert-to-Attention）中的通信，以及AG中的计算，通过沿输入张量的batch维度（对于AG）和token维度（对于EG）进行切分，分别产生了r_1r1r_1r1​和r_2r2r_2r2​个更小的任务。这种细粒度处理使得更大的并行化成为可能。

<img width="975" height="358" alt="image" src="https://github.com/user-attachments/assets/ba72eb9c-2196-4127-8504-f6582a4dec1f" />
<img width="920" height="281" alt="image" src="https://github.com/user-attachments/assets/e1043d14-a254-4a20-b127-65cdced9faf1" />

## token-filtering
Shaping capabilities with token-level data filtering 

https://arxiv.org/pdf/2601.21571 2026.1.30

https://github.com/neilrathi/token-filtering

1. 💡 本研究提出了一种在预训练阶段通过**token级别数据过滤**来塑造大型语言模型能力的新方法，其在**削弱不良能力方面**比文档级别过滤更有效。
2. 📈 该方法在模型规模越大时效果越显著，能使**遗忘领域（如医学）的计算效率降低7000倍**，并表现出比现有遗忘干预措施更高的对抗性微调鲁棒性。
3. ⚙️ 研究还引入了一种**基于稀疏自编码器的新型标记方法**，证实该过滤方法成本效益高、可容忍噪声标签，并且出人意料地增强了模型在遗忘领域进行拒绝训练的对齐能力。


该论文题为《通过令牌级数据过滤塑造能力》，旨在解决大型语言模型 (LLM) 在**预训练期间无意中习得有害能力的问题**。现有方法多为事后干预，例如通过 RLHF 或机器遗忘技术来降低风险，但这些方法容易被对抗性攻击（如越狱或微调）绕过，形成“猫捉老鼠”的局面。本文提出一种在预训练阶段通过数据过滤来塑造模型能力的自然替代方案，并证明了其在代理任务（移除医学能力）上的高效性、鲁棒性和经济性。

**核心思想与方法**

本文的核心贡献在于提出了**令牌级数据过滤**（token-level data filtering），并证明其优于传统的文档级数据过滤（document-level data filtering）。

1.  **能力划分：** 将模型能力划分为“**遗忘域**”（forget domain）和“**保留域**”（retain domain）。目标是**最大程度地削弱遗忘域能力，同时保持或尽可能少地损害保留域能力**。本研究选取移除“医学能力”作为遗忘域，保留“生物学能力”等相关领域。

2.  **数据过滤策略：**
    *   **文档级过滤：** 根据文档内容整体判断是否属于遗忘域，并移除整个文档。
    *   **令牌级过滤：** 识别文档中与遗忘域相关的特定令牌（token），并对其进行干预。提出两种令牌级过滤方法：
        *   **损失掩码（Loss Masking）：** 在模型的反向传播（backpass）过程中，将遗忘域令牌的梯度移除，使其不对模型参数更新产生贡献。在正向传播（forwards pass）中，模型仍能看到这些令牌，保留了上下文连贯性。
        *   **移除（Removal）：** 将遗忘域令牌替换为特殊的 `<|hidden|>` 令牌，并掩码其损失。这种方法以牺牲部分上下文连贯性为代价，实现对遗忘域内容的彻底移除。

3.  **计算慢化（Compute Slowdown）：** 衡量过滤效率的指标。定义为训练一个未经过滤的基线模型达到与过滤模型相同损失所需的计算量，与过滤模型实际使用的计算量的比值。高计算慢化表明过滤的效率更高，即过滤方法使模型在遗忘域上“扩展更差”（scale worse）。

**分类器训练技术**

为了实现精确的令牌级过滤，需要一个高质量的令牌级分类器。本文提出了一种低成本、高效率的弱监督流水线：

1.  **真值标签获取（Sourcing Ground-Truth Labels）：**
    *   利用预训练的稀疏自编码器（Sparse Autoencoders, SAE）来识别与遗忘域相关的语义特征（latents）。具体地，使用 Gemma 2 9B 模型的 SAE（层 31）提取 forget-domain latents。
    *   通过大型语言模型（如 Claude 3.5 Haiku 和 Claude Sonnet 4）解释这些 latents，并根据其解释将 latents 分类为“医学”或“非医学”相关。
    *   令牌被标记为“医学”当其在多个医学相关 latents 上激活值显著高于平均（例如，高于平均 4 个标准差且至少在两个医学 latents 上），或当其在至少一个医学相关 latent 上激活且与已标记为医学的相邻令牌相邻时，进行迭代扩展以覆盖令牌序列。
    *   此方法旨在生成“可爬坡”（hill climbable）的带噪数据集，以训练能够从有噪标签中泛化并学习“正确”真值方向的分类器。训练数据集包含 128k 份文档，其中 8.2M 令牌用于训练，并使用 PubMed、bioRxiv 等学术论文和 FineWeb-Edu 网页数据。

2.  **分类器模型训练（Training Classifiers）：**
    *   **双向上下文的重要性：** 令牌的含义高度依赖上下文（例如，“virus”在生物学和计算机安全中意义不同），因此使用双向上下文对分类至关重要。
    *   **线性探测（Linear Probes）：** 在预训练模型（如 BERT 变体）的固定表示上拟合线性探测器进行分类，以提高对虚假关联的鲁棒性。
    *   **小型任务特定基模型的优势：** 相比大型通用模型（如 ModernBERT-large，395M 参数），小型、任务特定的基模型表现更优且成本更低。研究发现：
        *   一个 65M 参数的 RoBERTa-like 模型在 FineWeb-Edu 上预训练（使用掩码语言建模目标）即可实现超越通用基线的 F1 分数。
        *   通过联合训练左右两个自回归模型来构建双向语言模型（biLM），将它们的表示拼接后进行分类，进一步提升性能。
        *   在领域上采样（domain-upsampled）的语料库（例如，50% PubMed + 50% FineWeb-Edu）上预训练 biLM，能进一步提高分类器性能。
        *   最终，一个 224M 参数的 biLM 分类器在验证集上达到 0.856 F1 分数，在测试集上达到 0.894 F1 分数。这表明领域特定的预训练能帮助模型构建对医学分类更显著的表示。
    *   **文档级分类器：** 使用相同的 224M biLM，但在文档级别进行标注，达到 0.922 F1 (val) 和 0.941 F1 (test)。

**实验设置与评估**

1.  **模型训练：** 训练计算最优的 Transformer 模型，参数规模从 61M 到 1.8B。模型采用增强版 GPT-2 架构，包含 RoPE、ReLU2 激活函数和 pre-RMSNorm 归一化。优化器采用 AdamW，学习率调度使用 μP 技术。
2.  **指令微调（Instruction Tuning）：** 针对最大的 1.8B 模型，进行指令微调以评估多项选择（multiple choice）和自由回答（free-response）能力。多项选择使用自定义数据集，聊天模型使用 smol-smoltalk mix。
3.  **评估指标：**
    *   **文本困惑度（Text Perplexity）：** 用于评估小模型在医学、生物学和通用非医学文本上的能力。
    *   **多项选择：** 针对医学知识使用 MedMCQA、MedQA-USMLE 和 MMLU 医学子集；保留域能力使用 MMLU 生物学、STEM 和非 STEM 子集。
    *   **自由回答：** 针对医学问题使用 HealthSearchQA，并用 Claude Sonnet 4 评估回答的关联性、连贯性和正确性；对照组使用 Alpaca 数据集。

**主要实验结果**

1.  **令牌级过滤的压倒性优势与可扩展性：**
    *   **Pareto 优势：** 令牌级过滤在达到同等遗忘域损失（例如，医学困惑度）的同时，能保持更低的保留域损失（例如，生物学困惑度），表现出更高的精度和同等召回率。
    *   **规模效应：** 令牌级过滤的有效性随模型规模增大而显著增强。对于 1.8B 参数模型，令牌移除可实现遗忘域上 7000 倍的计算慢化，而文档级过滤仅约 30 倍。这意味着随着模型规模增长，过滤在削弱 undesired capabilities 方面变得更加高效。
    *   **多项选择和自由回答评估：** 过滤后的模型在医学相关 MCQ 任务上表现接近随机，而在保留域任务上性能无明显下降。在自由回答中，令牌级过滤导致模型在医学查询上回答的连贯性、关联性和正确性显著下降（分别降低 4 倍和 10 倍）。

2.  **过滤的鲁棒性：**
    *   **对抗性微调（Adversarial Finetuning）对比：** 过滤方法比最先进的机器遗忘技术 RMU 更具鲁棒性。在对抗性微调攻击下，RMU 达到基线性能所需的微调令牌数量（作为预训练计算量的比例）随规模增长急剧下降，而数据过滤方法的鲁棒性下降速度则慢得多。对于 1.8B 模型，令牌移除比 RMU 鲁棒性高 13 倍。

3.  **对齐训练的兼容性：**
    *   **遗忘域分类：** 过滤训练后的模型仍能可靠地识别遗忘域令牌（通过线性探测），且识别能力随模型规模增大而提升。
    *   **拒绝训练（Refusal Training）：** 令人惊讶的是，令牌级过滤实际上使得遗忘域的对齐（alignment）更容易。模型在 HealthSearchQA 上生成拒绝回答的比例比基线高 2 倍，而在 Alpaca 上没有显著增加。这与先前关于毒性数据过滤可能使对齐更困难的发现形成对比。研究推断，这可能是因为过滤训练后的模型学会在“见过”和“未见过”的令牌之间进行区分，而非复杂地判断内容“是否有害”。

4.  **标签质量与过滤效果：**
    *   **有噪标签的影响：** 人为引入标签噪声会显著降低过滤效果，且这种影响呈幂律关系。
    *   **规模效应弥补噪声：** 即使是“糟糕”的分类器，在足够大的模型规模和高召回率（牺牲精度）的设置下，也能实现有效的过滤。通过大幅提高过滤比例，模型能接近理想的“高遗忘损失/低保留损失”前沿。
    *   **弱标签泛化（Weak-to-Strong Generalization）：** 令牌级分类器能够从粗粒度标签（如文档级或句子级标签）中进行有效泛化，性能仅略低于使用细粒度标签训练的分类器。然而，文档级分类器则难以从弱标签中泛化。


## OPSD自蒸馏
Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models 

https://arxiv.org/pdf/2601.18734 Meta等 2026.1.27

1. 提出了一种名为 **On-Policy Self-Distillation** (OPSD) 的新框架，其中单个大型语言模型通过**使用特权信息充当教师**并仅观察问题充当学生来实现自蒸馏。
2. OPSD 通过在**学生自己的生成轨迹上提供密集的、token-level 的监督**，有效解决了传统off-policy**蒸馏的分布不匹配**和强化学习的**稀疏奖励问题**，且无需外部教师模型。
3. Qwen3（1.7B, 4B, 8B），OPSD 在多个数学推理基准上优于SFT和GRPO，实现了**4-8倍的token效率提升**，并且效果随模型规模增大而增强，同时全词汇蒸馏优于采样token蒸馏。
4. 
<img width="799" height="308" alt="image" src="https://github.com/user-attachments/assets/75ab1a9c-eba3-47b5-be39-9fccc037d8f1" />
<img width="800" height="191" alt="image" src="https://github.com/user-attachments/assets/6abb08ff-8100-4252-a666-4263bf53ca52" />
<img width="800" height="374" alt="image" src="https://github.com/user-attachments/assets/b7ef4e67-0176-4db3-aeaf-8491ef279519" />

本文提出了一种名为 On-Policy Self-Distillation (OPSD) 的新型框架，旨在通过自蒸馏（self-distillation）来提升大型语言模型 (LLM) 在推理任务上的性能。该框架通过让**同一个 LLM 扮演教师和学生的角色**，解决了传统知识蒸馏（Knowledge Distillation, KD）方法中分布不匹配（distribution mismatch）以及需要额外大型教师模型的问题，并克服了强化学习方法如 GRPO面临的计算效率低下、奖励稀疏（sparse reward）及梯度消失（vanishing gradient）等挑战，同时弥补了监督微调（Supervised Fine-Tuning, SFT）在泛化能力上的不足。

<img width="907" height="533" alt="image" src="https://github.com/user-attachments/assets/a6a04127-bf95-47bc-a4b7-8ee58d5c311f" />
<img width="904" height="491" alt="image" src="https://github.com/user-attachments/assets/ed2415be-ca4a-4c29-b06b-399759ff7807" />
<img width="809" height="644" alt="image" src="https://github.com/user-attachments/assets/defb1af3-517d-498b-a9ab-efa23f8ba423" />


**实验与结果**
本文在 Qwen3 系列模型（1.7B, 4B, 8B）上进行了实验，并使用 OpenThoughts 的数学推理子集进行训练（包含 30K 个带有 CoT 的问题-解决方案对）。评估基准包括 AIME 2024、AIME 2025、HMMT 2025 和 Amo-Bench 等竞赛级数学数据集。对照基线为 SFT 和 GRPO。
<img width="743" height="430" alt="image" src="https://github.com/user-attachments/assets/28c4208d-3b71-43aa-96e3-7be8643dc911" />
<img width="795" height="281" alt="image" src="https://github.com/user-attachments/assets/4f517a3e-60c3-4a2f-9e73-7cbf102ff794" />

*   **主要性能**：OPSD 持续优于 SFT，并在 4B/8B 模型规模上匹配或超越 GRPO，在 1.7B 模型上性能相当。
*   **令牌效率**：OPSD 在每个问题上仅需采样 1 个响应，而 GRPO 需要 8 个响应，显著提升了样本效率。在 Qwen3-4B 模型上，OPSD 实现了 4-8 倍的令牌效率提升，用更少的生成令牌（2k vs. GRPO 的 16k）达到更高的性能，显著降低了采样成本和训练时间。
*   **模型规模效应**：实验发现 OPSD 的性能提升随着模型规模的增大而增强（在 1.7B 上提升有限，在 4B/8B 上提升显著），这验证了“自合理化（self-rationalization）能力需要足够的模型容量”的假设。
*   **生成长度效应**：增加策略内采样时学生的生成长度（从 1k 到 2k 再到 4k 令牌）可以提供更多的教师信号，从而持续提高 Pass@K 性能。
*   **学习目标比较**：全词汇表 Logits Distillation (GKD 风格) 优于采样令牌蒸馏（Lu & Lab 2025 风格）。全分布（full distribution）提供了更丰富的监督信息，但代价是更高的峰值内存使用。

**相关工作**
OPSD 与多种 LLM 训练范式相关，包括其他 OPD 方法（但 OPSD 采用自蒸馏而非独立教师）、通过 SFT 和 RL 改进 LLM 推理（SFT 易记忆而 RL 更泛化），以及 LLM 自训练（如 Self-Instruct, Self-Align, Context Distillation, ReST, STaR）。OPSD 的独特性在于其是策略内、令牌级的自蒸馏，模型在特权信息（ground-truth solutions）的引导下从自身输出中学习。

## SDFT 自蒸馏
Self-Distillation Enables Continual Learning 

https://arxiv.org/pdf/2601.19897 2026.1.28 MIT, Improbable AI Lab, ETH Zurich

https://github.com/idanshen/Self-Distillation

https://mp.weixin.qq.com/s/7BKxOoS5iqah27OrxY9BzA 

1. 提出了一种名为**自蒸馏微调（SDFT）的新方法**，它利用模型的**上下文学习能力**，通过将受**demo文案引导的模型作为教师进行自蒸馏**，从而直接从专家演示中实现策略学习，以解决持续学习中的遗忘问题。
2. SDFT 在技能学习和知识获取任务中始终优于监督微调（SFT），不仅显著**提高了新任务的准确性，还大幅减轻了灾难性遗忘**，展现出更强的泛化能力。
3. 主要基于Qwen2.5-7B-Instruct模型（最大14b），SDFT 使单一模型能够随时间逐步习得多项技能，同时保留现有能力，并且其有效性会随着模型规模和上下文学习能力的增强而提升，为基于演示的持续学习提供了实用路径。
- 代码和数据开源 单张H200可运行
 
<img width="790" height="376" alt="image" src="https://github.com/user-attachments/assets/1634259d-1d1a-48a3-9dbe-d9034aa07bff" />
<img width="875" height="319" alt="image" src="https://github.com/user-attachments/assets/9f896dfd-f85e-4cf0-a1a7-860a5d08c016" />


该论文提出了 Self-Distillation Fine-Tuning (SDFT)，这是一种简单有效的方法，旨在使 Foundation Models 能够进行 Continual Learning。Continual Learning 的核心挑战在于，模型在学习新技能或知识的同时，如何避免 Catastrophic Forgetting，即不退化其现有的能力。

**核心问题与现有方法的局限性：**
传统上，Foundation Models 部署后参数保持静态，无法持续学习和改进。现有解决 Continual Learning 的方法主要包括：
1.  **On-policy Reinforcement Learning (RL)**：虽然能有效减少遗忘，但其应用受限于需要明确的 Reward Function，这在许多实际场景中是难以获得的。
2.  **Learning from expert demonstrations**：
    *   **Supervised Fine-Tuning (SFT)**：这是从专家演示数据中学习的主流范式。然而，SFT 本质上是 Off-policy 的，导致在模型适应新任务或领域时出现严重的 Catastrophic Forgetting 和泛化能力不足。
    *   **Inverse Reinforcement Learning (IRL)**：旨在从演示中推断 Reward Function，然后进行 On-policy RL。尽管在概念上优美，但 IRL 通常需要对 Reward Structure 做出强假设，且扩展性差，实践中应用受限。

**SDFT 方法论：**
SDFT 通过利用模型的 In-context learning (ICL) 能力，直接从演示中实现 On-policy Learning。其核心思想是让模型在两个角色中运行：一个 Demonstration-conditioned 的“教师”和“学生”本身。
<img width="761" height="379" alt="image" src="https://github.com/user-attachments/assets/535e7f29-a90a-49f7-aef5-df2e6b94aeb8" />


**实验与结果 (Experiments and Results)：**
论文在两个 Continual Learning 场景中评估了 SDFT：
1.  **Skill Learning**：在 Science Q&A, Tool Use 和 Medical 任务中，模型需要掌握新的专业技能。
2.  **Knowledge Acquisition**：模型需要整合在 Pre-training Data 截止日期之后发生的真实新事实知识。

**主要实验结果：**
*   **减少 Catastrophic Forgetting 和提高泛化能力**：SDFT 在新任务上实现了更高的准确性，同时显著减少了在现有能力（通过 HellaSwag, MMLU 等通用基准衡量）上的 Catastrophic Forgetting，表现优于 SFT 及其增强版本（如 SFT + Re-invoke）和 DFT。
*   **多任务持续学习 (Multi-Task Continual Learning)**：在顺序学习三个不同任务的挑战性实验中，SDFT 使得单个模型能够稳定地积累技能，同时保持对先前学习任务和预先存在能力的性能，而 SFT 则表现出严重的干扰和性能振荡。
*   **模型规模效应 (Effect of Model Scale)**：SDFT 的有效性与模型的 ICL 能力紧密相关。随着模型规模的增大（例如从 3B 到 14B），SDFT 相对于 SFT 的性能增益显著增加，表明其在大模型上的优势。
*   **在缺乏推理数据的情况下训练推理模型 (Training Reasoning Models Without Reasoning Data)**：SDFT 能够使模型在只有最终答案而无中间 Chain-of-Thought (CoT) 的演示数据上进行有效适应。与 SFT 导致推理行为退化（表现为响应变短和准确率下降）不同，SDFT 通过使学生模型匹配 Demonstration-conditioned 教师模型的输出，保留了模型的内部推理风格和深度，提高了任务性能。
*   **On-policy Learning 的重要性**：论文通过消融实验证明，On-policy Learning 是 SDFT 性能优势的关键。即使是从同样的教师模型进行 Offline Distillation，其性能也远不及 SDFT 的 On-policy 方法，这突出了 On-policy Feedback 在持续改进中的核心作用。

**局限性与未来工作 (Limitations and Future Work)：**
*   **计算成本 (Computational Costs)**：SDFT 相较于标准 SFT，需要额外的 On-policy Rollout 生成，导致计算成本增加（约 2.5 倍 FLOPs，4 倍 Wall-clock Time）。
*   **学习到的伪影 (Learned Artifacts)**：学生模型可能继承教师模型的冗余语言模式（如“Based on the text...”），目前通过启发式掩码（Masking）来解决。
*   **模型能力要求 (Requirements for Model Capability)**：SDFT 严重依赖于基础模型的 ICL 能力；小型模型或需要根本性行为转变的场景（例如将非推理模型转变为推理模型）可能表现不佳。
*   **未来工作**：包括将 SDFT 与 On-policy RL 结合、进一步减少遗忘，以及探索从非专家或有噪声的演示中学习。
  
  
## VibeTensor
VibeTensor: System Software for Deep Learning, Fully Generated by AI Agents

From Node.js/Python to PTX assembly: a research deep-learning system fully generated by AI agents.

https://arxiv.org/pdf/2601.16238 2026.1.22 NVIDIA
https://github.com/NVLabs/vibetensor

1.  VIBETENSOR 是一个由**LLM驱动的AI代理“完全生成”的开源深度学习系统软件栈，**旨在探索AI生成复杂系统的能力。
2.  该系统提供PyTorch风格的即时张量库，其C++20核心支持CPU和CUDA，并通过Python及Node.js提供多语言前端，集成了包括autograd、CUDA运行时和诊断性缓存分配器在内的核心组件。
3. 通过构建、测试和差分检查进行验证，该项目展示了**AI代理生成连贯的深度学习运行时**（从API到CUDA内存管理）的可行性，但也揭示了“Frankenstein”合成效应等挑战。
   
<img width="759" height="573" alt="image" src="https://github.com/user-attachments/assets/789b8311-0e34-433a-a5f1-0699a7b01eed" />
<img width="956" height="522" alt="image" src="https://github.com/user-attachments/assets/c31a175a-b8c9-4f87-973a-0ac87dbdaf66" />
<img width="981" height="326" alt="image" src="https://github.com/user-attachments/assets/f6fffce3-c90a-4efd-b121-2befbeb30371" />

**代码量**：C++/CUDA 运行时包含超过 63,000 行非空代码，总代码量（包括插件、Python/Node.js overlay 和测试）超过 20 万行。
**性能**： AI 生成的 kernel 套件提供了 Triton 和 CuTeDSL 实现的微基准测试。例如，在 H100 GPU 上，对于 NanoChat 风格的 Attention (BF16, batch 32, seq 2048)，Triton kernel 在前向和后向传播上分别比 PyTorch SDPA/FlashAttention 快 1.54倍 和 1.26倍。然而，对于小 batch GQA prefill 场景，Triton 可能会落后（前向 0.67倍，后向 0.66倍）。
**端到端训练**： 在 H100 和 Blackwell GPU 上对 Sequence Reversal、CIFAR-10 ViT 和 miniGPT 模型进行了端到端训练验证。VIBETENSOR 虽然能够收敛并达到与 PyTorch 定性一致的结果，但目前性能比 PyTorch 慢 1.7 到 6.2 倍，这与其原型状态和已知的串行化瓶颈（如“Frankenstein”效应）一致。
**多GPU扩展**： 在 Blackwell GPU 上，VIBETENSOR 通过实验性 Fabric 子系统和 CUTLASS-based ring-allreduce 插件（非 NCCL）展示了多 GPU 扩展能力，但 Fabric 并非一个完整的分布式训练运行时。
**“Frankenstein”效应**： 生成系统的一个常见失败模式是，局部正确的子系统组合起来可能导致全局性能不佳。

该系统在约两个月内开发完成，主要依赖 LLM-powered 编码 Agent。人类提供高层需求和优先级，但不进行手工diff review。Agent作为“黑盒”，其工作流包括：

**明确目标与不变性** (Scoped Goal & Invariants)： 人类为Agent设定具体的开发目标和系统应满足的约束条件。
**代码生成与应用** (Code Generation & Application)： Agent根据目标生成代码更改并将其应用到代码库中。
**聚焦测试与验证** (Focused Testing & Validation)： Agent执行编译，并运行针对性的单元测试 (C++使用 CTest，Python使用 pytest)，确保局部功能的正确性。
**系统级组合验证 **(Broadened Validation)： 随着子系统的逐步组合，Agent会扩展验证范围，运行更全面的测试和端到端检查。
## errors modeling
A model of errors in transformers

https://arxiv.org/pdf/2601.14175 2026.1.20

探讨了LLM在**需要确定性输出和重复处理Token（例如算术任务）方面的错误率**。
作者提出，LLM的**错误源于其Attention机制中细微错误的累积**，最终导致错误向量的长度超出某个阈值。
基于这一洞察，论文推导出了一个定量的双参数模型，用于描述准确率与任务复杂度之间的关系。这两个参数——基本“噪声率”r和“合理错误方向”的数量 q ——随Prompt和模型而异。此分析借鉴了物理学中“Effective Field Theory”的视角，将LLM的众多底层参数重组为仅两个主导错误率的有效参数。

用Gemini 2.5 Flash, Gemini 2.5 Pro和DeepSeek R1三种SOTA LLM，对8种不同任务进行了广泛的经验测试，总计使用了20万个不同的Prompt。这些任务包括列表反转、嵌套线性变换、动态规划、汉诺塔、普通加法、算法加法、二进制加法和乘法。
论文还展示了如何通过精心设计的Prompt来降低错误率。例如，在乘法任务中，通过指导模型将数字转换为多项式进行中间计算，Flash模型的准确率显著提高，甚至超越了使用普通Prompt的Pro模型。这表明，通过Prompt引导模型更精确地关注相关Token可以有效减少Attention机制中的噪声积累。

## S3 attention
S3-Attention:Attention-Aligned Endogenous Retrieval for Memory-Bounded Long-Context Inference

https://arxiv.org/pdf/2601.17702 2026.1.29 北邮
1. S3-Attention 提出了一种将**内存受限的长上下文推理**转换为**流式、注意力对齐的内生检索过程的框架**，旨在解决传统 KV cache 扩展性和外部检索器语义不匹配的问题。
2. 该框架通过 **Top-k sparse autoencoders (SAEs) 将 Key 和 Query** 投影解码为稀疏语义特征，构建了 **CPU 上的倒排索引并立即丢弃 KV cache**，从而实现了 O(1) 的 GPU 内存占用。
3. Qwen7b/llama3-8b/Mixtral实验，S3-Hybrid 在 LongBench 基准测试中几乎保留了 full-context 的性能（如 Llama-3-8B 达到 99.4%），并通过有效过滤噪音提升了信息密集型任务的鲁棒性，甚至偶尔超越 full-context 基线。
- 不足：只评测检索/总结类负载，没测试普通问答 数学 代码等。
- 当前实现速度欠优化/延迟高：索引和检索阶段Python 级别的 Posting Lists 和频繁的CPU-GPU 同步
  
<img width="815" height="315" alt="image" src="https://github.com/user-attachments/assets/bede198a-7573-4ffc-815f-e44bbf792f7b" />

为解决大型语言模型 (LLM) 长上下文推理中的内存限制问题而提出的框架。当前 LLM 在处理长上下文时面临两大困境：一是维护完整的 KV cache 会导致 GPU 内存呈线性增长，进而迅速饱和；二是采用外部检索器（如 RAG）往往会遭遇“语义不匹配”问题，即检索到的段落可能在词汇上相似，但在因果关系上与模型的内部推理不相关，从而引入噪声并导致幻觉。S3-Attention 旨在弥合这一差距，将内存受限的推理过程转变为一个流式、注意力对齐的内生检索过程。
S3-Attention 的核心在于将模型的瞬态注意力状态解码为稀疏的特征 ID，并构建一个可搜索的内存索引，而无需保留庞大的 KV cache。其方法论可分为三个主要阶段：
<img width="1190" height="589" alt="image" src="https://github.com/user-attachments/assets/27cae263-e1ad-4f4e-928b-6a7ee8ce09f5" />
<img width="1198" height="148" alt="image" src="https://github.com/user-attachments/assets/4c437148-98b7-4043-b019-86bdc28b4c8e" />

## MARS
MARS: Unleashing the Power of Speculative Decoding via Margin-Aware Verification 

https://arxiv.org/pdf/2601.15498 2026.1.20 港大 麦克吉尔大学等

1. MARS (Margin-Aware Speculative Verification) 提出了一种训练无关的SD验证策略，旨在解决现有方法在LLM低裕度区域因**严格逐token验证导致的效率低下和高回滚成本问题**。
2. 通过引入**Logit Ratio来衡量目标模型对其预测的局部决策确定性**，并在模型对前两个候选项偏好微弱时（即低裕度）**自适应地放宽验证，允许接受次优草稿token**。
3. MARS在8B到235B的多种LLM和任务上，相对于现有基线实现了**显著的推理加速（最高达4.76倍），同时保持了近乎无损的生成质量**。
如果draft token 与目标token.Top2相同，且目标模型 top1/top2 token等logit差异很小（不确定性大），则也接受。
<img width="888" height="510" alt="image" src="https://github.com/user-attachments/assets/3dd717a7-0c35-4b8f-bdb6-703046bacbf1" />

大型语言模型（LLM）的自回归推理因内存带宽限制而面临高延迟问题。Speculative Decoding（SD）通过解耦生成（由轻量级 Draft Model 完成）和验证（由 Target Model 并行完成）来解决此瓶颈。然而，尽管 Medusa 和 EAGLE 等现有方法显著提升了 Drafting 质量，其验证机制仍主要依赖严格的 token-level rejection sampling。这种标准方法隐含地假设 Target Model 在每个步骤都具有明确的偏好，但实际上，LLM 常常在“Low-Margin Regimes”下运行，即排名靠前的候选 token 之间的似然差异统计上可忽略不计。在此类情况下，拒绝看似合理的 runner-up token 带来的信息增益微乎其微，却会产生巨大的 Rollback 成本，导致验证效率低下。

本文提出了一种名为 Margin-Aware Speculative Verification（MARS）的策略。MARS 是一种无需训练（training-free）且领域无关（domain-agnostic）的验证方法，它根据 Target Model 的“局部决策性”（local decisiveness）自适应调整。MARS 通过直接从 Target Model 的 Logits 测量决策稳定性（decision stability），仅在严格验证收益最小化时才放松拒绝条件。重要的是，该方法仅修改验证规则，与现有 Target-Coupled Speculative Decoding 框架完全兼容。

**核心方法（Methodology）**

<img width="1069" height="485" alt="image" src="https://github.com/user-attachments/assets/00f78f56-6aa9-4eea-bfb4-9376634c785c" />

<img width="1067" height="159" alt="image" src="https://github.com/user-attachments/assets/cc5c581f-b37b-4c6a-a275-82f9e6b6b700" />

<img width="864" height="519" alt="image" src="https://github.com/user-attachments/assets/02774d57-abca-4659-8f2e-5fbef9bf94c6" />

**整体性能（Overall Performance）：**
实验结果（如表 1 所示）表明，MARS 在所有模型系列和规模上都一致优于 EAGLE-3，实现了更高的 Speedup Ratio 和更长的平均接受长度 $\tau$。例如，在 Vicuna-13B 上，MARS 实现了 3.74x 的平均 Speedup，$\tau=7.20$，而 EAGLE-3 仅为 3.12x，$\tau=5.64$。在 LLaMA-3.3-70B 上，MARS 达到了 4.76x 的 Speedup。$\tau$ 的持续增加表明在随机解码下 Draft 效率更高，从而实现了可靠的端到端加速。

**精度保持（Accuracy Preservation）：**
尽管 MARS 是一个 lossy 的 SD 方法，其加速解码不保证与标准解码完全一致，但实验（图 3）显示，MARS 在各项任务和模型规模上实现了近乎完全的 Accuracy Recovery（98.1% 至 100%），在 HumanEval 上达到 100%，在 MBPP 上达到 99.2%-100%。在代码基准测试中观察到的退化可忽略不计，这可能是因为偶尔的分歧大多是表面性的（例如命名/格式），很少影响功能正确性。

**消融研究（Ablation Studies）：**
*   **Logit Ratio Threshold $\theta$：** $\theta$ 值控制 MARS 自适应放松的程度。 Speedup 随 $\theta$ 的增加而单调下降，而 Accuracy 通常在 $\theta \approx 0.90$ 附近达到峰值（图 4）。因此，0.90 被选作默认值，以实现 Accuracy-Speed 的良好权衡。
*   **Temperature t：** Speedup 和 $\tau$ 在不同 Temperature (0.2 到 1.0) 下保持相对稳定。Accuracy 随着 Temperature 的升高而降低（基线和 MARS 都如此），但 MARS 的效率提升不受影响。
*   **Draft Length K：** 增加 K 会导致更大的 $\tau$，但 Speedup 并非单调递增。中等 Draft Length（如 K=9）通常能实现最佳加速，过大的 K 可能增加 Draft/Verification 开销。


## LoPA

https://arxiv.org/pdf/2512.16229 上海交大 邓志杰

1. 💡 本文提出LoPA一种无需训练的即插即用算法，旨在通过识别并优化Token Filling Order (TFO) 来解决Diffusion Large Language Models (dLLMs) 推理中Tokens Per Forward pass (TPF) 较低的并行性瓶颈。
2. ✨ LoPA的核心机制是在每次迭代中并行探索多个候选TFO分支，并通过评估分支置信度来选择具有最高未来并行潜力的路径，从而有效提高解码效率。
3. 🚀 结合专门设计的Branch Parallel分布式推理系统LoPA-Dist，LoPA将D2F-Dream模型的TPF提升至GSM8K上的10.1，并在多GPU部署下实现了高达1073.9 tokens/s的单样本吞吐量，显著超越了现有基线。

      
## Tawa
Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References

https://arxiv.org/pdf/2510.14719 2025.12.10 康内尔 NVIDIA等 已经开源 提交到Triton

https://github.com/triton-lang/triton/tree/aref_auto_ws PR:6288

1. 提出了 Tawa，一个自动化编译器，旨在通过异步引用（`aref`）实现现代 GPU 的自动 warp spec，以解决 SIMT 编程模型与任务并行硬件之间的可编程性差距。
2. Tawa 的核心是一个新颖的 `aref` IR 抽象，它表达了**warp 级别的通信意图，使得编译器能够自动将程序划分为生产者-消费者角色并管理多粒度软件流水线**。
3. NVIDIA H100，Tawa对GEMM比cuBLAS12.7提速1.1 倍，Attention 工作负载上比 Triton 提速 1.2 倍，并可媲美手动优化的 CUTLASS 4.0 FlashAttention-3 性能。
   - 不支持B200

<img width="953" height="275" alt="image" src="https://github.com/user-attachments/assets/1c692646-a9a9-49b4-83bd-6209a2c87c18" />
<img width="459" height="233" alt="image" src="https://github.com/user-attachments/assets/99fdb697-472b-4b3e-8246-98dbd89f1449" />

Tawa 在 NVIDIA H100 SXM5 GPU 上使用 CUDA 12.7 进行了评估，与 cuBLAS、CUTLASS、Triton、ThunderKittens 和 TileLang 等先进框架和库进行比较。

**矩阵乘法 (GEMM)**：对于 M=N=8192M=N=8192M=N=8192M=N=8192，K 介于 256 到 16384 的 FP16 和 FP8 GEMM，Tawa 在大多数形状上达到了与高度优化的 cuBLAS 相同的性能水平，同时优于其他通用框架。在 FP16 中，Tawa 平均比 cuBLAS 快 1.01 倍，比 Triton 快 1.13 倍，比 TileLang 快 1.15 倍，比 ThunderKittens 快 1.09 倍。Triton 相对于 Tawa 的劣势在于其采用 Ampere 风格的软件流水线而非利用 Hopper 的硬件 warp 特化。在 FP8 中，Tawa 的优势更为明显，因为更小的瓦片和更快的计算使得内存传输和同步成为瓶颈，而 Tawa 的 warp 特化流水线通过持续数据流缓解了这一问题。

**GEMM 变体**：batched GEMM和grouped GEMM始终优于 Triton，并显著优于 TileLang。这些优势源于 Tawa 基于 aref 的分区和自动 warp 特化，它能够将一个 GEMM 的数据移动与另一个 GEMM 的计算重叠。

**多头注意力MHA**：对于序列长度 L \in [1024, 16384]L∈[1024,16384]L \in [1024, 16384]L∈[1024,16384] 的 MHA，Tawa 实现了与手写 FlashAttention-3 (FA3) 相似的性能，并持续比 Triton 快 1.21 倍。尤其在长序列（L \ge 4K≥4K\ge 4K≥4K）时，Tawa 显著优于 TileLang 和 ThunderKittens，表明其在数据移动和计算之间的重叠更有效。在 FP8 MHA 中，Tawa 的优势甚至更为显著。这些结果表明 aref 提供了一个原理性抽象，能将普通内核转换为数据流流水线，并在不同精度和语义下都表现良好。

**超参数选择与消融研究**：
评估了 aref 大小 D 和 MMA 深度 P 对性能的影响。结果表明，增大 D 有助于预取和隐藏延迟，而适度的 P（1-2）则平衡了重叠和资源压力。持久化内核能持续提升 5-10% 的性能。消融研究显示，自动 warp 特化带来了显著的性能提升（在 GEMM 中提升 3.78 倍，MHA 中提升 2.84 倍），结合协作 warp 组、大瓦片尺寸、持久化内核和 aref 优化，Tawa 最终将性能提升了近七倍。

## Twill
Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs

https://arxiv.org/pdf/2512.18134 2025.12.19 斯坦福，NVIDIA 未开源

1. 提出了 Twill，一个将软件流水线 (SWP) 和 Warp Specialization (WS) 建模为联合优化问题的新系统，旨在为 Tensor Core GPU 自动生成最优调度。
2. Twill 利用现成的约束求解器，通过整合模调度、内存感知和 warp 分配约束，自动发现无启发式、可扩展且保证最优的调度。
3. 针对FlashAttn前向/后向+Hopper/B200, Twill的FA算法重新发现，**并证明专家手动开发的 SWP 和 WS 策略是最佳的**，其性能可达到手调实现的 1-2% 以内。

Twill是首个能够自动推导此类迭代程序最优调度方案的系统，它不依赖启发式方法，易于扩展到新的GPU架构，并能保证生成最优调度。
<img width="930" height="474" alt="image" src="https://github.com/user-attachments/assets/524efbc9-c37c-4f8c-a4b6-16b19cb0be1b" />
<img width="946" height="527" alt="image" src="https://github.com/user-attachments/assets/401d128d-98cd-4401-9929-8f04a702088e" />

## Hummingbird
Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale

https://arxiv.org/pdf/2601.04071 2026.1.7 北大谢涛团队， 中科大等

1. Hummingbird是一种面向SLO的GPU共享-抢占调度系统，通过**微秒级GPU抢占**和有效利用空闲GPU时间片，解决了现有GPU共享技术在确保SLO和最大化效率方面的挑战。
2. 系统包含**kerel split更小抢占粒度(<400usec)**，运行时调度器动态管理，以及利用**NVLink进行分层内存卸载**的内存管理模块。
3. 截获CUDA driver API, 基于SOSP25 NEUTRINO提取PTX然后split kernel；对闭源cublas/cuddn则手工用cutlass替换kernel。部分cross-block sync cuda graph不支持split。
4. llama.cpp作推理；torch作训练。A100+cuda12.6(最多2机16卡；以及H100，L40)，Hummingbird将高优任务的SLO达成率分别提升9.7倍和3.5倍，同时将低优任务吞吐提高2.4x，且高优任务的SLO仅下降不到1%；平均抢占延迟121～165usec。
   
<img width="538" height="322" alt="image" src="https://github.com/user-attachments/assets/4d7b19cb-8486-4882-bc4e-9392c7479191" />

Hummingbird是一个面向服务水平目标（SLO）的GPU调度系统，旨在实现微秒级GPU抢占，并最大限度地利用GPU。针对现有GPU共享技术（包括空间共享和时间共享）在确保SLO的同时最大化效率方面所面临的挑战，Hummingbird通过在闭源GPU上启用微秒级抢占并有效利用空闲GPU时间片来克服这些困难。

**问题背景与现有方案局限性**

随着深度神经网络（DNN）模型（特别是Transformer-based架构如ChatGPT和Gemini）的普及，GPU资源面临前所未有的计算需求压力。然而，现有的GPU分配方案通常采用粗粒度方式，为特定任务独占分配多块GPU以保证SLO，导致GPU利用率极低（如微软GPU集群利用率仅52%，阿里巴巴甚至低于25%）。

为提升GPU利用率，业界提出了多种GPU共享技术：
1.  **空间共享（Spatial Sharing）**：如CUDA GPU Streams（`multi-streams`）、`Multi-Instance GPU` (MIG)和LithOS。
    *   **优点：** 允许不同任务在同一GPU上并发执行，通过提高SM（`Stream Multiprocessors`）内部/外部的并行度来提升GPU利用率。
    *   **局限性：** 难以提供严格的SLO保证。由于闭源GPU（如NVIDIA）不提供细粒度任务调度和资源隔离能力，导致L2 cache、HBM带宽和PCIe带宽等硬件资源无法被用户有效控制，从而产生严重的干扰。`multi-streams`虽然允许设置优先级，但仍难以控制干扰。`MIG`通过静态分区提供隔离，但无法根据应用的行为动态重新分配资源，导致低利用率或SLO违规。LithOS提供了TPC（`Thread Processing Cluster`）级别的计算控制，但HBM带宽和L2 cache干扰仍未解决。图1显示，空间共享方案如Orion和LithOS在SLO达标率方面表现较差（分别为7.2%和31%）。
2.  **时间共享（Temporal Sharing）**：如REEF。
    *   **优点：** 通过允许任务独占GPU来提供更好的SLO达标率。
    *   **局限性：**
        *   **高抢占延迟：** NVIDIA GPU不支持主动抢占（proactive preemption），高优先级任务必须等待当前运行的低优先级`kernel`完成。`kernel`执行时间差异巨大（从几微秒到数毫秒，如`GEMM`核可达7.49毫秒），导致不可预测的抢占时间。这可能导致SLO违规。
        *   **同步与重启动开销：** 低优先级任务频繁被抢占和重新调度，导致非微不足道的同步和重启动开销。现有方案通常限制设备队列容量（如REEF为4个`kernel`），以减少`kernel`逐出和重启动，但这又引入了额外的同步开销。
        *   **低优先级任务吞吐量受限：** 现有时间共享方案无法有效利用GPU的小型空闲时间片（`bubbles`），且频繁的同步开销显著限制了低优先级任务的吞吐量。REEF的低优先级任务吞吐量远低于Orion（2.1倍）。

**Hummingbird的设计理念与核心洞察**

Hummingbird的核心设计原则是确保高优先级任务以严格的性能隔离执行，而低优先级任务则伺机利用空闲时间片。关键在于，低优先级任务必须以微秒级（μs-scale）及时释放GPU。Hummingbird基于两大洞察：
1.  **微秒级抢占的可行性：** 尽管一个`kernel`的执行时间可能长达数毫秒，但单个线程块（`thread block`）的持续时间通常在微秒级别，因为每个块只处理一小部分工作以最大化并行性。通过调整启动的线程块数量（称为`split-kernel`），可以精细控制低优先级任务占用的GPU时间，从而创建一系列微秒级的抢占点。分析显示，对于广泛的AI工作负载，99.999%的线程块执行时间都在390微秒以内，因此将`kernel`执行时间限制在400微秒内是可行的。
2.  **高效利用GPU空闲时间片（`bubbles`）：** GPU时间轴上存在大量的空闲时间片，可分为两类：
    *   **大型`bubbles`：** 持续时间从秒到分钟不等，由请求波动引起（如真实世界GPT服务追踪中占GPU时间的23.6%）。
    *   **小型`bubbles`：** 持续时间在数百微秒级别，由内存操作和同步（如设备-主机内存传输、连续批处理的元数据更新）、GPU间通信和CPU端瓶颈引起。例如，Llama-8B和DeepSeek-16B推理中超过15%的GPU时间被这些`bubbles`消耗。在分布式设置中，小`bubbles`的比例甚至可以放大1.8倍。

**Hummingbird的系统架构与关键组件**

Hummingbird包含三个核心组件：
1.  **`Kernel Splitter`（`kernel`分割器）：** 分析低优先级`kernel`的特性和底层硬件的能力，确定每个`kernel`的最佳分割大小，并生成详细的分割日志以指导运行时调度器。
    *   **最优`split-kernel`大小：** 较小的`kernel`有助于降低抢占延迟并提供更多机会填充不同大小的`bubbles`，但可能导致GPU利用率不足和`kernel`启动开销增加。最佳`split-kernel`执行时间是当`kernel`线程数与GPU计算能力（即刚好填满SMs或饱和HBM带宽）对齐时。
    *   **两步分析法：**
        1.  计算`split-kernel`应包含的最大线程块数量（$N_{block}$），考虑SMs计算能力：
            $N_{block} = \frac{N_{SM} \cdot o \cdot SM\_MAX\_THREADS}{THREADS\_PER\_BLOCK}$
            其中，$N_{SM}$是SMs数量，$o$是`kernel`占用率（`occupancy`），$SM\_MAX\_THREADS$是单个SM内的线程数，$THREADS\_PER\_BLOCK$是开发人员指定的每块线程数。
        2.  从计算出的$N_{block}$开始，逐渐减少线程块数量，同时观察`kernel`的执行时间。如果减少块数量导致执行时间缩短，则表明`kernel`是内存密集型的。此过程持续到执行时间稳定，达到最短执行时间。
    *   **PTX `kernel`转换：** 通过PTX汇编级别的代码转换实现`kernel`分割，保证了通用性。通过PTX注入技术，修改`kernel`参数列表以接受额外的偏移参数，并注入算术指令来偏移原生的`blockIdx`（在PTX中表示为`ctaid`），从而保持原始的寻址语义。这种方法适用于AOT（如CUDA）和JIT（如Triton）编译的代码。
2.  **`Runtime Scheduler`（运行时调度器）：** 根据分割日志，将`kernel`分割成更小的`split-kernel`，并动态检测空闲GPU `bubbles`，自适应地合并`split-kernel`，并采用`kernel-tick`调度策略来提高系统吞吐量。
    *   **高优先级`kernel`调度：** 当高优先级`kernel`到达时，调度器会立即停止任何新的低优先级`kernel`启动，并启动高优先级`kernel`。Hummingbird实现了平均139微秒的抢占延迟。
    *   **低优先级`kernel`调度：**
        *   **`Kernel`分割：** 当接收到低优先级任务的`kernel`启动请求时，调度器查询`kernel`分割器的日志，并根据最优分割大小将`kernel`分割成`split-kernel`。
        *   **`Bubbles`检测：** 只有当GPU空闲（高优先级`kernel`队列为空）时才调度低优先级`kernel`。
            *   **小`bubbles`：** 主要源于跨设备数据传输和同步。Hummingbird提出了基于`hint`的`bubble`检测机制，通过分析特定框架和应用的API模式（如LLM推理中的`cudaMemcpyAsync`后接`cudaStreamSynchronize`，或`NCCL`通信API）来识别这些`bubbles`。在主机侧通过`cudaEvent`插入标记事件。
            *   **大`bubbles`：** 持续时间从数十毫秒到数秒，主要由请求波动或网络延迟引起。调度器周期性扫描GPU设备队列，当在一定时间阈值内没有高优先级`kernel`出现时，则识别为大`bubble`。识别到大`bubble`后，调度器会合并`kernel`（恢复原始`grid size`和重置偏移量），以减少启动开销。
        *   **`Kernel-tick`调度策略：** 在检测到`bubble`后，调度器会同步GPU等待当前高优先级`kernel`完成，然后设置`Pflag`为False，并启动一个异步线程来应用`kernel-tick`调度策略启动低优先级`kernel`。该线程会在`Pflag`被设为True时（表示有新的高优先级`kernel`或`bubbles`结束）停止。此策略通过限制GPU设备队列中最多只有一个`kernel`来保证抢占延迟被单个`split-kernel`的执行时间所限制。为避免每次`kernel`启动后的同步开销（约5微秒），调度器利用`kernel`执行时间的可预测性，计算启动间隔，在当前`kernel`即将完成时精确地启动下一个`kernel`，形成流水线，从而减少同步频率和开销。
3.  **`Memory Management`（内存管理）：** 集成了NVLink扩展的分层内存卸载，支持`hierarchical memory offloading`。
    *   **设计原则：** 高优先级任务必须保留对全部GPU内存容量的无障碍访问，而低优先级任务则伺机利用剩余内存。
    *   **优化：**
        1.  **优先级隔离：** 利用CUDA Driver APIs中的`cuMemAdvise`来优先分配高优先级任务的内存。当GPU内存满时，只允许驱逐低优先级任务的页面。
        2.  **干扰感知：** 改进了HUVM（`hierarchical unified virtual memory`）中的页面驱逐策略。Hummingbird整合了一个全局监视器，通过`ping-like`方法测量实时带宽，并在带宽冲突时优先交换到低竞争的GPU。如果NVLink连接的内存耗尽或硬件不支持，则回退到系统DRAM。

**系统实现**

Hummingbird在NVIDIA GPU上实现，大约8000行C++/CUDA代码。它不依赖特定硬件指令，支持所有代次的GPU架构。通过拦截低级CUDA Driver API（如`cuLaunchKernel`和`cuMemAlloc`）来实现通用性和透明性。其轻量级在线`profiler`使用`cudaEvent`记录`kernel`执行时间。`kernel`转换基于NEUTRINO的`probe`引擎，在运行时进行PTX `kernel`转换。

**评估结果**

Hummingbird在多种GPU（L40s、A100、H100）上，使用两类高优先级任务和四类低优先级任务（涵盖CNN到LLM，推理到训练）进行了全面评估，包括内存密集型场景和分布式设置（多达16块GPU）。
*   **高优先级任务SLO达标率：** Hummingbird在所有场景下均达到近99%的SLO达标率。相比Orion和LithOS（空间共享），SLO达标率分别提高了9.7倍和5.6倍；相比REEF（时间共享），提高了3.0倍。高优先级任务的SLO下降不到1%（与独占执行相比）。
*   **低优先级任务吞吐量：** Hummingbird的低优先级任务吞吐量比REEF高出2.4倍（平均1.9倍）。这得益于Hummingbird有效利用小`bubbles`的能力，以及通过`kernel-tick`调度策略减轻了同步开销。
*   **内存密集型场景：** 在内存密集型场景下，Hummingbird相比REEF实现了5.6倍的SLO改进和4.2倍的吞吐量提升。这证明了Hummingbird在确保高优先级任务SLO的同时，仍能有效提高低优先级任务吞吐量的能力。
*   **抢占延迟：** Hummingbird将平均抢占延迟显著降低至121-165微秒，比REEF快4.3-6.6倍。
*   **多GPU环境：** 在多GPU设置下，Hummingbird将SLO达标率提高了9.7倍，低优先级任务吞吐量提高了3.3倍，尤其在分布式环境中，由于频繁的GPU间数据传输和同步产生大量小`bubbles`，Hummingbird的优势更为明显。
*   **通用性：** Hummingbird在不同型号GPU上均表现出良好的通用性，确保了高优先级任务的SLO，同时最大化了GPU利用率。


## Tetris
Tetris: Efficient and Predictive KV Cache Offloading for Agentic and Reasoning Workloads

https://saa2025.github.io/papers/Tetris%20-%20Efficient%20and%20Predictive%20KV%20Cache%20Offloading%20for%20Agentic%20and%20Reasoning%20Workloads.pdf SOSP25 Workshop

<img width="399" height="269" alt="image" src="https://github.com/user-attachments/assets/99812df2-24ee-4abd-a17c-13ceac22ef91" />
<img width="329" height="263" alt="image" src="https://github.com/user-attachments/assets/60958310-ab87-4f2d-850f-7e8eef89b3f6" />

1. 💥 Tetris旨在解决**Agentic和推理LLM工作负载中**，因KV cache使用**量大和缺乏预测性而导致的级联抢占**问题。
2. 💡 该系统通过**轻量级逐token序列长度预测、动态结合重计算与卸载机制**，以及**分层 异步KV cache传输与预测调度**来缓解级联抢占。
3. 🚀 基于vLLM 0.8.3实现，Tetris显著**降低了抢占频率，并在内存受限环境下提升了P99 TPOT性能**。
Tetris是一种针对Agentic和推理LLM工作负载的高效、预测性KV Cache卸载系统，旨在解决推理时扩展和工具调用导致KV Cache使用量急剧增加的问题，尤其是在长中间推理步骤和API调用历史记录场景下。该研究指出，虽然长输入场景已得到广泛研究，但长输出场景（对于代理和推理工作负载，输出长度可达数十万token）仍未被充分探索，这导致了服务质量的下降，特别是“级联抢占（cascading preemption）”现象：由于受害者序列选择不当，内存不足时会连续发生多次抢占。
Tetris通过以下三个核心创新来缓解级联抢占并优化KV Cache管理：

轻量级逐token序列长度预测（Lightweight Per-token Sequence Length Prediction）
为了解决传统提前（AOT）的、基于代理LLM的预测方法（如[3, 6, 11]）计算量大、适应性差的问题，Tetris利用LLM隐藏状态编码未来输出长度结构信息的洞察（[1]）。它采用一个参数量仅为302K的MLP模型进行在线序列长度预测，这仅是LLM-based预测方法的1%。这种轻量级设计使得系统能够进行逐token的长度更新，并能通过在线更新MLP模型来适应新的工作负载。这种预测能力对于实现最优的受害者序列选择至关重要，因为它可以预判未来的KV Cache需求。

**重计算与卸载的权衡分析**
当系统内存不足需要进行抢占时，通常有两种策略：重计算（recomputation）和卸载（offloading）。
<img width="1044" height="437" alt="image" src="https://github.com/user-attachments/assets/0ff038c4-0337-46c3-8f52-61a90c104bf6" />

**分层异步KV Cache传输与预测调度**:
现有的系统如vLLM [4]和SGLang [10]通常采用**同步KV Cache卸载**，这会阻塞GPU的推理进程。Tetris通过识别并利用两个优化机会来缓解这种阻塞：
分层传输（Layerwise Transfer）：不再等待整个前向传播结束后再复制KV Cache，Tetris在当前层的KV Cache计算完成后立即开始复制。这使得设备到主机的KV Cache数据传输可以与后续的前馈网络（feedforward computation）计算重叠进行，从而提高GPU利用率。
预测调度/提前传输（Predictive Scheduling/Ahead-of-time Transfer）：借助其序列长度预测能力，Tetris能够预判未来的KV Cache需求并与可用内存进行比较，从而在实际抢占发生之前，异步地启动KV Cache的卸载操作。这种“in-flight offloading”机制可以在不中断序列生成的情况下驱逐受害者序列，从而最大程度地减少实际抢占时需要复制的KV Cache量。同样，在从主内存重新加载KV Cache到GPU时，Tetris也可以提前开始加载过程，以最小化GPU的停机时间。
为了解决分层传输中涉及大量小张量传输的效率问题，Tetris实现了一种打包机制，将批处理维度上的小张量打包在一起，以分摊卸载成本。

**评测**：
Tetris在vLLM v0.8.3上实现。其系统架构包括一个调度器（Scheduler），负责查询KV Cache可用性和预测的序列长度以选择受害者并构建输入批次。实际的KV Cache传输在一个专用的CUDA Stream和pinned memory上进行，以最大程度减少对主要解码操作的干扰。每个Worker在另一个CUDA Stream上执行解码步骤。解码完成后，系统会利用截获的隐藏状态更新序列长度预测。
Tetris的评估结果表明，在内存受限的环境下，它显著减少了抢占频率，并改善了P99 TPOT（Time-per-output-token）性能，同时保持了吞吐量。

## LLM-42
LLM-42: Enabling Determinism in LLM Inference with Verified Speculation

https://arxiv.org/pdf/2601.17768 2026.1.25 微软

https://github.com/microsoft/llm-42

1. 💡 LLM-42，一个受 speculative decoding 启发的方法，旨在解决 LLM 推理中因浮点非结合性、动态批处理和 GPU kernel 还原顺序变化导致的非确定性问题。
2. ⚙️ LLM-42 采用**decode–verify–rollback 协议**，使用**非确定性的 fast path 生成 token**，并通过一个**轻量级 verifier 在固定形状还原调度下**重放和验证 token，**仅在检测到不一致时进行回滚和 KV cache 修复**。
3. 🚀  Llama-3.1-8B-Instruct，LLM-42 在保持与**非确定性模式接近的性能的同时，显著优于批处理不变（batch-invariant）的确定性方法**，其开销与实际需要确定性的流量比例成正比。

<img width="407" height="333" alt="image" src="https://github.com/user-attachments/assets/70a1c11b-8993-401c-87c8-1a59723f2506" />
<img width="940" height="300" alt="image" src="https://github.com/user-attachments/assets/2f27658c-d217-4383-a59c-89700c01efcc" />
<img width="937" height="319" alt="image" src="https://github.com/user-attachments/assets/8913030a-69b0-4029-973e-5813fc701124" />

LLM-42 是一种新颖的调度方法，旨在解决大型语言模型 (LLM) 推理中的非确定性问题。LLM 推理的非确定性源于浮点运算的非结合性（non-associativity），结合动态批处理（dynamic batching）以及 GPU 核函数（kernel）随批次大小变化而改变的归约顺序（reduction order）。

传统的解决方案存在缺陷：禁用动态批处理会导致吞吐量严重下降；使核函数批次不变（batch-invariant）则要求重新实现核函数，这既增加了工程负担，又强制所有请求承担固定的运行时开销，无论它们是否需要确定性。例如，SGLang 和 VLLM 采用的批次不变计算方法，强制所有核函数使用统一的归约策略，牺牲了性能优化（如 GEMM 中的 split-K 策略）。实验表明，与非批次不变的 cuBLAS 相比，批次不变的 Triton GEMM **核函数性能下降 63%**；RMSNorm 核函数性能下降高达 50-700%。在端到端吞吐量测试中，当单个请求需要确定性时，强制整个批次使用批次不变核函数会导致**吞吐量骤降 56%**。

LLM-42 受到推测解码（speculative decoding）的启发，其核心观察基于以下四点（Observations）：
1.  **O1 (Token-level Inconsistencies are Rare):** 如果一个序列处于一致状态，下一个生成的 Token 通常也是一致的，即使在动态批处理下。序列级别的分歧主要发生在一个 Token 出现不一致后，自回归解码（autoregressive decoding）会逐渐放大这种差异。实证研究显示，许多初始 Token 能够匹配，但一旦出现差异，后续序列会迅速发散。
2.  **O2 (Most GPU Kernels use Shape-Consistent Reductions):** 大多数 GPU 核函数采用形状一致的归约策略。这意味着它们**对给定形状的所有输入都采用相同的归约策略**，只有当输入形状改变时，策略才会改变。这些核函数通常是位置不变的（position-invariant），即给定总批次大小，输入元素的输出与其在批次中的位置无关。
3.  **O3 (Determinism Requires Only Position-Consistent Reductions):** 对于确定性推理，只需确保给定 Token 位置在所有运行中采用相同的归约策略即可，不同 Token 位置之间或不同序列之间的归约策略可以不同。
4.  **O4 (Selective Determinism):** 实际 LLM 系统**并非所有任务都要求确定性**，例如创意性工作负载可能受益于随机性，而评估、审计或回归测试则需要确定性。因此，**全局强制确定性是一种资源浪费**。

基于这些观察，LLM-42 提出了一种解码-验证-回滚（Decode-Verify-Rollback, DVR）协议。其核心方法是：
1.  **快速路径（Fast Path）**：LLM-42 首先使用高吞吐量的非确定性执行（即，标准优化的 GPU 核函数和动态批处理）来乐观地生成候选 Token。
2.  **验证器（Verifier）**：**验证器周期性地对最近生成的一固定大小 Token 窗口进行重放和验证**。为了确保验证器的输出本身是确定性的，验证器总是处理固定数量的 Token（例如 $T=32$ 或 $64$），对于短序列会进行填充以保持形状一致性（利用 O2）。验证器使用特定的、确定性的归约策略，例如，对于 FlashAttention-3 核函数，将 `num_splits` 设置为 1；对于通信集体操作，优先使用 multimem-based AllReduce 或固定配置的 tree-based AllReduce，避免使用 ring-based AllReduce。
3.  **KV 缓存一致性（KV Cache Consistency）**：验证器生成的 KV 缓存条目会覆盖快速路径生成的对应条目，确保后续解码迭代的状态一致性。
4.  **回滚机制（Rollback）**：如果验证器发现快速路径生成的 Token 与其确定性重放的结果不一致，LLM-42 **会将序列回滚到最后一个匹配的 Token 位置，**并从该已知的一致状态继续解码。每个验证步骤至少会产生一个新的确定性 Token，从而保证了前向进展（forward progress）。
5.  **选择性确定性（Selective Determinism）**：LLM-42 通过引入一个 `is_deterministic=True|False` 的 API 标志，**允许用户按需启用确定性**。只有需要确定性输出的请求才会触发验证过程，其他请求则直接走非确定性快速路径，避免了不必要的开销（利用 O4）。

为了平衡验证成本和重计算成本，LLM-42 引入了**分组验证（Grouped Verification）**。验证操作在小窗口（例如，每个请求 $32$ 个 Token）下内存受限（memory-bound），单 Token 验证延迟较高（$0.75 \text{ ms}$）；在大窗口（例如 $512$ 个 Token）下计算受限（compute-bound），单 Token 验证延迟较低（$0.05 \text{ ms}$），但回滚时需要重计算的 Token 数量会大幅增加（例如，窗口大小为 $256$ 时，重计算开销可达 $46.41\%$）。分组验证通过同时验证多个请求的较小固定窗口（例如，8 个请求，每个 $32$ 个 Token），实现了高 GPU 利用率（如同大批次）的同时，保留了小窗口的回滚特性（限制了重计算成本）。

在评估方面，LLM-42 在 Llama-3.1-8B-Instruct 模型上进行了测试，并与 SGLang 的确定性模式和非确定性模式进行比较。
-   **离线推理（Offline Inference）**：SGLang 确定性模式的吞吐量比非确定性模式低 $24\% \sim 36\%$。而 LLM-42 在 $100\%$ 确定性流量下，吞吐量依然显著高于 SGLang 确定性模式，在多数场景下仅比 SGLang 非确定性模式慢很少（例如，在 ArXiv 数据集上，当 $5\%$ 确定性流量时，LLM-42 吞吐量在最佳情况的 $92\%$ 以内；在 $10\%$ 确定性流量时，吞吐量可达最佳情况的 $98\%$）。LLM-42 的回滚频率和重计算开销适中，最坏情况下（ArXiv 数据集，$100\%$ 确定性流量）平均每个请求回滚少于一次，总重计算 Token 占比最高为 $10.97\%$。
-   **在线推理（Online Inference）**：LLM-42 的端到端延迟分布曲线（CDF）紧密跟随 SGLang 非确定性基线，即使确定性流量比例增加，延迟也只适度增加。与 SGLang 确定性模式相比，LLM-42 在高负载下仍能保持显著更低的尾延迟（P99）。例如，在 $18 \text{ QPS}$ 下，LLM-42 在 $100\%$ 确定性流量下的 P90 TTFT 仅为 $101.2 \text{ ms}$，远低于 SGLang 确定性模式的 $171.6 \text{ ms}$。
-   **消融研究（Ablation Study）**：分组验证显著降低了端到端延迟，例如，在 $12 \text{ QPS}$ 下，未分组验证（批次大小 $1$）的最佳 P99 延迟为 $56.18 \text{ s}$（窗口大小 $128$），而分组验证（例如，批次大小 $8$，窗口大小 $32$）可将 P99 延迟降低到 $34 \sim 35 \text{ s}$。


## SOAR
Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability

https://arxiv.org/pdf/2601.18778 2026.1.27 MIT，Meta等

1. 大型语言模型在低初始成功率的推理任务中面临学习停滞，本研究为此引入了SOAR，一个**基于元RL的自改进框架**，旨在通过**模型自生成课程**来解决这一难题。
2. 🤖 SOAR的核心在于**教师模型通过学生在少量难题上的可衡量进步**来获得奖励，这种基于实际进展的奖励机制优于传统的内在奖励，有效避免了多样性崩溃和不稳定性。
3. ✨ llama3.2-3b实验 方法显著提升了**模型在极难数学基准上的性能**，并揭示出问题结构质量和良好定性对学习进展更为关键，即使答案不完全正确也能提供有效信号，为模型突破学习瓶颈提供了新途径。
   
<img width="723" height="305" alt="image" src="https://github.com/user-attachments/assets/7a050a3f-1550-4348-9705-4bf4461c18eb" />
<img width="716" height="347" alt="image" src="https://github.com/user-attachments/assets/4250c52a-4839-4ca5-b87a-ab64e691b545" />
<img width="525" height="400" alt="image" src="https://github.com/user-attachments/assets/a6a15e5b-8ea4-4ba1-b45f-f8c118c9af4c" />

大型语言模型（LLM）在解决高难度推理任务时，尤其是在数学和编程领域，常因奖励信号稀疏而陷入学习停滞。当初始成功率极低时，传统的基于可验证奖励的强化学习（RLVR）方法难以提供足够的训练信号，导致模型无法从其尚未解决的问题中学习。

本研究旨在探索一个根本性问题：预训练LLM能否利用其内在知识自主生成“垫脚石”课程，以克服其在难解问题上的学习瓶颈？为解答此问题，论文设计并提出了一种名为 SOAR (Self-Optimization via Asymmetric RL) 的自改进框架。SOAR是一个非对称的教师-学生 meta-RL (meta-Reinforcement Learning) 框架，旨在挖掘LLM中潜在的教学信号。

**核心方法论：SOAR框架**
<img width="861" height="308" alt="image" src="https://github.com/user-attachments/assets/33007de9-920d-4f15-b984-7d5ccd9996f1" />
<img width="946" height="471" alt="image" src="https://github.com/user-attachments/assets/ecb4e3cb-6388-4e90-a228-ce0bb8415a7b" />


**2. 内层循环：学生训练 (Inner Loop: Student Training)**
学生模型 $\pi^S_\theta$ 在教师生成的合成数据集 $X_k$ 上进行训练，同样使用 RLOO/RLVR。学生模型训练的步数较少（例如10步），以保持计算效率。学生训练的奖励信号由 Math-Verify 工具包提供，用于验证学生生成的答案的正确性。奖励函数定义为：
$$ R(y, a) = \begin{cases} 120.0 & \text{if has\_boxed}(y) \land \text{verify}(y, a) \\ 20.0 & \text{if has\_boxed}(y) \land \neg\text{verify}(...) \land a \in y_{ans} \\ 10.0 & \text{if has\_boxed}(y) \land \neg\text{verify}(...) \land a \notin y_{ans} \\ 0.0 & \text{otherwise} \end{cases} $$
其中 $y$ 是学生模型生成的解决方案，$a$ 是教师提供的正确答案。`has_boxed(y)` 检查答案格式，`verify(y, a)` 检查数学正确性，`a \in y_{ans}` 检查学生答案是否包含正确值。
**晋升机制 (Promotion Mechanism)**：为了让教师适应不断进步的学生，SOAR引入了一个晋升机制。当教师奖励的移动平均值 $\bar{R}_t$ 超过一个固定阈值 $\tau$（例如 $0.01$）时，表现最好的学生模型 $\pi^S_{\theta'_{k^*, j^*}}$ 会被“晋升”为新的学生基线 $\pi^S_\theta$。这意味着后续的奖励计算将以这个新的、更强的学生模型为参照，促使教师生成更高难度的“垫脚石”问题，从而持续推动学习进程。被晋升的学生模型所训练的数据集（Dbest）会被累积为 Promotion Questions (PQ)。
**拒绝采样 (Rejection Sampling)**：教师生成的 (q,a) 对需要遵循特定格式。对于不符合格式的生成结果，SOAR 采用拒绝采样（resample）策略。论文中证明了在 RLOO 的上下文下，这种拒绝采样不会影响梯度更新，因为 RLOO 的优势函数 (advantage function) 求和为零。

**实验设置与结果**
实验基于 Llama-3.2-3B-Instruct 模型，在数学推理基准 MATH, HARP 和 OlympiadBench 上进行。为模拟真实难题情境，研究专门筛选出在128次尝试中成功率为0的问题子集（fail@128数据集）。
**主要发现：**
1.  **Meta-RL 开启学习 (Meta-RL Discovers Effective Questions)**：SOAR 生成的合成问题（PQ）和晋升后的学生模型（PS）显著优于直接在难解问题上训练（Hard-Only）以及使用内在奖励（Intrinsic-T）的基线模型。例如，在 MATH 数据集上，SOAR 实现了 4倍的 pass@1 提升和 2倍的 pass@32 提升。这些合成问题即使在 OOD (Out-of-Distribution) 的 OlympiadBench 数据集上也能实现泛化，表明其捕获了可泛化的推理路径。此外，合成问题仅需少量（128-256个）即可恢复使用大量人工策展数据（如完整的 MATH 训练集）约75%的性能提升。这证明了模型的教学能力可以与解决问题的能力解耦，并且 meta-RL 能够从基础模型中提取和强化这种潜在能力。
2.  **接地奖励优于内在奖励 (Grounded Rewards over Intrinsic Rewards)**：将教师奖励与学生在真实问题上的进展挂钩，相比自玩（self-play）中常见的内在奖励，能显著提升性能。接地奖励训练的教师策略（Grounded-T）更为稳定且能保持生成问题的多样性（通过 Vendi Score 衡量，Grounded-T 的 Vendi Score 高于 Intrinsic-T，接近 Base-T），有效避免了内在奖励方法常出现的退化或多样性崩溃问题。
3.  **问题结构重于答案正确性 (Question Structure over Solution Correctness)**：研究发现，教师生成的有效“垫脚石”问题并非都拥有完全正确的答案。对 PQ 问题的分析显示，只有约32.8%的问题答案完全正确，但高达63%的问题在数学上是“well-posed”（结构良好、可解）。这表明对于处于学习停滞期的模型而言，问题的结构质量和概念内容比答案的精确正确性更重要，它们能够提供有用的学习信号。Meta-RL 还能减少问题中的歧义错误，进一步验证了问题连贯性的重要性。




## PrefixRL
Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes

https://arxiv.org/pdf/2601.18795 2026.1.27 Meta, CMU
1. 💡 PrefixRL是一种新的强化学习方法，它通过将历史off-policy**成功轨迹的前缀作为大型语言模型（LLM）推理任务的训练条件**，旨在更高效地利用FLOPs并解决“硬问题”。
2. 🔬 该方法通过将RL策略置于**更易获得奖励的状态来增强学习信号**，并被证明与**标准RL目标一致且更具样本效率**，同时规避了直接监督off-policy数据所导致的不稳定性。
3. 🏆 Qwem3-4b/Llam3-8b实验结果显示，PrefixRL在相同训练奖励下比**最强基线快2倍****，最终奖励提高3倍**，并且其收益可泛化到未见过的数据集，即使off-policy轨迹来源于不同模型家族也依然有效。
<img width="758" height="268" alt="image" src="https://github.com/user-attachments/assets/651fb689-f9cf-4dc7-ab3d-d95d8db04a5d" />
<img width="759" height="304" alt="image" src="https://github.com/user-attachments/assets/dbcdc1f1-895e-44b6-843e-ad25292389f6" />
<img width="1017" height="353" alt="image" src="https://github.com/user-attachments/assets/53a29590-78cc-4d54-84ad-b0287513fd04" />
<img width="1018" height="249" alt="image" src="https://github.com/user-attachments/assets/c9616902-a460-4402-8e16-5e5fdba35956" />
<img width="1017" height="380" alt="image" src="https://github.com/user-attachments/assets/5e45972a-a5e0-49dd-ba11-7e6f9d9182c7" />
<img width="1027" height="293" alt="image" src="https://github.com/user-attachments/assets/4e6678de-7f11-4573-89de-daa8df6102b1" />
<img width="1018" height="261" alt="image" src="https://github.com/user-attachments/assets/4ae5ab83-977b-4bf2-a2a1-52963a1e9be3" />
<img width="1010" height="392" alt="image" src="https://github.com/user-attachments/assets/2037a0c7-c87e-4947-984f-11afc975365f" />


提出了一种名为 PrefixRL 的强化学习（RL）方法，旨在解决大型语言模型（LLMs）在处理困难推理问题时计算资源浪费、策略梯度消失和学习停滞的问题。

**1. 问题背景与现有方法的局限性**
LLM 在数学和编码等推理任务中，通常采用 RL 进行训练。多数成功的 RL 方法是 `on-policy` 的，即从当前模型采样推理轨迹（`rollouts`），并根据正确（和不正确）的轨迹进行更新。然而，当问题难度很高，导致 `pass@k` （例如 `pass@2k`）接近 0 时，模型很难采样到正确轨迹，`on-policy` RL 学习信号稀疏，导致学习停滞，浪费大量 `FLOPs`。
为了利用之前计算（例如推理或旧的 RL 训练）产生的 `off-policy traces`，常见的尝试方法有：
*   **Supervised Fine-Tuning (SFT)**：将 `off-policy traces` 视为监督数据进行 `fine-tuning`，然后进行标准 `on-policy RL`。但 SFT 在小规模正确轨迹上可能导致模型记忆化（`memorization`）和 `entropy collapse`，损害后续 RL 的探索能力。
*   **Importance Weighting Off-policy RL**：直接在 RL 中使用 `importance weighting` 来处理 `off-policy` 数据。然而，由于 `off-policy traces` 在 RL 策略下概率极低，这通常会导致不稳定性，例如梯度估计方差高，甚至引发训练崩溃。

**2. PrefixRL 核心思想**
PrefixRL 提出了一种新的范式：不直接在 `off-policy traces` 上进行监督或更新，而是通过条件化（`conditioning`）于成功的 `off-policy traces` 的前缀（`prefixes`），然后运行 `on-policy RL` 来完成这些轨迹。
具体步骤如下：
1.  **提取并固定前缀**：从成功的 `off-policy traces` 中提取并固定其前缀 `yx1:h`。
2.  **创建带前缀的问题**：将这些前缀添加到原始问题 `x` 中，形成“带前缀的问题”（`prefixed problems`）$\text{concat}(x, y_{x}^{1:h})$。论文通常选择 $h$ 使得在给定前缀的情况下，`base LLM` 有合理的准确性。
3.  **运行 `on-policy RL`**：**在这些“带前缀的问题”和原始的“无前缀问题”（`no-prefix problems`）上同时运行 `on-policy RL`**。
关键在于，在处理“带前缀的问题”时，梯度计算在 `off-policy prefix` 上被遮蔽（`masked`）。这意味着模型不会修改生成前缀的参数，而**只学习如何从前缀之后的更高奖励状态进行有效探索和完成**。
PrefixRL 的训练目标函数定义为：
<img width="1011" height="462" alt="image" src="https://github.com/user-attachments/assets/d85404b9-6c3b-4c63-b484-6bf8007c1a9e" />

    *   **命题 3.4（`Worst-case separation with standard RL`）**：存在一个奖励函数和 `base LLM`，使得 `PrefixRL-NPG` 的性能显著优于标准 RL，尤其是在上下文长度（`context length`）$H$ 较长时。标准 RL 的性能以指数 $2^{-H}$ 衰减，而 PrefixRL 不会。

**4. `Back-Generalization` 现象**
论文发现一个关键的经验现象，称之为 `back-generalization`：仅在带前缀问题上训练 `on-policy RL`，可以显著提升模型在从未训练过的原始无前缀问题上的测试性能。
*   **超越 `stitching`**：`back-generalization` 不仅仅是模型学习如何更好地从 `off-policy` 中间状态继续，它还能影响未训练状态（即无前缀问题）的 `next-token distributions`。这表明 LLMs 中的有利函数逼近（`favorable function approximation`）起到了关键作用。
*   **在仅训练带前缀问题时提升无前缀性能**：实验表明，即使只在带前缀问题上训练，模型的无前缀性能也能提高。训练较长前缀可提高较短前缀的性能，并最终提高无前缀准确率（图 5）。
*   **发现新策略**：PrefixRL 可以发现并学习超出前缀中提供的策略。模型不只是简单地模仿 `off-policy prefix`。通过控制实验（图 6），PrefixRL 比标准 RL 更有效地放大成功策略和拒绝次优策略。模型甚至可以“遗忘”前缀中暗示的次优策略，转而发现更优的策略（例如，从“Erdős–Gallai theorem”转向“Dirichlet Theorem”）。这支持了 PrefixRL 目标的一致性，因为它能找到前缀中没有的更优解。
*   **`In-Context Learning` 下的 `Back-Generalization`**：PrefixRL 在 `in-context learning` 设置下进行分析（图 7），即在上下文中给定另一个问题及其解决方案轨迹的情况下运行 RL。
    *   **相关性影响**：当 `prefix` 和 `suffix` 在结构上相关时，`back-generalization` 效果最强。例如，训练解决 P2 | P3 (P2 以 P3 为上下文) 可以显著提高 P2 和 P3 的性能，而无关问题 (P1 | P3) 则没有这种效果。
    *   **非模仿性**：模型并不通过简单地记忆 `in-context trace` 来提高性能（图 8a）。`in-context` 解决方案的 `Negative Log-Likelihood (NLL)` 几乎没有下降，最终策略倾向于生成不同的 `token sequence` 也能获得正确答案。这表明模型学习的是深层表示和问题解决策略，而不是表面模仿。
*   **跨模型家族的 `Prefix` 来源**：即使 `off-policy prefix` 来自不同模型家族（例如，用 `Qwen3-4B-instruct` 的前缀训练 `Llama3.1-8B-instruct`），`back-generalization` 仍然有效。如果前缀长度分布足够广，可以为“无前缀问题”搭建“桥梁”，则 `off-policy` 数据的来源模型家族影响较小（图 8b, 8c）。

**5. 实验结果**
*   **计算效率和训练准确率提升**：PrefixRL 在计算匹配的评估中，即使计入初始 `rejection sampling` 成本，也比最强的 `baseline`（SFT+RL）提高 `compute-efficiency` 约 2 倍，并在无前缀训练问题上将最终训练准确率提高超过 45%（相对提升 3 倍）（图 2，图 9）。
*   **`Held-out` 基准测试性能提升**：PrefixRL 的优势可以迁移到 `held-out` 基准测试，例如 AIME '25、HMMT '25 和 IMO-AnswerBench（图 10）。在这些基准上，`pass@k` 提升了超过 10% 的绝对值，且 `pass@1` 提升显著，表明模型在轨迹早期更可能实例化正确的 `high-level plan`。
*   **扩展可解问题集**：PrefixRL 不仅提高了 `pass@1`，还稳定提高了 `pass@32`，表明它扩展了具有非零成功概率的问题集，而不是仅仅将固定 `pass@k` 转换为更高的 `pass@1`（图 11b）。它实现了训练问题 `pass@1` 的更均匀提升，避免了 RL 常见的“`ray interference`”问题（图 11a）。
*   **不同模型家族的有效性**：PrefixRL 在 `off-policy prefixes` 来自不同模型家族时仍然有效，显示了其在实际应用中的灵活性（图 12）。

**6. 训练动态分析**
*   **保留熵**：与 SFT 导致 `token-level entropy` 大幅下降不同，PrefixRL 在利用 `off-policy` 数据的同时，保留了大部分 `token-level entropy`，有利于 RL 探索（图 13 左）。
*   **减少 `all-negative batches`**：PrefixRL 显著减少了训练过程中 `all-negative problems` 的比例，这意味着它能更频繁地将策略置于可以获得非零奖励的状态，从而打破了 RL 的停滞状态（图 13 中）。
*   **迭代效率**：PrefixRL 能够以更少的采样 `token` 达到更高的准确率，因为正确轨迹通常更短，并且模型在内部化策略后能更快做出决定（图 13 右）。
*   **更高的信噪比（`Signal-to-Noise Ratio`）**：PrefixRL 同时拥有更高的梯度范数（`gradient norm`）和更低的梯度标准差（`gradient standard deviation`）（图 14），这意味着它在训练中具有更高的信噪比，优化更稳定。相比之下，`importance-weighted off-policy RL` 存在严重的梯度噪声和不稳定性。


## 阶跃Step3-VL-10b

https://www.paperscope.ai/hf/2601.09668 tech report
中文快速解读：https://mp.weixin.qq.com/s/DEeYuMgnw0CK1QhnH_NdZw


## Speculative Decoding
Speculative Decoding: Performance or Illusion?

https://arxiv.org/pdf/2601.11580 伯克利 2025.12.31

https://github.com/SpecDecode-Bench/simulator

1. 🔬 该研究首次在生产级vLLM推理引擎上系统评估了多种推测解码（SD）变体，发现SD在实际部署中能提升吞吐量，但其加速效果随批处理大小增加而减弱。
2. 📊 论文深入分析了SD的性能瓶颈，揭示目标模型验证是主要开销，且令牌接受率在不同输出位置、请求和数据集间存在显著波动，其中n-gram在代码编辑等重复性任务中表现突出。
3. 🚀 研究量化了SD的理论加速上限，并指出当前SD方法与理想性能存在较大差距，展望了通过优化验证过程和自适应结合不同SD策略（例如实现高达4.9倍的总加速）的未来优化方向。

<img width="1285" height="597" alt="image" src="https://github.com/user-attachments/assets/54889fa9-e135-43ae-956c-0b89602d1e6d" />
<img width="975" height="358" alt="image" src="https://github.com/user-attachments/assets/8dc23734-7294-44fc-923d-c553b1eb3e7f" />

本文对LLM (Large Language Model) 推理加速技术Speculative Decoding (SD) 进行了首次系统性研究，其评估基于生产级推理引擎vLLM，而非以往研究中常见的原型系统和不切实际的单Batch Size配置。

**1. 引言与背景**

SD 通过利用LLM评估候选Token序列的并行能力来加速推理。其核心思想是使用一个更小、更快的“草稿模型”(draft model) 或其他机制预测未来Token，然后目标大型模型并行验证这些预测。如果预测正确，则一次性接受多个Token，从而减少大型模型的自回归推理步数。

以往对SD的研究存在以下局限性：
*   多使用原型实现，缺乏生产级系统（如vLLM）的关键优化（例如：CUDA graphs）。
*   评估多在Batch Size为1的非现实配置下进行，无法反映真实部署场景。
*   缺乏对不同SD变体（如基于草稿模型、n-gram、树形结构等）的系统性比较和其适用场景的清晰指导。

本文旨在弥补这些不足，通过在vLLM上对多SD变体进行基准测试，并分析影响SD性能的关键因素。

**2. 核心SD变体**

文章研究了以下SD变体：
*   **Draft-model-based (草稿模型)：** 使用一个独立的、较小的LLM作为草稿模型。它需要与目标模型拥有相同的词汇表，并需进行量化(quantization) 或蒸馏(distillation) 才能有效工作。
*   **EAGLE (Li et al., 2024b) / EAGLE-3 (Li et al., 2025)：** 无草稿模型的方法。通过在Transformer block顶部添加辅助预测层 (auxiliary prediction heads) 直接提议Token。这些head需要进行额外的微调(fine-tuning)。
*   **Multi-Token Prediction (MTP) (Liu et al., 2025)：** 同样是无草稿模型的方法，其辅助预测head与主模型一同进行联合训练(co-trained)，因此可以“开箱即用”(out of the box)，并能实现较高的Token接受率(token-acceptance rates)。
*   **n-gram (Saxena, 2023)：** 训练无关(training-free) 的方法，通过从已生成文本中查找和复用重复的n-gram片段作为Token提议。特别适用于代码编辑等具有高局部重复性的场景。

**3. 端到端性能评估 (End-to-End Performance Evaluation)**

*   **实验设置：**
    *   **推理引擎：** vLLM v0.10.1.1，开启所有默认优化，包括KV cache管理、Continuous Batching、Chunked Prefill和CUDA Graphs。
    *   **硬件：** NVIDIA H100 (80 GB)。8B模型使用单GPU，70B/106B模型使用四块GPU并配置Tensor Parallelism。
    *   **模型：** Llama3.1-8B-Instruct, Llama3-70B-Instruct, Qwen3-8B, GLM-4.5-Air-106B。
    *   **工作负载：** CNN/DailyMail (摘要)、ShareGPT (多轮对话)、InstructCoder (代码编辑)、GSM8K (数学推理)，以及两个复杂推理数据集AIME22-24和GPQA-Main。
    *   **指标：** 由于LLM推理存在非确定性(nondeterminism) 导致生成长度波动，本文采用Token吞吐量(token throughput, 即每秒生成的Token数量) 作为核心性能指标。加速比(Speedup) 定义为SD吞吐量与无SD基线吞吐量之比。
*   **主要发现：**
    *   **Batch Size效应：** 尽管增加Batch Size能提高绝对吞吐量，但SD的相对加速比会降低。模型越大，这种效应越明显，因为大模型即使在小Batch Size下也可能已是计算密集型(compute-bound)，SD带来的额外计算开销（用于提议和验证最终被拒绝的Token）变得更加显著。
    *   **SD变体与数据集：**
        *   n-gram通常效果不如其他SD方法，但InstructCoder（代码编辑）除外。在代码编辑任务中，n-gram因其能有效利用Token复用而表现出色。
        *   基于草稿模型的方法在70B目标模型上表现最佳，但在8B目标模型上效果下降。这归因于草稿模型在小模型上的提议开销(proposing overhead) 相对于验证开销(verification cost) 变得过高。
    *   **推理工作负载：** 针对需要长文本生成的推理任务，能持续高接受率的SD方法（如EAGLE-3和n-gram）受益最大。MTP的性能受限于其单一MTP head在连续Token预测上的精度下降。

**4. 性能剖析：执行时间与内存分解 (Understanding Performance: Execution Time & Memory Breakdown)**

*   **执行时间：**
    *   文章将LLM推理的执行时间分解为：提议(Drafting)、验证(Verification)、拒绝采样(Rejection Sampling) 和其他开销(Other Overheads)。
    *   **发现：** 验证阶段占据了SD执行时间的最大部分（42%至95%），这表明大型目标模型执行仍然是主要计算瓶颈。提议阶段的开销因方法而异：n-gram极低（<2%），EAGLE/EAGLE-3在低Batch Size下为12-20%，而基于草稿模型的方法则较高（8B模型在Batch Size为1时高达47%）。拒绝采样时间微乎其微。
    *   **启示：** 验证成本是端到端SD执行的主导因素。对最终被拒绝的Token进行验证会造成大量计算浪费，这促使研究如何最小化验证成本。
*   **内存：**
    *   SD引入的内存开销通常很小，包括静态参数内存和每Token KV cache内存。
    *   **发现：** n-gram不产生GPU内存开销。EAGLE/EAGLE-3引入少量静态内存开销（因其额外的Transformer层）和适度的每Token KV cache开销。基于草稿模型的方法的开销取决于所选草稿模型的大小，其每Token KV cache开销显著高于EAGLE-based方法，因为它需要为额外的多层草稿模型维护KV cache。
    *   **计算方式：**
        *   静态内存： $M_{static,GiB} = (P_{target} + P_{draft}) \cdot \frac{2}{2^{30}}$ (对于FP16权重)。$P_{target}$ 和 $P_{draft}$ 分别为目标模型和草稿模型的参数量 (以十亿为单位)。
        *   每Token KV cache： $M_{KV/token,KiB} = L_h \cdot 2 \cdot n_{kv} \cdot d_{head} \cdot \frac{2}{2^{10}}$。其中 $L_h$ 为隐藏层数，$n_{kv}$ 为Key/Value head数，$d_{head}$ 为head维度。

**5. 接受行为分析**

文章详细分析了SD方法的Token接受行为，发现在一个请求内部、跨请求之间以及跨数据集之间，生成的Token长度（即接受的Token数量）存在显著差异。

*   **请求内部差异：** 针对长生成的推理工作负载，n-gram方法受益于重复模式，其接受Token数随生成进度显著增加，但接近结尾时（模型转向总结）接受率会下降。EAGLE-3则更平稳地增长。
*   **跨请求和数据集差异：**
    *   **基于草稿模型：** 通常实现最长的中位数生成长度，但方差较大，表明其性能高度依赖于请求，能否与目标模型紧密对齐。
    *   **EAGLE/EAGLE-3：** 展现出更稳定、方差较小的接受分布（通常为2-4个Token），这得益于其学习到的上下文表示。
    *   **n-gram：** 具有高方差和重尾分布(heavy-tailed distribution)。多数情况下接受长度较短，但在少数情况下（尤其在代码编辑中）能产生异常长的爆发性接受（超过15个Token），这表明它依赖于离散的模式匹配。

**6. n-gram在InstructCoder上的案例研究**

*   **假设：** n-gram在代码编辑任务上表现出色，是因为代码中固有的局部重复性。
*   **方法：** 使用BLEU-n (特别是BLEU-4) 量化Prompt和输出之间的重叠度。将请求按BLEU分数分组，比较n-gram和EAGLE/EAGLE-3的加速比。
*   **发现：** 更高的BLEU-n分数与n-gram更大的加速比呈强相关。当BLEU-4分数超过0.6时，n-gram在所有Batch Size下都持续优于EAGLE/EAGLE-3，其加速比可高出53%（提议长度3）甚至100%（提议长度5）。这证实了n-gram在代码编辑工作负载中直接受益于Prompt级别的重复性。

**7. SD理论加速上限**

*   **最小化验证成本：** 引入“Oracle”设置，假设系统能预知每个生成步骤中实际将被接受的Token数量，并精确提议该数量的Token，从而消除验证浪费。
    *   **发现：** Oracle加速比与固定提议长度的SD之间存在显著差距，且随着Batch Size的增加差距扩大。这表明当前SD方法在处理被拒绝Token的验证成本上仍有巨大优化空间。
*   **自适应组合方法实现最优加速：** 观察到EAGLE和n-gram在不同Token位置上具有互补优势（即在某些位置EAGLE接受更多，在另一些位置n-gram接受更多）。
    *   **方法：** 设想一个“Oracle Combine”方案，即一个完美的预测器能在每个位置选择表现最佳的SD方法，并准确预测其能接受的Token数量。
    *   **发现：** “Oracle Combine”方案能带来显著的额外加速空间（相较于最佳固定策略，可进一步加速2.2倍），特别是在InstructCoder等两种方法互补性强的任务上。
    *   **启示：** 这为未来研究指明了方向：开发一种准确且轻量级的预测器，能够自适应地根据工作负载、请求和Token位置的变化选择最优的SD策略。


## longcat-flash-thinking
LongCat-Flash-Thinking-2601 Technical Report 

https://arxiv.org/pdf/2601.16725 2026.1.23

https://github.com/meituan-longcat/LongCat-Flash-Thinking-2601
中文解读：https://mp.weixin.qq.com/s/p4f5fbNpdpW4QerMeP2SAw

1. 💡 560b-A27b开源MoE推理模型，在广泛的**agentic基准测试**（包括agentic搜索和工具使用）中，实现了开源模型的顶尖性能。
2. ⚙️ 其卓越的agentic推理能力得益于统一的训练框架，该框架结合了大规模**多环境训练、DORA异步强化学习系统**以及通过**注入真实世界噪声**提升模型鲁棒性的方法。
3. 🧠 模型还引入了“**Heavy Thinking**”模式，通过**并行思考增强推理的深度和广度**；同时，通过实验性的**Zigzag Attention机制，实现了高达1M tokens**的超长上下文高效处理。

<img width="1065" height="558" alt="image" src="https://github.com/user-attachments/assets/4e12afbd-c4db-459e-819d-f93683413e8e" />
<img width="1143" height="445" alt="image" src="https://github.com/user-attachments/assets/e8773df2-77e3-418b-9bca-2c148eb3f8d8" />
<img width="1027" height="393" alt="image" src="https://github.com/user-attachments/assets/fe560cc4-5896-4f80-bf03-cf359bd0f6a8" />
<img width="1144" height="416" alt="image" src="https://github.com/user-attachments/assets/a4f8d8e7-c599-4518-9e9e-b8fc383b075f" />
<img width="1039" height="482" alt="image" src="https://github.com/user-attachments/assets/34dd9d6b-6fd1-47ca-8415-1e9df3a17d1c" />
<img width="1036" height="522" alt="image" src="https://github.com/user-attachments/assets/a1a9cdc6-3d43-48d9-9f10-12333ff3ece8" />

LongCat-Flash-Thinking-2601技术报告介绍了LongCat-Flash-Thinking-2601模型，这是一个拥有560亿参数的开源Mixture-of-Experts (MoE) 推理模型，以其卓越的Agentic Reasoning能力脱颖而出。该模型在Agentic Search、Agentic Tool Use和Tool-Integrated Reasoning等广泛的Agentic基准测试中，在开源模型中达到了最先进的性能。

模型的核心能力源于一个统一的训练框架，该框架结合了领域并行专家训练与后续融合，以及从预训练到后训练阶段在数据构建、环境、算法和基础设施方面的端到端协同设计。该模型的Agentic Reasoning能力通过以下核心方法论得到增强：

**1. 预训练与中训练：**
LongCat-Flash-Thinking-2601的预训练沿袭了LongCat-Flash-Chat的方案，保留了原有数据分布以确保通用推理性能。在此基础上，通过精心设计的中训练阶段进一步将模型扩展到大规模Agentic Reasoning。
*   **长上下文建模：** 引入了分阶段、逐步增加上下文长度的中训练过程，分配了5000亿Token用于32K/128K阶段，并额外分配了400亿Token用于256K阶段。
*   **Agentic数据合成：** 为了弥补Agentic轨迹在真实世界语料库中的稀缺性，构建了一个混合数据合成流程。
    *   **Text-driven Synthesis (文本驱动合成)：** 从大规模文本语料库中挖掘并重构隐式过程为显式Agentic交互轨迹。包括文本过滤与工具提取、合成与精炼（增强Agentic模式多样性、严格质量过滤）。通过Tool Decomposition（逐步隐藏工具参数到环境中）和Reasoning Decomposition（生成多候选动作并合成模型推理以选择最佳）增强结构复杂度。
    *   **Environment-grounded Synthesis (环境驱动合成)：** 直接从可执行环境中构建Agentic数据，以保证逻辑正确性和执行一致性。通过轻量级Python环境、受控工具链采样和执行验证来生成轨迹。基于现有工具定义，实现可验证的Python环境，显式建模工具间依赖，并通过逆向合成（Reverse-Synthesis）和执行验证确保数据与实际执行逻辑相符。
*   **Planning-Oriented Data Augmentation (面向规划的数据增强)：** 专门设计了数据构建策略，将现有轨迹转化为以规划为中心的决策过程，以强化模型的规划能力。第一种侧重于合成有效的问题分解轨迹和正确的初始动作选择；第二种通过在每个决策步骤生成多个替代候选，并训练模型推理和选择，将线性轨迹转化为结构化的多步骤决策过程。

**2. 强化学习扩展：**
*   **RL准备：**
    *   **环境构建：** 针对Agentic任务，需要可靠、可扩展的环境基础。
        *   Code Sandbox：设计了可扩展的执行沙盒系统，提供统一工具接口和高并发调度器，统一处理搜索、文件读写、代码编辑和Shell执行等常用工具。
        *   Agentic Tool-Use Environment：设计了全自动Pipeline，将高层级领域规范转化为可执行图。从领域定义合成特定领域工具集，抽象为统一数据库Schema并生成工具代码，通过单元测试和辅助调试Agent验证。构建工具依赖图G，并基于G进行可验证性保持的环境扩展（Verifiability-preserving Environment Expansion），通过BFS式扩展添加新工具节点，仅当所有依赖已满足时，确保环境可执行性并随着任务难度增加复杂性。环境复杂性增加通过初始工具链$s_1$扩展为更大的子图$R(s_1)$，并基于当前环境结构复杂性$c(E_n)$、从剩余图中识别新有效工具链的难度$g(D_n)$以及剩余未使用的节点数量$|D_n|$来决定是否引入新的种子工具链$s_{n+1}$，概率为$p = f(c(E_n), g(D_n), |D_n|)$。
    *   **初始策略 (Cold-start Policy)：** 旨在为后续大规模探索提供高效初始策略。
        *   General Thinking：采用K-Center-Greedy (KCG) 结合Sliding-Window PPL进行数据选择，强调暴露模型推理能力差距的样本。
        *   Agentic Coding：构建严格的代码交互轨迹策展流程，要求轨迹完全可执行、可验证，并进行细粒度动作级过滤。
        *   Agentic Search：构建合成推理轨迹，强调正确性、推理完整性和抵御快捷行为。通过Graph-based QA Synthesis（从Wikipedia实体构建关系图，生成模糊问题的图基问答）和Agent-based QA Synthesis（多Agent协作，FSM编排，生成高吞吐量、高质量问答对）实现。
        *   Agentic Tool-Use：利用环境扩展Pipeline构建可扩展数据合成Pipeline，确保领域覆盖、轨迹结构和交互长度的多样性，并通过基于Rubric的验证和回合级质量控制进行严格过滤。
    *   **RL任务集：** 通过Agentic Search和Agentic Tool-Use环境构建。
*   **可扩展异步Agentic RL框架 (DORA)：**
    *   **架构：** 采用生产者-消费者架构（RolloutManager、SampleQueue、Trainer）与RPC进行协调。
    *   **全流式异步Pipeline：** 移除RolloutManager内的批处理障碍，实现LLM生成、环境执行和奖励评估的样本粒度执行，并支持多版本异步训练以处理长尾生成问题。
    *   **大规模Agentic训练扩展：** 将RolloutManager分解为Lightweight-RolloutManager和多个RolloutController，利用PyTorch RPC扩展实现CPU空闲感知的远程函数调用和对象实例化。
    *   **PD Disaggregation with CPU Swapping：** 对于MoE模型，将Prefill和Decode部署在独立设备组上，通过KV-cache交换（chunked异步传输）和CPU-resident KV-cache解决设备内存限制和重计算开销，提高生成效率和吞吐量。
![Uploading image.png…]()


**3. Test-Time Scaling (测试时扩展) 通过Heavy Thinking：**
为进一步提升推理能力，引入了Heavy Thinking模式，通过联合扩展推理的广度和深度来优化测试时计算。
*   **两阶段框架：**
    *   并行推理 (Parallel Reasoning)：让一个Thinking模型并行生成多个候选推理轨迹，扩展探索广度。
    *   重度思考 (Heavy Thinking)：利用一个Summary模型对这些轨迹进行反思性推理，综合中间结果以得出最终决策。
*   **上下文内存模块：** 支持工具使用和多轮对话场景，存储消息历史。
*   **Prompting策略：** 设计特定Prompt模板，组织当前回合并行轨迹的排列，促使Summary模型聚合或精炼答案。
*   **增强：** 引入额外的强化学习阶段专门用于Summary阶段。

**4. Zig-Zag Attention设计：**
为解决长上下文效率问题，引入了实验性的Zigzag Attention机制，并发布了LongCat-Flash-Thinking-ZigZag模型。
*   **核心机制：** 结合Multi-head Latent Attention (MLA) 和Streaming Sparse Attention (SSA)，将注意力限制在局部窗口和序列起始的一小组Token，实现亚二次方的计算复杂度。
*   **Zigzag Connectivity：** 层级交错稀疏化策略，约50%的全注意力层被SSA层替换，通过层间组合保持全局信息传递，形成“之”字形连接路径。
*   **Zigzag Integration：** 在中训练阶段引入，通过校准数据集估计注意力层的重要性，将最低重要性层替换为SSA层，并结合YaRN-based positional encoding扩展，支持高达1M Token的序列长度。
*   **效果：** 相较于全注意力，实现约1.5倍的推理速度提升，同时保持推理性能和Agentic能力。

LongCat-Flash-Thinking-2601在数学推理、Agentic Search、Agentic Tool-Use、General QA和Coding等多个基准测试中均展现出极强的竞争力，特别是在Agentic推理任务中达到开源模型的最先进水平，并显著缩小了与闭源领先模型之间的性能差距。
## Kitty 2bitKV
Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost 

https://arxiv.org/pdf/2511.18643 2025.11.23 

https://github.com/Summer-Summer/Kitty

1. 💡 针对LLM推理中KV cache的内存瓶颈及2-bit量化造成的精度显著下降，本文提出了Kitty，一种精确高效的**2-bit KV cache量化与动态通道级精度**提升的算法-系统协同设计方案。
2. ⚙️ Kitty的核心是**Dynamic Channel-wise Precision Boost**，它通过对**Key通道按敏感性排序**，仅将**12.5%~25%的key关键通道保持在高精度**，Value则采用 per-token 滑动窗口量化。
3. 🚀  Kitty设计了**page KV布局和Triton兼容的deQuant内核**，Qwen最大32b/llama8b~70b精度评测。Qwen3-8b A100KV内存减少近8倍，相同内存预算下实现～8倍的批处理量和**2.1~4.1x**的吞吐量提升。
   
<img width="396" height="416" alt="image" src="https://github.com/user-attachments/assets/865cfc61-6da1-4170-bc4e-3115ae8fae32" />
<img width="743" height="344" alt="image" src="https://github.com/user-attachments/assets/11049e21-9ef8-4018-85a6-787a57c75920" />
<img width="787" height="454" alt="image" src="https://github.com/user-attachments/assets/7375799f-1e39-4d38-a2b1-2d854471c10d" />
<img width="394" height="472" alt="image" src="https://github.com/user-attachments/assets/6361f26d-844c-4708-9c1e-07e644173c88" />

## Qwen困惑度悖论
Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs 

https://arxiv.org/abs/2601.11061 2026.1.16 南方科大 阿伯丁大学、穆罕默德·本·扎耶德人工智能大学等

https://github.com/idwts/How-RLVR-Activates-Memorization-Shortcuts 

https://mp.weixin.qq.com/s/qMg2rVabnqS5QbAIaO37pQ

白盒化分析某些模型在预训练的能力 如何在RL阶段激发出来。并且一些诡异的困惑度表现：虚假奖励中，模型对答案的困惑度（Perplexity）持续下降，**但对问题提示的困惑度却不降反升**。像一个学生，为了在考试中答对某道题，不去理解题目本身，而是死记硬背答案。研究团队将这一**现象命名为“困惑度悖论”（Perplexity Paradox），它成为识别记忆激活的关键现象**。

1. 📄 本研究揭示RLVR在LLM（如Qwen2.5/Qwen3）中即使使用**虚假奖励也能提升性能**，这是一种“困惑度悖论”，表明模型通过降低答案困惑度而牺牲提示连贯性来**激活预训练中的记忆**。
2. 💡 采用Path Patching、Logit Lens、JSD分析和Neural Differential Equations等方法，研究人员发现了**隐蔽的“Anchor-Adapter”电路，其中L18-L20是检索记忆化答案**的“Functional Anchor”，L21+则作**为“Structural Adapters”进行表示转换**。
3. 🚀 实验验证了这些电路对数据集的依赖性，并通过**干预特定MLP键**，研究者实现了**对模型性能的双向因果操控**，从而**证实了该机制在数据污染引起**的性能提升中的核心作用。

<img width="1082" height="586" alt="image" src="https://github.com/user-attachments/assets/930cb546-4fb2-450c-a90c-5f27f2144950" />
<img width="894" height="290" alt="image" src="https://github.com/user-attachments/assets/85d866bb-b74e-49d4-b4e0-563eeeae6827" />
<img width="656" height="402" alt="image" src="https://github.com/user-attachments/assets/4c842b32-f5e0-4e15-a6f7-4a4291bfc50f" />

本文深入探讨了“虚假奖励悖论 (Spurious Rewards Paradox)”现象，即强化学习结合可验证奖励 (RLVR) 即使使用虚假或不准确的奖励信号，仍能显著提升大型语言模型 (LLMs) 的性能，例如 Qwen 2.5 系列模型。研究揭示了这种性能提升并非源于模型泛化推理能力的增强，而是激活了模型在预训练阶段可能已记忆的“捷径 (memorization shortcuts)”。是否有办法针对污染路径进行反向干预？论文研究了一种方法：缩放任务相关神经元。

通过逐层投影隐藏状态到词表空间，研究团队直接观察到了答案token的“**诞生过程**”：

<img width="427" height="553" alt="image" src="https://github.com/user-attachments/assets/8dc19ab2-01a4-4a46-bfff-b8b8f6fd535e" />
<img width="363" height="661" alt="image" src="https://github.com/user-attachments/assets/b12da729-5ccf-4213-9811-dbedf864a653" />

在第19层，目标答案首次出现高概率
在第21层，这个概率短暂下降（表征转换）
在第23层，MLP显著注入正确答案，概率激增

**核心发现：**

1.  **困惑度悖论 (Perplexity Paradox)**：模型在虚假 RLVR 训练下，答案令牌 (answer-token) 的困惑度（Perplexity, PPL）显著下降，表明模型正在“记忆”答案；而同时，提示侧 (prompt-side) 的困惑度却上升，这暗示模型为**走捷径而牺牲了通用的语言建模能力和提示语的连贯性**。这与“过度记忆 (over-memorization)”的特征一致，即模型在任务准确性高的情况下，整体语言建模性能却可能下降。

2.  **锚点-适配器电路 (Anchor-Adapter Circuit)**：
    *   **功能性锚点 (Functional Anchor)**：位于模型中间层 (L18-20)。这是**因果决定记忆检索的关键点，高概率的触发令牌** (trigger token) 在此注入。
    *   **结构性适配器 (Structural Adapters)**：位于后续层 (L21+)。这些层**发生显著的权重变化，但并非为了存储新知识，而是进行表征转换** (representational transformation)，以适应并传播来自功能性锚点的捷径信号。
<img width="840" height="470" alt="image" src="https://github.com/user-attachments/assets/647c3e84-1d94-47de-b4ba-107e8c8a67d8" />


## DeepSeekOCR 能力来源质疑

Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR

https://arxiv.org/pdf/2601.03714 2026.1.8 中科院等

https://github.com/dududuck00/DeepSeekOCR

通过系统性实验，揭示了DeepSeek-OCR在高压缩比下表现出的高OCR精度可能**更多地依赖于语言先验，而非真正的视觉理解。**

**句子层面语义扰动的影响**：导致OCR准确率大幅下降，尤其是在高压缩模式下（Tiny模式平均下降11.2%，Small模式下降3.6%，Base模式下降0.6%）。这表明当视觉token稀缺时，全局语言先验知识显著辅助了文本重建。

**词语层面语义扰动的影响**：词语层面的扰动进一步降低了性能。10%的字母打乱（Shuffle）导致 Tiny 模式平均下降11.3%。在零先验随机文本实验中，DeepSeek-OCR 的性能急剧崩溃，Tiny 模式的准确率降至约20%。这证实了模型在很大程度上依赖词汇（n-gram）先验知识，其在压缩模式下报告的“准确率”大部分源于语言幻觉，而非真实的视觉识别。

**跨架构的语言先验依赖**：对13个 OCR/VLM 模型进行基准测试发现，所有端到端（end-to-end）架构均表现出对语言先验知识的严重依赖，在零先验随机文本上，它们的准确率普遍下降40-60%（DeepSeek-OCR Tiny 模式下降68.16%）。相比之下，传统的流水线（pipeline）OCR 方法，如 PaddleOCR-v5，表现出显著更高的抗语义扰动鲁棒性，准确率仅下降4.9%至89.53%，这归因于其将视觉识别与语言解码分离的架构。

**QA和VQA任务表现**：尽管 DeepSeek-OCR 声称 OCR 准确率很高，但在 VQA 任务上其性能接近随机水平（对于四选项问题约20%的准确率）。这表明视觉表征虽然足以触发解码器的语言先验进行文本重建，但未能捕获逻辑推理所需的深层语义关系。与此形成鲜明对比的是，标准 LLM 在直接提供文本内容时能达到90%以上的准确率。这种巨大差异证明，光学压缩破坏了推理所需的结构化语义信息。

**上下文长度限制**：DeepSeek-OCR 的所有模式在文本token数量达到8,000-10,500时均发生性能崩溃。Tiny 模式在约6,000 token后准确率急剧下降，并在8,500 token时降至零。即使是 Base 和 Large 模式也在此附近彻底崩溃。这揭示了当前光学压缩范式的根本限制：固定网格编码器能捕获的信息量是有限的，一旦文本密度超过该限制（约8.5k token/逻辑图像单位），信噪比便会降至解码器恢复阈值以下，使视觉token失去意义。


## Multiplex Thinking
Reasoning via Token-wise Branch-and-Merge 

https://arxiv.org/abs/2601.08808 2026.1.13
https://github.com/GMLR-Penn/Multiplex-Thinking 

1. 💡 Multiplex Thinking 是一种创新的推理机制，它通过在**每个思考步骤中独立采样 K 个候选 tokens 并将其 embeddings 聚合成一个连续的 multiplex token**，从而实现了高效推理。
2. ⚡ 这种方法在保留标准离散生成采样动态的同时，**引入了可追踪的概率分布**，使其能直接通过 on-policy RL 进行优化，并能自适应地在模型不确定时紧凑表示多个 plausible next steps。
3. 🚀 经验证，Multiplex Thinking 在挑战性数学推理基准上一致优于强大的离散 CoT 和 RL baseline，从 Pass@1 到 Pass@1024 都展现出更高的准确性，同时生成更短的序列。

<img width="786" height="284" alt="image" src="https://github.com/user-attachments/assets/c91c498a-34bb-47eb-a442-2f9dda3cf1a2" />
<img width="755" height="411" alt="image" src="https://github.com/user-attachments/assets/853377d4-5096-4b93-9cda-891db05d7df8" />
在每一步推理中，模型从当前概率分布中**独立采样K个离散令牌**，将它们的one-host向量平均后，通过**嵌入矩阵映射为连续多路令牌**。这个过程保留了词汇表嵌入的先验和采样动态：当分布集中时（低熵），**多路令牌退化为标准离散令牌**；当分布**分散时（高熵），它编码多种可能路径，实现高效探索**。
<img width="1047" height="325" alt="image" src="https://github.com/user-attachments/assets/57e02621-4f3a-495c-ab5d-cb5af6136c69" />

更重要的是，由于采样是独立的，整个多路轨迹的概率可以明确计算，这使得它能直接用于强化学习优化。公式上，多路令牌的概率是各采样令牌概率的乘积，从而支持基于策略的强化学习目标。报告还分析了熵的变化：多路令牌的熵随K值线性增长，相当于将探索空间从|V|扩展到|V|^K，大幅提升了发现正确路径的概率。

## MEPIC
MEPIC: Memory Efffcient Position Independent Caching for LLM Serving  

https://arxiv.org/pdf/2512.16822 2025.12.18 华为
中文解读：https://mp.weixin.qq.com/s/x8loHgF4LSdWR_BIBGUzVg

将chunk KV与分页存储对齐、将重计算从token级别转移到块级别（仅首个块为请求特定），以及利用注意力核中的RoPE融合来消除位置编码，从而使得剩余块可完全共享。
实验结果表明，MEPIC相较于最先进的PIC技术可减少高达2倍的HBM内存使用，对于长提示甚至可达5倍，同时保持了可比的延迟和准确性，且无需修改模型。基于vLLM。
不足：模型只选择了Mixtral-7b！！

<img width="757" height="476" alt="image" src="https://github.com/user-attachments/assets/b4bd5043-99d8-4726-b939-3592b90fe3e1" />

<img width="766" height="745" alt="image" src="https://github.com/user-attachments/assets/3913a686-5b6c-4bb3-ac42-3b953b893b48" />

<img width="939" height="429" alt="image" src="https://github.com/user-attachments/assets/774b1cd6-e248-4e20-8d66-e33eea993f1c" />

## SDC
中文汇总（通用 + 偏训练）https://mp.weixin.qq.com/s/_tlHqVmHjul8XvUvd0OQOg 

PyTorch 训练时可以从如下几方面保证：
随机种子：random 库、numpy 库，torch，torch.cuda 的随机种子等。
CUDNN 确定性：torch.backends.cudnn.deterministic = True 和 torch.backends.cudnn.benchmark = False。
PyTorch 的确定性算法：torch.use_deterministic_algorithms(True)。
环境的一致：不同的硬件，GPU Driver，CUDA Version 也可能引入误差。
DataLoader：DataLoader 也可能引入顺序的不一致，需要保证分布式场景 DataLoader 的不一致性。
NCCL 通信： NCCL 中的 AllReduce 可以通过指定算法、协议、拓扑等最大程度降低不确定性，但是依然无法严格保证。

https://arxiv.org/abs/2502.12340 Understanding Silent Data Corruption in LLM Training

https://www.opencompute.org/documents/sdc-in-ai-ocp-whitepaper-final-pdf
https://arxiv.org/abs/2509.16293
https://arxiv.org/abs/2509.01322
https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-535-288-01/index.html

## Uniqueness-Aware RL
Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs 

https://arxiv.org/abs/2601.08763 2026.1.15 MIT等

https://github.com/zhiyuanhubj/Uniqueness-Aware-RL 代码待开源，目前空。

中文：https://mp.weixin.qq.com/s/3qiXUQYhPzyPH8xkL-KznQ

**增加多样性**：现有的方法试图通过增加「随机性」（Entropy Bonus）来解决这个问题，但它们只是在**词元（Token） 层面**增加随机性。举个例子：模型可能会生成 设 x 为…… 和 令 x 等于……。这在 Token 上是不同的，但在解题策略上是完全一样的。**这某种程度上是「虚假的探索」**
应该重奖那些「使用了全新解题思路」的答案。这就是题目中 Rewarding the Rare（奖励稀缺） 的含义。
- 如果 10 个答案里，有 8 个用了「套公式法」，那这 8 个答案即便对了，奖励也要打折，因为它们太普通了。
- 如果有 1 个答案用了「对称性法」，且做对了，那这个答案非常珍贵，我们要给**它加倍的奖励**。
引入了策略层面的唯一性（Strategy-level Uniqueness）
模型：Qwen2.5-7b，Qwen3-8b，OLmo3-7b
<img width="761" height="392" alt="image" src="https://github.com/user-attachments/assets/965443d7-6dca-4750-b860-94568b66f9ed" />
![Uploading image.png…]()

## TTT-Discover
Learning to Discover at Test Time 

https://arxiv.org/abs/2601.16175 2026.1.22 斯坦福 NVIDIA TogetherAI等

https://github.com/test-time-training/discover

中文解读：https://mp.weixin.qq.com/s/uPcljKORt_7wnyD9q_BGRw

https://www.gpumode.com/v2/home 
https://www.gpumode.com/v2/leaderboard/496?tab=rankings 

它不追求「平均分」，它只想要那一次可复现的满分！
1. 💡 TTT-Discover提出一种测试时训练（Test-Time Training）方法，通过在特定科学问题上对大型语言模型（LLM）进行强化学习来发现新的state-of-the-art解决方案。
2. 🚀 该方法采用**熵目标函数和PUCT（Predictor-Update Confidence Tree）启发式状态重用机制**，旨在**优化生成单一最佳解决方案**而非追求平均性能。
3. 🏆 基于开源模型gpt-oss-120b实验，训练采用LoRA微调。在数学、GPU核工程、算法设计和生物学等多个领域刷新了SOTA。
局限：目前只能解决那些有连续奖励信号的问题，比如代码运行速度（越快越好）、数学边界（越小越好）。

与以往在测试时冻结 LLM 并通过搜索进行优化的方法不同，TTT-Discover 允许 LLM 在测试时继续训练，并通过特定于测试问题的经验进行学习。这种持续学习形式的目标是**找到一个最佳解决方案**，而非平均表现优异的多个解决方案，**并且专注于解决特定问题，而非泛化到其他问题**。

传统的 AI 系统在部署后通常保持静态，难以应对不断变化的世界和超出其训练数据范围的难题。例如，在科学发现问题中，解决方案可能需要超越现有知识的新思路。现有方法，如 AlphaEvolve，通过提示冻结的 LLM 进行搜索，并利用手工设计的启发式方法（如进化搜索）来生成新的提示。然而，这些方法无法使 LLM 本身进行学习和改进。本文认为，在**处理复杂问题时，学习（Learning）通常比搜索（Search）更具优势**，因此提出在**测试时对LLM进行持续训练，使其从解决特定问题的尝试中获取宝贵的、问题专属的训练数据**。

<img width="791" height="209" alt="image" src="https://github.com/user-attachments/assets/ea05d75a-5d37-428f-9228-b5e97d418af7" />
<img width="860" height="409" alt="image" src="https://github.com/user-attachments/assets/22808e22-a291-485f-bc11-e0171e1db209" />
<img width="879" height="563" alt="image" src="https://github.com/user-attachments/assets/b5f81857-ea6a-454a-aa29-1f650625145b" />
<img width="877" height="486" alt="image" src="https://github.com/user-attachments/assets/6f11adec-86d6-4656-975d-2385dedf9f22" />
<img width="650" height="309" alt="image" src="https://github.com/user-attachments/assets/5dc4bbaf-f647-497a-a804-3d9683b65407" />
<img width="1225" height="450" alt="image" src="https://github.com/user-attachments/assets/2755a79b-42e5-49f4-941d-b0e4660e6562" />
<img width="651" height="273" alt="image" src="https://github.com/user-attachments/assets/611194ce-4f94-4015-bdd2-9c9acd550334" />


## LLM in Sandbox
LLM-in-Sandbox Elicits General Agentic Intelligence

https://arxiv.org/abs/2601.16206v1 2026.1.22 微软 人大等

https://github.com/llm-in-sandbox/llm-in-sandbox

https://mp.weixin.qq.com/s/AebCuUQZ3RBI5oG0uU64Rg 

LLM-in-Sandbox不需要额外的训练，强大的大模型就能自发地**利用代码沙箱解决数学、物理、化学甚至生物医学等非代码领域的难题**。通过工具产生结果而非回归生成，显著降低了长文本场景下的**Token消耗**（最高达8倍）。
且sandbox就和LLM部署在同一台机器上。**论文的实验都是单台DGX GPU服务器：每个沙盒容器在空闲时仅消耗约 50 MB内存，峰值时～ 200 MB。对于一台拥有 2TB 内存的典型 GPU 服务器，即使同时运行 512 个并发沙盒**，也仅占用约 5% 的内存资源。且延迟极低。
<img width="649" height="329" alt="image" src="https://github.com/user-attachments/assets/1f344bae-2821-41ac-a492-da1eb4aaca29" />
<img width="675" height="457" alt="image" src="https://github.com/user-attachments/assets/d50f9575-77ac-47ea-8620-830aa9a669a2" />

<img width="799" height="393" alt="image" src="https://github.com/user-attachments/assets/a8ab30f2-f0c5-44b8-9f1c-6421b4d688cb" />
<img width="498" height="715" alt="image" src="https://github.com/user-attachments/assets/cdfe9bfa-6f90-4f11-8e26-8024348dc9a5" />
<img width="669" height="354" alt="image" src="https://github.com/user-attachments/assets/11f79ace-a6ea-4da2-91ff-a83857876ed7" />
<img width="932" height="268" alt="image" src="https://github.com/user-attachments/assets/80ca4a16-8118-4a0b-af46-b65d2260c852" />

**核心思想与方法 (Core Idea and Methodology)**

论文的核心在于将LLM与一个具有以下三种元能力（meta-capabilities）的虚拟计算机相结合：
1.  **外部资源访问 (External resource access)**：例如访问互联网、调用API。
2.  **文件管理 (File management)**：进行文件的读取、写入和组织。
3.  **代码执行 (Code execution)**：运行任意程序。

这使得LLM能够像人类使用计算机一样解决问题。

**LLM-in-Sandbox环境设计 (LLM-in-Sandbox Environment Design)**

LLM-in-Sandbox的设计遵循“最小化”和“探索性”原则。
*   **代码沙盒 (Code Sandbox)**：采用轻量级、通用目的的Docker容器，提供Ubuntu基础系统和终端访问。与现有SWE agents（例如Claude Code）使用的复杂、特定任务环境不同，**LLM-in-Sandbox环境仅预装标准的Python解释器和必要的科学计算库（如NumPy, SciPy）**。模型被鼓励在运行时自主安装所需的领域特定工具（如Java运行时、RDKit等）。这种设计保证了通用性（同一环境支持多种任务）和可扩展性（统一设置便于大规模推理和训练，存储占用仅约1.1 GB）。
*   **最小化工具集 (Minimal Toolset)**：LLM在沙盒中拥有三个基本工具：
    *   `execute_bash`: 执行任意bash命令，这是最基础且通用的接口，允许安装包、管理文件和运行程序。
    *   `str_replace_editor`: 用于文件创建、查看和编辑。
    *   `submit`: 指示任务完成。
*   **工作流 (Workflow)**：工作流基于ReAct框架，模型迭代地推理和行动。在每个回合中，模型生成一个工具调用（例如`execute_bash('ls -la /testbed/documents/')`），从**沙盒接收执行结果，并根据观察决定下一步行动。这个多回合交互持续到模型调用`submit`或达到最大回合限制。**
    *   **提示工程 (Prompting)**：系统提示（System Prompt）**引导模型充分利用沙盒，鼓励其使用计算工具而非自然语言进行计算**，并**通过程序执行获取答案，而不是直接硬编码结果。**
    *   **输入/输出处理 (Input/Output Handling)**：利用沙盒的文件系统灵活处理输入和输出。任务输入可从模型提示提供，也可通过文件（例如将长文本文档放置在`/testbed/documents/`中）。最终输出被指示写入指定位置（例如`/testbed/answer.txt`），确保仅包含最终结果。

**训练前的泛化能力 (Training-free Generalization Capabilities)**

论文首先展示了在不进行额外训练的情况下，强大的LLM（如Claude-Sonnet-4.5-Thinking、GPT-5、DeepSeek-V3.2-Thinking）如何自发地利用代码沙盒解决非代码任务，表现出显著的泛化能力。
*   **性能提升 (Performance Gains)**：在数学、物理、化学、生物医学、长上下文理解和指令遵循等六个非代码领域均获得显著性能提升。例如，Qwen3-Coder**在数学任务上性能提升高达+24.2%**。
*   **沙盒利用分析 (Sandbox Utilization Analysis)**：
    *   **案例研究 (Case Study)**：模型自**主安装领域特定工具（如化学任务中的Java和OPSIN库），利用文件管理工具（如`grep`和`sed`）处理长文档，并编写Python脚本进行字符计数、词语重叠检测等复杂计算**。
    *   **定量分析 (Quantitative Analysis)**：统计模型对外部资源访问、文件管理和计算能力的使用频率。强大的模型会根据任务需求调整使用模式，例如数学任务中计算频率高（43.4%），化学任务中外部资源访问频率高（18.4%）。在长上下文任务中，将文档置于沙盒而非提示中，能带来显著性能提升，最高可达8倍的token消耗降低。

**LLM-in-Sandbox强化学习 (LLM-in-Sandbox Reinforcement Learning, LLM-in-Sandbox-RL)**

为进一步提升模型的沙盒探索能力，论文提出了LLM-in-Sandbox-RL。
*   **方法 (Method)**：该方法在沙盒内使用通用的、非代理（non-agentic）的上下文相关（context-based）数据训练LLM。
    *   **数据源 (Data Source)**：采用来自Instruction Pre-Training的通用上下文相关任务数据集，包含百科、小说、专家材料、新闻等多种领域。
    *   **沙盒配置 (Sandbox Configuration)**：任务上下文以文件形式存储在沙盒内（`/testbed/documents/`），多文档或长上下文会被分割成多个文件，单文件上下文会添加无关的干扰文件，以鼓励模型主动探索和筛选信息。
    *   **RL训练 (RL Training)**：采用基于结果的奖励（outcome-based rewards）。与仅在文本模式下训练的LLM-RL基线不同，LLM-in-Sandbox-RL在沙盒模式下生成轨迹。
*   **实验结果 (Experimental Results)**：
    *   **广泛泛化 (Broad Generalization)**：LLM-in-Sandbox-RL在训练数据之外的所有评估领域（包括SWE任务）都取得了性能提升，即使训练数据与这些任务无重叠。
    *   **模型能力提升 (Improved Model Capabilities)**：对于初始代理能力较弱的模型（如Qwen3-4B-Instruct），LLM-in-Sandbox-RL训练后，其在沙盒模式下的性能显著优于LLM模式。对于强模型，仍能获得持续提升。
    *   **推理模式泛化 (Generalization across Inference Modes)**：尽管LLM-in-Sandbox-RL完全在沙盒模式下训练，但它也出人意料地提升了LLM模式（即非沙盒直接生成）的性能，甚至优于LLM-RL。这表明通过沙盒交互学习到的代理技能可以迁移到非代理生成中。
    *   **数据消融 (Data Ablation)**：对比了不同训练数据（数学、SWE、通用提示内、通用沙盒内）的效果，通用沙盒内训练效果最佳，强调了沙盒交互的重要性。
*   **泛化分析 (Analysis on Generalization)**：训练后，模型对沙盒能力的利用率显著提高，特别是弱模型，其平均回合数大幅减少（从23.7降至7.0），表明其学会了更有效和有目的的交互。此外，模型在非沙盒LLM模式下的推理模式（如结构化组织和验证行为）也得到了增强，这归因于沙盒交互中每次行动都能得到明确反馈的学习过程。

**效率部署 (Efficient Deployment)**

论文分析了LLM-in-Sandbox在实际系统中的效率。
*   **计算成本 (Computational Cost)**：
    *   **Token消耗 (Token Consumption)**：通常沙盒模式由于多回合探索会消耗更多token。然而，在长上下文任务中，通过将内容存储在本地文件而非提示中，可将token消耗减少高达8倍（从100K降至13K token）。总体而言，LLM-in-Sandbox模式的平均总token消耗仅为LLM模式的0.5-0.8倍。
    *   **速度 (Speed)**：沙盒模式下，大量token来自环境输出，这些token通过快速的Prefill（而非慢速的自回归解码）处理。环境执行时间仅占总时间的不到4%。总体查询吞吐量（Queries Per Minute, QPM）具有竞争力，部分模型甚至实现加速（如MiniMax实现2.2倍加速）。
*   **沙盒基础设施开销 (Sandbox Infrastructure Overhead)**：开销可忽略不计。存储方面，LLM-in-Sandbox使用一个通用的Docker镜像（约1.1 GB），显著小于特定任务SWE agents所需的数TB存储。内存方面，每个沙盒容器空闲时约50 MB，峰值约200 MB，即使并发512个沙盒也仅占用DGX节点总RAM的5%。

**超越文本生成 (Beyond Text Generation)**

LLM-in-Sandbox使LLM能够超越传统的“文本输入-文本输出”范式，解锁了独立LLM无法实现的能力：
*   **跨模态能力 (Cross-Modal Capabilities)**：通过编排沙盒内的专业软件，LLM可以处理和生成图像、视频、音频和交互式应用程序。
*   **文件级操作 (File-Level Operations)**：直接生成可用的实际文件（`.png`, `.mp4`, `.wav`, `.html`），并从实际执行中获得反馈。
*   **自主工具获取 (Autonomous Tool Acquisition)**：LLM可以按需自主发现、安装和学习使用任意软件库，实现无限的工具访问。
论文通过生成交互式地图（.html）、会议海报（.png）、动画视频（.mp4）和原创音乐（.wav/.mid）的案例研究，展示了其巨大的潜力。尽管当前结果仍有局限，但这一范式代表了迈向通用数字创作系统和通用智能的有力方向。

**结论与未来工作 (Conclusion and Future Work)**

论文认为LLM-in-Sandbox有望成为LLM服务的默认范式，将LLM从文本生成器转变为通用数字工作者。它也提供了一个评估代理能力的标准化基准，通过$\Delta = \text{LLM-in-Sandbox} - \text{LLM}$衡量模型利用计算环境的能力。未来工作包括推动“沙盒原生模型训练（Sandbox-Native Model Training）”，将沙盒交互作为首要训练目标，并通过大规模强化学习和预训练阶段融入沙盒式推理。

## PLA-Serve
PLA-Serve: A Prefill-Length-Aware LLM Serving System 

https://arxiv.org/pdf/2601.11589 2026.1.4

1. 💡 PLA-Serve通过识别并解耦大型语言模型服务中不同提示长度的请求，以解决**长预填充（compute-bound）和短预填充**（memory-bound）请求之间的计算-内存干扰问题。
2. ⚙️ 该系统为短预填充工作负载引入了长度感知智能批处理机制，结合自适应等待窗口（AWD）和基于CUDA Graph的聚类，以减少批处理延迟并提升吞吐量。
3. 📈 PLA-Serve采用双队列设计，支持单实例的时间解耦或多实例的空间解耦，在实际多轮对话场景中显著降低了预填充延迟和SLO违规率，并提高了请求吞吐量。

LA-Serve的核心洞察是，预填充阶段内，长序列（计算密集型）和短序列/重新预填充（re-prefill，通常为内存密集型）具有截然不同的性能瓶颈，混合调度会导致显著的相互干扰。
核心问题与背景
LLM serving面临的主要挑战之一是时延敏感性和高并发性。预填充阶段（计算第一个token）通常是计算密集型的，而解码阶段（自回归生成后续token）是内存密集型的。PD disaggregation（预填充与解码分离）将这两个阶段部署在不同的实例上，以避免跨阶段资源竞争。然而，即使在PD disaggregation之后，预填充实例内部仍存在干扰。论文指出，在LMsys-Chat-1M等真实世界多轮对话数据集中，大部分prompt是短的（<256 token），而长上下文请求（>1K token）相对较少。短的预填充/re-prefill请求通常是内存密集型的（受KV-cache读写限制），而长的预填充请求是计算密集型的（受GEMM吞吐量限制）。将它们混合在一个批次中进行处理，会导致“队头阻塞（head-of-line blocking）”和计算-内存干扰：短请求等待长的GEMM操作，导致TTFT（Time-to-First-Token）飙升；而长请求的有效FLOPs（浮点运算每秒）因短请求的大量KV流量而降低。

## JetRL-FP8
Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow

https://arxiv.org/abs/2601.14243  2026.1.19 NVIDIA MIT han song；伯克利；斯坦福等

1. 🤔 现有RL训练中，rollout阶段效率低下，BF16-train + FP8-rollout策略在长序列生成和复杂任务下表现出严重的训练不稳定性和灾难性精度下降。
2. 💡 为解决此问题，本文提出Jet-RL，一个**统一FP8精度流的on-policy RL训练框架**，确保训练和rollout**之间数值一致性，从而实现稳定优化**。W采用128*128 per-block；激活和梯度采用1-128量化。
3. 🚀 基于VeRL+vLLM，LLama/Qwen最大8b 最长16k，Jet-RL在保持与**BF16训练相当的收敛和精度下**，实现了高达**33%的rollout阶段加速、41%的训练阶段加速和16%的端到端加速**。
<img width="829" height="625" alt="image" src="https://github.com/user-attachments/assets/5453671c-a626-423f-be7e-54753af60a73" />

**研究背景与问题：**
强化学习在增强 LLM 复杂推理能力方面至关重要，但其训练流程计算效率低，资源消耗大。其中，rollout 阶段通常占据总训练时间的 70% 以上 (见 Figure 2)，成为主要瓶颈。FP8 量化因其显著的效率增益而被认为是加速 RL 训练的有效途径。目前普遍采用的策略是保持 BF16 精度进行训练，而在 rollout 阶段使用 FP8 量化（BF16-train + FP8-rollout）。然而，本文发现这种策略存在严重缺陷：
1.  **长期 Rollout 生成中的不稳定性：** 在长序列生成（例如超过 8K tokens）时，该策略会导致严重的精度下降甚至灾难性崩溃 (见 Figure 3)。这是因为训练和 rollout 精度之间的微小数值差异在长序列中会逐渐累积，放大离策略训练的影响，导致生成轨迹发散并使 RL 训练不稳定。
2.  **在挑战性任务中的失败：** 在面对更难的推理任务或使用较弱的基础模型时，BF16-train + FP8-rollout 策略的表现会迅速与 BF16 训练结果出现分歧 (见 Figure 4)。当模型对自身响应的信心不足时，量化引起的误差会严重扭曲 rollout 轨迹，导致训练不稳定和性能下降。
<img width="856" height="324" alt="image" src="https://github.com/user-attachments/assets/7d406449-fffc-4c2f-9257-723edbda9e61" />
<img width="865" height="553" alt="image" src="https://github.com/user-attachments/assets/87cdedb7-fbdf-4e67-b5b3-796a32449c76" />
<img width="423" height="262" alt="image" src="https://github.com/user-attachments/assets/0641071d-d091-460d-a1d6-b2d49656b2f2" />

这些问题源于训练和推理之间存在的“离策略”性质，即 FP8 量化 rollout 和 BF16 训练更新之间存在显著的数值不匹配。

**Jet-RL 核心方法：**
Jet-RL 的核心思想是强制在训练和 rollout 之间采用**统一的 FP8 精度流（unified FP8 precision flow）**，以实现真正的“在策略”（on-policy）FP8 RL 训练，从而稳定 RL 训练并消除策略不匹配。

1.  **统一精度流的建模与实现：**
    *   作者将模型中量化精度传播建模为有向图 $ \mathcal{G} = (\mathcal{V}, \mathcal{E}) $，其中节点 $ v_i \in \mathcal{V} $ 代表操作符或权重，边 $ (v, v') \in \mathcal{E} $ 描述了张量在连接操作符之间的传播及其精度和量化粒度。
    *   在 BF16-train + FP8-rollout 策略中，训练的前向图 $ \mathcal{G}_{\text{fwd}}^{\text{train}} $ 和推理图 $ \mathcal{G}^{\text{infer}} $ 是不同的，导致训练与 rollout 之间的不匹配。
    *   Jet-RL 通过强制 $ \mathcal{G}^{\text{infer}} $ 成为 $ \mathcal{G}_{\text{fwd}}^{\text{train}} $ 的子图来解决此问题，确保所有边（张量）的属性（精度和粒度）保持一致。
    *   尽管训练过程中会维护一个 BF16 的高精度主副本（master copy）以稳定训练，但**前向传播的量化行为在训练和推理框架**中保持一致，从而减轻了不匹配问题。
    *   对于反向传播，为了保持模型精度，操作符之间传输的**梯度保留在 BF16 精度**。然而，反向传播中的 **GEMM 操作（DGrad 和 WGrad）也量化为 FP8** 以加速。

2.  **GEMM 量化粒度：**
    *   考虑到 FP8 的逐张量（per-tensor）量化在 LLM 训练中不稳定，Jet-RL 采用了更细粒度的量化策略。
    *   **权重：** 使用 128x128 的**逐块（per-block）量化**。
    *   **激活和梯度：** 使用 1x128 的**逐组（per-group）量化**。
    *   **FProp (前向传播)：** 输入激活进行 1x128 逐组量化，权重进行 128x128 逐块量化。为满足硬件要求（row-wise $ \times $ column-wise），**激活和权重均以 row-wise 布局存储**。激活的量化可与前一个操作符融合以减少开销，权重的量化则在参数更新阶段进行，开销可忽略。
    *   **DGrad 和 WGrad (反向传播)：**
        *   DGrad 的工作负载与 FProp 相似，可重用相同的 1x128 量化矩阵乘以 128x128 量化矩阵的核配置。
        *   WGrad 遵循 DeepSeek-V3 的设计，将第一个矩阵量化为 1x128，第二个量化为 128x1（这种更细粒度的设计有助于稳定训练）。
        *   这两个操作都需要量化梯度，但分别要求 1x128 和 128x1 的量化，Jet-RL 融合了这些量化过程。
        *   权重的量化方案沿通道和行轴对称，在反向传播中值不变，只需进行转置。
        *   激活在正向传播中量化为 1x128，但在反向传播中需要量化为 128x1，这要求在反向传播中再次进行量化。

3.  **实现细节：** Jet-RL 使用 vLLM 作为推理引擎，VeRL 作为 RL 训练框架。量化 GEMM 借鉴了 DeepGEMM 的核，并使用 Triton 实现了量化、转置和融合的激活或 RMSNorm 核。

**实验评估：**
实验在 Llama3.1-8B, Qwen2.5-7B, 和 Qwen3-8B-Base 等模型上，使用 GSM8K + MATH 和 DeepMATH 数据集，以及 8K 和 16K 的 rollout 长度进行。
1.  **精度评估：**
    *   **8K Rollout：** BF16-train + FP8-rollout 方法表现出显著不稳定性，在 Qwen2.5-7B 上甚至无法收敛，在其他模型上性能大幅下降（Llama3.1-8B 平均分数下降 9.8%）。相比之下，Jet-RL 在所有场景下都能稳定收敛，并且将与 BF16 基线的性能差距大幅缩小（Qwen2.5-7B 仅下降 1.0%，Qwen3-8B-Base 仅下降 1.1%），在 Llama3.1-8B 上甚至超越了 BF16 基线 2.0% (见 Table 2)。
    *   **16K Rollout / DeepMATH：** BF16-train + FP8-rollout 在 16K rollout 的 Qwen3-8B-Base 上未能收敛，在 DeepMATH 上性能严重下降 10.3%。Jet-RL 成功解决了这些问题，收敛性更强，与 BF16 基线差距显著缩小（Qwen3-8B 仅下降 2.7%，DeepMATH 仅下降 0.9%，Qwen2.5-7B 仅下降 3.0%）(见 Table 3)。
2.  **效率评估：**
    *   **Rollout 加速：** FP8 相较于 BF16 实现了 1.07x 到 1.33x 的持续加速 (见 Table 4)。加速比随模型规模增大而提高（32B 模型达到 1.33x），但高张量并行度（TP）会降低加速效果（如 32B 模型在 TP=4 时加速比为 1.1x，而 TP=2 时为 1.3x），这表明通信开销成为限制因素。
    *   **端到端加速：** 对于 Qwen3-8B 模型和 8K rollout 长度，FP8 量化在 actor 更新阶段实现 1.54x 加速，在 Reference Model 推理阶段实现 1.80x 加速，共同使训练阶段吞吐量提高了 1.41x。结合 rollout 的加速，整体端到端步长耗时提速 1.16x。

# StaleFlow
Unleashing Efficient Asynchronous RL Post-Training via Staleness-Constrained Rollout Coordination

https://arxiv.org/pdf/2601.12784 2026.1.19 北大 崔斌团队 上交 腾讯等。

1. StaleFlow旨在解决**异步强化学习**（RL）后训练中普遍存在的**数据陈旧度（data staleness）和数据倾斜**（data skewness）两大挑战，通过引入**全局一致性协议严格控制陈旧度**，并创新性地重塑系统架构以有效缓解倾斜。
2. 该系统通过虚拟**陈旧度缓冲区**（virtual staleness buffer）实现**细粒度的轨迹级**（trajectory-level）陈旧度管理，确保即使在支持如**部分rollout和迁移等高级协调技术时，也能严格遵守预设的陈旧度上限**。
3. StaleFlow通过**Trajectory Server (TS) 和 Parameter Server (PS) 解耦**数据流，并由中央协调器（**centralized coordinator**）运用一**系列吞吐量导向的rollout协调策略**，最终实现了高达2.68倍的系统吞吐量提升，同时保持了收敛性（测了几十～100step）。

-   StaleFlow实现基于 Python 合计22K LoC: 1) Staleness: 2K; 2) 支持Megatron和FSDP，2K； 3）Rollout service: 10K, 其中PS 4K， TS 1K；vLLM: 3K, coordinator:2K; 利用 NVIDIA NIXL和UCX进行高效通信。
-   PS部署在 CPU 资源上，与训练和 Rollout 工作器协同工作，通过 PCIe DMA 和 RDMA 实现高效的参数传输。PS 的通信开销不随集群规模增加而显著增长。
-   Redundant Rollout 在消融研究中被证明可以进一步缩短平均轨迹长度并略微提升吞吐量，但考虑到它会改变生成长度分布，未在主要对比实验中启用。
  
<img width="552" height="375" alt="image" src="https://github.com/user-attachments/assets/147b4ca8-14d4-4328-85f1-67d1a3967159" />
<img width="550" height="362" alt="image" src="https://github.com/user-attachments/assets/be62e45c-fd67-45b5-8403-2570b68db965" />

异步 RL 系统中普遍存在的 **数据陈旧性（data staleness）** 和 **数据倾斜性（data skewness）** 问题。现有系统往往需要在 RL 收敛性和系统性能之间进行权衡，因为严格控制陈旧性会限制数据倾斜性缓解技术的使用，而激进的倾斜性缓解则会加剧数据陈旧性。StaleFlow 通过创新的架构设计和算法策略，在保持严格的陈旧性约束下，显著提高了系统吞吐量。
<img width="736" height="641" alt="image" src="https://github.com/user-attachments/assets/0e5dc929-5447-48c4-a978-432d53915fa9" />
<img width="1498" height="450" alt="image" src="https://github.com/user-attachments/assets/95a9821e-4f85-4005-807f-955a1bbdf1f2" />

**背景与动机**
异步执行带来了两个关键数据问题：
1.  **数据陈旧性（Data Staleness）**：由于 rollout 阶段使用的模型参数可能不是最新的，训练阶段会消耗**陈旧轨迹（stale trajectories）**，这可能损害 RL 的**收敛性（convergence）**。RL 算法通过重要性采样（Importance Sampling）或修正来容忍一定程度的模型不匹配，但过度的陈旧性会导致收敛性下降。可以通过设定**陈旧性上限（staleness bound）** $\eta$ 来限制。
2.  **数据倾斜性（Data Skewness）**：轨迹长度差异大，导致**工作负载倾斜（workload skewness）**。这体现在**实例内部（within-instance）**（短轨迹早完成，长尾轨迹长时间占用资源）和**实例之间（across-instance）**（不同实例进度不同，快实例需等待慢实例）。这两种倾斜都会降低资源利用率和系统吞吐量。

现有系统应对这些问题的方式各有侧重：
-   **严格陈旧性控制**（如 VeRL-Async、AReaL、Roll Flash）：允许用户定义 $\eta$，通过限制**在途数据（in-flight data）**量来强制执行，但限制了多种 **rollout coordination** 技术的使用。
-   **一步陈旧性**（如 VeRL-Pipeline、AsyncFlow、RhymeRL）：固定 $\eta=1$，保证了更灵活的协调，但无法根据需求放宽 $\eta$ 以获得更高性能。
-   **无陈旧性保证**（如 LlamaRL、Laminar、APRIL、SortedRL）：不限制陈旧性，自由应用协调技术，但可能导致陈旧性无界，损害收敛。

**rollout coordination 技术**主要包括：
-   **Partial Rollout Trajectories**：中断进行中的轨迹，同步模型，重计算 **KV Cache**，然后恢复生成。
-   **Redundant Rollout Trajectories**：过采样轨迹，丢弃长尾轨迹。
-   **Multi-Version Rollout Instances**：不同实例独立同步模型。
-   **Migration Across Rollout Instances**：动态迁移轨迹到其他实例。

**StaleFlow 的核心贡献**在于：
1.  提出了一种**全局一致性协议（global consistency protocol）**，在轨迹级别强制执行严格的陈旧性控制，同时支持高级 rollout coordination 技术。
2.  引入了架构创新，通过**轨迹服务器（Trajectory Server, TS）**和**参数服务器（Parameter Server, PS）**将数据移动解耦，实现了灵活的 rollout coordination。
3.  设计了一套**陈旧性感知（staleness-aware）**、**吞吐量导向（throughput-oriented）**的 rollout coordination 策略，通过**快照-命令循环（snapshot-command cycle）**统一各种技术以最大化系统吞吐量。

**核心方法学**

StaleFlow 的设计围绕**陈旧性管理器（Staleness Manager）**和增强的 **Rollout Service** 展开。

**1. 控制数据陈旧性：全局一致性协议**
StaleFlow 引入**虚拟陈旧性缓冲区（virtual staleness buffer）**抽象和**轨迹版本标识符（$V_{traj}$）**来精细化控制陈旧性。
-   **轨迹版本标识符（$V_{traj}$）**：每个初始轨迹都被分配一个 $V_{traj}$，表示用于生成该轨迹的模型版本。如果启用了 Partial Rollout，则 $V_{traj}$ 指的是可容忍的最旧模型版本。$V_{traj}$ 由 Rollout Coordinator 和 Staleness Manager 共同处理。
-   **虚拟陈旧性缓冲区（Virtual Staleness Buffer）**：
    -   Staleness Manager 维护一系列虚拟缓冲区，每个缓冲区容量等于**批次大小（batch size）**。
    -   每个缓冲区有一个**缓冲区版本（$V_{buf}$）**，表示该缓冲区中的轨迹将用于训练模型从 $V_{buf}$ 更新到 $V_{buf}+1$。
    -   **陈旧性约束**：所有轨迹必须满足 $V_{traj} + \eta \ge V_{buf}$，其中 $\eta$ 是用户定义的陈旧性上限。
    -   **核心原语**：
        1.  **Reserve**：当轨迹开始执行时，Staleness Manager 从 $V_{buf} = V_{traj} + \eta$ 开始向后扫描，保留一个空条目作为占位符。这代表了在不违反 $\eta$ 的情况下，轨迹可能驻留的最差情况缓冲区位置。
        2.  **Occupy**：当轨迹生成完成并计算奖励后，Staleness Manager 从前向扫描，贪婪地占用最早可用的空条目，将轨迹注册为可供消费。
        3.  **Consume**：训练工作器消耗一个完整的缓冲区后，该缓冲区的数据不再被跟踪。
    -   **缓冲区状态**：缓冲区可以处于 **Waiting**（有空条目），**Ready**（所有条目被 Occupy，可供训练），或 **Stuck**（已满但包含未完成的 Reserved 条目）。
    -   **条目删除和移动**：当一个 Reserved 条目完成并被 Occupy 时，其占位符被删除。协议会移动其他 Reserved 条目（例如，将满足 $V_B + \eta \ge V_{buf}$ 的最早 Reserved 条目 B 移动到已删除的条目 A 的位置），以确保 Ready 状态的条目尽可能早地被训练。
-   **与高级技术的兼容性**：该协议天然支持 Partial Rollout、Rollout Migration、Group Sampling（按组进行 Reserve/Occupy，组 $V_{traj}$ 为组内所有轨迹的最小 $V_{traj}$），以及 Redundant Rollout 和 Filtering（可以通过扩展缓冲区条目或每组轨迹数量来实现冗余，并选择性地丢弃轨迹）。

**2. 缓解数据倾斜性：Rollout Service 架构与策略**

StaleFlow 通过重新设计的 Rollout Service 架构和一套协调策略来缓解数据倾斜。

-   **Rollout Service 架构**：
    -   **TS（Trajectory Server）**：存储所有用于 Rollout 的轨迹，作为数据集和 Rollout 实例之间的中介。其容量为 $(\eta+1) \times \text{batch_size}$。它不断从数据集中采样 Prompt，并将其作为初始轨迹排队。当轨迹被中断时，它会被返回到 TS 以便重新路由。
    -   **PS（Parameter Server）**：存储最新的模型参数，部署在 CPU 资源上，在训练工作器和 Rollout 实例之间提供同步接口。训练完成后，训练工作器通过 **Push** 操作将更新的参数立即推送到 PS。Rollout Coordinator 指示 Rollout 实例根据需要从 PS **Pull** 参数。PS 使用读写锁机制确保正确性（Push 为排他写，Pull 为共享读）。
    -   **Rollout Coordinator**：作为中央控制平面，周期性地捕获系统**快照（snapshot）**，应用协调策略，并发出 Rollout 命令。
-   **快照-命令循环（Snapshot-Command Cycle）**：
    -   Coordinator 周期性捕获每个 Rollout 实例的快照 $S$，包含其 **KV Cache** 使用情况、正在运行的轨迹（**run_trajs**）、等待队列中的轨迹（**wait_trajs**）、已完成轨迹（**complete_trajs**）和当前模型版本（**inst_version**）。
    -   **推测状态（Speculative State, $P$）**：为了解决决策和系统状态变化之间的时间耦合问题，StaleFlow 引入 $P$ 来表示命令执行后的预期系统状态。每次发出命令（**Pull**、**Route**、**Interrupt**、**Abort**）后，都会更新 $P$。新的快照 $S_{t+1}$ 只有在与 $P$ 表示的预期状态一致时才被接受，确保 Coordinator 基于最新有效信息做出决策。
    -   **并发命令执行**：命令可以并发执行，StaleFlow 通过等待或挂起数据来处理命令之间的依赖关系。
-   **Rollout 协调策略**：StaleFlow 实现了一套陈旧性感知、吞吐量导向的策略：
    -   **成本模型（Cost Model）**：用于估计路由决策导致的吞吐量变化。实例 $i$ 的生成吞吐量 $T_i(S)$ 被建模为：
        $$T_i(S) = \frac{|\text{run_trajs}|}{k_1 \times \text{kv_cache} + \max(k_2, k_3 \times |\text{run_trajs}|) + k_4}$$
        其中 $k_1$、$k_2$、$k_3$、$k_4$ 是通过离线分析得到的常数系数。**attention** 操作的延迟与 **KV Cache** 大小呈线性关系（$k_1 \times \text{kv_cache}$），矩阵乘法延迟取决于运行轨迹数量（$\max(k_2, k_3 \times |\text{run_trajs}|)$），$k_4$ 是常数开销。该模型用于计算**边际吞吐量增益（marginal throughput gain）** $\Delta T_i$。
    -   **路由策略（Routing Strategy）**：
        1.  **多级队列（Multi-Level Queue, MLQ）**：TS 中的轨迹按 $V_{traj}$ 升序排序，优先处理陈旧性更高的轨迹。
        2.  **候选实例识别**：根据 Staleness Manager 规则和 Check Routable 算法（Algorithm 2），识别可路由轨迹的实例。
        3.  **瀑布模型（Waterfall Model）**：从最高优先级的实例组开始，选择具有最大估计边际吞吐量增益的实例。如果该增益超过阈值 $\mu \times \Delta T_{ideal}$（$\Delta T_{ideal}$ 为理想空闲实例的增益），则路由轨迹；否则，尝试下一个组，若无实例达标则暂缓路由。
    -   **同步策略（Synchronization Strategy）**：
        1.  **选择性同步**：仅当实例的模型版本落后于 PS 且在当前版本下无法接受更多轨迹时，才考虑同步。
        2.  **试探性更新**：通过模拟路由来评估更新模型版本是否会带来新的轨迹路由机会，若会，则进行同步。
    -   **迁移策略（Migration Strategy）**：
        1.  **处理过多等待轨迹**：若实例的 **wait_trajs** 数量超过阈值 $\phi_{wait}$，则中断超额轨迹并返回 TS。
        2.  **处理吞吐量不平衡**：若最高和最低吞吐量实例之间的差距超过 $\phi_{throughput}$，则中断最高吞吐量实例上的所有轨迹并返回 TS 进行再分配。

**评估与结果**
StaleFlow 在 128卡H20集群上进行评估，使用了 Qwen 系列模型（包括 Dense-32b 和 MoE 30b-A3b 模型），并与 VeRL（同步系统）、VeRL-Pipeline（一步陈旧性）以及 VeRL-Async、AReaL、Roll Flash（严格陈旧性控制）等基线系统进行了比较。

-   **端到端吞吐量**：StaleFlow 始终表现出最高的吞吐量，比同步系统 VeRL 高达 2.68 倍（平均 2.01 倍），比一步陈旧性系统 VeRL-Pipeline 高达 1.95 倍（平均 1.52 倍），比最佳严格陈旧性控制系统 VeRL-Async 高达 1.42 倍（平均 1.17 倍）。在更大的陈旧性上限下，StaleFlow 的优势更明显。
-   **收敛性**：在陈旧性上限 $\eta$ 设置在 1 到 3 之间时，StaleFlow 的收敛性与无陈旧性的 VeRL 相当，表明其能在提高吞吐量的同时，有效维持 RL 收敛。过大的 $\eta$（如 10）会导致训练崩溃。
-   **可扩展性**：StaleFlow 在响应长度、批次大小和 GPU 数量等方面的可扩展性均优于或媲美基线，尤其在长响应和大数据量下，其缓解数据倾斜的能力更强。
-   **性能分析**：
    -   **消融研究（Ablation Study）**：逐步引入 StaleFlow 的路由、同步和迁移策略，性能均有提升，三者结合效果最佳。StaleFlow 的策略能够更有效地管理资源和负载，例如，StaleFlow 的同步策略基于实际负载进行实例级同步，显著减少了 KV Cache 重计算开销。
    -   **陈旧性分布**：实验显示，没有轨迹的陈旧性超过设定的上限 $\eta$。大多数缓冲区中的轨迹陈旧性达到了 $\eta$，表明 StaleFlow 充分利用了最大的陈旧性容忍度来榨取系统吞吐量。
    -   **时间开销**：解码占总时间的绝大部分（89.9%），KV Cache 预填充或重填充占 7.9%。StaleFlow 的 Rollout 命令（Pull、Route、Interrupt）引入的开销低于 3%，TS 和 PS 作为中间件引入的额外开销微不足道。


# Injecting RL Skills
Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation

https://arxiv.org/pdf/2601.11258 2026.1.16 北大 Zhang muhan团队

1. 大型语言模型面临知识更新挑战，监督微调（SFT）虽能更新知识但难以提升推理技能，而强化学习（RL）虽能培养技能但成本高昂，本文提出PaST框架，利用**SFT和RL参数更新的近似正交性**，将RL技能模块化注入SFT模型。
2. 该方法通过在**源域中计算RL优化模型与SFT模型的参数差异**来提取领域无关的reasoning “Skill Vector”，随后将其线性注入到在目标域上进行了轻量SFT的模型中，实现了高效的技能适应。
3. 基于Qwen2.5-7b，PaST在SQuAD知识整合任务中超越现有SFT基线9.9分，在LooGLE长文本问答中提升8.0分，并在ToolBench工具使用中平均提高10.3%成功率，展现出其强大的可扩展性和跨域迁移能力。
   
<img width="829" height="709" alt="image" src="https://github.com/user-attachments/assets/ad8a0d9a-6e61-4b2d-81e7-30c583d94e37" />
<img width="407" height="383" alt="image" src="https://github.com/user-attachments/assets/33b18005-d62d-49c7-ade1-7f27713911da" />

大型语言模型 (LLMs) 面临着“知识截止”的挑战，其冻结的参数记忆限制了直接内化新信息的能力。尽管 Supervised Fine-Tuning (SFT) 常用于更新模型知识，但它通常仅更新事实内容，而不能可靠地提升模型运用新信息进行问答或决策的能力。强化学习 (RL) 对于获取推理技能至关重要，然而其高昂的计算成本使其难以进行高效的在线适应。本文实证观察到 SFT 和 RL 所引起的参数更新几乎是正交的。基于此观察，本文提出了 Parametric Skill Transfer (PaST) 框架，该框架支持模块化技能迁移，以实现高效且有效的知识适应。

**核心方法论：Parametric Skill Transfer (PaST)**

PaST 框架通过解耦知识（facts）和技能（skills）在参数空间中的表示，从而实现技能的高效迁移。其核心在于实证发现：**SFT 和 RL 导致的参数更新在参数空间中占据几乎正交的子空间**。这意味着通过 SFT 获得的知识和通过 RL 学习到的操作技能是相对独立的，可以单独优化和组合。
<img width="840" height="322" alt="image" src="https://github.com/user-attachments/assets/fd86a7b5-c3d1-40ad-87ca-3f2071296768" />

<img width="830" height="538" alt="image" src="https://github.com/user-attachments/assets/799e5cfd-b98a-4830-8672-8cf5e42b7611" />

**实验验证**：
<img width="842" height="738" alt="image" src="https://github.com/user-attachments/assets/99cf083f-ecb7-4e59-82c3-dcf8441b27e7" />
<img width="417" height="244" alt="image" src="https://github.com/user-attachments/assets/d092ee8f-d8ed-4e2e-8451-320013c89343" />
<img width="412" height="249" alt="image" src="https://github.com/user-attachments/assets/f90fc485-694b-4ee0-b5dd-dc41bc26f1a8" />

1.  **知识整合 QA (SQuAD)**：
    *   在闭卷 (closed-book) SQuAD 任务上，PaST 显著优于包括最先进的 SEAL (self-adapting baseline) 和 GPT-4.1 在内的基线模型。
    *   PaST (50x2) 达到 56.9% 的准确率，相比“Train on Passage + Synthetic”基线 (+17.2%) 和 SEAL (+9.9%) 有大幅提升。
    *   结果在单通道更新、Continual Pretraining (CPT, n=200 和 n=2067) 等不同机制下均保持优势，表明其鲁棒性和可扩展性。
    *   基础模型使用 Qwen2.5-7B，RL 采用 GRPO，奖励评估使用 GPT-4.1。

2.  **长上下文 QA (LooGLE)**：
    *   在处理超过 21k token 的长文档 LooGLE 数据集上，PaST 再次展示了其有效性。
    *   PaST 相比标准 Target SFT 基线提升了 8.0% 的绝对准确率 (从 30.1% 提升到 38.1%)。
    *   Source Set 使用 LooGLE 的最后 10 个文档，Evaluation Set 使用前 50 个文档。迭代技能获取进行 2 轮。
    *   SFT 阶段采用两阶段课程：(1) 上下文记忆 (通过多任务训练，如文本建模、扩展和压缩)；(2) 合成 QA 训练。
    *   基础模型使用 Qwen2.5-7B-Instruct，RL 采用 GRPO。

3.  **智能体工具使用 (ToolBench)**：
    *   在闭卷执行 (Closed-Book Execution) 设置下进行评估，模型仅提供 API 名称，需利用内部记忆回忆 API 模式并执行。
    *   源领域 (Source Domain) 选定为 Movies 类别，目标领域 (Target Domains) 包含 20 个 RL 训练中从未见过的类别。
    *   PaST 将平均成功率从 21.9% 提升到 32.2%，平均提升 10.3 个点。在某些领域甚至实现了从 0% 到 16.7% 的零样本激活。
    *   SFT 阶段用于建立 API 名称与功能的映射 (使用原始 API 模式、自然语言转录和双向 QA 对)，并对齐 ReAct 规范。
    *   RL 采用 PPO (基于 Search-R1 框架)，环境模拟器为 GPT-4o-mini，奖励信号是格式奖励、执行奖励和 GPT-4.1 判断的解决方案奖励的复合。
    *   基础模型使用 Qwen2.5-7B-Instruct。

**消融研究**：

1.  **迭代技能精炼的影响**：迭代策略在 SQuAD 和 LooGLE 任务上始终优于单轮训练 (即使总数据量相同)，证明其能促使技能向量捕获内容不变的执行逻辑，避免过拟合。
2.  **迁移策略的影响**：PaST 采用的“Post-hoc Composition”策略 ($ \theta_{final} = \theta_{sft}^T + \lambda \cdot v_{skill} $) 显著优于其他替代方案：
    *   “Sequential Fine-Tuning” ($\theta_{rl}^S$ 直接在目标文档上微调)：性能甚至略低于标准 SFT，可能因优化冲突破坏了 RL 学习到的推理电路。
    *   “Pre-Injection” ($v_{skill}$ 在目标 SFT 前注入 $\theta_{base}$)：性能中等，可能因为后续 SFT 会使权重流形发生偏移，导致预注入的技能错位。
    *   这强调了 PaST 先通过 SFT 锚定声明性知识，再嫁接执行逻辑的合理性。

**局限性**：
本文承认仍存在局限性，包括实验领域的多样性、$\lambda$ 系数的静态设定以及模型架构泛化性等，有待未来工作进一步探索。

## LithOS
LithOS: An Operating System for Efficient Machine Learning on GPUs

https://dl.acm.org/doi/pdf/10.1145/3731569.3764818 CMU,Meta. SOSP25

1. 💡 LithOS 是一种针对GPU的操作系统，旨在通过引入**细粒度资源管理和调度机制**，透明地提高机器学习（ML）工作负载的效率和GPU利用率。
2. ⚙️ 为此，它引入了创新的**TPC调度器、透明的内核原子化器**（Kernel Atomizer）、动态硬件资源调配和精细化电源管理等核心机制。
3. 🚀 扩展了Libsmctrl支持Hopper，但测试是在A100上，LithOS**显著降低了尾延迟**（与NVIDIA MPS相比，**推理堆叠可减少13倍**），提高了聚合吞吐量，并实现了显著的GPU容量和能源节约。
   
<img width="399" height="269" alt="image" src="https://github.com/user-attachments/assets/a9dab67e-837c-4d71-8588-802f6d2b3a21" />
<img width="794" height="231" alt="image" src="https://github.com/user-attachments/assets/fd500ff4-cb3e-4499-9f84-0b0a97d85b04" />
<img width="493" height="234" alt="image" src="https://github.com/user-attachments/assets/2adf419b-0275-43a9-b6ae-284eecc72534" />
<img width="1021" height="206" alt="image" src="https://github.com/user-attachments/assets/7fc3812f-697c-4901-91d8-3f97053c152e" />
<img width="592" height="597" alt="image" src="https://github.com/user-attachments/assets/a2e759f9-64d5-4c0a-83ff-759694f4c988" />

旨在解决数据中心GPU利用率低下且现有解决方案效率不足的问题。尽管机器学习 (ML) 的快速发展使得GPU在数据中心中不可或缺，但高利用率和多样化模型需求之间的平衡仍是根本性挑战。现有系统，如NVIDIA的Multi-Process Service (MPS) 和Multi-Instance GPU (MIG)，以及其他先进研究，要么颗粒度过粗，导致资源浪费和队头阻塞 (HoL blocking)，要么缺乏透明度，无法与现有ML软件栈无缝集成。为应对这些挑战，论文提出了LithOS，一个面向GPU的操作系统方法，旨在通过细粒度资源管理透明地提升利用率、能效和隔离性。

LithOS的核心思想是将GPU的调度控制从专有驱动和硬件转移到操作系统层，实现对GPU资源的细粒度、透明管理。它在GPU的Texture Processing Clusters (TPCs) 层面进行调度，而非传统的更粗粒度的Graphics Processing Clusters (GPCs) 或整个GPU。LithOS引入了以下创新抽象和机制：

1.  **TPC Scheduler**: 这是一个新颖的细粒度调度器，能够异步决定每个工作单元的计算单元分配和提交时间。它**在单个TPC的粒度上提供精确控制和强隔离**。调度器通过在线 **Kernel Latency Predictor** (核延迟预测器) 做出高效调度决策，并融入了 TPC Stealing 技术以提高GPU利用率。当某个TPC空闲时，其资源可以**动态地**“借用”给其他任务，从而减少浪费。为了避免优先级反转造成的HoL blocking，调度器会维护每个TPC的计时器，并结合原子化机制，确保高优先级任务能及时获得资源。

2.  **Kernel Atomizer**: LithOS的核心组件之一，它能够透明地将GPU内核 (kernel) 划分为**更小的可调度原子** (atoms)，每个原子包含部分线程块 (thread blocks)。这项功能**无需访问应用程序源代码或PTX代码**，确保了与ML软件栈的完全透明性。通过将长时间运行的内核（例如，持续数毫秒的内核）分割为持续时间较短（例如，250-500 $\mu$s）的原子，Kernel Atomizer 显著减少了HoL blocking，并允许在执行期间动态地重新配置TPC资源。其实现方式是通过**拦截 CUDA Driver API，修改用于启动内核的 Queue MetaData (QMD) 结构**，将原始内核的入口点替换为一个“Prelude”内核。Prelude 内核会根据传入的**原子元数据 (AtomMetadata) 检查 `block_idx` 是否在当前原子的指定范围内**，若在则调用原始内核的入口点，否则退出。这使得LithOS可以在线程块粒度而非内核粒度上调度工作。

3.  **Hardware Right-sizing**: 基于LithOS在TPC层面的细粒度调度能力，它引入了一种**动态的硬件资源优化机制**。该机制通过轻量级模型来确定每个内核及其原子所需的最小TPC资源量。模型通过拟合内核在全部TPCs和单个TPC上运行的延迟数据，得出一个形式为 $l = \frac{m}{t} + b$ 的曲线，其中 $l$ 是预测延迟，$t$ 是TPC数量，$m$ 和 $b$ 是常数。这个模型与Amdahl定律一致，可以捕捉不同内核的扩展行为。对于模型无法准确预测的短时运行内核，LithOS会根据线程块的占用率进行过滤。用户和管理员可以通过一个 `latency slip parameter` $k$ 来指定可接受的性能损失上限（例如，10%），LithOS会据此调整TPC分配，实现显著的容量节约。

4.  **Transparent Power Management**: LithOS通过细粒度的动态电压频率调节 (DVFS) 来实现透明的电源管理。与 Hardware Right-sizing 类似，它使用一个序列化的内核频率扩展模型来**指导DVFS决策**。每个内核被赋予一个权重 $w$ (占总运行时间的比例) 和一个敏感度 $s$。系统计算所有内核的聚合敏感度 $S = \sum w \cdot s$，并根据预设的 `latency slip parameter` $k$ 来确定最终频率 $f_{final} = \frac{f_{max}}{1 + k/S}$。**计算密集型内核的敏感度会引导频率接近最大值**，而**内存密集型内核则会根据其权重将频率调至较低水平**。**由于频率切换延迟较高，LithOS采取保守策略**，通过学习期来避免不必要的切换。

5.  **Online Latency Prediction**: 延迟预测模块学习内核的执行时间，为LithOS的所有组件提供支持。它提高了TPC Stealing 的准确性（通过估算未完成任务的持续时间），指导 Kernel Atomizer 的原子分割数量，并为 Hardware Right-sizing 和 DVFS 提供计算加速比所需的延迟数据。该模块无需大量的离线性能分析。为了准确预测，模块会根据CUDA事件记录内核延迟，并根据分配的TPC数量、GPU频率和原子化粒度进行调整。它还通过将内核启动与批处理中的 `ordinal index` 相关联，识别操作符节点，从而处理同一内核函数因输入参数变化而导致执行时间不同的情况。

**实施细节**:
LithOS **原型使用 Rust 语言实现，代码量约 5000 行**。它通过在 CUDA Driver API 层面进行拦截来实现对应用程序的完全透明性，从而支持各种未经修改的ML框架和库（如PyTorch, TensorFlow, JAX, TensorRT, cuDNN）。LithOS 重新实现了 `libsmctrl` 的功能，并通过逆向工程 Queue MetaData (QMD) 结构来识别TPC映射，实现动态TPC分配。对于 NVIDIA Hopper 架构上引入的 Thread Block Clusters，LithOS 也进行了逆向工程以确保原子化操作与其兼容。为了处理具有跨块同步或持久性内核等特殊情况，LithOS 会选择禁用原子化和TPC Stealing。

**评估结果**:
论文在 NVIDIA A100 GPU 上对LithOS进行了广泛评估，并与 NVIDIA 的 Time slicing, MPS, Priority, MIG 以及现有先进研究 (TGS, REEF, Orion) 进行了比较。

*   **仅推理多租户 (Inference-only Multitenancy)**：在包含两个高优先级 (HP) 推理应用和一个尽力而为 (BE) 应用的场景中，LithOS 实现了 100% 的 SLO (服务等级目标) 达成率和 1 倍的吞吐量（相较于单一应用独占设备），优于 MPS (45% SLO, 1.11x 吞吐量)。LithOS 在HP应用的 Goodput (有效吞吐量) 上领先，同时支持显著的BE吞吐量。在P99延迟方面，LithOS比MPS低13倍，比Orion低4倍，比TGS低1.2倍。

*   **混合推理/训练多租户 (Hybrid Inference/Training Multitenancy)**：在一个高优先级推理应用和一个尽力而为训练应用堆叠的场景中，LithOS 将P99尾延迟维持在理想值的20%以内。平均而言，它比REEF低2.34倍，比TGS低1.18倍，比原生MPS解决方案低4.7倍。LithOS还将训练吞吐量平均提高了34倍，总吞吐量比TGS提高了1.35倍。

*   **Kernel-SM Right-Sizing**：使用 `latency slip parameter` 为1.1时，LithOS实现了高达51%的GPU容量节约，平均节约率为26%。P99延迟和吞吐量的性能成本平均仅为4%，这表明其建模的准确性很高 ($R^2$ 值在 0.92 到 0.99 之间)。

*   **Kernel-Dependent DVFS**：在 `latency slip parameter` 为1.1时，LithOS实现了高达46%的GPU能耗节约，平均节约率为26%。P99延迟的平均增加仅为7%，这表明LithOS的DVFS策略在保证性能的同时显著节约了能源。

*   **消融研究 (Ablation Studies)**：
    *   单独启用 TPC Scheduler 可将HP尾延迟改善至理想值的1.38倍。
    *   Kernel Atomization 进一步将尾延迟降低至平均1.19倍（最高1.55倍），相较于REEF分别有6.5倍和3.9倍的提升，尤其在处理大批次训练和长序列推理时效果显著。
    *   延迟预测模块的误预测率很低（HP工作负载为0.38%-0.9%），P99误差很小（31-49 $\mu$s），证实了其在实践中提供足够性能隔离的能力。

*   **开销 (Overheads)**：LithOS的拦截和控制逻辑引入的开销很小，相较于原生NVIDIA驱动，仅增加了4%的开销，原子化本身增加了不到1%。

**讨论与未来工作**:
LithOS 专注于计算和功耗管理，但论文指出其原理可扩展到其他GPU资源，如内存、带宽和PCIe。论文也强调，未来的驱动和硬件支持（如内核到SM的直接分配、更细粒度（亚毫秒级）的DVFS、每个SM的功耗控制等）将进一步释放LithOS的潜力。核心教训是，高效的GPU多租户需要空间和时间上的细粒度分区。CUDA Driver API 作为稳定的拦截点，使得LithOS轻量、可移植且易于重定向。


## Hardware Compute Partitioning
Hardware Compute Partitioning on NVIDIA GPUs for Composable Systems

https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf

https://www.cs.unc.edu/~jbakita/ecrts25-ae/

<img width="515" height="220" alt="image" src="https://github.com/user-attachments/assets/8be1b2a0-16e0-49c3-939d-36db6c213ef1" />
<img width="743" height="380" alt="image" src="https://github.com/user-attachments/assets/de666d10-1b4c-4d6f-8952-960f10b2cf4e" />
<img width="740" height="540" alt="image" src="https://github.com/user-attachments/assets/d6d61a07-699a-467c-9c87-277cb5e3d0b4" />
<img width="737" height="469" alt="image" src="https://github.com/user-attachments/assets/8740517a-05c3-49a7-acd8-c3be6eea88c0" />

1. 🚀 开发了 `nvtaskset`，这是一种用于 NVIDIA GPU**空分共享和控制机制**，旨在无需修改任务、驱动程序或硬件的情况下，使多任务高效且可预测地运行。
2. 💡 `nvtaskset` 通过**结合 `libsmctrl` 的硬件强制分区能力和 NVIDIA MPS** 的多任务并发执行特性，实现了比 NVIDIA MPS 更强的分区强制性和更细粒度的控制。
3. 📊 评估结果表明，`nvtaskset` 具有**亚微秒级的开销**，并且在某些情况下，其分区强制能力甚至优于 NVIDIA MiG，同时揭示了 NVIDIA MiG 存在固有的性能损耗问题。但没有解决故障隔离（MPS）。
- recent: https://dl.acm.org/doi/pdf/10.1145/3731569.3764818
  
该论文提出了 `nvtaskset`，一个用于 NVIDIA GPU 的系统级空间计算分区机制，旨在解决嵌入式、安全关键系统中 GPU 任务调度面临的效率与时间可预测性之间的冲突。在这些系统中，单个 GPU 需要被多个任务共享。现有调度方法往往牺牲截止期满足能力或效率。`nvtaskset` 通过在 GPU 计算核心上实现空间分区，使得任务能够在不影响时间可预测性的前提下高效执行。该工具支持可组合系统，无需修改任务、驱动或硬件。在评估中，`nvtaskset` 展示了亚微秒级的开销、更强的分区强制能力和更细粒度的分区，优于 NVIDIA 的 `Multi-Process Service (MPS)` 或 `Multi-instance GPU (MiG)` 功能。

**核心问题与背景：**

随着深度神经网络 (DNN) 等 AI 任务在 GPU 上运行变得普遍，它们在自动驾驶等安全关键系统中的应用对实时响应提出了严格要求。然而，在多个 GPU 任务之间进行调度存在挑战：
1.  **竞争共享 (Competitive sharing)**：任务并发运行，争夺资源，效率高但时间可预测性差。
2.  **互斥 (Mutual exclusion)**：一次只运行一个任务，时间可预测但效率低。
这导致嵌入式系统设计者面临两难，尤其对于无法保证截止期满足的安全关键系统。此外，任务可能由不同团队开发，且不总是可修改，增加了调度系统的负担。

作者提出通过 GPU 计算核心的空间分区来解决此问题。空间分区允许任务在互斥的核心集上并发执行，最大限度地减少共享资源干扰，从而兼顾效率和时间可预测性。

**GPU 架构与调度：**

NVIDIA GPU 的典型架构（如 Ada-generation AD102）包括：
*   **Compute/Graphics Engine (计算/图形引擎)**：主要处理单元。
*   **General Processing Clusters (GPCs)**：计算引擎的子分区，每个 GPC 独立连接到 DRAM 控制器和 L2 缓存。
*   **Thread Processing Clusters (TPCs)**：每个 GPC 包含多个 TPC，每个 TPC 包含两个 `Streaming Multiprocessors (SMs)`。
*   **Streaming Multiprocessors (SMs)**：每个 SM 包含数十个 CUDA 核心和 L1 缓存。

GPU 调度管道如下：CPU 任务将核函数启动命令 (`TMDs` - Task Metadata Descriptors) 插入 CUDA `streams`，这些 `streams` 映射到 GPU `channels`。`runlist` 是核心仲裁器，决定哪个 GPU `channel` 或 `channel` 组可以访问计算/图形引擎。`runlist` 由 `time-slice groups (TSGs)` 组成，TSGs 包含来自同一上下文的 `channels`。TSGs 默认以工作守恒、抢占式的轮询方式执行，每个 TSG 有一个关联的时间片。这种调度方式导致 GPU 在一次只激活一个上下文时可能出现资源利用不足。

**空间分区的理想特性：**

一个理想的空间分区机制应具备以下特性：
*   **Portable (可移植)**：适用于广泛的 GPU 型号。
*   **Logically Isolated (逻辑隔离)**：保持任务间的虚拟地址空间隔离和独立异常处理。
*   **Transparent (透明)**：无需修改任务。
*   **Low-overhead (低开销)**：关键路径操作（如核函数启动）开销可忽略。
*   **Hardware-enforced (硬件强制)**：由硬件强制执行分区，以保护免受恶意或行为不当任务的影响。
*   **Dynamic (动态)**：无需重启任务即可重新配置分区。
*   **Granular (细粒度)**：分区以细粒度单位定义，如 TPC。

**现有解决方案及缺陷：**

1.  **学术界软件方案**：
    *   例如 [40, 16] 等：通过任务合作性地放弃未分配的计算核心来实现，易受行为不当任务的影响，无法强制执行分区。
    *   `libsmctrl` (作者先前工作)：通过修改 `TMD` 的 TPC 掩码来强制执行分区，但需要任务修改和共享地址空间，牺牲了逻辑隔离和透明性。

2.  **NVIDIA 官方方案**：
    *   **Multi-instance GPU (MiG)**：将 GPU 分割成静态、固定大小的分区，硬件强制执行。但分区选项有限，粒度粗（最小分区 14 个 SMs），且仅在数据中心 GPU 上可用（不可移植）。论文发现 MiG 存在 6-15% 的核心利用率损失，原因是为模拟统一 GPC 而禁用 TPC。
    *   **Multi-Process Service (MPS)**：可用于所有近期 NVIDIA GPU (Volta+)，允许计算/图形引擎并发执行多任务。
        *   **MPS 运作机制揭示 (核心技术点)**：
            *   MPS 采用客户端-服务器模式，CUDA 任务是 MPS 服务器的客户端。服务器作为 GPU 中介。
            *   **Runlist 修改**：MPS 客户端不再拥有独立的 TSG。多个 MPS 客户端共享 MPS 服务器的单个 TSG。所有 MPS 客户端的任务流并发执行，如同在一个程序中。这导致共享 TSG 的客户端可用 GPU 时间减少。
            *   **虚拟地址空间隔离**：MPS 通过为每个客户端分配唯一的 `Subcontext ID` 和独立的 `page table` 来保持虚拟地址空间隔离。`Subcontext ID` 随着命令传递给计算/图形引擎，用于访问和维护每个 `subcontext` 的 `page table` 状态。
            *   **MPS 缺陷（核心发现）**：
                *   **缺陷 1**：MPS 客户端共享每个服务器上的并发内核数量限制（任务槽耗尽），尽管专利声称是每个 `subcontext` 独立。
                *   **缺陷 2**：一个 MPS 客户端中的崩溃（如越界内存访问）可能导致所有 MPS 客户端崩溃，缺乏完全的故障隔离。
                *   **缺陷 3**：MPS 客户端不支持 GPU 上的内核启动 (CUDA Dynamic Parallelism, CDP)。
                *   **缺陷 4**：MPS 客户端默认支持的 CUDA `streams` 数量较少（默认 2 个，非 MPS 任务默认 8 个），可能导致隐式同步和调度行为变化。
                *   **缺陷 5**：MPS 客户端接收伪造的 SM ID (`%smid` 寄存器返回不一致的值)，影响 GPU 研究。
                *   **缺陷 6**：MPS 分区不绑定到特定 SMs。MPS 通过 "Execution Resource Provisioning" 功能（基于 `credits` 机制）限制任务可并发占用的 TPC 数量，但不能保证分配到哪些 TPC。
                *   **缺陷 7**：MPS 的硬件实现可能将两个任务分配到同一组 SMs，导致其他 SMs 空闲。当分配总和超过 100% 时，任务可能共享 TPC，但这种共享可能会持续，即使总分配容量恢复到 100% 也不会自动迁移到空闲 TPC，导致 GPU 资源浪费。
                *   **缺陷 8**：MPS 客户端的分区大小是静态的，创建后无法动态更改。
            *   **开销**：MPS 在核函数启动关键路径上没有额外开销。启动开销因为 MPS 服务器延迟初始化而有额外成本，但可以通过预加载规避。

**nvtaskset 的设计与实现：**

`nvtaskset` 结合了 MPS 的逻辑隔离和透明性与 `libsmctrl` 的分区能力。它类似 Linux 的 `taskset` 工具，允许用户通过命令行指定任务可使用的 GPU 核心（TPC 或 GPC）。

**`nvtaskset` 实现细节 (核心技术点)**：
1.  **基于 MPS 的并发执行**：`nvtaskset` 将通过它启动的所有任务与一个自动启动的 MPS 服务器关联，实现任务并发。为缓解 MPS 的“缺陷 4”，`nvtaskset` 将每个 CUDA 任务的 `channels` 数量配置为 8。
2.  **基于 `libsmctrl` 的分区强制**：`nvtaskset` 使用 `libsmctrl` 的机制，通过修改提交到 GPU 的 `TMD` 中的 TPC 掩码来实现分区。这使得 `nvtaskset` 能够像 `libsmctrl` 一样强制执行分区，且能指定具体 TPC，避免了 MPS 的“缺陷 6”和“缺陷 7”。
3.  **对未修改任务的支持 (透明性)**：`nvtaskset` 通过共享库拦截技术实现透明性。它将自己编译成 `libcuda.so.1`，利用加载器行为使其在 CUDA 库之前加载。其内部的加载时构造函数会加载真正的 CUDA 库，并注册 `TMD` 拦截回调函数。当任务执行核函数启动时，此回调函数会被触发，应用 TPC 掩码，无需任务修改。
4.  **动态可变分区**：`nvtaskset` 通过共享内存区域暴露每个 CUDA 任务的当前分区设置。`nvtaskset -p` 等命令可修改此设置，`nvtaskset` 的回调函数会自动检测并应用于后续的核函数启动，解决了 MPS 的“缺陷 8”。
5.  **GPC 粒度分区**：`nvtaskset` 支持按 GPC 指定分区，并将其内部转换为 TPC 集合。它修正了 `libsmctrl` 中关于 SM ID 到 GPC 映射的假设（SM ID 到 GPC 映射不是线性的，而是由 NVIDIA 驱动配置的条带状结构），确保分区与 GPC 边界对齐，这对于 NVIDIA 的 `Thread Block Groups` 功能（仅在访问完整 GPC 时有效）至关重要。

**`nvtaskset` 局限性：**

*   仅兼容 CUDA 任务。
*   不支持多 GPU 任务。
*   无法影响已启动内核的分区变化。
*   仍然受 MPS 的“缺陷 1-3”影响（内核并发限制、故障隔离不完全、不支持 CDP）。
*   每个 MPS 服务器最多支持 15 个客户端。
*   默认情况下受 MPS“缺陷 5”影响（伪造 SM ID），但可通过特定配置使其看似连续。

**评估：**

评估在 NVIDIA A100 GPU (支持 MiG, MPS, nvtaskset) 上进行，使用 57/43 分区比例（MiG 限制）。

1.  **分区开销**：
    *   **观察 1**：所有分区机制对启动或启动开销均无显著增加。
    *   **观察 2**：只有 MiG 降低了启动开销，可能因为它静态绑定了硬件调度管道到每个分区的 TPC，减少了设置和考虑的 TPC 数量。
    *   **观察 3**：MPS 和 `nvtaskset` 启动开销最低，因为作为 MPS 客户端，任务只需初始化 `channels` 和 `subcontexts`，MPS 服务器提供父上下文和 TSG，显著减少了开销。

2.  **分区强制能力**：
    *   测试方法：测量 6144x6144 矩阵乘法 (`MM6144`) 在 57% 分区内的执行时间，同时干扰任务（内存密集型 `random_walk` 和计算密集型 `mandelbrot`）在剩余 43% 分区执行。
    *   **观察 4**：MPS 的分区机制在可预测性和效率上表现更差，即使正确使用。对于内存密集型竞争，MPS 无法阻止内存争用，仅限制计算资源。
    *   **观察 5**：`nvtaskset` 能够提供接近 MiG 的分区强制能力，而无需硬件修改。对于内存密集型工作，`nvtaskset` 表现优于 MPS，接近 MiG。对于计算密集型工作，`nvtaskset` 在平均和最坏情况执行时间上甚至优于 MiG。这表明 `nvtaskset` 对 L0、L1 和 TLB 缓存的隐式分区（通过与 GPC 对齐）效果显著。

3.  **分区粒度**：
    *   测试方法：测量 8192x8192 矩阵乘法 (`MM8192`) 在各种可能分区大小下的执行时间。
    *   **观察 6**：`nvtaskset` 是粒度最细的分区机制。MPS 和 `nvtaskset` 都能达到每 TPC 粒度。但 `nvtaskset` 能指定具体 TPC，而 MPS 只能指定百分比。对于 54 TPC 的 GPU，`nvtaskset` 支持 $2^{54}$ 种分区设置，MPS 支持 54 种，MiG 仅支持 5 种。
    *   **观察 7**：MiG 无法访问 A100 GPU 核心的 9%（5 个 TPC）。A100 上 MiG 最大分区只包含 49 个 TPC，而 MPS 或 `nvtaskset` 能访问 54 个。这是由于 MiG 配置为 7 TPC 的倍数，而 54 不能被 7 整除，导致剩余的 5 个 TPC 被浪费。在 H100 GPU 上，损失甚至高达 6-15%。


## BitDecoding
BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

https://arxiv.org/abs/2503.18773 2025.5/2026.1.5 (last update, as HPCA26) 微软亚研院

https://github.com/OpenBitSys/BitDecoding 

1. 💡 BitDecoding 通过**协同利用 GPU 的 CUDA Cores 和 Tensor Cores**，解决了长上下文 LLMs 中低比特 KV Cache 解码效率低下，尤其是 Tensor Cores 利用率不足的问题。
2. ⚙️ 该系统通过生成兼容 Tensor Cores 的数据布局、引入 warp 级解量化并行以及采用支持混合精度执行的软件流水线解量化内核，优化了低比特 KV Cache 的处理。
3. 🚀 实验结果表明，BitDecoding 在 **Blackwell、Hopper 和 Ampere 等 GPU** 上实现了显著的性能提升，相较于 FP16 FlashDecoding-v2 平均提速 7.5 倍（Blackwell 上最高达 8.6 倍），并比现有先进方法 **QServe 提速达 4.3 倍**，同时大幅降低了端到端解码延迟。
<img width="942" height="263" alt="image" src="https://github.com/user-attachments/assets/90c916bf-c706-4698-ab56-187ba6b8db32" />
<img width="515" height="287" alt="image" src="https://github.com/user-attachments/assets/4c5c37b3-39ab-4b96-87c9-75c2b9157cb2" />
<img width="1031" height="275" alt="image" src="https://github.com/user-attachments/assets/f3cc5002-5579-4610-aa94-318ab5847b02" />
<img width="566" height="289" alt="image" src="https://github.com/user-attachments/assets/8bf38720-2fa6-44d8-a111-0227db27bb76" />
<img width="532" height="309" alt="image" src="https://github.com/user-attachments/assets/4375141c-ee22-40b1-a51a-849dd7c15fc2" />
<img width="577" height="396" alt="image" src="https://github.com/user-attachments/assets/ea0cbda0-3aef-405c-bd62-7bbff75eaa43" />
<img width="581" height="270" alt="image" src="https://github.com/user-attachments/assets/a3ab9e66-01ce-4513-92db-a45bae0dd234" />

当前LLMs在处理长上下文时，KV缓存会急剧增长，导致显存和带宽瓶颈。虽然低位KV缓存量化（如4比特或2比特）能显著减少显存占用，但现有系统在解码时效率低下，主要依赖CUDA核心，未能充分利用GPU上的主要计算资源——Tensor Core。BitDecoding是首个通过协同利用CUDA核心和Tensor Core来高效解码低位KV缓存的推理系统。

核心问题在于，KV缓存在自回归解码过程中是动态生成的，需要在线量化、打包和反量化。这与静态权重的低位矩阵乘法（mpGEMM）不同，后者可以离线预处理。现有的mpGEMM内核无法直接应用于KV缓存。BitDecoding的核心洞察是利用Tensor Core执行密集的矩阵乘法，同时高效利用CUDA核心进行KV缓存的反量化。

**BitDecoding解决以下关键挑战：**

1.  **Tensor Core低位布局不匹配：** Tensor Core要求反量化后的低位数据对齐到高精度格式，但动态增长的KV缓存难以满足。不同的指令和GPU代次有不同的fragment布局，低精度位宽进一步加剧了对齐问题。天真地进行`low-bit -> FP16`类型转换效率低下。
2.  **频繁停顿限制Tensor Core利用率：** 现有高性能注意力内核中的warp布局在引入反量化操作后效率下降，因为小warp tile需要顺序处理N维数据，导致反量化频繁停顿warp。Blackwell原生低精度格式的在线量化也会造成类似停顿。
3.  **缺乏通用可扩展的系统优化：** 不同的KV缓存量化方法（如tensor-wise和channel-wise）采用不同的量化粒度，增加了构建统一系统的难度。在线量化和打包引入了非平凡的运行时开销，辅助元数据（scale和zero-point）增加了内存流量。

**BitDecoding的核心方法论：**

BitDecoding的设计围绕优化低位布局和并行化GPU warp展开。

**A. 优化Tensor Core上的低位布局：**

1.  **通过硬件指令诱导优化低位布局：**
    *   核心思想：`ldmatrix`指令加载数据时已按Tensor Core的交错fragment布局组织。如果每个线程在本地进行量化和打包，则生成的低位打包数据会隐式保留FP16的交错布局。反量化时，值已与Tensor Core寄存器对齐，无需全局重塑。
    *   **Residual Kernel（残差内核）：** 融合新生成的FP16 KV张量的计算、量化和打包。它使用`ldmatrix`将高精度KV张量加载到为Tensor Core设计的寄存器结构中，执行矩阵操作（如$QK^T$或$PV$），然后每个线程量化并打包其寄存器中的部分。结果是交错的、布局兼容的低位数据直接写入全局内存，更新低位KV缓存。
    *   **Packing Kernel（打包内核）：** 融合反量化和计算。为保证解包时正确的寄存器布局，它镜像Residual Kernel的指令配置，使用相同的`ldmatrix`变体和相同的`mma`变体及warp-tiling配置。因此，当Packing Kernel通过`ldmatrix`加载打包的低位数据时，解包后的值天生与Tensor Core寄存器对齐，可直接参与矩阵乘法。
2.  **通过残差KV缓存对齐warp以饱和Tensor Core：**
    *   为了确保Tensor Core fragment完全填充，BitDecoding分配一个大小匹配Tensor Core tiling能力的残差缓冲区。
    *   KV缓存被划分为已打包的低位部分$X_{pack}$和半精度残差部分$X_{res}$，其中$X = X_{pack} \cup X_{res}$。
    *   残差块大小$N_r$计算公式为：$N_r = P_n \times W_n \times R$，其中$P_n$是每个warp tile处理的元素数量，$W_n$是N维度上的warp数量，$R = \omega / \beta$是打包比（$\omega$为打包字大小，$\beta$为量化位宽）。这确保了低位KV缓存fragment与Tensor Core操作的warp级别tiling精确对齐。
3.  **重新映射布局以加速反量化：**
    *   尽管与Tensor Core布局兼容，但直接将低位值转换为FP16（如通过`static_cast`）效率低下。
    *   BitDecoding设计了一种基于底层位操作和指令（受[14]启发）的更快反量化映射方法。加载打包数据到寄存器后，在映射到交错Tensor Core布局（遵循75316420模式）之前，将其转换为INT32。这种布局支持使用`lop3`指令进行位操作，高效将INT4/INT2数据转换为FP16，同时与Tensor Core计算模式对齐。
4.  **协调残差和打包内核与配置设置：**
    *   通过统一的指令配置协调Residual Kernel和Packing Kernel。指令配置（包括`ldmatrix`和`mma`变体）根据GPU架构确定。Residual Kernel加载高精度KV，通过Tensor Core计算，融合量化和打包，存入低位KV缓存。Packing Kernel使用相同配置，加载打包数据，高效反量化，并进行Tensor Core计算。

**B. Warp并行化策略：**

为了避免现有warp并行化策略在混合精度注意力中因频繁warp停顿导致的硬件利用率低，BitDecoding设计了新的warp布局。

1.  **增强低精度操作的warp并行性：**
    *   修改warp分区策略。在M维度上限制warp数量$W_m=1$（解码查询长度通常较小），将资源重新分配以增加N维度上的warp数量$W_n$。
    *   通过增加$W_n$，SM warp调度器可以有效地缓解反量化停顿，因为多个warp可以并行执行打包数据的反量化，然后进入基于Tensor Core的矩阵乘法。
2.  **利用内存层次结构进行warp同步：**
    *   由于结果现在分布在不同的寄存器和warp中，原始的寄存器级softmax变得不可行。
    *   BitDecoding利用多级内存层次结构（寄存器和共享内存）实现跨warp归约和softmax计算的同步。
    *   **多warp协作Softmax（Algorithm 1）：**
        *   引入两个额外的共享内存缓冲区：`sTMP`（用于行最大值归约）和`sAcc`（临时存储注意力分数P）。
        *   `sTMP`协助softmax期间的跨warp行最大值归约，首先进行寄存器内的warp内归约，然后通过共享内存进行warp间归约。
        *   `sAcc`临时存储在Tensor Core寄存器中计算的注意力分数P，并通过`ldmatrix`重新加载以确保后续Tensor Core `mma`操作的对齐。
        *   Algorithm 1:
            1.  $S_i = Q_i K_j^T$，其中$S_i \in \mathbb{R}^{T_m \times T_n}$。
            2.  $m_{new_i} = \max(m_i, \text{rowmax}(S_i, \text{sTMP}))$。
            3.  $P_i = \exp(S_i - m_{new_i})$，其中$P_i \in \mathbb{R}^{T_m \times T_n}$。
            4.  $\text{sAcc} = \text{tiled copy r2s}(P_i)$。
            5.  $P'_i = \text{tiled copy s2r}(\text{sAcc})$。
            6.  $O_{new_i} = P'_i V_j + \text{diag}(\exp(m_i - m_{new_i}))O_i$。
    *   由于$W_n$通常较小，复用`sTMP`的共享内存指针用于`sAcc`以最小化内存开销。Hopper Tensor Core支持WGMMA直接访问共享内存，无需显式数据从共享内存移动到寄存器。

**C. 系统实现：**

1.  **查询转换（Query Transformation）：**
    *   现代LLMs采用不同KV共享模式的注意力变体（如MHA、MQA、GQA）。解码时Q_len=1，导致查询张量维度很小，无法充分填充Tensor Core。
    *   BitDecoding将查询张量从$[1, (g_q, h_{kv})]$重塑为$[g_q, h_{kv}]$，形成更大的$Q_{tile}$，并行处理分组查询头，提高warp占用率和吞吐量。
2.  **Residual Kernel（残差内核）：**
    *   根据残差块大小$N_r$分区KV缓存。预填充时，前$N_p = L - (L \pmod{N_r})$个条目被量化并打包。剩余的$res\_len = L \pmod{N_r}$个KV张量存储在半精度残差KV缓存中。
    *   每次解码时，新生成的K、V张量追加到残差缓存中，并用于注意力计算。当残差缓存达到$N_r$时，Residual Kernel将其量化成打包格式。
    *   支持channel-wise和tensor-wise量化。
    *   **优化warp级指令归约：** 半精度KV数据在寄存器中以Tensor Core fragments形式存在。BitDecoding使用线程级归约获取局部min/max统计量，然后使用PTX指令`__shfl_xor_sync`在warp内聚合，实现高效warp级归约。当$W_n > 1$时，引入小型共享内存缓冲区协调warp间归约。
    *   计算量化参数后，每个线程在寄存器内进行量化并将低位值打包为INT16格式。scales和zero-points存储在`half2`格式，方便高效内存访问和反量化期间的融合乘加。
3.  **Packing Kernel（打包内核）：**
    *   辅助低位元数据（scale和zero-point）增加了内存流量，反量化仍在CUDA核心上运行。BitDecoding设计了一个细粒度异步流水线：CUDA核心处理反量化，Tensor Core执行矩阵乘法，两者与内存传输重叠。
    *   **异步数据移动优化：** 遵循FlashAttention的块级tiling和策略性重计算，处理共享内存中的Q、K、V tile。引入Kpack_params和Vpack_params的专用共享内存缓冲区，存储`half2`格式的scale和zeros。
    *   所有全局到共享内存的传输都使用`cp.async`指令异步执行。`cp.async.cg`用于Q、Kpack和Vpack，`cp.async.ca`用于Kpack_params和Vpack_params。Hopper架构利用`tma.copy`指令加载数据。
    *   **共享内存到寄存器：** 使用PTX指令`ldmatrix`高效加载Kpack、Vpack和sAcc到寄存器。使用sizzling scheme（`colid = rowid \oplus colid`）消除bank conflict，并重塑Kpack_params和Vpack_params的共享内存布局。
    *   **CUDA Core与Tensor Core重叠的异步流水线：** 实现寄存器级异步流水线，共享内存加载（`ldmatrix`）和反量化（Dequant）与Tensor Core矩阵乘法（`mma`）并发运行。当第i个切片由Tensor Core处理时，第i+1个切片同时从共享内存加载并反量化，维持连续的生产者-消费者流。
4.  **最新架构支持：**
    *   **Hopper加速：** 利用Hopper的Warpgroup Matrix Multiply-Accumulate（`wgmma`）指令。由于`wgmma`要求B矩阵在共享内存中，BitDecoding利用Hopper的STSM PTX指令将反量化后的FP16值高效存储到共享内存中，供`wgmma_SS`操作使用。WGMMA的异步特性使得存储与计算重叠。
    *   **Blackwell原生低精度格式加速：** Blackwell原生支持低精度张量操作，消除了显式反量化需求。BitDecoding直接利用Blackwell的低精度`mma`指令（尤其是支持微缩放格式如mxfp4/nvfp4的指令）在打包的4比特数据上执行GEMM操作。BitDecoding的布局转换策略与硬件强制格式自动对齐，确保与Blackwell原生Tensor流水线的无缝集成。

**D. 评估结果：**

*   **核性能：** BitDecoding在Blackwell（使用原生MXFP4）上达到高达8.6倍、Hopper上8.0倍、Ada上7.5倍的加速，相较于FP16 FlashDecoding-v2。相较于最先进的低位系统QServe，性能提升高达4.3倍。
*   **端到端模型性能：** 在LLaMA-3.1-8B的128K上下文长度单批次解码中，BitDecoding将延迟降低3倍，并实现了比QServe高4倍以上的服务吞吐量。
*   **精度：** 4比特量化仅导致0.2%的精度下降，2比特量化导致2.7%的精度下降，同时实现了显著的性能提升。
*   **开销分析：** 残差KV缓存的内存开销和运行时开销微乎其微。量化和打包开销（特别是在解码阶段）几乎可以忽略不计。反量化开销显著降低（从Atom/QServe的近一半时间降低到BitDecoding的低于15%）。多warp协作Softmax仅引入0.5%的开销。
*   **优化分解：** BitDecoding的性能提升主要来源于其布局设计（诱导Tensor Core兼容布局）、warp并行化策略以及流水线优化。

BitDecoding为高效低位KV缓存解码建立了一个新的系统基础，展示了CUDA核心和Tensor Core如何通过有原则的系统设计协同工作。其布局诱导和warp级协调技术适用于各种注意力变体、量化方案和GPU代次，并可自然扩展到新兴架构。

## Sparse-RL
Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning via Stable Sparse Rollouts

https://arxiv.org/pdf/2601.10079v1 2026.1.16 人大 蚂蚁等
中文解读：https://mp.weixin.qq.com/s/NjEd07gS9iIssWtcTbOHzQ 

1. 🤔 本文提出Sparse-RL框架以解决大型语言模型(LLM)强化学习rollout阶段**KV cache内存占用过高**的问题，并指出直接应用KV压缩会导致策略不匹配和训练崩溃。
2. 💡 为克服这一挑战，Sparse-RL引入了Sparsity-Aware **Rejection Sampling来过滤压缩引起的异常轨迹**，并采用Importance-based Reweighting来**修正off-policy偏差**，从而实现**稳定的稀疏rollout训练**。
3. 🚀 最大Qwen-7b，最长400 steps的实测，对比直接引入SnapKV等，基于Slime实现。Sparse-RL在-40% token，精度接近持平。

<img width="934" height="378" alt="image" src="https://github.com/user-attachments/assets/01e6f275-cb7a-4544-92d0-c974d8a99a2c" />

<img width="836" height="728" alt="image" src="https://github.com/user-attachments/assets/289ade64-fdbe-43c3-9605-b6cc56e3f8c9" />

<img width="1545" height="288" alt="image" src="https://github.com/user-attachments/assets/923f41f3-0b75-4787-a40d-c63e0f06086a" />

<img width="1167" height="321" alt="image" src="https://github.com/user-attachments/assets/10396759-8ccf-4a2c-8631-ec344939d1c7" />

<img width="1526" height="369" alt="image" src="https://github.com/user-attachments/assets/e85b43fa-f75e-4593-b2f9-bd15090bbcdf" />

## dLLM-Serve
Taming the Memory Footprint Crisis: System Design for Production Diffusion LLM Serving

https://arxiv.org/abs/2512.17077 

https://github.com/chosen-ox/dLLM-Serve
1. 🚀 dLLM-Serve 提出了一套针对 Diffusion LLMs (dLLMs) 的高效服务系统，旨在解决其特有的“显存占用危机”，该危机源于单体 **logit 张量和计算密集型“Refresh”阶段与带宽密集型“Reuse”阶段之间的资源振荡**。
2. 🧠 该系统通过 Logit-Aware Activation Budgeting 来**分解瞬时峰值激活内存**、Phase-Multiplexed Scheduler 来交错异构请求阶段，以及 Head-Centric Sparse Attention 来实现物理存储与逻辑稀疏性的解耦。
3. 📈 与现有最强基线相比（fast-dLLM， sparse-dLLM, dllm-cache等），dLLM-Serve显著提升了吞吐量（RTX 4090 1.81 倍，L40S 1.74 倍），同时在重负载下将尾延迟降低了近4倍，并能在更低的稀疏度下保持模型生成质量。
<img width="770" height="226" alt="image" src="https://github.com/user-attachments/assets/c59f8145-3663-4bb8-a480-328e35c631b4" />

LLaDA-8B-Instruct - Latent Language Diffusion Autoregressive model
Dream-v0-Instruct-7B - Dream diffusion model with shifted prediction

dLLM-Serve是一项旨在解决生产环境中扩散式大型语言模型（dLLMs）服务所面临内存和计算效率挑战的系统设计。与自回归模型（ARMs）通过顺序生成令牌不同，dLLMs采用并行解码和迭代去噪（iterative denoising）过程来生成完整序列。然而，这种并行性也带来了一系列独特的系统挑战，导致“内存占用危机”（memory footprint crisis）。

**核心问题与挑战：**
1.  **大规模激活内存占用（Monolithic Logit Tensors）**：dLLMs在每次迭代中为整个序列生成logits，导致一个形状为`[B, L, V]`的巨大 `logit` 张量。这个张量是瞬态的，但其峰值内存需求极大，会迅速耗尽GPU显存，严重限制并发请求的数量。现有的服务系统，如 `Fast-dLLM` 和 `dKV-Cache`，并未有效管理这种激活内存峰值。
2.  **资源振荡（Resource Oscillation）**：dLLM推理过程在计算密集型的“Refresh”阶段和带宽密集型的“Reuse”阶段之间交替。在“Refresh”阶段，模型更新整个序列的KV缓存，消耗大量计算资源；而在“Reuse”阶段，模型仅更新活动块的令牌，并重用缓存的KV状态。现有调度器通常以请求（request）粒度进行调度，无法有效利用“Reuse”阶段释放的资源，导致GPU利用率低下。
3.  **稀疏注意力效率不足（Inefficient Sparse Attention）**：尽管dLLMs由于其固有的稀疏性（大量 `[MASK]` 令牌）非常适合稀疏注意力，但现有方法（如 `Sparse-dLLM`）通常采用跨所有注意力头（attention heads）的统一掩码。这简化了内存管理，但牺牲了模型的准确性，因为不同注意力头可能关注不同的语义特征。此外，逻辑稀疏性（logical sparsity）并未转化为物理内存节省，系统仍分配并加载完整KV缓冲区，然后掩盖掉无关令牌。

**dLLM-Serve的核心方法：**
dLLM-Serve通过引入三大核心机制来解决这些问题：

1.  **Logit-Aware Activation Budgeting（Logit感知的激活预算）**：
    *   **问题**：传统的 `dLLM` 推理会一次性计算所有令牌的 `logits`，产生一个瞬时但巨大的 `[B, L, V]` 张量。例如，对于 `LLaDA-8B`，该张量可达8.3GB，是导致 `OOM` 的主要原因。
    *   **解决方案**：dLLM-Serve 引入 **Logit Decomposition（Logit分解）**。它定义了一个系统参数 `max_num_logits`，限制同时计算 `logits` 的最大令牌数量。在执行过程中，如果当前批次中需要计算 `logits` 的令牌总数 `N_logit` 超过 `max_num_logits`，运行时会将输出投影分解为串行子批次（serial sub-batches）。
    *   **实现细节**：推理引擎会遍历这些子批次，每次处理不超过 `max_num_logits` 个令牌。计算完 `logits` 后，立即执行解码操作（如 `ArgMax` 或 `Top-k sampling`）获取下一个令牌，并立即释放临时 `logit` 缓冲区，然后再处理下一个子批次。这种机制确保了瞬时激活内存占用严格受 `max_num_logits` 限制，无论完整序列长度或批次大小如何。
    *   **效益**：通过将 `logit` 内存峰值转换为更小的固定预算，dLLM-Serve能够将节省的显存重新分配给KV缓存池，从而显著提高并发请求数量。

2.  **Phase-Multiplexed Scheduler（阶段多路复用调度器）**：
    *   **问题**：dLLM的“Refresh”和“Reuse”阶段具有截然不同的资源需求。“Refresh”阶段计算成本高，内存占用大（全序列KV更新），而“Reuse”阶段计算成本低，内存占用小（仅活动块更新）。传统以请求为粒度的调度方式会为整个请求生命周期保守地预留“Refresh”阶段所需的最大资源，导致在“Reuse”阶段出现大量空闲的“headroom”。
    *   **解决方案**：dLLM-Serve在 **步骤粒度（step granularity）** 上进行调度，并采用令牌级别的打包（token-level packing）。调度器维护一个严格的不变量：打包批次中活动查询令牌的总数永不超过 `max_num_batched_tokens`。
    *   **实现细节**：在每个去噪迭代中，dLLM-Serve 构建一个跨请求的活动令牌打包批次。处于“Refresh”阶段的请求贡献完整序列长度 `L` 的查询令牌，而处于“Reuse”阶段的请求仅贡献活动块长度 `L_block` 的查询令牌。当运行中的请求从“Refresh”阶段过渡到“Reuse”阶段时，它们对打包批次的贡献从 `L` 骤降到 `L_block`，释放了大量查询令牌预算。调度器会立即根据 `FCFS` 原则准入排队中的新请求，使其开始“Refresh”阶段的工作，直到打包批次再次达到 `max_num_batched_tokens`。
    *   **效益**：这种动态调度策略将带宽密集型“Reuse”阶段产生的“headroom”转化为额外的计算密集型“Refresh”工作，从而最大化GPU利用率，并有效推迟系统饱和点。

3.  **Head-Centric Sparse KV Cache Management（以头为中心的稀疏KV缓存管理）**：
    *   **问题**：现有的稀疏注意力方法（如 `Sparse-dLLM`）为了保持内存连续性，通常强制所有注意力头使用一个共享的全局稀疏掩码。这牺牲了模型准确性，因为不同注意力头可能对不同的上下文令牌有特殊需求。同时，这些方法通常采用逻辑掩码，并未真正节省物理内存或减少带宽，因为数据仍按完整模式加载。
    *   **解决方案**：dLLM-Serve 实现了 **以头为中心（Head-Centric）** 的稀疏KV缓存，在保持物理存储连续性的同时，支持每个头独立的令牌保留。
    *   **算法公式**：与全局稀疏方法不同，dLLM-Serve 为每个注意力头独立计算重要性分数。对于每个头 $h$ 和位置 $j$，其重要性分数 $S^{h,j}$ 通过局部池化（kernel size $w$）计算：
        $$S^{h, j} = \max_{m \in [j - \frac{w}{2}, j + \frac{w}{2}]}(Q_{b,h} \cdot K_{m,h}^\top)$$
        然后，为每个头选择其各自的 `TopK` 索引集 $I_h = \text{TopK}(S^{h,:}, k)$，其中 $k = L \cdot r$ ($r$ 为保留率)。由于 $I_h$ 通常与 $I_{h'}$ 不同，这提供了更高的建模能力。
    *   **实现细节**：dLLM-Serve 实现了 **逻辑稀疏性与物理布局的解耦（Decoupling Logical Sparsity from Physical Layout）**。在“Refresh”阶段，系统利用选定的索引 $I_h$ 立即将稀疏令牌打包到物理连续的KV布局中。关键在于，稀疏索引图是瞬态的，仅用于此打包步骤，不保留在KV缓存中。这使得随后的“Reuse”阶段可以直接从连续内存中访问KV缓存，避免了间接寻址的开销。为支持这种打包执行，内存管理器为每个请求预分配一个固定大小的 `rL × sizeof(KV)` 块，并将其组织成形状为 `[Nheads, rL, Dhead]` 的连续密集张量。

**系统实现与评估：**
dLLM-Serve 基于 `Nano-vLLM` 运行时扩展，复用了其分页KV缓存存储和 `FlashAttention` 等高性能组件。它添加了扩散模型特有的控制流、迭代去噪循环、阶段跟踪和稀疏KV缓存管理。

实验在 `RTX 4090` (24GB) 和 `NVIDIA L40S` (48GB) 两种硬件平台上进行，并使用 `LLaDA-8B-Instruct` 模型。dLLM-Serve 与 `dLLM-Cache`、`Fast-dLLM` 和 `Sparse-dLLM` 三个基线进行比较。

*   **吞吐量与可扩展性**：在 `RTX 4090` 上，dLLM-Serve 的吞吐量比最强的基线（Fast-dLLM）提高了 **1.81倍**。在 `NVIDIA L40S` 上，吞吐量提升高达 **3.12倍**。dLLM-Serve 在高负载下仍能保持线性扩展，而基线系统则在较低请求率下达到“吞吐量瓶颈”。
*   **延迟敏感性与稳定性**：在高负载下，dLLM-Serve 的平均端到端延迟比基线系统降低了近 **4倍**。例如，在 `L40S` 上，它将 `OSC` 数据集的平均延迟从基线的 `6791s` 降低到 `1849s`。dLLM-Serve 还显著降低了延迟抖动（Jitter）和尾部延迟（Tail Latency），提高了服务预测性，标准差和尾部范围分别减少了 **56%** 和 **53%**。
*   **生成质量**：dLLM-Serve 的 Head-Centric 稀疏注意力在低保留率下表现出优越的生成质量。在 `HumanEval` 和 `GSM8K` 数据集上，当保留率低至 `10%` 时，dLLM-Serve 的准确性显著高于使用统一选择的基线，例如在 `GSM8K` 上，准确率从 `40.0%` 提升到 `75.1%`，相对提升 **87.7%**。这意味着 dLLM-Serve 在相同质量下可以允许更低的内存占用，从而支持 **2倍** 的并发请求。
*   **消融研究**：消融研究表明，定制的推理引擎、阶段多路复用调度器和 `Logit-Aware` 激活预算策略都对性能提升做出了贡献。其中，推理引擎和优化后的内存布局贡献最大，其次是智能调度器，最后是 `Logit-Aware` 预算策略。

dLLM-Serve 为可扩展的 `dLLM` 推理奠定了基础，将理论上的算法稀疏性转化为实际的墙钟加速，并在异构硬件上展现了其鲁棒性。
## Breadcrumbs Reasoning
Breadcrumbs Reasoning: Memory-Efficient Reasoning with Compression Beacons

https://arxiv.org/pdf/2510.13797 2025.12.29 康奈尔 哈佛

https://github.com/lil-lab/breadcrumbs-reasoning 待开源

1. 🍞 针对大型语言模型（LLM）在长上下文推理中KV缓存的内存和计算成本问题，本文提出了一种名为**Breadcrumbs Reasoning**的方法，通过定期将**生成的KV缓存压缩成一个特殊的“beacon(灯塔 信标)”令牌并清除旧条目来提高效率**。
2. 💡 该方法**采用改进的联合蒸馏与强化学习**（RL-distillation）框架进行训练，使**模型能够学习如何有效地压缩信息并同时进行推理**，并在**训练期间通过注意力掩码模拟KV缓存清除**。
3. ✨ Qwen2.5-1.5B和Phi-4-Mini实验结果表明，Breadcrumbs Reasoning在内存效率和推理准确性方面达到了更优的Pareto前沿，显著优于未压缩模型和现有的训练无关压缩技术，且能支持更长的推理链，尤其在固定内存预算下表现出色。
本论文提出了一种名为“Breadcrumbs Reasoning (BR)”的记忆高效推理方法，旨在解决大型语言模型（LLMs）在处理长上下文推理时，Transformer键值（KV）缓存呈线性增长所导致的内存和计算成本问题。
<img width="608" height="317" alt="image" src="https://github.com/user-attachments/assets/d6318b40-33bd-4937-bf9b-64208504e6bd" />
<img width="675" height="504" alt="image" src="https://github.com/user-attachments/assets/39fca0dc-5d94-4476-aca7-dfa39dd8fed2" />
<img width="663" height="514" alt="image" src="https://github.com/user-attachments/assets/0495e0b4-e737-4236-b569-fd31ca52aa08" />

**核心思想与方法（详细与技术层面）：**
该研究基于这样一个假设：当模型生成推理token时，之前生成token的信息价值会逐渐降低，从而为压缩提供了机会。BR方法的核心是在生成过程中周期性地压缩KV缓存。具体而言，它引入了一个特殊的、经过学习的“beacon”token，用于概括之前一系列token的信息。一旦这些信息被压缩并存储在“beacon”token中，原始的KV缓存条目就会被逐出，从而大幅节省内存。
<img width="1478" height="518" alt="image" src="https://github.com/user-attachments/assets/46dced5a-21fa-469b-8b89-ddd5be236852" />
<img width="663" height="314" alt="image" src="https://github.com/user-attachments/assets/25ca2eb1-06c1-45d1-acfe-1efe17fc78c7" />
<img width="666" height="397" alt="image" src="https://github.com/user-attachments/assets/fad12da2-255c-430d-9b9c-2dfa5eefe9df" />


**实验设置与结果：**
1.  **模型与任务：** 实验在Qwen2.5-1.5B和Phi-4-Mini模型上进行，评估了三个具有挑战性的推理基准：Countdown（算术谜题，强调避免重复尝试）、LinSys（线性方程组，强调结构化演绎推理）和StarGraph（星图路径查找，模型易出错的自回归任务）。
2.  **压缩比与训练模式：** 压缩比$c$设定为2、4、8、16和32。训练模式分为：
    *   SR BR (Single Ratio Breadcrumbs Reasoning)：每个模型针对单一压缩比进行训练。
    *   MR BR (Multi Ratio Breadcrumbs Reasoning)：单个模型在所有压缩比下进行训练，通过重复每个batch来适应不同的压缩比，这允许在推理时动态调整压缩比。
3.  **基线：** 对比了四种训练无关的缓存淘汰基线：PyramidKV、SnapKV、TOVA和StreamingLLM。这些方法也适用于KV缓存压缩，但不需要额外的训练。
4.  **评估指标：** 在固定最大缓存大小（1000条目）和固定最大生成长度（1000步）两种配置下评估准确率，并使用准确率-缓存大小曲线下面积（AUAC）来衡量模型在不同缓存约束下的鲁棒性。
5.  **主要发现：**
    *   **BR的卓越性能：** BR在精度-内存权衡上实现了Pareto改进。在相同的内存预算下，BR模型能够生成更长的推理链，从而匹配甚至超越教师模型（未压缩RL策略）的准确性。在Countdown和StarGraph任务上，BR在许多情况下甚至优于教师模型，因为它能在更小的缓存预算下进行更多的推理步骤。
    *   **记忆效率：** BR在固定生成长度下，使用2-32倍更少的KV缓存条目，同时保留了65.1%-89.8%的原始性能。
    *   **多比率训练（MR BR）的优势：** 尽管MR BR需要处理不同的压缩粒度，但其平均性能优于SR BR，表明联合训练有助于模型在不同比率之间共享有效的压缩策略，并提供了推理时的灵活性。
    *   **训练无关基线的劣势：** PyramidKV、SnapKV、TOVA和StreamingLLM等训练无关方法表现明显不佳，尤其是在较高压缩比和复杂推理任务上。这强调了对于复杂推理任务，学习到的压缩方案是必要的。
    *   **LinSys的挑战：** BR在LinSys任务上的性能提升不如其他任务明显，尤其是在高压缩比下。分析表明，主要问题在于算术错误而非记忆或信息检索失败，这可能与压缩数字的难度或对模型内部算术电路的影响有关。
    *   **联合RL-蒸馏训练的验证：** 实验证明，联合RL-蒸馏训练方法与两阶段训练（先完整训练RL策略，再生成数据蒸馏）相比，表现相当或更优，证明了其在训练效率上的优势。
    *   **推理时间：** BR的推理时间与教师模型相当，并且平均快于所有训练无关基线。

**讨论与未来工作：**
本研究表明，推理链中存在显著的压缩空间，并非所有信息都同等重要。BR通过有效压缩，在保留大部分推理性能的同时，实现了显著的内存节省。这在一定程度上是内存与时间之间的权衡。未来的研究方向包括：
*   实现动态自适应的压缩比选择，而非固定压缩比。
*   探索更激进的压缩方法，以在不增加推理步骤的情况下提高效率。
*   将BR与CoT（Chain-of-Thought）缩短方法相结合，以实现推理长度和KV缓存的双重优化。
*   在更广泛的推理基准场景中评估BR的行为。

## KVZap
KVzap: Fast, Adaptive, and Faithful KV Cache Pruning 

https://arxiv.org/pdf/2601.07891 NV 2026.1.13
https://huggingface.co/collections/nvidia/kvzap

1. ✨ KVzap提出了一种**快速、自适应**的KV cache剪枝方法，通过**训练轻量级Linear或MLP模型**来**预测重要性分数**，并基于固定阈值动态丢弃不重要的KV对。
2. ⚡️ 该方法解决了KVzip等现有方案在速度和解码阶段适用性上的局限，实现了**prefilling(更有针对性)和decoding的高效应用**，且计算开销可忽略不计。
3. 🏆 在Qwen3-8B、Llama-3.1-8B-Instruct和Qwen3-32B等模型及RULER、LongBench、AIME25等长上下文任务上，KVzap实现了2-4倍的KV cache压缩，同时保持了可忽略的准确度损失。
<img width="827" height="402" alt="image" src="https://github.com/user-attachments/assets/64cccc2a-ab62-4017-9960-d446910b7b10" />

<img width="762" height="138" alt="image" src="https://github.com/user-attachments/assets/2a3b52d0-800d-44bc-8538-58f25b495136" />

尽管已有许多针对KV Cache的架构优化（如GQA、MLA、滑动窗口注意力等）或剪枝方法，但它们或未能有效压缩时间($T$)轴上的KV Cache，或因速度-精度权衡、通用性、优化兼容性等问题而未被主流推理引擎采纳。KVzap旨在解决这些问题。

- KVzip是prefill阶段压缩的SOTA；**但性能慢（repeat input twice，计算相关性score）且只能用在prefill阶段**。对于每个注意力头，原prompt中位置i处的KV对的重要性得分被定义为模型在重复时，对位置i的最大注意力权重。直观上，如果模型**在重复提示时很少关注某个位置的KV对**，则该KV对的信息量较低，可以被丢弃。
<img width="1105" height="440" alt="image" src="https://github.com/user-attachments/assets/9beaedb1-0b37-450d-b2a4-d39557e799ae" />

**训练过程**方面，KVzap利用了大规模预训练数据集（如Nemotron-Pretraining-Dataset-sample）中的1.2M个(h, log(s+))对进行训练。值得注意的是，KVzap的剪枝策略是**基于阈值的 (thresholding)**，而非固定比例的top-$k$选择。它会丢弃预测得分低于固定阈值$\tau$的KV对。这种自适应性使得压缩率能根据提示的信息密度动态调整：对于复杂输入保留更多Token，对于冗余输入则保留更少。此外，为了保持局部上下文，KVzap还保留了一个固定大小（默认为$w=128$）的滑动窗口，以遵循StreamingLLM等方法。

<img width="748" height="313" alt="image" src="https://github.com/user-attachments/assets/f4731765-174c-41c7-b4d5-e2124a1ea034" />

<img width="761" height="338" alt="image" src="https://github.com/user-attachments/assets/9207d2be-f3b6-4560-a3ef-e5fdcdea0700" />

<img width="757" height="330" alt="image" src="https://github.com/user-attachments/assets/d3f7c144-4c13-4d83-869e-f3c5d8ef0df0" />

实验结果表明，KVzap在Qwen3-8B、Llama-3.1-8B-Instruct和Qwen3-32B等模型上，在RULER和LongBench等长上下文任务以及AIME25等推理任务中，实现了2-4倍的KV Cache压缩，同时保持了可忽略的精度损失。KVzap-MLP通常优于KVzap-Linear，但对于Llama-3.1-8B-Instruct，KVzap-Linear表现出奇的好。KVzap的计算开销极低（KVzap-MLP最高1.1%，KVzap-Linear最高0.02%），在长上下文场景中相对于Attention的二次计算成本可以忽略不计。KVzap的自适应阈值剪枝优于固定比例剪枝，并且滑动窗口对于保持性能至关重要。


## prefillonly
PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications 

https://arxiv.org/pdf/2505.07203 SOSP25 2025.5

1. 针对大型语言模型（LLM）在推荐、信用验证等判别任务中出现的“Prefill-only”工作负载（**即仅生成一个输出Token**），现有LLM推理引擎因其为任意长度输出设计而效率低下。
2. PrefillOnly引擎通过引入混合预填充（hybrid prefilling）技术，分块处理非注意力层并可选择性地丢弃或卸载Suffix KV Cache，从而**显著减少GPU内存**占用并**提高最大输入长度**（MIL），避免了传统并行化带来的吞吐量损失。
3. PrefillOnly利用Prefill-only请求固定的**Job Completion Time**（JCT），通过**连续JCT校准实现了高效的JCT感知调度**。2卡L40-8b/A100-32b/H100 70b，vLLM。最终保证较低平均和P99延迟的同时**，**4倍**的每秒查询量。
<img width="511" height="262" alt="image" src="https://github.com/user-attachments/assets/6c84841d-918e-4ebe-b50c-330a4c05ccfb" />
<img width="553" height="784" alt="image" src="https://github.com/user-attachments/assets/20c3b20c-ce4f-419e-8bc7-ea259cd2bc2a" />

<img width="518" height="302" alt="image" src="https://github.com/user-attachments/assets/07106e04-1d56-4569-9a27-9770d568969f" />
<img width="1116" height="568" alt="image" src="https://github.com/user-attachments/assets/e4592f38-4b44-4e2a-ae1d-84bdfc60b7e8" />
<img width="560" height="433" alt="image" src="https://github.com/user-attachments/assets/6ff04f24-8012-4522-baf8-c05a1b4becc6" />

**1. 问题与机遇**

现有的 LLM 引擎是为可变长度输出设计的，因此未能充分利用 prefill-only 工作负载的独特属性：
1.  **更小的 GPU 内存占用**: 传统的 LLM 推理为了重复使用 KV cache 会存储所有层的 KV cache。但对于 prefill-only 任务，生成的**KV cache 大部分不会被再次使用，因为只需生成一个 token**。这意味着可以显著减少 GPU 内存占用，从而处理更长的输入序列
2.  **JCT（Job Completion Time）确定性**: 传统 LLM 请求的 **JCT 难以预测**，因为输出长度不确定。而 prefill-only 请求的输出长度固定为 1，这使得引擎能**够精确预估 JCT，从而实现更高效的 JCT 感知调度策略**（如最短剩余作业优先 SRJF）。

**2. 现有引擎的局限性**

*   **处理长请求的吞吐量权衡**: 现有引擎为了处理长请求，必须存储所有 KV cache，这限制了最大输入长度（MIL）。当请求长度超过 MIL 时，通常采用以下方法，但都会降低吞吐量：
    *   **Chunked Prefilling**: 分块预填充，降低 attention kernel 性能。
    *   **Tensor Parallelism**: 跨 GPU 通信开销大，降低整体吞吐量。
    *   **Pipeline Parallelism**: 引入气泡（bubbles），导致次优的延迟-吞吐量权衡。
*   **调度算法不感知 JCT**: 现有 LLM 引擎通常使用 JCT 不感知调度（如 FCFS），无法利用 prefill-only 请求 JCT 可预测的优势。此外，简单的丢弃 KV cache 只能略微提高 MIL，因为 LLM 推理本身分配的临时张量也会消耗大量 GPU 内存。

**3. PrefillOnly 的核心技术**

**3.1. 混合 Prefilling (Hybrid Prefilling)**

这是 PrefillOnly 提高 MIL 的关键优化。作者发现，简单地丢弃 KV cache 效果不佳的原因在于，LLM 推理过程中线性层（linear layers）产生的**中间张量 (intermediate tensors) 才是 GPU 内存占用的主要瓶颈。**这些中间张量的大小远大于单层 KV cache。

*   **原理**: 混合 prefilling 策略是：对非 attention 层 (non-attention layers) 进行分块处理 (chunk-by-chunk)，而对 attention 层 (attention layers) 正常处理。
    *   非 attention 层（如 MLP 模块）是线性操作，其计算结果独立于其他部分，因此可以分块计算。分块处理意味着在任意时间点，只需为当前处理的 chunk 存储中间张量，从而显著降低峰值 GPU 内存使用。
    *   Attention 层仍按常规方式处理，因为其计算依赖于整个序列。
*   **实现**: 通过 `torch.compile` 实现。
    *   将连续的线性操作分组为虚拟层。
    *   分块通过这些虚拟层，并在末尾拼接输出张量。
    *   **优化**:
        *   **Output Preallocation (输出预分配)**: 提前分配好最终的输出张量空间，避免在拼接时重复分配内存。
        *   **In-place Computation (原地计算)**: 如果输入和输出张量形状相同，则复用输入张量的 GPU 内存来存储输出张量，进一步节省内存。

**3.2. Suffix KV Cache Discarding / Offloading (后缀 KV Cache 丢弃/卸载)**
*   **目的**: 允许 PrefillOnly 在不降低吞吐量的情况下处理更长的请求，同时仍能利用 prefix caching。
*   **机制**: PrefillOnly 会最大限度地保**留前缀 token 的 KV cache，并丢弃或卸载后缀 token 的 KV cache。**
*   **使能者**: 混合 prefilling 是实现这一点的关键，因为它确保了整个 prefilling 过程可以在一次 LLM 前向传播中完成，使得不再需要为后续解码保留全部 KV cache。

**3.3. Continuous JCT Calibration (持续 JCT 校准)**
*   **问题**: 传统的 JCT-based 调度（如 SRJF）在 prefix caching 场景下表现不佳，因为 KV cache 的存在会动态改变请求的实际 JCT。一个请求的 JCT 会在共享前缀的 KV cache 可用时降低，在 KV cache 被逐出时升高。这导致调度器无法及时优先处理那些能命中缓存的请求。
*   **解决方案**: PrefillOnly 在每次调度前，持续校准等待队列中请求的 JCT。
    *   **JCT 估计**: 对于每个请求 $r$，PrefillOnly 基于其输入 token 数量 $n_{input}$ 和命中 prefix cache 的 token 数量 $n_{cached}$ 来估计 JCT。经验上，作者发现未命中 cache 的 token 数量 $(n_{input} - n_{cached})$ 是 JCT 的良好代理（Pearson 相关系数 0.987）。
    *   **调度算法**: 调度算法可以表示为：
        $$ \text{score} = \text{get\_jct}(n_{input}, n_{cached}) - \lambda \cdot T_{queue} $$
        其中，$T_{queue}$ 是请求在队列中的等待时间，$\lambda$ 是一个公平性参数。
    *   **优势**:
        *   **提高 Cache 命中率**: 动态校准使得调度器能及时识别并优先处理那些能够命中现有 prefix cache 的请求，从而提高整体 cache 命中率，降低平均延迟。
        *   **公平性与防饥饿**: 引入 $T_{queue}$ 项可以防止长请求被短请求持续饿死，通过降低长时间等待请求的优先级得分来保证其最终被调度。
*   **不进行 Batching**: PrefillOnly 选择了逐个调度请求，而不是进行 Batching。原因在于 prefill-only 工作负载是 GPU 计算密集型 (computation-bound)，而不是 GPU 内存访问带宽密集型 (memory-bandwidth-bound)（像解码阶段）。Batching 能够显著提高内存带宽受限情况下的吞吐量，但对计算受限的情况改善不大，反而会增加平均延迟。

**4. 评估**

PrefillOnly 在 4 种硬件配置、3 种 LLM 模型和 2 种模拟工作负载（Post Recommendation 和 Credit Verification）上进行了评估。

*   **数据集**:
    *   **Post Recommendation (短上下文，高前缀缓存复用)**: 模拟社交媒体推荐场景，用户档案长度约 11,000-17,000 tokens，帖子长度约 150 tokens。每个用户对应 50 个请求，前缀缓存复用频繁。
    *   **Credit Verification (长输入长度)**: 模拟信用验证场景，信用历史长度约 40,000-60,000 tokens。每个用户 1 个请求，重点测试 MIL。
*   **基线**:
    *   PagedAttention (vLLM 的默认管理策略)
    *   Chunked Prefill
    *   Pipeline Parallel
    *   Tensor Parallel
*   **结果**:
    *   **QPS-Latency 权衡**: PrefillOnly 在高 QPS (Query Per Second) 下始终实现最低的平均延迟和 P99 延迟，表明其吞吐量显著高于基线。在低 QPS 下，PrefillOnly 的延迟可能高于基于并行化的基线（因为它们利用了多 GPU 服务单个请求），但在高 QPS 下，并行化基线由于通信开销，吞吐量远低于 PrefillOnly。
    *   **MIL 提升**: PrefillOnly 能够将 LLM 的最大输入长度提高达 5 倍（相比非并行化基线），且不降低吞吐量。混合 prefilling 策略本身可以将 MIL 提高达 8.7 倍。
    *   **性能来源**:
        *   在 Post Recommendation 场景中，PrefillOnly 的优势主要来自于其持续 JCT 校准机制，有效提高了前缀缓存命中率，避免了在高 QPS 下基线因缓存不足而导致的性能瓶颈。
        *   在 Credit Verification 场景中，PrefillOnly 的优势主要在于其能够处理长上下文而无需并行化 LLM 推理。并行化方案会因昂贵的 `all-reduce` 通信（Tensor Parallel）或流水线气泡（Pipeline Parallel）而导致 GPU 闲置时间。即使使用 NVLink，PrefillOnly 仍能保持更高的吞吐量。
    *   **公平性参数 $\lambda$**: 提高 $\lambda$ 值可以改善 P99 延迟，但会略微增加平均延迟，显示了公平性与平均性能之间的权衡。

**未来工作**:
*   将后缀 KV cache 卸载到 CPU（例如使用 LMCache），以允许未来请求复用被丢弃部分的计算。
*   将 PrefillOnly 应用于 prefill-decode 分离架构中的 prefill 节点。
*   进一步探索 prefill-only 工作负载的延迟优化，例如通过连续分配 GPU 缓冲区。
  
## Long decode KV
Hold Onto That Thought: Assessing KV Cache Compression On Reasoning

https://arxiv.org/pdf/2512.12008 2025.12.12 马里兰大学 芝加哥大学等

https://github.com/minghui-liu/kvpress 基于NVIDIA对KVpress 增加了一些方法实现

1. 💡 本研究全面评估了多种KV cache压缩策略在需要长生成（**long decoding**）的LLM推理任务上的性能。
2. 🚀 结果显示，对于推理模型，基于注意力（**attention-based**）的"heavy-hitter"策略，特别是H2O和**解码版SnapKV，表现出显著优势**。
3. 📉 论文还发现，在较低的缓存预算下，压缩策略可能导致模型生成更长的推理序列，揭示了缓存大小与推理成本之间的潜在权衡。

不足：最大模型14b; 没有MoE模型；没有高难度的AIME25评测集；最大decode长度只有2K。

<img width="788" height="326" alt="image" src="https://github.com/user-attachments/assets/e8a92209-ca70-4b62-8607-34e376ec0483" />

全面评估了大型语言模型（LLMs）在长序列decode中，各种流行KV缓存压缩策略的性能。LLMs在复杂多步推理任务上表现出色，但其KV缓存（用于加速注意力计算）会随上下文长度线性增长，导致内存瓶颈。现有压缩算法旨在通过驱逐“不重要”的token来缓解这一问题，但大多数评估集中于处理长提示的prefill阶段，而非需要长生成（long decoding）的推理任务。
<img width="742" height="245" alt="image" src="https://github.com/user-attachments/assets/7da31738-7575-407b-bc1e-edcdcffc03d5" />
<img width="518" height="170" alt="image" src="https://github.com/user-attachments/assets/a4215fee-497e-4928-8a0b-1723c6bd66b7" />
<img width="783" height="432" alt="image" src="https://github.com/user-attachments/assets/98043bd8-59e4-4916-a6a8-cc44a0ed07b6" />
<img width="785" height="424" alt="image" src="https://github.com/user-attachments/assets/bef409a1-c892-4951-a1fc-65340aeee66e" />

**研究动机与背景：**
推理基准测试（如GSM8K、MATH500）通常要求LLM生成比问题本身长得多的答案，形成数千token长的“思考序列”（thinking sequences）。这使得KV缓存的线性增长 ($O(n)$ 内存复杂度) 成为一个严重问题。专门的推理模型（如DeepSeek-R1、Llama-Nemotron系列）尤其以其冗长的推理轨迹而闻名。KV缓存压缩方法通过维护固定大小的缓存来解决此问题，其核心在于如何定义并驱逐“不重要”的token。现有方法基于不同的启发式规则，包括注意力分数、余弦相似度、嵌入范数和特定头的token类型偏好。然而，这些方法的评估往往忽略了生成长度主导内存使用的场景。

**核心贡献：**
1.  **综合基准测试：** 论文在八个推理基准（FOLIO, DROP, GSM8K, MATH-500, ReClor, StrategyQA, CommonSenseQA, OpenBookQA）上，全面评估了主流KV缓存压缩策略，包括StreamingLLM、H2O、解码增强型SnapKV（SnapKV-Decoding）、R-KV和KNorm。评估覆盖了Llama-3.1-8B-Instruct（非推理模型）以及DeepSeek-R1-Distill-Qwen-7B/14B、Nemotron-Nano-8B-v1、DeepSeek-R1-Distill-Llama-8B（推理模型）等多个模型，以及不同缓存预算（128, 256, 384, 512 token）和最大生成token限制（2048 token）。
2.  **重新关注基于注意力的压缩：** 研究发现，经典的基于注意力分数的“重击者”（heavy-hitter）策略，即H2O和论文提出的SnapKV-Decoding，在推理模型上表现卓越，甚至在某些情况下超越了完整缓存（full-cache）性能。这表明跟踪重击者token对于推理轨迹至关重要。
3.  **开发解码压缩库：** 论文实现并开源了NVIDIA `kvpress2` 库的一个分支，**增加了对解码阶段压缩的支持，并集成了R-KV和H2O方法，为KV缓存压缩研究提供了开放平台。**

**KV缓存压缩方法技术细节：**
论文评估的几种主要策略及其工作原理：
*   **StreamingLLM：** 保留前几个（如四个）token作为“注意力汇点”（attention sinks），并结合最近token的滑动窗口。
*   **H2O (Heavy-Hitter Oracle)：** 基于token在生成过程中累积获得的注意力分数，动态识别并保留重要的“重击者”token。缓存由最近token和H2 token两部分预算组成。
*   **SnapKV (SnapKV-Decoding)：** 原始SnapKV主要用于prefill阶段的提示压缩。**论文将其扩展为SnapKV-Decoding (SnapKV-D)，使其在解码阶段也可用**。它使用一个小的“观察窗口”（observation window）（默认大小 $w=128$）来预测重要性。窗口中查询的注意力分数被聚合，以“投票”选出prefix中的重要位置（重击者）。在解码过程中，观察窗口沿已解码序列滑动，每 $w$ 步，SnapKV-D会根据当前窗口中token对缓存中所有token的注意力分数，驱逐分数最低的token以维持预算。计算开销为 $O(\frac{N}{w}Bd)$，其中 $N$ 是解码序列长度，$B$ 是缓存预算，$d$ 是键向量维度。
*   **R-KV：** 专为推理轨迹压缩设计，结合累积注意力分数和token间的键余弦相似度来识别不重要token。
*   **KNorm：** 一种计算高效的方法，不依赖注意力分数。它观察到键向量L2范数较低的token通常能从后续查询中获得高注意力分数，因此驱逐L2范数最大的token。计算开销为 $O(N)$。
*   **ShadowKV：** 将KV缓存卸载到CPU，不属于严格意义上的压缩，但作为基线进行比较。

**主要发现与分析：**
1.  **注意力是推理模型的最佳指示器：** SnapKV-D和H2O在推理模型上表现最佳，显著优于所有预算和数据集下的其他策略。这两种方法都依赖于累积注意力分数来确定要保留的token。SnapKV-D通过其滑动观察窗口机制，在捕获关键token方面略优于H2O。对“关键token留存率”（critical token retention rate）的分析（如Table 8和Figure 5所示）表明，SnapKV-D和H2O能以更高比例保留与问题相关的关键token（如数值、专有名词），因为这些token在推理过程中倾向于持续表现出高注意力。推理模型输出中关键token的密度也更高，进一步佐证了这种策略的有效性。
2.  **非推理模型的策略选择：** 对于非推理模型Llama-3.1-8B-Instruct，没有单一的策略占据主导地位，性能高度依赖于数据集。例如，StreamingLLM在GSM8K上表现出色，但在其他任务上效果不佳。这可能因为非推理模型对关键token的依赖性不如推理模型。
3.  **低预算下的性能下降与生成长度增加：** 对于推理模型，压缩策略（尤其是低预算下）可能导致生成更长的推理轨迹（“更健谈”的模型），甚至不终止的循环回答（如KNorm在Deepseek-R1-Distill-Llama-8B上的表现）。这揭示了缓存大小与推理成本之间的一种权衡：过度压缩可能导致关键信息丢失，迫使模型生成更长的、无效的思考序列以尝试找到答案。非推理模型未出现此现象。
4.  **注意力损失与性能关联：** 对不同方法的注意力损失（pre-eviction与post-eviction注意力分数之间的绝对差值）分析表明，注意力损失越小，模型性能越好。SnapKV-D和H2O的注意力损失最小，与它们的高性能相符。
5.  **计算开销与延迟：** StreamingLLM和KNorm由于不计算累积注意力分数，计算开销相对较低，其延迟优于H2O和SnapKV-D（参见Figure 2和Table 13）。然而，性能差距有时可以弥补延迟。
6.  **扩大模型规模的验证：** 对R1-Distill-Qwen-14B（更大推理模型）的测试同样发现H2O和SnapKV-D显著优于其他方法，表明基于注意力的驱逐策略在大规模推理模型中依然有效。
7.  **与稀疏注意力方法的比较：** 尽管本研究主要关注KV缓存压缩，但与稀疏注意力方法SeerAttention的初步比较显示，SnapKV-D与SeerAttention性能接近，SnapKV-D略优。值得注意的是，SnapKV-D维护固定大小缓存，而SeerAttention需维护完整缓存。

**实践指导：**
*   避免使用极低的缓存预算，因为这可能反而增加输出长度。
*   对于大预算（B > 1024）和较小最大token限制，StreamingLLM表现良好。
*   在其他情况下，SnapKV-D和H2O是首选，尤其对于推理模型。
*   对于SnapKV-D，在不显著影响性能的前提下，增加观察窗口大小（w）可以减少计算开销。
*   累积注意力分数是推理模型token重要性的高质量度量。


##  Reasoning or Guessing
Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Model

https://arxiv.org/pdf/2601.10679 2026.1.16 上海期智，清华大学

猜测？顿悟？变异？

1.  🤔 本文通过对Hierarchical Reasoning Model (HRM)的机械分析发现，其在推理复杂任务时**更像“猜测”**而非“推理”，表现为**对极简谜题的失败**、“顿悟”式的性能提升及陷入虚假固定点。
2.  💡 研究指出，HRM的这些行为源于其“**固定点属性**”的违反以及在潜在空间中被“**虚假固定点**”（spurious fixed points）所困，这与传统的递进式推理模型概念相悖。
3.  🚀 为此，作者提出了数据增强、**输入扰动和模型自举**三种“猜测”扩展策略，将Augmented HRM在Sudoku-Extreme上的准确率**从54%提升至96.9%**。

本文对分层推理模型（Hierarchical Reasoning Model, HRM）进行了深入的机制分析，旨在理解其推理模式和潜在的失效模式。研究发现，HRM的表现并非直观的逐步推理，而是更接近于“猜测”，并提出了支持这一观点的三个惊人事实：(a) 在仅有一个未知单元格的极简谜题上，HRM仍可能失败，这归因于其违反了固定点属性（fixed point property），该属性是HRM理论的基础假设。(b) 在推理步骤中存在“grokking”动态，即答案并非均匀改进，而是在某个关键推理步骤突然变得正确。(c) 存在多个固定点，HRM会“猜测”第一个遇到的固定点，该固定点可能是错误的，并可能长时间或永久性地陷入其中。这些发现暗示HRM似乎在“猜测”而非“推理”。

<img width="820" height="365" alt="image" src="https://github.com/user-attachments/assets/67e3ae8a-08d2-4bd6-9c6b-6c74728815f4" />

基于“猜测”的视角，论文提出了三种策略来提升HRM的“猜测”能力：数据增强（data augmentation），用于提升猜测质量；**输入扰动**（input perturbation），通过利用**推理随机性来扩大猜测数量**；以及模型自举（model bootstrapping/boosting），通过利用**训练随机性来扩大猜测数量**。这些方法结合后，形成了Augmented HRM，将Sudoku-Extreme任务的准确率从54.5%提升至96.9%。
<img width="523" height="296" alt="image" src="https://github.com/user-attachments/assets/df107c44-9e6b-4c1a-a135-bf6294037944" />

**主要发现与解决方案：**
1.  **固定点违反：**
    *   **现象：** HRM在极简的数独谜题（如仅一个单元格待填）上表现不稳定，即使模型找到了正确答案，后续的推理段也无法保持该答案，甚至完全失败（图2）。
    *   **解释：** 单步梯度训练方式导致模型未被显式训练以保持在简单问题上的稳定性。训练数据集中仅包含高难度数独和最终解，如同扩散模型中缺少中间噪声级别的数据，导致模型未能有效学习到从部分解到完全解的稳定性过渡。
    *   **修复：** 引入数据增强，为训练集中的每个谜题创建简化副本，即揭示一部分原本隐藏的单元格。这为模型提供了不同难度级别的“中间状态”数据，从而帮助其恢复固定点属性，并使潜状态轨迹表现出对称性（图3）。此方法将准确率从54.5%提升至59.9%。

2.  **“Grokking”动态与“猜测”假说：**
    *   **现象：** 宏观上（所有样本平均）HRM的损失随推理段数平稳下降（图4）。但在单个样本上，损失曲线呈现“grokking”现象：损失先快速下降到较低值，随后进入长时间的平台期，最后才突然骤降至零（图5）。这表明HRM并非渐进式地细化答案。
    *   **推理模式分类：** 通过将潜状态轨迹投影到主成分分析（PCA）平面，可分为四种模式：Trivial Success（早期解决）、Non-trivial Success（先徘徊后突然跳跃解决）、Trivial Failure（徘徊或震荡，未解决）、Non-trivial Failure（收敛到错误固定点）（图6）。非平凡模式表明存在“虚假固定点”（spurious fixed points）。
    *   **“猜测”假说：** HRM的递归过程更像是对合理潜状态的多次“猜测”尝试，而非增量式推理。模型在大部分中间步骤中围绕虚假固定点徘徊，未取得实质进展，直到它“猜测”到正确的潜状态。

3.  **虚假固定点与“逃逸”策略：**
    *   **机制：** 虚假固定点是潜空间中的误导性吸引子，模型一旦进入其附近区域，就很难离开。这些虚假固定点与真实固定点之间存在竞争关系，形成清晰的分界线（图7左）。
    *   **局部极小值解释：** 通过定义一个衡量数独冲突数的启发式错误指标 $E(\hat{y}) = \sum_{u \in \{r(\hat{y}),c(\hat{y}),b(\hat{y})\}} \sum_{d=1}^9 \text{ReLU}(\text{count}(d, u)-1)$，发现误导性吸引子是该指标的浅层局部极小值。模型从虚假固定点向真实固定点移动时，该指标值会先略微升高再降至0，表明存在一个“能量势垒”（图7右）。
    *   **“逃逸”策略：**
        *   **输入扰动：** 利用数独固有的对称性（如数字重标签、行列交换等）创建语**义等效但形式不同的输入**。HRM对这些变换是“无知”的，因此**不同的输入版本能带来不同的推理轨迹**。通过数字重标签，**准确率提高了18.7%**（表1）。
        *   **模型自举：** 选取训练过程中**不同时间点的相邻模型检查点进行集成**。尽管这些检查点高度相关，但它们在解决特定测试样本上的能力差异显著。此**方法带来了10.2%准确率**提升（表1）。
    *   这些扰动技术在推理时进行多次前向传播，通过多数投票决定最终答案，有效增加了模型“猜测”到正确答案的机会。


## TRIM
TRIM: Hybrid Inference via Targeted Stepwise Routing in Multi-Step Reasoning Tasks

https://arxiv.org/pdf/2601.10245 2026.1.16 CMU, AWS等

1. 🧠 TRIM 提出了一种创新的**step-wise router**，通过仅将**关键步骤路由给更强大的大型语言模型**（Ms），而让**更小的模型（Mw）处理常规延续，从而显著提高了多步推理任务的成本效率**。
2. 🛠️ 该方法利用 Process Reward Models (**PRMs**) 识别错误步骤，并整合了多种路由策略，包括简单的 TRIM-Thr、基于 RL 训练的 TRIM-**Agg** 以及处理 PRM 噪音的 TRIM-POMDP，以在准确性和成本之间进行权衡。
3. 🚀 实验结果表明，TRIM 在 MATH-500 和 AIME 等基准测试上实现了显著的成本效率提升（最高达 6 倍），并展现出强大的跨数据集泛化能力，证明其 step-level 难度模式捕捉了推理的基本特性。

<img width="1160" height="275" alt="image" src="https://github.com/user-attachments/assets/15534718-f952-4314-83f8-9944ed315df6" />

针对现有 LLM 路由方法将整个查询分配给一个模型，以及多步骤推理中“瀑布式失败”现象（即早期一个错误步骤可能导致整个解决方案崩溃）的痛点，TRIM 引入了一种细粒度的步进式干预策略。

**核心思想：** 并非所有推理步骤都具有相同的难度或重要性。只在那些关键的、容易导致推理链脱轨的步骤进行干预，将这些“关键步骤”路由给更强大的大型语言模型 (Large Language Model, LLM)，而让小型、经济的模型处理常规的延续步骤。这种**靶向干预”显著提升了推理效率**，通过限制昂贵模型 (strong LLM, M_sMsM_sMs​) 的调用，精确作用于那些能有效防止级联错误的步骤。使用 Process Reward Models (PRMs) 来评估中间推理步骤的质量，并根据这些评估做出路由决策。
<img width="1370" height="496" alt="image" src="https://github.com/user-attachments/assets/c172943a-2e06-42ab-8677-9c74e8ac8187" />

模型配置：廉价模型 M_wMwM_wMw​ 为 Qwen2.5-3B-Instruct，昂贵模型 M_sMsM_sMs​ 为 Claude 3.7 Sonnet，PRM 为 Qwen2.5-Math-PRM-7B。
基准测试：MATH-500、AIME、OlympiadBench 和 Minerva Math。
基线方法：RouteLLM (BERT、Matrix Factorization、SW Ranking)、Smoothie、AutoMix，以及本文提出的 AutoMix-PRM 变体（用 PRM 评分替代原始的 self-verification 信号）。
评估指标：CPT (Cost-Performance Threshold)，表示达到特定性能水平所需的最小 token 成本；\DeltaΔ\DeltaΔIBC (Incremental Benefit Per Cost)，衡量相对于基线的成本效益增益；PGR (Performance Gap Recovered)，衡量 M_wMwM_wMw​ 和 M_sMsM_sMs​ 之间性能差距的恢复程度。
<img width="1155" height="650" alt="image" src="https://github.com/user-attachments/assets/24bb4db9-1d2c-4b96-af5e-1370e1695079" />

## XOR cache
The XOR Cache: A Catalyst for Compression" by Zhewen Pan, Joshua San Miguel

https://dl.acm.org/doi/abs/10.1145/3695053.3730995 ISCA25 Best Paper Honorable Mentions

1. 💡 XOR Cache 通过利用私有缓存和包容性策略的**数据冗余**，引入了一种新颖的压缩方案，解决LLC低效率问题。
2. ⚙️ 该新架构存储**缓存行对的按位 XOR 值**，既实现了**行间压缩（将存储的行数减半）**，又通过**降低相似数据的熵来促进行内压缩**。
3. 🚀 模拟器评估结果表明，XOR Cache 显著将 LLC 面积减少 1.93 倍，功耗减少 1.92 倍，而性能开销仅为 2.06%，从而使 energy-delay product 降低 26.3%。
   
<img width="660" height="304" alt="image" src="https://github.com/user-attachments/assets/ce2047c2-135b-4d04-a8d6-312b7ca8547e" />

## Platinum
Platinum: Path-Adaptable LUT-Based Accelerator Tailored for Low-Bit Weight Matrix Multiplication

https://arxiv.org/pdf/2511.21910 2025.11.26 杜克

1. 🚀 Platinum 提出了一种轻量级 ASIC 加速器，专为**利用查找表** (LUT) 加速**超低位量化**大语言模型的混合**精度矩阵乘法** (mpGEMM) 而设计。
2. 💡 该加速器通过离线生成构建路径来减少 LUT 构建开销，并通过自适应路径切换支持通用的位串行执行和优化的三元权重执行。
3. ⚡️ Platinum 在 BitNet **b1.58-3B** 上实现了显著的性能提升，相比现有基线在速度上最高达到 73.6 倍，在能耗上最高降低 32.4 倍，且芯片面积仅为 0.96mm²。
超低位量化为结果复用提供了大量机会，可通过查找表（LUTs）加速。然而，现有基于LUT的方法存在LUT构建计算和硬件开销过大，以及仅依赖位串行计算（对于三值权重网络次优）的问题。

## STEM
STEM: Scaling Transformers with Embedding Modules

https://arxiv.org/pdf/2601.10639 Meta CMU等 2026.1.15

Github: https://github.com/Infini-AI-Lab/STEM

Website: https://infini-ai-lab.github.io/STEM

中文： https://mp.weixin.qq.com/s/js1U3ag2xkXeF_jHpDOS8w

1. STEM（Scaling Transformers with Embedding Modules）是种**静态、token-indexed方法**，它通过层**局部嵌入查找替换FFN的up-projection**，旨在**提高参数容量，同时解决传统MoE训练不稳定、负载不均衡和通信开销大**等挑战。
2. 设计使得STEM在**极端稀疏性下仍能稳定训练，并在知识和推理密集型**基准测试（如ARC-Challenge、OpenBookQA、GSM8K、MMLU）中**实现显著的下游性能提升**，同时**增强了长上下文性能**。
3. STEM**还减少了per-token FLOPs和参数访问以提高效率**，并且其token-indexed的嵌入特性提供了更强的可解释性，支持知识编辑和注入。
<img width="873" height="342" alt="image" src="https://github.com/user-attachments/assets/2ef8dad0-a39f-437e-be7c-aa7f46a6718f" />

**STEM 的优势**
1. **训练稳定性**：尽管具有极高的稀疏性，STEM 在训练过程中没有表现出像 MoE 模型那样的损失尖峰 (loss spikes)，训练曲线更平滑，且在相同的训练 FLOPs 下能达到更低的损失。

2. **增强的知识存储容量**：STEM 学习到的嵌入空间展现出更大的角扩展 (angular spread)，即嵌入向量之间的余弦相似度较低（接近正交），这减少了表示干扰 (representational interference)，有效增加了存储和检索信息的“槽位”数量，从而提升了模型处理知识密集型任务（如 ARC-Challenge, OpenBookQA, GSM8K, MMLU）的能力。

3. **可解释性与可控性** (Interpretability & Controllability)：由于每个 STEM 嵌入都与一个特定的 token ID 相关联，个体“微专家”具有清晰的 token 级语义。这使得 STEM 模型具有独特的知识编辑 (knowledge editing) 能力：通过简单地交换或修改特定 token 的 STEM 嵌入，可以在不改变输入文本的情况下，引导模型的输出分布（例如，将关于“西班牙”的生成改为关于“德国”），这表明**事实知识在这些嵌入中是局部化的、可编辑的**。论文详细探讨了处理不同 token 长度实体时的编辑策略：左填充 (left padding)、复制 (copying)、子集选择 (subset selection) 和平均 (averaging)。
长上下文推理 (Long-context Inference)：随着序列长度的增加，STEM 能够激活更多不同的参数，实现测试时的容量扩展。在 Needle-in-a-Haystack (NIAH) 等长上下文任务中，STEM 模型的性能增益随上下文长度的增加而增强，并且在 LongBench 上也表现出一致的优势。

4. **效率提升**：通过**消除 FFN 的 up-projection 矩阵，STEM 减少了约三分之一的 FFN 参数**。这在训练和预填充 (prefill) 阶段降低了每 token 的 FLOPs，并在解码阶段减少了参数加载成本 (parameter loading cost)。论文提供了详细的理论分析，展示了在不同模型规模下 FLOPs 减少的百分比（例如，Qwen2.5-32B 可达 24.8%）。

5. VRAM 和通信节省：大的 STEM 嵌入表可以卸载到 CPU 内存，从而释放 GPU 内存。由于其 token 索引的特性，所需的嵌入可以异步预取 (asynchronous prefetch)，且无需像 MoE 那样进行所有专家之间的 all-to-all 通信，大大降低了通信开销。

**系统实现**
CPU Offloading：将大型 STEM 嵌入表卸载到 CPU 内存。
Asynchronous Prefetch：在 GPU 计算的同时，异步预取所需的嵌入。
Token Deduplication：对批次中的 token ID 进行去重，以减少 CPU-GPU 通信量。
LFU Caching：利用 token ID 的 Zipfian 分布特性，实现高效的 LFU 缓存，提高命中率。
Parallel Embeddings：在训练期间，将 STEM 嵌入表分片 (shard) 到多个 GPU 上。

**实验结果** 
STEM 在 350M MobileLLM 和 1B Llama3.2 模型上进行了评估，并与 dense baseline 和 Hash Layer MoE 进行了比较。结果显示，STEM 在下游任务上实现了高达 3–4% 的准确率提升，尤其是在知识密集型和推理任务（如 ARC-Challenge, OpenBookQA, GSM8K, MMLU）上表现显著。此外，STEM 展示了更高的训练投资回报率 (Training ROI)，即在更少的训练 FLOPs 下达到更高的模型准确率。消融研究进一步证实了 STEM 层数量增加对性能的积极影响，以及 STEM 在 FFN 中放置位置的重要性（up-projection vs. gate-projection）。

## Single-Stage Huffman Encoder 
Single-Stage Huffman Encoder for ML Compression 

https://arxiv.org/pdf/2601.10673 2026.1.15 Google

1. 👉 该论文提出了一种**单阶段 Huffman 编码器**，旨在解决大型语言模型 (LLM) **训练中网络带宽瓶颈问题**，同时克服传统三阶段 Huffman 编码器的延迟和开销限制。
2. 💡 该方法利用LLM层和分片之间观察到的**张量高统计相似性**，采用从平均概率分布**导出的固定码本**，从而**省去了运行时频率分析和码本传输的开销**。
3. ✅ 通过预计算和共享这些固定码本，对Gemma2B SFT BF16实现了接近单分片 Huffman 编码（0.5% 以内）和理想 Shannon 压缩（1% 以内）的压缩率，从而实现了**高效的无损压缩**。但论文没给具体实现。

旨在解决大型语言模型 (LLM) 训练和推理中集体操作**受网络带宽限制的问题**。传统的 Huffman 编码器由于其三阶段设计（即时**频率分析、码本生成和码本传输**）引入了计算、延迟和数据开销。

<img width="655" height="247" alt="image" src="https://github.com/user-attachments/assets/aaf7bfe0-9e04-4696-9b1e-be728b59e1ed" />

**核心问题与传统方法的局限性：**
LLM（例如 Gemini、Gemma、LLaMA、GPT）的训练和推理涉及数据分区 (sharding) 和跨多个加速器的并行计算。不同的并行策略（如 Data Parallelism、Tensor Parallelism、Pipeline Parallelism、Expert Parallelism 和 Sequence Parallelism）会调用各种集体操作 (Collective Operations)，例如 AllReduce、ReduceScatter、AllGather 和 AlltoAll。这些集体操作通常受到**网络带宽的限制。虽然无损压缩，特别是 Huffman 编码**，是减少网络流量和提高性能的有效方法，但其传统实现流程繁琐：
1.  **第一阶段：** 扫描整个输入以构建符号的频率表。
2.  **第二阶段：** 运行 Huffman 算法，为每个符号**生成变长编码**。
3.  **第三阶段：** 再次扫描输入，用相应的编码替换符号。

**提出的单阶段 Huffman 编码器方法：**
论文的核心创新在于利用**数据的统计相似性来消除传统 Huffman 编码的实时开销**。其关键思想是使用**固定码本 (fixed codebooks)**，这些码本是从之前批次数据的**平均概率分布 (average probability distribution)** 中预先推导出来的。

**核心方法学与技术细节：**
1.  **统计相似性分析：** 论文通过分析 Gemma 2B 模型在 Supervised Fine Tuning (SFT) 过程中的 FFN1 激活张量（共 18 层，分片到 64 个 TPU 上，总计 1152 个分片）来验证其核心假设。研究表明，不同分片之间的 **Probability Mass Function** (PMF) 具有高度统计相似性。具体来说，对所有 FFN1 激活分片的 PMF 进行平均，然后计算每个分片 PMF 相对于这个平均分布的 Kullback-Leibler (KL) Divergence。结果显示，大多数**分片的 KL Divergence 值非常小**（小于 0.06），这强烈支持了平均分布可以很好地近似真实分布的结论。这证明了在分布式系统中，不同计算单元处理的相似张量具有一致的统计特性。
2.  **固定码本的生成与应用：**
    *   **离线生成：** 基于上述统计相似性，Huffman 码本不再需要为每个数据批次或每个分片实时生成。相反，可以**利用历史数据批次或训练/推理运行的平均概率分布**，在非关键路径 (off the critical path) 上预先计算并生成一套 Huffman 码本。
    *   **多码本管理：** 系统可以**为不同类型的张量（例如 FFN1 激活、FFN2 权重梯度）和不同的数据类型**（如 bfloat16, e4m3 等）维护不同的预计算码本。
    *   **码本共享与识别：** 这些预计算的码本在所有参与的节点之间共享。在实际操作时，编码器只需发送编码后的数据以及所使用的码本 ID (code book id)，而无需传输整个码本。接收方根据码本 ID 选择对应的预存码本进行解码。
3.  **单阶段压缩：** 通过使用预先确定的固定码本，Huffman 编码过程可以简化为一个单阶段操作：**直接用预计算的码本对数据进行编码**。这消除了实时频率分析和码本生成/传输的开销，使得压缩能够高效地在**片间通信 (die-to-die communication) 等延迟敏感场景中**“即时 (on-the-fly)”完成。
4.  **压缩性能：** 实验结果表明，使用从平均分布派生出的固定码本，其压缩率达到了与理想 Shannon 可压缩性 (Shannon compressibility) **1% 以内**，并且与每个分片单独生成 Huffman 码本所能达到的压缩率仅差 0.5%。这表明在极大地降低了计算和延迟开销的同时，保持了非常高的压缩效率。

**实现方式：**
所提出的方法可以在软件或硬件中实现。在软件实现中，程序员选择特定的码本。在硬件实现中，可以并行评估多个码本以选择最佳压缩效果的码本。


## Proserve
PROSERVE: Unified Multi-Priority Request Scheduling for LLM Serving

https://arxiv.org/pdf/2512.12928 京东 中科大等 2025.12.15

1. PROSERVE针对LLM服务中现有调度器未能同时优化Service Level Objective (SLO)达成和客户端优先级的问题，将其形式化为服务增益最大化任务，并提出**Token-level Deadline-aware** Gain (TDG)作为衡量指标。
2. 引擎层，PROSERVE引入了**SlideBatching**，这是一种根**据实时负载动态调整批处理形成和请求排序的本地调度器**，通过**滑动边界机制平衡了“deadline-first”和“density-first”**策略。
3. 服务层，GoRouting执行面向**增益和能力感知的请求分发，主动为未来的高优先级或长请求保留容量**，从而使系统**增益提高高达35%**，SLO达成率提高高达52%。

基于京东等xLLM（c++），升腾910B单机16卡；Qwen7b/32b模型；对标vLLM基线，weighted VTC; FCFS/priority, FairBatching等。

<img width="574" height="142" alt="image" src="https://github.com/user-attachments/assets/bbf9650b-53b5-4be1-b45b-1fcf526fde4d" />
<img width="441" height="310" alt="image" src="https://github.com/user-attachments/assets/2fa0437b-d85f-4204-a8a6-64061fad0f06" />

   
## TokenScale
TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity

https://arxiv.org/pdf/2512.03416 新加坡国立等，2025.12.3

1. 现有LLM服务自动扩缩容策略因使用GPU利用率或请求计数等滞后指标，难以有效应对突发流量，导致SLO违规和资源浪费。
2. 为此，TokenScale引入了Token Velocity这一统一的预测指标，并设计了Convertible Decoders机制，使decoder GPU能在流量高峰时动态执行prefill任务。
3. 在生产流量追踪下的评估显示，TokenScale相比现有最先进系统将SLO达标率从50-88%提升至80-96%，并降低了4-14%的GPU成本。

基于VLLM0.9.2 LMCache, 2台A100机器（CX6）；2台H100机器（CX6）；llama3.18b/Qwen-32b。
对标基线：AIBrix；DistServe等。
<img width="525" height="303" alt="image" src="https://github.com/user-attachments/assets/56f779d2-daa3-4028-a607-3d0336139d93" />

   
## Dr. Zero
Dr. Zero: Self-Evolving Search Agents without Training Data

https://arxiv.org/abs/2601.07055 [Meta Superintelligence Labs 伊利诺伊大学] 2026.1.13

Code: https://github.com/facebookresearch/drzero

1. 💡 Dr. Zero是一个**无需训练数据的自演化框架**，旨在提升**大型语言模型（LLM）搜索代理的推理和搜索能力**。
2. ⚙️ 该框架设计了一个**提问者-解决者反馈循环**，并通过**难度引导奖励和跳跃分组**相对策略优化（HRPO）来生成**多样化且具有挑战性的问题**，从而显著提高训练效率。
3. 🎯 实验结果表明，Dr. Zero在复杂问答（QA）基准上能够匹配甚至超越完全监督的搜索代理，证明了**复杂推理和搜索能力可**以通过纯粹的自演化实现。
Dr. Zero 的迭代过程如下：

Proposer 生成： Proposer \pi_\thetaπθ\pi_\thetaπθ​ 生成一批具有不同跳数结构的问题-答案对。
Proposer 优化： 利用 Solver 的反馈，Proposer 通过 HRPO 进行优化，以生成可验证、多样化且具有挑战性的问题。
Solver 优化： Solver \pi_\phiπϕ\pi_\phiπϕ​ 利用这些生成的数据通过 GRPO 来优化其搜索和推理能力。
这个交替优化循环创建了一个共生反馈机制：随着 Solver 的改进，简单问题的奖励会降低，迫使 Proposer 探索更复杂的推理路径来最大化其回报。反之，日益困难的问题可以防止 Solver 的训练奖励停滞不前，从而使其能够不断扩展推理技能。两个模型都从相同的基础 LLM 初始化，并仅依靠外部搜索引擎来驱动其性能改进，无需任何训练数据。

<img width="807" height="286" alt="image" src="https://github.com/user-attachments/assets/573c45e5-babc-49d5-b4b2-ecca7cbbb815" />

<img width="789" height="338" alt="image" src="https://github.com/user-attachments/assets/7bf69673-1e06-484b-86cd-cddec8dc0a9d" />

<img width="789" height="509" alt="image" src="https://github.com/user-attachments/assets/cca2a84d-9472-4b45-bf38-ab43f8d7e427" />

## Gecko
Gecko: An Efficient Neural Architecture Inherently Processing Sequences with Arbitrary Lengths

https://arxiv.org/abs/2601.06463 [University of Southern California & Meta AI Research] 2026.1.10

https://github.com/XuezheMax/gecko-llm

1. 🚀 Gecko是一种基于Megalodon骨干的新型神经网络架构，通过引入**时间步衰减归一化**、滑动**分块注意力（**SCA）和**自适应工作记忆（AWM）**来高效、内在地处理**任意长度的序列**。
2. 💡 Gecko的时间步衰减归一化解决了传统**时间步归一化**的问题，SCA通过整合前一个块的信息改善了上下文边界，而AWM则**利用位置感知在线softmax核以分块级别捕获长期**信息。
3. 📈 在2万亿训练tokens的规模上，**Gecko-7B的训练损失**（1.68）显著优于Llama2-7B和Megalodon-7B，并展现了处理高达**4百万tokens序列和从注意力窗口4倍长上**下文检索信息的固有长上下文能力。

<img width="747" height="475" alt="image" src="https://github.com/user-attachments/assets/66f2739f-a19a-41af-959f-aac2484bbef2" />
<img width="1484" height="536" alt="image" src="https://github.com/user-attachments/assets/84ee8bf5-bd98-4905-a9e4-0479908d1e6d" />
<img width="1487" height="376" alt="image" src="https://github.com/user-attachments/assets/dbc68ac1-0dae-47c6-a584-ba621d641cad" />
<img width="1485" height="554" alt="image" src="https://github.com/user-attachments/assets/093fc92d-6fc7-441c-ae97-5cbc1e182eb3" />


## MiMo-V2-Flash
MiMo-V2-Flash Technical Report

https://arxiv.org/pdf/2601.02780 小米 2026.1.8

https://mp.weixin.qq.com/s/3a2xz8LYhyV6udSgxuQFoA 

1. 架构创新：采用混合注意力机制，以 5:1 的比例交替使用滑动窗口注意力（SWA，窗口大小仅 128）和全局注意力（GA），并引入可学习的 Attention Sink Bias，在大幅降低 KV Cache 的同时维持了长文性能。
2. 多 token 预测 (MTP) ：引入轻量级 MTP 模块作为推测解码的草稿模型，在推理阶段实现了最高 2.6 倍的加速。
3. MOPD 后训练范式：提出“多教师在线蒸馏”（Multi-Teacher On-Policy Distillation），通过三个阶段（通用 SFT -> 领域专家 RL -> 学生模型 MOPD）解决了传统模型合并中的能力互斥问题，使单一模型能同时掌握多个领域专家模型的巅峰能力。
4. 基础设施：引入 Rollout Routing Replay (R3) 解决 MoE 在 RL 训练中的路由不一致问题，并构建了支持大规模 Agent 训练的仿真环境。

<img width="738" height="415" alt="image" src="https://github.com/user-attachments/assets/a4dddbcc-8d38-4183-a1f8-d84c3b0ef656" />


## REVATI
REVATI: Transparent GPU-Free Time-Warp Emulation for LLM Serving 

https://arxiv.org/pdf/2601.00397 佐治亚理工 2026.1.1

中文解读：https://mp.weixin.qq.com/s/2GsXROvVqc4G3JebZJJAsw

<img width="631" height="485" alt="image" src="https://github.com/user-attachments/assets/87ed570c-19ec-4f7d-a8a6-9dc0593bc99e" />

1. 🤔 为解决LLM服务系统配置测试昂贵且传统模拟器难以跟上快速迭代的问题，REVATI提出了一种GPU-free的时间扭曲仿真器。
2. ⚙️ REVATI通过**拦截CUDA API调用来虚拟化设备管理**，并协调**分布式进程进行虚拟时间跳跃（time jumps）**，以快速**跳过GPU计算的等待时间并保持因果关系**。
3. 🚀 该系统在vLLM和SGLang上实现了**低于5%的预测误差**，同时运行速度**比实际GPU执行快5-17倍**，大幅提升了LLM服务性能评估的效率和成本效益。基于Maya (一个团队) 和 Vidur

中文解读 https://mp.weixin.qq.com/s/2GsXROvVqc4G3JebZJJAsw

<img width="505" height="446" alt="image" src="https://github.com/user-attachments/assets/17cc17f9-086f-474d-afb7-fa0d5766900a" />
<img width="523" height="594" alt="image" src="https://github.com/user-attachments/assets/13251562-08de-4997-8ed5-56ededc79a57" />
<img width="490" height="342" alt="image" src="https://github.com/user-attachments/assets/9537de06-37a7-4ed5-b61e-f48732cd77ca" />
<img width="505" height="380" alt="image" src="https://github.com/user-attachments/assets/24ff34a1-04cd-4348-b4c4-05b32561b0ce" />
<img width="506" height="529" alt="image" src="https://github.com/user-attachments/assets/99522f72-8638-4b96-bd08-48d0758b785a" />
<img width="511" height="289" alt="image" src="https://github.com/user-attachments/assets/fd9432cc-4a7c-434c-8355-99d8fb19c93c" />

## Maya
Maya: Optimizing Deep Learning Training Workloads using GPU Runtime Emulation

https://arxiv.org/pdf/2503.20191 佐治亚理工 (与Revati一个团队) NVIDIA；2025.11.15 EuroSys26 

1. 🚀 Maya 是一款性能**建模系统**，通过透明的 GPU **runtime emulation** 解决了深度学习**训练优化中**现有系统面临的语义鸿沟和通用性-易用性权衡问题。
2. 💡 该系统通过**拦截未经修改的训练代码**的设备 API 调用来**模拟 GPU 交互并捕获低级操作的详细轨迹，**从而在不依赖实际硬件的情况下实现准确的性能预测。
3. 🎯 Maya 在各种模型和优化策略中实现了低于5%的预测误差，并能识别出比现有方法**降低高达56%训练成本的配置**，同时通过 worker deduplication 和 trial pruning 大幅提升了搜索效率。
   H100/V100/A40 单机多机。Torch2.1 
   
<img width="1256" height="524" alt="image" src="https://github.com/user-attachments/assets/96880831-e593-4ddc-b2e4-9de2320b3f90" />
<img width="1248" height="523" alt="image" src="https://github.com/user-attachments/assets/cb154741-2b94-4280-bb01-c9d90d7db093" />
<img width="1244" height="295" alt="image" src="https://github.com/user-attachments/assets/6a23873a-9119-44e0-8713-64fdcfabeef9" />
<img width="601" height="667" alt="image" src="https://github.com/user-attachments/assets/05d7f028-df20-4cbc-8411-aa8de2e6f143" />
<img width="1253" height="378" alt="image" src="https://github.com/user-attachments/assets/ca15a731-93af-463b-9e5f-85f9be3208f3" />
<img width="608" height="487" alt="image" src="https://github.com/user-attachments/assets/55398ae2-e69c-41b9-997f-23436faf893a" />
<img width="1247" height="384" alt="image" src="https://github.com/user-attachments/assets/c2d0a34a-3460-470c-b890-49b5bccace4c" />
![Uploading image.png…]()


## GDPO
GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization

paper：https://arxiv.org/pdf/2601.05242

code：https://github.com/NVlabs/GDPO

project：https://nvlabs.github.io/GDPO/
中文解读：https://mp.weixin.qq.com/s/pv6qpKxh8qDgFbPYABGA3g 
https://mp.weixin.qq.com/s/K77IcMZWgk_nTHIPSM7fUQ
<img width="794" height="464" alt="image" src="https://github.com/user-attachments/assets/7a4eece3-5698-44b1-aa2a-3b42baafe070" />

<img width="799" height="554" alt="image" src="https://github.com/user-attachments/assets/9b55a8a2-c487-4a6f-a978-f3326dc42150" />

<img width="795" height="465" alt="image" src="https://github.com/user-attachments/assets/eb8a647d-a5c7-4746-ad9f-3aca8b152d46" />

<img width="800" height="355" alt="image" src="https://github.com/user-attachments/assets/0de25e19-13cf-486d-b23a-9e37294558e4" />

1. 📉 现有研究中，Group Relative Policy Optimization (GRPO) 在多奖励强化学习 (RL) 中直接对汇总奖励进行归一化，导致不同奖励组合坍缩为相同的优势值，从而降低了训练信号的分辨率并可能导致次优收敛或早期训练失败。
2. 💡 为解决此问题，本文提出了Group reward-Decoupled Normalization Policy Optimization (GDPO)，该方法通过解耦对每个单独奖励进行组内归一化，以更好地保留奖励间的相对差异，并通过批次优势归一化提高训练稳定性。
3. 🚀 qwen1.5b 小模型（最大7b），跑了100step（最长550steps）。在工具调用、数学推理和代码推理等多种任务中，GDPO始终优于GRPO一丢丢，展示了其在多奖励RL优化中的有效性和泛化性，并实现了更强的收敛和下游性能。

<img width="970" height="372" alt="image" src="https://github.com/user-attachments/assets/cb3ef74b-20c4-4548-820b-01e79d85573a" />
<img width="959" height="448" alt="image" src="https://github.com/user-attachments/assets/dca2ab47-22b5-48f3-8790-37e3b7affdb0" />
<img width="970" height="372" alt="image" src="https://github.com/user-attachments/assets/e6592f5b-22d8-43fd-a458-cde9eca7c016" />


## MoEBlaze
MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

https://arxiv.org/pdf/2601.05296v1 2026.1.8 Meta, Thinking machine
1. ✨ MoEBlaze提出了一种共设计系统方法，旨在解决MoE训练中由于**token路由和中间激活**引起的显著内存墙瓶颈。
2. ⚙️ 该方法通过端到端token调度和MoE训练，利用优化数据结构**消除中间buffer**，并采**用智能激活检查点的共同设计kernel来降低显存**占用。
3. 🚀 实验结果显示，MoEBlaze与现有MoE框架相比，H100单卡 1个MoElayer。实现了**4倍的训练速度提升和超过50%的内存节省**。

MoEBlaze 是一项旨在解决现代GPU上MoE（Mixture-of-Experts）训练中“内存墙”瓶颈的内存高效MoE训练框架。该框架通过共同设计的系统方法，包括优化的数据结构和智能激活检查点内核，显著减少了激活内存开销并提高了训练吞吐量。
<img width="829" height="376" alt="image" src="https://github.com/user-attachments/assets/6f6cda21-a257-4a08-bbea-70a1282a038b" />

<img width="516" height="469" alt="image" src="https://github.com/user-attachments/assets/8a1cd27f-95f7-4d97-bb74-72e5ec285655" />

<img width="546" height="316" alt="image" src="https://github.com/user-attachments/assets/737d4935-283a-4f43-b9b9-d71113dc2adb" />

<img width="555" height="435" alt="image" src="https://github.com/user-attachments/assets/bcac384e-1cf6-435b-a5e4-7bb7a7aed043" />

<img width="538" height="429" alt="image" src="https://github.com/user-attachments/assets/48fbe28d-3c20-4abe-874e-c95031367b9f" />

<img width="1089" height="507" alt="image" src="https://github.com/user-attachments/assets/1cdeb10e-e84c-4bd8-86b8-63b184faa5ab" />


**1. 问题背景与动机**

传统的计算模式中，处理器吞吐量的增长远超内存带宽和延迟，导致“内存墙”问题。MoE架构虽然通过稀疏激活实现了万亿级参数模型的可管理训练成本，但其固有的架构稀疏性导致计算密度降低，并引入了巨大的激活内存开销，主要体现在两个方面：
**Token Routing Buffers:** 为了路由token，需要大量的辅助缓冲区来压缩和存储激活。例如，一个典型的DeepSeek MoE层，对于 $L \approx 200万$ 个token、$K = 4$ 个活跃专家、$d = 6144$ 的模型维度，使用bfloat16存储routed token buffer，其内存占用高达约94GB。
**Intermediate Activation Storage:** FFN（Feed-Forward Network）计算过程中，尤其是采用SiLU和SwiGLU等现代非线性激活函数时，会产生大量中间激活张量。例如，在DeepSeek配置中，FFN的隐藏维度 $h = 24576$，中间激活内存占用可达约98GB。

这些内存压力限制了GPU上可容纳的最大batch size和sequence length，并导致过多的数据移动，从而阻碍了性能和模型的有效扩展。现有方法，如token dropping或padding，虽然管理了缓冲区，但往往牺牲了模型稳定性。

**2. MoEBlaze的核心贡献**

MoEBlaze通过以下方式解决上述挑战：
**高效的端到端token分派与训练方法：** 显著减少了token路由和激活materializing所需的中间激活缓冲区。该方法避免了填充和token dropping，在不牺牲准确性的前提下减少了内存使用和数据移动，同时提高了计算效率。
**高效的数据结构与算法：** 针对内存高效的计算方案设计，能够有效利用GPU的大规模并行和高带宽，并避免复杂的multi-kernel pipeline。
**与智能激活检查点机制共同设计的训练内核：** 进一步减轻了现代复杂激活函数带来的巨大内存占用，同时在GPU上实现了更好的计算效率。

**3. 核心方法论：内存高效的Token路由与训练算法**

MoEBlaze的核心思想是利用辅助索引列表，在token分派过程中跟踪路由决策，并**即时（on-the-fly）**地进行token访问和结果归约，从而避免了中间激活缓冲区的实例化。

**3.1 前向传播 (Forward Pass)**
**Token Dispatch (Token分派)：** 不为routed tokens创建专门的缓冲区。取而代之的是，基于前置门控（gating）阶段产生的评分，生成几个轻量级的索引数据结构：
    *   `per-expert token list`：跟踪分配给每个专家的token ID。
    *   `per-token expert list`：存储每个token选择的专家ID。
    *   在此阶段不为materialized routed token激活分配或保留内存。
**Expert Computation (专家计算)：** 使用`per-expert token list`中记录的索引，对原始的、未permute的激活张量进行即时收集（on-the-fly gathers）来执行专家MLP计算。为了最大化内存效率，只有两个MLP之间的中间结果（即第一个MLP的输出）被缓冲以用于反向传播。
**Output Aggregation (输出聚合)：** 专家输出被聚合以产生最终的 $(L, d)$ 输出。由于没有存储materialized token dispatch结果的激活缓冲区，聚合操作与第二个MLP计算紧密融合，并直接利用`per-token expert list`进行即时归约（on-the-fly reduction）到输出张量。

**3.2 反向传播 (Backward Pass)**
MoEBlaze的反向传播通过使用相同的逆映射索引，避免了传统方法中将 $(L, d)$ 梯度扩展到 $(L \times k, d)$ “routed gradient tokens”的中间步骤。
**Expert Summation Backward (专家聚合反向传播)：** 利用从分派元数据导出的token-mapping结构，将 $(L, d)$ 梯度张量映射回 $(L \times k, d)$ 的routed gradient tokens。这通过高效的“scatter”操作完成，将输出梯度分散到materialized中间MLP结果张量中的相应位置。
**Expert Computation Backward (专家计算反向传播)：** 梯度流经MLP反向传播。之前checkpointed的两个MLP之间的中间结果在此处用于计算权重梯度。
**Token Gradient Accumulation (Token梯度累积)：** 最后，累积来自所有专家的关于输入token的梯度。此步骤汇总每个token路由到的 $k$ 个专家的贡献，产生最终的 $(L, d)$ 输入激活梯度张量。由于没有激活存储来保存materialized routed token结果，这里也利用token索引数据结构进行即时归约。

**4. 核心方法论：高效且可并行化的分派与数据结构**

MoEBlaze定义并高效构建了以下关键数据结构，以支持上述内存高效的MoE训练算法，同时克服了GPU上常见的多对一映射导致的写竞争问题：

**`expert token indices`**: 一个紧凑的张量，存储分配给每个专家的token索引，这些索引在所有专家之间是连接在一起的。在token-choice MoE训练中，每个token选择 $k$ 个专家，因此 `expert token indices` 的大小为 $L \times k$。此列表是专家检索其指定输入token的基础。
**`expert token offsets`**: 一个长度为 $E+1$ 的数组，存储每个专家token计数的独占前缀和。对于专家 $i$，其分配的token的索引范围从 `expert token offsets[i]` 到 `expert token offsets[i+1] - 1`。
**`token expert indices`**: 基本上是 `expert token indices` 的逆映射。它存储了每个token的路由专家ID，并按token ID排序。其形状也是 $L \times k$。在按专家处理token时，此列表用于对中间materialized结果（例如，两个背靠背MLP之间）进行合并索引。
**`token index map`**: 一个 $L \times k$ 的紧凑张量，存储routed token在 `expert token indices` 列表中的位置。它按原始token ID $i \in L$ 逻辑分组，允许token高效地从中间缓冲区中找到并收集其 $k$ 个专家输出以进行最终组合。

为了高效地构建这些数据结构，MoEBlaze采用了一种三步法，每一步都设计为无原子操作且在GPU上高度并行化，从而避免了昂贵的全局内存访问和复杂的multi-kernel pipeline，解决了传统基于排序的方法（如radix sort）在性能上的瓶颈：

1.  **构建稠密Token-Expert Map (`dense token map`)：**
    *   第一步是构建一个稠密位图 `dense token map` 来编码top-k token到专家的路由。对于每个token $i$，考虑其选择的top-k专家 $\{e_{i,0}, \dots, e_{i,k-1}\}$。对于每个gate slot，将 `dense token map[i, e_{i,k}]` 设置为 $i$。其他条目保持未设置。
    *   `dense token map` 的构建在GPU上高度并行化。通过分配一个 $L \times E$ 的稠密map，并启动内核到CTA网格，每个warp处理一个不相交的token行（$i$）瓦片。由于每个token的专家ID是唯一的，保证了无intra-warp冲突。

2.  **计算专家长度 (`expert lengths`)：**
    *   利用已构建的 `dense token map`，下一步是高效计算每个专家的稀疏token ID列表的长度和偏移。
    *   启动一个自定义内核，CTA网格映射到 `dense token map` 的列（专家）。每个CTA专用于一个专家 $e_i$，并计算该列中非零条目（token到专家分配）的数量。使用warp-level reductions聚合CTA内的行和，产生 `expert lengths` 数组。`expert lengths[e_i]` 表示路由到专家 $e_i$ 的token的最终数量。
    *   计算长度后，通过在初始计数内核之外对 `expert lengths` 数组应用前缀和来导出 `expert offsets`。

3.  **路由索引到门控 (`Route Indices to Gates`)：**
    *   第三步是生成 `per-expert token id list expert token indices`，作为后续MLP计算的输入。为了在GPU上以无竞争的方式实现索引的紧凑、按专家连接，采用了一个两阶段过程，围绕生成一个 `location map`。
    *   `location map` 指定了 `dense token map` 中每个非零条目在 `expert token indices` 列表中的最终目标位置ID。
    *   一旦 `location map` 构建完成，一个简单的并行内核直接从 `dense token map` 读取元素，并将它们写入 `expert token indices` 中计算出的相应位置，确保完全并行且无原子操作。
    *   `location map` 的构建通过两步策略实现无原子操作：
        *   **tile-level scan:** 每个CTA处理一个专家。同一CTA内的线程处理分配给该专家的连续token在 `dense token map` 中。它们首先在共享内存中计算tile-level计数，然后执行CTA内部的独占扫描操作（前缀和）。
        *   **global offset addition:** 结果的CTA局部独占扫描计数与专家的预计算全局 `expert offsets` 相加。这个加法产生连接索引数组中正确、最终的位置ID。

**5. 核心方法论：端到端效率的训练内核协同设计**

MoEBlaze对MoE训练内核及其底层GPU内核进行了联合优化，以解决与某些高级激活方法相关的内存问题，尤其是SwiGLU。

**5.1 SwiGLU MoE与内存瓶颈**
SwiGLU定义为 $\text{SwiGLU}(x; W_1, W_2) = \text{SiLU}(xW_1) \cdot (xW_2)$，其中 $\text{SiLU}(u) = u \cdot \sigma(u)$。对于MoE层，这会引入两个投影：$a = xW_1$ 和 $b = xW_2$，以及后续的元素级操作 $\text{SiLU}(a)$ 和最终乘积 $\text{SiLU}(a) \odot b$。在传统内核中，需要materialize多个中间结果（$a$, $b$, $\sigma(a)$, $\text{SiLU}(a)$），并进行全局内存读写，这在模型和batch size扩大时成为显著瓶颈。

**5.2 激活检查点与内核协同设计**
MoEBlaze的优化基于以下观察：
*   **激活函数计算受内存带宽限制：** 激活函数计算主要是逐点操作，GPU对这类操作效率很高。在LLM训练中，$L \gg d$（token数量远大于嵌入维度），这种“高瘦”矩阵上的操作通常受内存带宽限制。
*   **激活函数的内存占用显著：** 即使计算量小，复杂激活函数也需要materialize并保存大量中间结果用于反向传播，导致内存分配与batch size、sequence length和FFN维度呈线性关系，在万亿token训练环境中成本高昂。

基于此，MoEBlaze提出了**联合激活检查点和内核融合**方法：
*   **内核融合：** 将SwiGLU中两个第一层投影和激活epilogue融合到一个单独的内核中。该内核消费非materialized的routed tokens，只加载输入 $x$ 一次，同时通过两个 $(W_1, W_2)$ GEMM流式传输，在寄存器/共享内存中计算 $\text{SiLU}(a)$，并立即执行与 $b$ 的乘法，只将最终输出写入全局内存。这种“epilogue融合”消除了 $a$, $b$ 的全局写入和后续的元素级操作重新读取，将计算从内存受限领域尽可能转移到计算受限领域。它还将 $x$ 的输入读取量减半。
*   **反向传播中的梯度聚合：** 在反向传播中，两个第一层投影的融合意味着来自两条路径的关于共享输入 $x$ 的梯度必须聚合。MoEBlaze的实现通过分块归约（tiled reductions）在原地（in-place）计算两个分支的激活导数并聚合梯度，完全消除了临时全局缓冲区。
*   **激活检查点：** 进一步应用激活检查点策略——在正向传播时跳过保存SwiGLU中间结果（$\text{SiLU}(a)$）。相反，在反向传播时采用重计算策略，利用 $\text{SiLU}$ 函数计算开销低的特点。

**6. 实验结果**

MoEBlaze在NVIDIA H100 GPU上与Megablocks进行了比较，涵盖了ReLU和SwiGLU激活函数下的7种MoE配置。

**内存效率（SiLU）：** MoEBlaze始终显著降低了激活内存消耗。在conf4（Dinput = 2048, E = 16, L = 1024, B = 32）配置下，MoEBlaze仅**需6,100 MB内存，相比Megablocks的22,000 MB减少了近3.6倍**。这种显著减少归因于**内存高效的token分派机制和智能重计算**（激活检查点）。
**训练速度（SiLU）：** MoEBlaze实现了1.4倍到3.7倍的显著性能提升。最大加速在conf4实现，表明MoEBlaze在大模型维度下扩展性良好。这得益于优化的token分派实现、高效的数据分派构造内核（避免了昂贵的排序和多核流水线）以及针对H100硬件优化的融合batched-GEMM内核。
**内存效率（SwiGLU）：** 即使在内存需求更高的SwiGLU激活函数下，MoEBlaze仍保持了显著的内存优势，峰值激活内存通常不到Megablocks的一半。例如，在conf3，Megablocks需要超过40,000 MB，而MoEBlaze控制在约10,000 MB，实现了4倍的内存减少。这证实了MoEBlaze的内存高效分派和智能重计算方案对于复杂激活函数也高度有效。
**训练速度（SwiGLU）：** 相比ReLU结果，SwiGLU下的加速因子通常更高且更一致，范围从2倍到6.2倍。这归因于SwiGLU更复杂的计算为MoEBlaze高度融合的内核提供了更大的优势，并且内存带宽节省在SwiGLU中更为关键，因为中间激活大小更大、更复杂。


## DRQ
Digital Red Queen: Adversarial Program Evolution in Core War with LLMs 

https://arxiv.org/abs/2601.03335 2026.1.6 SakanaAI

https://github.com/SakanaAI/drq/

中文解读：https://mp.weixin.qq.com/s/bf9E9RS7WwsAOKZ-GneDXg

<img width="860" height="604" alt="image" src="https://github.com/user-attachments/assets/e067bd3d-e7e7-4151-9ac5-17eb1ada62d0" />

该论文介绍了 Digital Red Queen (DRQ)，一种利用 LLM 在Core War游戏中演化 assembly 程序（称为 warrior）的简单 self-play 算法。Core War是一个 Turing-complete的沙盒环境，LLM演化的warrior在其中竞争虚拟机的控制权。
与传统的静态优化问题不同，DRQ 旨在通过对不断变化的目标进行持续适应来模拟现实世界中对抗性演化的 Red Queen 动态。

实验中观察到独立运行的多个并发DRQ实验（每个实验都从不同的战士开始初始化）会随时间推移，慢慢趋向于演化出具有相似行为的战士。值得注意的是，这种趋同并没有发生在源代码层面，这表明趋同的是「功能」而非「实现」。这一结果让人联想到生物学中的趋同进化 —— 即相似的功能特征通过不同的机制独立进化了多次。例如，鸟类和蝙蝠各自独立进化出了翅膀；蜘蛛和蛇独立进化出了毒液。尽管基础版 DRQ 算法本身较为简单，但它在《Core War》中表现出乎意料得好，这表明：即便是最简单的自对弈循环，也能揭示出复杂且鲁棒的策略。这使得 DRQ 成为探索其他竞争性多智能体仿真（如人工生命、生物学、药物设计、现实世界网络安全或市场生态系统）的有力候选方案。

1.  Digital Red Queen (DRQ) 是一种利用大型语言模型（LLMs）在 Core War 游戏中**演化对抗性程序**（warriors）的**自对弈算法**，通过持续适应不断变化的对手来模拟开放式“Red Queen”动态。
2. 该算法通过在每轮中演化出新的 warrior 以击败所有历史对手来驱动适应性进化，并在内部使用 Quality-Diversity (MAP-Elites) 算法确保程序多样性。
3. 实验结果表明，DRQ 能够生成对抗性更强的通用warrior，并观察到独立运行中行为的统计收敛（phenotypic convergence），而代码层面（genotypic）仍保持多样性，这与生物学的**趋同进化现象类似**。


**Core War 游戏机制**

Core War 是一款经典编程游戏，其中低级 assembly-like 程序（warrior）在共享的虚拟计算机中竞争。游戏的内存（Core）是一个固定大小的循环数组（通常 8,000 个单元），每个单元包含一个指令。warrior 的原始 assembly 代码被放置在内存中的随机位置，虚拟机会逐行执行它们的指令。由于代码和数据共享同一地址空间，因此 self-modifying logic 很常见，创建了一个高度不稳定的环境。每个 warrior 的目标是通过使对手程序崩溃来成为最后一个运行的程序，同时确保自身生存。常见的策略包括 bombing（在 Core 中放置 DAT 指令以终止对手进程）、replication（将自身代码复制到多个内存位置）和 scanning（探测 Core 以定位敌人）。

**DRQ 方法论**

DRQ 算法基于 self-play 和 coevolutionary training。其核心思想是让 LLM 演化新的 warrior，使其能够击败所有之前演化出的 warrior，从而形成一个不断适应的 warrior 序列。

1.  **Initialization:** 从一个基础 warrior $w_0$ 开始，可以是人类设计或 LLM 生成。
2.  **Adversarial Optimization:** 在第 $t$ 轮，演化一个新的 warrior $w_t$，使其在包含所有先前 warrior $\{w_0, \dots, w_{t-1}\}$ 的环境中最大化其预期 fitness：
    $$w_t = \arg \max_w \mathbb{E}[\text{Fitness}(w; \{w_0, \dots, w_{t-1}\})]$$
    其中期望值针对不同的评估 seeds。旧的 warrior 在 lineage 中不更新，以促进稳定性。
3.  **Iteration:** 重复 $T$ 轮，生成 warrior 的 lineage $\{w_0, w_1, \dots, w_T\}$。

**Intra-round Optimization with MAP-Elites:**
由于程序合成的搜索空间具有高度欺骗性，DRQ 在每一轮的内部优化循环中采用了 MAP-Elites 算法。MAP-Elites 是一种 quality-diversity 算法，通过将用户定义的 behavioral descriptor 空间离散化为 cells，并在每个 cell 中存储一个 elite solution，从而防止多样性崩溃。这使得算法能够维护一个广泛的 stepping stones 集合，以发现行为空间中不同区域的强大策略。
Fitness 函数定义为：
$$\text{Fitness}(w_i; \{w_j\}_{j \neq i}) = \sum_{\tau=1}^{T} \frac{N}{T} \frac{A_i^{\tau}}{\sum_j A_j^{\tau}}$$
其中 $A_i^{\tau}$ 是指示 warrior $i$ 在 simulation timestep $\tau$ 是否存活的 indicator function，$N$ 是 warrior 数量，$T$ 是 simulation timesteps。这个 fitness 函数激励 warrior 尽可能长时间地生存，并通过淘汰其他 warrior 来增加自己的份额。
Behavioral descriptor 函数 BD(·) 被定义为离散化的元组（total spawned processes, total memory coverage），用于捕获 warrior 在 simulation 期间的两个高级策略方面。网格在 log 空间中进行离散化。

**LLMs as the Mutation Operator:**
LLM（具体使用了 GPT-4.1 mini）用于生成新的 warrior 和对现有 warrior 进行 mutation。LLM 接收一个系统 prompt，描述 Core War 环境和 Redcode 汇编语言手册（包括 opcodes、addressing modes 和示例 warrior）。生成新 warrior 时，LLM 被指示生成新的 Redcode 程序。进行 mutation 时，LLM 会收到原始程序并被指示进行修改以改进性能。论文特意选择了 LLM 的简单用法，以将研究重点放在 Core War 和演化分析上。

**实验与结果**

1.  **Static Target Optimization Against Human Warriors:**
    作为基线，论文评估了针对 294 个真实人类 warrior 进行单轮 DRQ 的静态优化效果。LLM zero-shot 仅能击败 1.7% 的人类 warrior。通过 best-of-$N$ 采样，可以集体击败 22.1% 的人类 warrior。然而，针对每个人类 warrior 进行演化优化，可以产生 specialist warrior，集体击败 89.1% 或平局 96.3% 的人类 warrior。这表明演化可以显著提高性能。但这些 specialist warrior 缺乏鲁棒性，平均每个 warrior 只能击败或平局 27.9% 的人类 warrior，表明它们过度拟合了训练对手。

2.  **Iterative Red Queen Dynamics:**
    为了研究多轮 DRQ 的动态，论文在 96 个不同的 Core War 对局中进行了实验，并分析了 history length $K$（即每轮优化时考虑的先前 warrior 数量）的影响。
    *   **Phenotype Generality (表型通用性):** 随着 DRQ 轮次增加，warrior 的平均 generality 持续增长，即其击败或平局 unseen human warrior 的比例增加（Slope = 3.466, $R^2$ = 0.927），表明 DRQ 能够发现更鲁棒的 warrior。
    *   **Phenotype Convergence (表型收敛):** 跨独立 DRQ runs 的 warrior phenotype（对 unseen human warrior 的 fitness 值向量）的方差随着轮次减少（Slope = -0.003, $R^2$ = 0.573），表明在不同初始条件下存在向通用行为的收敛。同时，单个 run 内 phenotype 的变化率也随轮次降低（Slope = -0.006, $R^2$ = 0.756），表明趋于稳定。
    *   **Genotype Non-Convergence (基因型非收敛):** 与 phenotype 相反，genotype（源代码的 text embedding）的方差在多轮中保持大致不变（Slope = $2.10 \times 10^{-7}$, $R^2$ = 0.057），表明 DRQ 没有收敛到单一的实现方式。这种表型和基因型的分离类似于生物学中的 convergent evolution。

3.  **Cyclic Dynamics:**
    通过将 history length $K$ 从 1 增加到 10，DRQ 中 cycle（例如 rock–paper–scissors 动态）的总数减少了 77%。这与先前的研究一致，即在 self-play 中纳入历史对手可以减少循环行为。

4.  **What Makes a Good Core War Warrior?**
    通过分析 MAP-Elites archive，发现生成大量 spawned threads 的 warrior 性能最佳，因为终止它们需要停止所有 threads。对于生成较少 threads 的程序，最大化 memory coverage 是一种有效的替代策略。论文展示了两个 DRQ 演化出的 warrior 示例，"Ring Warrior Enhanced v9" 和 "Spiral Bomber Optimized v22"，它们展示了合成不同策略和生成高性能 warrior 的能力。

5.  **Does MAP-Elites Matter?**
    将 MAP-Elites 替换为 single-cell variant（移除 diversity preservation 机制）导致每一轮的优化性能显著下降，尤其是在后期轮次，突出了在 Core War 程序合成中保持多样性的重要性。

6.  **Is Fitness Predictable?**
    使用 OpenAI 的 text embedding 模型（text-embedding-3-small 和 text-embedding-3-large）将 warrior 的 Redcode 源代码嵌入，然后训练一个 linear probe 来预测 warrior 的 generality score。结果显示，generality 可以中等程度地从源代码嵌入中预测（test $R^2$ = 0.442 for small model, $R^2$ = 0.461 for large model）。这表明即使在高度混沌的 Core War 环境中，LLM 也能捕捉到源代码与其性能之间的复杂映射关系，为未来通过 surrogate model 绕过模拟提供了可能性。

**结论**

DRQ 展示了 LLM 驱动的对抗性程序演化如何在 Core War 这样丰富的 testbed 中产生鲁棒策略。通过针对不断增长的对手历史进行演化，DRQ 促进了策略的鲁棒性，并展示了跨独立 runs 的收敛性，这与生物学中的 convergent evolution 现象相似。这强调了从静态目标转向动态 Red Queen 目标的重要性。该工作将 Core War 定位为研究人工智能系统中对抗性适应的受控沙盒，并评估 LLM-based 演化方法的有效性。DRQ 算法的简洁性和有效性表明，类似的 minimalist self-play 方法可以在其他多 agent 对抗领域（如现实世界 cybersecurity 或药物耐药性研究）中发挥作用，以在受控环境中系统地探索潜在风险。

## GPU_ext
GPU_ext: Extensible OS Policies for GPUs via eBPF

https://arxiv.org/pdf/2512.12615 2025.12.20

1. 💡 `gpu_ext` 提出了一种基于 eBPF 的运行时，将 GPU 驱动和设备视为可编程的操作系统子系统，以克服传统 GPU 资源管理策略在灵活性和性能上的不足。
2. ⚙️ 该系统通过在 GPU 驱动中暴露安全的可编程 hook，并在 GPU 内核中引入设备端 eBPF 运行时来执行经过验证的策略逻辑，同时采用 SIMT-aware 验证和跨层 eBPF map 解决主机-设备异构性。
3. 🚀 实验结果表明，`gpu_ext` 支持的策略能够将吞吐量提升高达 4.8 倍(GPT-oss-120b FP4, 5090)，将尾延迟降低多达 2 倍，且对应用程序无侵入，证实了其在 GPU 资源管理中的有效性。

   
## KernelEvolve
KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

https://arxiv.org/pdf/2512.23236 2025.12.31 Meta

1. 🚀 KernelEvolve 是一个代理式内核编码框架，旨在解决 Meta **DLRM 训练和推理中模型、内核及**硬件异构带来的复杂优化挑战。
2. 💡 该系统通过图搜索、动态提示合成和包含硬件特定约束的持久知识库，自动化了 Triton 等编程抽象下的内核生成和优化，尤其支持了如 MTIA v3 等专有架构。
3. ⚡ KernelEvolve 在生产环境中已部署，不仅将开发时间从**数周缩短至数小时**，还在 **NVIDIA、AMD 和 MTIA 平台上实现了 PyTorch 基线高达 17 倍的**性能提升，有效缓解了新 AI 硬件的可编程性障碍。

KernelEvolve 是 Meta 推出的一种基于 Agent 的内核编码框架，旨在解决深度学习推荐模型 (DLRM) 训练和推理中存在的模型架构多样性、内核原语多样性以及硬件代际和架构异构性所带来的复杂优化难题。

**核心问题与挑战：**
现代广告系统面临三个维度上的复杂性，手动内核开发难以应对：
1.  **硬件多样性：** 涵盖 NVIDIA GPU、AMD GPU 和 Meta 自研的 MTIA 芯片。每种硬件都有独特的内存层次结构（如 NVIDIA 的 L2 Cache、AMD 的 Infinity Cache、MTIA 的片上 SRAM）、编程模型（CUDA、Triton、ROCm/HIP、CuTe、MTIA C++ DSL、TLX 等）和代际架构特性（如 NVIDIA Hopper 的 TMA、128 线程 Warp-Group），导致代码不直接移植，需针对性优化，且优化工作量巨大（每个平台需数周专家时间）。
2.  **模型多样性：** 多阶段推荐流水线（检索、早期排序、晚期排序）的模型复杂度差异巨大。Transformer-based 模型（如 InterFormer、Wukong）复杂度增加了 10-100 倍，且大型 Embedding Tables (100GB+) 对内存带宽构成压力。不同阶段需要不同的内核优化策略。
3.  **内核多样性：** 除了密集的矩阵乘法 (GEMM) 之外，广告排序模型还包含 200 多个数据预处理操作（如特征派生、密集归一化、稀疏归一化）。这些操作尽管算术强度低，但若缺乏在 AI 加速器上的原生实现，将强制采用分层服务架构（CPU 处理预处理，加速器处理神经网络），引入显著的网络延迟（10-20ms），增加 TCO，并阻碍新模型部署。因此，预处理内核覆盖率成为首要架构需求。

现有 AI 驱动的内核生成系统大多停留在研究原型阶段，存在优化范围窄、评估合成化、仅关注单一平台、Agent 能力有限、缺乏推理时扩展能力和检查点支持等局限性。

**KernelEvolve 的核心方法论：**
KernelEvolve 将内核优化视为一个图（或树）搜索问题，通过一个自改进的状态机来探索优化空间。其核心在于“通用操作符”设计、Agentic 的检索与自管理上下文机制，以及全面的评估与工具链。

**实验结果与案例研究：**
KernelEvolve 在 KernelBench 套件上所有 250 个问题（三个难度级别）和 160 个 PyTorch ATen 操作符（横跨三个硬件平台）上均实现了 100% 的通过率和正确性。在生产用例中，实现了 1.2 倍到 17 倍的性能提升，并将开发时间从数周缩短到数小时。

*   **卷积 Transformer (1D 卷积)：** 在 NVIDIA H100 上，FP16 精度下，针对生产形状 (2048, 96, 96, 200)，KernelEvolve 实现了相对于 PyTorch conv1d 基线 2.30 倍的 Speedup，相对于 conv2d 优化基线 1.62 倍的 Speedup。主要通过内核融合消除内存布局转换和中间张量物化来实现，结合 autotuning、双缓冲执行和缓存修饰符等架构优化。搜索轨迹显示，Agentic 搜索能系统性地发现高效实现。
*   **异构硬件上的卷积：** 对比 NVIDIA H100/A100、AMD MI300/MI350 和 MTIA v3 平台上的 conv1d 内核。KernelEvolve 在所有平台均实现对 conv1d 基线的 Speedup (1.75x - 6.54x)，其中 MTIA v3 表现最佳（6.54x），说明其能有效针对厂商库覆盖度不高的定制加速器。通过形状感知的调度，确保在目标工作负载上获得性能提升，同时保留对供应商库的降级路径。
*   **Wukong 和 InterFormer 中的内核融合：**
    *   **Optimized FM (Wukong)：** 针对 Factorization Machine 中的计算 $X \cdot (X^T Y)$，KernelEvolve 将两次 Batched Matrix Multiplication 融合，消除中间结果的 HBM 往返，并通过针对生产形状的定制 Tile 策略确保 SRAM 常驻。在生产形状上实现了 2-4 倍的 Speedup，尤其是在 N ≤ 64 的情况下。
    *   **PFFN (InterFormer)：** 融合了前馈网络、GELU 激活和 RMSNorm 等操作。KernelEvolve 生成的单通道内核，一次加载 Tile，在 SRAM 中完成所有操作链，仅一次写入 HBM，相比 PyTorch 基线多次加载和中间写入，显著减少内存流量。在生产形状上实现 1.2-2.0 倍的 Speedup，即使在批处理量增大时仍保持稳定增益。

**结论：**
KernelEvolve 是一个生产级的 AI 驱动内核优化系统，通过 Agentic 方法、图搜索、检索增强的上下文管理、自适应的通用操作符以及全面的评估工具链，成功实现了大规模 DLRM 在异构 AI 加速器上的高性能内核生成与优化。它不仅在通用操作符上实现了高正确性，还在复杂的生产用例中展现了显著的性能提升，并降低了新硬件的编程门槛，加速了 AI 系统的部署和创新。

## ClusterFusion
ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive 
https://arxiv.org/abs/2508.18850 2025.8.26 上海交大 DSM，H100单卡

https://github.com/xinhao-luo/ClusterFusion 


## ToolOrchestra
ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration 

paper: https://arxiv.org/abs/2511.21689 NVIDIA, 港大 2025.11.26

code:  https://github.com/NVlabs/ToolOrchestra
<img width="1033" height="337" alt="image" src="https://github.com/user-attachments/assets/35e89638-c42f-48d7-bc31-6c2bbe0199ad" />
<img width="1049" height="464" alt="image" src="https://github.com/user-attachments/assets/807bcce5-e284-423f-a631-13521a49df10" />

1.  ⚙️ 针对大型语言模型在处理复杂任务时面临的效率和成本挑战，本文引入ToolOrchestra方法，通过训练一个**小型编排模型8b来高效协调多种智能工具和模型**。
2.  🚀 该方法通过强化学习端到端地**训练这个8B参数的编排模型**，其奖励设计综合考虑了任务结果的准确性、资源使用效率和用户工具偏好。
3.  💡 实验证明，其训练出的Orchestrator模型在HLE、𝜏2-Bench和FRAMES等基准测试上**超越了GPT-5等前沿模型，以显著更低的成本**实现了更高的准确率，并能稳健泛化到未见工具。

<img width="440" height="426" alt="image" src="https://github.com/user-attachments/assets/31a4c1f5-1bd7-46f4-9557-ee310acead6e" />
<img width="1052" height="380" alt="image" src="https://github.com/user-attachments/assets/e74fb31d-1b25-4957-8d69-c0b079ca2ae8" />
<img width="1052" height="358" alt="image" src="https://github.com/user-attachments/assets/58d6c541-3756-46f8-8c2f-43ce3ae3f084" />
本文介绍了ToolOrchestra，一种通过训练小型编排模型（orchestrator model）来高效协调多样化模型和工具，以解决复杂agentic任务的方法。尽管大型语言模型（LLMs）能力强大，但在处理如“人类终极考试”（Humanity's Last Exam, HLE）等深层次复杂问题时，仍面临概念性和计算成本高的挑战。ToolOrchestra旨在通过一个轻量级编排器来管理其他智能工具和模型，从而提高智能上限并提升效率。

**核心方法（ToolOrchestra）**

ToolOrchestra通过强化学习（RL）端到端地训练一个小型语言模型（例如8B参数），使其作为异构工具使用agent的“大脑”，动态地选择和利用各种外部工具。

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

## Jisi
Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale

paper: https://arxiv.org/abs/2601.01330v1 上海AI Lab；港中文等。2026.1.4

code:  https://github.com/magent4aci/openJiSi

中文概述：https://mp.weixin.qq.com/s/79moEaUM1c0DQr30t-f7lg 

<img width="1018" height="428" alt="image" src="https://github.com/user-attachments/assets/0356ae71-b906-4220-88ce-79e6d68b131b" />
<img width="584" height="111" alt="image" src="https://github.com/user-attachments/assets/63c11dd4-3479-4349-a7af-70d184977af2" />
<img width="950" height="482" alt="image" src="https://github.com/user-attachments/assets/5a455045-b547-4be3-8848-ff9cfc72199e" />


1. 💡针对现有LLM **Routing仅依赖查询相似性、Aggregation方法静态以及Routing与Aggregation互补性不足**的瓶颈，本文提出了JiSi框架。
2. ⚙️JiSi框架引入了Query-Response Mixed Routing、Support-Set-based Aggregator Selection和Adaptive Routing-Aggregation Switch三大创新，以更有效地编排开源LLM的协作。
3. 🏆在九个基准测试中，JiSi通过**编排十个开源LLM，仅用47%的成本就超越了Gemini-3-Pro，平均性能提升1.15%**，并显著优于主流基线，展示了集体智慧的巨大潜力。
本文提出了一种名为 JiSi 的新型框架，旨在通过协调开源大型语言模型（LLMs）的协作，超越包括 Gemini-3-Pro 在内的领先闭源模型，探索集体智能作为单体模型无限扩展的替代路径。

**1. 现有路由和聚合方法的瓶颈：**
作者首先指出当前多 LLM 协作方法存在以下三个主要瓶颈：
*   **训练自由（training-free）路由器受限：** 现有路由器主要基于查询（query-based）范式，仅侧重于文本相似性，无法捕捉深层语义或问题难度。
*   **聚合方法静态且非自适应：** 近期聚合方法大多是静态的，难以针对不同任务自适应选择合适的聚合器（aggregator），导致在关键响应生成时受限。
*   **路由和聚合的互补性未充分利用：** 路由提供了稳定性但受限于单一模型，聚合可以超越个体模型限制但易受低质量输出和噪声影响，二者之间的互补性未被有效协同。
<img width="1002" height="397" alt="image" src="https://github.com/user-attachments/assets/c348d085-2fd4-4b37-818f-3c90b712e6b5" />
<img width="1015" height="416" alt="image" src="https://github.com/user-attachments/assets/372b0b46-3f22-47d1-a1cb-78525dbd87be" />
<img width="999" height="383" alt="image" src="https://github.com/user-attachments/assets/d656f9b9-190f-4126-a161-ff64ed90b85e" />
<img width="1025" height="505" alt="image" src="https://github.com/user-attachments/assets/8268c11e-5e1d-4ade-985e-91adc7202335" />

**2. JiSi 框架的核心创新：**
为解决上述瓶颈，JiSi 引入了三项核心创新，基于一个预构建的嵌入库（embedding bank）：

*   **2.1. Query-Response Mixed Routing（查询-响应混合路由）:**
    *   **问题：** 传统的基于查询嵌入的路由方法，仅捕获查询文本的表面相似性，忽略了深层语义和任务难度。例如，“证明每个大偶数都是两个素数之和”和“证明每个大偶数都是两个奇数之和”在词语上高度相似，但难度天壤之别，查询嵌入难以区分。
    *   **解决方案：** JiSi 利用 LLM 生成的响应嵌入和令牌消耗（token costs）来优化路由过程。
        *   **响应嵌入：** LLM 生成的响应包含潜在的语义信息，可以帮助区分语义相似但难度或深层含义不同的问题。例如，对于上述两个问题，LLM 可能会生成截然不同的响应，从而使得嵌入模型能够有效区分它们。
        *   **令牌消耗：** 难度更大的问题通常需要 LLM 消耗更多的令牌来生成响应。JiSi 将令牌消耗作为衡量问题难度的一种直观启发式指标。
    *   **技术细节：** 通过加权整合查询、响应和令牌消耗的相似性，JiSi 实现了更精确的模型路由。在模型推理时，对给定查询 $x$，首先通过候选 LLM 生成响应 $\hat{y}_k$ 和相应的推理成本 $c_k$。接着，将查询和响应映射到嵌入空间：$e = f^1(x)$ 和 $r_k = f^1(\hat{y}_k)$。
        *   用于过滤支持集的相似度得分 $s_{flt}$ 综合了查询相似度 $s$、响应相似度 $s_{res}$ 和成本相似度 $s_{cost}$：
            $s_{flt} = \epsilon s + \sigma s_{res} + \delta s_{cost}$
            其中，$s_{res,i} = \sum_{j=1}^K r_{i,j}^T r_j$，表示第 $i$ 个训练集问题下所有候选 LLM 响应与当前查询响应的相似度总和；$s_{cost,i} = \sum_{j=1}^K (1 - |c_{i,j}^2 - c_j^2| / c_{m,j})$，表示第 $i$ 个训练集问题下所有候选 LLM 响应成本与当前查询响应成本的相似度总和，其中 $c_{m,j}$ 是标准化因子。$\epsilon, \sigma, \delta$ 是预定义的权重，且 $\epsilon + \sigma + \delta = 1$。
        *   这种混合相似度用于计算细粒度模型得分 $g_f$，使得路由能更准确地匹配查询与 LLM 的能力。

*   **2.2. Support-Set-based Aggregator Selection（基于支持集的聚合器选择）:**
    *   **问题：** 现有聚合方法通常静态地选择聚合器，基于其整体性能。然而，一个聚合器在通用能力上表现出色，不代表其在特定领域也具有最佳聚合能力。理想的聚合器应在领域特定能力和综合聚合能力之间取得平衡。
    *   **解决方案：** JiSi 通过结合大规模嵌入支持集中的先验分数，动态选择合适的聚合器。
    *   **技术细节：** 对于给定查询 $x$，首先从嵌入库中选择一个支持问题集。该支持集的选择基于查询 $x$ 的嵌入 $e = f^1(x)$ 与嵌入库中所有问题嵌入 $E_Q$ 的余弦距离 $s$。
        *   通过预定义的参数 $N_{base}$ 和 $\gamma$，确定支持集中的问题数量 $N_{sup}$ 和具体索引 $I$：$I = \{i \in [N] | s_i \ge \gamma \cdot f^2_{N_{base}}(s)\}$。
        *   利用支持集 $I$ 中的问题及其对应的 LLM 正确性向量 $v_j$，计算粗粒度模型得分 $g \in \mathbb{R}^M_+$，其中 $g = v_{sup} s_{sup}$，$v_{sup}$ 是 LLM 在支持集上的正确性。
        *   最终，选择得分最高的 LLM 作为聚合器 $A_{agg} = A_k$，其中 $k$ 是 $g$ 中最大值的索引。这种方法确保了聚合器在更广泛的查询集上表现出良好的综合能力，同时也能兼顾不同领域的聚合需求。

*   **2.3. Adaptive Routing-Aggregation Switch（自适应路由-聚合切换）:**
    *   **问题：** 路由和聚合各有利弊。路由稳定但受限于单一模型性能上限；聚合能超越个体限制但易受低质量或噪声输入的干扰。简单地将二者结合可能无法有效利用各自优势。
    *   **解决方案：** JiSi 引入了一个自适应切换机制，根据模型响应的先验得分动态选择输出策略。
    *   **技术细节：** 在路由阶段，基于预构建的嵌入库，为每个候选响应计算精确的先验得分（即上面提到的细粒度模型得分 $g_f$）。
        *   通过预设的阈值 $t$，确定最终的聚合器（aggregatees）集合 $A_{fm}$：
            $A_{fm} = \left\{A_i \in A \mid i \in [M], g_{f,i} \ge [f^2(K)](g_f) \text{ and } \left(\frac{g_{f,i}}{[f^2(1)](g_f)}\right) \ge t\right\}$
            其中，$[f^2(K)](g_f)$ 表示 $g_f$ 向量中第 $K$ 大的值，确保至少选择 $K$ 个模型作为候选，然后通过阈值 $t$ 进一步筛选。
        *   **动态策略：**
            *   如果 $N_{fm} = 1$（只有一个响应的得分超过阈值），系统将跳过聚合过程，直接采用该单一响应作为最终输出（即退化为 Top-1 路由）。这显著提高了推理效率。
            *   如果 $N_{fm} > 1$（多个响应的得分超过阈值），则执行聚合操作。此时，由 $A_{agg}$ 对集合 $A_{fm}$ 中的 LLM 响应进行聚合：$\hat{y} = A_{agg}(\bar{x})$，其中 $\bar{x} = f^3(\{A_i(x) \mid A_i \in A_{fm}\})$ 是聚合器所需的所有响应的连接。
        *   这种机制能够剪除低质量或噪声响应，保证在触发聚合时其输入质量高，从而实现高效且高质量的决策。

**3. 嵌入库的构建：**
嵌入库是 JiSi 的基础，它存储了 LLM 的能力画像。它通过收集来自异构来源的大量查询，并获取候选 LLM 对这些查询的响应，包括响应文本、令牌消耗和正确性信息。所有查询和响应都通过预训练的嵌入模型 $A_{emb}$ 映射到连续嵌入空间：
$E = E_Q \cup E_R$
$E_Q = \{e_i = f^1(x_i) \mid i \in [N]\}$
$E_R = \{r_{i,j} = f^1(\hat{y}_{i,j}) \mid i \in [N], j \in [M]\}$
其中 $f^1(a) = A_{emb}(a) / \|A_{emb}(a)\|_2$。此外，每个 LLM $A_j$ 都对应一个能力向量 $v_j \in \{0, 1\}^N$，用于衡量其在所有问题上的正确性。

**4. 实验结果：**
*   **数据集和模型：** 在 OpenRouterBench 上使用九个基准测试集，十个开源 LLM（M=10）作为候选模型，并与七个闭源 LLM 和多种主流的路由及多智能体基线方法进行比较。
*   **性能卓越：** JiSi 在所有闭源和开源 LLM、路由器方法以及多智能体方法中平均性能排名第一。相较于最强大的开源 LLM (DeepSeek-V3.2-Speciale)，性能提升了 6.56%。甚至比最先进的闭源 LLM Gemini-3-Pro 性能提高了 1.15%。
*   **成本效率：** 尽管 JiSi 涉及多智能体聚合，但与领先的闭源 LLM 相比，它显著降低了总成本，同时保持了卓越的性能。例如，JiSi 相比 Gemini-3-Pro 节省了 53.23% 的成本。这得益于开源 LLM API 价格更低以及自适应路由-聚合切换机制减少了聚合工作量。
*   **可扩展性：** JiSi 表现出强大的可扩展性，随着候选 LLM 数量的增加，在推理与知识、编码与工程、通用聊天与事实性三大领域均展现出稳定且持续的性能提升，证实了其作为动态系统的潜力。
*   **LLM 选择分布：** JiSi 倾向于选择高性能 LLM，同时保持任务感知的专业化和丰富的使用多样性。例如，在推理密集型任务中偏爱 DeepSeek-V3.2-Speciale，而在事实性问答中偏向 Qwen3-235B-A22B。同时，它避免了系统过度依赖单一 LLM 的问题，保证了聚合的互补性。


## CLO
https://arxiv.org/pdf/2511.14510 中科大 华为等 2025.11.18

https://github.com/CommediaJW/CLO
1. 🚀 CLO提出了一种算法-系统协同设计的方法，旨在解决现有LLM KVCache卸载系统中存在的CPU瓶颈，包括细粒度缓存管理开销、PCIe带宽利用率低以及CPU中心化同步导致的GPU停滞。
2. 💡 该系统通过引入粗粒度、头粒度的近似GPU缓存策略、零拷贝传输引擎和GPU中心化同步机制来优化KVCache管理和数据传输。
3. ⚡ 实验结果表明，CLO在保持与SOTA系统相当的准确性下，显著降低了CPU开销，充分利用了PCIe带宽，并将解码吞吐量提高了9.3%至66.6%。

## RetrievalAttn
RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval

https://chenc10.github.io/assets/pdf/2025_neurips_retrievalattention.pdf 微软 NIPS25

https://github.com/microsoft/RetrievalAttention 

1. 🚀 RetrievalAttention是一种训练无关方法，通过为固定上下文预构建KV向量索引并将其存储在CPU内存中以供高效检索，从而加速长上下文LLM推理并减少GPU内存消耗。
2. 💡 该方法创新性地将近似最近邻搜索 (ANNS) 集成到注意力计算中，并提出了一种“**注意力感知向量索引**”来解决查询 (Q) 和键 (K) 向量之间的分布不一致 (OOD) 问题。
3. ⚡ 实验结果表明，RetrievalAttention在保持接近全注意力精度的同时，仅访问**1-3%的数据**，显著降低了推理成本，使得8B参数的LLM在单张RTX4090 (24GB) 上能处理128K tokens，解码速度达到0.107秒/token。

RetrievalAttention是一项旨在加速长上下文大型语言模型（LLM）推理并降低GPU内存消耗的无训练方法。该方法通过为固定上下文预构建Key-Value (KV) 向量索引并将其保存在CPU内存中以供高效检索，从而解决了Transformer-based LLM在处理长上下文时面临的推理速度慢和KV缓存内存占用高的问题。

**核心思想与挑战：**
传统上，KV缓存方法在解码阶段仍需所有KV向量参与注意力计算，导致GPU内存和推理延迟随上下文长度线性增长。RetrievalAttention的核心洞察在于注意力机制的内在稀疏性：对于当前查询向量，只有一小部分关键（critical）Key-Value向量对输出有显著贡献。作者提出将近似最近邻搜索（ANNS）整合到注意力计算中，以高效地识别这些关键tokens。然而，现有的ANNS技术常因注意力机制中查询（Q）向量和键（K）向量之间的**分布外（Out-of-Distribution, OOD）**特性而失效。研究发现，Q向量与K向量之间的Mahalanobis距离显著大于K向量彼此之间的距离（超过10倍），导致传统ANNS索引在Q-K搜索中性能不佳，需要扫描超过30%的数据才能达到高召回率。

**RetrievalAttention方法详解：**

1.  **近似注意力 (Approximated Attention)：**
    RetrievalAttention基于以下假设近似全注意力输出：$o_t$ 可以通过只关注那些注意力权重超过阈值 $\epsilon$ 的KV向量（即定义为 $I_{t,\epsilon}$ 的token索引子集）来高效近似。数学上，这表示为：
    $$o_t = \sum_{i \in I_{t,\epsilon}} a_{t,i} \cdot v_i + \sum_{i \notin I_{t,\epsilon}} a_{t,i} \cdot v_i \approx \sum_{i \in I_{t,\epsilon}} \tilde{a}_{t,i} \cdot v_i \quad \text{where} \quad \tilde{a}_{t,i} = \frac{e^{z_i}}{\sum_{j \in I_{t,\epsilon}} e^{z_j}}$$
    其中，$z_i = q_t \cdot K_i^T / \sqrt{d}$ 是点积，$\tilde{a}_{t,i}$ 是在稀疏上下文 $I_{t,\epsilon}$ 下重新归一化的注意力权重。

2.  **注意力感知向量索引 (Attention-aware Vector Index)：**
    为了克服Q-K OOD问题，RetrievalAttention构建了一种特殊的向量索引。
    *   **索引构建：** 在离线阶段，不再仅仅基于Key向量的相似性构建索引。相反，它显式地建立从查询向量到其最近邻Key向量（精确K-Nearest Neighbors, KNN）的连接。这些KNN连接充当了弥合Q和K分布差异的“桥梁”（如图4b所示）。
    *   **投影技术 (Projection Technique)：** 进一步地，该方法借鉴了跨模态ANNS索引RoarGraph的投影技术。它将KNN连接“投影”到Key向量之间，即如果两个Key向量连接到同一个查询向量，则认为它们是相关的并建立连接。这消除了在索引中存储和访问查询向量的需要，同时仍能保留查询到Key的内在关系。这种投影后的索引使得在解码查询时，能够首先在“查询视角”下找到相关的Key向量，然后利用传统的图索引遍历策略进行高效搜索。该策略使得索引在搜索时仅需扫描1-3%的Key向量即可达到高召回率。

3.  **CPU-GPU协同执行 (CPU-GPU Co-Execution)：**
    RetrievalAttention将注意力计算分解为两部分，并利用CPU和GPU的优势：
    *   **持久性KV向量 (Persistent KV Vectors)：** 一小部分“静态”KV向量（例如，初始tokens和最近滑动窗口内的tokens，如640个tokens的固定模式）被预测为始终重要，并常驻在GPU内存中。
    *   **动态检索KV向量 (Dynamically Retrieved KV Vectors)：** 剩余的固定上下文的KV向量及其注意力感知索引被卸载到CPU内存中。
    *   **预填充阶段 (Prefill Phase)：** 对于用户查询，固定上下文的KV缓存从CPU加载到GPU以计算第一批tokens。此阶段结束后，除静态tokens外，其余KV缓存被丢弃以节省GPU内存。
    *   **解码阶段 (Decoding Phase)：** 在每个解码步骤中：
        1.  GPU并行计算静态模式KV向量的局部注意力输出（利用FlashAttention等高效内核）。
        2.  同时，当前查询向量被发送到CPU侧。CPU利用注意力感知向量索引高效检索最相关的动态KV向量，并计算这部分KV向量的局部注意力输出。
        3.  CPU和GPU的局部注意力输出（$o_W$ 和 $o_\Omega$）被合并以得到最终的近似注意力输出 $o_t$。合并过程类似于FlashAttention，通过重新缩放因子 $\gamma_1, \gamma_2$ 来确保结果的准确性：
            $$o_t = \gamma_1 \cdot o_W + \gamma_2 \cdot o_\Omega$$
            其中，$o_W = \text{Attn}(q_t, K[W,:], V[W,:])$ 且 $o_\Omega = \text{Attn}(q_t, K[\Omega,:], V[\Omega,:])$。缩放因子根据局部最大点积 $\tilde{z}_1, \tilde{z}_2$ 和全局最大点积 $\tilde{z}$ 进行计算：
            $$\gamma_1 = \frac{e^{\tilde{z}_1 - \tilde{z}} \cdot \sum_{i \in W} e^{z_i - \tilde{z}_1}}{\sum_{j \in I_{t,\epsilon}} e^{z_j - \tilde{z}}} \quad \text{and} \quad \gamma_2 = \frac{e^{\tilde{z}_2 - \tilde{z}} \cdot \sum_{i \in \Omega} e^{z_i - \tilde{z}_2}}{\sum_{j \in I_{t,\epsilon}} e^{z_j - \tilde{z}}}$$
    这种CPU-GPU协同执行实现了计算重叠和极低的数据传输量，显著减少了PCIe带宽的瓶颈。

**实验评估：**
RetrievalAttention在Llama-3-8B-Instruct-262k、Yi-9B-200K和Yi-6B-200K等长上下文LLM上进行了评估。
*   **准确性：** 在RULER、∞-Bench和Needle-in-a-haystack等基准测试中，RetrievalAttention保持了与全注意力（full attention）几乎相同的任务准确性（例如，在Llama-3-8B上，RULER平均准确率仅下降2.21%）。这显著优于其他零训练（training-free）的稀疏注意力方法，如StreamingLLM、SnapKV等，这些方法通常会经历显著的准确性下降。
*   **推理延迟：** 对于128K上下文，在NVIDIA RTX4090 GPU（24GB）上，RetrievalAttention实现了7.93倍于精确KNN（Flat）和2.80倍于传统ANNS索引（IVF）的解码延迟降低，每token解码速度达到0.107秒。延迟随上下文长度的增长缓慢，体现了其亚线性时间复杂度。
*   **内存效率：** 得益于将大部分KV缓存和索引卸载到CPU内存，RetrievalAttention使得8B参数模型在单张RTX4090 GPU上处理128K tokens成为可能，解决了商品级GPU的内存限制。
*   **可扩展性：** 吞吐量随batch size的增加而提高，并能受益于更多的CPU核心，进一步验证了其CPU-GPU协同架构的效率。
*   **索引质量：** 注意力感知索引仅需扫描1-3%的Key向量即可达到高召回率（recall@100高于0.95），显著优于传统方法。

## ASA
Optimizing Native Sparse Attention with Latent Attention and Local Global Alternating Strategies

https://arxiv.org/pdf/2511.00819 美团 人大；2025.11.2

1. 🤔 本文系统分析了Native Sparse Attention (NSA) 的组成部分，发现其滑动窗口注意力在常识推理中起主导作用，而压缩和选择性注意力则主要用于丰富全局上下文信息。
2. ✨ 基于此洞察，文章提出了Alternating Sparse Attention (ASA)，通过在Transformer层间**交替使用局部（滑动窗口）和全局（压缩/选择性）注意力**，并引入Multi-head Latent Attention (**MLA**) 和Group-head Latent Attention (**GLA**) 来优化效率和表达能力。
3. 🚀 实验结果表明，ASA在通用**常识推理和长文本理解任务上均达到或超越了全注意力与原生稀疏注意力**，同时将KV-cache**内存占用比NSA降低了50%**。
<img width="526" height="355" alt="image" src="https://github.com/user-attachments/assets/60dc3397-3409-4284-820f-0c2333928978" />
<img width="898" height="332" alt="image" src="https://github.com/user-attachments/assets/b8c1cdf5-81c9-4567-a634-4251248dfe25" />
<img width="814" height="346" alt="image" src="https://github.com/user-attachments/assets/f6280af4-94d2-4213-a5a9-49b86982c04f" />
<img width="1063" height="321" alt="image" src="https://github.com/user-attachments/assets/f157e205-832b-44a5-81c6-5fae5ec063a8" />
<img width="515" height="770" alt="image" src="https://github.com/user-attachments/assets/5c79a316-d83c-4d74-b161-80438240e450" />

本文提出了一种名为 Alternating Sparse Attention (ASA) 的新型稀疏注意力架构，旨在优化大型语言模型 (LLMs) 中的长上下文建模，同时显著降低 KV-cache 的内存消耗。ASA 基于对现有 Native Sparse Attention (NSA) 机制的深入分析，并对其进行了多项针对性改进。

**1. NSA 功能分析与洞察**
NSA 机制将传统注意力分解为三个分支：滑动窗口注意力 (sliding-window attention)、压缩注意力 (compressed attention) 和选择性注意力 (selective attention)。通过消融实验，作者发现：
*   **滑动窗口注意力**主要影响模型的通用常识推理能力。移除该分支会导致常识推理任务性能显著下降。
*   **选择性注意力**在增强长上下文检索能力方面发挥关键作用。移除选择性注意力会导致上下文检索任务性能大幅下降。
*   **压缩注意力**主要作为选择性注意力的辅助机制。即使使用更细粒度的压缩注意力，也无法有效弥补移除选择性注意力造成的检索性能损失。
*   滑动窗口注意力与选择性注意力同时存在时，可能会“捷足先登”地学习到局部依赖，从而降低选择性注意力在检索任务中的有效性。

此外，作者还发现，在不同层之间交替使用不同的注意力模式，相比于在所有层使用统一的稀疏度设置，能带来更好的长上下文检索性能，同时保持可比的常识推理能力，并能将 KV-Cache 存储开销减少一半。

**2. ASA 核心方法**
基于上述洞察，ASA 提出了以下核心改进：

*   **分层注意力模式交替 (Alternating Layer-wise Attention Patterns)**:
    ASA 将 NSA 中原本集成在每个注意力层内的三个注意力分支重新分配到不同的层中。具体而言，模型的连续注意力层将严格地以一对一模式交替使用两种互补的注意力类型：
    *   一类层专注于处理**压缩注意力与选择性注意力**，旨在高效捕捉长距离全局上下文信息。
    *   另一类层则专注于**滑动窗口注意力**，有效建模局部上下文信息。
    这种分层策略确保了每个注意力头专注于单一的稀疏模式，从而减少了干扰并提高了表示聚焦。

*   **引入 Latent Attention 机制**:
    ASA 用 Latent Attention 机制取代了 NSA 中原有的 Grouped Query Attention (GQA)，以增强模型的表达能力。
    *   **Multi-head Latent Attention (MLA)** 增强了滑动窗口分支。MLA 最初由 DeepSeek-V2 引入，在训练时行为与 Multi-head Attention (MHA) 相同，但在推理时通过存储低维度的 Latent states $c$ 而表现出 Multi-query Attention (MQA) 的内存效率。
        MLA 的形式化表达为：
        $$ \text{MLA}(q_{i,t}, c_{\le t}) = \text{Softmax}(q_{i,t}(c_{\le t}W_{i,k})^\top)c_{\le t}W_{i,v}W_{i,o} $$
        其中，$H$ 是注意力头数量，$q_{i,t}$ 是头 $i$ 在时间步 $t$ 的查询，$c_{\le t}$ 是直到时间步 $t$ 的 Latent states。

    *   **Grouped-head Latent Attention (GLA)** 应用于压缩和选择性分支。由于 MLA 在训练时独立投影键值，与需要共享键值表示的稀疏注意力机制不兼容，因此 ASA 在 MLA 中引入了分组机制，形成了 GLA。在 GLA 中，多个查询头共享相同的键和值投影矩阵，同时保留独立的输出投影，这使得 MLA 能够更好地适应需要共享 KV 存储的稀疏注意力。
        GLA 的形式化表达为：
        $$ \text{GLA}(q_{i,t}, c_{\le t}) = \frac{1}{H/G} \sum_{i=1}^{H/G} \sum_{j=1}^{G} \text{Softmax}(q_{iG+j,t}(c_{\le t}W_{j,k})^\top)c_{\le t}W_{j,v}W_{iG+j,o} $$
        其中，$G$ 是分组大小，每组 $G$ 个头共享相同的键值投影 $W_{j,k}$ 和 $W_{j,v}$，但保持不同的输出投影 $W_{iG+j,o}$。

*   **内核优化 (Kernel Optimization)**:
    为了提高训练效率，NSA 改进了 Flash Attention 内核。ASA 在此基础上，进一步优化了内核，允许在一个块内的所有查询共享第一个查询所选择的 KV 块。实验表明，该优化将前向计算时间减少了约 30%，后向计算时间减少了约 13%，同时对模型性能影响甚微。

**3. 实验结果**
ASA 在 340M 和 1.3B 参数量的模型上进行了广泛评估，训练数据集为 SlimPajama（15B 和 100B tokens）。
*   **常识推理任务**: ASA 的性能略优于 GQA 和 NSA，这得益于 MLA 对 GQA 的替代。
*   **上下文检索任务**: ASA 显著优于 NSA，尤其是在 Needle-In-A-Haystack (NIAH) 基准测试中，这证明了交替使用混合窗口注意力和选择性注意力的有效性。GLA 的集成通过增加注意力计算过程中的键值维度，使得 ASA 在 S-NIAH-2 任务上甚至超越了 GQA 基线。
*   **长上下文理解任务**: ASA 在几乎所有 LongBench 基准测试中都始终优于 GQA 和 NSA。

**4. 贡献**
ASA 提供了一种实用且可扩展的内存高效语言模型解决方案，通过将局部（滑动窗口）和全局（压缩/选择性）注意力机制与 Latent Attention 增强相结合，实现了高效的长上下文建模。它在性能上与 GQA 和 NSA 持平或超越，同时将 KV-cache 存储减少了 50%。

## MIT RLM
Recursive Language Model 

https://arxiv.org/pdf/2512.24601 MIT 2025.12.31

<img width="820" height="378" alt="image" src="https://github.com/user-attachments/assets/da963468-608f-4439-9b46-4e266beef503" />
1. 💡 本文提出了递归语言模型 (RLMs)，这是一种通用的推理策略，允许大型语言模型 (LLMs) 将长**提示作为外部环境的一部分进行处理，并通过 REPL 环境以编程方式检查、分解并递归调**用自身。
2. 📈 实验表明，RLMs 能够有效处理**超出模型上下文窗口两个数量级的输入**，并且在四项长上下文任务中，其性能显著优于基础 LLMs 和现有长上下文方法，同时保持相似或更低的查询成本。
3. 🔍 RLMs 展示了过滤输入信息、分块递归调用 LLMs 以及通过变量传递递归输出等新兴行为，这使得它们在信息密集型任务上表现出色，并能有效应对上下文长度和问题复杂性带来的性能下降。
大型语言模型（LLMs）在推理和工具使用方面取得了迅速进展，但仍面临上下文窗口有限的问题，且即使在这些限制内，模型质量也会随着上下文长度增加而显著下降，即“上下文腐烂”（context rot）。为了应对LLM处理任意长提示的挑战，本文提出了Recursive Language Models (RLMs)，一种通用的推理策略。
<img width="823" height="446" alt="image" src="https://github.com/user-attachments/assets/cf27fd2f-ec5d-4466-9265-40ce77da02a9" />
<img width="817" height="632" alt="image" src="https://github.com/user-attachments/assets/0e6888ee-7d5e-44bb-89ad-3f3f44e583eb" />
<img width="823" height="303" alt="image" src="https://github.com/user-attachments/assets/1037f02d-ef13-4166-a5e5-27bfa1459f85" />
<img width="837" height="687" alt="image" src="https://github.com/user-attachments/assets/7eb4e2f4-33cc-497c-b810-3649926dc976" />

**核心思想与方法论 (Core Idea and Methodology):**
RLMs的核心思想是将长提示（arbitrarily long prompts）视为外部环境（external environment）的一部分，而非直接馈送给神经网络（如Transformer）。如图2所示，RLM接收一个任意结构和长度的字符串提示 $P$，并产生一个字符串响应。其工作机制如下：
1.  **环境初始化 (Environment Initialization):** RLM初始化一个Python Read-Eval-Print Loop (REPL) 编程环境 $E$。
2.  **提示作为变量 (Prompt as a Variable):** 输入提示 $P$ 被设置为环境 $E$ 中的一个变量（例如，名为 `context`）。LLM会获得关于该REPL环境的通用上下文信息（如变量 $P$ 的长度）。
3.  **程序化交互 (Programmatic Interaction):** LLM被允许编写Python代码，通过执行这些代码来与`context`变量进行交互。这些代码可以实现：
    *   **检查和分解 (Examine and Decompose):** LLM可以编写代码来“窥视”（peek into）`context`变量的不同部分，对其进行分解（decompose），例如通过切片、正则表达式（regex queries）或按行/块划分。
    *   **观察副作用 (Observe Side Effects):** LLM可以迭代地观察代码执行的任何副作用，并利用`print()`语句将信息输出给自身进行推理。
    *   **递归调用自身 (Recursive Self-Calling):** 关键在于，RLM鼓励LLM在其生成的代码中程序化地构造子任务（sub-tasks），并使用一个特殊的`llm_query`函数来递归调用（invoke recursively）自身（即sub-LLM）来处理这些子任务。例如，LLM可以将`context`的某个片段传递给`llm_query`函数，要求sub-LLM对该片段进行分析或提炼。

这种设计将上下文管理隐式地交由LLM自身处理，通过编程逻辑和迭代反馈循环，使得RLM能够处理远超其内部模型上下文窗口限制的输入长度。与传统上将所有输入直接送入模型（导致“上下文腐烂”）或使用有损压缩（lossy compression）方法（如概括或截断）不同，RLM通过程序化地选择性访问和处理上下文片段，从而在不丢失细粒度信息的前提下实现长上下文处理。

**实验与发现 (Experiments and Findings):**
本文在四项多样化的长上下文任务上评估了RLMs：S-NIAH (简单信息检索)、BrowseComp-Plus (多跳问答，需要信息聚合)、OOLONG (复杂推理，需要语义转换和聚合) 和 OOLONG-Pairs (信息密度极高的配对推理任务，处理成本呈二次方增长)。实验使用了前沿的闭源模型GPT-5和开源模型Qwen3-Coder-480B-A35B，并与直接LLM调用、上下文压缩（Summary Agent）、以及带工具（BM25检索）的代码生成代理（CodeAct）等基线进行比较。

**主要发现 (Key Observations):**
1.  **大规模长上下文处理能力 (Scalability to 10M+ tokens):** RLMs能够成功处理高达千万甚至亿级别令牌（10M+ token）的输入，比基础模型的上下文窗口大两个数量级，并在所有任务上显著优于基线模型，性能提升高达2倍，同时保持了相当或更低的平均查询成本。
2.  **对复杂任务的卓越表现 (Strong Performance on Complex Tasks):** 在OOLONG和OOLONG-Pairs等信息密集型任务上，RLMs的性能相较于基础模型有巨大飞跃（例如，OOLONG-Pairs上GPT-5的F1分数从不到0.1%提升至58.00%），展示了处理极端信息密度任务的 emergent capability。
3.  **REPL环境与递归调用的作用 (Role of REPL and Recursive Calls):** REPL环境本身使得模型能够处理超长输入。而递归子调用对于信息密集型任务至关重要，它允许RLM对上下文进行细粒度的语义转换和聚合。在某些任务中，即使没有子调用能力，REPL环境也能让RLM超越基础模型的上下文限制，但在信息密集型任务上，有子调用能力的RLM性能明显更好（10%-59%的提升）。
4.  **成本与复杂性 (Cost and Complexity):** RLM的推理成本与基础模型调用相当，但在某些情况下成本较高，因为任务复杂性导致轨迹长度差异大。虽然RLM的尾部成本（tail end costs）可能很高，但其中位数成本通常与或低于基础模型。RLM能够选择性地查看上下文，使其比Summary Agent等完全摄取输入的基线更高效。
5.  **模型行为差异 (Model-Agnostic but Different Behaviors):** 尽管RLM是一种模型无关的推理策略，但不同模型（如GPT-5和Qwen3-Coder）作为RLM时，在上下文管理和子调用决策上展现出不同行为模式。例如，Qwen3-Coder倾向于更频繁地进行子调用，有时会导致效率低下。

**涌现模式 (Emergent Patterns):**
RLMs无需显式训练就展现出有趣的上下文管理和问题分解行为：
*   **信息过滤 (Information Filtering):** RLM能够通过代码执行（如正则表达式查询）过滤输入信息，结合模型先验知识缩小搜索空间，只处理少量相关令牌。
*   **分块与递归 (Chunking and Recursive Sub-calling):** RLM将无限制长度的推理链下放到子R(LM)调用。常见的分解策略包括统一分块和关键词搜索。
*   **答案验证 (Answer Verification):** RLM通过子LM调用或代码执行进行答案验证，有时会隐式避免上下文腐烂。
*   **处理长输出任务 (Handling Long Output Tasks):** RLM通过在REPL中迭代构建变量来生成超出基础LM限制的任意长度输出，结合程序化输出和子R(LM)输出。

**局限性与未来工作 (Limitations and Future Work):**
目前RLMs仍有改进空间：
*   **同步子调用 (Synchronous Sub-calls):** 当前实现采用同步/顺序子调用，导致运行时较慢。异步调用和沙盒REPL有望显著降低运行时和成本。
*   **递归深度 (Recursion Depth):** 本文主要探索了最大递归深度为1的情况，未来可研究更深层次的递归。
*   **模型训练 (Model Training):** 现有模型并非为作为RLM而设计，显式训练模型（作为root或sub-LM）可能带来显著性能提升。RLM轨迹可视为一种推理形式，可通过自举（bootstrapping）现有模型进行训练。

  
## ShinkaEvolve
ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution

https://arxiv.org/pdf/2509.19349 Sakana AI 2025.12.31

https://github.com/SakanaAI/ShinkaEvolve

1. 🚀 ShinkaEvolve是一个开源框架，通过结合**父程序采样、代码新颖性拒绝采样**和bandit-based LLM ensemble selection等创新算法，显著提升了LLM驱动程序进化的样本效率。
2. 💡 该框架在多个任务中取得了SOTA成果，包括仅用150次评估就找到新的圆堆积（Circle Packing）解决方案，设计高性能AIME数学推理Agentic harnesses，改进ALE-Bench竞争编程方案，并发现新型Mixture-of-Expert load balancing loss functions。
3. 🌐 ShinkaEvolve的广泛适用性和卓越样本效率，以及其开源特性，旨在促进开放式科学发现的民主化和可及性。
<img width="918" height="417" alt="image" src="https://github.com/user-attachments/assets/aa11616d-5ab2-4899-a096-91bc799f22f0" />
<img width="921" height="330" alt="image" src="https://github.com/user-attachments/assets/88d17153-f5ed-4b8b-ab16-2ec43ee36ee1" />
<img width="935" height="415" alt="image" src="https://github.com/user-attachments/assets/4cbe6dc8-3c68-4610-99b1-7d891a6f519e" />
![Uploading image.png…]()
![Uploading image.png…]()

论文提出了一种名为 ShinkaEvolve 的新型开源框架，旨在通过利用大语言模型（LLMs）显著**提升程序演化过程的样本效率和解决方案质量**，以加速科学发现。该框架旨在克服现有 LLM 驱动的代码演化方法在样本效率低下（通常需要数千次评估）和闭源限制方面的缺点。

ShinkaEvolve 通过三项核心算法创新实现其目标：
1.  **父程序和启发式程序采样（Parent and inspiration sampling）**：ShinkaEvolve 维护一个固定大小的已评估程序档案，其中包含程序的适应度分数和元信息。它采用“岛屿模型”（island model）方法，通过独立的子群体并行演化以增强多样性并防止过早收敛。父程序选择策略平衡了探索与利用：
    *   **幂律采样（Power Law Sampling）**：程序根据适应度排名$r_i$（最佳程序$r_i=1$），选择概率遵循$p_i = \frac{r_i^{-\alpha}}{\sum_{j=1}^{n} r_j^{-\alpha}}$。其中，$\alpha$控制利用强度，$\alpha=0$为均匀采样（纯探索），$\alpha \to \infty$为爬山算法（纯利用）。
    *   **加权采样（Weighted Sampling）**：融合了性能和新颖性。性能分量$s_i = \sigma(\lambda \cdot (F(P_i) - \alpha_0))$，其中$\alpha_0$是所有程序的适应度中位数，$F(P_i)$是程序$P_i$的适应度，$\sigma(x) = \frac{1}{1+e^{-x}}$是Sigmoid函数，$\lambda$控制选择压力。新颖性分量$h_i = \frac{1}{1+N(P_i)}$，偏好具有较少子代的程序。最终概率为$p_i = \frac{w_i}{\sum_{j=1}^n w_j}$，其中$w_i = s_i \cdot h_i$。

2.  **程序变异和新颖性评估（Program mutation and novelty assessment）**：
    *   **LLM 引导的程序变异（LLM-Guided Program Mutations）**：ShinkaEvolve 从预设的 LLM 池中选择一个 LLM 和采样参数。它采用三种变异方法：
        *   **Diff-Based Edits**：利用 LLMs 进行目标性修改。
        *   **Full Rewrites**：允许对程序进行完全重写，同时确保不可变代码块保持不变。
        *   **Crossover Mutation**：采样一个额外的档案程序，并提示 LLM 将其与父程序结合。
    *   **新颖性拒绝采样（Novelty Rejection Sampling）**：为增强生成代码提案的创造性，ShinkaEvolve 使用一个嵌入模型（如 `text-embedding-3-small`）来嵌入程序的可变代码部分。如果新生成程序与岛屿子群体中现有程序的最大余弦相似度超过某个阈值（例如$\eta = 0.95$），则会查询另一个 LLM 来进一步评估该程序是否具有有意义的新颖性，以避免冗余变异。

3.  **执行和世界反馈（Execution and world feedback）**：
    *   **多目标评估与文本反馈（Multi-Objective Optimization & Textual Feedback）**：执行程序后，ShinkaEvolve 进行多目标评估，得到标量适应度值$r_i$、一组“公共指标”和文本反馈，并将这些信息存储在程序档案中，作为未来 LLM 变异的上下文。
    *   **自适应 LLM 采样演化（Adaptive LLM sampling evolution）**：不同 LLMs 在不同问题领域和不同演化阶段的表现各异。ShinkaEvolve 采用基于 UCB1 算法（Auer et al., 2002）的自适应策略，根据 LLM 生成变异的性能来更新其采样概率。LLM 的奖励$r_u^i = \exp(\max(r_i - r_b^i, 0)) - 1$，其中$r_b^i$是基线奖励（父程序或初始程序中的最大值），这促进了能够提出大胆、高风险、高回报变异的 LLMs。
    *   **元草稿本和在线优化（Meta-Scratchpad & Online Refinement）**：ShinkaEvolve 实现了元草稿本系统，每$T$代周期性分析成功的解决方案，总结优化策略和设计原则，并将这些见解综合为可操作的建议，附加到变异提示中，为 LLM 提供高层次的指导。

实验结果表明，ShinkaEvolve 在多个任务中表现出色：
*   **圆盘堆积问题（Circle Packing）**：ShinkaEvolve 仅用 150 次评估就发现了一个新的 State-of-the-Art 圆盘堆积解决方案，比现有方法所需的数千次评估显著提高了样本效率。其发现的算法结合了复杂的初始化策略、SLSQP 梯度优化与模拟退火（Simulated Annealing）的混合优化方法，以及智能扰动机制。
*   **AIME 数学推理任务（AIME Mathematical Reasoning Tasks）**：ShinkaEvolve 能够演化出高效的 Agent Scaffold 设计，在 10 次 LLM 查询限制下，显著超越了人工设计的基线方案。其最终方案是三阶段架构，包含多样化的专家角色、严格的同行评审和合成机制。
*   **ALE-Bench 竞技编程（ALE-Bench Competitive Programming）**：ShinkaEvolve 成功改进了 ALE-Agent 发现的高性能解决方案，平均提升约 2.3%。例如，在 ahc039 任务中，解决方案排名从第 5 位提升至第 2 位。
*   **MoE 负载均衡损失函数（Mixture-of-Expert Load Balancing Loss）**：ShinkaEvolve 发现了一种创新的 MoE 负载均衡损失函数，其在保持模型表达力的同时，有效激励了专家之间的效率和特化。其发现的损失函数$L_{\text{LBL}}$是对已有的“全局批次负载均衡损失”（Global-batch LBL）的扩展，引入了一个新项以正则化那些未充分特化的专家：
    $L_{\text{LBL}} = N_E \cdot \frac{1}{L} \sum_{\ell=1}^L \sum_{i=1}^{N_E} f_{\ell,i} P_{\ell,i} + 0.1 \sum_{\ell=1}^L s(P_\ell) \sum_{i=1}^{N_E} \max(0, \tau - f_{\ell,i})$
    其中，$N_E$是专家数量，$L$是层数，$f_{\ell,i}$是专家$i$在层$\ell$的令牌分配频率，$P_{\ell,i}$是路由器的平均软分配概率，$s(P_\ell) = 0.5 + (\frac{1 - H(P_\ell)}{\log N_E})$是路由熵$H(P_\ell)$的归一化补数，$\tau = 0.064/N_E$是最小使用阈值。该新项在层路由熵较低、路由器集中于少数主导专家时，会更强地推动那些令牌分配低于阈值的专家。

通过开源实现和卓越的样本效率，ShinkaEvolve 旨在普及开放式发现，降低计算资源门槛。

## MSched
MSched: GPU Multitasking via Proactive Memory Scheduling 

https://arxiv.org/pdf/2512.24637 陈海波团队 2025.12.31

1. 🤔 针对GPU HBM容量有限导致多任务下传统需求分页性能严重下降的问题，MSched提出通过主动调度工作集来扩展GPU上下文切换。
2. 💡 为实现此目标，MSched采用了基于模板的方法对kernel工作集进行精确的空间预测，并设计了任务调度器与内存管理器之间的协同机制，利用调度器的时间线推断全局内存访问序列以优化页面放置。
3. 🚀 基于XSched实现，5080GPU（16GB PCIGen5*16）实验证明，MSched在内存超额订阅下显著优于传统需求分页（LLM推理高达57.88倍），并通过减少RT任务延迟和提高BE任务吞吐量来提升并发GPU任务的整体性能和效率。
人为精心设计的场景

<img width="582" height="300" alt="image" src="https://github.com/user-attachments/assets/38506e44-7ebf-4ffd-a453-78b5fdfc4045" />
<img width="1205" height="429" alt="image" src="https://github.com/user-attachments/assets/f28cc987-129f-42c7-a5eb-4449ec804598" />
<img width="581" height="325" alt="image" src="https://github.com/user-attachments/assets/b3bd45cf-6157-4703-8fe2-1f35eda7a65b" />
<img width="1226" height="270" alt="image" src="https://github.com/user-attachments/assets/9135a9fd-aa22-4762-8d9d-14d583a520f2" />
<img width="1213" height="621" alt="image" src="https://github.com/user-attachments/assets/c6d5fe51-c65a-452c-a34a-3fb7c75c3239" />


## Mirage Persistent Kernel(MPK)
Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs

paper: https://arxiv.org/pdf/2512.22219 CMU 2025.12.22

code: https://github.com/mirage-project/mirage

1. 🚀 Mirage Persistent Kernel (MPK): 新的编译器和运行时系统，旨在自动将 **multi-GPU 模型**推理**自动的**转化为单个高性能 mega-kernel。
2. 💡 MPK 引入了 **SM-level 的 tGraph** 表示来捕获细粒度数据依赖，并通过其编译器生成优化的任务图，同时 **in-kernel parallel runtime 采用去中心化调度**在 SMs 上执行这些任务。
3. 📈 MPK引入了新的pipeling和overlap机会，在**PyTorch中作为一个**`kernel backend`实现：torch.compile(backend=MPK)。A100/H100/B200的单卡/多卡推理实验表明，MPK 显著优于现有的系统，将推理**延迟降低高达 1.7 倍**。
- 实现：40K行C++, 84K行CUDA，10K python。生成kernel，通信的底层调用的是nvshmem接口。每卡预留了4个SM做scheduler（4*4=16 warps）。为了模型推理e2e评测，实现了continous batch和paged attn。
- Qwen和Llama系列模型，dense最大8b，MoE采用Qwen-30b-a3b，offline模式 64->1024 对标了SGLang，vLLM。模型越小 加速越大（1.5x+）；batch越小 越明显；H100/B200 8b最多1.2x（A100上有1.5x），MoE 1.1x
  
  
MPK解决了现有`kernel-per-operator`（每个操作一个核）执行模式的局限性，例如`kernel barrier`限制`cross-operator software pipelining`和`fine-grained compute-communication overlap`，以及频繁的`kernel launch overhead`。
<img width="622" height="294" alt="image" src="https://github.com/user-attachments/assets/332749a9-fdac-4637-8179-d94aca5df99e" />

<img width="1353" height="507" alt="image" src="https://github.com/user-attachments/assets/40a7145f-ef61-4009-815c-61d5ee31e54d" />
<img width="1206" height="375" alt="image" src="https://github.com/user-attachments/assets/70320b43-aace-4754-b446-9a8aee9541df" />
<img width="1337" height="620" alt="image" src="https://github.com/user-attachments/assets/6a3f385d-473e-4cbf-b339-f9a7de983fc0" />
<img width="636" height="450" alt="image" src="https://github.com/user-attachments/assets/821cad02-a87c-453a-90c7-bdd572f3e2bc" />
<img width="653" height="416" alt="image" src="https://github.com/user-attachments/assets/11a43af0-a230-493f-ae9d-e337fc126567" />
<img width="642" height="435" alt="image" src="https://github.com/user-attachments/assets/d7de9a47-b04b-44df-a081-ba8684503bc2" />

MPK的核心思想是将**计算和GPU间通信的粒度细化**到单个`streaming multiprocessor` (SM)。为此，MPK引入了`SM-level graph representation`，称为`tGraph`。`tGraph`的节点表示在单个SM上执行的任务（`task`）或同步点（`event`），边表示任务之间的细粒度依赖关系。这种表示方式揭示了额外的并行性，并使得跨操作符的软件流水线和细粒度核重叠成为可能。

MPK系统包含两个主要组件：
1.  **MPK Compiler**: 将`tensor program`和推理配置作为输入，自动将其计算图转换为针对给定配置和GPU架构优化的`SM-level tGraph`。编译器采用多种优化技术来减少同步开销并最大化生成的`tGraph`的性能。
    *   **Operator decomposition**: 将输入计算图的每个操作符分解为一组任务，通过划分操作符的输出张量，使所有任务计算输出的互不相交的子集，从而在SM之间并行执行。分区策略旨在最小化从`device memory`到`shared memory`的数据加载。
    *   **Dependency analysis**: MPK使用`event`来捕获任务间的依赖关系。对于共享张量的两个操作符，MPK会遍历这两个操作符的所有任务对($t_1, t_2$)。如果任务$t_1$产生的输出区域与任务$t_2$消耗的输入区域有重叠，则在$t_1$和$t_2$之间引入一个`event` $e$。该`event`作为同步点，确保$t_2$在$t_1$完成其所需数据生成之前不会开始执行。相应的，MPK会在`tGraph`中插入两条边$(t_1, e)$和$(e, t_2)$。这种细粒度的依赖分析在保留`producer-consumer`依赖的同时，暴露了独立任务之间的最大并行度。
    *   **Event fusion**: 消除冗余同步点并简化`tGraph`。
        *   `Successor-set fusion`: 合并作为同一组消费任务先决条件的`event`。若`OutTasks(e1) = OutTasks(e2)`，则将$e_1, e_2$融合成新的`event` $e'$，其`InTasks(e') = InTasks(e1) \cup InTasks(e2)`，`OutTasks(e') = OutTasks(e1)`。
        *   `Predecessor-set fusion`: 合并依赖于相同生产任务集的`event`。若`InTasks(e1) = InTasks(e2)`，则将$e_1, e_2$融合成新的`event` $e'$，其`InTasks(e') = InTasks(e1)`，`OutTasks(e') = OutTasks(e1) \cup OutTasks(e2)$。
    *   **tGraph normalization**: 解决任务具有任意数量依赖/触发`event`的内存开销问题。通过引入新的`event`和不执行计算的“空”任务（`empty new tasks`），确保每个任务最多只有一个依赖`event`和一个触发`event`（如图6所示）。
    *   **tGraph linearization**: 解决`event`触发大量任务时的存储问题。使用基于`BFS`的算法（Algorithm 1）来线性化`tGraph`，确保所有由同一`event`触发的任务在最终任务排序中被分配连续的索引。这样，`event`的`fan-out`可以通过存储第一个和最后一个任务索引来紧凑地编码，而无需存储显式依赖任务列表。
    *   **Task Implementation Generation**: 利用现有的`superoptimization`技术，如`Mirage superoptimizer`，自动为**每个任务生成高性能的CUDA实现**，包括`intra-SM optimizations`。

2.  **In-Kernel Parallel Runtime**: 在单个`mega-kernel`内执行`SM-level tGraph`。这种设计消除了`kernel launch overhead`，并实现了**对调度、同步和执行顺序的细粒度控制**。
    *   **SM Partitioning**: 将GPU的**SMs划分为**`workers`和`schedulers`。每个`worker`运行在一个物理SM上，维护一个独立的任务队列并以`FIFO`顺序执行任务。`schedulers`则维护任务间的依赖关系，并在先决条件满足时分配任务。`schedulers`以`warp`粒度组织。
    *   **Event-Driven Execution**: 运行时采用`event-driven model`执行`tGraph`。一个`tGraph`从一个无先决条件的`start event`开始。当一个`event`被`scheduler`取出后，`scheduler`会启动所有依赖于它的任务。每个启动的任务被分派给一个`worker`，`worker`执行任务并在完成后通知该任务所关联的触发`event`。一个`event`在其所有先决条件任务完成并触发它达到所需次数后被激活。
    *   **Hybrid Task Launch**: 结合了`Just-in-time` (JIT) 和`Ahead-of-time` (AOT) **两种任务启动机制**。
        *   `JIT`: `scheduler`仅在依赖`event`完全激活后才将任务分配给`worker`。这使得MPK能够适应工作负载不平衡（如注意力操作的数据依赖性），实现动态负载均衡，但引入了额外的`worker`–`scheduler`通信延迟。
        *   `AOT`: 运行时在先决`event`激活之前预先将任务排队到`worker`。`worker`只需等待`event`激活。这减少了任务启动延迟，但对静态工作负载更有效。
        *   混合策略: 编译器根据任务执行时间是否具有数据依赖性和是否可能导致运行时不平衡来分类。具有数据依赖性的操作符（如注意力）被标记为`JIT`，直到遇到全局屏障；其他则标记为`AOT`以最小化调度开销。`workers`优先处理`JIT`任务，当`JIT`队列为空时检查`AOT`任务。`AOT`任务在执行前预先以`round-robin`方式分配到`workers`。
    *   **Runtime Optimizations**:
        *   `Paged shared-memory abstraction`: 将`shared memory`划分为固定大小的页面。任务根据需要获取和释放页面。这使得`cross-task software pipelining`成为可能，即当前任务的计算可以与后续任务的数据预加载重叠。
        *   `Cross-task software pipelining`: 将每个任务分解为`pre-loading phase`和`compute phase`。在当前任务的`compute phase`执行期间，可以同时启动下一个任务的`pre-loading phase`，从而在**SM上实现跨任务的流水线**并行。
        *   `Pre-fetching task descriptions`: 将即将到来的任务描述预取到`shared memory`，以减少`device memory`访问延迟和`enqueue/dequeue`延迟。
        *   轻量级`worker`和`scheduler`，`event`和任务队列实现为`circular buffers`，利用`atomicAdd`指令，以及`decentralized scheduling`以避免全局协调开销。

实验结果表明，MPK显著优于现有`kernel-per-operator` LLM服务系统，端到端推理延迟最高可降低1.7倍，将LLM推理性能推向接近硬件极限。MPK在**PyTorch中作为一个**`kernel backend`实现，只需少量代码修改即可将PyTorch模型编译为MPK `mega-kernel`。

## qTTT 
Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs

paper：https://arxiv.org/pdf/2512.13898 Meta 2025.12.30

中文解读： https://mp.weixin.qq.com/s/1OA61YW28D_Pribdg1Eaiw 

## TTT-E2E 
End-to-End Test-Time Training for Long Context
paper：https://arxiv.org/abs/2512.23675 Astera 研究所 NVIDIA 斯坦福等， 2025.12.31 

code：https://github.com/test-time-training/e2e

中文解读：https://mp.weixin.qq.com/s/30ysdCrari7V2Rd9SUF4Kg
针对长序列，提出一种混合模型（训练3B），结合SWA，full-attn，精度优于混合线形；128k时速度比full-attn 快2.7x。总体比较复杂。
TTT-E2E 的核心思想是将模型在测试阶段（推理阶段）的行为定义为一个在线优化过程。当模型读取长上下文时，它不仅仅是在做前向传播，还**在同步进行梯度下降**。方法基于这样一个逻辑：如果我们将上下文看作一份学习资料，那么模型在预测下一个 token 之前，可以先在已经读过的 token 上进行自监督学习。
通过这种方式，上下文中的信息就被编码进了**模型的权重 W 中，而不是存储在外部的 KV Cache 里**。这就像是在阅读一本书时，你不断根据新读到的内容修正自己的认知模型。

首先是元学习（Meta-Learning）。传统的模型在预训练时并未考虑测试时的更新逻辑，这会导致训练与测试的脱节。TTT-E2E 通过外层循环（Outer Loop）优化模型的初始化参数，使得模型「学会如何学习」，即经过少量测试时梯度更新后，能达到最优的预测效果。
其次是架构的微调与滑动窗口的结合。该团队意识到，如果完全摒弃注意力机制，模型会丧失局部精确记忆能力。因此，TTT-E2E 采用了一种混合架构：使用一个固定大小（如 8K）的滑动窗口注意力（SWA）来处理短期记忆，确保局部逻辑的严密；而对于超出窗口的长期记忆，则交给 TTT 更新后的 MLP 层来承担。

## RL TP确定性计算
Rice Univ以及 明尼苏达大学等的两篇相关论文，论文2是最新版。

论文1: Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference
https://openreview.net/pdf?id=Q3qAsZAEZw (NeurIPS 2025, Oral)

论文2: Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch 针对TP不同size引入的差异性 提出统一的tree based GEMM和AllReduce kernel “TBIK”, trition实现，vLLM/FSDP集成，kernel开销有点大；通信影响更大。E2E下降50%
https://arxiv.org/pdf/2511.17826 2025.11.21 

code： https://github.com/nanomaoli/llm_reproducibility

官方blog：https://festive-clam-15f.notion.site/Enabling-Large-Scale-True-on-Policy-RL-by-Bringing-Tensor-Parallelism-to-Order-2b039f5cabfa807b9770fcbe339f0f9b

论文1
1. 🔍 该研究揭示，LLM推理的重现性在系统配置变化（如GPU数量、批处理大小）下表现出脆弱性，尤其对于推理模型影响显著，其根本原因在于有限数值精度下的浮点算术非结合性。
2. 📊 实验表明，BF16精度下的**贪婪解码（greedy decoding）导致显著的输出变异**，而F**P32则提供近乎完美的重现性**，表明数值精度对LLM输出稳定至关重要。
3. 💡 为平衡内存效率和重现性，该文提出LayerCast，一种混合精度推理方案，其将**模型权重存储为BF16但在计算时上转换为FP32**，实现了接近FP32的确定性。
<img width="815" height="323" alt="image" src="https://github.com/user-attachments/assets/c832ac7f-8f0f-48c5-8a14-cd4f625e5bad" />

**主要发现**：
**贪婪解码的非确定性**：BF16 精度下的贪婪解码表现出显著的不稳定性。例如，DeepSeek-R1-Distill-Qwen-7B 在 AIME’24 上，BF16 精度可导致高达9%的准确率标准差，而 FP32 几乎为零。BF16 还会导致响应长度显著波动（可达9,000 token），这严重影响了长文本推理和高效推理研究。

**数值精度影响**：FP32 提供了近乎完美的确定性，**FP16 表现出中等可变性**，而 **BF16 表现出显著的可变性**。这是因为 BF16 具有有限的尾数位（7位，FP16为10位），导致概率计算中引入更大的误差。当这些波动与 top-1 和 top-2 token 概率之间的微小差距重叠时，token 翻转的可能性增加。FP32 的高精度（23位尾数位）使运行时变化几乎可以忽略不计。

**分歧点**：FP32 显著减少了分歧的例子数量，并将分歧点推迟到序列的更晚位置。例如，在 MATH500 上，**BF16 有超过90%的例子在早期就出现分歧**，而 FP32 只有2.2%的例子出现分歧。

**随机采样的不稳定性**：即使在随机采样设置下，数值精度也引入了额外的变异来源。**BF16 等低精度格式往往产生更高的输出方差**。这意味着在低精度下，研究人员可能需要更多的运行次数才能达到与高精度格式相同的统计置信度。

运行时配置的影响：
**GPU数量**：**4个GPU通常比2个GPU表现出更高的 top-1 token 概率变异**，可能是因为并行计算的增加引入了更多变化的浮点操作顺序。
**批处理大小**：**较小的批处理大小反而导致更高的 token 概率方差**，而较**大的批处理则通过CUDA优化内核的并行计算限制了误差积累**。
GPU类型：**A100 GPU通常比L40S GPU表现出略高的概率方差**，这可能归因于硬件级浮点实现和内存层次结构的差异。这些影响在BF16精度下最为显著。

**解决方案 LayerCast**：鉴于 FP32 的高内存和推理时间成本，论文提出了 LayerCast，一种混合精度方法。
模型参数**最初以 FP32 **精度加载。所有**线性层权重和偏置明确地转换为 BF16 存储**。推理运行时，每个权重在矩阵乘法之前即时（just-in-time）地向上转换为 FP32。

**论文2**:
1. 💡 针对大型语言模型（LLM）推理中存在的非确定性问题，特别是**不同Tensor Parallel (TP) 大小导致的结果不一致（同batch内 通过ThinkMachine等方法可实现一致）**，本文旨在解决这一关键挑战，该问题在RL训练中尤为突出。
2. ⚙️ 本文提出了**Tree-Based Invariant Kernels** (TBIK)，通过**统一的层次化二叉树结构对MatMul和All-Reduce操作进行编排**，确保无论TP大小如何，都能实现**比特级相同的浮点计算结果**。
3. ✅ TBIK已在**Triton中实现并集成到vLLM和FSDP中**，实验证明其在不同TP尺寸下能实现零概率发散和比特级可复现性，成功**消除了vLLM和FSDP之**间在RL训练中的精度不匹配问题。具体是：
   1）linear层：对QKV_proj, up/gate和lm_head激活BIO对batch不变；对Out-proj, down-proj采用列切的TBIK
   2）attn：关闭vLLM的prefix-cache; 训推采用相同的TritonAttn（固定tile大小）
   3）RMSNorm，Silu，RopE embedding：FSDP中采用列vLLM的实现。
没测MoE模型；没测改进后的分数？
<img width="716" height="257" alt="image" src="https://github.com/user-attachments/assets/93beeee3-f5ab-4c90-80fa-2a3911a60d32" />

<img width="832" height="331" alt="image" src="https://github.com/user-attachments/assets/ac0c7424-39d7-4423-bfc1-814d25d3fcc6" />
<img width="817" height="339" alt="image" src="https://github.com/user-attachments/assets/d30c22f2-972b-42ca-88fb-f97a1faa3365" />
<img width="486" height="193" alt="image" src="https://github.com/user-attachments/assets/58410b22-ad28-4ea0-bff5-d2f55189e9d1" />
<img width="481" height="323" alt="image" src="https://github.com/user-attachments/assets/c2c6a589-2cb8-4ee2-bd92-e4c7974d1222" />
<img width="452" height="298" alt="image" src="https://github.com/user-attachments/assets/7d0a2252-94f2-4ca8-b362-738e1d743227" />
<img width="996" height="386" alt="image" src="https://github.com/user-attachments/assets/32132450-4554-4e98-b6e3-eac5bf829707" />
<img width="1007" height="467" alt="image" src="https://github.com/user-attachments/assets/a8f162fc-b1e0-4066-8eb0-edd0bc354136" />
<img width="476" height="339" alt="image" src="https://github.com/user-attachments/assets/9d007a1d-a163-4559-acf5-98f2b66ddc6f" />
<img width="493" height="632" alt="image" src="https://github.com/user-attachments/assets/01697d25-ff56-448c-8be9-dfe6574ccb19" />

<img width="481" height="389" alt="image" src="https://github.com/user-attachments/assets/b521f842-efc2-48d0-92ba-f32fa05473da" />
<img width="514" height="570" alt="image" src="https://github.com/user-attachments/assets/b66e9c40-0cce-4d53-8485-31388cbe167b" />

**核心问题与挑战：**
LLM 推理的非确定性源于浮点运算的非结合性，即计算顺序的变化会导致累积的舍入误差，从而产生不同的结果。现有系统中的多种因素都会改变浮点运算顺序，包括：
1.  **连续批处理 (Continuous Batching)**：动态改变批次中的请求集和批次大小。
2.  **不同操作实现**：例如，MatMul 中 Split-K 与 Non-Split-K 的使用。
3.  **操作超参数**：如 MatMul 和 Flash-Attention 的 Block Size。
4.  **并行系统中的集合操作 (Collective Operations)**：特别是 All-Reduce。
5.  **并行策略**：尤其是张量并行 (TP)，它将工作负载分片到多个 GPU 上。
6.  **不同 GPU 架构**：可能使用不同的底层指令集。

先前的研究，如 Batch-Invariant Operations (BIO)，通过沿批次维度并行化计算，解决了批次大小引起的非确定性问题。但 BIO 仅限于消除批次大小的方差，**无法解决由 TP 引起的非确定性**。

**TP 导致非确定性的具体原因：**
在张量并行 (TP) 中，模型的权重矩阵被分片。对于行并行层（如自注意力中的 `o_proj` 和前馈网络中的 `down_proj`），输入和权重矩阵在 K 维度上被分片到不同的 GPU。每个 GPU 计算部分结果 $X_i W_i$，最终结果通过跨 GPU 的求和（All-Reduce）聚合得到。
问题在于，当 TP 配置改变时，All-Reduce 操作的参与设备数量和通信模式（如 NCCL 的 ring 或 tree 归约）也会改变，这导致浮点数累积顺序的变化。例如，标准的 **cuBLAS GEMM 会先在每个 GPU 上沿 K 维度顺序执行局部归约，然后通过 NCCL 进行跨 GPU 归约**。这种两阶段的归约顺序会**随 TP 大小而异，从**而产生不同的输出。

**TBIK 方法论：**
该论文的核心思想是通过**统一的、分层的二叉树结构来对齐** GPU 内部 (intra-GPU) 和 GPU 之间 (inter-GPU) 的归约顺序。无论 TP 大小如何，这种方法都能确保计算顺序的一致性。

1.  **树形归约 MatMul 内核 (Tree-Based MatMul Kernel)**：
    *   **设计理念**：将每个矩阵乘法分解为沿归约维度 K 的 `tiles`。然后，这些 `tiles` 的累积严格遵循一个固定的、预定义的二叉树结构。
    *   **实现细节** (基于 Triton，参考 Algorithm 1)：
        *   每个 GPU 负责总瓦片数量 $T$ 中的 $T/C$ 块，其中 $C$ 是 GPU 数量。
        *   为了实现树状累积，中间的部分结果被存储在一个名为 `S` 的累加器缓冲区中，其形状为 $[L, \text{BlockM}, \text{BlockN}]$，其中 $L = \log_2(T / \text{Kfirst})$ 是二叉归约树的深度。`Kfirst` 是在第一次进位前累积的瓦片数量。
        *   `Count` 数组 `Count[L]` 用于记录每个级别已合并的瓦片数量。
        *   当某个级别 `l` 的计数器达到 `Kfirst` (对于第一级) 或 2 (对于其他级别) 时，触发进位操作：该级别对应的两个部分和被归约到下一级别 (`S[l+1] = S[l+1] + S[l]`)，然后当前级别的值和计数器被清零。
        *   这个过程重复进行，直到 K 维度上的所有瓦片都被处理，最终归约结果存储在 `S[L]` 中。

2.  **树形 All-Reduce (Tree All-Reduce)**：
    *   **设计理念**：在每个 GPU 完成本地 MatMul 累积后，使用 All-Gather 集体操作同步各 GPU 的部分结果。然后，这些收集到的结果通过两两求和的方式进行归约，确保跨 GPU 的归约也遵循相同的固定树结构。
    *   **实现细节** (参考 Algorithm 2)：
        *   论文明确指出，不直接使用 NCCL 内置的树形 All-Reduce，因为 NCCL 仅允许在节点间指定树形拓扑，而节点内部仍默认为链式归约。
        *   因此，论文实现了一个自定义的树形 All-Reduce 算法，确保节点内的归约也严格遵循树状拓扑。

**理论证明 (Theorem 1)：**
如果总瓦片数 $N$ 可以被 TP 大小 $C$ (且 $C$ 是 2 的幂) 整除，并且瓦片在 GPU 间均匀分布，那么在定义的树形操作符 $T(\cdot)$ 下，分层归约顺序是固定的，与 TP 大小无关。这意味着计算结果将是位级别相同的。

**实验评估：**
论文通过在 Qwen3-8B/32B、Mistral-7B-Instruct 和 Llama-3.1-8B-Instruct 模型上，使用 AIME24 和 AMC23 数据集进行了评估。

1.  **可复现性**：
    *   **唯一输出计数 (Count of Unique Outputs)**：对于相同的输入，输出序列不同的情况数。BIO+TBIK 始终达到 1，表示在所有测试配置（不同 TP 大小、批次大小）下，模型产生位级别相同的推理结果。
    *   **最大概率偏差 (Maximum Probability Divergence)**：衡量 Top-5 预测概率的差异。BIO+TBIK 实现了严格的零偏差，验证了位级别的确定性。

2.  **性能开销**：
    *   **核函数层面**：TBIK MatMul 核函数在 BF16 模式下比 cuBLAS 慢，吞吐量约为其 63%。
    *   **端到端延迟**：在 Qwen3-8B 模型上，BIO+TBIK 相比 vanilla BF16 引入了 56% 到 135% 的显著开销。
    *   **开销分解**：树形 MatMul 核函数仅占总开销的 3-14%，因为它在模型总计算量中占比小。树形 All-Reduce 操作贡献了更大的开销 (28-50%)，主要由于其当前的未优化实现以及测试环境中缺少 NVLink，导致通信效率低下。

3.  **弥合 RL 中的概率鸿沟**：
    *   通过将 TBIK 集成到 vLLM 和 FSDP 中，并确保所有相关核函数（如 Attention、RMSNorm、RoPE）和超参数的一致性，甚至禁用了 vLLM 中的 chunked prefill 以避免隐式调度行为的影响。
    *   结果显示，vLLM (TP=4) 和 FSDP (TP=1) 之间预测的 `token` 概率达到了位级别相同，完全消除了训练和推理引擎之间的精度不匹配问题。


## FUSCO
FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

paper: https://www.arxiv.org/abs/2512.22036 清华,无问芯穹 ；2025.12.26

code: https://github.com/infinigence/FUSCO

中文解读：https://mp.weixin.qq.com/s/3SMExYsXdINkop8-_qxUVw 

核心是libfusco.so：基于NCCL修改（2.26， 2K c++/cuda)暴露API gather-send和scatter-recv（类似allreduce), + 1K python实现planner和LB；其他～500LOC接入到Megatron—LM和SGLang。

1. 💡 FUSCO针对大型Mixture-of-Experts (MoE) 模型中分布式**数据shuffle**效率低下的问题，通过融合数据转换和通信来解决MoE专家布局与设备布局冲突导致的冗余操作。
2. ⚙️ 该库引入了Data-Fused Communication Engine (**dComm**) 以及**Communication Planner**和Online Load Balancer机制，实现了细粒度的数据布局捕获和流水线式传输，**无需中间数据重新排列**。
3. ⚡️ FUSCO在通信基准测试中实现了高达3.84倍和**2.01倍的NCCL和DeepEP加**速，并在端到端MoE训练和推理任务中分别将延迟降低了1.17-1.39倍和1.06-1.19倍。

<img width="474" height="463" alt="image" src="https://github.com/user-attachments/assets/b07ea0fe-fca7-4b16-aee1-0bfe74c42953" />
<img width="412" height="479" alt="image" src="https://github.com/user-attachments/assets/9ac49d76-cb36-4db6-8586-506f1f127443" />
<img width="464" height="305" alt="image" src="https://github.com/user-attachments/assets/5fdfface-babd-4020-90a8-b34dda65c5a5" />
<img width="490" height="386" alt="image" src="https://github.com/user-attachments/assets/0573afd4-c707-4f51-a1a9-1dbf8459115e" />
<img width="481" height="343" alt="image" src="https://github.com/user-attachments/assets/ff449e3a-f26a-4614-b6c9-291a93030d98" />
<img width="473" height="314" alt="image" src="https://github.com/user-attachments/assets/9198b67a-c570-4711-89c9-99a57affc0e7" />
FUSCO 的核心思想是融合数据转换与通信 (fused data transformation and communication)。它不再将数据布局的改变视为本地预处理和后处理，而是将细粒度的数据布局语义直接嵌入到通信操作中。FUSCO 将待交换的结构化数据（如 MoE 中的 tokens）建模为一系列 segments，每个 segment 代表一个连续的逻辑工作单元。为了捕获结构化数据的移动，FUSCO 引入了 segment descriptor，它记录了每个 segment 的内存信息（地址和大小），指示如何从非连续内存中收集数据或将数据分散到指定非连续位置。
FUSCO 的核心是 Data-Fused Communication Engine (dComm)。dComm 能够根据 segment descriptor 高效地重排数据 segments，其通过流水线设计 (pipelined design) 实现内存操作与网络传输的重叠。具体而言，对于 intra-node 传输，dComm 利用 GPUDirect P2P 实现 GPU 到 GPU 的直接复制，将 descriptor 解释嵌入到复制路径中，从而在数据传输过程中同步完成布局转换，避免额外的内存拷贝。对于 cross-node 传输，dComm 以 slices（包含多个逻辑 segments）为单位发送数据，以摊销 descriptor 处理开销并保持 NIC 持续饱和。GPU 作为生产者，根据 descriptors 获取 segments 并执行布局转换（在从 GPU 全局内存到 ring buffer 的复制中 piggybacked），NIC 作为消费者，一旦数据就绪便通过网络传输。这种 producer-consumer 模式确保了数据准备与网络传输的完全流水线化。

## BuPO
Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies

paper: https://arxiv.org/abs/2512.19673 中科院自动化所，腾讯等 2025.12.22

code: https://github.com/Trae1ounG/BuPO

1. 💡 本文将大型语言模型（LLM）策略分解为内部层策略和内部模块策略，揭示了Transformer**残差流中隐藏的推理机制**。
2. 📈 通过对内部策略熵的分析，研究发现LLM具有**从探索到收敛的通用推理结构，而Qwen系列（尤其是Qwen3）展现出分阶段的、类似人类**的渐进式推理模式。
3. 🚀 受此启发，作者提出了Bottom-up Policy Optimization (BuPO)，一种新的强化学习范式，通过在早期训练阶段优化内部层策略，显著提高了LLM在复杂推理任务上的性能。
主要关注dense模型，FFN。没有MoE，没有Attn等机理。

<img width="372" height="541" alt="image" src="https://github.com/user-attachments/assets/2dc92ba2-f726-4f42-b0ab-fa85042e806f" />
这篇论文题为“Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies”，深入探讨了大型语言模型（LLMs）内部的策略演变机制，并基于这些发现提出了一种新颖的强化学习（RL）优化范式。
<img width="876" height="388" alt="image" src="https://github.com/user-attachments/assets/07b2ad36-1caa-4dcc-9dbc-a1e775454959" />
<img width="878" height="575" alt="image" src="https://github.com/user-attachments/assets/8e45b041-2342-47dd-97ff-83510a2cc441" />

**核心问题与背景**
现有的强化学习方法通常将LLM视为一个单一的、统一的策略，忽略了其复杂的内部机制。理解策略如何在Transformer的不同层和模块中演化，对于实现更具针对性的优化和揭示复杂的推理机制至关重要。论文指出，虽然奖励构造和熵正则化等表面算法设计是RLVR（Reinforcement Learning with Verifiable Rewards）研究的焦点，但LLM策略的内在性质及其内部残差流中的信息却常被忽视。Logit lens框架提供了初步见解，但并未从策略角度深入探讨。img width="879" height="380" alt="image" src="https://github.com/user-attachments/assets/49a2a533-bc50-4f56-bcd0-ce98938d5ef1" />

**论文贡献与核心方法**

1.  **内部策略分解 (Internal Policy Decomposition)**
    *   **基础洞察：** 论文基于两个核心洞察来分解LLM策略。
        *   **残差流的加性分解：** Transformer的残差连接天生支持加性分解。这意味着任何子模块（attention或FFN）的输入等于所有前序输出之和加上原始嵌入。第 $l$ 层的隐藏状态 $H_l$ 可以表示为初始嵌入 $H^{(0)}$ 加上所有前 $l$ 个注意力模块 ($A_i$) 和FFN模块 ($F_j$) 的输出之和：$H_l = H^{(0)} + \sum_{i=1}^l A_i + \sum_{j=1}^l F_j$。
        *   **隐藏状态与可采样策略的等价性：** 论文提出，任何内部隐藏状态 $H$ 与 unembedding matrix $E_u$ 结合都可以生成一个可采样的策略（即词汇空间上的概率分布）。最终的语言模型策略 $\pi_\theta$ 等同于 $P = \text{softmax}(\text{LN}(H^{(2L)})E_u^T)$。
    *   **定义：** 基于上述洞察，论文形式化定义了两种粒度的内部策略：
        *   **内部层策略 (Internal Layer Policy) $\pi_l^{\text{Layer}}$：** 利用来自每个层 $l$ 的隐藏状态 $H_l$ 与 $E_u$ 结合形成：$\pi_l^{\text{Layer}} \equiv P_l^{\text{Layer}} = \text{softmax}(H_l E_u^T)$。这捕获了截至第 $l$ 层的累积推理。
        *   **内部模块策略 (Internal Modular Policy) $\pi_l^{\text{Module}}$：** 将 $E_u$ 与特定模块（attention或FFN）的隐藏状态结合，隔离了它们各自的贡献：
            *   注意力模块：$\pi_l^{\text{ATTN}} = \text{softmax}(A_l E_u^T)$
            *   FFN模块：$\pi_l^{\text{FFN}} = \text{softmax}(F_l E_u^T)$
    *   **与Logit Lens的区别：** 论文强调其内部策略定义与Logit Lens不同，后者通常在应用层归一化（LN）之后进行解码以检查离散 token，而本论文的定义更侧重于可采样的概率分布，并且出于经验考量，有意省略了LN以获得更稳定的熵动态。

2.  **基于内部策略熵的分析 (Internal Policy Entropy-based Analysis)**
    *   **度量：** 论文采用熵作为主要指标，因为它与RL中的策略行为密切相关。内部策略熵定义为：$H_l^{\text{Layer}} = -\sum_{j=1}^{|V|} P_{l,j}^{\text{Layer}} \cdot \log(P_{l,j}^{\text{Layer}})$。
    *   **熵变化 (Entropy Change) $\Delta H_l$：** 为量化信息增益或损失，定义为模块输入和输出之间内部策略熵的差异，例如 $\Delta H_l^{\text{FFN}} = H_l^{\text{Output}} - H_l^{\text{Input}}$。
        *   $\Delta H_l^{\text{FFN}} > 0$ 表示探索空间扩展。
        *   $\Delta H_l^{\text{FFN}} \approx 0$ 表示内部知识整合。
        *   $\Delta H_l^{\text{FFN}} < 0$ 表示推理过程中的预测收敛。
    *   **发现：**
        *   **一致的内部推理结构：** 所有模型都展现出通用模式：早期层保持高熵以探索解决方案空间，而顶层收敛到接近零熵以进行最终预测。
        *   **独特的内部推理模式：**
            *   Llama系列：预测空间在最后三层才突然收敛，FFN的熵变化持续为正，表明探索贯穿始终，中间整合有限。
            *   Qwen系列（特别是Qwen3）：展现出渐进式的收缩。Qwen3的FFN表现出明显的“探索-整合-收敛” (EIC) 三阶段模式：较低层（1-6）熵增加（$\Delta H_l^{\text{FFN}} > 0$）进行探索，中间层（7-26）熵变化接近零（$\Delta H_l^{\text{FFN}} \approx 0$）整合参数知识，上层（27-36）熵减小（$\Delta H_l^{\text{FFN}} < 0$）逐渐收敛。这种模式与人类推理的阶段性过程类似。
        *   **残差余弦相似度分析：** Qwen3的自注意力持续增强残差流，与其正熵变化和扩展探索行为一致。FFN在不同阶段以不同方式调制残差流：较低层注入正交特征以支持探索，中间层抑制模糊信号并整合参数知识，上层则放大并整合特征以推动收敛。

3.  **自下而上的策略优化 (Bottom-up Policy Optimization, BuPO)**
    *   **动机：** 由于推理是自下而上逐步出现的，BuPO提出从底部视角进行优化。通过前期实验发现，单独优化内部策略会导致模型内部状态的显著特征细化，即优化后的底层会预先捕获高级推理信息，为后续推理提供更稳健的基础。
    *   **优化目标：** BuPO分阶段优化。首先优化内部层策略 $\pi_l^{\text{Layer}}$，然后优化整体语言模型策略 $\pi_\theta$。训练目标 $J_{\text{BuPO}}$ 定义为：
        $J_{\text{BuPO}}(\pi_\theta, \pi_l^{\text{Layer}}) = \begin{cases} J_{\text{InterGRPO}}(\pi_\theta, \pi_l^{\text{Layer}}), & \text{当前步 } s_{\text{cur}} \le s_{\text{inter}} \\ J_{\text{GRPO}}(\pi_\theta), & \text{当前步 } s_{\text{cur}} > s_{\text{inter}} \end{cases}$
        其中，$s_{\text{inter}}$ 是内部层策略的训练步数。
    *   **InterGRPO目标：** 在第一阶段，BuPO通过修改GRPO目标来直接优化选定的内部层策略 $\pi_l^{\text{Layer}}$。其目标函数为：
        $J_{\text{InterGRPO}}(\pi_\theta, \pi_l^{\text{Layer}}) = E_{q \sim Q,\{o_i\}_{i=1}^G \sim \pi_{\theta}^{\text{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \hat{r}_{i,t} \hat{A}_{i,t}, \text{clip}(\hat{r}_{i,t}, 1 - \epsilon, 1 + \epsilon) \hat{A}_{i,t} \right) \right]$
        其中，$\hat{r}_{i,t} = \frac{\pi_l^{\text{Layer}}(o_{i,t}|q,o_{i,<t})}{\pi_l^{\text{Layer,old}}(o_{i,t}|q,o_{i,<t})}$ 是针对当前优化策略 $\pi_l^{\text{Layer}}$ 的重要性采样比率，而采样仍然来自旧的整体语言模型策略 $\pi_{\theta}^{\text{old}}$。
    *   **梯度流控制：** 在InterGRPO优化第 $l$ 层时，梯度只会更新第 $0$ 层到第 $l$ 层的参数以及 unembedding matrix $E_u$，而不会影响更高层 ($k > l$) 的参数。这确保了内部策略优化对所选层及其以下层提供直接监督，强化基础推理能力。
    *   **实验结果：** 在MATH、AMC23、AIME24、AIME25等复杂推理基准测试中，BuPO在Qwen和Llama系列模型上始终优于GRPO、PPO等RL基线算法，平均性能表现卓越。例如，在Qwen3-4B上，BuPO在AIME24和AIME25上分别比GRPO提高了4.58和0.76个点。对Llama系列模型也观察到类似提升。Pass@K评估显示BuPO在不同K值下均保持领先。
    *   **训练动态分析：** BuPO训练初期能增强熵探索。适度的底层优化能够促进整体模型学习能力，但过度优化会导致模型崩溃。这表明选择合适的 $s_{\text{inter}}$ 至关重要。

<img width="884" height="429" alt="image" src="https://github.com/user-attachments/assets/d6093ca7-9f24-4154-beed-43703769eb44" />
![Uploading image.png…]()


## SpecFormer 
Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios

paper: https://arxiv.org/pdf/2511.20340 小米 武汉大学等 2025.11.25

code: https://github.com/ShiLuohe/SpecFormer 待开源 基于HF, Medusa

1. 📝 现有 Speculative Decoding (SD) 方法在存在**大batch场景下效率低下**，因为 batching 减少了可用计算资源，并且其 draft 模型过度依赖 position-dependent 参数，导致难以扩展。
2. 🚀 针对此问题，本文提出 SpecFormer，这是一种结合了 unidirectional 和 bidirectional attention 的新颖架构，旨在通过**更精准的 draft 生成** 在有限的 draft token 预算下实现**高效并行预测**。
3. ⚡ 实验证明，SpecFormer 在多种模型规模和 batch size 下均能提供稳定的加速效果，尤其在 draft token 预算受限时，其吞吐量 (TPS) 和转换效率 (κ-to-TPS) 均优于基线方法。
<img width="393" height="264" alt="image" src="https://github.com/user-attachments/assets/6c8d5604-4f95-4599-9de2-e2a02c444588" />
<img width="850" height="436" alt="image" src="https://github.com/user-attachments/assets/195e6ae2-c18f-43f0-a9c4-6c544ca826bb" />

<img width="673" height="370" alt="image" src="https://github.com/user-attachments/assets/7d4c2d0d-4665-4503-98e7-32679e1aa021" />
这篇论文提出了一种名为 SpecFormer 的新型 Speculative Decoding (SD) 架构，旨在解决现有 SD 方法在大型批处理（large-batch）场景下的效率瓶颈。

**1. 背景与问题**
大型语言模型（LLMs）的推理通常采用自回归（autoregressive）解码，即逐个生成 token。这种方式的缺点是算术强度（arithmetic intensity, AI）低，导致计算资源（尤其是 GPU）在等待内存-芯片数据传输时大量闲置。Speculative Decoding (SD) 通过利用这些闲置资源，同时生成多个 draft token 来加速推理，并通过一个大型模型（LLM）进行验证。
然而，主流的 LLM 推理系统广泛采用 batching 等技术来提高 GPU 利用率，这反而压缩了 SD 可用的闲置计算资源。在 batch size 增大的情况下，每个参数的计算强度更高，用于生成 draft token 的计算周期随之减少。现有的 SD 方法，无论是自回归还是非自回归，都通常依赖生成复杂且大规模的 draft tree 来提高预测准确率。这些方法的问题在于其大量的“位置依赖参数”（position-dependent parameters），这使得它们难以有效扩展以适应更少的可用计算资源和更严格的效率要求。因此，如何在低验证资源和低调度成本下进行 SD 成为一个重要研究问题。

**2. SpecFormer 方法**
SpecFormer 的核心思想是提高 draft generation 模型的预测能力，使其在有限的 draft token 预算下也能高效运行。它结合了自回归模型从整个输入序列中提取信息的能力，以及非自回归模型并行生成 token 的优势。该架构集成了单向（unidirectional）和双向（bidirectional）注意力机制，从而消除了对大型 prefix tree 的依赖。
<img width="1509" height="471" alt="image" src="https://github.com/user-attachments/assets/0357157d-998d-4325-a84d-d0e44f0a97b0" />
<img width="1518" height="263" alt="image" src="https://github.com/user-attachments/assets/dc9fd3c6-b782-4557-9267-50e30a8eafc8" />


**2.4 实现优化**
*   **高效 Grouped RMS Norm**: 为解决 RMS Normalization 的性能瓶颈，使用了定制的 Triton GPU kernel。
*   **批内梯度累积 (Intra-batch Gradient Accumulation)**: 在 LM Head 周围采用梯度累积策略，顺序计算每个位置的损失，避免了昂贵的 softmax 存储开销。

**3. 实验与结果**
*   **训练语料**: 在 UltraChat-200K (UC) 数据集上进行训练，并强调了“自蒸馏”（self-distillation）的重要性。通过使用基础 LLM 重新生成补全部分，确保 draft 模型学习到的分布与基础模型严格对齐。
*   **基础 LLM**: 评估了 Qwen2.5-3B, Qwen3-8B, Qwen3-14B 和 LLaMA-3.1-8B。
*   **评估**: 使用 UC 测试集以及 MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DM 等流行基准进行评估，关注无损加速下的性能，即保持输出与原模型一致。

**3.1 吞吐量比较 (Table 1)**
在受限的 draft token 预算下，SpecFormer 在不同 batch size 设置下始终优于基线方法 (HASS, EAGLE-3)。论文指出，基线方法未达到其宣称的性能，是因为在实验中强制限制了 draft token 预算，以模拟计算冗余有限的场景。SpecFormer 在高吞吐量的同时，不需要大量的 draft tokens，这得益于其卓越的预测能力和更高的 $\kappa$-to-TPS 转换效率，表明其非自回归设计带来了更高的算术强度和更低的平均每 token 开销。

**3.2 特殊案例研究**
*   **自蒸馏的影响 (Table 2)**: 结果显示，没有自蒸馏的模型加速效果可以忽略不计，因为其学习到的 token 分布与基础模型不符。自蒸馏是确保 draft 模型与基础模型输出严格对齐的关键步骤。
*   **基础 LLM 大小 (Table 3)**: 随着模型尺寸增大，预测器准确预测未来 token 的能力有所削弱，导致加速收益略有下降（例如，4B 模型加速比为 1.56×，而 14B 模型为 1.47×）。然而，较大模型展现出更有利的 $\theta$ 值（$\kappa$-to-TPS 转换比），意味着预测器引入的相对开销更小。这归因于较大模型层数增加使得预测器参数占比更小，以及大模型中更大的权重矩阵稀释了调度开销。

**3.3 模块消融研究 (Table 4)**
在 Qwen3-4B 模型上进行消融研究，结果表明：
*   双向注意力对模型能力有提升，但提升不显著。但考虑到其对推理时间影响可忽略，保留该结构。
*   Positional FFN 对性能提升贡献很大，符合预期（因其参数量较大）。
*   更大的模型尺寸能带来显著的性能提升。这暗示，在基础模型总参数量更大的情况下，扩大 draft 模型尺寸可以抵消其负面影响，并提升 draft 模型预测能力。

**4. 结论**
SpecFormer 提出了一种创新的 SD 方法，通过结合单向和双向注意力机制，在有限的 draft token 预算下实现了高效的未来 token 并行生成。该方法通过从完整上下文提取信息，并以参数高效的方式注入位置信息，解决了现有 SD 方法在 large-batch 场景下因计算资源受限和位置依赖参数过多而导致的扩展性问题。实验证明，SpecFormer 在不同 batch size 和不同 LLM 规模下均能提供持续的加速效果，且对较小模型尤其有效，为 LLM 推理的扩展性设定了新标准。

## Step-level verifier
**Step-level verifer-guided hybrid test-time scaling for LLM**
paper: https://aclanthology.org/2025.emnlp-main.931 EMNLP25 东北大学 字节等

code: https://github.com/Lucky-259/Hybrid_TTS

中文解读：https://mp.weixin.qq.com/s/RUKY-1_8Vh3d047PsE3_Hw?poc_token=HKYSUWmj_5Obo6wRroDVmf0mgxYGriyB_Vv7gaux

1. ✨ 本文提出了一种名为 Step-level Verifier-guided Hybrid Test-Time Scaling 的新推理范式，旨在通过结合**多种训练无关的 Test-Time Scaling (TTS) 方法来提升大型语言模型在推理任务上的探索性能**。
2. 💡 该方法首先引入了 Conditional **Step-level Self-refinement** 以实现细粒度顺序缩放 +～3分；并在此基础上，将并行缩放（如 Best-of-N 和 MCTS）与该顺序缩放方法在**步级别进行融合**，所有过程均由高质量的 **Process Reward Model** (PRM) 指导。
3. 🚀 广泛的实验表明，该混合策略显著提升了不同规模和系列的 instruction-tuned LLM 的推理能力，例如在 GPQA Diamond 数据集上使 Qwen2.5-3B-Instruct 模型超越了经 RL 增强的 DeepSeek-R1-Distill-Qwen-7B 模型，凸显了训练无关 TTS 的巨大潜力。

按步生成： 模型每生成一个推理步骤就暂停
实时验证： PRM 给这一步打分
按需反思： 只有分数低（没信心）时才触发反思；如果分数高，直接跳过，避免画蛇添足
择优录取： 只有修正后的步骤得分更高，才采纳；否则保留原样
<img width="1080" height="794" alt="image" src="https://github.com/user-attachments/assets/d04e421d-377e-4626-99cd-473225d7ed46" />

MCTS (蒙特卡洛树搜索) 框架下的深度扩展，遵循 “Best-of-the-Best” 原则，每一个推理步骤都经过了“千锤百炼”：

广度探索：通过 Best-of-N 采样多个candidate steps

深度优化：对 Best candidate step 进行 Conditional Step-level Self-Refinement

动态选择：基于 PRM 评分选择 Best step 作为叶节点，以构建最优推理路径

就像是让模型在每走一步棋时，既考虑了多种走法 (Best-of-N)，又对选定的走法进行了仔细推敲 (Self-refinement)，不仅保证了广度，也保证了深度。

https://mmbiz.qpic.cn/sz_mmbiz_png/djrV8EtuicU2kuDtnVnxLa3IfJGFBQ0lSbludFBHzicdR8SCEGdhACT8IiclvQxWmibvwKIfBHUd3VhEToqbUIT4Rg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4
## UCCL-EP 
UCCL-EP: Portable Expert-Parallel Communication

paper: https://arxiv.org/abs/2512.19849 伯克利，AWS 等，2025.12.22 

code: https://github.com/uccl-project/uccl/tree/main/ep

在DeepEP代码基础上修改，增加2万行c++和1K行python，API保持和DeepEP兼容。支持了2类GPU（NV，AMD）X 3家网卡（NV Mellanox，AWS EFA，Broadcom），支持LL和HT模式，总体吞吐相比原生DeepEP接近（-5%）； 延迟差于PPLX。
现在移植到新的GPU/NIC只需要3人月 ：）

1. 📄 本文提出UCCL-EP，旨在解决DeepEP等现有高性能专家并行 (EP) 通信系统在**异构GPU和NIC平台之间移植性差**的问题，其根源在于**GPU直接操作NIC导致紧密耦合**。
2. 💡 UCCL-EP的核心思想是**利用CPU作为中间层解耦GPU与NIC**，通过一个**高吞吐量GPU-CPU控制通道**，将**GPU发起的token路由指令传递给多线程CPU代理**，再由**代理执行GPUDirect RDMA操作**。
3. 🚀 UCCL-EP在NVIDIA+EFA和AMD+Broadcom等异构平台上实现了DeepEP级别的性能，在dispatch和combine吞吐量上最高提升2.1倍，并使SGLang和DeepSeek-V3训练吞吐量分别提高了40%和45%。
<img width="519" height="246" alt="image" src="https://github.com/user-attachments/assets/2b98d3fe-cd39-46bd-ace6-58e593fd093e" />
<img width="756" height="321" alt="image" src="https://github.com/user-attachments/assets/43713a75-9185-445d-9092-a0072b8f9504" />

本文提出了一种名为UCCL-EP的通信系统，旨在解决稀疏MoE模型中`Expert Parallelism (EP)`通信的`可移植性 (portability)`问题，同时保持高性能。
<img width="1078" height="321" alt="image" src="https://github.com/user-attachments/assets/263f858e-27e5-46ba-bab2-d3298c59702d" />
<img width="1229" height="229" alt="image" src="https://github.com/user-attachments/assets/06e47631-b6c1-4e34-be4d-7a5eb06808d2" />

**问题背景：**
大型语言模型（LLMs），如`DeepSeek-V3`，越来越多地采用`Mixture-of-Experts (MoE)`架构，以在保持计算效率的同时实现巨大的参数容量。`MoE`层中的`专家并行 (EP)`需要将`token activations`在GPU之间进行`all-to-all`通信，这构成了`MoE`性能的关键瓶颈。传统的`all-reduce`或`pipeline-parallel`通信模式不适用于`MoE`的细粒度（`fine-grained`）、不规则（`irregular`）和`token-level`传输特性。`DeepEP`等`state-of-the-art`系统通过`GPU-initiated token-level communication`解决了这一问题，利用NVIDIA的`IBGDA (InfiniBand GPUDirect Async)`技术，允许GPU直接操作`RDMA NICs`。这种方法实现了高效的`token`去重（`deduplication`）和分层`reduce`，但其设计导致了极差的`可移植性`。核心问题在于GPU直接写入`NIC`的`driver/MMIO`接口，造成GPU与`NIC`之间的紧密耦合，以及GPU内核对网络层严格的`排序 (ordering)`和交付语义假设（例如`write-then-atomic`），这在异构`NICs`（如AWS EFA）上难以满足。现有解决方案需要`O(m × n)`的开发工作来支持`m`种GPU和`n`种`NIC`。
<img width="767" height="386" alt="image" src="https://github.com/user-attachments/assets/fa0f9c88-7a0c-4616-894c-7228c612e919" />

**核心思想与方法：**
UCCL-EP的核心思想是利用CPU作为`可移植`的中间层，打破GPU和`NIC`之间的紧密耦合。CPU通过`libibverbs`库与各种`NIC`兼容，并通过`PCIe`/`NVLink-C2C`等高速互连与GPU通信。UCCL-EP将通信的`启动 (initiation)`与`执行 (execution)`解耦：GPU负责启动细粒度的`token`控制，但将实际的通信任务委托给`host CPU`上的`CPU proxy`。这样，UCCL-EP实现了`O(m)`的`可移植性`工作量。

**主要设计和技术细节：**
<img width="639" height="393" alt="image" src="https://github.com/user-attachments/assets/925a5d6f-2d39-48ac-a433-ede8e99bb9b7" />
<img width="587" height="291" alt="image" src="https://github.com/user-attachments/assets/aed4e3e6-29d3-4983-9201-f048be1ac004" />

1.  **高效GPU-CPU通信通道 (`Efficient CPU-GPU Communication Channel`)：**
    *   **`TransferCmd`：** GPU将轻量级、固定大小的命令描述符`TransferCmd`（128位，16字节）排队到共享的`lock-free FIFO channels`中。16字节可通过单个GPU指令和`MMIO doorbell`写入，减少开销。
    *   **FIFO通道设计：** GPU写入`FIFO`尾部，CPU`proxy`作为消费者从`FIFO`头部读取。通道大小由`kMaxInflight`参数限制，用于控制GPU发送速率。
    *   **内存一致性：** `FIFO`的头部（`head`）放置在CPU内存，尾部（`tail`）放置在GPU内存，以优化各自的访问效率。通过绕过GPU硬件缓存和CPU L2缓存刷新确保内存一致性。
    *   **GPU侧争用：** 采用多个`FIFO channels`，并引导不同GPU线程写入对应的通道，减少争用。`TransferCmds`在同一`FIFO`内保证有序。
    *   **通道API：**
        *   CPU侧：`Poll`（读取但不移除命令），`Pop`（移除命令）。
        *   GPU侧：`Push`（推入命令并获取`Idx`），`Check-completion(Idx)`（检查命令是否被CPU侧移除）。
    *   **`TransferCmd`类型：**
        *   `Write`：委托CPU`proxy`执行写入请求（包含源/目的地址偏移、长度、目的`rank`）。
        *   `Atomics`：委托CPU`proxy`执行独立的`atomic`操作（包含目的偏移、`atomic`值、目的`rank`）。
        *   `Drain`：委托CPU`proxy`清空`RDMA completion queue`，确保所有未完成的`RDMA`操作完成。
        *   `Barrier`：委托CPU`proxy`建立同步`barrier`（支持`all-peer barrier`和`same-rail barrier`）。

2.  **灵活的CPU代理 (`Flexible CPU proxy`)：**
    *   **多线程设计：** 每个GPU对应一个CPU`proxy`，该`proxy`包含多个不共享状态的线程，实现并发处理，提高小消息吞吐量。
    *   **对称内存 (`Symmetric Memory`)：** CPU`proxy`在初始化时注册内存区域并交换基地址，实现了对称内存的抽象。GPU只需传递偏移量，CPU`proxy`负责地址转换，减少控制消息大小，并消除对`NVSHMEM`等特定供应商库的依赖。
    *   **处理交付语义 (`Addressing Delivery Semantics`)：**
        *   **问题：** 异构`NICs`可能不保证`RDMA`写入的`有序交付 (in-order delivery)`。
        *   **解决方案：** CPU`proxy`在每次`RDMA`写入中通过`immediate data`（`RoCEv2`包头中的32位字段）嵌入序列号。接收方CPU`proxy`解析`immediate data`，如果消息乱序到达，则暂时将其`atomic`消息缓冲在`control buffer`中。只有当所有先前的写入都完成并应用后（例如，`Low-Latency (LL)`模式中的`partial completion fence`，或`High-Throughput (HT)`模式中的`per-channel locally ordered`），才应用`atomic`更新。这种`receiver-side`的语义强制比`sender-side`更高效。

**实现细节与可移植性：**

*   UCCL-EP在`DeepEP`基础上扩展，保持API兼容性。
*   通过去除GPU供应商特定的软件栈（如`NVSHMEM`）和将CUDA特定的`PTX intrinsics`迁移到`ROCm`替代方案，并适应AMD `wavefront`编程模型，实现了对NVIDIA和AMD GPU的广泛支持。
*   **EFA支持：** EFA `NICs`不支持硬件`RDMA atomics`。UCCL-EP通过软件模拟`atomics`：发送方发出负载写入后，跟着一个带有编码`counter`或`flag`的`immediate value`的微小`RDMA`写入。接收方CPU`proxy`检测到`immediate data`后更新主机内存上的本地`completion counter`。

**性能评估：**

*   **测试平台：** 涵盖了NVIDIA GPU（H200, B200, H100, GH200）和AMD GPU（MI300X），以及AWS EFA、NVIDIA ConnectX-7 IB和Broadcom Thor-2等多种`NICs`。
*   **微基准测试：**
    *   **NVIDIA+EFA平台 (AWS)：** 在`dispatch`和`combine`吞吐量上，UCCL-EP在处理大量`tokens`时，比`PPLX`（第二好的`EP`解决方案）性能提升高达2.1倍。对于小批量`tokens`，`PPLX`可能略有优势，因为UCCL-EP/DeepEP默认发送7KB的`token`粒度消息，而EFA固件对小消息处理效率不高。
    *   **NVIDIA+CX7 IB平台：** UCCL-EP在`HT`模式下实现了与原始`DeepEP`相当的性能（`dispatch`性能在5%以内），同时优于`PPLX`和`CPU-assisted IBGDA`。
    *   **GH200 (NVLink-C2C)：** UCCL-EP在`LL`模式下实现了比原始`DeepEP`更低的传输延迟，表明其在`cache-coherent CPU-GPU`互连上表现出色。
    *   **AMD+Broadcom/CX7平台：** UCCL-EP在异构`NICs`（Broadcom和IB）上表现相似，验证了其在AMD平台上的`可移植性`。
*   **应用性能：**
    *   **SGLang推理 (NVIDIA+EFA)：** 在`SGLang`中，UCCL-EP在`DeepSeek R1`和`Qwen3`模型上的`token`吞吐量比`NCCL`提高高达40%，且在更大`EP`配置下表现更佳。
    *   **DeepSeek-V3训练 (AMD Primus/Megatron-LM)：** 在16节点`AMD+Broadcom`平台上，UCCL-EP使`DeepSeek-V3`训练吞吐量比`RCCL`提高了高达45%。
*   **设计剖析：**
    *   UCCL-EP FIFO的延迟远小于网络延迟，能够处理高`QPS`（例如8 Mops）。
    *   CPU线程数量对性能有显著影响，增加CPU线程（例如到4个）可显著提升性能。CPU利用率适度增加。
<img width="589" height="469" alt="image" src="https://github.com/user-attachments/assets/c10c8d3f-6592-49e2-8095-2aa20d5a5013" />
<img width="581" height="285" alt="image" src="https://github.com/user-attachments/assets/68fbfbde-50cd-47f1-9277-e9dac2cc98f5" />

**讨论与未来工作：**
UCCL-EP的设计为未来的改进提供了基础，例如在CPU`proxy`中实现更灵活的`拥塞控制 (congestion control)`和`流控制 (flow control)`机制，以处理`tail latency`和`incast`问题。此外，CPU`proxy`可以支持`弹性EP (Elastic EP)`，**在不影响GPU内核逻辑的情况下处理故障和伸缩事件**。未来的优化可能包括进一步优化`LL`模式下`token`打包以提升小消息处理效率，以及扩展到`TPUs`和`AWS Trainium`等其他AI加速器。

## H-Neurons幻觉来源与控制
H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs

paper: https://arxiv.org/pdf/2512.01797 清华大学 2025.12.2 
未见开源
中文解读：https://mp.weixin.qq.com/s/JESXy8qHFSdGJ1ku4seTzg

1. 🧠 研究发现，LLM中一个极其稀疏的神经元子集（少于总数的0.1%），即H-Neurons，能够可靠地预测幻觉的发生，并在各种场景下展现出强大的泛化能力。
2. 💡 通过受控干预，研究表明H-Neurons与模型的过度顺从行为（over-compliance）存在因果关系，放大这些神经元的激活会系统性地增加模型对错误前提、误导性上下文和有害指令的顺从性。
3. 🕰️ 进一步的追溯分析揭示，H-Neurons起源于预训练阶段，并在指令微调（instruction tuning）过程中保持了其预测能力，这表明幻觉行为深植于LLM的基本训练目标之中。


## ThinkARM
Schoenfeld's Anatomy of Mathematical Reasoning by Language Models

paper: https://arxiv.org/abs/2512.19995 马里兰大学等 2025.12.23

code: https://github.com/MingLiiii/ThinkARM

中文解读：https://mp.weixin.qq.com/s/otEp-SPfRiDfoTHG4k0zzw 

对LLM数学推理的结构分析和深入挖掘。采用了 Alan Schoenfeld 于 1985 年提出的片段理论（Episode Theory）。Schoenfeld 的理论最初用于分析人类（学生与数学家）解决数学问题的过程，将其概念化为一系列功能性片段。
研究将大语言模型（LLM）的推理思维链（CoT）抽象为理解 规划 执行 探索 验证/反思等基础能力，具体包括 Read、Analyze、Plan、Implement、Explore、Verify、Monitor 和 Answer 等八个功能性片段。
<img width="944" height="330" alt="image" src="https://github.com/user-attachments/assets/38b85e68-2370-4a6f-a85c-6e54e4867d35" />

通过对 15 个模型（包括 DeepSeek-R1、OpenAI o1/o3-mini 等）的推理轨迹进行句子级标注与量化分析，研究揭示了推理模型与非推理模型在结构上的本质差异：
- 推理模型呈现出一种从抽象分析到具体执行再到评估控制的“心跳”模式（Heartbeat Pattern），并包含大量的迭代循环。
- Explore（探索）片段**是不确定性的关键分支点**，其**后续流向（转向 Monitor 还是继续 Implement）与最终答案的正确性高度相关**。
- 此外，**效率导向的模型优化（如蒸馏或长度惩罚）往往倾向于压缩评估性片段**，从而改变了基础模型的推理拓扑结构。
<img width="1044" height="577" alt="image" src="https://github.com/user-attachments/assets/fbc2f1f2-4e83-428d-aa25-a1545ee31c6a" />
<img width="1034" height="490" alt="image" src="https://github.com/user-attachments/assets/ef45eb74-d343-4f23-b3f9-eb35b6493cb0" />

## Mesh-Attention
Mesh-Attention: A New Communication-Efficient Distributed Attention with Improved Data Locality

https://arxiv.org/pdf/2512.20968v1 清华 普渡 字节等 2025.12.24 
基线不做overlap？自己的做overlap？

1. 💡 Mesh-Attention通过引入新的矩阵基（AM）模型，重新思考分布式注意力机制设计空间，将二维计算块（tiles）分配给每个GPU，旨在解决现有分布式注意力（如Ring-Attention）因过度通信而限制LLM上下文窗口扩展的问题。
2. ⚙️ 该方法通过优化通信-计算（CommCom）比率显著降低通信复杂度，并提出一种贪婪算法来高效调度GPU内的计算与通信，从而实现最大化重叠，同时支持前向/后向传播和因果掩码。
3. 🚀 实验结果显示，与Ring-Attention相比，Mesh-Attention在**多达256个GPU上实现了高达3.4倍的加速（平均2.9倍）和高达85.4%的通信量减少**（平均79.0%），展现出卓越的扩展性和效率。

<img width="1077" height="325" alt="image" src="https://github.com/user-attachments/assets/d71ce5d5-4ab6-4828-b69d-401def4320de" />
<img width="699" height="444" alt="image" src="https://github.com/user-attachments/assets/cc69901a-01a4-48f6-a486-8754c4016936" />
<img width="523" height="323" alt="image" src="https://github.com/user-attachments/assets/1d876924-05bd-4841-a62a-558212f6071f" />
<img width="541" height="477" alt="image" src="https://github.com/user-attachments/assets/bf89a707-6d00-46d0-bdc8-28b93f5c23c4" />
<img width="539" height="587" alt="image" src="https://github.com/user-attachments/assets/42292be8-8caf-4b72-9302-1a3478c8f8dd" />
<img width="531" height="269" alt="image" src="https://github.com/user-attachments/assets/22dccd8e-15fc-4b09-80ec-4347006f69d5" />


## BLASST
BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

https://arxiv.org/pdf/2512.12087 2025.12.12 NVDIIA, MIT

kernel集成到flashInfer 但未见开源；e2e评测基于 https://github.com/NVIDIA-NeMo/Skills

1. 💡 BLASST 是一种即插即用的 sparse attention 方法，通过利用 FlashAttention 的在线 softmax 机制和固定阈值，在不进行预计算或使用代理分数的情况下，动态剪枝 negligible attention scores，从而跳过计算和 Value block 加载。
2. 🚀 该方法在保持高准确度的同时，在现代 GPU 上实现了显著的加速，Prefill 阶段最高可达 **1.62×，Decode 阶段最高可达 1.48×**，并在所有 attention 变体和预填充/解码阶段提供统一的解决方案。
3. 🛠️ 为确保其鲁棒部署，BLASST 引入了**自动校准程序以确定最优阈值**，并探索了 sparsity-aware training 作为扩展，进一步提升了 accuracy-sparsity frontier。

<img width="396" height="533" alt="image" src="https://github.com/user-attachments/assets/078ecc45-d707-4a18-ba21-de5823451175" />
<img width="406" height="487" alt="image" src="https://github.com/user-attachments/assets/4e6651a3-1e4d-4c09-9a54-c4b2dad50b1c" />
<img width="1102" height="468" alt="image" src="https://github.com/user-attachments/assets/aae9b41e-393c-41e9-b575-d914b1b7266f" />
<img width="1090" height="264" alt="image" src="https://github.com/user-attachments/assets/4cf89e71-b30a-47ca-8586-206f91d263ba" />
<img width="1092" height="498" alt="image" src="https://github.com/user-attachments/assets/b7d03cb8-f46a-4e06-918c-89dfe9535971" />
<img width="526" height="416" alt="image" src="https://github.com/user-attachments/assets/4e7066fa-f943-441e-8816-0cd1d316dc3d" />
![Uploading image.png…]()

**核心方法学：动态 Attention 稀疏性与 Softmax 阈值**

BLASST 的核心思想在于，在 FlashAttention 的块级在线 Softmax 计算过程中，可以**利用已有的信息动态识别并跳过那些对最终输出贡献可以忽略的** Attention 块。

1.  **关键洞察与剪枝原理**:
    在 FlashAttention 的分块计算中，对于每个 Query $Q_i$ 而言，其对 Key-Value 块 $K_j, V_j$ 的 Attention 分数 $S_{ij} = Q_i K_j^\top$ 会参与 Softmax 运算，最终得到 Attention 权重 $P_{ij}$。标准的 Softmax 运算定义为：
    $P_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}$
    FlashAttention 为了数值稳定性和内存效率，采用了在线 Softmax 算法，该算法在处理每个块时会维护一个运行最大值 $m^{(j)}_i$ 和一个行和 $l^{(j)}_i$。对于当前处理的块 $j$，计算其内部 Attention 分数的局部最大值 $\tilde{m}^{(j)}_i = \max_{p,q} S_{ipq}$。
    BLASST 的关键观察是，如果一个块的局部最大值 $\tilde{m}^{(j)}_i$ 显著小于当前的运行最大值 $m^{(j)}_i$，即满足条件 $\tilde{m}^{(j)}_i - m^{(j)}_i < \ln(\lambda)$，其中 $\lambda$ 是一个预设的阈值。根据指数函数的性质，这意味着 $\exp(\tilde{m}^{(j)}_i - m^{(j)}_i) < \lambda$。由于 Softmax 的指数项通常会除以一个大的归一化因子（$\exp(m^{(j)}_i)$），如果一个块的最大 Attention 分数与行最大值之差小于 $\ln(\lambda)$，那么该块内所有分数经 Softmax 后的值都将非常接近零，对最终 Attention 输出的贡献微乎其微。

2.  **剪枝操作与节省**:
    当满足上述剪枝条件 $\tilde{m}^{(j)}_i - m^{(j)}_i < \ln(\lambda)$ 时，BLASST 会对当前 Attention 块跳过以下三种高开销操作（对应于 Algorithm 1 的第 7-12 行）：
    *   **计算节省（CUDA Cores）**:
        *   跳过计算 Attention 权重 $\tilde{P}_{ij} = \exp(S_{ij} - m^{(j)}_i)$ 所需的昂贵 $\exp(\cdot)$ 操作。
        *   跳过用于归一化 Attention 权重的行和（rowsum）规约操作。
        *   这节省了数千个 CUDA core 指令。
    *   **计算节省（Tensor Cores）**:
        *   跳过 Attention 权重与 Value 块的矩阵乘法 $\tilde{P}_{ij} V_j$ (BMM2)。在 Prefill 阶段，该操作是计算瓶颈。
    *   **内存带宽节省**:
        *   跳过从高带宽内存 (HBM) 加载 Value 块 $V_j$ 到 SRAM 的操作。这在 Decode 阶段尤为关键，因为 Attention 通常是内存瓶颈。

**内核设计**

BLASST 针对 Prefill 和 Decode 阶段的不同计算特性，设计了专门优化的 CUDA 内核。设计目标是最小化开销并复用 FlashAttention 已计算的统计信息。

*   **Prefill 内核（计算密集型优化）**:
    Prefill 阶段通常受限于计算吞吐量（CUDA Cores 和 Tensor Cores），而非内存带宽。因此，BLASST Prefill 内核主要跳过 Softmax 计算和 MMA (Matrix Multiply Accumulate) 操作。尽管 Value 块仍会从 HBM 加载，这是因为内存带宽不是瓶颈，且预取流水线受益于可预测的内存访问模式，同时有条件地加载 Value 会引入额外延迟。这种优化使得 Prefill 速度提升几乎与稀疏度呈线性关系。

*   **Decode 内核（内存密集型优化）**:
    Decode 阶段通常受限于 HBM 带宽，因为 Attention 涉及单个 Query 与所有 Key 的比较（KV Cache）。BLASST Decode 内核的核心优化是跳过内存密集型的 Value 矩阵 $V_j$ 加载。通过减少内存流量，显著提升了速度。对于像 MLA (Multi-head Latent Attention) 这样在 Decode 阶段也计算密集型的 Attention 机制，BLASST 还会额外跳过 Softmax 操作以进一步加速。

**关键技术与增强**

1.  **稀疏度校准**:
    为了在不同上下文长度下保持一致的稀疏度和准确性，BLASST 引入了自动化校准程序。研究发现，最优阈值 $\lambda$ 与上下文长度 $L$ 之间存在简单的反比关系：$\lambda = a/L$，其中 $a$ 是模型特定的常数。该校准过程通过在不同上下文长度下，经验性地寻找达到目标稀疏度 $S$ 的最佳阈值 $\lambda_{best}$，然后对数据点 $(1/L_k, \lambda_{best})$ 进行线性回归来确定参数 $a$。这确保了在不同上下文长度下，可以预测性地控制稀疏度并获得一致的计算加速。

2.  **稀疏感知训练 (Sparsity-Aware Training)**:
    BLASST 进一步探索了稀疏感知训练作为其自然延伸。在微调阶段，在前向传播中应用 BLASST 阈值剪枝。由于跳过的块在后向传播中不会收到梯度，这鼓励模型在训练过程中将重要信息集中到高分 Attention 块中，从而使其 Attention 模式更适应稀疏性。这种方法不需要架构修改或辅助损失函数，只需在训练时使用与推理时相同的稀疏 Attention 机制。


## who is Adam?
Blog: https://www.notion.so/sagnikm/Who-is-Adam-SGD-Might-Be-All-We-Need-For-RLVR-In-LLMs-1cd2c74770c080de9cbbf74db14286b6

中文：https://mp.weixin.qq.com/s/kDnLL7qZyGw7P7_w5w3XXg

## Prompt Repetition
Prompt Repetition Improves Non-Reasoning LLMs
https://arxiv.org/abs/2512.14982 Google 2025.12.17

1. 💡 提出了一种名为 Prompt Repetition 的方法，即将输入查询 `<QUERY>` 重复为 `<QUERY><QUERY>`，结果显示在非reasoning模式下，该方法能显著提升 Gemini、GPT、Claude 和 Deepseek 等主流 LLM 在多项基准测试上的表现，**且不增加生成令牌或延迟**。reasoning模式下，中性略提升。
2. ⚙️ 这种方法通过允许**每个提示令牌关注其他所有提示令牌来提高性能**，并且由于它仅影响并行化的 prefill 阶段，因此高效且可直接集成到现有系统中。
3. 🎯 实验证明，Prompt Repetition 在47/70个**测试中表现出统计学上的显著优势**，尤其在“选项在前”的多项选择题和自定义任务上效果明显，而在启用推理时，其影响则趋于中性或略有积极。
<img width="586" height="328" alt="image" src="https://github.com/user-attachments/assets/0e88e33e-3346-4640-8d7b-5291f5019e13" />
<img width="724" height="454" alt="image" src="https://github.com/user-attachments/assets/8cb3d2c2-a4ae-42f4-95f3-4606f6a2d660" />
<img width="736" height="406" alt="image" src="https://github.com/user-attachments/assets/dfffcd31-cd53-47c1-ae7a-5e32a9797fb2" />

几乎是“免费”的。
• 生成长度不变：重复输入Prompt并**不会让模型生成的答案变长**。
• 延迟几乎不变：虽然输入的Prompt变长了，但这部分计算发生在**预填充阶段**（Prefill Stage）。现代推理引擎对预填充阶段有极高的并行优化能力。

## RePo
RePo: Language Models with Context Re-Positioning 
https://www.arxiv.org/abs/2512.14391 SakanaAI, 2025.12.16
https://github.com/SakanaAI/repo

中文解读：https://mp.weixin.qq.com/s/NRgtlMGeSoFdKWHIUwpzQg

1. ✨ 本文提出了一种名为REPO（Context Re-Positioning）的新机制，通过可微分模块fϕ根据令牌的内在关联性在连续、非线性空间中动态分配位置，以减少大型语言模型（LLMs）中固定上下文结构造成的额外认知负荷。
2. 🚀 在**OLMo-2 1B**模型上进行的持续预训练实验表明，REPO在处理嘈杂上下文、结构化数据和长上下文任务时显著提升了模型性能，同时在通用短上下文任务上保持了竞争力。
3. 💡 详细分析揭示，REPO能够成功地将更多注意力分配给遥远但相关的信息，打破了局部性偏见，并学习到能够捕捉输入上下文内在结构（如few-shot示例分割）的密集非线性位置模式。
<img width="668" height="212" alt="image" src="https://github.com/user-attachments/assets/2508b6b6-73c3-464f-83f8-1bdff4d47e9a" />
本文提出了一种名为 REPO（Context Re-Positioning）的新机制，旨在通过重新组织上下文来降低大型语言模型（LLMs）的额外认知负荷。现有的 LLMs 架构通常采用刚性且固定的上下文结构，通过分配线性或恒定的位置索引来处理信息。作者认为，这种非信息性的结构增加了外部认知负荷，消耗了有限的工作记忆容量，而这部分容量本应分配给深度推理和注意力分配。

为了解决这个问题，REPO 引入了一个可微分模块 $f_\phi$，该模块根据上下文依赖性为每个 token 分配位置，而非依赖预定义的整数范围。$f_\phi$ 可以为 LLM 的每个注意力头独立学习。

**核心方法 (REPO) 详解：**

REPO 模块 $f_\phi$ 旨在根据 token 的相关性为其分配更合适的位置。它包含两个主要组件：

1.  **位置表示 (Position Representation)**：
    此组件负责从 token 的隐藏状态中显式提取位置信息。具体实现上，它使用一个轻量级的 SwiGLU 子层来完成此任务：
    $r_i = \text{Swish}(h_i W_g) \odot (h_i W_c)$
    其中，$r_i \in \mathbb{R}^{d_p}$ 是 token $x_i$ 的位置表示，$h_i \in \mathbb{R}^d$ 是其隐藏状态，$W_g, W_c \in \mathbb{R}^{d \times d_p}$ 是线性变换矩阵，$\text{Swish}(\cdot)$ 是激活函数。为了保持轻量级，通常设置 $d_p < d$。

2.  **位置分配 (Position Assignment)**：
    此组件在每个注意力头中为 token $x_i$ 分配一个新的位置值 $z_i$。它通过一个线性变换实现：
    $z_i = r_i W_z$
    其中，$W_z \in \mathbb{R}^{d_p \times 1}$ 是一个线性变换矩阵。

通过结合这两个组件，REPO 模块 $f_\phi$ 的形式化定义为：
$f_\phi(h_i) = \text{Swish}(h_i W_g) \odot (h_i W_c) W_z$

当与现代可微分的位置编码方法（例如 RoPE）结合使用时，注意力分数的计算变为：
$A_{\text{Repo}_{i,j}} = q_i^\top g_\theta (z_j - z_i) k_j$
其中，$g_\theta$ 是位置编码函数。REPO 模块在实践中从第 $l$ 层（例如 $l=5$）开始应用，而较低层则保持标准位置编码。此设计基于 LLM 较低层主要捕获依赖局部信息的表面特征的发现。为了效率，REPO 仅使用分配的位置 $z_i$ 和 $z_j$ 来影响注意力计算中的位置编码，而保持 $q_i$ 或 $k_i$ 在上下文中的自回归顺序不变，从而避免了 KV 缓存的重新计算开销。

**实验与结果：**

研究人员在 OLMo-2 1B 模型的基础上，通过 50B token 的第二阶段数据持续预训练，将 REPO 与多种基线模型（ROPE, NOPE, R2N1, N2R1）进行了比较。评估涵盖了三个维度：

1.  **噪声上下文 (Noisy Context)**：在 RULER 基准测试上进行评估，该基准测试故意在上下文中注入无关信息。REPO 在训练上下文长度（4K tokens）内，性能平均超过 ROPE 11.04 分，表明其在处理噪声和干扰信息时的鲁棒性。
2.  **结构化数据 (Structured Data)**：在 NLGraph 和 HybridQA 数据集上进行评估，这些数据集涉及图和表格数据。REPO 在表格数据上表现出色，平均比标准 ROPE 提高了 1.94 EM 分。
3.  **长上下文 (Longer Context)**：通过 YaRN 方法将测试上下文长度扩展到 16K tokens（训练期间未见），并在 RULER 子集和 LongBench 上进行评估。REPO 在 4K tokens 长度时就已超越所有基线，并在 8K 和 16K tokens 时性能差距进一步扩大，平均比其他基线至少高 5.48 分。

此外，REPO 在广泛的通用基准测试（如 ARC-C, MMLU-Pro 等）上保持了与 ROPE 相当的性能，尽管其从线性位置分配转变为 REPO 导致预训练和持续训练之间存在不一致。这表明 REPO 在通用数据上具有良好的泛化能力。

**分析：**

1.  **注意力质量分配 (Attention Mass on Relevant Tokens)**：
    在 Needle-in-a-Haystack (NIAH) 任务中，REPO 相比线性 (RoPE) 和常量 (NoPE) 位置分配策略，能将显著更多的注意力分配给距离较远但对生成至关重要的“needle”tokens，而对最近的“query”tokens 的注意力分配较少。这表明 REPO 能够动态调整注意力模式，打破典型的局部性偏置，更好地捕获长距离依赖关系。

2.  **学习到的位置模式 (Position Patterns Learned by REPO)**：
    *   **位置范围**：REPO 分配的位置距离 $d_{k,h} = \text{max}(z_{k,h}) - \text{min}(z_{k,h})$ 在较长上下文中更大，但仍远小于原始上下文长度。这暗示模型可能不需要将位置范围扩展到完整的输入上下文长度。
    *   **局部模式**：将上下文分割成非重叠的 chunks，分析了三种模式：
        *   **Constant**：分配的位置接近一个常量。
        *   **Mono**：位置单调递增或递减。
        *   **Hybrid**：其他混合模式。
        研究发现，“Mono”模式非常罕见（占 4%），模型更偏爱“Constant”模式（占 22%），而“Hybrid”模式占据主导地位（约 70%）。这表明 REPO 学习到的位置模式与之前工作中预定义的不同，它能动态地在“Constant”和“Linear”模式之间选择或混合。

3.  **案例研究**：在包含 few-shot 示例的 MMLU-Pro 基准测试中，REPO 分配的位置模式与 few-shot 示例的语义分割大致对齐，表明 REPO 能够捕获输入上下文的内在结构。此外，REOP 分配的位置可能出现负值，这在 RoPE 框架下可被解释为反向旋转。

**效率：**
REPO 机制非常轻量级，仅增加了 0.9% 的参数量，且在训练上下文长度内的推理时间与原始模型相当，FLOPs 增加也微乎其微（0.9%）。

**总结：**
REPO 通过引入一个可学习的可微分模块 $f_\phi$，允许 LLM 根据 token 的相关性动态地重新定位上下文信息，从而减少了由刚性位置编码带来的额外认知负荷。实验证明，REPO 在噪声上下文、结构化数据和长上下文任务上显著提升了性能，同时在通用短上下文任务上保持了竞争力。分析表明，REPO 能够有效地将注意力引向远距离但相关的信息，并在稠密非线性空间中分配位置，捕获输入上下文的内在结构。

## SuperOffload
SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips

https://arxiv.org/abs/2509.21271 2025.9.25

针对超节点例如GH200的offload方案.集成到DeepSpeed.

中文解读：https://mp.weixin.qq.com/s/fj_XLlnNnNEAWI7fzXFEFw
<img width="674" height="387" alt="image" src="https://github.com/user-attachments/assets/b4ec6edc-9331-459c-99e5-ab94f0c82ac5" />

针对超节点例如GH200的offload方案.


## SD Speed-of-Light
Speculative Decoding Speed-of-Light: Optimal Lower Bounds via Branching Random Walks

paper: https://arxiv.org/pdf/2512.11718 Dan ISTA, 2025.12.12
1. 📚 该论文首次为大型语言模型（LLMs）的Speculative Decoding技术**建立了严格的运行时下限**，揭示了其加速的根本限制。
2. 💡 通过将Token生成过程与Branching Random Walks（BRW）理论联系起来，研究证明了每次推测迭代中成功预测的Token期望数量 E[X] **与验证器容量 P 的对数呈线性关系**，并**与目标模型输出分布的预期熵 μ 呈反比**。
3. ⚙️ 这一结果表明**并行度 P 的增加会带来收益递减**，并获得了Llama模型上经验评估的验证，为未来Speculative Decoding系统的设计提供了重要指导。
<img width="853" height="503" alt="image" src="https://github.com/user-attachments/assets/c7ab5357-acd0-49e7-9422-aa2b518f52a6" />

实验验证

论文在Llama 3.1 8B Instruct、Llama 3.3 70B Instruct、DeepSeek R1 Distill Llama 8B和Qwen3 8B等流行模型上，通过HumanEval、MT-bench、GSM8K、CNN/Daily Mail和Natural Questions等基准测试，评估了\muμ\muμ和\mu^{(2)}μ(2)\mu^{(2)}μ(2)参数。
结果显示，更大的模型（如Llama 3.3 70B）具有更低且更稳定的熵（\muμ\muμ）和二阶矩（\mu^{(2)}μ(2)\mu^{(2)}μ(2)），表明其并行化潜力更高。Qwen3相比Llama 3.1在大多数基准测试中表现出更低的期望熵。
通过将引理7的下限与EAGLE-3（一种先进的推测解码技术）在不同模型上的实际性能进行比较，实验结果显示出清晰的线性关系，验证了理论预测的紧密性。
图2展示了EAGLE-3的加速比随推测大小PPPP的变化。理论上的精确上限（定理1的蓝色虚线）与EAGLE-3的实际性能（实黑线和星形标记）之间存在差距。这种差距主要归因于：a) 理想分析中假设的最优推测器与EAGLE-3实现中的实际推测器之间的差距；b) 完美知识与不完美知识之间的差距。这表明在设计更优的实用推测算法方面仍有改进空间。

局限性

I.i.d. 假设：语言模型的上下文依赖性是固有的，输出熵会随上下文变化，i.i.d.假设是对实际情况的简化。
简化时序模型：忽略了KV缓存增长导致的延迟增加和草稿器的计算开销。
完美知识和确定性草稿：上限是为具有完美知识的最优确定性草稿策略推导的，未考虑随机推测策略。

总而言之，该论文通过将推测解码与BRW理论联系起来，为推测解码算法的性能提供了严格的理论上限。其核心发现是，加速比与并行度呈对数关系，并与目标模型的期望熵呈负相关，揭示了大规模并行本身无法克服概率瓶颈的根本限制。这些理论洞察得到了实验验证，并为未来的推测解码系统设计提供了指导，强调了优化模型特性（降低有效熵）比单纯增加并行预算更为重要。

## FlashFuser
FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
https://arxiv.org/pdf/2512.12949 上海交大 2025.12.15
1. 💡 FlashFuser是一个开创性的编译器框架，它利用现代GPU的**Distributed Shared Memory (DSM)** 机制，解决了现有**kernel fusion因SMEM容量限制**而无法处理大型中间结果的问题。
2. 🛠️ 该框架的核心贡献包括**DSM-based communication abstractio**n、一个**量化跨内存层级数据移动成本的数据流分析器**，以及一个能高效发现最佳执行计划的统一搜索引擎。
3. 🚀 在NVIDIA **H100 GPU**上的评估表明，FlashFuser将**内存访问减少了58%，实现了高达4.1倍的kernel speedup**，并带来了1.24倍的端到端性能提升。

FlashFuser是一款突破性的编译器框架，旨在解决现代深度学习工作负载中日益突出的“内存墙”问题。鉴于计算吞吐量的增长速度持续超越内存带宽的提升，许多深度学习模型（特别是Transformer中的Feed Forward Network (FFN)层和卷积块）受限于内存带宽，表现为Memory-bound。现有的Kernel fusion技术受限于单SM（Streaming Multiprocessor）内的局部Scratchpad memory（如寄存器和SMEM），当中间结果量超出其有限容量时（如大型FFN），融合就会失败。现代GPU（如NVIDIA H100）引入了互连核心机制，即Distributed Shared Memory (DSM)，提供了一个**更大、高带宽、低延迟的片上内存池**。然而，这一硬件潜力尚未被现有软件框架充分利用。FlashFuser是首个利用互核连接（DSM）进行Kernel fusion的编译器框架，旨在弥合这一硬件与软件之间的鸿沟。
<img width="826" height="299" alt="image" src="https://github.com/user-attachments/assets/940020af-694c-4847-9a42-dff87d135211" />
<img width="409" height="260" alt="image" src="https://github.com/user-attachments/assets/01835b4c-e6a7-4560-9b73-d7b23a77e7a0" />
<img width="453" height="224" alt="image" src="https://github.com/user-attachments/assets/83c02701-7fbd-4889-911a-1a492937a71c" />
<img width="1009" height="249" alt="image" src="https://github.com/user-attachments/assets/5ae282c5-c957-4fc7-b035-49e3a9382c5e" />
<img width="402" height="244" alt="image" src="https://github.com/user-attachments/assets/e05f9ea7-7ca5-44fd-8933-e3902ab783a6" />
<img width="407" height="276" alt="image" src="https://github.com/user-attachments/assets/cfb90d4c-defe-43cc-88b7-2444323a1959" />

FlashFuser通过三大核心贡献将现有融合技术扩展到DSM领域：

1.  **DSM-based communication abstraction (dsm_comm primitive)**:
    传统的GPU编程模型主要关注线程块（thread block）级别的单一分块层级。DSM的引入要求在线程块集群（thread block cluster）层面引入更高层级的Tiling，从而需要显式处理Cluster内部和Cluster之间的通信。
    FlashFuser定义了一系列`dsm_comm`原语来形式化复杂的基于Cluster的数据交换模式，包括：
    *   `dsm_all_exchange`：实现Cluster内的All-Reduce操作。例如，在两阶段GEMM链中，当K维度在Cluster内被空间分区到多个Blocks时，这些Blocks需要对中间结果执行Cluster内累加。`dsm_all_exchange`确保每个Block在继续下一步之前都持有完整的累加中间结果。它具有操作灵活性，可执行加法（如标准FFN）或乘法（如Gated FFN）。
    *   `dsm_shuffle`：用于数据在GEMM计算过程中在Shuffle Group内进行交换。例如，计算输出矩阵E的某个Block需要访问中间矩阵C的整行数据，同一Shuffle Group内的Blocks通过此原语交换各自的C矩阵分片。
    *   `dsm_reduce_scatter`：在Store阶段，用于将部分和进行两级层次化规约。首先进行Cluster内规约，多个贡献Shuffle Group通过此操作进行累加，并采用Scatter模式以避免数据冗余。
    *   `inter_cluster_reduce`：通过NVIDIA Hopper架构的Tensor Memory Accelerator (TMA)的`cp.reduce.async.bulk`指令实现，用于异步聚合来自所有参与Cluster的部分和。
    这些原语的定义依赖于两个关键参数：`clsi`（Cluster在维度$i$上的并行Blocks数量）和`blki`（Block在维度$i$上计算的数据粒度）。此外，还派生出`clsshuffle`（单个Shuffle Group中的Blocks数量，$clsshuffle = clsl / clsk$）和`clsreduce`（参与Reduce操作的Shuffle Groups数量，$clsreduce = clsn / clsshuffle$）。这些参数的灵活配置对于将不同大小的问题高效映射到硬件至关重要。
    通过这些原语，复杂的融合Kernel可以被抽象为直观的Tile graph来描述数据流，例如支持标准FFN和更复杂的Gated FFN。

2.  **Dataflow Analyzer**:
    这个组件负责在给定参数集下，评估任何融合方案的可行性和数据移动成本，并确定中间数据如何在内存层次结构中高效放置以实现复用。它能够量化跨内存层级的数据移动量。与传统方法只需考虑寄存器和SMEM不同，FlashFuser的分析器还协调新引入的DSM层。
    核心逻辑（如Algorithm 1所示）如下：
    *   **Loop Scheduling**: 定义了操作链的循环执行顺序。它将所有操作符的相互依赖循环维度统一为一组独立的维度$X = \{x_0, x_1, \dots, x_{J-1}\}$。然后定义一个排列$s$来设置嵌套顺序，并将维度分区为Spatial（并行处理）或Temporal（顺序处理）。不同的循环调度会影响需要缓存的张量大小，从而决定是否需要将数据从寄存器溢出到SMEM、DSM，甚至L2/Global。
    *   **Tile Selection**: 定义了三个层级的Tiling大小：`tile.cluster`（工作如何在Cluster间分布）和`tile.block`（每个Block计算的Tile大小）。这些Tiling直接影响内存使用和数据流模式。
    *   **Resource Mapping**: 采用启发式方法将张量绑定到不同的内存层级。对于可重用张量，它首先获取其Footprint (`DF`)。然后，采用贪婪算法将张量放置在尽可能高层级的内存中。如果某个层级容量不足，剩余部分会溢出到下一个层级。在整个过程中，计算每个缓存层级的数据移动量(`DV`)，尤其关注DSM流量，并根据`dsm_comm`原语的Cluster大小和数据Footprint来计算DSM流量。最终输出总数据移动量和最终的执行计划（包括调度、Tiling和资源映射）。

3.  **Fusion Search Engine**:
    DSM的引入极大地扩展了融合的可能性，从而带来了巨大的搜索空间。该引擎通过分析成本模型和剪枝策略，高效地探索由Loop schedules、Tiling sizes和Resource mapping构成的巨大搜索空间，以发现最优融合计划。
    *   **Cost Model**: 受到Chimera的启发，FlashFuser对L个内存层级的数据移动成本进行建模。将数据传输到层级$l$的成本$C_l$由所需数据量$V_l$和内存带宽$B_l$决定：
        $$ C_l(T_l) = \frac{V_l(T_l)}{B_l} $$
        为优化整体性能，目标是最小化所有内存层级中最慢的数据移动阶段，这被表述为一个minimax优化问题：
        $$ \min_{T_1, \dots, T_L} \left( \max_{l=1, \dots, L} (C_l(T_l)) \right) $$
        此优化受限于每个层级的内存容量约束：
        $$ U_l(T_l) \le \text{Cap}_l, \forall l \in \{1, \dots, L\} $$
    *   **Pruning Strategies**: 相比于先前工作（如MCFuser的$10^4$可能性），FlashFuser的初始搜索空间高达$2.75 \times 10^{13}$。因此，引入了更严格的剪枝策略：
        *   Rule 1, Divisible Tile Sizes: Tile大小必须是硬件感知的，且能整除问题尺寸。
        *   Rule 2, Cluster Size Constraint: 每个GEMM在M, N, K维度的Cluster维度乘积必须小于硬件限制（H100为16），且连续GEMM的Cluster维度必须一致。
        *   Rule 3, Activation constraint: 为确保连续GEMM间激活的正确性，前一个GEMM的累加维度必须放在最内层循环。
        *   Rule 4, Dependency constraint: 如果L维度被设为Spatial，且GEMM存在依赖，Spatial Tile需要中间张量但无法直接通信时，融合会失败。
        *   Rule 5, Memory Capacity Limit: 张量不能超过其可溢出到的最低层级缓存容量。
    *   **Search Algorithm**: (Algorithm 2) 首先利用剪枝策略过滤搜索空间。然后，合法候选通过Dataflow Analyzer进行详细分析，获取具体数据流细节和数据移动量。接着，使用成本模型评估每个配置，并维护一个top-K候选列表。最后，在硬件上对top-K候选进行Profiling以确定最终执行计划。为了提高效率，该搜索是离线进行的，运行时通过Binning和查表选择预编译的Kernel。

**实现细节**:
FlashFuser**基于NVIDIA CUTLASS构建**，前端是Python实现的搜索引擎，后端负责将优化计划翻译成高性能CUDA代码。Dataflow Analyzer的启发式计划通过计算理论寄存器使用量来决定寄存器和SMEM的使用，若SMEM不足，数据将溢出到DSM。`dsm_comm`原语的SHUFFLE, MUL, REDUCE操作通过TMA进行数据移动，并使用`mbarrier`内联函数进行多对多同步。通过**扩展CUTLASS Kernel的结构**（prologue, mainloop, epilogue），FlashFuser将**Cluster级数据流集成到Kernel中，例如在prologue中初始化DSM信号量**，mainloop中执行DSM Mul和Shuffle，epilogue中执行DSM Reduce。

**评估**:
在NVIDIA H100 GPU上进行评估，FlashFuser在多种Compute-intensive operator chains（GEMM链、卷积链、Gated FFN）上表现出色：
*   **GEMM链**：相较于BOLT、Chimera、Relay、TASO、TensorRT和PyTorch，平均速度提升分别为5.4x、4.6x、4.7x、3.4x、2.4x和3.1x。
*   **卷积链**：平均速度提升分别为6.3x、6.4x、5.6x、4.3x、3.3x和3.9x。
*   **内存访问**：FlashFuser显著减少了全局内存访问，相比未融合方法（如PyTorch），平均减少了2.4倍的全局内存流量，确认减少Off-chip内存访问是性能提升的主要来源。
*   **成本模型和搜索策略验证**：成本模型能够持续识别性能最优或接近最优的配置。实验表明选择Top-K=11能获得接近100%的预测准确率。搜索引擎相较于暴力搜索，编译时间加速12-864倍。
*   **`dsm_comm`性能**：测得`dsm_comm`原语（Shuffle, Reduce, Mul）在不同Cluster Size下的带宽和利用率，发现带宽随Cluster Size增加而下降，但带宽利用率保持稳定。Shuffle原语性能优于Reduce和Mul，因为后者有额外计算开销。
*   **Ablation Study**：对核心设计（dsm_comm (DC), Dataflow Analyzer (DA), Search Engine (SE)）进行消融实验，结果显示，完整系统(`All`)相较于无融合基线提升3.29x，仅`DC+DA`（随机配置）提升2.11x，仅`DA`（仅SMEM/全局内存融合）提升1.52x，证明了各组件的有效性。
*   **端到端性能**：在SGLang框架下对真实LLM和CNN模型进行端到端推理性能评估，FlashFuser平均提升1.24x。

FlashFuser提出的融合策略不局限于特定架构，其`dsm_comm`核心抽象是拓扑无关的集体通信概念。虽然目前实现主要针对H100，但对于具有Crossbar互连（如Graphcore IPU）或Mesh架构（如Cerebras WSE）的架构也具有潜在适用性。

总结来说，FlashFuser通过引入DSM、创新的`dsm_comm`原语、精细的数据流分析器以及高效的融合搜索引擎，成功克服了现有Kernel fusion在处理大中间数据时的容量限制，显著提升了Compute-intensive operator chains在现代GPU上的性能，为深度学习编译领域带来了重要进展。

## Efficient-DLM
Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed

paper：https://arxiv.org/pdf/2512.14067 佐治亚 MIT NV等，2025.12.16
1. 🎯 本文提出Efficient-DLM框架，旨在将**预训练的自回归（AR）模型高效转换为扩散语言模型**（dLM），以实现**更高的吞吐量并保持甚至略微提升任务准确性**。
2. 💡 关键创新包括采用**块状注意力模式**（block-wise attention）并使用**干净上下文**（clean context）以更好地保留AR模型能力，以及提出**位置依赖的token掩码策略**（position-dependent token masking）以弥合训练与测试间的差距。
3. 🚀 实验结果表明，Efficient-DLM系列模型4B～8B在准确性-吞吐量权衡方面超越了现有SOTA的AR和dLM，并能自适应调整以平衡准确性和效率，同时在文本嵌入任务中展现出强大优势。
<img width="406" height="322" alt="image" src="https://github.com/user-attachments/assets/764adbba-ec5c-4110-99df-21808c07527c" />
<img width="883" height="388" alt="image" src="https://github.com/user-attachments/assets/8a0f8678-e2d9-4565-a509-2a8c7e8307f0" />
<img width="887" height="273" alt="image" src="https://github.com/user-attachments/assets/28bc3d20-b4ef-410a-9440-a2bfcf81cf21" />
<img width="870" height="582" alt="image" src="https://github.com/user-attachments/assets/39a8a401-ec14-4e04-aec5-953a0ef831c5" />

该论文提出了一种名为 Efficient-DLM 的新型扩散语言模型 (dLM) 家族，旨在解决自回归 (AR) 语言模型在生成吞吐量方面的限制，同时克服现有 dLM 在学习效率和实际速度方面的不足。核心思想是将预训练的 AR 模型高效地转换为 dLM，从而在保持 AR 模型任务准确性的同时实现更高的生成速度。

该研究通过深入分析现有 AR-to-dLM 方法的注意力模式和训练目标，提出了关键的改进原则和方法。

**核心方法学与技术细节：**

1.  **注意力模式的优化：**
    *   **问题识别：** 现有的 AR-to-dLM 方法（如 Dream）通常采用完全双向注意力模式，即序列中的所有 token 都可以相互看到。这种模式存在缺陷：1) 难以应用 KV caching；2) 上下文过度损坏，特别是对于后续 token；3) 与 AR 模型的因果性注意力模式差异大，导致从 AR 模型初始化时权重漂移严重。
    *   **提出方案：** 引入一种“块级注意力模式 (block-wise attention pattern)”，并结合“基于干净上下文的条件化 (conditioning on clean context)”。
        *   **块级注意力 (block-wise attention)：** 在块间保持因果性 (causal across blocks)，而在每个块内部允许双向建模 (bidirectional modeling within each block)。这使得模型能够原生支持 KV caching，并且其块级因果性更接近预训练 AR 模型的 token 级因果性，从而更好地保留了 AR 模型的能力，减少了权重漂移（如图2(e)所示，与完全双向注意力相比，权重漂移更小）。
        *   **基于干净上下文的条件化：** 解决了训练与测试之间的 gap。在测试时，已解码的上下文是干净的（不包含 masked token），而传统块级注意力（如图2(c)）在训练时每个块的上下文仍可能包含 masked token。为了模拟测试时的行为，训练时将当前噪声块 $\tilde{\mathbf{x}}_b^t$ 与其前面已解码的干净上下文 $\mathbf{x}_{<b}$ 进行拼接作为输入。注意力机制设计为：噪声块内的 token 可以相互注意，并且可以注意干净上下文；干净上下文内的 token 也可以相互注意。
        *   **训练目标：** 形式化为：
            $$ \mathcal{L}(\theta) = E_{t \sim \mathcal{U}[0,1]} E_{\tilde{\mathbf{x}}_b^t \sim q(\cdot|\mathbf{x}_b)} - \frac{1}{t} \sum_{b=1}^B \log p_\theta(\mathbf{x}_b | \tilde{\mathbf{x}}_b^t, \mathbf{x}_{<b}) $$
            其中，$\mathbf{x}_b$ 是第 $b$ 个干净块，$\tilde{\mathbf{x}}_b^t$ 是其在噪声水平 $t$ 下的噪声版本，$q(\cdot|\mathbf{x}_b)$ 是腐蚀过程，$\mathbf{x}_{<b}$ 是第 $b$ 个块之前的干净上下文。模型 $\theta$ 在此目标下进行连续预训练，从预训练的 AR 模型（其训练损失为 $\mathcal{L}_{AR}(\theta) = - \sum_{l=1}^L \log p_\theta(x_l | x_{<l})$）进行初始化。
    *   **去除 Token Shift：** 传统 AR 模型通过预测下一个 token 进行训练。早期 dLM 转换工作认为保留 Token Shift 有益。该研究发现，去除 Token Shift（即直接预测 masked token 本身，而非其下一个 token）反而能提高准确性，表明 AR 模型的 Token Shift 可以被更好地适应，并且直接预测 masked token 更加直接和容易。

2.  **位置依赖的 Token Masking 策略：**
    *   **训练-测试 Masking Gap：** 现有的 dLM 通常采用均匀 Token Masking，即 Masked token 的选择只依赖于噪声水平 $t$，与位置无关。然而，在推理时，基于置信度的采样 (confidence-based sampling) 过程显示出明显的从左到右的解码倾向（语言的自回归特性），导致后续 token 往往需要更多的去噪步骤（如图6(a)）。这意味着在推理结束时，Masked token 更可能出现在块的末尾。这种训练与测试行为的不一致造成了 gap。
    *   **提出方案：** 引入位置依赖的 Token Masking 策略。对于一个序列 $\mathbf{x} = (\mathbf{x}_1, \dots, \mathbf{x}_L)$ 和给定的噪声水平 $t$，在块内相对位置 $i \in [L']$ 上的 Masking 概率 $w_i(t)$ 定义为：
        $$ w_i(t) = \exp \left( \beta (1-t) i \right) $$
        其中 $\beta \geq 0$ 是控制位置偏差强度的超参数。通过将其参数化为半衰期比率 $\lambda = \ln 2 / (\beta L')$，较低的 $\lambda$ 表示更强的位置先验。在去噪结束时（$t \to 0$），$w_i(t)$ 会给靠后的 token 赋予更大的权重，使得 Masked token 更可能出现在块的末尾，从而更好地模仿测试时的行为。这种策略能提高对块中较难（损失更大）的靠后 token 的 Masking 概率。

3.  **训练动态分析：**
    *   研究发现，dLM 的训练动态表现为：低训练成本（10B token 级别）即可从 AR 模型中恢复任务准确性；更长的训练（100B token 级别）能持续提高似然估计 (likelihood estimation)，并在下游任务上实现更高的准确性。更强的似然估计能生成更准确和可靠的置信度分数，从而在基于置信度的并行生成中提高生成质量，支持更激进的并行 token 生成。

**Efficient-DLM 模型家族：**

综合上述洞察，研究团队开发了 Efficient-DLM 模型家族（1.5B/4B/8B），分别从 Qwen2.5-1.5B、Qwen3 4B 和 Qwen3 8B 连续预训练而来。这些模型集成了：1) 块级注意力模式（带干净上下文，无 Token Shift）；2) 位置依赖的 Token Masking 策略（$\lambda = 0.1$）。模型经过更长时间的训练（1.5B/4B 训练 300B token，8B 训练 500B token）。

**实验结果与优势：**
<img width="883" height="218" alt="image" src="https://github.com/user-attachments/assets/c605fdf6-61c0-4a18-9d99-df28d839065d" />

*   **性能提升：** Efficient-DLM 在精度-吞吐量权衡方面超越了 SOTA 的 AR 模型和 dLM。例如，Efficient-DLM 8B 与 Qwen3 8B 相比准确率相当（略好），与 Dream 7B 相比准确率提高 5.35%，吞吐量提高 4.5 倍；与 Qwen3 4B 相比准确率提高 2.68%，吞吐量提高 2.7 倍。
*   **灵活性：** 单个 Efficient-DLM 模型能够通过调整置信度阈值在准确性和吞吐量之间进行平衡，实现“一劳永逸 (one-for-all)”的部署灵活性。
*   **文本嵌入优势：** 由于其双向建模能力，Efficient-DLM 在文本嵌入任务上表现出明显优势。在 1.5B 和 4B 规模上，Efficient-DLM 平均比同尺寸的 AR Qwen 模型高出 7.71% 和 9.91% 的 MTEB 分数。

**结论：**

该研究为将预训练 AR 模型转换为高效 dLM 提供了系统性的框架，通过优化注意力模式、Masking 策略和理解训练动态，成功构建了兼具高准确性和高速度的 Efficient-DLM 模型家族。这为 dLM 的未来发展提供了重要的实践指导和研究方向。


## TurboDiffusion
TurboDiffusion: Accelerating Video Diffusion Models by 100–200 Times

paper: https://arxiv.org/pdf/2512.16093 张金涛 清华 伯克利等 2025.12.18

code : https://github.com/thu-ml/TurboDiffusion
1. ✨ TurboDiffusion 是一种视频生成加速框架，能够在保持视频质量的同时，将端到端扩散生成速度提高 100-200 倍。
2. 🚀 该框架集成4种优化：低比特 **SageAttention++**、可训练的 **Sparse-Linear Attention** (SLA)、用于**高效步长蒸馏的 rCM**，以及 **权重和激活的W8A8 INT8量化**, **优化版的kernel： LayerNorm 和 RMSNorm 等**操作多种核心技术实现加速。
3. 💡 在多个 Wan 视频扩散模型上进行实验，结果显示 TurboDiffusion **在单张 RTX 5090 GPU** 上显著缩短了生成时间，从而使高质量视频生成更高效和实用。
<img width="672" height="291" alt="image" src="https://github.com/user-attachments/assets/ac20eb5d-ed11-4d38-befd-511ae1f9f476" />
<img width="676" height="227" alt="image" src="https://github.com/user-attachments/assets/4d3be692-cdae-42c8-a446-c818ffe074db" />

TurboDiffusion 是一项旨在加速视频扩散模型的框架，能在保持视频质量的同时，将端到端扩散生成速度提升100至200倍。该框架主要通过算法和系统协同优化实现其显著的加速效果。

**核心方法 (Main Techniques):**
TurboDiffusion 主要整合了四种技术来加速扩散模型：
1.  **Attention Acceleration (注意力加速):**
    *   **Low-bit SageAttention:** TurboDiffusion 采用了 SageAttention，特别是其变体 SageAttention2++，用于实现低比特量化的注意力计算。SageAttention 家族（SageAttention, SageAttention2, SageAttention2++, SageAttention3）专注于对注意力机制进行8-bit或更低比特的量化，以利用 Tensor Core 的加速能力，并解决量化带来的精度损失问题。
    *   **Sparse-Linear Attention (SLA):** 此外，TurboDiffusion 使用 Sparse-Linear Attention [5] 来加速稀疏注意力计算。SLA 允许对注意力机制进行稀疏化，减少计算量。由于稀疏计算与低比特 Tensor Core 加速是正交的，SLA 可以在 SageAttention 的基础上提供累积的加速效果。在推理阶段，SLA 会被专门的 CUDA 实现 SageSLA 替换，后者基于 SageAttention 构建。

2.  **Step Distillation (步长蒸馏):**
    *   **rCM (score-Regularized Continuous-time consistency):** TurboDiffusion 采用 rCM [6]（一种前沿的扩散模型蒸馏方法）来大幅减少采样步数。rCM 通过模型权重合并的方式，自然地继承了注意力层面的加速优化。通过 rCM 蒸馏，可以将原始模型通常需要100步的采样过程减少到极小的步数，例如3或4步，从而显著缩短生成时间。

3.  **W8A8 Quantization (W8A8 量化):**
    *   **Linear Layer Acceleration and Compression:** TurboDiffusion 对模型的参数（权重）和激活值都进行了8比特（INT8）量化。具体而言，线性层（Linear layers）的参数被量化为 INT8，量化粒度为 block-wise，块大小为 128 × 128。在推理过程中，线性层的激活值也实时量化为 INT8，并利用 INT8 Tensor Cores 进行计算。这种 W8A8 量化不仅加速了线性层的计算，还将模型大小压缩了约一半。

4.  **Other Optimizations (其他优化):**
    *   为了进一步提升效率，TurboDiffusion 还重新实现了如 LayerNorm 和 RMSNorm 等操作，使用 Triton 或 CUDA 进行优化。

**训练流程 (Training Process):**
TurboDiffusion 的训练流程基于一个预训练的视频扩散模型进行：
1.  首先，将全注意力（full attention）替换为 Sparse-Linear Attention (SLA) 并对预训练模型进行微调，使其适应稀疏性。
2.  同时，利用 rCM [6] 将预训练模型蒸馏成一个采样步数更少的学生模型。
3.  最后，将 SLA 微调和 rCM 训练过程中产生的参数更新合并到一个模型中。整个训练过程可以利用真实数据或合成数据。

**推理阶段 (Inference):**
在推理阶段，通过上述训练得到的模型进行部署，并应用以下加速策略：
*   **Attention acceleration:** 将训练后的 SLA 替换为优化过的 CUDA 实现 SageSLA。
*   **Step distillation:** 将采样步数从100步减少到4步或3步。
*   **Linear layer quantization:** 对线性层参数和运行时激活值执行 W8A8 量化，并利用 INT8 Tensor Cores 进行计算。

**实验评估 (Evaluations):**
TurboDiffusion 在 Wan2.2-I2V-A14B-720P、Wan2.1-T2V-1.3B-480P、Wan2.1-T2V-14B-720P 和 Wan2.1-T2V-14B-480P 等视频扩散模型上进行了实验。在单个 RTX 5090 GPU 上，TurboDiffusion 实现了以下显著加速：
*   Wan2.2-I2V-A14B-720P：从 4549s 降至 38s，加速 120x。
*   Wan2.1-T2V-1.3B-480P：从 184s 降至 1.9s，加速 97x。
*   Wan2.1-T2V-14B-480P：从 1676s 降至 9.9s，加速 170x。
*   Wan2.1-T2V-14B-720P：从 4767s 降至 24s，加速 199x。

相比基线模型（如原始 Wan 模型和 FastVideo），TurboDiffusion 在保持可比视频质量的同时，实现了更高的效率。例如，在 Wan2.1-T2V-1.3B-480P 模型上，原始延迟为 184s，FastVideo 延迟为 5.3s，而 TurboDiffusion 仅需 1.9s。在 Wan2.1-T2V-14B-720P 模型上，原始延迟高达 4767s，FastVideo 为 72.6s，而 TurboDiffusion 仅需 24s。图4的分解实验表明，W8A8 & FusedNorm、rCM 以及 SageSLA 的组合贡献了累计的加速效果，将 Wan2.1-T2V-14B-720P 的推理延迟从 4767s 降低到 24s，实现了约 200 倍的端到端加速。

**结论 (Conclusion):**
TurboDiffusion 成功地将视频生成时间缩短至小于1分钟（在单个 RTX 5090 GPU 上），显著提高了高质量视频生成的效率和实用性。未来工作计划将该框架扩展到更多视频生成范式，例如自回归视频扩散。

## B200 microbenchmark
Microbenchmarking NVIDIA’s Blackwell Architecture: An in-depth Architectural Analysis

https://arxiv.org/abs/2512.02189v1 Delaware Univ. 2025.12.1
1. 🚀 这项工作首次通过微基准测试深入分析了NVIDIA Blackwell B200 GPU，揭示了其5th-gen Tensor Cores、Tensor Memory (TMEM) 和 Decompression Engine (DE) 等架构创新带来的显著性能提升。
2. 💡 研究发现，Blackwell的TMEM使缓存未命中延迟降低58%，并提供16 TB/s的读带宽，而新引入的FP4/FP6精度Tensor Cores则实现了高达7702.5 TFLOPS的计算能力。
3. 📈 在实际工作负载中，B200在LLM推理、科学计算和混合精度训练方面均展现出比H200显著的性能优势，例如，其Tensor Core增强实现了1.56倍更高的混合精度吞吐量和42%更优的能效。

本文旨在通过深入的微基准测试（microbenchmarking）对 NVIDIA 最新的 Blackwell (B200) GPU 架构进行全面的性能分析，填补当前硬件发展与性能理解之间的空白。随着 exascale 计算和机器学习对 GPU 性能需求的快速增长，Blackwell 引入了多项关键架构创新，包括 5th-generation Tensor Cores、Tensor Memory (TMEM)、Decompression Engine (DE) 以及双芯片设计。然而，这些创新的实际性能影响和优化潜力在广泛工作负载中仍未被充分理解。本研究提供了一个开源微基准测试套件，旨在帮助应用程序开发者做出明智的架构决策，并指导未来的 GPU 设计方向。

本文的核心贡献包括：首次详细表征 NVIDIA Blackwell B200 的关键组件；量化 TMEM 对矩阵密集型工作负载的影响及其在减少张量计算内存瓶颈中的作用；评估 DE 在不同格式下的吞吐量并确定最佳使用方式；通过新的 `tcgen05` PTX 指令分析 5th-gen Tensor Core 执行的性能含义；评估 FP4/FP6 性能和精度权衡；以及通过 LLM 推理、训练和科学计算 kernels 演示其实际性能增益。
100MB数据压缩：
<img width="577" height="115" alt="image" src="https://github.com/user-attachments/assets/87b5698f-58db-41b1-b54d-13b1583c1b77" />
<img width="403" height="103" alt="image" src="https://github.com/user-attachments/assets/21081714-baf4-4efc-8b6c-ebefa7e89459" />
<img width="294" height="108" alt="image" src="https://github.com/user-attachments/assets/9005639d-d1fb-404f-9ce2-02693f25e5e8" />
<img width="432" height="131" alt="image" src="https://github.com/user-attachments/assets/58d6bca3-7479-4636-b31f-840ef1f96b9b" />
<img width="1017" height="258" alt="image" src="https://github.com/user-attachments/assets/af7144da-c693-4dd6-af2f-f30e2527cbce" />
<img width="978" height="199" alt="image" src="https://github.com/user-attachments/assets/b9b9122c-fe7a-4112-8a60-0d14a22ab6c1" />

**Blackwell 架构概述：**
B200 GPU 采用了双芯片配置，通过 NVIDIA High-Bandwidth Interface (NV-HBI) 连接，为软件提供统一的 192 GB HBM3e 内存空间。主要架构改进包括：
1.  **5th-generation Tensor Cores**: 相较于 Volta、Ampere 和 Hopper 等早期架构的 `warp-synchronous` (如 `mma.sync` 或 `wgmma`) 范式，Blackwell 引入了 `tcgen05.mma`，这是一种 `single-thread instruction`，允许每个线程独立地执行 `MMA` 操作，消除了 `warp-level` 同步需求，提高了调度灵活性。
2.  **Tensor Memory (TMEM)**: 新增的 `on-chip memory`，专用于张量数据移动，减少了对 `shared memory` (SMEM) 和 `register files` (RF) 的依赖。数据移动通过 `tcgen05.ld`, `tcgen05.st`, `tcgen05.cp` 等新指令显式管理。
3.  **Decompression Engine (DE)**: 硬件级别的解压缩引擎，旨在加速模型权重和大型数据库表的加载。支持多种算法（如 LZ4, Snappy, Zstandard, GZIP, Cascaded, Bitcomp, ANS）。
4.  **Extended Precision**: Tensor Cores 原生支持 FP4 和 FP6 浮点精度。
5.  **CTA Pair Execution**: 两个 `Cooperative Thread Arrays` (CTAs) 共享操作数，减少冗余数据移动。

**核心方法论：PTX 微基准测试**
本文采用基于 NVIDIA Parallel Thread Execution (PTX) 的微基准测试方法来表征 Blackwell 的微架构特征。PTX 允许对寄存器和内存操作进行显式控制，其代码编译为 Streaming Assembler (SASS) 指令。通过记录 PTX 到 SASS 的转换并验证性能，确保基准测试能准确隔离和测量特定的微架构行为。

1.  **TMEM 特征化**:
    *   **延迟基准测试**: 使用 `pointer-chase` 基准测试比较传统 `shared memory` 与 TMEM 之间的内存访问延迟，通过创建依赖内存访问来防止 `pipeline overlap`，以揭示每种内存层级的基本访问成本。
    *   **指令比较**: 系统性比较 `tcgen05.*` 系列新数据移动指令与传统指令（如 `wmma.load`, `ldmatrix`, `ld.shared`, `cp.async`）在不同访问模式下的性能。
    *   **带宽饱和点**: 改变操作数大小和访问步长，识别带宽饱和点和每访问延迟，揭示新指令集的性能能力和限制。

2.  **Decompression Engine (DE) 特征化**:
    *   **自定义套件**: 开发定制微基准测试套件，针对七种压缩格式（LZ4, Snappy, Zstandard, GZIP, Cascaded, Bitcomp, ANS）。
    *   **测量指标**: 测量 100MB 数据集的端到端解压缩吞吐量（`input throughput` 和 `output throughput`）和延迟。
    *   **数据多样性**: 使用不同熵的合成数据集（随机数据、混合字母数字、重复模式、全零缓冲）来隔离压缩比效应。
    *   **参数调整**: 系统性地改变 `chunk size` (32KB, 64KB, 128KB, 256KB) 和 `batch concurrency` (1–1024 `concurrent operations`)，以识别最佳并行级别，并定义 `pipeline depth` 和 `saturation point`。

3.  **Tensor Core 特征化**:
    *   **自定义 GPU kernels**: 执行 `D = A × B + D` 形式的 `MMA` 操作，使用 Blackwell 新引入的 `tcgen05` 指令集。
    *   **延迟和吞吐量测量**: 针对不同指令类型、矩阵 `tile shapes` 和操作数布局进行测量。
    *   **单指令延迟隔离**: 隔离 `tcgen05.mma` 的单指令延迟 (`SI-LAT`)，并与 Hopper 的 `wgmma` 进行比较。

4.  **Extended Precision (FP4 和 FP6) 特征化**:
    *   **系统性基准测试**: 首次为 Blackwell 的 FP4 和 FP6 `MMA` 指令 (`tcgen05` PTX `opcode` 和 `e2m1`/`e3m2`/`e2m3` 编码格式) 开发系统性基准测试。
    *   **依赖链方法**: 采用 `dependency-chain` 方法隔离这些超低精度操作的真实指令延迟。

**关键发现与性能分析：**

1.  **内存子系统**:
    *   **TMEM 性能**: TMEM 在 `cache-miss` 场景下实现 420 时钟周期端到端内存访问延迟，比 Hopper 的 `global memory latency` 减少 58%。这归因于 TMEM 专用的仲裁逻辑绕过了传统内存层次结构中 L2 cache 分区竞争。TMEM 提供每 SM 16 TB/s 的读带宽和 8 TB/s 的写带宽，且与 L1/SMEM 带宽累加。FP8 数据上的背靠背 `MMA` 操作可维持 8 TB/s 带宽，比传统 `ld.global` 路径的 3.8 TB/s 提高 2.1 倍。
    *   **TMEM 最佳实践**: 最佳效率在 64×64 元素 `tile` (FP8 精度为 4KB) 时实现，完全利用 1024-bit 内存接口宽度。这与 Hopper 32×32 的最佳 `tile size` 不同，需要算法调整。较小 `tile` 利用率低，较大 `tile` 引入 `pipeline stalls`。TMEM 适用于大型工作集的 `multi-stage tensor pipelines`，通过在 TMEM 中保持中间结果，可以显著减少数据移动（估计每秒 12 TB）。功率效率分析显示，对于大型矩阵，将 `Matrix-D accumulators` 放在 TMEM 中可降低 15% 的 `board-level power consumption`。
    *   **Decompression Engine (DE) 性能**:
        *   **格式特定优化**: DE 在不同压缩格式下吞吐量差异显著 (42 到 462 GB/s)。Bitcomp 针对数值数据实现 462.4 GB/s 的出色输出吞吐量和 0.227ms 的低延迟。Zstandard 在通用工作负载中表现平衡。
        *   **输入带宽瓶颈**: DE 的性能主要受 `compressed input bandwidth` 限制，而非解压缩计算能力。压缩比越高，输入吞吐量越低。然而，输出吞吐量保持在约 160-220 GB/s 之间，表明 DE 架构优先考虑持续输出带宽。
        *   **延迟一致性**: 无论数据模式如何，解压缩延迟始终较低 (100MB 数据集为 0.477-0.660ms)，表明 DE 实现了复杂的工作负载平衡机制。
        *   **管线深度与并发**: `pipeline depth` 随 `chunk size` 增加而减少。32KB `chunks` 支持 16 个并发操作，256KB `chunks` 支持 4 个。峰值吞吐量随 `chunk size` 增加而提升，从 53.8 GB/s 增加到 151.6 GB/s。

2.  **GPU Cores 微架构**:
    *   **5th-Generation Tensor Cores**: `tcgen05.mma` PTX 指令编译成不同的 SASS 指令（如 `OMMA` 用于 FP4）。Blackwell 的 `tcgen05.mma` 实现了 2.9–11.6 倍的单指令延迟降低（相比 Hopper 的 `wgmma`），并且延迟在不同 `tile sizes` 下几乎保持不变（11.0–11.4 周期），这表明 Blackwell 采用了 `spatial array` 设计而非 Hopper 的 `temporal pipelining`。`Warp-level` 粒度减少了调度器 `stalls` 18–23%。
    *   **精度影响**: 尽管 FP64 (44.8 TFLOPS) 与 FP4 (7702.5 TFLOPS) 之间吞吐量差异 177 倍，但延迟仅变化 1.27 倍（11.2–14.2 周期），这证实吞吐量提升是通过增加并行性实现，而非更深的管线。FP32 累加器将吞吐量减半，表明累加器数据路径是瓶颈。INT8 略优于 FP8，FP4 优于 FP8。

3.  **性能分析与案例研究**:
    *   **LLM 推理**:
        *   **精度模式影响**: 降低精度带来性能提升。FP8 和 FP4 对于 Mistral-7B 分别实现 1.73 倍和 2.5 倍吞吐量提升，接近理论带宽上限（2x 和 4x）。内存流量减少和 `cache locality` 提升是关键因素，L2 cache `hit rates` 从 68% 增加到 84%。B200 比 H200 始终保持 1.57–1.59 倍的吞吐量优势。FP8 引入的 `perplexity` 增加最小 (+1.9% to +2.4%)，FP4 较大但仍可接受 (+7.7% to +9.1%)。
        *   **Batch Size 敏感性**: 在小 `batch size` 下 (如 `batch size` 1)，B200 比 H200 有 1.48–1.52 倍的优异性能提升，因为 `pipeline` 自动重配置减少了处理阶段。在大 `batch size` 下，系统优化吞吐量，性能比稳定在 1.44 倍。
    *   **科学计算**:
        *   **FP64 性能**: B200 在大型矩阵尺寸下达到 36.3 TFLOPS 的 FP64 DGEMM 性能，利用率达 80.7% (H200 为 18.9 TFLOPS，55.6%)。这归因于 TMEM 启用的累加器和改进的内存访问合并模式。
        *   **持续内存带宽**: STREAM Triad 基准测试显示，当工作集超出 `cache capacity` 时，B200 和 H200 均实现超过 90% 的内存带宽利用率。B200 达到 7.48 TB/s，H200 达到 4.38 TB/s，B200 有 1.71 倍的速度提升，与原始带宽比（8.0/4.8）相符。
        *   **稀疏操作**: 利用 DE 进行稀疏矩阵-向量乘法 (SpMV) 可实现 3.16 倍的持续加速。DE 引入的延迟开销小于 5%，同时为 `pointer-intensive workloads` 减少 35% 的内存流量。
    *   **混合精度训练**:
        *   **端到端训练性能**: 在 ResNet-50 和 GPT-1.3B 训练中，实现 1.54–1.56 倍的速度提升。分解为 SM 数量 (1.09x)、CTA 配对 (1.27x) 和 TMEM (1.26x)。
        *   **能效**: 即使功耗增加 14%，GPT 训练能效仍提高 42%。

**讨论与结论：**
Blackwell 架构通过 TMEM、双模式 Tensor Cores 和解压缩引擎等创新，显著提升了性能，尽管晶体管数量有所增加。256KB 的 TMEM (`per SM`) 实现 61–82% 的 `hit rates`，验证了其尺寸设计。CUDA 13.0 提供了 TMEM/CTA 的初步支持，但软件工具链仍在发展。FP4/FP6 的 `per-layer precision selection` 对于保持精度和利用性能至关重要。对于 LLM 推理，B200 提供了 1.8–3.9 倍的优势；训练性能提升 (1.54–1.56x) 支持 33% 更大的 `batches`；HPC 性能 (1.92x FP64) 在科学计算中也极具竞争力。

总而言之，NVIDIA B200 GPU 标志着 GPU 架构的重大转变。本文提供了首次详细的微基准测试特性，量化了 TMEM 对矩阵密集型工作负载的影响，评估了硬件解压缩引擎的吞吐量和最佳使用方式，并分析了通过新 `tcgen05` PTX 指令实现的 5th-generation Tensor Core 执行。研究进一步评估了 FP4 和 FP6 精度权衡，基准测试了 Blackwell 在 LLM 推理、科学内核和混合精度训练等多种工作负载下的性能，并为开发人员提供了可操作的性能指南。
## SonicMoE
SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

https://arxiv.org/abs/2512.14080 Tri Dao 普林斯顿 伯克利等， 2025.12.16

https://github.com/Dao-AILab/sonic-moe

https://github.com/open-lm-engine/lm-engine

1. 🤔 SonicMoE提出了一种针对细粒度和稀疏MoE模型训练效率低下的共同设计方案，通过优化算法和GPU内核来解决**其激活内存占用大和硬件IO开销高**的问题。
2. 🚀 该方法通过内存高效的算法减少了MoE**反向传播的激活内存占用达45%**，并设计了新的GPU内核，在NVIDIA Hopper GPU上通过**IO和计算重叠实现了1.86倍**的吞吐量提升。
3. 💡 此外，SonicMoE引入了**创新的“token rounding”路由方法**，通过将token数量四舍五入到Grouped GEMM瓦片大小的倍数，有效减少了填充造成的FLOPs浪费，在高稀疏场景下核执行时间可额外**加速1.16倍，**同时保持模型质量。
<img width="944" height="281" alt="image" src="https://github.com/user-attachments/assets/98475f93-57a7-48f5-895b-faf43e3c17ec" />

<img width="923" height="376" alt="image" src="https://github.com/user-attachments/assets/fae6b7ea-0690-4483-a8e8-3369d0f18b9a" />

<img width="926" height="554" alt="image" src="https://github.com/user-attachments/assets/c336f327-d746-4df8-9072-dbc9706f86ef" />

<img width="779" height="277" alt="image" src="https://github.com/user-attachments/assets/3347e6e4-ce78-453e-8ecb-4e13a9cd0988" />

<img width="723" height="327" alt="image" src="https://github.com/user-attachments/assets/5a65cf35-c1f4-4135-98e6-5e7bdff1a7df" />

SonicMoE是一项针对Mixture of Experts (MoE) 模型训练效率的协同设计解决方案，旨在解决细粒度 (fine-grained) 和稀疏 (sparse) MoE所面临的激活内存占用过高、I/O成本增加以及Grouped GEMM中填充 (padding) 导致计算浪费等挑战。

**背景与挑战**
MoE模型通过仅激活部分专家来扩展语言模型，从而在不显著增加计算成本的情况下扩大模型规模。然而，现代MoE模型正趋向于更高的专家粒度（专家中间维度$n$更小，即$d/n$更大）和更高的稀疏度（专家总数$E$增加但激活专家数$K$不变）。这导致了硬件效率低下：
1.  **激活内存占用高**: 随着专家粒度增加，激活内存占用线性增长。
2.  **I/O成本和算术强度降低**: 细粒度专家导致每个专家处理的tokens更少，使得算术强度降低，I/O成为瓶颈。
3.  **Grouped GEMM填充浪费**: 在高度稀疏的MoE中，每个专家接收的tokens数量可能无法整除GEMM的tile size，导致额外的计算浪费。
<img width="825" height="285" alt="image" src="https://github.com/user-attachments/assets/558ac2bd-ff41-44c3-8ff5-e17798573643" />

**核心贡献**
SonicMoE通过三个主要贡献解决了上述问题：
1.  **内存高效的MoE前向和后向算法**: 最小化反向传播所需的激活缓存，消除激活内存占用随专家粒度增加而线性增长的问题。
2.  **I/O感知型GPU kernels**: 利用Hopper和Blackwell GPU的硬件特性，实现内存I/O与计算的重叠，提高吞吐量。
3.  **Tile感知型token rounding方法**: 最小化Grouped GEMM中因填充造成的计算浪费。
<img width="841" height="277" alt="image" src="https://github.com/user-attachments/assets/f66d2844-f6fe-429d-b7be-3ff4a45caf3b" />

**核心方法学**

1.  **内存高效的MoE算法**
    SonicMoE通过重新设计计算图来优化激活内存使用，特别是在反向传播过程中。
    -   **避免缓存$Y$和$Xe$**: 传统的MoE实现会缓存$Y$ (down-projection输出) 和$Xe$ (gathered $X$) 用于反向传播。SonicMoE通过融合gather操作与HBM加载来避免对$Xe$的显式物化和缓存，并为$dY$（Y的梯度）找到替代计算路径，避免将其写入HBM。
    -   **优化$dS$和$dH$计算**:
        -   **$dS$（router scores的梯度）计算**: 对于专家$e$，输入$X_e \in \mathbb{R}^{T_e \times d}$，专家权重$W_{1,e} \in \mathbb{R}^{d \times 2n}$, $W_{2,e} \in \mathbb{R}^{n \times d}$。前向计算为$H_e = X_e W_{1,e}$, $A_e = \text{SwiGLU}(H_e)$, $Y_e = A_e W_{2,e}$。聚合输出为$O_t = \sum_{e \in [E]} \pi_{t,e} S_{t,e} Y_{e,t}$。
        -   SonicMoE计算$dS_{t,e}$为$\langle dA'_{e,t}, A_{e,t} \rangle$，其中$dA'_{e,t} = dO_{e,t} W_{2,e}^{\top}$。这种方法比传统的$\langle dO_{t}, Y_{e,t} \rangle$更优，因为它：
            -   **节省HBM流量**: $dA'$和$A$在$dH$ kernel中已经计算得出，避免了额外的HBM加载（2$T K d$ bytes）。
            -   **节省缓存激活内存**: 避免缓存$Y$ (2$T K d$ bytes)。
            -   **减少并行归约轮次**: $\langle dA', A \rangle$在中间维度$n$上归约，而$\langle dO, Y \rangle$在维度$d$上归约，前者通常更快，因为$n$远小于$d$。
        -   **$dH$（$H$的梯度）计算**: $dH_e = d\text{SwiGLU}(dA_e, H_e)$。SonicMoE将$dH$和$dS$的计算融合到反向传播的down-proj激活梯度(dH) kernel的epilogue中，减少了额外的kernel启动和内存访问。

2.  **I/O感知型kernel设计**
    SonicMoE的Grouped GEMM kernels通过以下策略最大化吞吐量：
    -   **Gather fusion with HBM load**: 对于变量M维度Grouped GEMM（如前向的up-proj），SonicMoE将输入tokens的gather操作（从非连续内存位置收集数据）与从HBM到SMEM的数据加载融合。这通过Hopper/Blackwell GPU的`cp.async`指令实现，并针对Blackwell的2-CTA集群做了优化（使用relay warp传递完成信号），避免了额外的gather kernel启动和大量I/O。
    -   **Epilogue fusion**:
        -   将SwiGLU和dSwiGLU激活函数融合到GEMM的epilogue阶段。
        -   如上所述，将$dH$和$dS$的计算融合到down-proj激活梯度(dH) kernel的epilogue中。
    -   **MMA与异步I/O重叠 (Ping-Pong Scheduling)**:
        -   在Hopper GPU上，GEMM使用producer-consumer模式。SonicMoE利用"Ping-Pong"调度策略，让两个consumer warpgroups交替进行MMA计算和I/O操作，使得I/O延迟与计算重叠。这对于具有重度epilogue操作的kernels（如前向down-proj $Y$和反向down-proj激活梯度$dH$）特别有效。
        -   SonicMoE选择异步TMA (Tensor Memory Access) store进行HBM写入，而非同步`st.global`指令。同步的`st.global`会阻塞下一个tile的MMA执行，导致吞吐量显著下降（高达20%），尤其是在需要scatter操作的情况下。SonicMoE不融合scatter与HBM store，而是让每个token在expert aggregation kernel中gather专家输出，从而避免了同步scatter的性能开销。

3.  **Token Rounding Routing**
    针对稀疏MoE的"tile quantization effect"（由于token数量不能整除GEMM tile size而导致的填充浪费），SonicMoE引入了Token Rounding (TR) 路由方法。
    -   **算法概述**:
        1.  执行标准的Top-K token choice (TC) 路由，得到每个token选择的专家以及对应的分数。
        2.  对于每个专家，计算其接收到的token频率($f_e$)。
        3.  根据预定义的tile size ($M_{tile}$)，将$f_e$向上取整到最近的$M_{tile}$倍数（$\lceil f_e \rceil_{M_{tile}}$）或向下取整（$\lfloor f_e \rfloor_{M_{tile}}$）。
        4.  TR算法根据某种舍入策略（例如“nearest rounding to Mtile-multiples via expert frequency”，NR-f）选择是填充额外的tokens以达到下一个tile大小，还是丢弃部分tokens以匹配前一个tile大小。
        5.  构建一个分数矩阵$S'$，使得TC选择的tokens总是优先于其他tokens。
        6.  对每个专家，根据$S'$和舍入后的token数量重新分配tokens。
    -   **核心保证**: 对于每个专家，token数量与原始TC路由结果的最大偏差被限制在一个tile内。
    -   **优点**: 有效消除了Grouped GEMM中的填充浪费，同时在训练期间保持了相似的总token数量，并在下游任务中维持了模型质量。在高度稀疏MoE训练中，TR可以显著提高计算吞吐量。

**实验结果**

-   **激活内存**: SonicMoE在单MoE层上的峰值激活内存占用最低。对于7B MoE模型，激活内存使用量比ScatterMoE减少45%，对于30B和120B模型，节省更为显著（120B模型相比MoMoE节省超过3GiB）。SonicMoE的激活内存占用不随专家粒度增加而改变。
-   **训练吞吐量**:
    -   **Kernel级别**: SonicMoE在H100 GPU上实现了1.86x的BF16 MoE kernel计算吞吐量提升。它在不同模型规模下持续实现最高的TFLOPS，对于细粒度MoE，其相对DeepGEMM++的加速比更高。
    -   **端到端训练**: SonicMoE在64个H100 GPU上实现了每天2130亿tokens的训练吞吐量，与ScatterMoE在96个H100 GPU上实现的2250亿tokens/天相当，展现了显著的效率提升。
    -   **Token Rounding的额外加速**: 在高MoE稀疏设置下，tile-aware token rounding算法在kernel执行时间上额外带来1.16x的加速，同时保持了相似的下游任务性能。对于256个专家（K/E=1/128），Token Rounding路由在前向和反向传播中分别带来了25.7%和11.8%的TFLOPS提升，端到端提升15.9%。
-   **Token Rounding质量**: 实验证明，即使在训练时使用TR，在推理时切换回标准Top-K TC路由，模型质量也与TC训练的模型相近甚至略优，尤其是在极端稀疏的MoE配置下。TR算法对不同的舍入子程序以及微批次大小和tile size的变化表现出较强的鲁棒性，只要$T_e / M_{tile} \ge 2$。

**总结**
SonicMoE通过内存高效的算法、I/O感知型GPU kernels和创新的token rounding路由方法，全面提升了MoE模型的训练效率，特别是在处理当前和未来趋势下的细粒度、稀疏MoE模型时，能有效降低内存占用并提高计算吞吐量。

## ParallelKittens 
ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels 

paper: https://arxiv.org/abs/2511.13940 斯坦福 Christopher团队，2025.11.17

code: https://github.com/HazyResearch/ThunderKittens 

blog: https://hazyresearch.stanford.edu/blog/2025-11-17-pk

1. 🚀 ParallelKittens (PK) 提出并实现了一套基于**传输机制、调度策略和设计开销**三大原则的 CUDA 框架，旨在系统性地简化和优化 multi-GPU AI kernels 的开发。
2. ✨ 该框架通过利用 Tensor Memory Accelerator (TMA) 和 register-level instructions 等高效数据传输方式，支持 **intra-SM 和 inter-SM 两种 overlapping 调度策略**，并大幅减少了现有通信库的同步和缓冲开销。
3. ⚡ PK 在 Hopper 和 Blackwell 架构上展示了卓越性能，相较于 baseline 和现有 hand-tuned 以及 compiler-based 方法，为 data-、tensor-、sequence- 和 expert-parallel workloads 带来了**显著的性能提升，且代码量极少~50LOC。**

ParallelKittens (PK) 是一项旨在系统化并简化多GPU AI内核开发的框架，其核心目标是解决多GPU工作负载中GPU间通信日益成为瓶颈的问题。现有系统主要通过计算-通信重叠来缓解这一问题，但往往未能达到理论峰值性能，且在异构工作负载和新型加速器上表现不佳。PK通过提炼和封装一套简单的、可复用的原则，构建了一个最小的CUDA框架，大大简化了重叠式多GPU内核的开发。

**核心方法论 (Core Methodology)**

PK的核心方法论基于对影响多GPU性能的三个关键因素的全面分析：数据传输机制、资源调度和设计开销。

1.  **传输机制 (Transfer Mechanism)**
    *   **主机发起 (Host-initiated) vs. 设备发起 (Device-initiated) 通信：**
        *   **拷贝引擎 (Copy Engine)**：由主机发起，支持连续内存传输。在H100上可达368.82 GB/s (理论峰值82%)，在B200上为726.13 GB/s (理论峰值81%)。对于饱和传输，需要至少256MB的大消息粒度。PK认为这种方式主要适用于大规模连续数据传输（如全分片数据并行中的权重移动），且重叠通常是直观的（在不同CUDA Stream中启动）。
        *   **设备发起 (Device-initiated)**：由设备上的Streaming Multiprocessors (SMs) 发起。PK主要依赖这种方式进行细粒度通信。
            *   **张量内存加速器 (Tensor Memory Accelerator, TMA)**：支持NVLink传输和NVSwitch加速的广播。TMA可以在H100上达到350.01 GB/s (78%)，B200上达到669.12 GB/s (74%)。其关键优势在于可以由单个线程异步启动，不增加寄存器压力，允许同一SM中的其他线程重叠计算或内存工作（intra-SM overlap）。TMA在2KB消息粒度下即可达到接近峰值带宽利用率。
            *   **寄存器级操作 (Register-level instructions, `ld`, `st`)**：效率相对较低，在B200上约70%的峰值带宽。这些指令是同步的，需要高SM占用率（数千线程并发）来饱和NVLink带宽，同时带来更高的寄存器压力和手动内存合并。仅当拷贝引擎和TMA无法提供所需功能时才使用，例如NVSwitch的in-network reduction (如`multimem.ld.reduce`和`multimem.red`)。

    *   **传输机制选择总结：**
        *   拷贝引擎：点对点传输，fabric内广播。
        *   TMA：点对点传输，fabric内广播，点对点规约（P2P Reduction）。
        *   寄存器操作：点对点传输，fabric内广播，点对点规约，fabric内规约（in-fabric reduction），元素级传输。

2.  **调度 (Scheduling)**
    PK确定了两种主要的调度策略来重叠计算和通信：
    *   **SM内部重叠 (Intra-SM Overlapping)**：将一个SM内的线程划分为两部分，一部分执行计算/内存指令，另一部分执行通信指令。
        *   **优势**：当计算和通信粒度对齐时非常有效。所有计算单元（Tensor Core）可以充分利用，因为计算吞吐量与执行计算的SM数量线性相关。相比SM间重叠，SM内部重叠引入的同步开销更低（例如，使用`mbarrier`对象进行intra-SM同步约为64 ns，而通过HBM进行inter-SM同步约为832 ns）。
        *   **通信隐藏条件**：对于BF16 GEMM，当输入维度$K \gtrsim \frac{sR}{2B}$时（其中$s$为每元素大小，$R$为Tensor Core吞吐量，$B$为NVLink带宽），通信可以被计算完全隐藏。例如，在H100上，当$K \gtrsim 2197$时，通信开销几乎完全隐藏。
    *   **SM间重叠 (Inter-SM Overlapping)**：将SMs划分为两部分，一部分专门用于计算，另一部分专门用于通信。
        *   **优势**：能够利用in-network acceleration（如NVSwitch的规约功能），显著减少通信量。例如，GEMM all-reduce通过SM间重叠可实现3.62倍的性能提升。此外，可以有效处理远程L2缓存重用问题，因为对等GPU的HBM访问只在源设备缓存，每此远程访问都受NVLink带宽限制。通过通信专用SM批量传输数据到本地HBM，可以提高L2重用。
        *   **SM划分**：需要平衡计算和通信的SM数量，最优划分取决于输入规模，大型工作负载倾向于更多计算SMs，小型工作负载倾向于更多通信SMs。PK允许用户在运行时自动搜索最优SM分配。

3.  **设计开销 (Design Overheads)**
    PK通过避免传统通信库（如NCCL、NVSHMEM）中的不必要开销来提升性能：
    *   **双向同步和中间缓冲**：NCCL等库通常强制执行每操作双向同步和使用预分配的中间缓冲区。这些在细粒度通信中导致显著开销。PK通过使用预分配的目的缓冲区实现直接、单向传输，避免了中间暂存，从而在全规约等纯通信内核中实现了高达1.79倍的性能提升。
    *   **对等内存访问和同步**：NVSHMEM的API函数在远程对等访问时会引入全局内存加载（`ldg`）来检索对等地址，并强制进行组同步（`syncthreads`）。PK通过将对等地址保存在寄存器中并移除不必要的同步，将元素级NVLink访问延迟降低了4.5倍，并将带宽利用率提高了约20 GB/s。

**ParallelKittens 抽象**

PK建立在ThunderKittens (TK) 框架之上，提供一套最小且互补的通信原语：
*   **数据结构**：PK为GPU内存层次结构的每个级别定义了数据结构。
    *   **寄存器级**：最小执行单元为16x16 Tile。
    *   **共享内存级 (SMEM)**：`shared tiles`支持单线程异步地从对等HBM加载和存储，可选支持对等内存上的原子规约和通过in-network广播进行多播。
    *   **HBM级**：引入**并行全局布局 (Parallel Global Layout, PGL)**，表示在所有设备上分配的形状和大小相同的内存区域，作为异步P2P传输、广播和同步fabric内多播/规约的中心数据结构。
*   **多GPU操作**：PK引入了八个新的核心原语，足以实现所有实验中展示的内核。
    *   **P2P通信原语**：`store_async` (将共享Tile存储到多播内存), `store_add_async` (原子性地将共享Tile添加到多播内存)。这些是异步单线程操作，支持与其他操作融合。
    *   **网络加速通信原语**：`reduce` (将数据从多播内存规约到本地HBM), `all_reduce` (对多播内存上的数据执行全规约)。这些需要至少Warp级的参与以获得最优吞吐量。
    *   **设备间和SM间同步原语**：`signal` (向特定设备的barrier发送信号), `signal_all` (同时向所有设备的barrier发送信号), `wait` (等待特定设备的barrier达到期望值), `barrier` (等待所有设备到达此点)。
*   **编程模板 (Program Template)**：PK提供一个统一的“加载-计算-存储-通信 (LCSC)”模板，用于实现各种多GPU内核。该模板定义了四个工作组件：
    *   `loader`：执行本地或对等HBM读取。
    *   `storer`：处理本地或对等HBM写入。
    *   `consumer`：执行Tensor Core或CUDA Core的本地计算。
    *   `communicator`：专门执行设备间通信，在独立的通信SMs上运行，实现SM间重叠。
    该模板自动化了内核配置、共享内存和TMA设置、barrier和同步管理以及SM/Warp划分的优化，允许用户专注于每Tile的计算和通信逻辑。

**实验结果**

PK在H100和Blackwell架构上对数据并行、张量并行、序列并行和专家并行等多种AI工作负载进行了验证。
*   **数据和张量并行**：对于AG+GEMM, GEMM+RS, GEMM+AR工作负载，PK相对于非重叠基线（cuBLAS + NCCL）实现了1.06-1.68倍的加速，相较于编译器方法（Triton Distributed）性能提升1.07-5.63倍。PK与Flux和CUTLASS等手工优化内核性能持平或超越，例如相对于Flux加速0.97-2.33倍。非重叠通信时间占比可降至1%以下。
*   **序列并行**：
    *   **Ring Attention**：PK通过精确的SM间重叠融合KV交换和计算，相对于xDiT基线实现了1.07-4.08倍的加速，将非重叠通信时间占比降至9%。
    *   **DeepSpeed-Ulysses**：PK实现了细粒度的all-to-all内核，移除了传统方法中因NCCL不支持内部维度all-to-all而引入的张量重塑开销，实现了1.01-1.39倍的加速。
*   **专家并行**：PK在令牌分派和第一层专家MLP重叠中，与Comet等手工优化基线性能持平或超越，实现了0.92-1.22倍的性能提升。
所有PK内核的通信部分，通常只需要少于50行设备代码。

**总结**

PK提供了一个最小化、系统化的框架，通过对传输机制、调度策略和设计开销的深入分析，实现高性能多GPU内核。它证明了小范围的核心原语即可匹配或超越手工优化内核的性能，同时大大简化了实现复杂性。该工作专注于节点内执行，为未来节点间通信扩展提供了基础。

## RoEP++ real
Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs

https://arxiv.org/abs/2512.07525 复旦 邱锡朋 2025.12.8
https://github.com/OpenMOSS/rope_pp 

1. 💡 针对标准RoPE在计算注意力分数时丢弃**复数点积虚部**的问题，本文提出了RoPE++，通过重新引入这一**虚部来增强旋转位置编码**。
2. 🎉 RoPE++利用完整的复数表示来创建**双组分注意力分数**，并提供了两种配置：**RoPE++EH**在注意力头数相同的情**况下将KV缓存减半**，而**RoPE++EC则在缓存大小相同的情况下将注意力头数翻倍。**
3. 🚀 实验结果表明，RoPE++在短上下文和长上下文基准测试中均持续优于传统RoPE，尤其是在**长上下文场景下表现出显著提升**，同时RoPE++EH还能有效降低内存成本并加速解码。
<img width="599" height="375" alt="image" src="https://github.com/user-attachments/assets/0c7c6baf-96a5-4690-b2f7-22a88756da8f" />

该论文提出 SII-OpenMOSS (RoPE++), 旨在通过重新引入旋转位置编码 (RoPE) 中被舍弃的虚部信息，以增强大型语言模型 (LLMs) 的长上下文建模能力。
<img width="702" height="623" alt="image" src="https://github.com/user-attachments/assets/3fb52a3c-31a6-421a-b4df-869079a0c9dd" />

**问题背景：**
RoPE 已成为 LLMs 中常用的位置编码方式，它通过对查询（query）和键（key）向量在复平面进行旋转来编码序列顺序。然而，标准实现只使用复值点积的实部来计算注意力分数，这导致虚部（包含宝贵的相位信息）被丢弃，可能损失了对建模长上下文依赖至关重要的关系细节。

**核心方法论：**
RoPE++ 的核心思想是重新利用 RoPE 计算中被忽略的虚部信息。

1.  **复数形式的重新审视与虚部恢复：**
    标准的注意力分数计算 $A_{t,s}$ 在复数形式下表示为：
    $$A_{t,s} = \text{Re}\left[\sum_{n=0}^{d/2-1} \tilde{q}^{(n)}_t (\tilde{k}^{(n)}_s)^* e^{-i\theta_n(t-s)}\right]$$
    其中 $\tilde{q}^{(n)}_t = q^{(2n)}_t + i \cdot q^{(2n+1)}_t$ 和 $\tilde{k}^{(n)}_s = k^{(2n)}_s + i \cdot k^{(2n+1)}_s$。
    论文恢复了被丢弃的虚部，严格来说是负虚部，并将其定义为虚部注意力 ($A_{Im_{t,s}}$)：
    $$A_{Im_{t,s}} = -\text{Im}\left[\sum_{n=0}^{d/2-1} \tilde{q}^{(n)}_t (\tilde{k}^{(n)}_s)^* e^{-i\theta_n(t-s)}\right]$$
    通过对 $q_t$ 向量进行 $-\pi/2$ 的旋转，虚部注意力也可以表示为绝对位置编码的形式。假设 $R_{\Theta,t}$ 是应用于 $q_t$ 的 RoPE 旋转矩阵，则实部注意力可以表示为 $A_{Re_{t,s}} = (R_{\Theta,t}q_t)^\top (R_{\Theta,s}k_s) = q_t^\top R_{\Theta,s-t}k_s$。相应地，虚部注意力可表示为：
    $$A_{Im_{t,s}} = (R_{-\frac{\pi}{2}}q_t)^\top R_{\Theta,s-t}k_s$$
    这意味着计算虚部注意力只需将 $q_t$ 向量旋转 $-\pi/2$ 后再进行标准的位置编码，而 $k_s$ 的位置嵌入保持不变。

2.  **长依赖捕获能力分析：**
    实部注意力（RoPE）的特征曲线近似为余弦积分函数 $\tilde{c}_{Re}(\Delta t) = \text{Ci}(\Delta t) - \text{Ci}(\frac{\Delta t}{10^4})$，它在相对距离 $\Delta t$ 增加时表现出衰减特性，更倾向于局部语义关联。
    虚部注意力则近似为正弦积分函数 $\tilde{c}_{Im}(\Delta t) = \text{Si}(\Delta t) - \text{Si}(\frac{\Delta t}{10^4})$。尽管 $\sin(\theta \Delta t)$ 在 $\Delta t=0$ 时为零且有波动，但论文发现虚部注意力特征曲线在超过一定距离后衰减非常缓慢，表明它能够更好地捕获更远距离的信息。图1的对比也印证了虚部注意力在长上下文区域分配了更多权重，从而有助于 LLMs 检索长上下文信息。此外，虚部注意力也具备语义聚合特性。

3.  **缓存与参数效率考量 (RoPE++EH 和 RoPE++EC)：**
    由于虚部注意力的计算方式与实部注意力高度相似，它们可以并行计算。
    *   **RoPE++EC (Equal Cache size)：** 保持与原始 RoPE 相同的 KV 缓存大小。通过将经过 $-\pi/2$ 旋转的 $q_t$ 和原始 $q_t$ 交错处理，可以在 FlashAttention 中一次性完成实部和虚部注意力的计算，不引入额外的 KV 缓存成本。这会使注意力头数量翻倍。
    *   **RoPE++EH (Equal Head number)：** 保持与原始 RoPE 相同的注意力头数量。这会使 QKV 参数和 KV 缓存大小减半。这种配置在长上下文场景中能显著降低内存消耗并提高吞吐量。
    两种配置都要求实部和虚部注意力共享相同的参数 $W_q$，因为虚部注意力是相对实部注意力定义的，不能独立存在。

4.  **长度外推能力提升：**
    标准 RoPE 中，偶数索引的 $q^{(2n)}$ 和奇数索引的 $k^{(2n+1)}$ 维度仅与 $\cos \theta_n(t-s)$ 和 $\sin \theta_n(t-s)$ 相乘，在超出训练上下文长度时可能遇到分布外（OOD）的负嵌入值。RoPE++ 通过引入虚部注意力，使得这些维度在训练阶段就接触到负的 $\cos \theta_n(t-s)$ 和正的 $\sin \theta_n(t-s)$，从而观察到更完整的正弦波值范围（包括负值），缓解了长度外推问题。虽然 RoPE++ 本身不是即插即用的长度外推方案，但其困惑度曲线在超出训练长度后上升更缓慢。

**实验结果：**
论文在 376M、776M 和 1.5B 不同规模的模型上进行了预训练和评估，训练数据为 DCLM-Baseline-1.0，最大上下文长度 4k，并进行了长上下文继续预训练（32k 长度， rotary base 从 10000 扩展到 500000）。

*   **短上下文评估：** RoPE++EH 和 RoPE++EC 在 WikiText、LAMBADA 和 Open LLM Leaderboard 任务上的平均分数优于标准 RoPE 和其他基线（FoPE, Pythia, ALiBi）。RoPE++EH 在 KV 缓存和 QKV 参数减半的情况下仍超越了标准 RoPE。
*   **长上下文评估：** 在 RULER 和 BABILong 等合成基准测试中，RoPE++ 在 64k 上下文长度下取得了最高分数。RoPE++EC 在相同缓存成本下显著优于 RoPE，而 RoPE++EH 在 KV 缓存减半的情况下表现与 RoPE 相当甚至更优。
*   **效率：** RoPE++EH 显著降低了内存成本并加速了解码，尤其是在上下文长度增加时，优势更为明显。
*   **注意力模式分析：** 对 RoPE++ 模型的注意力模式检查表明，虚部注意力头更倾向于关注全局信息和初始位置，而实部注意力头则更关注局部上下文。通过对虚部注意力施加噪声，模型在长上下文任务上的性能下降更为显著，证实了虚部注意力在长上下文建模中的主导作用。
*   **与其他长上下文技术结合：** RoPE++ 可以与 NTK-RoPE、Linear PI 和 YaRN 等现有长上下文技术结合使用，并在此类组合中持续展现性能优势。

**结论：**
RoPE++ 通过重新引入 RoPE 计算中被忽略的虚部，并将其作为一个新的注意力头组，成功增强了 LLMs 的长上下文建模能力。其两种配置 (RoPE++EH 和 RoPE++EC) 在短上下文任务上实现了更优性能，并在长上下文场景中带来了显著提升，同时在某些配置下（RoPE++EH）还实现了参数和缓存效率的提高。对注意力模式的分析证实了虚部注意力在长上下文建模中的关键作用。

## MicroEP 
MicroMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

https://arxiv.org/pdf/2511.16947 北大 2025.11.21

1.  ✨ 本文提出了一种新颖的MoE负载均衡策略MicroEP，通过在每个Micro-batch内进行Token Scheduling，实现了GPU之间的细粒度负载均衡。
2.  🌍 该方法将负载均衡问题建模为线性规划问题并高效求解，并通过 Expert Placement 和 Adaptive Replacement 机制来优化 Token Scheduling 的能力。
3.  🚀 实验结果表明，MicroMoE 系统相比现有最先进系统，端到端训练吞吐量最高提升47.6%，并且能够几乎一致地在GPU之间实现最佳负载均衡。
<img width="506" height="300" alt="image" src="https://github.com/user-attachments/assets/620b8c47-07f6-4e4a-807f-7eb31dab63fe" />

<img width="453" height="279" alt="image" src="https://github.com/user-attachments/assets/d0a5c8a7-4008-432f-b9f5-ffc59a89c191" />

<img width="457" height="705" alt="image" src="https://github.com/user-attachments/assets/4b85956f-6438-4e0a-bf79-05077cae4606" />

MoE (Mixture-of-Experts)模型因其稀疏激活特性而能有效扩展深度学习模型，但**其动态路由会导致专家之间负载不均衡，严重影响分布式训练效率**。现有解决方案要么牺牲模型精度（如通过修改路由算法引入负载均衡损失或丢弃tokens），要么引入额外的系统开销（如通过调整专家-GPU映射），均未能实现细粒度的负载均衡，而这对于优化训练效率至关重要。

本文提出了MicroEP，一种新颖的并行化策略，旨在通过token调度实现MoE系统的细粒度负载均衡。在此基础上，本文设计并实现了高效的分布式MoE训练系统MicroMoE。

**核心方法学：MicroEP**

MicroEP通过token调度在每个微批次内实现GPU间的负载均衡，其关键在于在EDP（Expert Data Parallelism）组之间调度tokens，并通过重新洗牌专家放置（expert placement）来扩展调度空间。

1.  **调度空间与前提：**
    *   传统的EP（Expert Parallelism）范式中，每个EP组包含每个专家的一个副本，token到GPU的映射由token到专家的映射固定，没有调度空间。
    *   MicroEP的关键观察是，如果DP（Data Parallelism）度大于EP度，且每个GPU承载多个专家副本，那么一个token可以在其指定专家的EDP组内任何GPU上的副本进行计算。这允许合并多个EP组形成一个MicroEP组，并在其中进行token调度。
    *   为了扩大调度空间并实现全局负载均衡，MicroEP在合并的EP组内随机打乱专家放置，使得不同专家的EDP组可以相互交叉。

2.  **Token调度算法：**
    *   **确定副本负载（Determining Replica Loads）：**
        将负载均衡问题建模为一个线性规划问题（LPP 1），目标是最小化MicroEP组内所有GPU上的最大负载。
        设 $E$ 为所有专家的集合，$G_{MicroEP}$ 为MicroEP组内所有GPU的集合，$G^e_{EDP}$ 为专家 $e$ 的EDP组中的GPU集合。变量 $x^g_e$ 表示专家 $e$ 在GPU $g$ 上的副本负载（即tokens数量）。$load_e$ 表示专家 $e$ 的总负载。
        优化问题表述为：
        $$ \text{minimize } \max_{g \in G_{MicroEP}}\left\{\sum_{e \in E:g \in G^e_{EDP}} x^g_e \right\} $$
        $$ \text{subject to } \sum_{g \in G^e_{EDP}} x^g_e = load_e, \quad \forall e \in E $$
        $$ x^g_e \ge 0, \quad \forall e \in E, g \in G^e_{EDP} $$
        该LPP可在每个微批次中高效求解，通过复用前一个解的中间状态进行“热启动”（warm-start），显著减少优化开销。
    *   **路由Tokens到副本（Routing Tokens to Replicas）：**
        一旦确定了每个副本的负载 $x^g_e$，就需要将tokens路由到具体的专家副本以强制执行这些负载。这通过一个顺序路由策略实现。
        为了减少All-to-All（A2A）通信量，MicroEP采用局部性感知路由（locality-aware routing）：当GPU $g$ 包含专家 $e$ 的副本时，优先将来自GPU $g$ 的tokens路由到其本地专家副本，然后再考虑远程副本。此外，LPP也可以扩展为通信感知调度，将通信时间纳入优化目标，通过 $\alpha$ 参数平衡计算和通信的权重。

3.  **分布式调度与延迟隐藏：**
    *   **分布式调度：** 调度算法在所有设备上分布式执行，每个设备通过All-gather操作收集全局负载信息，然后独立执行确定性调度算法，确保结果一致性。
    *   **延迟隐藏：** 通过与Megatron-LM中现有操作（如token排列）重叠来隐藏调度延迟。对于没有合适重叠操作的框架，MicroEP提出了一种流水线机制（pipelining），将tokens分成两部分，交错执行EP的A2A通信和MicroEP的调度，以进一步隐藏延迟。

4.  **专家放置策略（Expert Placement）：**
    专家放置决定了EDP组的构成，从而直接影响负载均衡能力。
    *   **最优专家放置分析：** 通过图论抽象（GPU为顶点，专家为连接EDP组中所有顶点的超边），LPP 1的最优目标值 $m$ 等于图G中所有诱导子图密度（induced subgraph density）的最大值。
        $$ m = \max_{G_{max} \subseteq G_{MicroEP}}\left\{\frac{1}{|G_{max}|} \sum_{e \in E:G^e_{EDP} \subseteq G_{max}} load_e \right\} $$
        因此，最优专家放置是使最大诱导子图密度最小化的图。
    *   **对称放置（Symmetric Placement）：** 在没有预先专家负载分布知识时，采用对称放置，将所有专家同等对待。本文提出使用Cayley图（Cayley graphs）来构建近乎最优的对称放置，利用其固有的对称性确保边的均衡分布。
    *   **非对称放置（Asymmetric Placement）：** 如果已知真实的专家负载分布，则可以构建非对称放置。本文采用启发式两步策略：首先通过贪心算法确定每个专家的副本数量，然后通过Monte Carlo采样确定专家副本在GPU上的具体放置。
    *   **自适应替换（Adaptive Replacement, AR）：** AR机制补充了token调度，处理长期、粗粒度的负载不均衡。放置管理器在训练期间监视专家负载分布，并通过时间序列分析预测未来负载。如果未来性能下降到特定阈值以下，它将生成新的最优非对称放置并重新初始化模型状态。与现有专家调度方案不同，AR是MicroMoE中token调度的优化，而非主要负载均衡手段。

<img width="947" height="602" alt="image" src="https://github.com/user-attachments/assets/d59e82ca-6b02-4709-9467-4bfaa7262dc6" />

<img width="451" height="281" alt="image" src="https://github.com/user-attachments/assets/bfcefc4c-2a14-4920-bbac-6a5f61245b64" />

<img width="476" height="292" alt="image" src="https://github.com/user-attachments/assets/d1c05fad-f71b-43ab-bb14-9845f734b289" />

**实验结果：**
MicroMoE相比Megatron-LM端到端训练吞吐量提升高达47.6%，平均提升36.9%。即使在高度不平衡的工作负载下，也能几乎持续地实现GPU间的最佳负载均衡。调度开销极低（64 GPU和256专家下小于1ms），而自适应替换开销（模型状态迁移）在数百毫秒级别，需权衡替换频率。消融研究表明，热启动、局部性感知路由和调度重叠等优化有效减少了调度时间。

4节点H100（4*8=32卡），400Gb RDMA x2 per node。基于Megatorn-LM修改。模型：Mixtral， GPT
## ThreadWeaver
ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models

https://arxiv.org/pdf/2512.07843 Meta 田渊栋等，2025.12.10

1.  🚀 ThreadWeaver是一个**自适应并行推理框架**，通过**分解问题解决过程**为并发推理线程，显著降低了大型语言模型（LLMs）在复杂任务上的推理延迟。
2.  💡 该框架的核心创新包括**两阶段并行trajectory生成器**、基于**trie的训练-推理协同设计**（无需修改现有推理引擎例如kv cache），以及**并行化感知的强化学习框架**P-GRPO，以平衡准确性和高效并行化。
3.  🏆 在六项数学推理基准测试中，ThreadWeaver在Qwen3-**8B上实现了与顶尖顺序推理模型相当的准确率**（平均71.9%），同时将token **latency平均加速了1.53倍**，建立了新的性能-效率帕累托前沿。
vllm + FSDP；

<img width="834" height="453" alt="image" src="https://github.com/user-attachments/assets/298920a8-b3aa-46c1-b3e3-b0cbd4871597" />
ThreadWeaver是一种为大型语言模型（LLM）设计的自适应并行推理框架，旨在解决现有方法在复杂任务上推理延迟高以及准确性下降的问题。该框架通过引入并发推理线程，显著降低了推理延迟，同时保持了与先进序列推理模型相当的准确性。

**核心挑战：**
文章指出了当前自适应并行推理面临的三个主要挑战：
1.  **高质量并行推理轨迹难以获取：** 为真实世界复杂问题生成高质量的并行推理轨迹进行训练成本高昂，且现有LLM在生成复杂并行轨迹方面仍有困难。
2.  **依赖定制推理引擎：** 大多数现有方法需要修改LLM的 `position embeddings`、 `KV caches` 或 `attention masks`，导致**部署复杂且难以兼容标准推理框架**。
3.  **强化学习（RL）探索不足且难以扩展：** 在并行推理轨迹上进行RL训练引入了额外的建模和系统挑战，例如跨分支的 `advantage calculation` 和训练与测试执行之间的一致性。

**ThreadWeaver的创新与方法：**
ThreadWeaver通过三项关键创新来解决这些挑战：
1.  **两阶段并行轨迹生成器（Two-Stage Parallel Trajectory Generator）：**
    *   **第一阶段：LLM重写（LLM Rewriting）**：首先从Qwen3-8B生成的**顺序推理轨迹中提取，并使用一个更强大的LLM（如GPT-5）对其进行轻量级重写和注释**，以**识别并标记可并行化的部分**。重写过程仅进行局部修改，例如移除跨线程依赖、平滑过渡，并添加 `<Outlines>`。它不会重新生成整个 `chain-of-thought`，从而保留了原始推理轨迹的探索和自我反思。**生成的轨迹遵循特定的** `fork-join` 格式，包含 `<think>`、`<Parallel>`（包含 `<Outlines>` 和多个 `<Thread>`）等控制令牌。
    *   **第二阶段：自训练与奖励过滤（Self-training with Reward-based Filtering）**：在第一阶段生成的少量高质量数据上进行 `supervised fine-tuning` (SFT) 后，模型会根据自身生成能力在完**整数据集上生成大量并行轨迹**。这些轨迹通过 `answer correctness` 和 `structural validity` 过滤，形成一**个更大、与模型自身生成模式更匹配的数据集**，用于进一步的SFT，从而稳定并行结构的生成。
<img width="815" height="520" alt="image" src="https://github.com/user-attachments/assets/1835c583-5fdf-4e1f-a8b6-6384a90a7cb6" />

2.  **基于Trie的训练与推理协同设计（Trie-Based Training and Inference Co-Design）：**
    *   **并行推理机制：** ThreadWeaver的推理采用状态机模式，通过轻量级控制令牌实现 `fork-join` 结构。模型生成到 `<Outlines>` 标签时停止顺序解码，解析出 `<Outline>` 条目。然后，每个 `<Thread>` 的内容作为独立的 `completion request` 并行发送给标准 `autoregressive inference engine` (如vLLM或SGLang)，并以 `</Thread>` 为停止符。所有并行线程完成后，结果被合并，主线程继续顺序解码。这种设计无需修改底层LLM架构或 `KV caches`，兼容现有服务优化（如 `paged attention`、 `prefix caching`）。
    *   **Trie构造与损失计算：** 为了训练模型生成这种结构，训练数据通过构建 `token-level prefix tree` (trie) 来处理。每个 `<context, completion>` 对（对应推理时的API请求-响应）被插入到trie中，共享前缀合并，分支代表不同的延续。通过深度优先遍历将trie扁平化为一个单一的训练序列，并应用 `ancestor-only attention mask` 以防止跨线程信息泄露。损失仅应用于 `completion token`。这种协同设计确保了训练时和推理时 `position embeddings` 和上下文的一致性，同时允许模型在纯顺序模式下运行。
<img width="840" height="424" alt="image" src="https://github.com/user-attachments/assets/cd199c9d-bc1b-4a1e-a535-b42435c3c62c" />

3.  **并行化感知强化学习（P-GRPO - Parallelization-Aware GRPO）：**
    *   **GRPO的修改：** ThreadWeaver采用 Group Relative Policy Optimization (GRPO) 的变体P-GRPO。它将轨迹级别的 `advantage` $A_{P-GRPO_{p,i}}$ 广播到该轨迹中的所有令牌。
    *   **轨迹分解：** 一个并行轨迹 $\tau_{(p,i)}$ 被分解为一系列有序的 `(context, completion)` 单元 $(cont_{(i,m)}, comp_{(i,m)})$。整个轨迹的对数概率可以分解为所有 `completion segment` 的对数概率之和：
        $\log \pi_{\theta} (\tau_{(p,i)}) = \sum_{m=1}^{M_i} \log \pi_{\theta}(comp_{(i,m)} | cont_{(i,m)})$
    *   **优势计算：** P-GRPO计算 `group-normalized advantage`，但为了提高稳定性并维持 `correctness` 和 `acceleration` 奖励之间的平衡，它移除了标准差归一化，仅使用均值中心化：
        $A_{P-GRPO_{p,i}} = r_{p,i} - \mu_p$
        其中 $r_{p,i}$ 是轨迹的奖励，$\mu_p$ 是该 `prompt` 所有 `rollout` 的平均奖励。
    *   **并行化感知奖励：** 总奖励 $r(\tau)$ 由两部分组成：
        *   `Correctness reward` $R_{correct}(\tau) = \mathbf{1}_{\{Correct(\tau)\}}$：如果最终答案正确则为1。
        *   `Acceleration reward` $R_{accel}(s) = \mathbf{1}_{\{Correct(\tau)\}} \min (\rho \cdot \eta(s), \rho_{clip})$：只有当答案正确时才给予，鼓励更有效的推理路径。
            加速比 $\eta(s) = 1 - \frac{L_{longest}}{L_{total}}$，其中 $L_{longest}$ 是最长线程的序列长度（`token latency`），$L_{total}$ 是整个轨迹中的总令牌数。$\rho$ 是加速奖励比例，$\rho_{clip}$ 是裁剪阈值。
<img width="848" height="422" alt="image" src="https://github.com/user-attachments/assets/10084aa0-75ce-4848-8661-bfd14dc648a8" />

**实验与结果：**
<img width="844" height="397" alt="image" src="https://github.com/user-attachments/assets/a96dca2d-5386-47e5-a037-4539ab25c8b3" />

<img width="865" height="327" alt="image" src="https://github.com/user-attachments/assets/ddddcd8f-32d9-44d4-ad87-191f05e0b28c" />

ThreadWeaver在Qwen3-8B模型上进行训练，并在AIME24、AIME25、AMC23、MATH500、Minerva Math和OlympiadBench等六个数学推理基准上进行评估。
*   **性能提升：** ThreadWeaver的平均准确率（71.9%）与顺序RL基线（72.2%）相当，但在 `token latency` 上实现了显著降低（平均从15.1k降至13.2k），平均加速比达到1.22倍，最高可达1.53倍。在AIME24上准确率甚至略有提升（79.9% vs 78.3%）。这建立了准确性和效率之间的新 `Pareto frontier`。
*   **与现有方法的比较：** 相比Multiverse和Parallel-R1，ThreadWeaver在AIME24上实现了更高的准确率（79.9% vs 53.8%或19.4%）和更高的 `self-parallelism speedup`。ThreadWeaver在 `long chain-of-thought` 模式下运行，与强大的顺序推理模型相匹配。
*   ** `Wall-clock` 加速：** 在实际部署中，通过将并行线程分配到多GPU上，ThreadWeaver实现了1.14倍的 `wall-clock latency` 降低，验证了 `critical-path` 长度减少能够转化为实际的端到端加速。
*   **消融研究：**
    *   移除GRPO中的标准差归一化提高了训练稳定性，并带来了更高的准确率和更短的 `critical path`。
    *   高质量、与模型对齐的SFT数据对下游性能至关重要，优于使用其他LLM生成的轨迹（即使教师模型更强大）。
    *   RL阶段的并行 `rollout` 和自训练对提高准确率和推理效率均有贡献。
<img width="850" height="449" alt="image" src="https://github.com/user-attachments/assets/1d954d14-9388-436f-a658-6fa99caccd35" />


## CMU RL的能力来源和条件
On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models 

https://arxiv.org/abs/2512.07783 CMU 2025.12.8

中文解读：https://mp.weixin.qq.com/s/PrjOBHUuCktIEyXOmotPGg

<img width="668" height="202" alt="image" src="https://github.com/user-attachments/assets/1d8746a8-7c8e-4d0e-a423-44524af85a26" />
构建了一个完全可控的实验框架，隔离预训练、中间训练和基于RL的后训练的因果贡献。该框架基于三个原则：
(1) 具有显式原子操作和DAG定义依赖结构的完全可控合成推理任务；
(2) 可观察、可解析的推理过程，支持过程级评估并减少奖励或评估欺骗；
(3) 系统操纵预训练/中间训练/后训练分布，将因果效应归因于每个阶段。
<img width="1048" height="427" alt="image" src="https://github.com/user-attachments/assets/839c0503-53ed-4bef-bf92-fd67d0207d93" />

论文使用100M参数的Qwen2.5模型，在包含30B Token的大规模合成推理数据集上训练，该数据集跨越多个操作范围和上下文模板，并被划分为不相交的预训练、中间训练和后训练分割以避免分布污染。
论文沿两个关键维度：
- 外推泛化 (Extrapolative Generalization) (深度)：评估模型解决比预训练中遇到的问题更复杂的能力，即 op(G)op(G)op(G)op(G) 超出训练范围的问题。问题分为三类：In-Distribution (ID, op=2-10op=2−10op=2-10op=2−10)、OOD-edge (op=11-14op=11−14op=11-14op=11−14) 和 OOD-hard (op=15-20op=15−20op=15-20op=15−20)。
- 上下文泛化 (Contextual Generalization) (广度)：评估模型将推理技能迁移到表面形式不同但底层逻辑相似的新颖上下文的能力。

**发现1：RL能力提升的条件**

仅在下面两个条件同时成立时 RL才能产生提升能力边界（pass@128)：

(1) 任务在预训练期间没有被大量覆盖，留下足够的探索空间。

(2) 模型的"能力边界"（Edge of Competence）处，既问题不太简单（分布内）也不能太困难（分布外）。

分布内任务（op=2-10）上，无论RL数据方案如何，pass@1有明显的性能提升但pass@128没有提升，表明**RL锐化了现有能力而没有扩展它们**。
分布外任务（op=11-14和op=15-20）：对于边界（op=11-14）时，RL**始终提升pass@128性能**，达到+42%的提升，证明了超越预训练的真正能力提升

<img width="1039" height="368" alt="image" src="https://github.com/user-attachments/assets/755e9702-cdda-4939-90dc-0de45b03168f" />

**发现2：广度泛化能力提升等条件**

ctxB**需要在预训练中有最小曝光**。当预训练排除长尾上下文B或提供很少曝光（0%或0.1%）时，RL无法迁移到上下文B。
**引入1%的**上下文B数据，post-training泛化就能显著增强，甚至可以泛化到最困难的op=20任务，达到+60% pass@128的提升。

<img width="1059" height="367" alt="image" src="https://github.com/user-attachments/assets/313b59b0-8e14-472f-b56b-30104231992d" />

**发现3：Mid+RL混合效果：问题越难 越需要更多RL**
固定计算预算下，引入一个桥接预训练和后训练分布的中间训练阶段显著增强了分布内和分布外性能。中间训练+RL在OOD-hard任务上比仅RL高出+10.8%。
比较了五种mid+RL配比：完全中间训练、完全RL、Light-RL（β=0.2）、Medium-RL（β=0.5）和Heavy-RL（β=0.8）。结果显示，
在OOD-edge任务上，完全中间训练和轻量RL配置优于重量或完全RL；

**对于OOD-hard任务，将更多预算重新分配给重量RL显著提升了最困难实例的性能**

<img width="1054" height="376" alt="image" src="https://github.com/user-attachments/assets/9b6748a6-5b90-40e2-b8a9-24c07cb5dc66" />

**发现4：结果奖励容易引发欺骗；过程奖励可以改善。二者结合最好**
过程级奖励（Process-Level Rewards）能够减少奖励欺骗（Reward Hacking），提升推理的真实性。
通过将过程验证纳入奖励函数，使强化信号与有效推理行为对齐，在复杂的组合设置下，pass@1在外推任务（op=15-20）上提升了4-5%。
<img width="1060" height="314" alt="image" src="https://github.com/user-attachments/assets/3f894c9f-ede1-436e-b3ea-190a404e91af" />

<img width="812" height="351" alt="image" src="https://github.com/user-attachments/assets/2f3fd6ce-f65d-469e-a570-98873417c976" />


## NPR
Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning
https://arxiv.org/abs/2512.07461 朱松纯团队 2025.12.8

https://github.com/bigai-nlco/Native-Parallel-Reasoner

https://huggingface.co/bigai-NPR

中文解读：https://mp.weixin.qq.com/s/LusROLFlk-m0giJ3wXbALw 
1. 💡 本文提出了 Native Parallel Reasoner (NPR)，这是一个**teacher-free框架**，通过自蒸馏渐进式训练使 Large Language Models (LLMs) 能够自主演化出真正的并行推理能力。
2. 🚀 NPR 引入了新型 Parallel-Aware Policy Optimization (**PAPO**) 算法，直接在执行**图中优化分支策略以学习自适应分解**，并设计了鲁棒的NPR Engine确**保大规模并行RL训练的稳定性**。
3. 📈 在八个推理基准测试中，NPR 在 Qwen3-4B 上实现了24.5% 的性能提升和 **4.6 倍的推理加速**，并展现了 100% 的 genuine parallel execution

<img width="1012" height="314" alt="image" src="https://github.com/user-attachments/assets/2c22c7b1-34a3-4edb-86ec-d151c8840869" />

这篇论文介绍了**Native Parallel Reasoner** (NPR)，这是一个teacher-free的框架，旨在使大型语言模型（LLMs）能够自我演化出原生的并行推理能力。传统LLMs在“更深”的测试时扩展（single-path reasoning）上表现出色，但在需要探索多条轨迹的“更广”推理能力方面（parallel reasoning）存在不足。现有的并行推理实现面临三大挑战：
1.  **算法与架构不兼容**：主流的推理引擎和强化学习（RL）算法未能**有效支持原生分支操作**，尤其是特殊令牌（special tokens）的梯度裁剪，阻碍了模型学习严格的并行结构。
2.  **低效的手工并行化**：早期方法通过独立采样实现并行，但**未能利用共享的Key-Value (KV)状态，导致冗余计算和线性（$O(N)$）的延迟成本**。
3.  **依赖监督蒸馏**：例如Multiverse等框架**依赖于从更强的教师模型中蒸馏数据**，这限制了**学生模型发现新颖、模型内在**的并行策略的能力。

为此，NPR提出了一个**三阶段的渐进式训练范式**，将模型从**顺序模拟**（sequential emulation）**转变为原生的并行认知**（native parallel cognition）。
<img width="683" height="386" alt="image" src="https://github.com/user-attachments/assets/2ce503a5-1a4f-49ee-8e48-e99988c79a94" />

**核心方法（Methodology）**

NPR的训练过程分为三个阶段：
<img width="824" height="527" alt="image" src="https://github.com/user-attachments/assets/db08f78d-e000-4384-aa99-504dd43e7f2b" />

**1. 阶段1：格式遵循强化学习**
*   **目标**：在没有外部标注的情况下，引导模型自发发现并生成结构化的并行推理格式。
*   **并行格式**：NPR采用了一种简化的“Map-Process-Reduce”范式，灵感来源于Multiverse但结构更精简。每个并行块以`<guideline>...</guideline>`开始，包含`<plan>...</plan>`条目定义Map阶段。Process阶段通过`<step>...</step>`块独立并行执行子任务。最后，**Reduce阶段通过`<takeaway>...</takeaway>`聚合结果**。这种基于标签的格式便于解析和验证。
*   **奖励函数**：使用DAPO (Yu et al., 2025) 算法。奖励函数结合了格式（format）和准确性（accuracy）信号。
    *   **格式奖励**：通过格式检查的输出获得0.0奖励，失败则惩罚在$(0.0, -2.0]$之间。
    *   **准确性奖励**：格式检查通过后，正确答案获得+1.0，错误答案获得-1.0。
*   **产物**：此阶段训练出的模型称为NPR-ZERO，它主要学习生成所需的结构化格式。其生成的数据用于后续阶段的大规模自蒸馏（self-distillation）。

**2. 阶段2：拒绝采样与并行热身**
*   **目标**：通过对NPR-ZERO**生成的轨迹进行拒绝采样**，构建高质量的自蒸馏数据集，并进行监督微调（SFT），以稳定化模型对并行原语的生成能力。
*   **结构化轨迹收集**：
    *   对于数据集中的每个问题$q_i$，模型生成$K$个候选推理轨迹和答案$\{(\hat{r}_{ij}, \hat{a}_{ij})\}_{j=1}^K$。
    *   **拒绝采样过滤器**：
        *   **结果正确性 (Outcome Correctness)**：丢弃答案$\hat{a}$与真实答案$a_i$不匹配的轨迹，即$1_{\text{correct}}(\hat{a})$。
        *   **结构化并行性 (Structured Parallelism)**：丢弃未能遵循预定义结构化输出格式的轨迹，即$1_{\text{format}}(r)$。
    *   **接受标准**：样本**仅在满足两个标准时被接受**：$1_{\text{accept}}(r, \hat{a}) = 1_{\text{correct}}(\hat{a}) \cdot 1_{\text{format}}(r)$。
    *   **数据集**：接受的轨迹形成自蒸馏数据集$D_{\text{accept}} = \{(q_i, r_{ij}, a_{ij}) | i \leq N, j \leq K, \text{s.t.} (r_{ij}, a_{ij}) \sim \pi_{\theta}(\cdot|q_i), 1_{\text{accept}}(r_{ij}, \hat{a}_{ij}) = 1\}$。
*   **并行注意力掩码与位置编码 (Parallel Attention Mask & Positional Encoding)**：
    *   采用**Multiverse Attention (Yang et al., 2025a) 的核心设计**，构建并行注意力掩码和对应的位置编码。这使得**多个推理路径可以在一个前向传播中并行存在**，并**支持KV缓存共享以减少推理开销**。
    *   通过初始化特殊令牌来暴露结构化标签，以确保模型可以生成这些标签。
*   **并行热身**：在$D_{\text{accept}}$上对模型进行监督训练（使用标准负对数似然）。此阶段产物为NPR-BETA，作为后续并行RL的稳定初始化。

**3. 阶段3：Native-parallel RL)**
*   **目标**：进一步**放大和泛化并行推理能力，**通过直接优化模型在并行执行图中的分支策略。
*   **Parallel-Aware Policy Optimization (PAPO)**：对标准RL（**基于DAPO）进行多项实践修改以适应并行语义：**
    1.  **并行Rollout**：使用NPR Engine (§2.5) 进行采样，确保**所有生成的轨迹都遵循预期的Map–Process–Reduce流程**。
    2.  **结构化过滤 (Structural Filtering)**：在Rollout期间，**对不符合并行Schema的malformed序列进行过滤**，确保**保留的轨迹严格遵守目标结构**。过滤后奖励仅基于准确性（正确最终答案+1，否则-1）。
    3.  **批次级优势归一化 (Batch-level Advantage Normalization)**：由于违反格式的样本被移除，导致组级方差塌陷，因此采用**Lite-PPO **(Liu et al., 2025a) 风格的优势函数，将组级方差替换为批次级方差：
        $\hat{A}_{i,t} := \frac{R_i - \text{mean}(\{R_1, R_2, \cdots, R_G \})}{\text{std}(\{R_1, R_2, \cdots, R_G, \cdots, R_{N \times G}\})}$
        其中$N$是批次大小，$G$是组大小，$R$是准确性奖励。
    4.  **保留特殊令牌的梯度 (Preserve Gradients on Special Tokens)**：**特殊令牌对于维持并行语义至关重要，因此移除了其梯度裁剪**，确保它们**始终获得梯度**。为了避免PPO中重要性采样的不稳定性，NPR消除了重要性采样，采用**严格的on-policy目标。**
*   **PAPO目标函数**：
    $J (\theta) = E_{(q,y)\sim D, \{\hat{y}_i \}^G_{i=1}\sim\pi_{\theta} (\cdot|q)} \left[ -\frac{1}{\sum_{i=1}^G |\hat{y}_i|} \sum_{i=1}^G |\hat{y}_i| \sum_{t=1} \left( \pi_{\theta} ( \hat{y}_{i,t} | q, \hat{y}_{i,<t}) \cdot \text{sg}[\pi_{\theta} ( \hat{y}_{i,t} | q, \hat{y}_{i,<t})] \cdot \hat{A}_{i,t} \right) \right]$
    其中$\text{sg}[\cdot]$表示stop-gradient操作。
    
<img width="829" height="456" alt="image" src="https://github.com/user-attachments/assets/f7f70c00-d288-438f-9adb-73edee5e01a8" />

**NPR Engine**
为了支持大规模并行RL的稳定Rollout，NPR重新设计了**Multiverse的并行生成引擎**，解决了**SGLang在生产规模下存**在的以下稳定性问题：
1.  **KV缓存双重释放和内存损坏 (KV-cache double-free and memory corruption)**：在高并行分支下，**共享的radix-tree KV路径可能被多次回收，导致上下文损坏和GPU内存泄漏**。**解决方案**：用显式、预算感知的回收策略替代了机会性回收。
2.  **全局令牌预算低估 (Underestimated global token budget)**：并行解码会数倍增加总令牌消耗，但原始引擎仅跟踪最长的单个分支。**解决方案**：扩展了长度核算以适应分支，引擎现在记录每个扩展处的活跃分支因子并相应更新全局令牌账本。
3.  **非法并行Schema导致的未定义状态 (Undefined states from illegal parallel schemas)**：某些并行分支布局导致引擎条件逻辑出现未定义状态。**解决方案**：增加了轻量级预分支格式验证器，强制执行少量结构不变性检查。
4.  **`<step>`块内部局部重复 (Local repetition inside `<step>` blocks)**：细粒度步骤流容易出现局部重复。**解决方案**：对`<step>...</step>`上下文中的令牌施加温和、选择性的重复惩罚（系数1.02）。

**实验结果**
<img width="825" height="325" alt="image" src="https://github.com/user-attachments/assets/e6f61023-fefe-4d86-8470-b8faf7122636" />

<img width="811" height="182" alt="image" src="https://github.com/user-attachments/assets/887098f9-b858-4223-a354-fb8384ca2cbe" />

<img width="843" height="273" alt="image" src="https://github.com/user-attachments/assets/c2739fd7-dbc2-46ae-8250-12a7b3801d1a" />

<img width="810" height="303" alt="image" src="https://github.com/user-attachments/assets/21c8380b-89bf-4326-9f01-7a4709868452" />

<img width="828" height="612" alt="image" src="https://github.com/user-attachments/assets/cb8e6e40-91be-4a16-bed4-3a59c8e771be" />

<img width="743" height="598" alt="image" src="https://github.com/user-attachments/assets/de462382-f1f1-4e58-a525-85c1191a3f64" />

NPR在八个推理基准测试（如AIME25, AIME24, HMMT25, OlympiadBench, Minerva-Math, ZebraLogic, AMC23, MATH500）上进行了评估，使用Qwen3-4B-Instruct-2507和Qwen3-4B（非思考模式）作为基础模型。
*   **性能提升**：NPR相对于基线模型实现了显著的性能提升，例如在Qwen3-4B上平均性能提升超过24.5%。
*   **自蒸馏数据优势**：NPR的自蒸馏数据集平均比Multiverse的教师生成轨迹高10.1分。
*   **并行效率和有效性**：NPR-BETA和NPR-RL均显著优于直接的顺序RL基线（如DAPO），证实自适应并行策略提供了优于单路径Rollout的搜索机制。
*   **100%原生并行执行**：NPR在所有评估测试案例中实现了100%的原生并行推理，没有隐藏的自回归（AR）回退或伪并行行为，而之前的基线存在30%以上的AR回退。
*   **推理加速**：NPR提供了高达4.6倍的推理加速（相对于AR解码），特别是在更难的问题上加速效果更明显。
*   **测试时可扩展性**：NPR在测试时可靠地增加了oracle覆盖率，尤其在基础模型较弱时提升更显著。
*   **定性案例研究**：NPR能够根据问题类型自适应其并行度，例如在创造性任务中进行广泛探索，在逻辑任务中进行严格的交叉验证。

**结论**

NPR框架通过自蒸馏并行SFT和agentic并行RL，实现了LLM自主演化原生并行推理能力。该方法使模型能够学习自适应分解、多样化的并行规划和可靠的聚合，而非模拟或脚本化的行为。这些结果表明，原生并行推理是通往更通用和可扩展智能的未来方向。

## TileRT
中文介绍 https://mp.weixin.qq.com/s/5T-93n5kk7UbHXj_I3NIvw 

code：https://github.com/tile-ai/TileRT

## DS-GRPO
Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning 

https://arxiv.org/pdf/2511.19942 CMU 清华等，2025.11.25
1. 📚 该研究深入分析了RL微调大型语言模型时出现的输出**多样性崩溃**问题，并从第一性原理上提出了**选择偏置和强化偏置**是导致正确轨迹多样性下降的根本原因。
2. 💡 基于此诊断，论文提出了一种名为 **Differential Smoothing **(DS-GRPO) 的原则性方法，通过对正确轨迹应用熵奖励并对不正确轨迹应用熵惩罚，实现了**差异化的奖励修改**。
3. 📈 理论证明DS-GRPO能够同时**提高模型的正确性**（Pass@1）和**多样性（Pass@K）**，并通过在**CountDown和数学推理任务上的广泛实验**，一致性地展示出其优于现有方法和香草RL的性能。

深入探究了强化学习（RL）微调大型语言模型（LLM）时普遍存在的**多样性崩溃（diversity collapse）**问题。研究指出，RL微调模型通常会导致**输出缺乏多样性**，从而在诸**如Pass@K等指标上表现不佳**，甚至在K值较大时可能不如原始基础模型。现有缓解该问题的方法往往需要在正确性（Pass@1）和多样性（Pass@K）之间进行权衡，效果不一致，甚至相互矛盾。
核心诊断：RL中的**选择偏置和强化偏置**（Selection Bias and Reinforcement Bias）
该研究首先从第一性原理出发，对RL微调导致多样性崩溃的机制进行了形式化分析，并引入了两个关键概念：
<img width="1026" height="321" alt="image" src="https://github.com/user-attachments/assets/ef0a8a0f-c21a-4f13-a747-837412a5721e" />
<img width="959" height="358" alt="image" src="https://github.com/user-attachments/assets/71146cf3-8bca-447a-8205-6cb811e161c4" />
<img width="1061" height="319" alt="image" src="https://github.com/user-attachments/assets/7c47f3cb-8511-4fe7-bcd0-a8a2e47c9350" />
<img width="1082" height="295" alt="image" src="https://github.com/user-attachments/assets/ac21df42-e1d4-498e-bac9-a5efd6e07719" />
<img width="1070" height="437" alt="image" src="https://github.com/user-attachments/assets/f0207d30-82e1-4b69-906c-75bf1969170f" />

研究强调，这些偏置特指对**正确轨迹**的影响。相反，不正确的轨迹则表现出相反的动态：高概率的错误会受到更强的惩罚，使得不正确轨迹的概率分布变得扁平化。这种不对称的效应导致了多样性与正确性之间的内在权衡。

**核心方法：差分平滑（Differential Smoothing, DS）**
<img width="924" height="188" alt="image" src="https://github.com/user-attachments/assets/eabbb4d5-7f43-47d3-96ae-95bf4ce4ef49" />
<img width="973" height="141" alt="image" src="https://github.com/user-attachments/assets/e88fb2ed-2a0f-46e9-aa55-10845cafe2cb" />

基于上述诊断，该研究提出了一种名为**差分平滑（Differential Smoothing）**的原则性方法。核心思想是：只需要在**正确轨迹**上修改奖励函数以防止多样性崩溃，而不正确轨迹可以继续使用原始奖励或甚至鼓励“锐化”的修改。这种对正确和不正确轨迹施加不同压力的差异化奖励机制，能够同时改善正确性和多样性，克服了现有启发式方法的局限性。

差分平滑通过修改奖励函数$r_{\text{DS}}(\tau)$来实现：
$$
r_{\text{DS}}(\tau) = \begin{cases} \hat{r}(\tau) - \gamma_p \cdot \log(\pi_{\text{base}}(\tau)) & \text{if } r(\tau) > 0 \quad (\text{correct trajectories}) \\ r(\tau) + \gamma_n \cdot \log(\pi_{\text{base}}(\tau)) & \text{if } r(\tau) \le 0 \quad (\text{incorrect trajectories}) \end{cases}
$$
其中，$\gamma_p, \gamma_n \ge 0$是超参数。

*   **对正确轨迹（$r(\tau) > 0$）**：通过减去一个与轨迹在基础模型下对数概率（$\log \pi_{\text{base}}(\tau)$）成比例的项（即**熵奖励，entropy bonus**），来鼓励模型给那些在基础模型下不那么常见的正确轨迹分配更高的概率。这直接对抗了选择偏置和强化偏置，促进了正确解决方案的多样性。
*   **对不正确轨迹（$r(\tau) \le 0$）**：通过加上一个与轨迹在基础模型下对数概率成比例的项（即**熵惩罚，entropy penalty**），鼓励模型给那些不正确的轨迹分配更低的概率。这进一步锐化了对错误轨迹的惩罚，增强了模型的正确性，同时不影响正确解决方案的多样性。

**理论支撑与实现（DS-GRPO）**

研究在理论上证明了这种差分奖励修改的优越性：
**定理 3.4**：假设模型对所有轨迹的奖励有正确估计。对于在熵正则化策略$\pi_{\text{ent}}$中使用的任意参数$\gamma_{\text{ent}} \ge 0$和$\beta_{\text{ent}} > 0$，如果它满足接近度约束$K_{\rho}(\pi_{\text{ent}}, \pi_{\text{base}}) \le \kappa$，那么差分平滑策略$\pi_{\text{DS}}$存在参数$\gamma_{\text{DS}} \ge 0$和$\beta_{\text{DS}} > 0$，使其也满足$K_{\rho}(\pi_{\text{DS}}, \pi_{\text{base}}) \le \kappa$，并且同时满足正确性$C(\pi_{\text{DS}}) \ge C(\pi_{\text{ent}})$和多样性$\sigma(\pi_{\text{DS}}) \ge \sigma(\pi_{\text{ent}})$。其中$C(\pi)$定义为策略$\pi$的正确性，$\sigma(\pi)$定义为正确解决方案的归一化方差，用于衡量多样性。该定理对多种KL散度度量均成立。这意味着DS方法在保持与基础模型接近度的同时，能够同时提升正确性和多样性。

在实践中，差分平滑被集成到Group Relative Policy Optimization (GRPO)框架中，形成了**DS-GRPO**算法。DS-GRPO通过修改GRPO的优势函数（advantage function）$\text{ADS}_i$来实现：
$$
\text{ADS}_i = A_i + \begin{cases} - \gamma_p \log \pi_{\theta_{\text{old}}}(y_i | x) & \text{if } r_i = 1 \\ + \gamma_n \log \pi_{\theta_{\text{old}}}(y_i | x) & \text{otherwise} \end{cases}
$$
这里，$A_i$是原始GRPO的标准化优势，$r_i$是完成$y_i$的奖励。理论分析中使用$\pi_{\text{base}}$，但实践中发现使用前一策略迭代的$\pi_{\theta_{\text{old}}}$在稳定性上表现更优，且两者在参数重整化下被证明是等价的。

**实验验证**

研究在CountDown和数学推理等多个领域，使用1.5B到8B不同大小的模型进行了广泛实验。
*   **对比Vanilla GRPO**：DS-GRPO在Pass@1和Pass@K上均取得了一致的提升，最高可达6.7%的Pass@K提升。同时，它还能在保持相同Pass@K性能的情况下，显著提高推理速度（例如，实现4倍的推理加速）。
*   **对比现有启发式方法**：包括基于熵正则化（全局熵奖励或熵惩罚）和Pass@K优化等方法。实验结果表明，DS-GRPO在所有测试数据集和模型上都表现出鲁棒的优越性，而现有方法往往只在特定设置下有效。
*   **对熵控制的澄清**：研究解释了为什么全局熵奖励和熵惩罚会在不同任务上产生矛盾的效果。全局熵奖励提高多样性但可能损害正确性，适用于具有高“解决方案多样性（Solution Multiplicity）”的任务（即每问题有多个正确解决方案）；而全局熵惩罚提高正确性但牺牲多样性，适用于低解决方案多样性任务。DS-GRPO的差分方法通过对正确样本应用熵奖励，对不正确样本应用熵惩罚，成功地结合了两者的优点，实现了同时提升正确性和多样性。

**结论**

这项工作为RL微调中的多样性崩溃提供了严格的理论基础，并基于此提出了一种新颖且有原则性的差分平滑方法。该方法在理论上和经验上均被证明能够同时改善模型的正确性（Pass@1）和多样性（Pass@K），超越了现有的RL方法和启发式策略。此外，它还澄清了熵控制在LLM微调中的复杂且任务依赖的角色，为未来的研究指明了方向。

## LoRA for RL
https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus 2025.12.2 MindLab

对Kimi K2 base (1T) 采用LoRA进行强化后训练，64卡H100，colocated, on-policy
- 混合并行: 训练（EP TP PP SP），推理（DP TP）
- 稳定性：Replay router，TIS
- LoRA：r=128，dense 和 MoE都用LoRA adaptor；LoRA本身也fully shared，fusion
- 集成实现：verl+megatron-bridge

相关patch
https://github.com/volcengine/verl/pull/4063
https://github.com/modelscope/ms-swift/pull/6714
https://github.com/modelscope/ms-swift/pull/6720
https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/1310
https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/1380
<img width="747" height="624" alt="image" src="https://github.com/user-attachments/assets/96d63c6a-6e82-4435-acdd-0981b00cfb64" />

## 阶跃 PaCoRe-8B 推荐！

paper: https://github.com/stepfun-ai/PaCoRe/blob/main/pacore_report.pdf 2025.12.9

model: https://huggingface.co/stepfun-ai/PaCoRe-8B 

https://github.com/stepfun-ai/PaCoRe
Model:
PaCoRe-8B: Our final PaCoRe-trained model checkpoint!
RLVR-8B-0926: The initial checkpoint of our study, conducted strong reasoning-oriented post-trained on Qwen3-8B-Base.
📚 Data: PaCoRe-Train-8k The high-quality training corpus, including opensource_math, public_mathcontest, synthetic_math and code:

🤗 Stage1-3k: PaCoRe-Train-Stage1-3k

🤗 Stage2-5k: PaCoRe-Train-Stage2-5k

解读：https://mp.weixin.qq.com/s/9sZLZvk2BoY41_a7bSBTXw
<img width="1065" height="435" alt="image" src="https://github.com/user-attachments/assets/df699125-f320-414b-899b-40646c644b24" />
<img width="1081" height="732" alt="image" src="https://github.com/user-attachments/assets/c0142f92-fc48-49f2-8596-04b63e487fe7" />
<img width="1083" height="534" alt="image" src="https://github.com/user-attachments/assets/6d290659-8a0b-407b-9d68-b2141503cafe" />
<img width="918" height="407" alt="image" src="https://github.com/user-attachments/assets/cef65557-236e-41a5-b89c-0449398bbc27" />
<img width="893" height="410" alt="image" src="https://github.com/user-attachments/assets/849795e3-5c91-4f34-b088-dba93bbdd2d6" />
<img width="944" height="228" alt="image" src="https://github.com/user-attachments/assets/63e26ca4-bc5d-4d9a-9721-cfb61282f6e0" />



## 微软NextLat
Next-Latent Prediction Transformers Learn Compact World Models

https://arxiv.org/pdf/2511.05963v1 微软 2025.11.8
Next-Latent Prediction 训练方法解决标准Transformer缺乏压缩历史信息动机的问题，通过在标准自回归训练中加入一个**辅助的自监督任务**——预测下一时刻的潜在状态，引导模型学习紧凑且具有一致动态性的世界模型。这个方法**不改变模型架构和推理过程**，理论上证明了其能促使模型学习到“信念状态”。实验表明，NextLat在world modeling、reasoning、planning 和 language modeling任务上均取得显著提升，在曼哈顿出租车路径规划中，其有效潜在秩比GPT低3倍以上（52.7 vs 160.1），序列压缩率提升了9.2%。
- 能够使 Transformer 的 latent representations 收敛到 belief states，形成更 compact 的内部 world models 和 transition dynamics，从而注入了 recurrent inductive bias。
<img width="846" height="368" alt="image" src="https://github.com/user-attachments/assets/69a17e9b-2119-4c27-8ed6-3c13fa6e8985" />
<img width="340" height="377" alt="image" src="https://github.com/user-attachments/assets/2966ae63-7da5-43d2-893b-e04b1d558226" />
<img width="816" height="329" alt="image" src="https://github.com/user-attachments/assets/ca62accb-7484-4286-9c67-52682a9a2370" />

**核心问题与动机**：标准Transformer通过自注意力机制可以回顾整个历史序列，**缺乏将历史信息压缩成紧凑状态的内在动力**，这可能**导致模型学习到泛化能力差的复杂捷径**。本文旨在为Transformer引入一种**循环归纳偏置**，促使其**学习紧凑、一致的内部世界模型，从而提升泛化能力**。
**NextLat 方法原理**：NextLat在标准的NTP 损失基础上，增加了一个辅助的自监督损失，引入一个轻量级的“潜在动态模型”，该模型接收当前时刻的隐藏状态和下一时刻的输入token，来预测下一时刻的隐藏状态。这个过程强迫模型学习到的隐藏状态必须包含预测未来所需的所有信息，从而使其收敛为“信念状态”。
**理论保障**：从理论上证明，只要同时满足“下一Token 一致性”和“状态转移一致性”，模型学习到的隐藏状态必然会收敛为一个信念状态。信念状态是历史信息的充分统计量，对于预测未来至关重要，这是标准Transformer所不具备的理论保证。

实现细节与损失函数：NextLat的总损失函数由三部分构成，L_next-token 是标准的交叉熵损失，用于预测下一个词元。L_next-h 是预测的下一隐藏状态与真实下一隐藏状态之间的Smooth L1回归损失，为了防止表示崩溃，目标状态的梯度被停止。L_KL 是预测的下一隐藏状态所对应的词元分布与真实下一隐藏状态所对应的词元分布之间的KL散度损失。这类似于知识蒸馏，确保了潜在空间预测的语义一致性。

**对比**：
vs. 标准GPT：GPT没有学习信念状态的压力，容易过拟合。
vs. RNN：NextLat 引入了循环归纳偏置，但避免了RNN 训练时完全串行的瓶颈，保留了Transformer 的并行训练效率。
vs. BST/JTP：与其他旨在学习信念状态的方法BST、JTP相比，NextLat 在计算上更高效，且理论保证更通用，不依赖于特定的可观测性假设，在各类任务中表现也更优越。
vs. MTP：**MTP 等方法在词元空间进行多步预测，容易受局部n-gram 规律影响而产生短视行为**。NextLat 在潜在空间进行预测，更能捕捉深层结构和长期依赖。
实验结果：

**世界建模**（Manhattan Taxi Rides）： NextLat在轨迹有效性、序列压缩、有效latent rank和绕道鲁棒性等指标上表现最佳，学习到的世界模型与真实世界模型更一致，且表示更紧凑。所有模型在next-token test上均达到100%准确率，但NextLat在OOD泛化和compactness上显著优越。
**推理**（Countdown）： NextLat持续优于所有基线，即使在较浅的预测horizon d=1d=1d=1d=1时，也**大幅超越MTP和JTP**。这表明NextLat能够更好地进行前瞻规划，减少了局部性错误，在最终方程的有效性上表现出更高的一致性。
**规划**（Path-Star Graph）： NextLat在所**有图拓扑结构上都接近100%的解决率**，而其他方法（包括BST）在较大图上开始失效。这表明latent space预测更能避免shortcuts learning，并产生更具泛化性的解决方案，从而提升了long-horizon planning能力。
**语言建模**（TinyStories）： NextLat在保持GPT的next-token prediction性能的同时，展**示了最强的长horizon预测能力**（最远可达20个token）。相比之下，BST、JTP和MTP的额外token-level预测目标导致next-token性能下降，并且其表示在长horizon预测上表现不佳。

与现有方法的比较：

与BST和JTP： NextLat与BST和JTP等belief-learning方法进行了比较。BST训练成本极高（O(T^2T2T^2T2)梯度信号），且推理时需要双Transformer编码器。NextLat的训练速度显著快于BST（在TinyStories上快10倍以上），且在belief state学习方面拥有更强的理论保证（与预测horizon dddd无关），避免了JTP对k-observability的严格依赖。
Token-level预测的局限性： 论文指出，token-level监督通常是短视的。NextLat通过将预测转移到latent space，强调latent transition modeling，避免了next-token性能的下降，并通过鼓励学习结构化、预测性表示而非浅层token-level相关性，改善了下游泛化。
与Recurrent Neural Networks (RNNs)： NextLat通过latent transition prediction引入了循环归纳偏置，但避免了RNNs在训练时的严格顺序计算瓶颈。NextLat仅在训练时引入与rollout horizon dddd成比例的额外顺序成本，远小于整个序列长度TTTT，从而保留了Transformer的并行训练效率。NextLat的梯度传播方式也不同于RNNs的截断反向传播，它对Transformer的计算图进行完全反向传播，而latent dynamics model则在“外循环”中展开，避免了梯度估计的偏差。

结论与未来工作：
NextLat通过将next-token训练与自监督latent-space预测相结合，使Transformer能够学习belief-state representations。该方法简单、高效，且在不改变Transformer架构、并行训练效率或推理过程的前提下，在世界建模、推理、规划和语言建模等任务中产生了更紧凑、更具预测性且更可泛化的表示。未来的工作将包括在大规模语言模型预训练中评估NextLat，将其作为微调目标以提升现有模型能力，以及探索更高维度或分层belief states的架构扩展。

## MISA
MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling

https://arxiv.org/abs/2511.00056 北京大学 2025.12.4

https://github.com/pkumelon/MISA

from misa.optimizer import BlockCoordinateDesentOptimizer

base_optimizer = AdamW(model.parameters(), lr=args.learning_rate)

optimizer = BlockCoordinateDesentOptimizer(
      base_optimizer=base_optimizer,
      named_parameters_list=list(model.named_parameters()),
      param_ratio_limit=0.03, # The ratio of trainable parameters
)

1. ✨ MISA提出了一种模块级重要性采样（Module-wise Importance SAmpling）方法，通过将大型语言模型（LLMs）层细**分为更小的模块**，并**基于梯度范数动态分配重要性分数**，从而实现**内存高效的优化**。
2. 💡 该方法采用**加权随机采样机制**，在**确保探索性的同时有效利用重要模块**，并提供在**包含Adam优化器**和随机梯度的非凸随机条件下O(1/√K)的收敛速率保证。
3. 🚀 实验结果表明，MISA在多种微调和预训练任务中，相比于LoRA和现有分层优化方法，在内存效率和性能方面均展现出显著优势。

<img width="1071" height="281" alt="image" src="https://github.com/user-attachments/assets/9d6428a0-4700-4ff7-b0ac-a10ff416e751" />

<img width="1066" height="366" alt="image" src="https://github.com/user-attachments/assets/699434ac-15ef-4d4c-a1b4-bfea22855481" />
<img width="1045" height="313" alt="image" src="https://github.com/user-attachments/assets/0be77e26-ad9f-49e3-b260-ddd535be1be3" />

MISA (Module-wise Importance SAmpling) 是一项针对大型语言模型（LLMs）优化提出的新方法，旨在解决LLM预训练和微调过程中巨大的内存消耗问题。

**背景与现有方法局限性：**
LLMs的训练需要存储优化器状态、梯度和中间激活，导致高昂的内存开销。
1.  **PEFT方法（如LoRA）**：通过冻结大部分预训练参数，仅优化小部分低秩矩阵来节省内存。然而，这种方法限制了模型的适应性，通常导致性能次优。
2.  **Layer-wise优化方法（如BAdam, LISA）**：基于块坐标下降（BCD）思想，顺序优化Transformer层，冻结其他层以节省内存和激活。它们能够实现接近全参数微调的性能，并比LoRA更节省内存。但现有Layer-wise方法存在以下问题：
    *   **Q1：分区策略不佳**：将Transformer层视为同质单元，忽略了层内不同模块（如多头注意力、前馈网络、归一化层）重要性的差异，可能导致不佳的优化性能。
    *   **Q2：采样策略次优**：主要依赖循环或均匀采样模式，未充分利用不同层或模块的重要性差异。LISA虽然考虑了Embedding层和LLM头部的重要性，但对Transformer层仍采用均匀采样。
    *   **Q3：收敛性保证不足**：现有理论分析通常假设无噪声梯度或每个采样块只更新一次，与LLM实际训练中Adam优化器、随机梯度和多步更新的情况不符。

**MISA的核心贡献与方法：**

MISA通过引入模块级优化和改进的重要性采样机制来克服上述限制，并提供严格的收敛性保证。

1.  **C1：模块级优化 (Module-wise Optimization)**：
    *   **模块定义**：MISA将LLM的权重划分为更细粒度的“模块”，而非粗粒度的“层”。一个模块定义为Transformer层内与权重梯度相关的矩阵参数。例如，多头注意力机制中的$W_q, W_k, W_v, W_o$以及前馈网络中的$W_{up}, W_{down}$均被视为独立模块。
    *   **动机**：经验观察表明（如图1所示），同一Transformer层内不同模块的梯度范数差异显著，表明其重要性不同。对层内所有参数进行统一更新可能导致次优结果。理论上，将层分解为更小的模块能保留更多梯度信息。
    *   **内存效益**：这种细粒度的更新策略消除了将整个层加载到内存的必要性，比Layer-wise优化方法更节省内存。

2.  **C2：改进的重要性采样 (Improved Importance Sampling)**：
    *   **原理**：为了在优化效率和探索性之间取得平衡，MISA设计了一种加权随机采样机制。它通过最大化目标函数$\max_{\{p_b\}_{b=1}^B} \sum_{b=1}^B p_b \|g_n^b\|^2 - (1/\eta) \text{KL}(p_b, q_B)$来确定模块的采样概率，其中$\|g_n^b\|^2$是模块$b$的梯度范数（重要性度量），$p_b$是采样概率，$q_B = 1/B$是均匀分布，$\eta > 0$是控制探索与利用权衡的系数。
    *   **采样概率**：该优化问题的闭式解为$p_n^b = \frac{\exp(\eta \|g_n^b\|^2)}{\sum_{j=1}^B \exp(\eta \|g_n^j\|^2)}$。
    *   **实际实现**：由于全批次梯度范数$\|g_n^b\|^2$在LLM训练中不可访问，MISA使用聚合历史随机梯度的经验平均来近似，即$G_n^b = \beta G_{n-1}^b + (1-\beta) \frac{1}{T} \sum_{t=1}^T \|g_{n,t}^b\|^2$，其中$G_n^b$跟踪模块$b$在迭代$n$时的重要性度量，$T$是内循环更新步数。为了消除不同模块参数量对梯度范数计算的影响，实际中采用按参数量缩放后的梯度范数。
    *   **优势**：这种动态、基于重要性的采样策略优于传统的循环或均匀采样，同时通过$\eta$参数确保了对所有模块的探索。

3.  **C3：收敛性保证 (Convergence Guarantees)**：
    *   MISA在非凸、随机和Adam优化器条件下，实现了$O(1/\sqrt{K})$的收敛速率，其中$K$是总的块更新次数。
    *   **挑战与创新**：传统BCD分析依赖于块梯度是全梯度的无偏估计，但这在MISA的多步更新（repeated block updating）策略下不再成立，引入了偏差和噪声的复杂相互作用。MISA通过以下创新解决：
        *   **偏差传播分析**：推导了梯度偏差和随机噪声如何在连续的块更新中累积的递归关系。
        *   **连接块级和全梯度**：算法1中额外的Adam步（Algorithm 1, Line 16: $\theta^{n+1,0}_{\tau_n} \leftarrow \theta^{n,T}_{\tau_n} - \alpha \frac{\beta_1}{1-\beta_1} \frac{m^{n,T}_{\tau_n}}{\sqrt{v^{n,T}_{\tau_n} + \epsilon}}$）在确保从局部块更新到全局变量优化的平滑过渡中起关键作用。
        *   **新分析工具**：开发了共同控制梯度偏差和放大随机噪声的分析工具。

**MISA算法流程 (Algorithm 1)：**
MISA采用双循环结构：
*   **外循环 (Outer Loop)**：迭代$N$次，每次选择一个或一组模块进行优化。
    *   首先将模型划分为$B$个模块。
    *   根据当前模块的重要性估计$G_n^b$更新模块的采样概率$p_n^b$。
    *   根据采样概率$P^n$选择一个（或一组）模块$\tau_n$，并确保可训练参数占总参数的比例低于$\delta$。
*   **内循环 (Inner Loop)**：对于选定的模块$\tau_n$，执行$T$次更新。
    *   每次更新中，采样一个mini-batch数据$\xi_{n,t}$计算随机块梯度$g_{n,t}^{\tau_n}$。
    *   使用Adam优化器更新模块参数$\theta_{n,t}^{\tau_n}$，并更新一阶和二阶动量$m_{n,t}^{\tau_n}, v_{n,t}^{\tau_n}$。
    *   采用AMSGram-type归一化$\tilde{v}_{n,t}^{\tau_n} = \max(v_{n,t}^{\tau_n}, \|\tilde{v}_{n,t-1}^{\tau_n}\|_{\max})$。
*   **优化器状态清除**：在外循环结束后，清除当前模块的优化器状态，以确保持续的内存效率。

**内存分析：**
MISA在长序列微调任务中显著优于LoRA。对于LLaMA3-8B模型，MISA在序列长度增加时，比LoRA具有更优的内存效率。当采样比阈值$\delta$较小时，MISA比Layer-wise方法更具内存效率。

**实验结果：**
MISA在多种LLM（LLaMA3-8B、Qwen2.5-7B、TinyLLaMA、LLaMA2-7B、Mistral-7B）和多项任务（常识推理、数学推理、指令微调、预训练）上表现出色：
*   **微调**：在常识推理和数学推理任务上，MISA在相当的内存约束下，性能优于LoRA、DoRA、LISA和BAdam等基线方法。例如，在LLaMA3-8B的常识推理任务中，MISA ($\delta = 3\%$) 达到86.6%的平均准确率，略低于全参数微调，但显著高于其他PEFT和Layer-wise方法。
*   **指令微调**：在Alpaca GPT-4数据集上微调TinyLLaMA、LLaMA2-7B和Mistral-7B，MISA在MMLU、MMLU-pro和MT-Bench等评估基准上在多数情况下优于所有基线。
*   **收敛性**：MISA相比LISA和BAdam，在训练时间维度上展现出更好的验证损失收敛性（图3）。
*   **预训练**：在C4数据集上预训练LLaMA2 130M和350M模型，MISA在130M模型上超越了GaLore和Adam，在350M模型上优于GaLore并接近Adam的性能，显示出其作为Adam正则化的潜力。

**局限性：**
*   MISA的预训练实验规模相对较小，其在大规模LLMs（如7B、70B或更大模型）预训练中的可扩展性仍待验证。
*   MISA目前仅在文本Transformer架构LLMs上进行了验证，其在多模态模型或非Transformer架构上的适用性需进一步探索。
  
##
Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning

https://arxiv.org/pdf/2512.04359 南京大学 高阳等；2025.12.4

1. 🧩 针对大型语言模型 (LLM) 推理中强化学习 (RL) 存在的**熵坍塌问题**，本文提出了一种名为 SENT 的高效 RL 框架，该框架在语义和 Token 级别上利用熵信号来提升推理能力。
2. 📚 SENT 在**数据层面通过语义熵引导课程学习，逐步提供难度递增的任务**；在**算法层面，则对低熵 Token 施加 KL 正则化，并对其中高协方差部分施加更强约束以鼓励探索。**
3. 🚀 实验结果表明，SENT 在 6 个基准测试和 3 种不同参数规模的基础模型上均优于其他基于熵的方法，有效缓解了熵坍塌，并显著增强了 LLM 的推理性能。

该论文提出了一种名为 SENT（Semantic ENtropy with Token-level entropy optimization）的高效强化学习（RL）框架，旨在解决大型语言模型（LLM）推理能力提升过程中常见的熵崩溃（entropy collapse）问题。传统的RLVR（Reinforcement Learning with Verifiable Rewards）方法通常以准确性为导向，但这会导致策略探索不足和局部最优，表现为训练过程中策略熵（policy entropy）的急剧下降，进而限制了模型生成响应的多样性和推理能力。基于verl
<img width="1181" height="605" alt="image" src="https://github.com/user-attachments/assets/291a99d5-6a2a-4451-b8fd-31fc31af9bae" />
<img width="1138" height="589" alt="image" src="https://github.com/user-attachments/assets/573c775c-937a-4d57-ae4e-934e5a3cdc55" />

SENT 框架通过在数据组织和算法优化两个层面同时利用熵信号来解决这一挑战。

**1. 数据层面：语义熵引导的课程学习 (Semantic Entropy-Guided Curriculum Learning)**
为了避免训练样本难度突变导致的探索不稳定，SENT引入了基于语义熵（Semantic Entropy, SE）的课程学习。语义熵量化了针对给定问题语义上不同解决方案的多样性，而非单纯的 token 预测不确定性。
- **语义熵计算：**
    1.  **响应生成 (Response Generation)：** 对于训练数据集 $\mathcal{D}$ 中的每个查询 $q$，从当前策略 $\pi_\theta$ 中采样 $M$ 个响应 $\{o_1, o_2, \ldots, o_M\}$ 及其对应的概率 $\{P(o_1|q), \ldots, P(o_M|q)\}$。
    2.  **语义聚类 (Semantic Clustering)：** 根据语义等效性（例如，在数学推理任务中，最终答案相同则视为语义等效）将生成的响应聚类为 $K$ 个语义等效类 $C = \{C_1, C_2, \ldots, C_K\}$。
    3.  **熵计算 (Entropy Calculation)：** 计算每个语义簇的概率 $P(C_i|q) = \sum_{o \in C_i} P(o|q)$，并对其进行归一化 $\hat{P}(C_i|q) = \frac{P(C_i|q)}{\sum_{j=1}^K P(C_j|q)}$。
    4.  最终的语义熵通过以下公式估计：
        $$H_{SE}(q) = - \sum_{i=1}^K \hat{P}(C_i|q) \log \hat{P}(C_i|q)$$
        高语义熵表明模型生成了多样化的推理路径和解决方案，任务难度较高；低语义熵则表示模型倾向于收敛到相同解决方案，任务相对容易。
- **课程设计 (Curriculum Design)：** 在训练前，使用初始策略（通常是 SFT 模型）计算每个查询的语义熵，然后将整个训练数据集 $\mathcal{D}$ 按语义熵升序排序，得到 $\mathcal{D}_{sorted} = \text{sort}(\mathcal{D}, \text{key} = H_{SE})$。最后，将排序后的数据集划分为 $N$ 个课程阶段，模型逐阶段进行训练。这种设计使得模型能够从简单到复杂逐步适应推理任务，从而维持稳定的探索。

**2. 算法层面：Token 级别熵优化 (Token-Level Entropy Optimization)**
为了防止对低熵 token 的过度优化，SENT 引入了 token-选择性 KL 正则化。
- **识别低熵 Token：** 低熵 token 被定义为策略在预测这些 token 时具有高置信度（即不确定性低）的 token。这些 token 容易导致熵崩溃。
    $$\text{Tlow} = \{o_t | H_t(q, o_{<t}) < \tau_H\}$$
    其中 $H_t(q, o_{<t})$ 是在给定上下文 $q, o_{<t}$ 下 token $o_t$ 的熵，$\tau_H$ 是低熵阈值。
- **计算协方差 (Covariance)：** 在低熵 token 中，那些具有高协方差的 token 对熵动态影响最大。协方差衡量了模型置信度（log 概率）与学习信号（优势 $A_t$）之间的相关性。对于每个低熵 token $o_t \in \text{Tlow}$，其协方差计算如下：
    $$\text{Cov}_{o_t \sim \pi_\theta (\cdot|q,o_{<t})} = \left(\log \pi_\theta (o_t|q, o_{<t}) - \frac{1}{N}\sum_{j=1}^N \log \pi_\theta (o_j |q, o_{<j} )\right) \cdot \left(A_t - \frac{1}{N}\sum_{j=1}^N A_j \right)$$
    其中 $N$ 是 rollout token 的批次大小。
- **识别高协方差低熵 Token：**
    $$\text{Thigh-cov} = \{o_t \in \text{Tlow} | \text{Cov}_t > \tau_{cov}\}$$
    其中 $\tau_{cov}$ 是高协方差阈值。
- **自适应 KL 正则化：** 根据 token 的熵和协方差，应用不同的 KL 正则化系数 $\beta_{con}$：
    $$\beta_{con} = \begin{cases} \beta_{low} & \text{if } o_t \in \text{Tlow} \setminus \text{Thigh-cov}, \\ \beta_{high} & \text{if } o_t \in \text{Thigh-cov} \subseteq \text{Tlow}, \\ 0 & \text{if } o_t \notin \text{Tlow}. \end{cases}$$
    其中 $\beta_{high} > \beta_{low} > 0$。这意味着对于高协方差的低熵 token 施加最强的约束，以最大限度地保持熵。
- **优化目标：** 结合数据层面的课程学习和 token 级别选择性正则化，SENT 的优化目标为：
    $$J_{SENT}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D}_n, \{o_i\}_{i=1}^G\sim\pi_{old}(\cdot|q)}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(r_t(\theta) \hat{A}_i, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta_{con} D_{KL}(\pi_\theta \parallel \pi_{ref})\right]\right]$$
    其中 $r_t(\theta) = \frac{\pi_\theta (o_t|q, o_{<t})}{\pi_{old} (o_t|q, o_{<t})}$ 是 likelihood ratio，$\hat{A}_i$ 是 Group Relative Policy Optimization (GRPO) 中计算的优势估计，$\mathcal{D}_n$ 是当前课程阶段的数据集，$\pi_{ref}$ 是参考模型。

**3. 理论分析 (Theoretical Analysis)：**
论文通过理论分析阐述了 SENT 如何控制熵动态。策略熵的变化与动作 log-概率及其 logit 变化之间的协方差成反比。
- **Logit 变化 (Logit Change)：** 在 Policy Gradient 算法中，logit 变化满足：
    $$\theta^{k+1}_{s_t,o_t} - \theta^k_{s_t,o_t} = \eta \cdot \pi^k_\theta (o_t|s_t)A_t$$
- **熵变化 (Entropy Change)：**
    $$H(\pi^{k+1}_\theta) - H(\pi^k_\theta) \approx -\eta \mathbb{E}_{s_t}\left[\text{Cov}\left(\log \pi^k_\theta (o_t|s_t), \pi^k_\theta (o_t|s_t)A_t\right)\right]$$
- **带有 KL 正则化的熵变化：** SENT 的 logit 更新包含 KL 正则化项，导致熵变化为：
    $$H(\pi^{k+1}_\theta) - H(\pi^k_\theta) \approx -\eta \mathbb{E}_{s_t}\left[\text{Cov}\left(\log \pi^k_\theta (o_t|s_t), \pi^k_\theta (o_t|s_t)A_t\right)\right] + \eta \mathbb{E}_{s_t}\left[\beta_{con} \text{Cov}\left(\log \pi^k_\theta (o_t|s_t), \nabla_\theta D_{KL}(\pi^k_\theta \parallel \pi_{ref})\right)\right]$$
    第一项是香草 Policy Gradient 导致的熵衰减，第二项是 KL 正则化导致的熵保持效果。SENT 的分层 $\beta_{con}$ 确保只在熵崩溃最可能发生（低熵）和影响最大（高协方差）的 token 上进行干预，从而实现熵的保持，同时最小化对优化过程的干扰。数据层面的课程学习进一步稳定了训练，使得模型在处理更复杂任务时能动态调整熵控制。

**4. 实验结果 (Experimental Results)：**
- SENT 在 1.5B、7B 和 14B 参数规模的基础模型上进行了广泛实验，数据集为 DAPO-MATH-17K。
- **性能比较：** 在 AIME、AMC、MATH500、OlympiadBench 和 Minerva 等 6 个数学推理基准测试中，SENT 在 Pass@K 和 Avg@K 指标上均显著优于现有熵基方法，尤其是在 1.5B 模型上，SENT 在 Pass@K 的平均得分上超过次优方法 3.26 分。在 7B 和 14B 模型上，SENT 也展示了持续的优势，在 MATH500 Pass@16 上甚至达到 100% 的最优表现。
- **响应长度 (Response Length)：** SENT 生成的响应长度显著增加（平均多出 1000 多个 token），表明模型进行了更深入、多步骤的推理探索。
- **熵变化分析：** 实验结果图（Fig. 4, Fig. 11, Fig. 12）显示，GRPO 经历快速熵崩溃，而直接熵最大化（w/ En）导致熵爆炸。SENT 则实现了受控的熵变化，避免了持续的熵衰减，并在训练后期维持了健康的探索水平。这证明了 SENT 成功缓解了熵崩溃。
- **渐进性能 (Asymptotic Performance)：** SENT 在训练初期展现出更快的学习效率，在后期课程阶段（数据复杂度增加）虽有短暂性能下降，但能迅速恢复并最终超越基线方法，表明其在应对逐步复杂数据时的鲁棒性。
- **泛化能力 (Generalization Ability)：** 在 LiveCodeBench 编程任务上的评估表明，SENT 在域外任务上也能保持优越的性能，证明了其所培养的推理能力具有良好的泛化性。
- **消融研究 (Ablation Study)：** 独立的课程学习、低熵 token 约束和高协方差约束均对性能有提升，而三者结合的 SENT 达到最佳效果，验证了各组件的有效性和协同作用。

综上所述，SENT 通过语义熵引导的课程学习和 token 级别自适应 KL 正则化，有效地缓解了 RLVR 中 LLM 推理的熵崩溃问题，同时显著提升了 LLM 在各类数学推理任务上的性能和探索深度。

## SpePV
SpecPV: Improving Self-Speculative Decoding for Long-Context Generation via Partial Verification

https://arxiv.org/pdf/2512.02337 西安交大 2025.12.2
https://github.com/TanZhendong/SpecPV

1. 📝 针对长上下文生成中推测解码的验证瓶颈，该论文提出了SpecPV，一种采**用部分KV状态进行快速验证，并周期性地进行完整验证以消除累积误差的自推测**解码方法。
2. ⚡️ SpecPV通过构建包含sink tokens、检索tokens和局部tokens的部分KV缓存来加速验证过程，该缓存远小于完整KV缓存，在内存受限时尤其能降低PCIe传输开销。
3. 🚀 在LLaMA-3.1-8B-Instruct和Qwen3系列模型上的实验结果表明，SpecPV相比标准自回归解码，可实现高达6倍的解码加速，且仅有轻微的准确性下降。

<img width="881" height="285" alt="image" src="https://github.com/user-attachments/assets/69cec556-0576-46a0-9a2d-b11730e03168" />

<img width="641" height="571" alt="image" src="https://github.com/user-attachments/assets/3b361348-7e4b-4b74-93e8-bc1e84c7e81b" />

本文提出了一种名为SpecPV的自推测解码（self-speculative decoding）方法，旨在通过部分验证（partial verification）加速长上下文（long-context）文本生成，以应对大型语言模型（LLMs）在代码生成、深度推理和长文档理解等任务中对长上下文生成日益增长的需求。

**背景与动机**
随着上下文长度的增长，LLMs的生成效率面临挑战。推测解码（speculative decoding）是一种有效的加速方法，其遵循草稿-验证（draft-verify）范式：一个轻量级草稿模型（draft model）提出候选tokens，然后目标模型（target model）进行验证。然而，研究发现，在长上下文生成中，验证阶段逐渐成为主要的性能瓶颈，其时间占比从短上下文的约60%增长到长上下文的近80%（根据EAGLE-3的测量）。这促使作者思考是否能更积极地利用部分Key-Value (KV) cache进行验证，从而减轻完整验证的开销，同时保持最小的性能损失。鉴于LLMs在处理长上下文时注意力（attention）表现出稀疏性，通过精心设计的检索策略，可以利用部分KV cache实现有效验证。

**核心方法：SpecPV**
SpecPV的核心思想是在自推测解码框架下，主要采用部分KV cache进行快速验证，并周期性地进行完整验证以纠正累积误差，确保生成结果与原始输出一致。

1.  **自推测解码框架 (Self-Speculative Decoding Framework)**
    自推测解码的关键特点是草稿模型重用目标LLM的层特征。在预填充（prefilling）阶段之后，模型预测下一个token并生成一组层特征，这些特征作为自推测草稿的输入。草稿阶段仅涉及草稿模块的计算，无需额外调用目标LLM的前向传播，因为验证阶段已生成了所有必要层特征。

2.  **部分验证 (Partial Verification)**
    为提高解码效率，SpecPV采用部分KV cache。部分KV cache在block级别组织，包含四个组成部分：
    *   **Sink Tokens**: 始终保留初始token的KV blocks，以维持注意力sink机制的性能。
    *   **Retrieval Tokens**: 这是部分KV cache的主体。受Quest启发，关键tokens的集合取决于当前query。由于自然语言的语义连续性，注意力具有内在的局部性，因此部分KV cache无需每步更新即可保持良好性能。为加速检索过程，系统缓存每个block的Key-states summary：
        $$S_i = \begin{pmatrix} K_{max}^i \\ K_{min}^i \end{pmatrix}$$
        其中，$K_{max}^i$ 和 $K_{min}^i$ 分别表示block $i$ 中Key states的元素级最大值和最小值。对于每个block，基于输入query states计算分数 $s_i$。在验证阶段，草稿模型生成一系列候选tokens，产生多个query states。因此，首先计算每个query与每个block的得分 $s_{i,j}$：
        $$s_{i,j} = \max(\mathbf{q}_j (K_{max}^i)^\top, \mathbf{q}_j (K_{min}^i)^\top)$$
        然后通过归约函数（如max、mean或last，其中last指最近验证token的query state）聚合为单个重要性得分：
        $$s_i = f(s_{i,1}, s_{i,2}, \dots, s_{i,M})$$
        根据这些得分选择要保留的KV blocks。
    *   **Local Tokens**: 保留少量最近的tokens，作为一个固定大小的窗口。
    *   **Buffer**: 临时存放已部分验证且等待纠正的tokens。在验证完成后，无效tokens将从buffer中移除。
    这些组件在token顺序上是连续的，并在注意力前向传播期间作为单个连续KV段处理。

3.  **通过完整验证纠正 (Rectified with Full Verification)**
    部分验证引入的误差会逐渐累积，导致生成输出随时间漂移。这源于部分验证tokens的KV states不准确，以及检索到的上下文随生成进程变化。SpecPV通过周期性地应用完整验证来消除这些累积误差：
    *   在进入验证前向传播之前，将已部分验证的tokens添加到新生成的候选tokens之前。
    *   在验证过程中，使用完整KV cache执行注意力计算，并相应地刷新部分KV cache。
    *   在此前向传播中，计算部分验证tokens和候选tokens的KV states。
    *   更新检索tokens和局部窗口tokens。
    *   评估候选tokens后，从部分和完整KV cache中移除无效tokens。
    当序列长度小于部分KV cache预算时，禁用部分验证。当序列增长超过部分KV cache预算时，执行一次完整验证以初始化部分KV cache，然后平滑过渡到部分验证。当部分验证和候选tokens的总数超过设定的最大buffer size时，切换回完整验证并刷新部分KV cache。通过调整buffer size，可以控制触发完整验证的频率。
    此外，在内存受限（如RTX 4090 GPU）的KV cache offloading场景下，由于部分KV cache远小于完整KV cache，仅offload完整KV cache到主机内存，将部分KV cache保留在设备上，从而减少PCIe传输开销，提高推理效率。

**实验评估**

1.  **设置**
    *   **模型**: LLaMA-3.1-8B-Instruct和Qwen3系列（4B, 8B, 14B）。草稿模型采用YARN扩展的EAGLE-3框架。
    *   **任务**: 效率评估采用PG-19上的故事续写任务；性能评估使用LongBench v1和v2上的文档摘要和问答任务。
    *   **基线**: TriForce (独立草稿模型, 分层推测), TokenSwift (Medusa-style草稿头, 部分KV cache for drafting), EAGLE3-YARN (直接使用YARN适配的EAGLE-3模型)。
    *   **指标**:
        *   效率: 总加速比（$\alpha = \text{Throughput}_{\text{SpecPV}} / \text{Throughput}_{\text{AR}}$），草稿接受长度（$\tau$）。
        *   质量: 问答任务使用Exact-Match Accuracy；摘要任务使用ROUGE-L和BLEURT。
    *   **硬件**: A100 80GB GPU，以及RTX 4090 24GB GPU（KV cache offloading）。

2.  **生成效率**
    在长上下文场景下，SpecPV相比EAGLE3-YARN、TriForce和TokenSwift取得了显著的加速。例如，在60K上下文长度下，SpecPV-4K在LLaMA3.1-8B-Instruct上实现了6.15x的加速比，而EAGLE3-YARN为3.48x。这表明部分验证在长上下文时效率提升更加显著。不同部分KV预算下，SpecPV均能保持较高的草稿接受长度（约3.0-3.6）。在KV cache offloading场景下，SpecPV由于减少了完整cache访问和PCIe传输，展现出更显著的加速。

3.  **生成质量**
    SpecPV在摘要和问答任务上保持了与完整验证高度一致的生成质量。在摘要任务中，SpecPV与完整生成结果的相似度（ROUGE-L和BLEURT）仅有轻微下降，且差异与完整生成和朴素自回归解码之间的固有数值差异相当。在问答任务中，当KV cache预算为4096 tokens时，SpecPV的性能与完整验证相当，甚至在某些数据集上略有超越，作者推测这可能是因为丢弃部分KV cache有时能减少长上下文中的噪声。

4.  **消融研究 (Ablation Study)**
    *   **检索分数聚合策略**: 比较了Mean、Max和Last三种归约函数，发现Mean归约函数在LLaMA3.1和Qwen3上均能产生最高的相似度和略长的草稿接受长度，因此在其他实验中均采用Mean归约。
    *   **刷新间隔（Buffer Size）**: 随着刷新间隔（即buffer size）的增加，SpecPV与完整验证结果的相似度逐渐降低，表明周期性部分KV cache更新有助于保持模型性能。为平衡效率与准确性，buffer size通常设定为单次验证步骤中处理的token数量加上20。

**结论**
SpecPV通过引入部分KV cache进行快速验证，并辅以周期性完整验证以消除累积误差，有效地加速了长上下文场景下的推测解码。实验结果表明，SpecPV在不显著牺牲准确性的前提下，相比标准自回归解码实现了高达6倍的解码加速，相比完整验证在60K上下文长度下也实现了约2倍的加速。该工作为更高效、实用的长上下文LLM应用提供了有益探索。

## DeepMind Evo-Memory 
Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory

https://arxiv.org/abs/2511.20857 DeepMind等 2025.11.25
中文解读：https://mp.weixin.qq.com/s/TQsCt3cnGUoP4rLfcy07Dw 

1. 💡 Evo-Memory是一个全面的benchmark和框架，旨在评估LLM agents在连续任务流中进行test-time learning和self-evolving memory的能力。
2. 🚀 该研究提出了ExpRAG作为经验重用的baseline，并引入了ReMem，一个action–think–memory refine pipeline，用于持续改进agent的性能。
3. 📈 实验结果表明，ReMem和ExpRAG等self-evolving memory方法显著提升了agent在多轮和单轮任务中的性能、效率和robustness。
Evo-Memory是一项旨在评估大型语言模型（LLM）代理在部署过程中“测试时学习”（test-time learning）和“自进化记忆”（self-evolving memory）能力的综合性基准测试和框架。
上下文已经具备了权重的核心属性：抗噪性、纠错性和泛化性。模**型在没有更新任何参数的情况下，通过在记忆中沉淀经验，实现了行为的永久性矫正。**

以后，我们也许不需要真的去算梯度，不需要更新参数。仅仅通过自然语言层面的反思和沉淀，就能让模型表现出好像被训练过一样的行为矫正。
<img width="991" height="486" alt="image" src="https://github.com/user-attachments/assets/b7307b8b-8390-494f-a1a0-8492afec89cc" />

**1. 研究动机与核心问题**
LLM代理要实现长期规划和问题解决，状态维持能力至关重要，这使得记忆成为其关键组成部分。然而，现有评估大多集中于静态对话设置中的记忆，即从对话中被动检索信息以回答查询，忽视了代理在连续任务流中积累和重用经验的动态能力。在诸如交互式问题助手或具身代理（embodied agents）等真实世界环境中，LLMs需要处理连续的任务流，但往往无法从积累的交互中学习，从而丢失宝贵的上下文洞察。这种限制促使研究者关注测试时进化，即LLMs在部署期间持续检索、整合和更新记忆的能力。该论文旨在弥合这一差距，通过引入Evo-Memory，一个用于评估LLM代理自进化记忆的综合性流式基准测试和框架。

**2. Evo-Memory框架**
Evo-Memory将数据集重构为顺序任务流，要求LLMs在每次交互后搜索、适应和进化记忆。它涵盖多轮目标导向环境以及单轮推理和问答任务，明确测试LLMs在部署过程中积累知识和优化策略的能力，这一过程被称为测试时进化。该框架统一并实现了十余种代表性记忆模块，包括基于检索、工作流和分层记忆系统，以研究它们的适应行为。

**2.1 问题形式化**
一个通用的记忆增强代理被形式化为一个四元组 $(F, U, R, C)$，其中：
- $F$ 是基础LLM。
- $U$ 是记忆更新流水线。
- $R$ 是检索模块。
- $C$ 是上下文构建机制，将检索到的内容转换为最终的工作上下文。

在时间步 $t$，代理接收输入 $x_t$，维护一个不断进化的记忆状态 $M_t$，并执行以下迭代过程：
1.  **Search (检索)**：代理首先从 $M_t$ 中检索相关记忆条目：
    $R_t = R(M_t, x_t)$
    这里的 $R$ 可以是相似性搜索、基于索引的查找或对存储嵌入的注意力机制。
2.  **Synthesis (合成)**：代理将检索到的信息 $R_t$ 重构为与当前输入 $x_t$ 相符的简洁工作上下文 $\tilde{C}_t$。这可能涉及构建结构化提示、选择关键记忆项或合并检索内容。
    $\hat{y}_t = F(\tilde{C}_t)$
3.  **Evolve (进化)**：在获得输出 $\hat{y}_t$ 后，代理构建一个新的记忆条目 $m_t = h(x_t, \hat{y}_t, f_t)$，其中 $f_t$ 是反馈（例如任务是否完成）。记忆状态 $M_t$ 随后通过更新函数 $U$ 进行更新：
    $M_{t+1} = U(M_t, m_t)$
    不同的算法会以不同的方式实例化 $U$，例如直接追加、总结或压缩，或替换。

**2.2 ExpRAG (Experience Retrieval and Aggregation)**
作为一种简单且有效的基线，ExpRAG是一个任务级别的检索增强代理。每个记忆条目 $m_i = S(x_i, \hat{y}_i, f_i)$ 编码了一个带有模板 $S$ 的结构化经验文本。在时间步 $t$，代理根据检索分数 $\phi$ 从记忆中检索 $k$ 个相似经验：
$R_t = \text{Top-k}_{m_i \in M_t} \phi(x_t, m_i)$
模型根据这些检索到的示例进行上下文学习：
$\hat{y}_t = F(x_t, R_t)$
并将新的经验追加到记忆中：
$M_{t+1} = M_t \cup \{(x_t, \hat{y}_t, f_t)\}$
ExpRAG通过检索和聚合实现一次性经验重用，但缺乏迭代推理或推理过程中的自适应优化。

**2.3 ReMem (Synergizing Reasoning, Acting, and Memory)**
ReMem是一个统一推理、行动和记忆优化的框架。与将记忆视为静态上下文的传统检索增强或ReAct风格方法不同，ReMem引入了记忆推理的第三个维度，允许代理在问题解决过程中主动评估、重组和进化自己的记忆。
在每个时间步 $t$，给定当前输入 $x_t$、记忆状态 $M_t$ 和此步骤之前的推理痕迹 $o_{1:n-1}^t$，代理选择三种操作之一：
$a_n^t \in \{\text{Think, Act, Refine}\}$
然后执行该操作并进行状态转换：
$o_n^t = \text{Agent}(x_t, M_t, a_n^t)$
其中 $o_n^t$ 是在 $n$ 次操作后在时间步 $t$ 生成的输出，例如中间推理痕迹、外部行动或记忆优化思考。
- **Think**：产生内部推理痕迹，帮助分解任务并指导后续行动。
- **Act**：执行环境中的操作或向用户输出最终响应。
- **Refine**：对记忆进行元推理，利用有用经验、清除噪声并重组 $M_t$，以更好地支持未来的推理和行动。
在一个步骤中，代理可以执行多轮“Think”和“Refine”操作，并在选择“Act”操作后终止该步骤。这形成了一个马尔可夫决策过程，其中状态 $s_n^t = (x_t, M_t, o_{1:n-1}^t)$，行动空间是 $\{\text{Think, Act, Refine}\}$，转换动力学由Agent算子和环境响应给出。通过这种扩展，记忆成为一个与实时推理交互的自适应组件，而非被动上下文。

**3. 实验设置与评估**
Evo-Memory在多种数据集上进行评估，涵盖事实知识、推理、数学、编程和目标导向交互：
- **单轮数据集**：MMLU-Pro、GPQA-Diamond、AIME-24/25、ToolBench。
- **多轮/目标导向数据集**：Alf World、BabyAI、ScienceWorld、Jericho、PDDL任务。
评估指标包括：
- **准确率 (Answer accuracy)**：衡量单轮任务中模型的正确输出。
- **成功率 (Success rate)**：衡量多轮任务中目标完成情况。
- **步骤效率 (Step efficiency)**：达到目标所需的步骤数。
- **序列鲁棒性 (Sequence robustness)**：在不同任务顺序下性能的稳定性。

**4. 主要发现**
实验结果表明，自进化记忆架构能够持续提升LLM代理的性能。
- **单轮任务**：自进化记忆方法（特别是ReMem）显示出一致的改进，但在单轮设置中提升幅度相对温和。
- **多轮任务**：ReMem和ExpRAG在多轮推理环境中表现出强大且稳定的性能，持续的记忆反思和优化显著改善了过程性知识的积累。性能提升在多轮设置中尤为显著，这强调了持续适应在任务范围延长时变得越来越有价值。
- **任务相似性**：ReMem的性能提升与数据集内任务相似性（由嵌入聚类比率衡量）呈强相关性，表明重复的任务结构有助于记忆重用和泛化。
- **步骤效率**：进化记忆方法始终需要更少的步骤来完成任务，ReMem实现了最强且最稳定的减少，展示了更高效的任务执行。
- **任务难度序列**：在“简单到困难”和“困难到简单”的任务难度序列变化下，自进化记忆代理（特别是ReMem）保持了强劲且一致的性能，表明其能够保留可转移知识，即使任务复杂性发生变化，也增强了鲁棒性。
- **反馈类型**：当记忆中包含成功和失败的任务经验时，基线方法性能会下降，而进化记忆方法（尤其是ReMem）通过主动优化存储经验保持鲁棒性，强调了从成功中学习并适当地利用失败信息的重要性。
- **时间步累积性能**：ReMem在长期任务序列中表现出更快的适应和更稳定的留存，证明了其在测试时学习中的鲁棒性。

**5. 总结**
Evo-Memory通过将静态数据集转换为流式轨迹，系统地评估了LLMs如何通过交互检索、适应和优化记忆，填补了现有研究的空白。研究结果表明记忆可以显著提升性能，但也揭示了其在稳定性和程序性重用方面的脆弱性。为促进该领域的发展，论文引入了用于经验检索的ExpRAG和用于交错推理、行动与记忆更新的ReMem。Evo-Memory旨在作为一个统一平台，促进开发具有可靠且持续改进记忆的LLMs。

## Native Top-K sparse
A Preliminary Study on the Promises and Challenges of Native Top-k Sparse Attention

https://arxiv.org/abs/2512.03494 美团 2025.12.3

中文解读：https://mp.weixin.qq.com/s/d0c6w_S7ppox5oPqd20RbQ 
测的偏长序列，没有纯粹的数学/代码？？

1. ✨ 本研究验证了在**解码阶段使用精确的Top-k Decoding**能显著降低计算成本，同时在**长上下文任务上保持甚至超越全注意力**（full attention）的性能。
2. 📚 进一步的实验表明，在监督微调（SFT）阶段引入**原生Top-k Attention训练机制**，可确保**训练与推理一致性，从而显著提升模型在Top-k Decoding下**的表现。
3. 💡 论文探讨了近似Top-k算法的精度对下游任务的影响，并从熵的角度解释了Top-k SFT模型在低熵任务环境下表现更优的理论基础。
   
<img width="598" height="405" alt="image" src="https://github.com/user-attachments/assets/5445a9fa-1b85-4d79-8361-d12d37f9ec8a" />
<img width="941" height="558" alt="image" src="https://github.com/user-attachments/assets/c1aa4f6a-8bc7-458c-9319-0937dbee5740" />

大型语言模型 (LLMs) 在长上下文建模领域日益普及，但其推理计算成本已成为阻碍智能体和多模态应用发展的重要瓶颈。本报告围绕加速长上下文推理的核心问题，对 Top-k Attention 机制在解码和训练阶段的有效性及其理论机制进行了初步研究。

首先，研究验证了精确 Top-k Decoding 的有效性。在解码阶段，仅保留与 Query 相似度最高的关键 Key 作为上下文窗口，这不仅显著降低了计算开销，而且在 HELMET 和 LongBench v2 等下游任务上实现了与 Full Attention 相当甚至超越的性能。Top-k Ratio ($\rho$) 定义为所选关键 tokens ($W$) 占总上下文 tokens ($N$) 的比例：
$\rho = \frac{W}{N} = \frac{|K_{top}|}{|K|}$
其中 $K_{top}$ 是从所有 Key tokens $K$ 中选出的 $W$ 个最显著的 tokens。当 $\rho$ 较低时，计算更稀疏；$\rho = 1$ 则对应 Full Attention。实验结果表明，即使在低 $\rho$ 值下，Top-k Decoding 也能保持高性能。

其次，研究进一步探索了原生 Top-k Attention 训练策略。通过在 Llama-3-8B-ProLong-512k-Base 模型进行 Supervised Fine-Tuning (SFT) 阶段引入 Top-k Attention kernel，获得了 Llama-3-8B-ProLong-Instruct-512K-TopK-SFT 模型。具体而言，该策略通过封装 FLASHATTENTION kernel，预先计算精确 Top-k 索引和分数，并将其传递以更新相应掩码，从而在 1% 的 Top-k 比例下实现变长 Top-k Attention 训练。实验证实，训练和推理阶段 Top-k Attention 操作的一致性有助于进一步释放 Top-k Decoding 的潜力，显著提升模型性能。

再者，考虑到精确 Top-k Attention 的高计算复杂性，研究探讨了近似 Top-k 算法精度对下游任务的影响。为此，引入了 Retrieval Precision ($p$) 的概念：
$p = \frac{|K_{approx} \cap K_{top}|}{|K_{approx}|} = \frac{|K_{approx} \cap K_{top}|}{W}$
其中 $K_{top}$ 是精确 Top-W scores 对应的 tokens 集合，而 $K_{approx}$ 是近似方法实际检索到的 tokens 集合。该指标衡量了检索到的集合与真实值之间的重叠率。研究证实了下游任务性能与近似保真度之间存在正相关关系，即 $p$ 值越高，性能越好。对 DeepSeek-V3.2-Exp 模型中 Lightning Indexer 的精度进行了统计评估，发现其平均精度约为 60%，但结合其庞大的参数规模，仍能提供优越的端到端性能。Lightning Indexer 虽然理论复杂度仍为 $O(N^2)$，但通过减少 attention heads 数量、降低 embedding 维度和 FP8 量化有效降低了计算系数，并通过 Multi-Query Attention (MQA) 模式实现了近似。

最后，报告从信息熵的角度提供了理论解释。实验观察表明，经过 Top-k Attention SFT 训练的模型在下游任务中表现出明显的熵降低现象。这验证了低熵状态更适应 Top-k Decoding 的假设，为稀疏 attention 机制的有效性提供了坚实的理论基础。经过 Top-k SFT 的模型相对于原始模型在不同任务上都显示出 attention 熵的降低。

## TokenFlow
TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling 

https://arxiv.org/abs/2510.02758 上交 陈guihai团队 2025.8

1. 🌍 TokenFlow通过引入前瞻性调度和主动KV缓存管理，显著提升了LLM文本流服务在请求突发情况下的响应速度和有效吞吐量。
2. 💡 其核心机制包括一个基于实时令牌缓冲区状态和消耗率动态调整请求优先级的buffer-aware调度器，以及一个通过write-through策略和I/O重叠来最小化preemption开销的分层KV缓存管理模块。
3. 🚀 在多GPU和多模型实验中，TokenFlow与现有SOTA基线相比，实现了高达82.5%的有效吞吐量提升和80.2%的P99 TTFT降低，同时保持了相当的整体令牌吞吐量。

TokenFlow是一篇专注于LLM文本流式服务（Text Streaming Serving）的论文，旨在解决请求突发（request burst）场景下，LLM服务在响应性（responsiveness，即Time-To-First-Token, TTFT）和生成稳定性（steady generation，即Time-Between-Tokens, TBT）之间难以平衡的问题。现有系统因非抢占式调度和被动式内存管理，导致资源利用率低和请求处理并行度不足。TokenFlow提出了一种新颖的LLM服务系统，通过抢占式请求调度和主动Key-Value (KV) 缓存管理来增强文本流式性能。

**核心问题与动机：**
传统的LLM推理系统通常采用先来先服务（FCFS）调度策略，并优先处理Prefill阶段，以最大化整体吞吐量。然而，对于用户交互式应用，TTFT和流畅的Token交付至关重要。研究发现，用户对超过1.3秒的初始响应延迟难以接受，且当生成速率低于特定阈值（如12 tokens/s）或波动超过30%时，用户体验会下降。现有系统无法在请求突发时同时优化TTFT和TBT，导致请求排队时间过长，而**正在服务的请求则以远超用户阅读速度的速率生成Token，**造成资源浪费和用户体验下降。此外，LLM的KV Cache会随序列长度线性增长，占用大量GPU内存，限制了并发请求数。现有内存管理策略通常是被动触发的，只有当GPU内存达到阈值时才进行卸载，导致显著的I/O开销，并且缺乏与调度器的紧密协调。

**TokenFlow的核心贡献：**
TokenFlow通过协同设计（Co-design）的调度器和KV Cache管理器，解决了上述挑战。其主要创新点在于：
1.  **Buffer-Aware 调度器：** 根据实时Token缓冲区占用率和Token消耗速率动态调整请求优先级，**实现主动抢占**，确保资源利用与用户消费速度匹配。
2.  **层次化KV Cache管理：** 在**后台主动地在GPU和CPU内存之间传输KV Cache**，并重叠I/O与计算，以最小化请求抢占开销。
3.  **QoS度量：** 引入了一个**综合性的服务质量**（Quality of Service, QoS）度量，平衡了Token的有用性（usefulness）、TTFT和回放停顿惩罚（rebuffering penalties），更准确地反映用户体验和系统效率。

**详细方法：**

1.  **QoS 度量：**
    为了更准确地评估文本流式服务的质量，TokenFlow定义了一个综合性的QoS指标，它考虑了Token的实用价值、首次Token的延迟以及回放停顿带来的惩罚。
    对于每个请求 $i$，其生成Token $j$ 的实用性权重 $w_{i,j}$ 定义为：
    $$
    w_{i,j} = \begin{cases}
        1, & \text{if } B_{i,j} \le \tau \\
        \max(1 - \alpha \cdot (B_{i,j} - \tau), 0), & \text{if } B_{i,j} > \tau
    \end{cases}
    $$
    其中，$B_{i,j}$ 是Token $j$ 生成时请求 $i$ 输出缓冲区的Token数量，$\tau$ 是一个阈值，超过该阈值Token的实用性开始衰减，$\alpha > 0$ 是可调的衰减因子。
    最终的QoS指标定义为：
    $$
    \text{QoS} = \frac{1}{T} \sum_{i=1}^{N} \left( \sum_{j=1}^{L_i} w_{i,j} - \lambda \cdot t_{\text{ttft}_i} - \mu \cdot \text{Rebuffer}_i \right)
    $$
    其中，$T$ 是总处理时间，$N$ 是请求总数，$L_i$ 是请求 $i$ 生成的Token数，$t_{\text{ttft}_i}$ 是请求 $i$ 的TTFT，$\text{Rebuffer}_i$ 是请求 $i$ 经历的缓冲区空闲时间总和，$\lambda$ 和 $\mu$ 是惩罚系数。

2.  **调度问题形式化与目标：**
    TokenFlow将LLM请求调度形式化为一个在线组合优化问题，其目标是选择一个请求子集在每个时间间隔 $t$ 内执行，以最大化预期生成的有效Token数量，同时惩罚可能导致缓冲区欠载（underflow）和回放停顿的调度决策。
    每个请求 $i$ 的效用函数 $U_i$ 定义为：
    $$
    U_i = v_i \cdot t - \gamma \cdot \phi(b_{\text{rem}_i})
    $$
    其中，$t$ 是当前调度步中分配给该请求的执行时间，$b_{\text{rem}_i}$ 是其输出缓冲区中未读Token的数量，$v_i$ 是估计的Token价值，$\phi(\cdot)$ 是一个惩罚函数，当缓冲区过低时该函数值增加（增加优先级），$\gamma$ 是调节惩罚强度的系数。
    考虑到硬件约束，包括GPU内存限制和Batch Size对计算的权衡，优化问题可表示为：
    $$
    \max_{x_i \in \mathcal{A}} \sum x_i \cdot \left( v_i (t - t_{\text{overhead}_i}) - \gamma \phi(b_{\text{pred}_i}) \right) \\
    \text{s.t.} \quad x_i \in \{0, 1\} \quad \forall i \in \mathcal{A} \\
    \sum x_i \le B \\
    \sum x_i l_i \le M
    $$
    其中，$\mathcal{A}$ 是所有活跃请求的集合，$x_i$ 是二进制变量（1表示调度，0表示不调度），$B$ 是最大并发运行请求数，$l_i$ 是请求 $i$ 的上下文长度，$M$ 是总可用GPU内存。
    该公式还引入了两个关键改进：
    *   **有效执行时间 $t - t_{\text{overhead}_i}$：** 考虑了内存操作引起的上下文切换延迟。$t_{\text{overhead}_i}$ 估计为 $\min(t_{\text{IO}}, t_{\text{recompute}})$，其中 $t_{\text{IO}}$ 通过内存管理器的I/O吞吐量历史数据获得，$t_{\text{recompute}}$ 通过滑动窗口平均的Per-Token延迟估计。
    *   **预测缓冲区状态 $b_{\text{pred}_i}$：** 预测在系统开销下Token累积情况。

3.  **Buffer-Aware 请求调度器：**
    TokenFlow的调度器采用两阶段启发式算法来近似解决优化问题：
    *   **确定工作集 (Working Set Determination)：**
        工作集 $W$ 是系统当前活跃处理的请求集合。静态上限 $W_{\text{static}} = \lfloor M / \beta \rfloor$，其中 $\beta$ 是每个请求估计的内存占用。运行时，工作集大小 $W$ 会根据当前运行请求数 $N_{\text{running}}$ 动态调整：
        $$
        W_{\text{scheduled}} = W_{\text{static}} - \lambda \cdot (W_{\text{static}} - N_{\text{running}})
        $$
        新的请求被允许进入工作集需要满足两个条件：工作集有可用容量（$W_{\text{current}} < W_{\text{scheduled}}$），且现有请求的缓冲区大小满足 $b_{\text{rem}_i} \ge \mu \cdot r_i \cdot (\tau_{\text{evict}} + \tau_{\text{load}} + \tau_{\text{schedule}})$，其中 $\mu \ge 1$ 是安全系数，$r_i$ 是请求 $i$ 所需的输出速率。
    *   **缓冲区平衡 (Buffer Balancing)：**
        在工作集内部，调度器根据效用函数 $U_i$ 动态分配优先级，优先处理那些缓冲区较小（接近耗尽）的请求，以防止停顿。效用函数中 $\phi(b_{\text{rem}_i}) = e^{-b_{\text{rem}_i}}$，确保缓冲区接近清空时优先级最高。调度器采用贪婪算法结合局部搜索来选择最佳的执行请求子集。
        *   **重计算与加载的平衡：** 调度器会动态决定是从CPU内存重新加载卸载的请求还是重新计算它们。这取决于 $t_{\text{IO}}$（卸载/加载队列等待时间 + 传输时间）和 $t_{\text{recompute}}$（Prefill时间）的比较。如果重新计算更快，则选择重新计算，反之则加载。

4.  **层次化KV Cache管理器：**
    为了支持调度器的主动抢占和频繁的请求切换，KV Cache管理器需要比现有系统更高效和主动。
    *   **Write-Through 策略：** 将GPU内存视为更大CPU内存的高速缓存。KV Cache一旦生成，就立即同步写入CPU内存，而不是等到抢占时才写入（Write-back）。这消除了抢占模式预测的需求，最大化了PCIe写入带宽利用率，并支持增量更新。
    *   **同步分块写入 (Synchronous Chunked Writing)：** 每一轮推理生成的KV Cache都会被缓冲。在下一次迭代开始前，系统会估计执行时间，并从写入缓冲区拉取适当大小的数据块，启动与估计计算时间匹配的写入操作。这确保了写入操作在后续计算间隔内完成，避免了调度器因I/O等待而停顿，并最大化了PCIe带宽。
    *   **加载-驱逐重叠 (Load-Evict Overlap)：** 在抢占请求并恢复其他请求时，TokenFlow允许并行执行KV Cache的驱逐（eviction）和加载（loading）操作。当一个请求的KV Cache被驱逐时，已同步的部分可以立即释放。同时，其他请求的KV Cache可以分块加载，与驱逐过程重叠。这减少了传输延迟和内存碎片。

**实施：**
TokenFlow在SGLang框架之上实现，替换了其默认调度器，并增加了请求跟踪和管理模块。KV Cache管理器利用并行的CUDA流和Python多线程实现计算和内存操作的完全重叠。

**评估：**
*   **实验设置：** 在NVIDIA RTX 4090、A6000和H200等多种GPU平台以及Llama3-8B、Qwen2-7B和Qwen2.5-32B等模型上进行。使用ShareGPT、BurstGPT和真实生产 traces 数据集，以及受控的合成请求分布（突发和泊松分布）。
*   **评估指标：** TTFT、吞吐量（Throughput）和有效吞吐量（Effective Throughput）。有效吞吐量根据用户消费模式对Token进行加权：当缓冲区低于总输出长度的10%时，Token完全计算；在10%到20%之间线性衰减；超过20%的Token不计入有效吞吐量。
*   **基线：** SGLang（传统）、SGLang (chunked) 和 Andes (QoE-aware)。
*   **主要结果：** 在真实世界 traces 中，TokenFlow平均降低了52.6%的TTFT（P50最高达88.7%），同时在A6000上有效吞吐量提高了45.1%，H200上提高了37.1%。在受控突发工作负载下，P99 TTFT降低高达80.2%，平均TTFT降低48.4%，有效吞吐量提高52.9%。在泊松分布工作负载下，有效吞吐量最高提高82.5%（RTX 4090），TTFT降低53.7%（H200）。TokenFlow在保持与SGLang相似的原始吞吐量的前提下，大幅提升了有效吞吐量和降低了TTFT。
*   **微实验：**
    *   **Token生成时间线：** TokenFlow能更早地开始服务，并以所需速率精准交付Token，克服了SGLang的队头阻塞问题。
    *   **抢占式调度可视化：** 展示了当请求缓冲区达到阈值时，系统如何重新分配资源，暂停部分请求（表现为时间线上的平稳段），待其缓冲区接近耗尽时再恢复。
    *   **多速率请求调度：** 成功支持不同目标生成速率的请求，高速率请求因缓冲区消耗更快而获得更高的调度优先级。
    *   **异构硬件支持：** 在华为Ascend 910B上同样表现良好。
*   **超参数敏感性：** 重调度间隔（reschedule interval）和缓冲区保守度（buffer conservativeness）对性能有影响。较短的间隔略微提升有效吞吐量和TTFT，但会增加调度开销。缓冲区保守度可调节响应性和稳定性之间的权衡。
*   **开销和消融研究：** 调度算法和新设计的请求管理器的运行时开销极低（从SGLang的0.07ms增加到0.4ms）。消融研究显示，写直通（Write-Through）和分层卸载设计是性能提升的主要贡献者。

**讨论：**
TokenFlow与Andes的主要区别在于，Andes侧重于用户体验质量（QoE），而TokenFlow则引入了一个更全面的QoS指标，同时考虑了延迟、吞吐量和用户体验。TokenFlow通过调度器与内存管理器的双向交互（调度器传递抢占需求，内存管理器提供反馈），实现了更有效的资源分配。对于多节点和分布式系统，TokenFlow的调度和KV管理可以扩展，例如引入节点间缓存层。对于不同类型的客户端，TokenFlow可以通过指定目标输出速率，或在未来工作通过系统负载推断有效速率来支持异构场景。

**总结：**
TokenFlow通过其创新的Buffer-Aware调度器和层次化KV Cache管理器，显著提升了LLM文本流式服务的性能。它能够动态匹配Token生成速率与用户消费模式，并主动管理GPU内存，从而在请求突发场景下实现更高的有效吞吐量和更低的TTFT。这一研究为实时LLM应用提供了一个鲁棒且高效的解决方案。

## DualSparse-MoE
DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction 
https://arxiv.org/abs/2508.18376 2025.8 港科广

1. 💡 论文指出Mixture of Experts (MoE) 模型存在tensor-level和neuron-level双重稀疏性，并引入post-training expert partitioning方法来在不重新训练的情况下增强tensor-level稀疏性，从而提升精度和效率。
2. 🚀 基于此，作者提出了DualSparse-MoE推理系统，该系统通过动态的tensor-level计算dropping和静态的neuron-level reconstruction来协同利用双重稀疏性。
3. 📊 实验证明，DualSparse-MoE在实现显著计算加速的同时能将精度损失降至最低（例如，MoE module speedup可达1.41倍，平均精度仅下降0.5%），并在Mixtral、OLMoE和DeepSeek模型上均展现出优异性能。

该论文提出了 DualSparse-MoE，一个用于高效部署大规模语言模型（LLMs）中 Mixture of Experts (MoE) 架构的推理系统。MoE 通过**引入TP稀疏性来降低每 token 的计算量**，但其巨大的计算规模和不可预测的激活模式仍带来挑战。论文的核心思想是**利用预训练 MoE 模型中固有的双重稀疏性**：张量级稀疏性（不同专家间的激活选择）和神经元级稀疏性（单个专家内部的神经元激活模式），以**在保证精度最低损失的前提下提升效率。**

**核心方法论：**

1.  **Expert Partition（专家分区）**
    为了在后训练阶段**诱导更细粒度的张量级稀疏性**，论文提出了两种专家分区方法：

    *   **Complete Transformation（完全变换）**
        目标：通过更细粒度的专家结构来提升模型微调时的精度。
        机制：将预训练 MoE 模型中的每个**专家划分为 $P$ 个更细粒度的专家。**
        步骤：
        1.  **重复 gating network 权重：** 原始的 gating network 权重 $\mathbf{W}_g \in \mathbb{R}^{d_{model} \times E}$ 被复制 $P$ 次，形成新的 $\mathbf{W}_g^P \in \mathbb{R}^{d_{model} \times (E \times P)}$。这意味着对于每个原始专家 $e$，其对应的 $P$ 个新专家共享相同的 gating logit $l_e$。
        2.  **分区专家神经元：** 将原始专家的神经元均匀地分配给这 $P$ 个新专家。例如，如果原始专家 FFN 的中间维度为 $d_{ffn}$，则每个新专家 FFN 的中间维度变为 $d_{ffn}/P$。
        3.  **缩放 down-projection 权重：** 每个分区后的专家 FFN 的 down-projection 权重 $\mathbf{W}_2$ 被乘以 $P$。
        数学一致性证明：
        原始 MoE 输出 $\mathbf{y}_i = \sum_{e=1}^E s_e \cdot f_e(\mathbf{x}_i)$，其中 $s_e = \frac{\exp(l_e)}{\sum_{j=1}^E \exp(l_j)}$。
        在完全变换后，每个新专家 $(e,p)$ 的 gating score 为 $s_{e,p} = \frac{\exp(l_{e,p})}{\sum_{j=1}^E \sum_{k=1}^P \exp(l_{j,k})} = \frac{1}{P} \frac{\exp(l_e)}{\sum_{j=1}^E \exp(l_j)}$，因为 $l_{e,p} = l_e$ 且分母中共有 $E \times P$ 项。
        每个原始专家 $f_e(\mathbf{x}_i)$ 的输出是其 $P$ 个分区专家 $f_{e,p}(\mathbf{x}_i)$ 输出之和：$f_e(\mathbf{x}_i) = \sum_{p=1}^P f_{e,p}(\mathbf{x}_i)$。
        因此，分区后的 MoE 输出 $\mathbf{y}_i^P = \sum_{e=1}^E \sum_{p=1}^P s_{e,p} \cdot f_{e,p}(\mathbf{x}_i) = \sum_{e=1}^E \left(\frac{1}{P} s_e\right) \left(\sum_{p=1}^P f_{e,p}(\mathbf{x}_i)\right) = \sum_{e=1}^E \frac{1}{P} s_e \cdot f_e(\mathbf{x}_i) = \frac{1}{P} \mathbf{y}_i$。
        为了使 $\mathbf{y}_i^P = \mathbf{y}_i$，需要将 down-projection 权重 $\mathbf{W}_2$ 乘以 $P$。

    *   **Partial Transformation（部分变换）**
        目标：主要用于提高系统效率，例如 Soft Expert-Tensor Parallelism (S-ETP) 和 DualSparse-MoE。
        机制：在不修改 gating network 的情况下对专家进行分区。
        步骤：
        1.  **重复 gating scores：** 将 Top-K 选定专家的 gating scores 复制 $P$ 次。
        2.  **重映射专家索引：** 原始专家索引被重新映射，使得每个原始专家对应的 $P$ 个分区专家按顺序排列。
        3.  **分区专家神经元：** 与 Complete Transformation 类似，将原始专家的神经元均匀地分配给 $P$ 个新专家，但不缩放 $\mathbf{W}_2$。
        数学一致性：由于 gating scores 被重复且 $\mathbf{W}_2$ 不缩放，模型输出 $\mathbf{y}_i$ 保持不变。

    *   **Soft Expert-Tensor Parallelism (S-ETP)**
        S-ETP 利用 Partial Transformation **实现 tensor-level 分区，以优化 MoE 模型在分布式部署中的通信模式。**它通过 **AlltoAll 操作取代了 ETP 中 AlltoAll+AllGather **和 **ReduceScatter+AlltoAll 的**复杂模式，减少了核函数启动和同步开销，提高了通信效率。

2.  **DualSparse-MoE 推理系统**
    DualSparse-MoE 旨在通过协调张量级和神经元级稀疏性，在不重新训练的情况下提升推理效率并最小化精度损失。它由以下策略组成：

    *   **Static Expert Partition and Reconstruction（静态专家分区与重构）**
        *   **专家分区：** 采用 Partial Transformation 策略将专家分为更细粒度的子专家，增强张量级稀疏性。
        *   **神经元重要性分析：** 通过校准样本（calibration samples）对每个专家内的神经元进行重要性分析。论文测试了四种 profiling 方法：累积 gate value、累积绝对 gate value、累积 gate-up value、累积绝对 gate-up value。经验表明，基于绝对值的 profiling 方法更有效。
        *   **重构子专家：** 根据重要性分析，将每个专家重构为两个子专家：一个包含高重要性神经元的 "major sub-expert"，和一个包含低重要性神经元的 "minor sub-expert"。这种静态重构避免了运行时动态识别神经元激活的复杂性。

    *   **Dynamic Token-Expert Computation Dropping（动态 token-expert 计算丢弃）**
        *   **1T-Drop (Single-Threshold Drop)：** 基于归一化 gating score，如果低于某个阈值 $T^{1}_{drop}$，则丢弃该 token-expert 的计算。论文发现，适当低的阈值甚至能略微提高精度。
        *   **2T-Drop (Dual-Threshold Drop)：** 结合 Major 和 Minor 子专家，引入双阈值机制：
            *   **Minor Threshold ($T^{2}_{minor}$):** 对 Minor sub-expert 应用较高的阈值。如果 token-expert 的 gating score 低于 $T^{2}_{minor}$，则其 Minor sub-expert 的计算被丢弃。
            *   **Major Threshold ($T^{2}_{major}$):** 对 Major sub-expert 应用较低的阈值。如果 token-expert 的 gating score 低于 $T^{2}_{major}$，则该 token-expert 的所有计算（包括 Major 和 Minor）都被完全丢弃。
            *   对于 gating scores 介于 $T^{2}_{major}$ 和 $T^{2}_{minor}$ 之间的 token-expert，只计算 Major sub-expert 的部分。
            *   这种机制通过对不同重要性的神经元应用不同的丢弃策略，更精细地控制计算量，从而在保持计算量削减的同时，显著降低精度损失。论文中，经验设置为 $T^{2}_{major} = T^{1}_{drop} - 0.01$ 和 $T^{2}_{minor} = T^{1}_{drop} + 0.01$。
        *   **优化 Triton kernel：** 为了将计算丢弃转化为实际的性能提升，对 token-expert 分组 GEMM (grouped-GEMM) 的 Triton kernel 进行了优化。

    *   **Load-Aware Thresholding（负载感知阈值）**
        目标：解决专家并行 (EP) 中分布式设备间的负载不均衡问题，在相同加速效果下最大程度保留精度。
        机制：动态调整每个设备的丢弃阈值。
        *   高负载设备应用更高的丢弃阈值，丢弃更多计算。
        *   低负载设备使用更低的丢弃阈值，减少不必要的计算丢弃。
        *   通过计算实际负载与理想平衡负载的比率来调整阈值。如果比率大于1，阈值设定为预定义的最大值；如果小于1，阈值按比例降低。
        效果：确保所有设备尽可能少地丢弃计算，同时将负载控制在原始最高负载设备水平之下，从而在EP部署中实现更高的推理精度和相同水平的加速。

**实验结果：**

*   **专家分区对精度和效率的影响：**
    *   **精度提升：** 将 Mixtral-8x7B 模型从 8 专家分区为 32 专家（P=4）后，微调损失显著降低（图4）。微调后，分区模型在下游任务上的平均精度从 70.53% 提升到 71.12% (Table 1)，即使在 23.9% 的计算丢弃率下，其精度仍高于未分区模型的基线。
    *   **效率提升 (S-ETP)：** S-ETP 相较于 ETP 显著提升了通信带宽。在 8xH20 节点上的实际测试中，带宽提升 3.0%~29.9%。在 NVL72 和 CloudMatrix384 等高性能互联模拟环境中，带宽提升 9.9%~80.4%。
*   **DualSparse-MoE 推理系统的表现：**
    *   **精度：** 在约 25% 的计算丢弃率下，DualSparse-MoE (2T-Drop with Reconstruct) 相较于基线的平均精度损失极小：Mixtral 仅降低 0.08%，OLMoE 降低 0.28%，DeepSeek 降低 0.18% (Table 2)。相比之下，1T-Drop 会导致更大的精度下降。部分情况下，丢弃计算甚至略微提高了精度，表明低贡献计算可能存在负面影响。
    *   **效率：** 22%~27% 的 MoE 计算丢弃率能有效转化为 MoE 模块 1.17x~1.23x 的实际加速，以及端到端 1.07x~1.12x 的加速 (Figure 10)。这得益于其张量级丢弃特性，与现有硬件兼容性好。
    *   **负载感知阈值：** 结合负载感知阈值后，DeepSeek 模型在 8xEP 部署中实现了 1.41x MoE 模块加速和 1.13x 端到端加速，而平均精度损失仅为 0.5% (Figure 11)。

**总结：**

DualSparse-MoE 通过引入后训练专家分区来增强张量级稀疏性，并设计了一种结合动态张量级计算丢弃与静态神经元级重构的双阈值策略，实现了在分布式 MoE 模型部署中显著的效率提升和极小的精度损失。负载感知阈值的引入进一步优化了专家并行下的负载均衡，有效提升了整体性能。该方法为 MoE 模型在服务器端推理的高效部署提供了有力的解决方案。

## 华为 Nexus
Nexus: Higher-Order Attention Mechanisms in Transformers

https://arxiv.org/abs/2512.03377 华为 2025.12.4
1. 🌟 本文提出Nexus，一种通过嵌套自注意力循环递归地精炼Query和Key向量的Transformer架构，旨在捕捉单层内复杂的更高阶依赖关系。
2. 💡 Nexus通过跨递归步骤共享权重实现了O(1)的额外参数，并在理论上证明其能够突破标准注意力机制的线性瓶颈。
3. 🚀 经验证，Nexus在多个基准测试中超越了标准Transformer，尤其在多步推理任务上表现出色，并能有效提升现有LLM的推理能力。

本文提出了一种名为 Nexus 的新型 Transformer 架构，旨在通过递归框架增强模型的表征能力，以解决标准 Transformer 中一阶注意力机制存在的“**低秩瓶颈**”（low-rank bottleneck）问题，该问题**限制了其捕获复杂、多跳关系（multi-hop relationships）的能力**。


<img width="1042" height="542" alt="image" src="https://github.com/user-attachments/assets/e10e3153-27ee-4363-b907-b72454b2b567" />

<img width="1042" height="345" alt="image" src="https://github.com/user-attachments/assets/8690fb10-c741-4f13-a280-f9d4e8aff1b1" />

<img width="1042" height="618" alt="image" src="https://github.com/user-attachments/assets/05663721-5009-4cf6-8edd-f814b1d46857" />


**1. 引言与动机**

标准 Transformer 架构依赖自注意力机制（self-attention mechanism）来捕获长距离依赖（long-range dependencies）。然而，现有理论研究表明，自注意力矩阵存在秩塌陷（rank collapse）问题，限制了其对复杂、分层关系（hierarchical relationships）的建模能力，尤其是在需要多**步推理（multi-step reasoning）和符号操作（symbolic manipulation）的任务中**。标准的注意力权重 $A = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) \in \mathbb{R}^{n \times n}$ 仅建模令牌（token）之间的成对（pairwise）交互，这阻碍了模型执行多步推理或捕获更复杂、分层关系的能力。例如，对**于三个令牌** $x_i, x_j, x_k$ 之间的三元交互（triadic interaction），标准自注意力机制需要通过多层堆叠或迭代推理才能推断其组合效应，这不仅增加了计算负担，还可能导致信息损失和梯度消失问题。

**2. 核心方法：高阶注意力机制（Higher-Order Attention Mechanism）**

Nexus 的核心思想是**动态地改进 Query 和 Key 的表示，使其本身成为内部注意力循环（inner attention loops）的输出。这意味着在最终注意力计算之前，令牌可以聚合全局上下文（global context）并建模高阶关联**（high-order correlations）。

**2.1 高阶注意力（H-Attention）**

H-Attention 通过首先对 Query 和 Key 应用自注意力机制来细化它们的表示：
$$ \text{H-Attention}(X) = \text{Attention}(\text{Attention}_q (X), \text{Attention}_k(X), V) \quad (6) $$
其中，
$$ \text{Attention}_q (X) = \text{softmax}\left( \frac{QQ^\top}{\sqrt{d_k}} \right)Q \quad (7) $$
$$ \text{Attention}_k(X) = \text{softmax}\left( \frac{KK^\top}{\sqrt{d_k}} \right)K \quad (8) $$
因此，高阶注意力可以形式化为：
$$ \text{H-Attention}(X) = \text{softmax}\left( \frac{\text{Attention}_q (X) (\text{Attention}_k (X))^\top}{\sqrt{d_k}} \right)V \quad (9) $$
通过这种方式，$\text{Attention}_q (X)$ 和 $\text{Attention}_k (X)$ 封装了来自多个令牌的聚合信息，使得后续的注意力操作能够直接考虑多令牌依赖（multi-token dependencies），从而在一个注意力层内捕获高阶交互。

**2.2 递归高阶注意力（Recursive Higher-Order Attention）**

该机制可以递归地扩展，以进一步增强模型捕获分层和长距离依赖的能力。第 $m$ 阶注意力可以递归定义为：
$$ H_m\text{-Attention}(X) = \text{Attention}(H_{m-1}\text{-Attention}_q (X), H_{m-1}\text{-Attention}_k (X), V) \quad (11) $$
每个递归步骤进一步处理 Query 和 Key 向量，使模型能够捕获多级依赖和更复杂的内部关系结构。

**2.3 参数高效的权重共享（Parameter-Efficient Weight Sharing）**

为了避免高阶机制带来的参数量增加，Nexus 引入了一种权重共享策略。基于语义转换在递归层级上相似的假设，Nexus 强制内部（inner）和外部（outer）注意力层之间共享参数 $\theta = \{W_q, W_k, W_v\}$。这意味着内部注意力机制重用与外部主循环相同的投影权重。
$$ H\text{-Attention}_q (X; \theta) = \text{Attention}(X, X, X; \theta) \cdot W_q \quad (12) $$
这一约束确保了 Nexus 的参数复杂度相对于递归阶数 $m$ 保持 $O(1)$，与标准 Transformer 相同，从而实现了参数高效性。

**2.4 复杂度分析**

引入高阶注意力机制会增加计算复杂度。第 $m$ 阶高阶注意力机制的计算复杂度为 $O(2^m n^2 d_k)$。尽管时间复杂度与递归阶数 $m$ 呈指数关系，但在实践中，较小的递归深度（例如 $m=2$）就足以实现显著的性能提升。因此，实际计算开销相对于标准注意力而言是一个常数因子（例如 $m=2$ 时约为 2 倍）。而参数复杂度通过权重共享仍保持为 $O(1)$。

**2.5 标准注意力机制的线性瓶颈（Linear Bottleneck）**

Bhojanapalli et al. (2020) 指出，当 $d_k < n$ 时，标准注意力机制缺乏表达任意注意力权重 $A$ 的能力，即存在低秩瓶颈。Theorem 3.1 进一步阐述了这一问题：
1.  给定 $N$ 个不同的输入 $X_m \in \mathbb{R}^{n \times d}$ 和对应的行随机矩阵 $A_m \in \mathbb{R}^{n \times n}$，只要 $\text{rank}(\log(A_m)) \le d_k$，就存在映射 $Q, K: \mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times d_k}$ 使得 $\text{softmax}\left( \frac{Q(X_m)K(X_m)^\top}{\sqrt{d_k}} \right) = A_m$。
2.  如果 $d < n-1$，存在满足 $\text{rank}(\log(A_m))=1$ 的 $A_m \in \mathbb{R}^{n \times n}$，但对于所有线性变换 $Q(X) = XW_q, K(X) = XW_k$，上述等式仍然不成立。
这表明标准注意力机制无法充分表示低秩矩阵，即使是秩为 1 的 $\log$-attention-weight 矩阵也无法表示。Nexus 通过引入非线性的 $Q(X) = \text{Attention}_q (X)$ 和 $K(X) = \text{Attention}_k (X)$ 映射来解决这一问题，使其更加灵活。

**3. 实验验证**

**3.1 不同规模模型上的评估**

Nexus 在 Pythia 模型上进行验证，并在 PIQA、Hellaswag、SciQ、ARC-E、ARC-C 和 LogiQA 等六个公共数据集上进行了评估。结果显示，Nexus 在所有模型规模上都优于标准 Transformer baseline，尤其是在需要多步推理或长上下文整合的任务上（如 SciQ 和 PiQA）表现出显著提升。

**3.2 消融研究（Ablation Study）**

在 70M 参数规模模型上进行了消融研究，主要发现：
*   **高阶组件的选择**：对 Query 和 Key 同时应用高阶注意力（Nexus-QK）能带来显著性能提升，而对 Value 应用则无额外收益。这表明核心优势在于细化 Query 和 Key 之间的关联对齐。
*   **参数效率与权重共享**：权重共享策略（Nexus-QK-Shared）在保持参数量与标准 Transformer 相当的同时，性能略有下降但仍显著优于 baseline，实现了高表达能力与参数效率的平衡。
*   **递归阶数的影响**：增加到 3 阶注意力（Nexus-Recursive）进一步提升了性能，验证了高阶递归捕获更复杂依赖的能力。综合效率和准确性，2 阶共享配置（Nexus-QK-Shared）被选为默认设置。

**3.3 注意力模式可视化（Visualization of Attention Patterns）**

可视化结果（图 2）显示：
*   **因果结构（Causal Structure）的保留**：Nexus 的外部注意力（Nexus outer）与标准 Transformer 类似，都保持了因果语言模型（causal language models）的基本结构，如对局部上下文（local context）的强对角线关注和对句首令牌（beginning-of-sentence token）的显著关注，表明 Nexus 在增强表达能力的同时保持了稳定性。
*   **内部注意力（Inner Attentions）的作用**：内部 Key-Attention（Inner K-Attention）显示出独特的垂直条纹，表明它充当了语义高亮器（semantic highlighter），在主注意力机制计算最终分数之前，识别并聚合全局相关信息（如关键词或实体）到 Key 表示中。
*   **上下文感知投影（Contextualized Projections）**：Nexus 网络中的 Q 和 K 向量是上下文感知的表示，通过内部循环从先前令牌聚合而来，实现了“预推理”步骤，简化了最终外部注意力层任务。

**3.4 改造标准 Transformer 以进行推理（Retrofitting Standard Transformers for Reasoning）**

Nexus 还可作为现有 LLM 的“升级套件”。通过将预训练的 Qwen2.5-Base 模型（1.5B 和 7B）在 SFT 阶段改造为 Nexus 架构，并在 MATH-500、AIME24 和 GPQA-Diamond 等推理基准上进行评估，Nexus-SFT 取得了持续的性能提升，尤其是在数学推理能力上。这表明 Nexus 架构可以有效升级现有模型，以较低的适应成本释放其在复杂推理任务中的潜力。

**4. 结论**

Nexus 通过引入递归高阶注意力机制，迭代处理 Query 和 Key 向量，构建了多级注意力框架，从而增强了模型捕获复杂分层关系的能力，解决了标准一阶注意力的低秩瓶颈。理论分析表明其具有更强的表达能力，经验评估则证实了 Nexus 在多个基准测试中优于标准 Transformer。此外，Nexus 还支持现有预训练模型的有效“升级改造”。未来的工作将侧重于优化高阶注意力的计算效率，并探索其在视觉和多模态学习等更广泛领域的适用性。

## Arbitrage 投机加速
Arbitrage: Efficient Reasoning via Advantage-Aware Speculation

https://arxiv.org/abs/2512.05033 伯克利 apple等，2025.12.4

1. 🤔 现有的步级 Speculative Decoding (SD) 方法，例如 RSD，在推理任务中**由于频繁且无意义地重新生成步骤**，导致了大量的计算浪费。
2. 💡 ARBITRAGE 提出了一种**优势感知型步级推测生成框架**，它使用轻量级 ARBITRAGE ROUTER 来**预测目标模型何时能提供显著更好的步骤**，从而最大限度地减少冗余计算。
3. ✨ 在多个数学推理基准测试中，ARBITRAGE 始终优于先前的基线，在匹配精度下可将推理延迟降低高达约 2 倍。
基于草稿模型和目标模型在特定推理步骤上的预期质量差异。它动态地**在草稿模型和目标模型之间进行路由**，仅在预期目标模型**能够提供“有意义的改进”时才调用目标模型**。
基于SGLang A6000


<img width="972" height="697" alt="image" src="https://github.com/user-attachments/assets/bfbf94e5-4458-48dd-bf11-bf6baa8d01d0" />

<img width="1104" height="718" alt="image" src="https://github.com/user-attachments/assets/3212c14a-2235-42d7-901b-797b20b2547f" />

<img width="1104" height="566" alt="image" src="https://github.com/user-attachments/assets/2278e241-164c-4d1a-b6d7-fc5e75145d85" />


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
