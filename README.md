# AwesomePaper-for-AI
Awesome system papers for AI

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
Seer在生产级RL工作负载上进行了评估，结果表明，与最先进的同步RL系统相比，Seer的端到端Rollout吞吐量提高了74%至97%，长尾延迟降低了75%至93%，显著加速了RL训练迭代。消融实验也验证了分段Rollout、上下文感知调度和自适应分组推测解码各自的有效性。例如，上下文感知调度实现了接近Oracle LFS调度器95%的吞吐性能，而自适应分组推测解码相比无SD基线提高了30%的端到端吞吐量。
