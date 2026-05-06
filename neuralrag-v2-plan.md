# NeuralRAG — 可行性方案 v2

## 核心调整（相比原方案）

| 维度 | 原方案 | v2 | 理由 |
|------|--------|-----|------|
| 定位 | Industrial-Grade | **Production-Ready Demo** | 4周做不出工业级，但可以做出面试能深度讲的 |
| 框架 | LangChain + LlamaIndex 双框架 | **LlamaIndex 为主 + 轻量 Agent** | 减少胶水代码，聚焦检索核心 |
| 监控 | Prometheus + Grafana | **结构化日志 + /health + /metrics** | 6个Docker服务太重，面试不会真跑Grafana |
| 接口 | generate/embed/fine_tune 混在一个类 | **三个独立接口** | 符合接口隔离原则，面试不扣分 |
| Chunk | 512 token + 64 overlap (12.5%) | **512 token + 128 overlap (25%)** | 中文长句不会被切断 |
| 周计划 | 无 buffer，每天塞满 | **每周留 1 天 buffer** | 任何一个模块卡住不会整体崩塌 |
| 风险 | 无 | **有风险清单 + 缓解方案** | 面试被问"项目遇到什么困难"时能答 |

---

## 一、项目定位

**一句话：** 一个能跑在服务器上、面试时可以逐层深讲的 RAG-Agent 系统。

**目标用户：** 面试官（不是生产用户）

**核心原则：**
- 每个设计决策都要能回答「为什么」
- 宁可功能少但每层都扎实，不要功能多但每层都有坑
- 代码即文档 — README 就是面试讲解稿

---

## 二、技术选型（精简版）

| 组件 | 选择 | 一句话理由 |
|------|------|-----------|
| 检索框架 | **LlamaIndex** | 检索是核心，LlamaIndex 的数据连接器和检索策略最成熟 |
| Agent 编排 | **手写轻量 ReAct** | 避免引入 LangChain 全套依赖，100 行代码搞定 |
| 向量库 | **Qdrant** (Docker) | Rust 实现性能好，原生支持 hybrid search，单容器部署 |
| Embedding | **BGE-M3** (本地) | 中英混合强，支持 dense+sparse 双路，面试加分 |
| LLM | **Claude API → OpenAI API** 二级降级 | 开发用 Claude（长上下文），降级用 OpenAI（稳定） |
| 重排序 | **BGE-Reranker-v2-m3** | 和 BGE-M3 同系列，减少模型碎片化 |
| API | **FastAPI** | 异步原生，自动文档，面试常考题 |
| 内存 | **Redis** (Docker) | 会话状态 + 对话历史，轻量 |
| 部署 | **Docker Compose** (4 服务) | api + qdrant + redis + nginx |

**砍掉的内容：**
- ~~LangChain~~ — Agent 编排手写，检索用 LlamaIndex
- ~~Prometheus + Grafana~~ — 换成结构化日志 + /metrics 端点
- ~~Ollama Provider (Week 1-3)~~ — Week 4 作为可选加分项
- ~~MiniMind Provider 完整实现~~ — 只留接口桩，不强上 160M 模型做 RAG

---

## 三、接口设计（修正版 — 符合 SOLID）

### 3.1 三个独立接口

```python
# core/base.py

from abc import ABC, abstractmethod
from typing import AsyncIterator, List

class BaseLLMProvider(ABC):
    """LLM 推理接口 — 只管生成"""
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]: ...


class BaseEmbeddingProvider(ABC):
    """Embedding 接口 — 独立于 LLM"""
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]: ...

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]: ...


class BaseReranker(ABC):
    """重排序接口 — 独立抽象"""
    @abstractmethod
    async def rerank(
        self, query: str, documents: List[str], top_k: int
    ) -> List[tuple[int, float]]: ...  # [(index, score), ...]
```

**为什么要拆？** 面试官如果问到 SOLID 原则，你可以直接指着这段代码说「我在这里做了接口隔离，因为 Embedding 模型和 LLM 是不同的服务生命周期，BGE-M3 常驻 GPU 而 LLM 走 API，混在一起违反单一职责」。

### 3.2 ProviderRegistry（保持不变，核心亮点）

```python
# core/registry.py

class ProviderRegistry:
    """配置驱动的 Provider 热切换 — 不重启服务"""
    def __init__(self, config_path: str): ...
    def register(self, name: str, provider: BaseLLMProvider): ...
    def get_active(self) -> BaseLLMProvider: ...
    def switch(self, name: str): ...        # 热切换
    def reload(self): ...                   # 重载配置文件
```

面试亮点：修改 yaml 配置 → 调用 `/admin/switch` 端点 → Provider 实时切换，无需重启。

---

## 四、数据流设计（保持，微调参数）

### 4.1 摄入流程

```
文档(PDF/MD/TXT) → Loader → 原始文本
    → SemanticChunker (512 token, 128 overlap, 25%)
    → MetadataEnhancer (自动标题/来源/时间戳)
    → BGE-M3 Embedding (1024d dense + sparse)
    → Qdrant.upsert(vectors + payload + sparse_vectors)
```

### 4.2 查询流程

```
用户查询
    → QueryExpander (可选 — 配置开关，不是默认)
        生成 2 个变体（并行调用，不是串行）
    → HybridSearch
        BM25 (weight: 0.3) + Vector (weight: 0.7)
        → RRF 融合 → Top-50
    → BGE-Reranker (50 → 10)
    → ContextBuilder (拼接 + 来源标注)
    → Claude.generate_stream()
    → 流式输出 + 来源引用
```

**改动点：**
- overlap 64→128（25%，中文友好）
- QueryExpander 改为可配置开关，并行调用
- BM25 weight 标注为「默认值，需在评估集上验证」

---

## 五、Agent 编排（轻量手写，不引入 LangChain）

```python
# agent/orchestrator.py

class AgentOrchestrator:
    """轻量 ReAct 编排 — ~150 行"""
    def __init__(self, llm: BaseLLMProvider, tools: list, memory: MemoryManager): ...

    async def run(self, session_id: str, user_message: str) -> AgentResponse:
        # 1. 加载对话历史 (Redis)
        # 2. 构建 ReAct prompt（思考→行动→观察 循环）
        # 3. LLM 决策：是否需要工具？
        #    - 需要 → 执行工具 → 观察结果 → 继续循环 (max 3 轮)
        #    - 不需要 → 生成最终回答
        # 4. 保存对话历史
        # 5. 返回答案 + 工具调用记录
```

**为什么手写而不引入 LangChain？**
- ReAct 模式核心就是 50 行 prompt + 一个 while 循环
- LangChain 的 AgentExecutor 有大量抽象层，调试困难
- 面试时手写实现可以展示你对 Agent 底层机制的理解
- 代码量不大（~150 行），不增加时间风险

**工具列表（Week 3 实现 2 个）：**
1. `KnowledgeSearch` — 调用检索 pipeline
2. `Calculator` — 安全的数学表达式求值

---

## 六、4 周可行性计划（2026.05.06 - 06.02）

### Week 1：基础设施层（5 月 6-12 日）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 1-2 | 项目脚手架（uv/pyproject.toml）、目录结构、三个 base 接口、config 加载 | 可运行的骨架 |
| Day 3-4 | Claude Provider + OpenAI Provider + ProviderRegistry | 热切换演示脚本 |
| Day 5 | Qdrant Docker 部署、连接测试、collection 创建 | Qdrant 可读写 |
| Day 6 | BGE-M3 本地加载、embedding 测试 | 向量化通路 |
| Day 7 | **Buffer + 单元测试补充** | 覆盖率 >60% |

**里程碑：** `python demo_provider_switch.py` 跑通，输出「Claude: xxx」→ 切换到「OpenAI: xxx」

---

### Week 2：检索管道（5 月 13-19 日）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 8-9 | Document Loader (PDF/MD/TXT) + SemanticChunker | 文档→chunk 通路 |
| Day 10-11 | Ingestion Pipeline（Loader → Chunk → Embed → Qdrant upsert） | 端到端摄入 |
| Day 12-13 | HybridSearch (BM25 + Vector + RRF) + BGE-Reranker | 检索管道完成 |
| Day 14 | **Buffer + 检索评估**（造 50 条 QA，测 Recall@5/MRR） | 评估报告 |

**里程碑：** `python eval_retrieval.py` 输出 Recall@5 > 0.80，检索延迟 < 500ms

---

### Week 3：Agent + API 层（5 月 20-26 日）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 15-16 | MemoryManager (Redis 会话存储) + 对话历史管理 | 多轮记忆通路 |
| Day 17-18 | ReAct AgentOrchestrator + Calculator Tool | Agent 决策循环 |
| Day 19-20 | FastAPI 路由（/query, /ingest, /health）+ middleware | 完整 API |
| Day 21 | **Buffer + 集成测试** | 端到端 5 轮对话测试 |

**里程碑：** `curl /query` 完成一次完整的 RAG 问答，5 轮对话内存正确

---

### Week 4：UI + 部署 + 打磨（5 月 27 日 - 6 月 2 日）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 22-23 | Gradio Web UI（查询框 + 来源展示 + Provider 切换按钮） | 可视化界面 |
| Day 24-25 | Docker Compose（api+qdrant+redis+nginx）+ Nginx 反代 | 一键启动 |
| Day 26-27 | README + 架构文档 + Demo 脚本录制 | 完整文档 |
| Day 28 | **Buffer + 整体联调 + Bug 修复** | 可演示 |

**里程碑：** `docker compose up -d` → 浏览器打开 → 提问 → 看到来源引用 → 切换 Provider → 多轮对话

---

## 七、目录结构（精简版）

```
neuralrag/
├── core/
│   ├── base.py              # 三个抽象接口
│   ├── registry.py          # Provider 注册与热切换
│   └── config.py            # YAML 配置加载
├── providers/
│   ├── claude_provider.py   # Claude (anthropic SDK)
│   ├── openai_provider.py   # OpenAI
│   ├── bge_embedding.py     # BGE-M3 (sentence-transformers)
│   └── bge_reranker.py      # BGE-Reranker-v2-m3
├── ingestion/
│   ├── loader.py            # 文档加载 (LlamaIndex readers)
│   ├── chunker.py           # 语义分块
│   └── pipeline.py          # 摄入编排
├── retrieval/
│   ├── hybrid_search.py     # BM25 + Vector + RRF
│   └── query_expander.py    # 可选查询扩展
├── agent/
│   ├── orchestrator.py      # ReAct 编排 (~150行)
│   ├── memory.py            # Redis 会话管理
│   └── tools.py             # Search / Calculator
├── api/
│   ├── main.py              # FastAPI 入口
│   ├── routes/
│   │   ├── query.py         # POST /query
│   │   ├── ingest.py        # POST /ingest
│   │   └── admin.py         # GET /health, POST /admin/switch
│   └── middleware.py         # retry + fallback + circuit_breaker
├── web/
│   └── app.py               # Gradio UI
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── tests/
│   ├── test_providers.py
│   ├── test_retrieval.py
│   └── test_agent.py
├── config/
│   ├── providers.yaml       # LLM/Embedding 配置
│   └── retrieval.yaml       # 检索策略参数
├── data/                    # 测试文档 + 50 条 QA
├── pyproject.toml
└── README.md
```

---

## 八、验证计划

### 8.1 功能验证

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| Provider 热切换 | 运行时改 providers.yaml，调用 /admin/switch | 无需重启，下个请求使用新 Provider |
| 检索质量 | 50 条中文 QA 对，测 Recall@5 / MRR | Recall@5 > 0.80, MRR > 0.65 |
| 混合检索 vs 纯向量 | 同一测试集对比 | Hybrid 至少提升 5% Recall |
| 多轮对话 | 5 轮连续对话，验证上下文记忆 | 第 5 轮能正确引用第 1 轮的信息 |
| 降级机制 | 模拟 Claude API 超时 | 自动切换到 OpenAI |
| 并发 | 10 并发请求 | 响应延迟 p95 < 2s |

### 8.2 面试演示流程（5 分钟）

1. 打开 Gradio UI，上传一篇 PDF 文档
2. 问一个问题 → 展示回答 + 来源 chunk 高亮
3. 追问（多轮记忆验证）
4. 切换到 OpenAI Provider → 再问 → 展示切换无缝
5. 打开 /health 端点展示系统状态

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| BGE-M3 显存不足 | 中 | 高 | 使用 CPU 推理做 fallback；GPU 不够时 downgrade 到 BGE-small-zh |
| Qdrant hybrid search API 变化 | 低 | 中 | Week 1 即验证 API，锁定版本号 |
| LlamaIndex 版本兼容问题 | 中 | 中 | 锁定 `llama-index==0.11.x`，不追新版本 |
| 时间不够 | 中 | 高 | 每周日晚上评估进度；必要时砍 Gradio UI（FastAPI docs 也可演示）|
| MiniMind 预训练未完成 | 低 | 低 | MiniMind 只是 Provider 桩，不阻塞主流程 |

---

## 十、面试深度问答准备（核心 5 问）

**Q1: 为什么手写 Agent 编排而不直接用 LangChain？**
> LangChain 的 AgentExecutor 封装了太多内部状态，调试时很难知道 LLM 每一步的 raw output 是什么。我的手写 ReAct 循环只有 ~150 行，每一步的 prompt、response、tool output 都有结构化日志，可控性和可调试性远高于框架黑盒。对于 4 周的项目，清晰 > 方便。

**Q2: 为什么选 LlamaIndex 而不是 LangChain 的检索模块？**
> LlamaIndex 的数据连接器生态更成熟（50+ 文档格式），检索策略（HyDE、混合检索、递归检索）是它的核心战场。LangChain 的检索是后来补的，抽象层次太多。选型原则是：谁的核心能力匹配就用谁，不做无意义的全家桶绑定。

**Q3: BM25 + Vector 混合检索，权重 0.3/0.7 是怎么定的？**
> 0.3/0.7 是默认起点，参考了 BGE-M3 论文和 MTEB 排行榜上的经验值。但实际需要在自有评估集上做网格搜索验证，我在 retrieval.yaml 里把权重做成可配置参数，方便调优。

**Q4: 为什么 Embedding 和 LLM 拆成两个接口？**
> 两个原因。第一，生命周期不同 — BGE-M3 是本地 GPU 常驻服务，LLM 是外部 API 调用，混在同一个类里会导致初始化逻辑混乱。第二，接口隔离原则 — 调用方如果只需要 embedding，不应该依赖 generate() 方法。

**Q5: 这个系统如果上线生产，还缺什么？**
> 缺三样：一是认证鉴权（API Key/JWT），目前是裸奔；二是向量库的持久化备份策略；三是 A/B 评估框架（对比不同 chunk 策略/检索策略的效果）。这些我在架构文档里做了「生产扩展」章节，4 周时间不够但思路已经有了。

---

## 十一、和原方案的关键差异总结

| 原方案问题 | v2 修复 |
|-----------|---------|
| LangChain+LlamaIndex 双框架胶水成本高 | LlamaIndex 为主，Agent 手写 ~150 行 |
| `fine_tune()` 混在 `BaseLLMProvider` 上 | 独立 `BaseTrainingAdapter`，MiniMind 只是桩 |
| 512+64 overlap 仅 12.5% | 512+128 overlap 25% |
| Prometheus+Grafana 太重 | 结构化日志 + /metrics，面试够用 |
| Week 1 塞了 8 个任务 | 砍到 5 个核心任务 + 1 天 buffer |
| 没有 risk management | 5 条风险 + 缓解方案 |
| 检索验证无量化指标 | Recall@5 > 0.80, MRR > 0.65 |
| MiniMind 定位为「模型能力展示」 | 定位为「架构扩展性验证」 |
