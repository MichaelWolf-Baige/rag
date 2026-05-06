# NeuralRAG 工业级智能检索Agent系统

## Context

**项目目标**：一个月内完成一个工业级、高扩展性的RAG-Agent系统，用于深圳暑期实习求职。

**用户需求**：
- 有服务器可用（Docker部署）
- API灵活切换（开发用Claude/OpenAI，演示可切换开源模型）
- 先独立开发，预留MiniMind集成接口
- 目标是大厂实习和AI创业公司
- 需要体现系统设计能力和技术深度

---

## 一、项目定位与命名

**项目名称**：NeuralRAG - Industrial-grade Intelligent Retrieval Agent

**一句话亮点**：
"LangChain编排Agent能力 + LlamaIndex优化检索质量 + Hybrid Search两阶段检索 + 多Provider热切换 + Docker生产部署 + MiniMind预留接口"

---

## 二、技术栈选择

### 核心框架（双层架构 - 面试关键卖点）

| 框架 | 用途 | 选择理由 |
|------|------|---------|
| **LlamaIndex** | 数据层/检索层 | 专注检索质量，数据连接器丰富，6ms overhead极低 |
| **LangChain** | Agent编排层 | 工具链、Memory管理成熟，社区活跃 |

**架构亮点**：不是简单选一个框架，而是组合两者优势，体现架构设计能力。

### 向量数据库

| 选择 | 替代方案 | 选择理由 |
|------|---------|---------|
| **Qdrant** | Milvus/Pinecone/pgvector | Rust实现高性能、Hybrid原生支持、Docker友好、单容器部署 |

### Embedding模型

| 选择 | 替代方案 | 选择理由 |
|------|---------|---------|
| **BGE-M3** | text-embedding-3-small | 开源免费、中英文混合效果好、多粒度检索(dense/sparse/colbert)、本地部署 |

### LLM Provider（可插拔）

| Provider | 用途 | 特点 |
|----------|------|------|
| Claude | 开发/高质量演示 | 上下文长、推理强 |
| OpenAI | Fallback备选 | 稳定、兼容性好 |
| Ollama+Qwen | 纯开源演示 | 免费、本地部署 |
| MiniMind | 后续集成 | 预留接口 |

---

## 三、六层模块化架构

```
neuralrag/
├── core/
│   ├── base.py          # 抽象接口契约（Provider可插拔设计）
│   ├── registry.py      # Provider注册管理（配置驱动切换）
│   └── config.py        # 配置加载与验证
│
├── providers/
│   ├── claude_provider.py    # Claude API实现
│   ├── openai_provider.py    # OpenAI API实现
│   ├── ollama_provider.py    # Ollama本地模型实现
│   └── minimind_provider.py  # MiniMind预留接口（TODO）
│
├── retrieval/
│   ├── hybrid_search.py      # BM25 + Vector混合检索
│   ├── reranker.py           # BGE-Reranker精排
│   └── query_expansion.py    # 多查询扩展
│
├── ingestion/
│   ├── loader.py             # 文档加载（PDF/MD/TXT）
│   ├── chunker.py            # 语义分块策略
│   ├── metadata.py           # 元数据增强
│   └── pipeline.py           # 导入流水线
│
├── agent/
│   ├── chains.py             # RAG Chain定义
│   ├── memory.py             # 对话记忆管理
│   ├── tools.py              # 工具定义(Calculator/Search)
│   └── orchestrator.py       # Agent编排器
│
├── api/
│   ├── main.py               # FastAPI入口
│   ├── routes/
│   │   ├── query.py          # 查询接口
│   │   ├── ingest.py         # 导入接口
│   │   └── health.py         # 健康检查
│   └── middleware/
│   │   ├── retry.py          # 重试机制
│   │   ├── fallback.py       # 备选切换
│   │   └── circuit_breaker.py # 熔断器
│
├── monitoring/
│   ├── metrics.py            # Prometheus指标
│   ├── grafana/              # Dashboard配置
│
├── web/
│   └── app.py                # Gradio Web界面
│
├── docker/
│   ├── Dockerfile            # 应用镜像
│   ├── docker-compose.yml    # 服务编排
│   └── nginx.conf            # 反向代理
│
├── tests/
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   └── benchmarks/           # 性能测试
│
├── docs/
│   ├── architecture.md       # 架构文档
│   ├── api.md                # API文档
│   └── deployment.md         # 部署文档
│
├── config/
│   ├── providers.yaml        # Provider配置
│   ├── retrieval.yaml        # 检索配置
│   └── agent.yaml            # Agent配置
│
└── README.md                 # 项目介绍
```

---

## 四、关键接口设计（预留扩展点）

### 1. BaseLLMProvider - Provider抽象接口

```python
# core/base.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional

class BaseLLMProvider(ABC):
    """LLM Provider抽象接口 - 支持多种模型热切换"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """生成响应（支持流式）"""
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """生成向量嵌入"""
        pass

    @abstractmethod
    async def fine_tune(
        self,
        dataset: str,
        config: Dict[str, Any]
    ) -> str:
        """预留微调接口 - MiniMind集成点"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider标识"""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, bool]:
        """能力声明（是否支持stream/embed/fine_tune）"""
        pass
```

### 2. ProviderRegistry - 配置驱动切换

```python
# core/registry.py
class ProviderRegistry:
    """Provider注册管理 - 配置驱动，无需改代码"""

    _providers: Dict[str, BaseLLMProvider] = {}
    _active_provider: str = "claude"

    def register(self, name: str, provider: BaseLLMProvider):
        """注册Provider"""
        self._providers[name] = provider

    def get_active(self) -> BaseLLMProvider:
        """获取当前活跃Provider"""
        return self._providers[self._active_provider]

    def switch_provider(self, name: str) -> bool:
        """切换Provider（配置热更新）"""
        if name not in self._providers:
            raise ValueError(f"Provider {name} not registered")
        self._active_provider = name
        return True

    def reload_from_config(self, config_path: str):
        """从配置文件重新加载（无需重启）"""
        config = yaml.safe_load(open(config_path))
        self._active_provider = config['active_provider']
```

### 3. RetrievalPipeline - 检索策略可配置

```python
# retrieval/hybrid_search.py
class RetrievalPipeline:
    """检索流水线 - 策略可配置"""

    strategies = {
        'semantic': SemanticSearch,
        'hybrid': HybridSearch,
        'multi_query': MultiQuerySearch
    }

    def __init__(self, strategy: str = 'hybrid', config: Dict = None):
        self.strategy = self.strategies[strategy](config)

    async def retrieve(self, query: str, top_k: int = 10):
        return await self.strategy.search(query, top_k)
```

---

## 五、核心数据流设计

### 导入流程

```
文档文件(PDF/MD/TXT)
    ↓
Loader.load() → 原始文本
    ↓
SemanticChunker.chunk() → 文本块(512 tokens, overlap=64)
    ↓
MetadataEnhancer.enrich() → 增强元数据(标题/摘要/来源)
    ↓
BGE-M3.embed() → 向量
    ↓
Qdrant.upsert() → 存储(向量+原文+元数据)
```

### 查询流程

```
用户Query
    ↓
QueryExpansion.expand() → 多查询扩展(原Query + 2变体)
    ↓
HybridSearch.search()
    ├── BM25检索(权重0.3) → 精确匹配候选
    └── Vector检索(权重0.7) → 语义相似候选
    ↓
RRF融合(Reciprocal Rank Fusion) → 合并排序
    ↓
BGE-Reranker.rerank(top_k=50 → rerank_top_k=10) → 精排
    ↓
ContextBuilder.build() → 构建Prompt上下文(来源引用)
    ↓
Claude.generate(stream=True) → 流式响应
    ↓
ResponseFormatter.format() → 最终输出(答案+来源+置信度)
```

### Agent多轮对话流程

```
用户消息
    ↓
MemoryManager.load(session_id) → 加载历史对话(Redis存储)
    ↓
AgentOrchestrator.plan() → 分析意图，决定工具调用
    ├── Calculator(数学问题)
    ├── Search(实时搜索)
    └── Knowledge(RAG检索)
    ↓
ToolExecutor.execute() → 执行工具，获取结果
    ↓
MemoryManager.save(session_id) → 保存对话状态
    ↓
EntityTracker.update() → 更新实体记忆(人名/地点/时间)
```

---

## 六、部署方案

### Docker Compose架构

```yaml
# docker/docker-compose.yml
version: '3.8'
services:
  neuralrag-api:
    build: .
    ports: ["8000:8000"]
    environment:
      - ACTIVE_PROVIDER=claude
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=http://redis:6379
    depends_on: [qdrant, redis]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data:/qdrant/storage"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  nginx:
    image: nginx:alpine
    ports: ["80:80", "443:443"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf"]

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes: ["prometheus_data:/prometheus"]

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
```

### 生产可靠性机制

| 机制 | 实现 | 说明 |
|------|------|------|
| **Retry** | exponential_backoff | 最多3次，间隔1s→2s→4s |
| **Fallback** | Claude→OpenAI→Ollama | 三层备选 |
| **Circuit Breaker** | 连续5次失败后暂停30s | 防止雪崩 |
| **Rate Limit** | 100 req/min | 保护API配额 |
| **Monitoring** | Prometheus+Grafana | 延迟>1000ms告警 |

---

## 七、4周开发计划（28天）

### Week 1：核心架构搭建（Day 1-7）

| Day | 任务 | 验证点 |
|-----|------|--------|
| 1 | 项目初始化、目录结构 | `tree neuralrag` 输出完整 |
| 2 | core/base.py抽象接口 | 单元测试通过 |
| 3 | Claude Provider实现 | `pytest tests/unit/providers/` 通过 |
| 4 | OpenAI Provider实现 | Provider切换测试通过 |
| 5 | ProviderRegistry实现 | 配置文件切换生效 |
| 6 | Qdrant向量库配置 | Docker启动成功 |
| 7 | BGE-M3 Embedding集成 | 向量化测试通过 |

**Week 1里程碑**：Provider热切换可用，单元测试覆盖率>70%

### Week 2：检索流程完善（Day 8-14）

| Day | 任务 | 验证点 |
|-----|------|--------|
| 8 | SemanticChunker实现 | 分块粒度测试 |
| 9 | MetadataEnhancer实现 | 元数据字段完整 |
| 10 | Hybrid Search实现 | BM25+Vector融合 |
| 11 | RRF融合算法实现 | 排序效果测试 |
| 12 | BGE-Reranker集成 | Top50→Top10精排 |
| 13 | 检索Pipeline整合 | 端到端检索测试 |
| 14 | Benchmark测试 | 延迟<500ms |

**Week 2里程碑**：Hybrid检索效果优于单一检索，延迟<500ms

### Week 3：Agent能力+API（Day 15-21）

| Day | 任务 | 验证点 |
|-----|------|--------|
| 15 | RAG Chain实现 | LangChain集成测试 |
| 16 | Memory Manager实现 | Redis存储对话状态 |
| 17 | Tools定义 | Calculator/Search工具可用 |
| 18 | AgentOrchestrator实现 | 多轮对话测试 |
| 19 | FastAPI路由实现 | API文档自动生成 |
| 20 | Retry/Fallback/CircuitBreaker | 可靠性测试 |
| 21 | Prometheus Metrics | 监控指标可访问 |

**Week 3里程碑**：多轮对话流畅，API完整，监控可视化

### Week 4：Web界面+部署（Day 22-28）

| Day | 任务 | 验证点 |
|-----|------|--------|
| 22 | Gradio界面实现 | Web可访问 |
| 23 | Dockerfile编写 | 镜像构建成功 |
| 24 | docker-compose配置 | 一键启动 |
| 25 | Nginx反向代理 | HTTPS访问 |
| 26 | Grafana Dashboard | 监控可视化 |
| 27 | Demo演示录制 | 视频上传 |
| 28 | README+文档完善 | 项目开源标准 |

**Week 4里程碑**：Gradio界面可用，Docker一键启动，Demo视频录制

---

## 八、面试亮点提炼

### 架构层
**一句话**："配置驱动切换Provider，无需改代码"
**深度问题**：
- Q: Claude不稳定怎么办？→ Retry+Fallback+熔断器三层保障
- Q: 为什么不直接调用API？→ 抽象接口便于切换，体现架构思维

### 检索层
**一句话**："Hybrid Search + Reranking两阶段检索"
**深度问题**：
- Q: 为什么不用纯Vector？→ BM25精确匹配更强（如产品型号）
- Q: RRF融合原理？→ 互补融合，降低单一策略风险

### 数据层
**一句话**："语义分块+元数据增强+增量索引"
**深度问题**：
- Q: chunk_size怎么选？→ 512平衡上下文和粒度
- Q: 元数据有什么用？→ 过滤检索范围，提升召回精准度

### Agent层
**一句话**："LangChain编排+Memory管理+Tool集成"
**深度问题**：
- Q: 为什么用LangChain？→ 开箱即用的编排能力，减少重复开发
- Q: Memory怎么管理？→ Redis存储会话状态，支持跨对话记忆

### API层
**一句话**："FastAPI异步+类型安全+自动文档"
**深度问题**：
- Q: 如何处理并发？→ 异步处理+Redis状态管理
- Q: 为什么用FastAPI？→ 自动API文档、类型提示、性能优异

### 部署层
**一句话**："Docker编排+Prometheus监控+告警"
**深度问题**：
- Q: 生产环境怎么保障？→ 监控告警+自动恢复+日志追踪

---

## 九、MiniMind集成路径

### 预留接口位置

```python
# providers/minimind_provider.py
class MiniMindProvider(BaseLLMProvider):
    """MiniMind本地模型Provider - 预留实现"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = self._load_model(model_path)
        self.device = device

    async def generate(self, prompt: str, context: str = None, stream: bool = False):
        # 调用MiniMind推理
        pass

    async def fine_tune(self, dataset: str, config: Dict):
        # 预留微调接口 - 连接MiniMind训练流程
        pass
```

### 配置文件添加

```yaml
# config/providers.yaml
providers:
  minimind:
    type: "minimind"
    model_path: "/models/minimind-160m"
    device: "cuda"
    capabilities:
      stream: true
      embed: false
      fine_tune: true
```

### 集成优势
- 无需修改核心代码，只需实现新Provider
- 体现架构前瞻性，面试加分点
- 可展示"自己训练模型+自己部署应用"全链路能力

---

## 十、验证方案

### 功能验证
1. Provider切换测试：修改config/providers.yaml，无需重启生效
2. 检索效果测试：Hybrid Search召回率>单一检索
3. 多轮对话测试：连续5轮对话，Memory保持正确
4. 压力测试：100并发请求，响应延迟<2s

### 部署验证
1. Docker一键启动：`docker-compose up -d` 成功
2. 监控可视化：Grafana Dashboard显示指标
3. API文档：FastAPI自动生成，可访问测试

### 面试演示流程
1. 打开Web界面，输入Query
2. 展示检索结果+来源引用
3. 切换Provider（Claude→Ollama）
4. 多轮对话展示Memory
5. 打开Grafana展示监控

---

## 关键文件清单

| 文件 | 用途 | 优先级 |
|------|------|--------|
| `core/base.py` | 抽象接口 | P0 |
| `core/registry.py` | Provider注册 | P0 |
| `providers/claude_provider.py` | Claude实现 | P0 |
| `retrieval/hybrid_search.py` | 混合检索 | P0 |
| `api/main.py` | FastAPI入口 | P1 |
| `web/app.py` | Web界面 | P1 |
| `docker/docker-compose.yml` | 部署配置 | P1 |
| `providers/minimind_provider.py` | MiniMind预留 | P2 |