const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const deckPath = path.join(root, "tenstorrent-ggml-scheduler-slides.html");
const notesPath = path.join(root, "tenstorrent-ggml-scheduler-speaker-notes.html");

function esc(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function flow(steps, className = "") {
  return `<div class="flow ${className}">${steps
    .map(
      (step, index) =>
        `<div class="flow-step ${step.tone || ""}"><span>${esc(step.label)}</span><strong>${esc(step.value)}</strong></div>${
          index < steps.length - 1 ? '<span class="arrow">→</span>' : ""
        }`,
    )
    .join("")}</div>`;
}

function ref(text) {
  return `<div class="slide-ref">${text}</div>`;
}

const reportAnchors = {
  cover: "从模型到芯片执行-以-tenstorrent-适配-llama.cpp-ggml-为例",
  questions: "从模型到芯片执行-以-tenstorrent-适配-llama.cpp-ggml-为例",
  entry: "tenstorrent-软件栈",
  roles: "llama.cpp-ggml-与-backend-边界",
  transformer: "模型到算子的执行路径",
  formats: "权重与量化-进入运行时前的信息",
  checkpoint: "权重与量化-进入运行时前的信息",
  safetensors: "权重与量化-进入运行时前的信息",
  gguf: "权重与量化-进入运行时前的信息",
  quant: "权重与量化-进入运行时前的信息",
  "quant-execution": "权重与量化-进入运行时前的信息",
  "runtime-graph": "从-gguf-到-ggml_cgraph",
  cgraph: "从-gguf-到-ggml_cgraph",
  "two-adaptations": "从-gguf-到-ggml_cgraph",
  "scheduler-inputs": "scheduler-如何切分-cgraph",
  placement: "scheduler-如何切分-cgraph",
  passes: "scheduler-如何切分-cgraph",
  copy: "scheduler-如何切分-cgraph",
  allocator: "scheduler-如何切分-cgraph",
  backend: "tenstorrent-backend-如何执行-split",
  "backend-quant": "tenstorrent-backend-如何执行-split",
  "tt-stack": "tenstorrent-backend-如何执行-split",
  hardware: "tenstorrent-wormhole-硬件约束",
  "chip-grid": "tenstorrent-wormhole-硬件约束",
  noc: "tenstorrent-wormhole-硬件约束",
  conclusion: "tenstorrent-wormhole-硬件约束",
  "appendix-files": "权重与量化-进入运行时前的信息",
  "appendix-cgraph": "从-gguf-到-ggml_cgraph",
  "appendix-scheduler": "scheduler-如何切分-cgraph",
  "appendix-backend": "tenstorrent-backend-如何执行-split",
  references: "参考文献",
};

function reportLink(id) {
  const anchor = reportAnchors[id] || "从模型到芯片执行-以-tenstorrent-适配-llama.cpp-ggml-为例";
  return `tenstorrent-ggml-scheduler-integrated-report.html#${anchor}`;
}

function routeInteractive() {
  return `
  <div class="route-demo" data-route-demo>
    <nav class="route-tabs" aria-label="推理路径选择">
      <button data-route="compiler" class="active">Compiler-first</button>
      <button data-route="pytorch">PyTorch backend</button>
      <button data-route="vllm">vLLM runtime</button>
      <button data-route="ggml">llama.cpp / ggml</button>
      <button data-route="manual">Hand-optimized</button>
    </nav>
    <div class="route-body">
      <img data-route-image src="pics/ttSDK_path_compiler_first.png" alt="Compiler-first 进入 Tenstorrent 软件栈的路径">
      <div class="route-readout">
        <div class="eyebrow" data-route-label>Compiler-first</div>
        <h3 data-route-title>图转换和 lowering 先行</h3>
        <p data-route-copy>框架模型经过 Forge / MLIR lowering，最终落到 TT-NN 与 Metalium。</p>
        <strong data-route-focus>关注点：图覆盖、layout legalization、codegen</strong>
      </div>
    </div>
  </div>`;
}

function toyGraph() {
  return `
  <svg class="toy-graph stage-model" data-cgraph-svg viewBox="0 0 1140 790" role="img" aria-label="Toy decoder block 中 tensor 和 op 的依赖关系">
    <defs>
      <marker id="arrow-toy" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
        <path d="M0,0 L9,4.5 L0,9 Z"></path>
      </marker>
    </defs>
    <g class="toy-lines">
      <path d="M116 98 V137"></path><path d="M116 197 V235"></path>
      <path d="M232 265 H392"></path><path d="M530 195 V235"></path><path d="M530 295 V330"></path>
      <path d="M530 390 V410 H568 V448"></path><path d="M116 197 V410 H492 V448"></path>
      <path d="M530 508 V535"></path><path d="M662 565 H735"></path><path d="M860 505 V535"></path>
      <path d="M860 595 V608 H900 V620"></path><path d="M530 508 V608 H820 V620"></path>
      <path d="M860 674 V710"></path><path d="M630 737 H735"></path>
    </g>
    <g class="toy-node input"><rect x="20" y="40" width="192" height="58"></rect><text x="116" y="75">tokens + embedding</text></g>
    <g class="toy-node"><rect x="20" y="137" width="192" height="60"></rect><text x="116" y="162">x0</text><text class="sub" x="116" y="184">GET_ROWS</text></g>
    <g class="toy-node"><rect x="20" y="235" width="212" height="60"></rect><text x="126" y="260">n0</text><text class="sub" x="126" y="282">RMS_NORM</text></g>
    <g class="toy-node weight"><rect x="410" y="135" width="240" height="60"></rect><text x="530" y="171">wq / wk / wv</text></g>
    <g class="toy-node active"><rect x="392" y="235" width="276" height="60"></rect><text x="530" y="260">q, k, v</text><text class="sub" x="530" y="282">MUL_MAT</text></g>
    <g class="toy-node active"><rect x="392" y="330" width="276" height="60"></rect><text x="530" y="355">attn_out</text><text class="sub" x="530" y="377">ATTENTION path</text></g>
    <g class="toy-node"><rect x="392" y="448" width="276" height="60"></rect><text x="530" y="473">x1</text><text class="sub" x="530" y="495">ADD + residual</text></g>
    <g class="toy-node"><rect x="392" y="535" width="270" height="60"></rect><text x="527" y="560">n1</text><text class="sub" x="527" y="582">RMS_NORM</text></g>
    <g class="toy-node weight"><rect x="740" y="445" width="242" height="60"></rect><text x="861" y="481">w_gate / w_up / w_down</text></g>
    <g class="toy-node active"><rect x="735" y="535" width="250" height="60"></rect><text x="860" y="560">ffn_out</text><text class="sub" x="860" y="582">FFN path</text></g>
    <g class="toy-node"><rect x="735" y="620" width="250" height="54"></rect><text x="860" y="642">x2</text><text class="sub" x="860" y="661">ADD + residual</text></g>
    <g class="toy-node weight"><rect x="410" y="710" width="220" height="54"></rect><text x="520" y="742">lm_head</text></g>
    <g class="toy-node final"><rect x="735" y="710" width="250" height="54"></rect><text x="860" y="732">logits</text><text class="sub" x="860" y="751">MUL_MAT</text></g>
  </svg>`;
}

function ggufStructure() {
  return `
  <div class="gguf-layout" data-gguf-layout>
    <div class="blob-block header active" data-blob="header"><span>Header</span><strong>GGUF · version · tensor count · KV count</strong></div>
    <div class="blob-block meta" data-blob="meta"><span>KV metadata</span><strong>architecture · tokenizer · model properties</strong></div>
    <div class="blob-block directory" data-blob="directory"><span>Tensor directory</span><strong>name · shape · ggml_type · data_offset</strong></div>
    <div class="blob-block bytes" data-blob="bytes"><span>Aligned binary blob</span><strong>[ tensor bytes ][ pad ][ tensor bytes ] ...</strong></div>
    <div class="offset-note">data_offset 指向实际权重数据</div>
  </div>`;
}

function ggufInteractive() {
  return `
  <div class="gguf-demo" data-gguf-demo>
    <nav class="blob-tabs" aria-label="GGUF 文件结构">
      <button data-blob-tab="header" class="active">Header</button>
      <button data-blob-tab="meta">Metadata</button>
      <button data-blob-tab="directory">Tensor directory</button>
      <button data-blob-tab="bytes">Data blob</button>
    </nav>
    <div class="gguf-slide">
      ${ggufStructure()}
      <div class="gguf-takeaway">
        <span data-blob-label>文件入口</span>
        <strong data-blob-title>先知道目录规模</strong>
        <p data-blob-copy>magic、版本、tensor 数量和 metadata 数量先确定后续读取结构。</p>
      </div>
    </div>
  </div>`;
}

function cgraphInteractive() {
  return `
  <div class="cgraph-demo" data-cgraph-demo>
    <nav class="cgraph-tabs" aria-label="cgraph 查看步骤">
      <button data-cgraph-stage="model" class="active">模型路径</button>
      <button data-cgraph-stage="ops">op 与 tensor</button>
      <button data-cgraph-stage="inputs">输入来源</button>
    </nav>
    ${toyGraph()}
    <div class="cgraph-caption" data-cgraph-copy><strong>先看路径：</strong>一个 decoder block 输出 logits 的简化计算链。</div>
  </div>`;
}

function schedulerInteractive() {
  const nodes = ["x0", "n0", "qkv", "score", "prob", "attn", "x1", "n1", "ffn", "x2", "logits"];
  return `
  <div class="sched-demo" data-scheduler-demo>
    <nav class="pass-tabs">
      ${["输入", "Pass 1", "Pass 2", "Pass 3", "Pass 4", "Pass 5"].map((title, index) => `<button data-pass="${index}" class="${index === 0 ? "active" : ""}">${title}</button>`).join("")}
    </nav>
    <div class="sched-body">
      <div class="sched-left">
        <div class="legend"><span class="dot tt"></span>TT <span class="dot cpu"></span>CPU <span class="dot pending"></span>未分配 <span class="dot copy"></span>copy</div>
        <div class="node-chain">
          ${nodes.map((node) => `<div class="sched-node pending" data-node="${node}">${node}</div>`).join('<span class="chain-link">→</span>')}
        </div>
        <div class="split-output" data-splits></div>
      </div>
      <div class="pass-text">
        <div class="eyebrow" data-pass-label>输入图</div>
        <h3 data-pass-title>只有部分 tensor 已有位置</h3>
        <p data-pass-copy>权重或输入可能已经在某个 buffer 上，中间结果还没有 assignment。</p>
      </div>
    </div>
  </div>`;
}

const slides = [
  {
    id: "cover",
    section: "Tenstorrent / ggml backend adaptation",
    title: "从模型到芯片执行",
    subtitle: "以 Tenstorrent 适配 llama.cpp / ggml 为例",
    duration: "0:50",
    layout: "cover",
    content: `
      <div class="cover-main">
        <div class="kicker">LLM INFERENCE STACK / INTERNAL STUDY</div>
        <h1>从模型到<br>芯片执行</h1>
        <p>以 Tenstorrent 适配 llama.cpp / ggml 为例</p>
      </div>
      ${flow([
        { label: "模型文件", value: "GGUF" },
        { label: "运行时图", value: "ggml_cgraph" },
        { label: "调度", value: "scheduler" },
        { label: "设备接口", value: "TT backend", tone: "accent" },
        { label: "执行", value: "Wormhole" },
      ], "cover-flow")}
      <div class="cover-footer">完整阅读版：tenstorrent-ggml-scheduler-integrated-report.html</div>`,
    notes: {
      opening: "今天不从某个 kernel 开始。先看一条完整链路。",
      points: ["一个模型文件，怎么进入运行时", "一张计算图，怎么切给 backend", "一个新 backend，怎么真正落到芯片"],
      transition: "先把今天要回答的问题钉住。",
    },
  },
  {
    id: "questions",
    section: "01 / 主线",
    title: "三道边界，决定模型能不能跑起来",
    duration: "1:10",
    content: `
      <div class="three-boundaries">
        <div><span>01</span><h3>文件边界</h3><p>权重、metadata、量化格式<br>怎样进入 runtime？</p></div>
        <div><span>02</span><h3>图与调度</h3><p>cgraph 里的 node<br>怎样形成 split？</p></div>
        <div><span>03</span><h3>设备执行</h3><p>split 怎样变成<br>TT-NN / Metalium 调用？</p></div>
      </div>
      <p class="claim">重点不是“写出一个算子”，而是把这三处接口接通，并能验证。</p>`,
    notes: {
      opening: "我把问题压成三道边界。",
      points: ["第一道：文件里到底给了什么", "第二道：运行时怎样切图", "第三道：芯片 backend 怎样执行"],
      transition: "先看为什么选择 llama.cpp/ggml 这条入口。",
    },
  },
  {
    id: "entry",
    section: "02 / 软件栈入口",
    title: "同一底层执行栈，承接不同入口",
    duration: "2:10",
    content: `
      ${routeInteractive()}
      ${ref("来源：[2, 3, 5, 6]")}
    `,
    notes: {
      opening: "这张图先别一次看完。我按入口依次点过去。",
      points: ["compiler-first：图转换与 lowering 在前", "vLLM：服务调度和 KV cache 管理在前", "llama.cpp/ggml：本地加载、量化和多 backend 执行在前", "入口不同，底部仍汇到 TT-NN、Metalium、LLK"],
      transition: "下面固定在 llama.cpp/ggml 这条路径，明确三部分各自负责什么。",
    },
  },
  {
    id: "roles",
    section: "02 / 软件栈入口",
    title: "选择 ggml backend，是为了观察真实执行边界",
    duration: "1:25",
    content: `
      <div class="responsibility-grid">
        <div><span>llama.cpp</span><strong>组织 LLM 推理</strong><p>模型加载、tokenizer、KV cache、生成流程</p></div>
        <div><span>ggml</span><strong>表达与调度计算</strong><p>tensor、op、cgraph、buffer、scheduler</p></div>
        <div class="accent"><span>TT backend</span><strong>把 split 交给设备</strong><p>supports_op、copy/layout、graph_compute</p></div>
      </div>
      <p class="claim small">前端模型捕获暂时不在这条线内；量化 tensor、跨 backend copy 和设备执行仍全部存在。</p>
      ${ref("来源：[2, 5, 6, 17, 20]")}
    `,
    notes: {
      opening: "选这条路线，不是因为它最简单，而是因为边界足够清楚。",
      points: ["llama.cpp 组织一次完整推理", "ggml 把计算表示成 tensor 和 cgraph", "TT backend 处理被分到设备上的执行片段", "编译器前端不在视线里，但执行难点都保留"],
      transition: "先从模型本身开始：Transformer 最后怎样变成这些 op。",
    },
  },
  {
    id: "transformer",
    section: "03 / 模型到算子",
    title: "Transformer block 最终会展开为 op",
    duration: "1:05",
    content: `
      <div class="media-and-line">
        <img src="pics/transformer_arch.jpg" alt="Transformer architecture">
        <div class="op-strip">
          <span>RMSNorm</span><span>MUL_MAT</span><span>Softmax</span><span>SiLU</span><span>ADD</span>
          <p>Attention、MLP/FFN、Norm 与 residual<br>进入 cgraph 后成为具体 op 与 tensor 依赖</p>
        </div>
      </div>
      ${ref("来源：[1]")}
    `,
    notes: {
      opening: "图里是 Transformer，芯片看到的不会是这个大方块。",
      points: ["它会拆成矩阵乘、norm、softmax、激活、残差", "KV cache 围绕 attention 的 K/V tensor 长期存在", "backend 最终处理的是 op、tensor、layout"],
      transition: "op 之前还有一步：权重以什么格式加载进来。",
    },
  },
  {
    id: "formats",
    section: "04 / 权重格式",
    title: "格式演进来自使用阶段的变化",
    duration: "1:25",
    content: `
      <div class="stage-track">
        <div class="format-stage">
          <div class="tag">训练 / 调试</div>
          <h3>.pt / .pth / .bin</h3>
          <p>state_dict 或 checkpoint<br>灵活，能携带训练状态</p>
          <small>默认路径常依赖 pickle</small>
        </div>
        <span class="track-arrow">→</span>
        <div class="format-stage safe">
          <div class="tag">权重分发</div>
          <h3>safetensors</h3>
          <p>tensor-only<br>名称、shape、dtype、offset</p>
          <small>安全、规整、适合读取</small>
        </div>
        <span class="track-arrow">→</span>
        <div class="format-stage active">
          <div class="tag">本地推理加载</div>
          <h3>GGUF</h3>
          <p>权重 + tokenizer + metadata<br>+ ggml_type</p>
          <small>llama.cpp / ggml 的模型入口</small>
        </div>
      </div>
      ${ref("来源：[7, 8, 9, 10, 11]")}
    `,
    notes: {
      opening: "这不是三个互相替代的后缀，是三个不同阶段的需求。",
      points: ["训练要恢复现场，所以 checkpoint 灵活", "分发主要传权重，所以 safetensors 收窄范围", "本地 C/C++ 推理还要直接拿到 tokenizer 和量化类型，所以到 GGUF"],
      transition: "GGUF 里面，最值得后端关心的是 tensor directory。",
    },
  },
  {
    id: "checkpoint",
    section: "04 / 权重格式",
    title: "训练阶段首先需要保存和恢复状态",
    duration: "1:15",
    content: `
      <div class="state-dict-slide">
        <div class="code-window">
          <div class="code-title">model.state_dict()</div>
          <pre>conv1.weight  -&gt; tensor(...)
conv1.bias    -&gt; tensor(...)
fc1.weight    -&gt; tensor(...)
fc1.bias      -&gt; tensor(...)</pre>
        </div>
        <div class="checkpoint-flow">
          <span>名称 → tensor 字典</span>
          <strong>torch.save(...)</strong>
          <span>.pt / .pth / .bin</span>
          <p>也可以把 optimizer、epoch 等训练恢复状态一起保存为 checkpoint。</p>
        </div>
      </div>
      ${ref("来源：[7, 8]")}
    `,
    notes: {
      opening: "文件格式之前，先把 state_dict 这条线说明白。",
      points: ["state_dict 是名称到 tensor 的字典", "torch.save 可以只写这组权重，也可以写完整 checkpoint", "训练恢复需要这种灵活性", "分发阶段不再需要恢复整个训练现场"],
      transition: "只需要交付权重时，格式的目标会变窄。",
    },
  },
  {
    id: "safetensors",
    section: "04 / 权重格式",
    title: "权重分发需要规整的 tensor-only 容器",
    duration: "1:20",
    content: `
      <div class="safe-layout">
        <div class="safe-file">
          <div class="safe-head">JSON header</div>
          <div>name · dtype · shape · offsets</div>
          <div class="safe-data">raw tensor bytes</div>
        </div>
        <div class="model-folder">
          <div class="eyebrow">Hugging Face 模型目录</div>
          <pre>config.json
tokenizer.json / tokenizer.model
generation_config.json
model-00001-of-000xx.safetensors
model.safetensors.index.json</pre>
        </div>
      </div>
      <p class="claim small">safetensors 收窄到权重存储；模型配置和 tokenizer 仍由目录中的其他文件补齐。</p>
      ${ref("来源：[8, 9]")}
    `,
    notes: {
      opening: "safetensors 对应的是权重分发，而不是训练恢复。",
      points: ["它保存 tensor 描述和原始数据，不恢复任意 Python 对象", "安全是原因之一，读取结构和按需访问也更适合分发", "完整模型仍是多个文件共同构成的目录"],
      transition: "llama.cpp 希望轻量 runtime 直接读一个推理文件，这就引出 GGUF。",
    },
  },
  {
    id: "gguf",
    section: "04 / 权重格式",
    title: "GGUF：让 runtime 直接找到并解释权重",
    duration: "1:50",
    content: `
      ${ggufInteractive()}
      ${ref("来源：[10, 11]")}
    `,
    notes: {
      opening: "GGUF 可以理解为带目录和 metadata 的权重 blob。我按四层展开。",
      points: ["header 先给读取范围", "metadata 携带架构与 tokenizer 信息", "tensor directory 告诉 runtime 每段权重的名称、形状和类型", "最后的 data blob 才保存真实 bytes", "计算图不存进 GGUF，仍在运行时创建"],
      transition: "这里出现的 ggml_type，会直接带出量化问题。",
    },
  },
  {
    id: "quant",
    section: "05 / 量化",
    title: "Q4_K_M 不是每个 tensor 都四比特",
    duration: "1:30",
    content: `
      <div class="quant-title">
        <span>Q4</span><span>K</span><span>M</span>
        <div>主体等级</div><div>blockwise 系列</div><div>模型级 preset 档位</div>
      </div>
      <div class="tensor-types">
        <div><label>多数权重</label><strong>Q4_K</strong></div>
        <div class="higher"><label>敏感 tensor</label><strong>Q5_K</strong></div>
        <div class="higher"><label>更高保真需求</label><strong>Q6_K</strong></div>
      </div>
      <p class="claim small">文件名给出量化 preset；真正执行时仍要逐 tensor 读取 ggml_type。</p>
      ${ref("来源：[10, 12, 13, 14, 15, 16]")}
    `,
    notes: {
      opening: "这一页只抓住一件事：Q4_K_M 是模型级标签。",
      points: ["不同 tensor 对量化误差敏感度不同", "K-quants 是 blockwise 存储格式", "同一个 Q4_K_M 文件里，可能同时存在 Q5_K、Q6_K tensor", "backend 不能只看文件名"],
      transition: "文件加载完成以后，讨论对象从权重转成运行时图。",
    },
  },
  {
    id: "quant-execution",
    section: "05 / 量化",
    title: "量化格式进入 backend 后有三条处理路径",
    duration: "1:35",
    content: `
      <div class="quant-paths">
        <div class="quant-input"><strong>逐 tensor 检查</strong><span>ggml_type · shape/stride · buffer · layout</span></div>
        <div class="quant-choice direct"><strong>直接执行</strong><p>已有经过验证的量化 kernel 或内部重排路径</p></div>
        <div class="quant-choice convert"><strong>转换后执行</strong><p>dequant / requant / 转成设备可消费格式</p></div>
        <div class="quant-choice fallback"><strong>拒绝分配</strong><p>supports_op 返回 false，交给其他 backend</p></div>
      </div>
      <p class="claim small">不能把 Q4_K/Q5_K 的支持写成硬件天然能力；必须落到具体 backend 实现。</p>
      ${ref("来源：[12, 16, 20]")}
    `,
    notes: {
      opening: "量化对适配的影响，不停在模型文件大小。",
      points: ["backend 实际读取的是每个 tensor 的 ggml_type", "有专门实现，才能直接执行", "否则先转换到支持格式，或者让 scheduler 留给其他 backend", "这会直接改变 split 数量与 copy 数量"],
      transition: "有了权重表示，再看运行时怎样生成 cgraph。",
    },
  },
  {
    id: "runtime-graph",
    section: "06 / 运行时建图",
    title: "GGUF 提供权重；build_graph 产生计算图",
    duration: "1:10",
    content: `
      ${flow([
        { label: "文件", value: "GGUF" },
        { label: "加载", value: "arch / vocab / tensors" },
        { label: "权重", value: "ggml tensors" },
        { label: "运行时", value: "model.build_graph" },
        { label: "输出", value: "ggml_cgraph", tone: "accent" },
        { label: "设备分配", value: "scheduler" },
      ], "runtime-flow")}
      <div class="two-statements">
        <p><strong>新增模型架构</strong><br>让 loader / builder 认识模型</p>
        <p><strong>新增硬件 backend</strong><br>让 scheduler 分配后的图能在设备执行</p>
      </div>
      ${ref("来源：[27, 28, 29]")}
    `,
    notes: {
      opening: "这里要把两个工作分开。",
      points: ["GGUF 只是加载信息，不提前保存计算图", "build_graph 根据模型结构生成 cgraph", "我们关心的是后半段：新 backend 接图、算图"],
      transition: "用一个 toy decoder block 看 cgraph 长什么样。",
    },
  },
  {
    id: "cgraph",
    section: "06 / 运行时建图",
    title: "cgraph 记录结果 tensor、op 与输入来源",
    duration: "2:00",
    content: `
      ${cgraphInteractive()}
      ${ref("图中省略 RoPE、mask、KV cache 与 layout；来源：[12, 28, 29]")}
    `,
    notes: {
      opening: "这张图不是 Transformer 架构图，是运行时依赖图。我分三步点。",
      points: ["先看一条 decoder 输出路径", "再看每个结果 tensor 关联的 op", "最后看 src 指针记录输入来源", "wq、w_gate、lm_head 是已加载权重，logits 是最终结果"],
      transition: "有了 cgraph，才轮到 scheduler 决定交给谁算。",
    },
  },
  {
    id: "two-adaptations",
    section: "07 / 适配位置",
    title: "模型适配与硬件适配，入口不同",
    duration: "1:00",
    content: `
      <div class="parallel-tracks">
        <div class="track">
          <div class="track-title">新增模型架构</div>
          <div>architecture / tensor names</div><span>→</span><div>builder</div><span>→</span><div>正确 cgraph</div>
        </div>
        <div class="track emphasis">
          <div class="track-title">新增 Tenstorrent backend</div>
          <div>device / buffer</div><span>→</span><div>supports_op</div><span>→</span><div>graph_compute</div>
        </div>
      </div>
      <p class="claim">本文重点：让已经建立的 cgraph 被新的 backend 正确承接。</p>
    `,
    notes: {
      opening: "这里很容易把两件事混在一起。",
      points: ["新增模型：改 loader 和 builder", "新增芯片：实现 backend 能力", "这次汇报盯的是第二条"],
      transition: "scheduler 在分图前，手里有哪些信息？",
    },
  },
  {
    id: "scheduler-inputs",
    section: "08 / Scheduler",
    title: "scheduler 先看位置，再问能力",
    duration: "1:35",
    content: `
      <div class="sched-inputs">
        <div class="known">
          <h3>已经有 buffer</h3>
          <p>weights</p><p>KV cache</p><p>graph input</p>
          <small>已有位置形成初始锚点</small>
        </div>
        <div class="decision">
          <div class="decision-label">判断条件</div>
          <strong>backend 顺序</strong>
          <strong>supports_op</strong>
          <strong>buffer 兼容性</strong>
          <strong>copy 代价</strong>
        </div>
        <div class="unknown">
          <h3>尚未分配</h3>
          <p>activations</p><p>temporary outputs</p><p>copy tensors</p>
          <small>由 assignment 与 allocator 落地</small>
        </div>
      </div>
      <p class="claim small">supports_op 回答“能否正确执行”，不保证“这是最快方案”。</p>
      ${ref("来源：[17, 18, 19]")}
    `,
    notes: {
      opening: "scheduler 不认识 Qwen，也不认识 TinyLLaMA。",
      points: ["它看到 tensor、op、输入依赖", "已有 buffer 是锚点", "未分配中间结果要推导归属", "supports_op 太宽会错，太窄会碎"],
      transition: "接下来把 split_graph 的五步压成一张可播放的图。",
    },
  },
  {
    id: "placement",
    section: "08 / Scheduler",
    title: "权重位置先给出调度锚点，中间结果随后落地",
    duration: "1:40",
    content: `
      <div class="placement-flow">
        <div><span>模型加载 / context 初始化</span><strong>weights · KV cache</strong><p>可能已经位于某个 backend buffer</p></div>
        <div class="arrow-block">→</div>
        <div class="accent"><span>split_graph</span><strong>node assignment</strong><p>结合已有位置、能力与 backend 顺序切 split</p></div>
        <div class="arrow-block">→</div>
        <div><span>分配与执行</span><strong>activations · copy tensors</strong><p>随后在对应 backend buffer 中获得空间</p></div>
      </div>
      <div class="placement-note"><strong>多设备时：</strong>layer/device 分配与 tensor_split 影响已有权重位置；它不是 split_graph 生成的 split。</div>
      ${ref("来源：[17, 19, 26]")}
    `,
    notes: {
      opening: "这里把两个容易混的 split 分开。",
      points: ["tensor_split 影响模型加载时权重怎样放到多个设备", "split_graph 处理的是运行时 cgraph 怎样划分执行段", "已有权重位置会成为调度锚点", "中间结果与 copy tensor 在后续分配阶段落实空间"],
      transition: "明确锚点以后，再逐步看 split_graph 怎样形成执行片段。",
    },
  },
  {
    id: "passes",
    section: "08 / Scheduler",
    title: "split_graph：assignment 如何形成 split",
    duration: "2:15",
    content: `
      ${schedulerInteractive()}
      ${ref("演示中的 TT/CPU 支持组合仅用于解释调度过程；来源：[17, 19]")}
    `,
    notes: {
      opening: "这页可以慢一点，按 pass 走。",
      points: ["Pass 1：从已有位置找锚点", "Pass 2：沿可支持节点扩展", "Pass 3：升级或补漏，偏向少 copy", "Pass 4：补齐 src 和 view", "Pass 5：backend 变化处切 split，并插 copy"],
      transition: "为什么 copy 很关键？因为 split 之间真的要搬 tensor。",
    },
  },
  {
    id: "copy",
    section: "08 / Scheduler",
    title: "跨 backend 依赖，会显式变成 copy tensor",
    duration: "1:20",
    content: `
      <div class="copy-path">
        <div class="compute tt">TT split 0<br><strong>MUL_MAT</strong></div>
        <div class="copy-hop">copy<br>tensor</div>
        <div class="compute cpu">CPU split 1<br><strong>fallback op</strong></div>
        <div class="copy-hop">copy<br>tensor</div>
        <div class="compute tt">TT split 2<br><strong>MUL_MAT</strong></div>
      </div>
      <div class="copy-reasons">
        <span>前一段结果被另一 backend 消费</span>
        <span>同一个 tensor 被不同 backend 引用</span>
        <span>权重所在 device 不支持当前 op</span>
      </div>
      ${ref("来源：[17, 19]")}
    `,
    notes: {
      opening: "copy 不是因为 tensor 太大要拆开。",
      points: ["它首先是设备边界问题", "结果在 TT，下一段在 CPU，就要搬", "一个 tensor 被两条不同设备路径消费，也可能出现 copy", "split 变碎，copy 通常也会变多"],
      transition: "图切完后，Tenstorrent backend 才真正开始执行。",
    },
  },
  {
    id: "allocator",
    section: "08 / Scheduler",
    title: "切图决定在哪算；allocator 决定数据放在哪",
    duration: "1:35",
    content: `
      <div class="allocation-pipeline">
        <div><span>01</span><strong>split_graph</strong><p>为 node 决定 backend<br>形成 split 与跨设备输入</p></div>
        <div><span>02</span><strong>ggml_gallocr</strong><p>按 backend buffer type<br>规划中间 tensor 空间</p></div>
        <div><span>03</span><strong>compute_splits</strong><p>先完成所需 copy<br>再提交 split 执行</p></div>
      </div>
      <p class="claim small">ggml_cgraph 记录计算依赖；buffer 的实际空间由外部 allocator 与 backend 落实。</p>
      ${ref("来源：[17, 19]")}
    `,
    notes: {
      opening: "cgraph 本身不是一块装好数据的设备内存。",
      points: ["split_graph 的输出是执行分配与 copy 需求", "ggml_gallocr 再为中间 tensor 规划对应 buffer 空间", "执行 split 之前，跨 backend 输入先拷到目标 backend", "这也是 TT buffer 适配会影响运行的原因"],
      transition: "下面进入 TT backend 自己必须实现的能力。",
    },
  },
  {
    id: "backend",
    section: "09 / Backend",
    title: "TT backend 要同时接住类型、布局和执行",
    duration: "1:35",
    content: `
      <div class="backend-layers">
        <div><span>01</span><strong>supports_op</strong><p>op · ggml_type · shape<br>contiguity · stride</p></div>
        <div><span>02</span><strong>buffer / copy</strong><p>设备 tensor<br>与 ggml allocator 对接</p></div>
        <div><span>03</span><strong>graph_compute</strong><p>TT-NN 或 Metalium<br>实际提交执行</p></div>
      </div>
      <div class="dtype-line"><strong>示例映射</strong><span>Q4_K → BFLOAT4_B</span><span>Q5_K / Q6_K → BFLOAT8_B</span><small>取决于具体 backend 实现版本</small></div>
      ${ref("来源：[20]")}
    `,
    notes: {
      opening: "到这里才进入 TT backend 本身。",
      points: ["第一层：敢不敢接受这个 op", "第二层：buffer 怎么和 ggml 的分配模型接起来", "第三层：实际调用 TT-NN 或 Metalium", "类型映射属于实现能力，不等于硬件天然支持所有 GGML 格式"],
      transition: "这些软件选择背后，是 Wormhole 的执行模型。",
    },
  },
  {
    id: "backend-quant",
    section: "09 / Backend",
    title: "量化支持最终落在 supports_op 与转换路径上",
    duration: "1:35",
    content: `
      <div class="support-table">
        <div class="support-head"><span>输入状态</span><span>backend 判断</span><span>执行结果</span></div>
        <div><span>类型、layout、kernel 路径齐备</span><strong>supports_op = true</strong><em>TT split 直接执行</em></div>
        <div><span>原格式不可直接消费，但可转换</span><strong>convert / dequant</strong><em>以支持格式执行</em></div>
        <div><span>没有可验证路径</span><strong>supports_op = false</strong><em>留给其他 backend</em></div>
      </div>
      <p class="claim small">转换带来额外数据搬运或存储开销；拒绝则可能增加 split 与跨 backend copy。</p>
      ${ref("来源：[16, 20]")}
    `,
    notes: {
      opening: "前面提到的 dequant，在这里才成为具体执行判断。",
      points: ["支持某个 ggml_type 需要 kernel 或转换路径证据", "转换路径可以执行，但带来额外成本", "拒绝路径保证正确性，却可能把图切碎", "因此扩大支持面要同时验证结果与数据移动代价"],
      transition: "TT backend 再往下，会落入 Tenstorrent 自己的软件层次。",
    },
  },
  {
    id: "tt-stack",
    section: "09 / Backend",
    title: "TT backend 的执行落点：算子、runtime 与 kernel",
    duration: "1:20",
    content: `
      <div class="tt-stack">
        <div class="incoming">ggml split<br><strong>tensor + op</strong></div>
        <span>→</span>
        <div class="stack-layers">
          <div><strong>TT-NN</strong><small>神经网络算子与布局能力</small></div>
          <div><strong>TT-Metalium</strong><small>设备、内存、数据搬运与 kernel launch</small></div>
          <div><strong>TT-LLK</strong><small>更接近硬件的低层 kernel</small></div>
        </div>
        <span>→</span>
        <div class="incoming chip">Wormhole<br><strong>Tensix / NoC</strong></div>
      </div>
      <p class="claim small">适配初期优先复用可验证算子路径；性能瓶颈明确后再下沉到更低层实现。</p>
      ${ref("来源：[2, 20, 22]")}
    `,
    notes: {
      opening: "TT backend 不等于直接写最底层 kernel。",
      points: ["能由 TT-NN 承接的 op 先落到成熟算子路径", "Metalium 负责设备和数据流相关的运行机制", "只有关键路径需要进一步下沉 LLK 或自定义 kernel", "这和前面的多入口软件栈重新汇合"],
      transition: "最后看这些选择对应的 Wormhole 硬件约束。",
    },
  },
  {
    id: "hardware",
    section: "10 / Hardware",
    title: "多 chip 拓扑先约束数据应放在哪里",
    duration: "1:25",
    content: `
      <div class="hardware-feature">
        <img src="pics/tt_hw_board_topology.png" alt="Wormhole n300 board topology">
        <div>
          <strong>Wormhole n300 board</strong>
          <p>板上两个 chip，host 到不同 chip 的路径不完全对称。</p>
          <p>多设备放置、peer copy 与 host 中转要考虑实际互联拓扑。</p>
        </div>
      </div>
      ${ref("来源：[22]")}
    `,
    notes: {
      opening: "先看板级拓扑，它会影响最上层的数据放置。",
      points: ["两个 chip 并非拥有完全相同的 host 路径", "多 device 分配不能只看容量，也要看搬运路径", "backend 的 copy 与 device 表达最终受物理拓扑约束"],
      transition: "进入单个 chip 后，约束从拓扑转到 tile 与本地存储。",
    },
  },
  {
    id: "chip-grid",
    section: "10 / Hardware",
    title: "单 chip 内，计算与数据搬运由 tile 网络组织",
    duration: "1:35",
    content: `
      <div class="hardware-duo">
        <figure><img src="pics/tt_hw_chip_tile_grid.png" alt="Wormhole tile grid"><figcaption>Compute、DRAM、PCIe、Ethernet tiles</figcaption></figure>
        <figure><img src="pics/tt_hw_tensix_pipeline.png" alt="Tensix compute pipeline"><figcaption>Compute tile 内的 Tensix 执行路径</figcaption></figure>
      </div>
      <p class="claim small">算子执行依赖本地 SRAM、显式数据流与 Tensix pipeline；这决定自定义 kernel 的实现成本。</p>
      ${ref("来源：[21, 22, 23]")}
    `,
    notes: {
      opening: "单个 chip 也不是一块统一共享缓存的计算表面。",
      points: ["边缘 tile 承接外部通信和内存", "compute tile 用本地 SRAM 协作完成计算", "Tensix 中计算发射、unpack、matrix、pack 都与数据流配合"],
      transition: "tile 之间怎么搬数据，直接关系到 layout 与 copy 成本。",
    },
  },
  {
    id: "noc",
    section: "10 / Hardware",
    title: "NoC 与 tiled layout 把软件语义变成搬运成本",
    duration: "1:45",
    content: `
      <div class="noc-layout">
        <img src="pics/tt_hw_noc_async_memcpy.png" alt="NoC asynchronous data movement">
        <div class="layout-conversion">
          <span>ggml 侧</span><strong>row-major · stride · view</strong>
          <div>→</div>
          <span>TT 执行路径</span><strong>tile · padding · materialize · copy</strong>
        </div>
      </div>
      <p class="claim small">NoC 搬运为异步并需要同步；非连续或不对齐的数据路径是否昂贵，需要结合具体 kernel 与 benchmark 验证。</p>
      ${ref("来源：[22, 23, 24, 25]")}
    `,
    notes: {
      opening: "这页把前面的 layout/copy 判断接回硬件。",
      points: ["跨 tile 数据流由 NoC 明确搬运", "主要计算路径常按数据 tile 处理，边界尺寸要 padding", "ggml 的 stride 与 view 可能要求转换或物化", "这些成本最终要在实际 op 路径上测量"],
      transition: "到这里，验证顺序就可以收束出来。",
    },
  },
  {
    id: "conclusion",
    section: "11 / 验证",
    title: "先验证执行路径，再谈性能优化",
    duration: "1:30",
    content: `
      <div class="checks">
        <div><span>01</span>assignment 为什么这样分？</div>
        <div><span>02</span>supports_op 为什么 true / false？</div>
        <div><span>03</span>哪些 tensor 发生跨 backend copy？</div>
        <div><span>04</span>TT split 能否完整正确执行？</div>
      </div>
      <div class="landing">
        <strong>底层统一，入口多样；</strong>
        <strong>适配先求可解释，再求更快。</strong>
      </div>
      <a class="reading-link" href="tenstorrent-ggml-scheduler-integrated-report.html">完整阅读版 →</a>
    `,
    notes: {
      opening: "最后我用四个问题判断第一阶段是否完成。",
      points: ["先能解释 placement", "再能解释拒绝原因和 copy", "再对齐 CPU baseline", "最后才扩大覆盖、减少 fallback、优化 layout"],
      transition: "后面是参考资料，问题可以沿这四个点展开。",
    },
  },
  {
    id: "appendix-files",
    section: "Appendix / A1",
    title: "三类容器的使用阶段与信息范围",
    duration: "0:55",
    content: `
      <div class="appendix-table formats-table">
        <div class="table-head"><span>容器</span><span>主要阶段</span><span>保留的信息</span><span>进入 llama.cpp 前</span></div>
        <div><strong>.pt / .pth / .bin</strong><span>训练 / 调试</span><span>权重或完整 checkpoint</span><span>通常需转换</span></div>
        <div><strong>safetensors</strong><span>权重分发</span><span>命名 tensor + 少量 metadata</span><span>需转换到 GGUF</span></div>
        <div class="highlight"><strong>GGUF</strong><span>本地推理</span><span>tensor + tokenizer + ggml_type</span><span>直接加载</span></div>
      </div>
      ${ref("来源：[5, 7, 9, 10, 11]")}
    `,
    notes: {
      opening: "这页用来回应格式对比的问题。",
      points: ["三者服务的阶段不同", "safetensors 不承担 tokenizer 和 GGML 类型表达", "GGUF 是 llama.cpp 常规加载入口"],
      transition: "若问题转到数据结构，再看 cgraph 的结构对应。",
    },
  },
  {
    id: "appendix-cgraph",
    section: "Appendix / A2",
    title: "结果 tensor 如何记录 op 与输入来源",
    duration: "0:55",
    content: `
      <div class="struct-example">
        <pre>struct ggml_tensor {
    enum ggml_type type;
    enum ggml_op   op;
    struct ggml_tensor * src[GGML_MAX_SRC];
    ...
};</pre>
        <div>
          <strong>x1 = ADD(x0, attn_out)</strong>
          <p>x1.op = GGML_OP_ADD</p>
          <p>x1.src[0] = x0</p>
          <p>x1.src[1] = attn_out</p>
        </div>
      </div>
      ${ref("简化示意；来源：[12, 28, 29]")}
    `,
    notes: {
      opening: "如果有人追问 src 是什么，这页直接回答。",
      points: ["结果 tensor 本身携带 op 与 source tensor 指针", "cgraph 从结果沿来源关系收集依赖", "示意只保留本文需要解释的字段"],
      transition: "再追到 scheduler，就看五步划分。",
    },
  },
  {
    id: "appendix-scheduler",
    section: "Appendix / A3",
    title: "split_graph 的五步与 copy 生成条件",
    duration: "1:00",
    content: `
      <div class="pass-summary">
        <div><span>1</span>已有 buffer 找锚点</div>
        <div><span>2</span>沿可支持 node 扩展</div>
        <div><span>3</span>升级 / 补漏分配</div>
        <div><span>4</span>补齐 src 与 view 归属</div>
        <div class="accent"><span>5</span>backend 变化处切 split；不兼容输入生成 copy tensor</div>
      </div>
      <p class="claim small">split 是一段交给同一 backend 的 node 子图，不等同于 Transformer layer。</p>
      ${ref("来源：[17, 19]")}
    `,
    notes: {
      opening: "这页是交互演示的静态提要。",
      points: ["前四步主要形成 node assignment", "第五步真正产生 split 和跨设备输入", "split 粒度由调度结果决定，不与模型层一一对应"],
      transition: "最后保留一个 backend 实现边界的追问入口。",
    },
  },
  {
    id: "appendix-backend",
    section: "Appendix / A4",
    title: "实验 backend 暴露了 allocator 对接难点",
    duration: "1:05",
    content: `
      <div class="implementation-boundary">
        <div><strong>ggml allocator 期望</strong><p>可用 base + offset 表达 tensor 在 buffer 中的位置</p></div>
        <span>≠</span>
        <div><strong>Metalium tensor 管理</strong><p>设备 tensor 不直接按同样线性偏移模型暴露</p></div>
      </div>
      <div class="implementation-note">公开实验实现使用 dummy base 与延迟创建真实 device tensor 的适配方式；它用于说明接口摩擦，不应被视为最终架构结论。</div>
      ${ref("来源：[20]")}
    `,
    notes: {
      opening: "这页只在讨论实现深水区时使用。",
      points: ["scheduler 能切图，不代表 buffer 接口天然匹配", "allocator 假设和设备 tensor 管理方式可能冲突", "实验实现提供了一个可研究的适配方式，不代表产品实现"],
      transition: "参考资料页列出进一步核查入口。",
    },
  },
  {
    id: "references",
    section: "Appendix",
    title: "核心资料入口",
    duration: "0:20",
    content: `
      <div class="sources">
        <p><span>[5-6]</span> llama.cpp README / ggml README</p>
        <p><span>[9-11]</span> Hugging Face safetensors / GGUF；GGUF specification</p>
        <p><span>[12-19]</span> ggml type、backend、scheduler 与 loader 本地源码快照</p>
        <p><span>[20]</span> Metalium backend documentation and source</p>
        <p><span>[22-24]</span> FOSDEM Wormhole talk；Tenstorrent Metalium docs</p>
      </div>
      <div class="appendix-links">
        <a href="tenstorrent-ggml-scheduler-integrated-report.html">打开完整 HTML 阅读版</a>
        <a href="tenstorrent-ggml-scheduler-speaker-notes.html">打开演讲稿</a>
      </div>
    `,
    notes: {
      opening: "资料入口都在完整阅读版里，这里只留主要来源。",
      points: ["代码事实以本地源码快照为基准", "硬件部分来自 Tenstorrent 文档与 FOSDEM slides", "会后可直接打开长文查细节"],
      transition: "结束。",
    },
  },
];

function renderSlide(slide, index) {
  const count = String(index + 1).padStart(2, "0");
  return `<section class="slide ${index === 0 ? "active" : ""} ${slide.layout || ""}" id="slide-${count}" data-index="${index}" aria-hidden="${index === 0 ? "false" : "true"}">
    <header class="slide-header"><span>${esc(slide.section)}</span><span class="header-right"><a class="section-link" href="${reportLink(slide.id)}" target="_blank" rel="noreferrer">完整文档 · 本节 ↗</a><span>${count} / ${String(slides.length).padStart(2, "0")}</span></span></header>
    ${slide.layout === "cover" ? slide.content : `<h2>${esc(slide.title)}</h2><div class="slide-content">${slide.content}</div>`}
  </section>`;
}

function renderDeck() {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>从模型到芯片执行 - Slides</title>
  <style>
    :root {
      --bg: #f5f7f6; --paper: #ffffff; --ink: #142124; --muted: #5b686b; --line: #d4ddda;
      --teal: #087e78; --teal-soft: #dcefeb; --purple: #5d57a5; --purple-soft: #eceafb;
      --amber: #a65f12; --amber-soft: #ffefd9; --cpu: #58646b; --cpu-soft: #e9edef;
      --pad-x: clamp(46px, 5vw, 78px); --pad-y: clamp(34px, 4vh, 52px);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #e8eeec; color: var(--ink); }
    body { display: grid; place-items: center; }
    .deck { position: relative; width: min(100vw, calc(100vh * 16 / 9)); height: min(100vh, calc(100vw * 9 / 16)); aspect-ratio: 16 / 9; background: var(--bg); box-shadow: 0 2px 30px rgba(25,37,39,.08); overflow: hidden; }
    .slide { display: none; position: absolute; inset: 0; padding: var(--pad-y) var(--pad-x) 48px; background: var(--bg); }
    .slide.active { display: block; }
    .slide-header { display: flex; justify-content: space-between; margin-bottom: clamp(20px, 2.7vh, 32px); color: var(--muted); font-size: clamp(11px, .88vw, 13px); font-weight: 600; text-transform: uppercase; }
    .header-right { display: flex; align-items: center; gap: 22px; }
    .section-link { color: var(--teal); font-weight: 600; text-transform: none; text-decoration: none; }
    .section-link:hover { text-decoration: underline; }
    h1 { margin: 0; font-size: clamp(62px, 7vw, 102px); line-height: 1.07; font-weight: 720; }
    h2 { margin: 0 0 clamp(24px, 3.7vh, 42px); font-size: clamp(32px, 3.4vw, 49px); line-height: 1.18; font-weight: 700; }
    h3 { margin: 0; font-size: clamp(18px, 1.6vw, 25px); line-height: 1.35; }
    p { margin: 0; }
    .slide-content { height: calc(100% - 116px); position: relative; }
    .slide-ref { position: absolute; left: 0; bottom: 0; color: var(--muted); font-size: clamp(10px, .73vw, 12px); }
    .cover-main { padding-top: clamp(54px, 8vh, 94px); }
    .kicker { margin-bottom: 22px; color: var(--teal); font-size: clamp(13px, 1vw, 15px); font-weight: 700; }
    .cover-main p { margin-top: 26px; font-size: clamp(22px, 2vw, 29px); color: var(--muted); }
    .cover-flow { position: absolute; left: var(--pad-x); right: var(--pad-x); bottom: 96px; }
    .cover-footer { position: absolute; bottom: 34px; left: var(--pad-x); color: var(--muted); font-size: 13px; }
    .flow { display: flex; align-items: stretch; gap: clamp(7px, 1vw, 14px); }
    .flow-step { flex: 1; min-width: 0; display: grid; gap: 8px; min-height: clamp(68px, 8vh, 86px); padding: 13px 15px; background: var(--paper); border: 1px solid var(--line); }
    .flow-step span { color: var(--muted); font-size: clamp(11px, .8vw, 13px); }
    .flow-step strong { font-size: clamp(14px, 1.16vw, 19px); line-height: 1.35; }
    .flow-step.accent { border-color: var(--teal); background: var(--teal-soft); }
    .arrow { display: grid; place-items: center; color: var(--teal); font-size: 23px; }
    .three-boundaries { display: grid; grid-template-columns: repeat(3, 1fr); gap: clamp(20px, 2.5vw, 38px); padding-top: 30px; }
    .three-boundaries div { min-height: 290px; padding-top: 24px; border-top: 3px solid var(--teal); }
    .three-boundaries span { color: var(--teal); font-size: 18px; font-weight: 700; }
    .three-boundaries h3 { margin: 28px 0 20px; font-size: clamp(26px, 2.25vw, 33px); }
    .three-boundaries p { color: var(--muted); font-size: clamp(17px, 1.3vw, 21px); line-height: 1.65; }
    .claim { position: absolute; bottom: 34px; left: 0; right: 0; color: var(--ink); font-size: clamp(21px, 1.85vw, 28px); font-weight: 600; }
    .claim.small { font-size: clamp(18px, 1.4vw, 23px); }
    .route-demo { height: calc(100% - 16px); }
    .route-tabs { display: flex; justify-content: center; gap: 7px; margin: -10px 0 16px; }
    .route-tabs button, .blob-tabs button, .cgraph-tabs button { padding: 9px 15px; border: 1px solid var(--line); background: var(--paper); color: var(--muted); font-size: 14px; cursor: pointer; }
    .route-tabs button.active, .blob-tabs button.active, .cgraph-tabs button.active { color: var(--teal); border-color: var(--teal); background: var(--teal-soft); font-weight: 650; }
    .route-body { display: grid; grid-template-columns: minmax(480px, 1.55fr) minmax(275px, .72fr); gap: 30px; align-items: center; height: min(52vh, 470px); min-height: 0; }
    .route-body img { display: block; width: 100%; height: 100%; max-height: 470px; min-height: 0; object-fit: contain; }
    .route-readout { border-left: 3px solid var(--teal); padding: 15px 0 15px 22px; }
    .route-readout .eyebrow { margin-bottom: 17px; color: var(--teal); font-size: 14px; font-weight: 700; }
    .route-readout h3 { margin-bottom: 17px; font-size: clamp(22px, 1.65vw, 27px); }
    .route-readout p { margin-bottom: 22px; color: var(--muted); font-size: 16px; line-height: 1.65; }
    .route-readout strong { color: var(--teal); font-size: 15px; line-height: 1.55; }
    .responsibility-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; padding-top: 42px; }
    .responsibility-grid div { min-height: 270px; padding: 25px; border-top: 4px solid #93a29f; background: var(--paper); }
    .responsibility-grid div.accent { border-color: var(--teal); background: var(--teal-soft); }
    .responsibility-grid span { color: var(--teal); font-size: 15px; font-weight: 650; }
    .responsibility-grid strong { display: block; margin: 39px 0 21px; font-size: clamp(22px, 1.7vw, 28px); }
    .responsibility-grid p { color: var(--muted); font-size: 17px; line-height: 1.65; }
    .media-and-line { height: 79%; display: flex; align-items: center; justify-content: center; gap: clamp(58px, 7vw, 100px); }
    .media-and-line img { width: min(330px, 29%); height: 88%; object-fit: contain; }
    .op-strip { max-width: 510px; display: flex; flex-wrap: wrap; gap: 10px; align-content: center; }
    .op-strip span { padding: 13px 17px; color: var(--teal); background: var(--teal-soft); font-size: clamp(17px, 1.2vw, 20px); font-weight: 600; }
    .op-strip p { width: 100%; margin-top: 27px; color: var(--muted); font-size: clamp(17px, 1.28vw, 21px); line-height: 1.75; }
    .stage-track { display: flex; align-items: center; justify-content: space-between; gap: clamp(14px, 1.2vw, 22px); padding-top: 32px; }
    .format-stage { flex: 1; min-height: 310px; padding: 27px 25px; border-top: 4px solid #8e9d99; background: var(--paper); }
    .format-stage.safe { border-color: var(--purple); }
    .format-stage.active { border-color: var(--teal); background: var(--teal-soft); }
    .format-stage .tag { color: var(--muted); font-size: 14px; font-weight: 600; }
    .format-stage h3 { margin: 40px 0 24px; font-size: clamp(25px, 2vw, 31px); }
    .format-stage p { font-size: clamp(16px, 1.2vw, 19px); line-height: 1.7; }
    .format-stage small { display: block; margin-top: 30px; color: var(--muted); font-size: 14px; }
    .track-arrow { color: var(--teal); font-size: 28px; }
    .state-dict-slide { display: grid; grid-template-columns: 1.05fr .95fr; gap: 52px; align-items: center; height: 72%; }
    .code-window { padding: 0 0 24px; border: 1px solid var(--line); background: var(--paper); }
    .code-title { padding: 13px 18px; border-bottom: 1px solid var(--line); color: var(--teal); font-weight: 650; }
    .code-window pre { margin: 25px 27px 0; color: #2d3b3f; font: clamp(15px, 1.1vw, 18px)/1.85 Consolas, monospace; }
    .checkpoint-flow { display: grid; gap: 17px; text-align: center; justify-items: center; }
    .checkpoint-flow span { width: 88%; padding: 16px 20px; background: var(--paper); border: 1px solid var(--line); font-size: 19px; }
    .checkpoint-flow strong { padding: 14px 40px; color: var(--teal); border: 2px solid var(--teal); font-size: 22px; }
    .checkpoint-flow p { margin-top: 12px; color: var(--muted); font-size: 16px; line-height: 1.65; }
    .safe-layout { display: grid; grid-template-columns: .88fr 1.12fr; align-items: center; gap: 60px; height: 69%; }
    .safe-file { display: grid; gap: 8px; padding: 13px; border: 1px solid var(--line); background: var(--paper); font: 17px/1.5 Consolas, monospace; }
    .safe-file div { padding: 17px 19px; background: #f1f5f4; }
    .safe-file .safe-head { color: var(--teal); background: var(--teal-soft); font-weight: 700; }
    .safe-file .safe-data { min-height: 104px; display: grid; align-items: center; }
    .model-folder .eyebrow { margin-bottom: 20px; color: var(--teal); font-weight: 700; }
    .model-folder pre { margin: 0; padding: 21px 25px; background: var(--paper); border-left: 3px solid var(--purple); color: #334347; font: clamp(16px, 1.2vw, 20px)/1.75 Consolas, monospace; }
    .gguf-demo { height: calc(100% - 20px); }
    .blob-tabs { display: flex; gap: 7px; margin: -8px 0 20px; }
    .gguf-slide { display: grid; grid-template-columns: minmax(470px, 1.1fr) minmax(300px, .72fr); gap: clamp(35px, 4vw, 62px); height: 76%; align-items: center; }
    .gguf-layout { display: grid; gap: 7px; }
    .blob-block { display: grid; gap: 6px; padding: 18px 20px; border: 1px solid var(--line); background: var(--paper); }
    .blob-block span { color: var(--teal); font-weight: 700; font-size: 15px; }
    .blob-block strong { font: 15px/1.5 Consolas, monospace; }
    .blob-block.active { border-color: var(--teal); background: var(--teal-soft); }
    .blob-block.bytes { background: #edf2f0; }
    .offset-note { justify-self: center; padding: 8px 17px; color: var(--amber); background: var(--amber-soft); font-size: 15px; }
    .gguf-takeaway span { display: block; margin-bottom: 25px; color: var(--muted); font-size: 16px; }
    .gguf-takeaway strong { display: block; margin-bottom: 16px; color: var(--teal); font-size: clamp(27px, 2vw, 33px); }
    .gguf-takeaway p { margin-top: 31px; color: var(--muted); font-size: 17px; line-height: 1.7; }
    .quant-title { display: grid; grid-template-columns: repeat(3, 1fr); max-width: 820px; margin: 34px auto 45px; text-align: center; }
    .quant-title span { color: var(--teal); font-size: clamp(65px, 6vw, 88px); font-weight: 720; border-right: 1px solid var(--line); }
    .quant-title span:nth-child(3) { border-right: 0; }
    .quant-title div { margin-top: 9px; color: var(--muted); font-size: 16px; }
    .tensor-types { display: flex; justify-content: center; gap: 12px; }
    .tensor-types div { width: 230px; padding: 16px 20px; border-top: 3px solid var(--teal); background: var(--paper); display: grid; gap: 11px; }
    .tensor-types .higher { border-color: var(--purple); }
    .tensor-types label { color: var(--muted); font-size: 14px; }
    .tensor-types strong { font-size: 27px; }
    .quant-paths { display: grid; grid-template-columns: 1.12fr repeat(3, 1fr); gap: 15px; padding-top: 75px; }
    .quant-input, .quant-choice { min-height: 242px; padding: 27px 22px; background: var(--paper); border-top: 4px solid #98a6a3; }
    .quant-input { display: grid; align-content: center; gap: 22px; border-color: var(--purple); }
    .quant-input strong { font-size: 27px; }
    .quant-input span { color: var(--muted); font-size: 17px; line-height: 1.75; }
    .quant-choice strong { display: block; margin-bottom: 33px; font-size: 25px; }
    .quant-choice p { color: var(--muted); font-size: 16px; line-height: 1.7; }
    .quant-choice.direct { border-color: var(--teal); background: var(--teal-soft); }
    .quant-choice.convert { border-color: var(--amber); }
    .runtime-flow { margin-top: 80px; }
    .two-statements { display: flex; justify-content: center; gap: 38px; margin-top: 75px; }
    .two-statements p { width: min(350px, 34%); padding-top: 16px; border-top: 3px solid var(--line); color: var(--muted); font-size: 17px; line-height: 1.75; }
    .two-statements strong { color: var(--teal); font-size: 21px; }
    .cgraph-demo { height: calc(100% - 6px); position: relative; }
    .cgraph-tabs { position: absolute; left: 0; top: 0; z-index: 2; display: flex; gap: 6px; }
    .toy-graph { display: block; width: min(1080px, 100%); height: calc(100% - 94px); margin: 44px auto 0; }
    .toy-lines path { fill: none; stroke: #aeb9b7; stroke-width: 2.4; marker-end: url(#arrow-toy); }
    .toy-graph marker path { fill: #aeb9b7; }
    .toy-node rect { fill: var(--paper); stroke: #acb7b4; rx: 5; }
    .toy-node text { text-anchor: middle; fill: var(--ink); font-size: 16px; font-weight: 600; }
    .toy-node .sub { fill: var(--muted); font-size: 12px; font-weight: 500; }
    .toy-node.input rect { fill: var(--purple-soft); stroke: var(--purple); }
    .toy-node.weight rect { fill: #eef5f4; stroke: #79938e; }
    .toy-node.weight text { fill: #36514d; font-size: 13px; }
    .toy-node.active rect { fill: var(--teal-soft); stroke: var(--teal); }
    .toy-node.final rect { fill: var(--amber-soft); stroke: var(--amber); }
    .toy-graph.stage-ops .toy-node.weight { opacity: .32; }
    .toy-graph.stage-ops .toy-node.active rect, .toy-graph.stage-ops .toy-node.final rect { stroke-width: 3; }
    .toy-graph.stage-inputs .toy-lines path { stroke: var(--teal); stroke-width: 3; }
    .toy-graph.stage-inputs marker path { fill: var(--teal); }
    .cgraph-caption { position: absolute; right: 0; bottom: 34px; padding: 11px 16px; color: var(--muted); background: var(--paper); border: 1px solid var(--line); font-size: 14px; }
    .parallel-tracks { display: grid; gap: 36px; padding: 60px 16px 0; }
    .track { display: grid; grid-template-columns: 220px 1fr 32px 1fr 32px 1fr; gap: 13px; align-items: center; }
    .track-title { font-size: 24px; font-weight: 650; }
    .track div:not(.track-title) { min-height: 75px; display: grid; place-items: center; border: 1px solid var(--line); background: var(--paper); font-size: 19px; }
    .track span { color: var(--teal); font-size: 25px; text-align: center; }
    .track.emphasis .track-title { color: var(--teal); }
    .track.emphasis div:not(.track-title) { border-color: var(--teal); background: var(--teal-soft); }
    .sched-inputs { display: grid; grid-template-columns: 1fr 1.04fr 1fr; gap: 26px; align-items: start; padding-top: 36px; }
    .sched-inputs h3 { margin-bottom: 22px; font-size: 26px; }
    .sched-inputs p { margin: 9px 0; color: var(--ink); font-size: 19px; }
    .sched-inputs small { display: block; margin-top: 28px; color: var(--muted); font-size: 14px; }
    .known, .unknown { min-height: 300px; padding: 24px 27px; border-top: 3px solid var(--purple); background: var(--paper); }
    .unknown { border-color: #97a4a1; }
    .decision { min-height: 320px; display: grid; gap: 13px; padding: 24px; border: 2px solid var(--teal); background: var(--teal-soft); }
    .decision-label { color: var(--teal); font-size: 14px; font-weight: 700; }
    .decision strong { padding: 10px 0; border-bottom: 1px solid rgba(8,126,120,.22); font-size: 21px; }
    .placement-flow { display: grid; grid-template-columns: 1fr 52px 1.15fr 52px 1fr; align-items: center; gap: 12px; padding-top: 72px; }
    .placement-flow > div:not(.arrow-block) { min-height: 224px; padding: 26px 23px; border-top: 4px solid #91a19d; background: var(--paper); }
    .placement-flow > div.accent { border-color: var(--teal); background: var(--teal-soft); }
    .placement-flow span { display: block; color: var(--teal); font-size: 14px; font-weight: 700; margin-bottom: 31px; }
    .placement-flow strong { display: block; font-size: 25px; margin-bottom: 20px; }
    .placement-flow p { color: var(--muted); font-size: 16px; line-height: 1.65; }
    .placement-flow .arrow-block { color: var(--teal); text-align: center; font-size: 30px; }
    .placement-note { margin: 39px auto 0; width: fit-content; color: var(--muted); font-size: 17px; }
    .placement-note strong { color: var(--teal); }
    .sched-demo { height: calc(100% - 30px); }
    .pass-tabs { display: flex; gap: 7px; margin: -7px 0 27px; }
    .pass-tabs button { padding: 10px 19px; border: 1px solid var(--line); background: var(--paper); color: var(--muted); font-size: 15px; cursor: pointer; }
    .pass-tabs button.active { border-color: var(--teal); color: var(--teal); background: var(--teal-soft); font-weight: 650; }
    .sched-body { display: grid; grid-template-columns: minmax(600px, 1.8fr) minmax(285px, .8fr); gap: 34px; }
    .legend { margin-bottom: 29px; display: flex; gap: 22px; align-items: center; color: var(--muted); font-size: 14px; }
    .dot { width: 12px; height: 12px; display: inline-block; margin-right: 6px; vertical-align: middle; }
    .dot.tt { background: var(--teal); } .dot.cpu { background: var(--cpu); } .dot.pending { border: 1px solid #a8b3b0; } .dot.copy { background: var(--amber); }
    .node-chain { display: flex; flex-wrap: wrap; gap: 7px 5px; align-items: center; }
    .sched-node { padding: 14px 13px; min-width: 58px; text-align: center; border: 1px solid var(--line); font-size: 15px; font-weight: 600; }
    .sched-node.tt { color: var(--teal); border-color: var(--teal); background: var(--teal-soft); }
    .sched-node.cpu { color: var(--cpu); border-color: #99a5a8; background: var(--cpu-soft); }
    .sched-node.pending { color: #8d9795; background: var(--paper); }
    .chain-link { color: #aeb8b5; font-size: 18px; }
    .split-output { display: flex; flex-wrap: wrap; gap: 8px; min-height: 120px; margin-top: 48px; }
    .split { padding: 11px 13px; border-top: 3px solid var(--teal); background: var(--paper); font-size: 13px; }
    .split.cpu { border-color: var(--cpu); }
    .split b { display: block; margin-bottom: 6px; font-size: 14px; }
    .copy-tag { color: var(--amber); }
    .pass-text { padding: 17px 22px; border-left: 3px solid var(--teal); }
    .pass-text .eyebrow { color: var(--teal); font-size: 13px; font-weight: 700; margin-bottom: 20px; }
    .pass-text h3 { margin-bottom: 20px; font-size: 28px; }
    .pass-text p { color: var(--muted); font-size: 17px; line-height: 1.7; }
    .copy-path { display: flex; justify-content: center; align-items: center; gap: 0; margin-top: 86px; }
    .compute { width: 245px; min-height: 137px; display: grid; place-items: center; text-align: center; font-size: 18px; line-height: 1.7; border-top: 5px solid; background: var(--paper); }
    .compute.tt { border-color: var(--teal); } .compute.cpu { border-color: var(--cpu); }
    .copy-hop { position: relative; width: 122px; color: var(--amber); text-align: center; font-size: 14px; font-weight: 600; }
    .copy-hop::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; border-top: 2px solid var(--amber); z-index: -1; }
    .copy-hop { background: var(--bg); }
    .copy-reasons { display: flex; justify-content: center; gap: 15px; margin-top: 65px; }
    .copy-reasons span { padding: 12px 18px; border-top: 2px solid var(--line); color: var(--muted); font-size: 15px; }
    .allocation-pipeline { display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px; padding: 70px 55px 0; }
    .allocation-pipeline div { min-height: 260px; padding: 25px; border-top: 4px solid var(--teal); background: var(--paper); }
    .allocation-pipeline span { color: var(--teal); font-weight: 700; font-size: 16px; }
    .allocation-pipeline strong { display: block; margin: 42px 0 18px; font-size: clamp(25px, 2vw, 31px); }
    .allocation-pipeline p { color: var(--muted); font-size: 16px; line-height: 1.7; }
    .backend-layers { display: flex; justify-content: center; gap: 22px; padding-top: 45px; }
    .backend-layers div { width: 29%; min-height: 235px; padding: 24px 22px; border-top: 4px solid var(--teal); background: var(--paper); }
    .backend-layers span { color: var(--teal); font-weight: 700; }
    .backend-layers strong { display: block; margin: 38px 0 18px; font-size: 28px; }
    .backend-layers p { color: var(--muted); font-size: 17px; line-height: 1.7; }
    .dtype-line { display: flex; align-items: center; gap: 19px; width: fit-content; margin: 42px auto 0; font-size: 16px; }
    .dtype-line span { padding: 9px 13px; background: var(--teal-soft); color: var(--teal); }
    .dtype-line small { color: var(--muted); }
    .support-table { display: grid; max-width: 1060px; margin: 53px auto 0; font-size: 17px; }
    .support-table > div { display: grid; grid-template-columns: 1.35fr 1fr 1fr; gap: 17px; align-items: center; min-height: 74px; padding: 0 22px; border-bottom: 1px solid var(--line); }
    .support-table .support-head { min-height: 50px; color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }
    .support-table strong { color: var(--teal); }
    .support-table em { font-style: normal; color: var(--muted); }
    .tt-stack { display: flex; align-items: center; justify-content: center; gap: 26px; padding-top: 49px; }
    .tt-stack > span { color: var(--teal); font-size: 30px; }
    .incoming { min-width: 190px; padding: 29px 20px; text-align: center; background: var(--paper); border-top: 4px solid var(--purple); color: var(--muted); line-height: 1.7; }
    .incoming strong { color: var(--ink); font-size: 22px; }
    .incoming.chip { border-color: var(--teal); }
    .stack-layers { display: grid; gap: 8px; min-width: 375px; }
    .stack-layers div { display: flex; align-items: center; gap: 29px; padding: 17px 22px; background: var(--teal-soft); border-left: 4px solid var(--teal); }
    .stack-layers strong { min-width: 125px; font-size: 24px; color: var(--teal); }
    .stack-layers small { color: var(--muted); font-size: 15px; }
    .hardware-feature { display: grid; grid-template-columns: 1.32fr .68fr; gap: 45px; height: 68%; align-items: center; }
    .hardware-feature img { width: 100%; height: 100%; object-fit: contain; background: var(--paper); border: 1px solid var(--line); }
    .hardware-feature strong { display: block; color: var(--teal); font-size: 26px; margin-bottom: 32px; }
    .hardware-feature p { margin-bottom: 25px; color: var(--muted); font-size: 18px; line-height: 1.7; }
    .hardware-duo { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; height: 65%; }
    .hardware-duo figure { margin: 0; display: grid; grid-template-rows: 1fr auto; min-height: 0; }
    .hardware-duo img { width: 100%; height: 100%; min-height: 0; object-fit: contain; background: var(--paper); border: 1px solid var(--line); }
    .hardware-duo figcaption { padding: 10px 2px; color: var(--muted); font-size: 15px; }
    .noc-layout { display: grid; grid-template-columns: 1.18fr .82fr; align-items: center; gap: 48px; height: 65%; }
    .noc-layout img { width: 100%; height: 100%; object-fit: contain; background: var(--paper); border: 1px solid var(--line); }
    .layout-conversion { display: grid; gap: 16px; }
    .layout-conversion span { color: var(--teal); font-weight: 700; font-size: 15px; }
    .layout-conversion strong { padding: 16px; background: var(--paper); font-size: 21px; }
    .layout-conversion div { color: var(--teal); text-align: center; font-size: 29px; }
    .checks { max-width: 970px; margin: 45px auto 0; display: grid; grid-template-columns: 1fr 1fr; gap: 13px; }
    .checks div { display: flex; align-items: center; min-height: 86px; padding: 16px 22px; background: var(--paper); border-left: 4px solid var(--teal); font-size: clamp(18px, 1.4vw, 22px); }
    .checks span { margin-right: 18px; color: var(--teal); font-weight: 700; }
    .landing { margin: 47px auto 0; text-align: center; font-size: clamp(28px, 2.6vw, 40px); line-height: 1.5; }
    .landing strong { display: block; }
    .reading-link { display: block; width: fit-content; margin: 33px auto; color: var(--teal); font-size: 17px; text-decoration: none; }
    .appendix-table { display: grid; max-width: 1120px; margin: 52px auto 0; font-size: 16px; }
    .appendix-table > div { display: grid; grid-template-columns: 1.05fr .95fr 1.7fr 1fr; gap: 14px; align-items: center; min-height: 72px; padding: 0 20px; border-bottom: 1px solid var(--line); }
    .appendix-table .table-head { min-height: 48px; color: var(--muted); font-size: 13px; font-weight: 700; }
    .appendix-table .highlight { background: var(--teal-soft); }
    .appendix-table strong { color: var(--teal); font-size: 20px; }
    .struct-example { display: grid; grid-template-columns: 1.15fr .85fr; gap: 55px; align-items: center; height: 68%; padding: 0 62px; }
    .struct-example pre { margin: 0; padding: 30px; background: var(--paper); border-left: 4px solid var(--teal); color: #334347; font: clamp(16px, 1.3vw, 20px)/1.8 Consolas, monospace; }
    .struct-example strong { display: block; margin-bottom: 30px; color: var(--teal); font-size: 25px; }
    .struct-example p { margin-bottom: 17px; padding: 11px 16px; background: var(--paper); font: 18px Consolas, monospace; }
    .pass-summary { display: grid; max-width: 940px; margin: 48px auto 0; gap: 8px; }
    .pass-summary div { display: flex; align-items: center; gap: 22px; min-height: 61px; padding: 0 22px; background: var(--paper); font-size: 19px; }
    .pass-summary span { display: grid; place-items: center; width: 28px; height: 28px; background: var(--teal-soft); color: var(--teal); font-weight: 700; }
    .pass-summary .accent { border-left: 4px solid var(--teal); background: var(--teal-soft); }
    .implementation-boundary { display: grid; grid-template-columns: 1fr 64px 1fr; align-items: center; gap: 21px; max-width: 1020px; margin: 68px auto 46px; }
    .implementation-boundary div { min-height: 188px; padding: 30px; background: var(--paper); border-top: 4px solid var(--teal); }
    .implementation-boundary strong { display: block; color: var(--teal); font-size: 24px; margin-bottom: 28px; }
    .implementation-boundary p { color: var(--muted); font-size: 18px; line-height: 1.65; }
    .implementation-boundary > span { color: var(--amber); font-size: 35px; text-align: center; }
    .implementation-note { max-width: 1020px; margin: 0 auto; padding-top: 19px; border-top: 2px solid var(--line); color: var(--muted); font-size: 17px; line-height: 1.65; }
    .sources { max-width: 930px; margin: 40px auto 0; font-size: clamp(19px, 1.45vw, 23px); line-height: 2.15; }
    .sources span { display: inline-block; width: 100px; color: var(--teal); font-weight: 700; }
    .appendix-links { display: flex; gap: 22px; margin: 58px auto 0; width: fit-content; }
    .appendix-links a { padding: 14px 23px; border: 1px solid var(--teal); color: var(--teal); text-decoration: none; }
    .controls { position: absolute; right: 22px; bottom: 17px; display: flex; align-items: center; gap: 8px; z-index: 12; }
    .controls button { width: 40px; height: 36px; border: 1px solid var(--line); background: rgba(255,255,255,.92); color: var(--ink); cursor: pointer; font-size: 18px; }
    .controls button:hover { border-color: var(--teal); color: var(--teal); }
    .progress { position: absolute; left: 0; bottom: 0; height: 4px; background: var(--teal); width: 0; transition: width .2s ease; z-index: 11; }
    .help { position: absolute; left: 50%; transform: translateX(-50%); bottom: 18px; color: var(--muted); font-size: 12px; z-index: 12; }
    .help a { color: var(--teal); }
    @media (max-aspect-ratio: 4/3) {
      .deck { width: 100vw; height: auto; }
    }
  </style>
</head>
<body>
  <main class="deck" id="deck">
    ${slides.map(renderSlide).join("\n")}
    <div class="help">← → 翻页　F 全屏　<a href="tenstorrent-ggml-scheduler-speaker-notes.html">讲稿</a></div>
    <div class="controls"><button id="prev" aria-label="上一页">‹</button><button id="next" aria-label="下一页">›</button></div>
    <div class="progress" id="progress"></div>
  </main>
  <script>
    (function () {
      var slides = Array.from(document.querySelectorAll(".slide"));
      var current = 0;
      function indexFromHash() {
        var match = location.hash.match(/^#slide-(\\d+)$/);
        if (!match) return 0;
        return Math.max(0, Math.min(slides.length - 1, Number(match[1]) - 1));
      }
      function show(index, updateHash) {
        current = Math.max(0, Math.min(slides.length - 1, index));
        slides.forEach(function (slide, i) {
          var active = i === current;
          slide.classList.toggle("active", active);
          slide.setAttribute("aria-hidden", active ? "false" : "true");
        });
        document.getElementById("progress").style.width = ((current + 1) / slides.length * 100) + "%";
        if (updateHash) history.replaceState(null, "", "#slide-" + String(current + 1).padStart(2, "0"));
      }
      function next() { show(current + 1, true); }
      function prev() { show(current - 1, true); }
      document.getElementById("next").addEventListener("click", next);
      document.getElementById("prev").addEventListener("click", prev);
      document.addEventListener("keydown", function (event) {
        if (event.key === "ArrowRight" || event.key === "PageDown" || event.key === " ") { event.preventDefault(); next(); }
        if (event.key === "ArrowLeft" || event.key === "PageUp") { event.preventDefault(); prev(); }
        if (event.key.toLowerCase() === "f" && document.documentElement.requestFullscreen) { document.documentElement.requestFullscreen(); }
        if (event.key === "Home") show(0, true);
        if (event.key === "End") show(slides.length - 1, true);
      });
      window.addEventListener("hashchange", function () { show(indexFromHash(), false); });
      show(indexFromHash(), !location.hash);

      var routeDemo = document.querySelector("[data-route-demo]");
      if (routeDemo) {
        var routes = {
          compiler: {
            image: "pics/ttSDK_path_compiler_first.png",
            alt: "Compiler-first 进入 Tenstorrent 软件栈的路径",
            label: "Compiler-first",
            title: "图转换和 lowering 先行",
            copy: "框架模型经过 Forge / MLIR lowering，最终落到 TT-NN 与 Metalium。",
            focus: "关注点：图覆盖、layout legalization、codegen"
          },
          pytorch: {
            image: "pics/ttSDK_path_pytorch_backend.png",
            alt: "PyTorch backend 进入 Tenstorrent 软件栈的路径",
            label: "PyTorch backend",
            title: "前端图捕获后交给设备后端",
            copy: "PyTorch 2.0 backend 捕获和分发图，后续可进入 TT-MLIR 或算子执行路径。",
            focus: "关注点：捕获边界、fallback、算子覆盖"
          },
          vllm: {
            image: "pics/ttSDK_path_runtime_vllm.png",
            alt: "vLLM runtime 进入 Tenstorrent 软件栈的路径",
            label: "vLLM runtime",
            title: "服务调度和 KV cache 在前",
            copy: "请求批处理、KV cache 和生成调度由 runtime 组织，设备侧承接实际算子执行。",
            focus: "关注点：吞吐、cache、prefill / decode 调度"
          },
          ggml: {
            image: "pics/ttSDK_path_ggml_llamacpp.png",
            alt: "llama.cpp 和 ggml 进入 Tenstorrent 软件栈的路径",
            label: "llama.cpp / ggml",
            title: "本地加载、量化与多 backend 执行",
            copy: "GGUF、ggml_cgraph 与 scheduler 保留了真实推理执行路径，适合观察新 backend 适配。",
            focus: "本文路径：类型、split、copy、设备执行"
          },
          manual: {
            image: "pics/ttSDK_path_hand_optimized.png",
            alt: "手工优化模型进入 Tenstorrent 软件栈的路径",
            label: "Hand-optimized",
            title: "固定关键路径直接优化",
            copy: "针对少数模型或热点 kernel，直接组织算子与低层实现换取定向性能。",
            focus: "关注点：可控性能与较低通用性"
          }
        };
        function setRoute(routeKey) {
          var route = routes[routeKey];
          routeDemo.querySelector("[data-route-image]").src = route.image;
          routeDemo.querySelector("[data-route-image]").alt = route.alt;
          routeDemo.querySelector("[data-route-label]").textContent = route.label;
          routeDemo.querySelector("[data-route-title]").textContent = route.title;
          routeDemo.querySelector("[data-route-copy]").textContent = route.copy;
          routeDemo.querySelector("[data-route-focus]").textContent = route.focus;
          routeDemo.querySelectorAll("[data-route]").forEach(function (button) {
            button.classList.toggle("active", button.dataset.route === routeKey);
          });
        }
        routeDemo.querySelectorAll("[data-route]").forEach(function (button) {
          button.addEventListener("click", function () { setRoute(button.dataset.route); });
        });
        setRoute("ggml");
      }

      var ggufDemo = document.querySelector("[data-gguf-demo]");
      if (ggufDemo) {
        var blobStates = {
          header: ["文件入口", "先知道目录规模", "magic、版本、tensor 数量和 metadata 数量先确定后续读取结构。"],
          meta: ["模型级信息", "runtime 取得架构与 tokenizer", "KV metadata 放入模型属性和 tokenizer 等读取模型所需的信息。"],
          directory: ["逐 tensor 描述", "这里出现 ggml_type", "名称、shape、ggml_type 与 data_offset 决定每段权重怎样被找到和解释。"],
          bytes: ["真实数据区", "按 alignment 读取 bytes", "data_offset 指向对齐后的二进制权重区；量化编码实际保存在这里。"]
        };
        function setBlob(key) {
          var state = blobStates[key];
          ggufDemo.querySelectorAll("[data-blob-tab]").forEach(function (button) {
            button.classList.toggle("active", button.dataset.blobTab === key);
          });
          ggufDemo.querySelectorAll("[data-blob]").forEach(function (block) {
            block.classList.toggle("active", block.dataset.blob === key);
          });
          ggufDemo.querySelector("[data-blob-label]").textContent = state[0];
          ggufDemo.querySelector("[data-blob-title]").textContent = state[1];
          ggufDemo.querySelector("[data-blob-copy]").textContent = state[2];
        }
        ggufDemo.querySelectorAll("[data-blob-tab]").forEach(function (button) {
          button.addEventListener("click", function () { setBlob(button.dataset.blobTab); });
        });
        setBlob("header");
      }

      var cgraphDemo = document.querySelector("[data-cgraph-demo]");
      if (cgraphDemo) {
        var graphStates = {
          model: "<strong>先看路径：</strong>一个 decoder block 输出 logits 的简化计算链。",
          ops: "<strong>再看 op：</strong>中间结果 tensor 同时记录产生它的 ADD、MUL_MAT、RMS_NORM 等操作。",
          inputs: "<strong>最后看 src：</strong>x1.op = ADD，x1.src = [ x0, attn_out ]，箭头表示输入来源。"
        };
        function setCgraph(stage) {
          var svg = cgraphDemo.querySelector("[data-cgraph-svg]");
          svg.classList.remove("stage-model", "stage-ops", "stage-inputs");
          svg.classList.add("stage-" + stage);
          cgraphDemo.querySelector("[data-cgraph-copy]").innerHTML = graphStates[stage];
          cgraphDemo.querySelectorAll("[data-cgraph-stage]").forEach(function (button) {
            button.classList.toggle("active", button.dataset.cgraphStage === stage);
          });
        }
        cgraphDemo.querySelectorAll("[data-cgraph-stage]").forEach(function (button) {
          button.addEventListener("click", function () { setCgraph(button.dataset.cgraphStage); });
        });
        setCgraph("model");
      }

      var demo = document.querySelector("[data-scheduler-demo]");
      if (demo) {
      var states = [
        { label: "输入图", title: "只有部分 tensor 已有位置", copy: "权重或输入可能已经在某个 buffer 上，中间结果还没有 assignment。", map: {} },
        { label: "Pass 1 / 初始分配", title: "从已有 buffer 找锚点", copy: "已有权重位置先固定若干计算路径，作为后续扩展的起点。", map: { x0:"cpu", qkv:"tt", attn:"tt", ffn:"tt", logits:"tt" } },
        { label: "Pass 2 / 扩展", title: "沿相邻可执行节点传播", copy: "能通过 supports_op 的相邻节点尽量跟随已确定的 backend。", map: { x0:"cpu", n0:"cpu", qkv:"tt", score:"tt", attn:"tt", x1:"tt", ffn:"tt", x2:"tt", logits:"tt" } },
        { label: "Pass 3 / 补漏", title: "剩余节点选择可执行 backend", copy: "未分配节点依据可兼容输入数量选择 backend；这是启发式选择。", map: { x0:"cpu", n0:"cpu", qkv:"tt", score:"tt", prob:"cpu", attn:"tt", x1:"tt", n1:"cpu", ffn:"tt", x2:"tt", logits:"tt" } },
        { label: "Pass 4 / 输入归属", title: "补齐 src 与 view 关系", copy: "没有显式归属的输入跟随消费它的节点；view 与原始 tensor 保持同位置。", map: { x0:"cpu", n0:"cpu", qkv:"tt", score:"tt", prob:"cpu", attn:"tt", x1:"tt", n1:"cpu", ffn:"tt", x2:"tt", logits:"tt" } },
        { label: "Pass 5 / split + copy", title: "backend 变化形成执行段", copy: "跨 backend 且 buffer 不兼容的输入会登记 copy tensor。", map: { x0:"cpu", n0:"cpu", qkv:"tt", score:"tt", prob:"cpu", attn:"tt", x1:"tt", n1:"cpu", ffn:"tt", x2:"tt", logits:"tt" },
          splits: [
            ["CPU", "x0, n0"], ["TT", "qkv, score", "copy n0"], ["CPU", "prob", "copy score"],
            ["TT", "attn, x1", "copy prob"], ["CPU", "n1", "copy x1"], ["TT", "ffn, x2, logits", "copy n1"]
          ] }
      ];
      function setPass(pass) {
        var state = states[pass];
        demo.querySelectorAll(".pass-tabs button").forEach(function (button, i) { button.classList.toggle("active", i === pass); });
        demo.querySelectorAll(".sched-node").forEach(function (node) {
          node.classList.remove("tt", "cpu", "pending");
          node.classList.add(state.map[node.dataset.node] || "pending");
        });
        demo.querySelector("[data-pass-label]").textContent = state.label;
        demo.querySelector("[data-pass-title]").textContent = state.title;
        demo.querySelector("[data-pass-copy]").textContent = state.copy;
        var splitOut = demo.querySelector("[data-splits]");
        splitOut.innerHTML = state.splits ? state.splits.map(function (item) {
          return '<div class="split ' + item[0].toLowerCase() + '"><b>' + item[0] + '</b>' + item[1] + (item[2] ? '<div class="copy-tag">' + item[2] + '</div>' : '') + '</div>';
        }).join("") : "";
      }
      demo.querySelectorAll(".pass-tabs button").forEach(function (button) {
        button.addEventListener("click", function () { setPass(Number(button.dataset.pass)); });
      });
      setPass(0);
      }
    })();
  </script>
</body>
</html>`;
}

function renderNotes() {
  const total = slides.reduce((sum, slide) => {
    const [minutes, seconds] = slide.duration.split(":").map(Number);
    return sum + minutes * 60 + seconds;
  }, 0);
  const totalText = `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>从模型到芯片执行 - 演讲稿</title>
  <style>
    :root { --ink:#162124; --muted:#59686b; --teal:#087e78; --line:#d5ddda; --soft:#f3f6f5; font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif; letter-spacing:0; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#fff; line-height:1.65; }
    main { max-width:1060px; margin:0 auto; padding:46px 34px 70px; }
    h1 { margin:0 0 12px; font-size:38px; line-height:1.3; }
    .intro { margin:0 0 40px; color:var(--muted); font-size:17px; }
    .toolbar { display:flex; gap:14px; margin-bottom:38px; }
    .toolbar a { color:var(--teal); border:1px solid var(--teal); padding:9px 15px; text-decoration:none; }
    .overview { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:0 0 42px; }
    .overview div { padding:16px; border-top:3px solid var(--teal); background:var(--soft); }
    .overview span { display:block; color:var(--muted); font-size:13px; }
    .overview strong { font-size:21px; }
    .note { margin:0 0 26px; padding:22px 26px; border:1px solid var(--line); page-break-inside:avoid; }
    .note-head { display:flex; justify-content:space-between; gap:16px; margin-bottom:18px; }
    .number { color:var(--teal); font-size:14px; font-weight:700; }
    h2 { margin:4px 0 0; font-size:23px; }
    .time { color:var(--muted); font-size:14px; white-space:nowrap; }
    .script-grid { display:grid; grid-template-columns:100px 1fr; gap:10px 16px; font-size:16px; }
    .label { color:var(--teal); font-weight:650; }
    .line { color:var(--ink); }
    ul { margin:0; padding-left:19px; }
    li { margin:3px 0; }
    .transition { color:#435357; }
    @media print { main { padding:0; } .toolbar { display:none; } .note { border-color:#bbb; } }
    @media (max-width:720px) { main { padding:25px 17px; } .overview { grid-template-columns:1fr 1fr; } .script-grid { grid-template-columns:1fr; gap:3px; } }
  </style>
</head>
<body>
  <main>
    <h1>从模型到芯片执行｜演讲稿</h1>
    <p class="intro">与 HTML slides 逐页对应。句子刻意保持短、口语化，现场按节奏取用即可。</p>
    <div class="toolbar"><a href="tenstorrent-ggml-scheduler-slides.html">打开幻灯片</a><a href="tenstorrent-ggml-scheduler-integrated-report.html">打开完整阅读版</a></div>
    <div class="overview">
      <div><span>页面</span><strong>${slides.length} 页</strong></div>
      <div><span>预计主讲</span><strong>${totalText}</strong></div>
      <div><span>听众</span><strong>混合技术听众</strong></div>
      <div><span>主线</span><strong>文件 → 图 → 设备</strong></div>
    </div>
    ${slides
      .map(
        (slide, index) => `<section class="note" id="note-${String(index + 1).padStart(2, "0")}">
      <div class="note-head"><div><div class="number">${String(index + 1).padStart(2, "0")} / ${slides.length}</div><h2>${esc(slide.title)}</h2></div><div class="time">${slide.duration}</div></div>
      <div class="script-grid">
        <div class="label">开口</div><div class="line">${esc(slide.notes.opening)}</div>
        <div class="label">指图</div><div class="line"><ul>${slide.notes.points.map((point) => `<li>${esc(point)}</li>`).join("")}</ul></div>
        <div class="label">转场</div><div class="line transition">${esc(slide.notes.transition)}</div>
      </div>
    </section>`,
      )
      .join("")}
  </main>
</body>
</html>`;
}

fs.writeFileSync(deckPath, renderDeck(), "utf8");
fs.writeFileSync(notesPath, renderNotes(), "utf8");
console.log(`Generated ${path.basename(deckPath)}`);
console.log(`Generated ${path.basename(notesPath)}`);
