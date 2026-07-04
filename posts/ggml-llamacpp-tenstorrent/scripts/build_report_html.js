const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const inputPath = path.join(root, "tenstorrent-ggml-scheduler-integrated-report.md");
const outputPath = path.join(root, "tenstorrent-ggml-scheduler-integrated-report.html");
const source = fs.readFileSync(inputPath, "utf8").replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function slugify(text) {
  return text
    .replace(/[：、，。/（）()[\]]/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
}

function hrefFor(rawHref) {
  if (/^[A-Za-z]:\//.test(rawHref)) {
    return `file:///${rawHref}`;
  }
  return rawHref;
}

function inline(text) {
  let rendered = escapeHtml(text);
  rendered = rendered.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    const target = hrefFor(href);
    const external = /^https?:\/\//.test(target) ? ' target="_blank" rel="noreferrer"' : "";
    return `<a href="${escapeHtml(target)}"${external}>${label}</a>`;
  });
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  rendered = rendered.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  rendered = rendered.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (_, numbers) => {
    const refs = numbers.split(",").map((value) => value.trim());
    return `<span class="citation">[${refs.map((ref) => `<a href="#ref-${ref}">${ref}</a>`).join(", ")}]</span>`;
  });
  return rendered;
}

function tabs(items, targetClass) {
  return `<div class="tabs" role="tablist">${items
    .map(
      (item, index) =>
        `<button type="button" role="tab" class="${index === 0 ? "active" : ""}" data-target="${targetClass}-${item.key}" aria-selected="${index === 0}">${item.label}</button>`,
    )
    .join("")}</div>`;
}

function softwareRoutes() {
  const routes = [
    {
      key: "compiler",
      label: "Compiler-first",
      title: "通用模型图编译入口",
      src: "pics/ttSDK_path_compiler_first.png",
      text: "PyTorch、ONNX 或 JAX 模型经前端与 TT-MLIR lowering 进入 Tenstorrent 执行栈。",
    },
    {
      key: "pytorch",
      label: "PyTorch backend",
      title: "PyTorch 原生 backend 入口",
      src: "pics/ttSDK_path_pytorch_backend.png",
      text: "PyTorch 2.0 backend 捕获并分发图，后续落到 TT-MLIR 或 TT-NN。",
    },
    {
      key: "vllm",
      label: "vLLM runtime",
      title: "服务型推理 runtime 入口",
      src: "pics/ttSDK_path_runtime_vllm.png",
      text: "vLLM 或推理服务负责请求调度与 KV cache，设备计算由 Tenstorrent 栈承接。",
    },
    {
      key: "ggml",
      label: "llama.cpp / ggml",
      title: "本地量化推理 backend 入口",
      src: "pics/ttSDK_path_ggml_llamacpp.png",
      text: "llama.cpp/ggml 保留模型加载、量化和 scheduler 逻辑，Tenstorrent 实现新的硬件 backend。",
    },
    {
      key: "manual",
      label: "Hand-optimized",
      title: "手工优化入口",
      src: "pics/ttSDK_path_hand_optimized.png",
      text: "固定模型或关键 kernel 可直接组织 TT-NN、TT-Metalium 与 TT-LLK 调用。",
    },
  ];
  return `
<section class="visual wide tab-component route-component" data-tab-component>
  <div class="visual-head">
    <div>
      <div class="visual-label">交互图</div>
      <h4>同一底层栈上的不同接入路径</h4>
    </div>
    ${tabs(routes, "route")}
  </div>
  ${routes
    .map(
      (route, index) => `
  <figure class="route-panel tab-panel ${index === 0 ? "active" : ""}" id="route-${route.key}">
    <img src="${route.src}" alt="${route.title}">
    <figcaption><strong>${route.title}</strong> ${route.text}</figcaption>
  </figure>`,
    )
    .join("")}
</section>`;
}

function binaryDiagram(kind) {
  const isSafe = kind === "safetensors";
  const title = isSafe ? "model.safetensors 的文件布局" : "GGUF 的文件布局";
  const label = isSafe ? "safetensors" : "GGUF";
  const rows = isSafe
    ? [
        ["Header length", "8-byte little-endian integer", "指出后续 JSON header 的字节长度。"],
        ["JSON header", "metadata + tensor descriptions", "保存名称、dtype、shape 与 data_offsets。"],
        ["Tensor data bytes", "[tensor_0][tensor_1][tensor_2] ...", "data_offsets 指向这段连续数据区。"],
      ]
    : [
        ["GGUF header", 'magic = "GGUF" | version | n_tensors | n_kv', "识别文件并给出目录规模。"],
        ["KV metadata", "architecture | tokenizer | model properties", "保存模型级属性与 tokenizer 信息。"],
        ["Tensor directory", "name | shape | ggml_type | data_offset", "逐 tensor 描述实际存储类型和位置。"],
        ["Aligned tensor blob", "[tensor_0][pad][tensor_1] ...", "data_offset 指向按 alignment 放置的数据区。"],
      ];
  return `
<figure class="visual binary-component" data-binary>
  <div class="visual-label">结构图 · ${label}</div>
  <h4>${title}</h4>
  <div class="binary-grid">
    <div class="binary-stack" role="list">
      ${rows
        .map(
          (row, index) => `<button type="button" class="binary-block ${index === 0 ? "active" : ""}" data-description="${escapeHtml(row[2])}">
        <strong>${row[0]}</strong><span>${row[1]}</span>
      </button>`,
        )
        .join("")}
    </div>
    <div class="binary-callout">
      <span class="callout-title">选中字段</span>
      <p>${rows[0][2]}</p>
      <div class="offset-arrow">${isSafe ? "data_offsets -> tensor data bytes" : "data_offset -> aligned tensor blob"}</div>
    </div>
  </div>
</figure>`;
}

function toyGraph() {
  const rows = [
    ["x0", "GET_ROWS", "tok_embeddings, tokens"],
    ["n0", "RMS_NORM", "x0, attn_norm"],
    ["q / k / v", "MUL_MAT", "wq / wk / wv, n0"],
    ["attn_out", "ATTENTION path", "q, k, v"],
    ["x1", "ADD", "x0, attn_out"],
    ["n1", "RMS_NORM", "x1, ffn_norm"],
    ["ffn_out", "FFN path", "w_gate / w_up / w_down, n1"],
    ["x2", "ADD", "x1, ffn_out"],
    ["logits", "MUL_MAT", "lm_head, x2"],
  ];
  return `
<section class="visual wide tab-component cgraph-component" data-tab-component>
  <div class="visual-head">
    <div>
      <div class="visual-label">Toy decoder block</div>
      <h4>模型数据流与 ggml 记录方式</h4>
    </div>
    ${tabs(
      [
        { key: "flow", label: "依赖图" },
        { key: "records", label: "ggml 记录" },
      ],
      "cgraph",
    )}
  </div>
  <div class="tab-panel active graph-panel" id="cgraph-flow">
    <svg class="cgraph-svg" viewBox="0 0 1020 760" role="img" aria-label="Toy decoder block 的 tensor 和 op 依赖图">
      <defs>
        <marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z"></path>
        </marker>
      </defs>
      <g class="flow-lines">
        <path d="M155 96 V125"></path><path d="M155 181 V210"></path>
        <path d="M260 240 H390"></path><path d="M525 168 V210"></path><path d="M525 266 V300"></path>
        <path d="M525 356 V372 H560 V400"></path><path d="M155 181 V372 H490 V400"></path>
        <path d="M525 456 V485"></path><path d="M640 515 H680"></path><path d="M815 443 V485"></path>
        <path d="M815 541 V552 H850 V580"></path><path d="M525 456 V552 H780 V580"></path>
        <path d="M815 636 V665"></path><path d="M590 693 H680"></path>
      </g>
      <g class="node source"><rect x="50" y="40" width="210" height="56"></rect><text x="155" y="75">tokens + tok_embeddings</text></g>
      <g class="node"><rect x="50" y="125" width="210" height="56"></rect><text x="155" y="148">x0</text><text class="op" x="155" y="168">GET_ROWS / EMBED</text></g>
      <g class="node"><rect x="50" y="210" width="210" height="56"></rect><text x="155" y="233">n0</text><text class="op" x="155" y="253">RMS_NORM</text></g>
      <g class="node weight"><rect x="405" y="120" width="240" height="48"></rect><text x="525" y="150">wq / wk / wv</text></g>
      <g class="node accent"><rect x="390" y="210" width="270" height="56"></rect><text x="525" y="233">q, k, v</text><text class="op" x="525" y="253">MUL_MAT</text></g>
      <g class="node accent"><rect x="390" y="300" width="270" height="56"></rect><text x="525" y="323">attn_out</text><text class="op" x="525" y="343">ATTENTION path</text></g>
      <g class="node"><rect x="390" y="400" width="270" height="56"></rect><text x="525" y="423">x1</text><text class="op" x="525" y="443">ADD + residual</text></g>
      <g class="node"><rect x="390" y="485" width="250" height="56"></rect><text x="515" y="508">n1</text><text class="op" x="515" y="528">RMS_NORM</text></g>
      <g class="node weight"><rect x="695" y="395" width="240" height="48"></rect><text x="815" y="425">w_gate / w_up / w_down</text></g>
      <g class="node accent"><rect x="680" y="485" width="270" height="56"></rect><text x="815" y="508">ffn_out</text><text class="op" x="815" y="528">FFN path</text></g>
      <g class="node"><rect x="680" y="580" width="270" height="56"></rect><text x="815" y="603">x2</text><text class="op" x="815" y="623">ADD + residual</text></g>
      <g class="node weight"><rect x="420" y="669" width="170" height="48"></rect><text x="505" y="697">lm_head</text></g>
      <g class="node final"><rect x="680" y="665" width="270" height="56"></rect><text x="815" y="688">logits</text><text class="op" x="815" y="708">MUL_MAT</text></g>
    </svg>
    <p class="diagram-note">方框表示结果 tensor 与其对应 op；连线表示输入来源。图中省略 RoPE、mask、KV cache 与 layout 细节。</p>
  </div>
  <div class="tab-panel" id="cgraph-records">
    <div class="table-wrap compact-table">
      <table>
        <thead><tr><th>result tensor</th><th>op</th><th>src：输入来源指针</th></tr></thead>
        <tbody>
          ${rows.map((row) => `<tr><td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td></tr>`).join("")}
        </tbody>
      </table>
    </div>
    <p class="diagram-note">每个结果 tensor 自己携带 op 和 src；ggml 从 logits 沿 src 反向追溯，收集成 ggml_cgraph。</p>
  </div>
</section>`;
}

function runtimePipeline() {
  const steps = [
    ["模型文件", "GGUF"],
    ["读取信息", "load_arch / hparams / vocab / tensors"],
    ["权重对象", "ggml weight tensors"],
    ["建图", "model.build_graph"],
    ["架构路径", "architecture-specific builder"],
    ["运行时图", "ggml_cgraph"],
    ["执行入口", "scheduler / backend"],
  ];
  return `
<figure class="visual wide pipeline">
  <div class="visual-label">执行链路</div>
  <h4>模型文件到 backend 入口</h4>
  <div class="pipeline-track">
    ${steps
      .map(
        (step, index) => `<div class="pipe-step"><span>${step[0]}</span><strong>${step[1]}</strong></div>${
          index < steps.length - 1 ? '<div class="pipe-arrow">-></div>' : ""
        }`,
      )
      .join("")}
  </div>
</figure>`;
}

function schedulerFrame() {
  return `
<section class="visual scheduler-frame wide">
  <div class="visual-label">交互图 · split_graph</div>
  <h4>同一 toy cgraph 上的五阶段 assignment 与 split</h4>
  <p class="diagram-note">组件中的 TT/CPU 支持组合用于解释 scheduler 规则，不表示任何具体 Tenstorrent backend 版本完整支持或拒绝这些 op。</p>
  <iframe src="pics/split_graph_toy_cgraph_visualizer.html" title="split_graph 五阶段交互演示" loading="lazy"></iframe>
</section>`;
}

function allocationPipeline() {
  const blocks = [
    ["cgraph", "权重 / KV cache / graph input / 中间激活"],
    ["split_graph", "backend 优先级 | supports_op | buffer 归属 | split + copy"],
    ["ggml_gallocr", "按 graph 分配 | 引用计数复用 | inplace 复用"],
    ["tensor -> buffer", "执行前空间就位"],
  ];
  return `
<figure class="visual allocation-pipeline">
  <div class="visual-label">调度与分配</div>
  <h4>在哪算，与空间怎么划</h4>
  <div class="allocation-track">
    ${blocks
      .map(
        (block, index) => `<div class="allocation-step"><strong>${block[0]}</strong><span>${block[1]}</span></div>${
          index < blocks.length - 1 ? '<div class="vertical-arrow">down</div>' : ""
        }`,
      )
      .join("")}
  </div>
</figure>`;
}

function validationChecklist() {
  return `
<div class="verification-list" aria-label="执行路径验证清单">
  <div><span>01</span>当前 assignment 由 backend 顺序和 tensor buffer 归属怎样决定</div>
  <div><span>02</span>每个 supports_op 返回 true 或 false 的原因：dtype、shape、layout</div>
  <div><span>03</span>当前 split 的跨 backend copy tensor 列表及其来源</div>
  <div><span>04</span>graph_compute 能否完整执行分配给 TT 的 split</div>
</div>`;
}

function readFence(lines, start) {
  const language = lines[start].slice(3).trim();
  const content = [];
  let index = start + 1;
  while (index < lines.length && !lines[index].startsWith("```")) {
    content.push(lines[index]);
    index += 1;
  }
  return { language, content: content.join("\n"), next: index + 1 };
}

function renderTable(lines, start) {
  const tableLines = [];
  let index = start;
  while (index < lines.length && /^\|.*\|\s*$/.test(lines[index])) {
    tableLines.push(lines[index]);
    index += 1;
  }
  const cells = tableLines.map((line) =>
    line
      .slice(1, -1)
      .split("|")
      .map((cell) => cell.trim()),
  );
  const head = cells[0];
  const body = cells.slice(2);
  return {
    next: index,
    html: `<div class="table-wrap"><table><thead><tr>${head.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead><tbody>${body
      .map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`)
      .join("")}</tbody></table></div>`,
  };
}

function renderMarkdown(markdown) {
  const lines = markdown.split("\n");
  const output = [];
  const headings = [];
  let index = 0;
  let skipDirectory = false;
  let inReferences = false;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const title = heading[2].trim();
      if (level === 2 && title === "目录") {
        skipDirectory = true;
        index += 1;
        continue;
      }
      if (skipDirectory && level !== 2) {
        index += 1;
        continue;
      }
      if (skipDirectory && level === 2) {
        skipDirectory = false;
      }
      const id = slugify(title);
      if (level > 1) {
        headings.push({ level, title, id });
      }
      if (title === "参考文献") {
        inReferences = true;
      }
      output.push(`<h${level} id="${id}">${inline(title)}${level > 1 ? `<a class="anchor" href="#${id}" aria-label="链接到本节">#</a>` : ""}</h${level}>`);
      index += 1;
      continue;
    }

    if (skipDirectory) {
      index += 1;
      continue;
    }

    if (line.startsWith("<p align=\"center\">")) {
      const raw = [line];
      index += 1;
      while (index < lines.length && !lines[index].includes("</p>")) {
        raw.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        raw.push(lines[index]);
        index += 1;
      }
      const image = raw.join("\n").match(/<img src="([^"]+)" alt="([^"]+)" width="([^"]+)">/);
      let caption = "";
      while (index < lines.length && !lines[index].trim()) {
        index += 1;
      }
      if (index < lines.length && /^\*图：.*\*$/.test(lines[index])) {
        caption = lines[index].slice(1, -1);
        index += 1;
      }
      if (image) {
        output.push(`<figure class="media"><img src="${image[1]}" alt="${image[2]}" style="--preferred-width:${image[3]}px">${caption ? `<figcaption>${inline(caption)}</figcaption>` : ""}</figure>`);
      }
      continue;
    }

    if (line.startsWith("```")) {
      const fence = readFence(lines, index);
      const content = fence.content;
      if (content.includes("Compiler-first / model-graph lowering")) {
        output.push(softwareRoutes());
        index = fence.next;
        let remaining = 3;
        while (remaining > 0 && index < lines.length) {
          while (index < lines.length && !lines[index].trim()) index += 1;
          if (lines[index] && lines[index].startsWith("```")) {
            index = readFence(lines, index).next;
            remaining -= 1;
          } else {
            break;
          }
        }
      } else if (content.includes("model.safetensors") && content.includes("Header length")) {
        output.push(binaryDiagram("safetensors"));
        index = fence.next;
      } else if (content.includes("GGUF header")) {
        output.push(binaryDiagram("gguf"));
        index = fence.next;
      } else if (content.includes("Toy decoder block cgraph")) {
        output.push(toyGraph());
        index = fence.next;
      } else if (content.includes("Toy decoder block as simplified view")) {
        index = fence.next;
      } else if (content.startsWith("GGUF 文件")) {
        output.push(runtimePipeline());
        index = fence.next;
      } else if (content.includes("cgraph") && content.includes("split_graph") && content.includes("ggml_gallocr")) {
        output.push(allocationPipeline());
        index = fence.next;
      } else if (content.startsWith("1. 当前 assignment")) {
        output.push(validationChecklist());
        index = fence.next;
      } else {
        output.push(`<pre class="code-block"><code>${escapeHtml(content)}</code></pre>`);
        index = fence.next;
      }
      continue;
    }

    if (line.startsWith("> ")) {
      const quote = [];
      while (index < lines.length && lines[index].startsWith("> ")) {
        quote.push(lines[index].slice(2));
        index += 1;
      }
      output.push(`<blockquote>${inline(quote.join(" "))}</blockquote>`);
      continue;
    }

    if (/^\|.*\|\s*$/.test(line) && index + 1 < lines.length && /^\|\s*[-:| ]+\|\s*$/.test(lines[index + 1])) {
      const table = renderTable(lines, index);
      output.push(table.html);
      index = table.next;
      continue;
    }

    if (line === "---") {
      output.push("<hr>");
      index += 1;
      continue;
    }

    if (inReferences && /^\d+\.\s/.test(line)) {
      const refs = [];
      while (index < lines.length && /^\d+\.\s/.test(lines[index])) {
        const match = lines[index].match(/^(\d+)\.\s(.+)$/);
        refs.push(`<li id="ref-${match[1]}">${inline(match[2])}</li>`);
        index += 1;
      }
      output.push(`<ol class="references">${refs.join("")}</ol>`);
      continue;
    }

    if (/^-\s/.test(line)) {
      const entries = [];
      while (index < lines.length && /^(\s*)-\s(.+)$/.test(lines[index])) {
        const match = lines[index].match(/^(\s*)-\s(.+)$/);
        entries.push({ indent: match[1].length, text: match[2] });
        index += 1;
      }
      output.push(`<ul>${entries.map((entry) => `<li class="${entry.indent ? "nested" : ""}">${inline(entry.text)}</li>`).join("")}</ul>`);
      continue;
    }

    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,3})\s/.test(lines[index]) &&
      !lines[index].startsWith("```") &&
      !lines[index].startsWith("> ") &&
      !lines[index].startsWith("<p align=\"center\">") &&
      lines[index] !== "---" &&
      !(/^\|.*\|\s*$/.test(lines[index]) && index + 1 < lines.length && /^\|\s*[-:| ]+\|\s*$/.test(lines[index + 1])) &&
      !(inReferences && /^\d+\.\s/.test(lines[index]))
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    const paragraphText = paragraph.join(" ");
    output.push(`<p>${inline(paragraphText)}</p>`);
    if (paragraphText.includes("split_graph 的 assignment 过程分为五个 pass")) {
      output.push(schedulerFrame());
    }
  }

  return { body: output.join("\n"), headings };
}

function toc(headings) {
  return headings
    .filter((heading) => heading.level === 2 || heading.level === 3)
    .map(
      (heading) =>
        `<a class="toc-link level-${heading.level}" href="#${heading.id}">${escapeHtml(heading.title)}</a>`,
    )
    .join("");
}

const rendered = renderMarkdown(source);
const document = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>从模型到芯片执行：以 Tenstorrent 适配 llama.cpp/ggml 为例</title>
  <style>
    :root {
      --bg: #f6f7f6;
      --surface: #ffffff;
      --surface-alt: #f0f4f3;
      --ink: #172024;
      --muted: #58676b;
      --line: #d6dddb;
      --line-strong: #adb9b6;
      --teal: #087d78;
      --teal-soft: #dcefeb;
      --amber: #a96316;
      --amber-soft: #fff0da;
      --purple: #5d57a5;
      --purple-soft: #eceafb;
      --radius: 7px;
      --reading: 880px;
      --wide: 1140px;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; color: var(--ink); background: var(--bg); font-size: 16px; line-height: 1.75; overflow-x: hidden; }
    a { color: var(--teal); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .shell { width: 100%; max-width: 1370px; margin: 0 auto; display: grid; grid-template-columns: 244px minmax(0, 1fr); gap: 38px; padding: 30px 28px 64px; }
    .toc { position: sticky; top: 26px; height: calc(100vh - 52px); overflow: auto; padding: 18px 16px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
    .toc-title { display: block; margin-bottom: 12px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .toc-link { display: block; margin: 0 -8px; padding: 6px 8px; border-radius: 5px; color: var(--ink); font-size: 13px; line-height: 1.45; }
    .toc-link:hover { color: var(--teal); background: var(--surface-alt); text-decoration: none; }
    .toc-link.level-3 { padding-left: 19px; color: var(--muted); }
    article { width: 100%; max-width: var(--reading); min-width: 0; padding-bottom: 40px; overflow-wrap: break-word; }
    h1 { margin: 4px 0 20px; font-size: clamp(29px, 3vw, 38px); line-height: 1.25; font-weight: 700; }
    h2 { margin: 66px 0 19px; padding-top: 5px; font-size: 27px; line-height: 1.35; }
    h3 { margin: 45px 0 15px; font-size: 21px; line-height: 1.4; }
    h4 { margin: 3px 0 12px; font-size: 18px; line-height: 1.4; }
    .anchor { margin-left: 10px; color: var(--line-strong); font-size: .72em; opacity: 0; }
    h2:hover .anchor, h3:hover .anchor { opacity: 1; }
    p { margin: 0 0 17px; }
    blockquote { margin: 0 0 35px; padding: 16px 19px; border-left: 4px solid var(--teal); border-radius: 0 var(--radius) var(--radius) 0; background: var(--surface); color: #263338; font-size: 17px; }
    hr { border: 0; border-top: 1px solid var(--line); margin: 40px 0; }
    .citation { white-space: nowrap; font-size: .93em; }
    .citation a { color: var(--teal); }
    .media { margin: 24px auto 27px; text-align: center; }
    .media img { display: block; width: min(var(--preferred-width), 100%); max-height: 520px; object-fit: contain; margin: 0 auto; border: 1px solid var(--line); }
    figcaption { margin: 11px auto 0; max-width: 700px; color: var(--muted); font-size: 13px; line-height: 1.6; text-align: left; }
    .table-wrap { margin: 21px 0 28px; overflow-x: auto; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
    table { width: 100%; border-collapse: collapse; min-width: 650px; font-size: 14px; line-height: 1.55; }
    th { text-align: left; color: #29383b; background: var(--surface-alt); font-weight: 650; }
    th, td { padding: 11px 13px; border-bottom: 1px solid var(--line); vertical-align: top; }
    tbody tr:last-child td { border-bottom: 0; }
    .code-block { margin: 18px 0 27px; overflow-x: auto; padding: 17px 18px; border: 1px solid #d4dcda; border-radius: var(--radius); background: #f1f4f3; color: #1d2b2d; font: 13px/1.6 Consolas, "Cascadia Mono", monospace; }
    .visual { max-width: 100%; margin: 25px 0 31px; padding: 20px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); overflow: hidden; }
    .wide { width: min(var(--wide), calc(100vw - 334px)); max-width: none; }
    .visual-label { color: var(--teal); font-size: 12px; line-height: 1.2; font-weight: 700; text-transform: uppercase; margin-bottom: 7px; }
    .visual-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: end; gap: 15px; margin-bottom: 14px; }
    .tabs { display: flex; flex-wrap: wrap; gap: 5px; }
    .tabs button { padding: 7px 11px; border: 1px solid var(--line); border-radius: 5px; background: #fff; color: var(--muted); cursor: pointer; font-size: 13px; }
    .tabs button.active { border-color: var(--teal); background: var(--teal-soft); color: var(--teal); font-weight: 600; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .route-panel img { display: block; width: min(620px, 100%); max-height: 410px; object-fit: contain; margin: 3px auto 14px; }
    .route-panel figcaption { max-width: 760px; text-align: center; }
    .binary-grid { display: grid; grid-template-columns: minmax(280px, 1fr) 260px; gap: 18px; align-items: center; }
    .binary-stack { display: grid; gap: 6px; }
    .binary-block { text-align: left; width: 100%; display: grid; gap: 3px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 5px; background: var(--surface-alt); cursor: pointer; }
    .binary-block strong { font-size: 14px; color: var(--ink); }
    .binary-block span { color: var(--muted); font: 12px/1.5 Consolas, monospace; }
    .binary-block.active { border-color: var(--teal); background: var(--teal-soft); }
    .binary-callout { min-height: 148px; padding: 14px; border: 1px dashed var(--line-strong); border-radius: 5px; color: var(--muted); font-size: 14px; }
    .callout-title { display: block; margin-bottom: 8px; color: var(--teal); font-size: 12px; font-weight: 700; }
    .binary-callout p { margin-bottom: 18px; }
    .offset-arrow { padding: 9px; background: var(--amber-soft); color: var(--amber); font: 12px/1.45 Consolas, monospace; }
    .cgraph-svg { width: 100%; height: auto; display: block; margin: 0 auto; }
    .cgraph-svg .flow-lines path { fill: none; stroke: #b2bdbb; stroke-width: 2; marker-end: url(#flow-arrow); }
    .cgraph-svg marker path { fill: #b2bdbb; }
    .cgraph-svg .node rect { fill: #f3f6f5; stroke: #aeb9b6; rx: 5; }
    .cgraph-svg .node.accent rect { fill: var(--teal-soft); stroke: var(--teal); }
    .cgraph-svg .node.source rect { fill: var(--purple-soft); stroke: var(--purple); }
    .cgraph-svg .node.weight rect { fill: #eef5f4; stroke: #7b9692; }
    .cgraph-svg .node.weight text { font-size: 13px; fill: #38514e; }
    .cgraph-svg .node.final rect { fill: var(--amber-soft); stroke: var(--amber); }
    .cgraph-svg .node text { text-anchor: middle; font-size: 15px; fill: var(--ink); font-weight: 600; }
    .cgraph-svg .node .op { fill: var(--muted); font-size: 12px; font-weight: 500; }
    .cgraph-svg .weight-labels text { fill: var(--teal); font-size: 12px; }
    .diagram-note { color: var(--muted); font-size: 13px; line-height: 1.6; margin: 12px 0 0; }
    .compact-table table { min-width: 500px; }
    .pipeline-track { display: flex; align-items: stretch; gap: 6px; overflow-x: auto; padding: 5px 0 4px; }
    .pipe-step { flex: 0 0 135px; display: grid; gap: 8px; align-content: center; min-height: 83px; padding: 10px; border: 1px solid var(--line); background: var(--surface-alt); border-radius: 5px; }
    .pipe-step span { color: var(--muted); font-size: 11px; }
    .pipe-step strong { font-size: 13px; line-height: 1.4; }
    .pipe-arrow { flex: 0 0 18px; display: grid; align-items: center; color: var(--teal); font-weight: bold; }
    .scheduler-frame iframe { display: block; width: 100%; height: 1000px; margin-top: 17px; border: 1px solid var(--line); border-radius: 5px; background: #f5f7f5; }
    .allocation-track { max-width: 640px; margin: 12px auto 0; display: grid; justify-items: stretch; }
    .allocation-step { padding: 13px 15px; border: 1px solid var(--line); border-radius: 5px; background: var(--surface-alt); display: grid; gap: 4px; }
    .allocation-step strong { color: var(--teal); }
    .allocation-step span { color: var(--muted); font-size: 13px; }
    .vertical-arrow { height: 29px; color: var(--teal); text-align: center; font-size: 11px; line-height: 29px; text-transform: uppercase; }
    .verification-list { margin: 21px 0 27px; display: grid; gap: 7px; }
    .verification-list div { display: flex; gap: 13px; padding: 11px 13px; border: 1px solid var(--line); border-radius: 5px; background: var(--surface); }
    .verification-list span { color: var(--teal); font-weight: 700; font-variant-numeric: tabular-nums; }
    .references { margin: 12px 0; padding-left: 29px; font-size: 14px; line-height: 1.65; }
    .references li { padding: 4px 0 4px 5px; }
    @media (max-width: 1060px) {
      .shell { display: block; max-width: 930px; padding: 18px 18px 50px; }
      .toc { position: static; height: auto; margin-bottom: 34px; display: block; }
      .toc-title { width: auto; }
      .toc-link.level-3 { padding-left: 19px; }
      article { max-width: none; }
      .wide { width: 100%; }
    }
    @media (max-width: 680px) {
      body { font-size: 15px; }
      h1 { font-size: 27px; overflow-wrap: anywhere; }
      h2 { font-size: 23px; margin-top: 50px; }
      .visual { padding: 14px; }
      .binary-grid { grid-template-columns: 1fr; }
      .scheduler-frame iframe { height: 880px; }
      .media img { max-height: 360px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <nav class="toc" aria-label="目录">
      <span class="toc-title">目录</span>
      ${toc(rendered.headings)}
    </nav>
    <article>
      ${rendered.body}
    </article>
  </div>
  <script>
    (function () {
      document.querySelectorAll("[data-tab-component]").forEach(function (component) {
        component.querySelectorAll(".tabs button").forEach(function (button) {
          button.addEventListener("click", function () {
            component.querySelectorAll(".tabs button").forEach(function (item) {
              item.classList.remove("active");
              item.setAttribute("aria-selected", "false");
            });
            component.querySelectorAll(".tab-panel").forEach(function (panel) {
              panel.classList.remove("active");
            });
            button.classList.add("active");
            button.setAttribute("aria-selected", "true");
            component.querySelector("#" + button.dataset.target).classList.add("active");
          });
        });
      });
      document.querySelectorAll("[data-binary]").forEach(function (diagram) {
        var note = diagram.querySelector(".binary-callout p");
        diagram.querySelectorAll(".binary-block").forEach(function (block) {
          block.addEventListener("click", function () {
            diagram.querySelectorAll(".binary-block").forEach(function (item) {
              item.classList.remove("active");
            });
            block.classList.add("active");
            note.textContent = block.dataset.description;
          });
        });
      });
    })();
  </script>
</body>
</html>`;

fs.writeFileSync(outputPath, document, "utf8");
console.log(`Generated ${path.relative(root, outputPath)}`);
