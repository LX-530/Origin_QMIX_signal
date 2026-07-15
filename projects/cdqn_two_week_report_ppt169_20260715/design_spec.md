# cdqn_two_week_report - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | 级联 DQN 改造两周成果汇报 |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 10 |
| **Design Style** | General Consulting + 技术研究汇报 |
| **Target Audience** | 导师、课题组、项目评审 |
| **Use Case** | 阶段性成果汇报：说明为什么改、改了什么、当前结果可信到什么程度 |
| **Created Date** | 2026-07-15 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 56px, top 48px, bottom 42px |
| **Content Area** | 1168×600 |

---

## III. Visual Theme

### Theme Style

- **Style**: clean consulting report, technical but not decorative.
- **Theme**: light theme.
- **Tone**: restrained, precise, evidence-first.

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#F8FAFC` | Page background |
| **Surface** | `#FFFFFF` | Tables, panels, code blocks |
| **Primary** | `#0B1F3A` | Titles, major lines, conclusion blocks |
| **Accent** | `#2563EB` | Algorithm / CDQN emphasis |
| **Secondary accent** | `#14B8A6` | Verified / passed status |
| **Warning** | `#E11D48` | Risk, missing 24-agent definition |
| **Body text** | `#111827` | Main text |
| **Secondary text** | `#475569` | Captions, notes |
| **Tertiary text** | `#94A3B8` | Footers, muted labels |
| **Border/divider** | `#CBD5E1` | Tables, separators |
| **Soft blue** | `#DBEAFE` | CDQN light background |
| **Soft teal** | `#CCFBF1` | Verified light background |
| **Soft red** | `#FFE4E6` | Risk light background |

### Gradient Scheme

Use linear gradients only for subtle section bands. Avoid decorative blobs or large gradients.

---

## IV. Typography System

### Font Plan

**Typography direction**: PPT-safe modern CJK sans + monospace for code.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei"` | Arial | sans-serif |
| **Body** | `"Microsoft YaHei"` | Arial | sans-serif |
| **Emphasis** | `"Microsoft YaHei"` | Arial | sans-serif |
| **Code** | - | Consolas, `"Courier New"` | monospace |

**Per-role font stacks**

- Title: `"Microsoft YaHei", Arial, sans-serif`
- Body: `"Microsoft YaHei", Arial, sans-serif`
- Emphasis: `"Microsoft YaHei", Arial, sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 20px.

| Purpose | Size |
| ------- | ---- |
| Cover title | 46-56px |
| Page title | 30-36px |
| Subtitle | 22-26px |
| Body | 18-21px |
| Annotation | 13-15px |
| Code | 16-18px |

Formula policy: `text-only`. The deck contains short expressions such as `4^24` and `24×4`, which should stay editable.

---

## V. Layout Principles

### Page Structure

- **Header area**: 48-110px, contains page title and one-line conclusion.
- **Content area**: 500-560px, uses one primary information structure per page.
- **Footer area**: 30-42px, contains page index and source/status note.

### Layout Pattern Library

- Cover and conclusion pages use large-title anchor layouts.
- Audit and implementation pages use dense tables or 2-column evidence layout.
- Method pages use clean process/architecture diagrams.
- Result pages avoid overclaiming: small validated facts + placeholder for server-scale results.

### Spacing Specification

| Element | Recommended Range | Current Project |
| ------- | ---------------- | --------------- |
| Safe margin from canvas edge | 48-60px | 56px |
| Content block gap | 24-36px | 28px |
| Icon-text gap | 8-14px | 10px |
| Panel radius | 6-8px | 8px |
| Panel padding | 20-28px | 24px |

---

## VI. Icon Usage Specification

### Source

- Built-in icon library: `tabler-outline`.
- Stroke width: `2`.
- Approved inventory: `alert-triangle`, `arrows-shuffle`, `brain`, `chart-bar`, `chart-line`, `check`, `code`, `cpu`, `database`, `flame`, `git-branch`, `list-check`, `network`, `route`, `settings`, `shield-check`, `target`, `timeline`, `users`.

---

## VII. Visualization Reference List

Catalog read: 71 templates

| Page | Template | Path | Summary-quote (verbatim from `charts_index.json`) | Usage |
| ---- | -------- | ---- | ------------------------------------------------- | ----- |
| P04 | process_flow | `templates/charts/process_flow.svg` | "Pick for 3-8 sequential steps connected by simple arrows 鈥?approval workflows, customer onboarding, request handling, lifecycle stages. Skip if cyclical (use circular_stages) or stages produce named outputs (use pipeline_with_stages)." | Show cascading action selection from Q1 to Qk |
| P06 | layered_architecture | `templates/charts/layered_architecture.svg` | "Pick for 3-4 horizontal architecture layers (presentation/service/data), 2-4 module cards per layer, each card = title + 1-line description (description required, even if source brief). Skip if no per-module descriptions (use icon_grid) or no horizontal layering (use module_composition)." | Explain CDQN implementation modules |
| P09 | project_schedule_table | `templates/charts/project_schedule_table.svg` | "Pick for table-style task tracker (task / owner / status / timeline). Skip for true Gantt with dependencies (use gantt_chart) or schedule without ownership (use roadmap_vertical)." | Show two-week deliverables by outcome |

**Runners-up considered**

- `agenda_list` | rejected for P02: the audit page is substantive, not merely a table of contents.
- `kpi_cards` | rejected for P08: local results are smoke-test facts, not mature KPI evidence.
- `line_chart` | rejected for P08: the available curve has too few points and would overclaim.

---

## VIII. Image Resource List

No raster images are used. Existing `compare_curves.png` was visually reviewed and rejected for formal use because it contains too few evaluation points and is only a smoke/sanity artifact.

---

## IX. Content Outline

### Slide 01 - Cover

- **Layout**: Anchor page, left title + right compact status ledger.
- **Title**: 级联 DQN 改造两周成果汇报
- **Subtitle**: 从 QMIX 训练问题审计到 CDQN 最小可运行实现
- **Content**: project status, date, no-overclaim note.

### Slide 02 - Core Question

- **Layout**: Situation-Complication-Answer strip.
- **Title**: 要解决的不是“换算法”这么简单
- **Content**: 24-agent convergence issue; current code audited as 18 agents; first fix training baseline and action search complexity.

### Slide 03 - Environment Audit

- **Layout**: Dense table + risk callouts.
- **Title**: 当前仓库审计：18 个智能体，两个训练高风险点
- **Visualization**: audit table.

### Slide 04 - Paper Insight

- **Layout**: Sequential flow.
- **Title**: 论文启发：把组合动作拆成级联选择
- **Visualization**: process_flow adaptation.

### Slide 05 - Project Adaptation

- **Layout**: Two-column contrast.
- **Title**: 本项目采用的是 CDQN-style autoregressive DQN
- **Content**: paper setting vs signal-control setting; action repetition allowed; masks mandatory.

### Slide 06 - Network and TD Target

- **Layout**: Layered architecture + equation box.
- **Title**: 最小网络：共享编码器 + GRU 级联 + Double DQN
- **Visualization**: layered_architecture adaptation.

### Slide 07 - Code Changes

- **Layout**: Module map.
- **Title**: 代码改动集中在 7 个职责点
- **Content**: new files, modified files, routing, logging.

### Slide 08 - Verified Local Results

- **Layout**: Evidence cards + restrained mini trend.
- **Title**: 本地小样验证通过，但不包装成最终性能结论
- **Content**: smoke test, CDQN 900 steps, illegal=0, loss下降, compare plot rejected.

### Slide 09 - Two-Week Deliverables

- **Layout**: Outcome table.
- **Title**: 两周工作量按成果交付，而不是按天数堆任务
- **Visualization**: project_schedule_table adaptation.

### Slide 10 - Decision and Next Step

- **Layout**: Conclusion-first, three decision gates.
- **Title**: 下一步：补齐 24 智能体定义后做服务器全量对比
- **Content**: keep CDQN if same-budget 5-seed comparison is not worse in completion and improves stability.

---

## X. Speaker Notes Requirements

- One total notes file first: `notes/total.md`.
- Notes style: formal, concise, conclusion-first.
- Duration target: 8-10 minutes.
- Purpose: report progress and align next experiments.

---

## XI. Technical Constraints Reminder

1. viewBox: `0 0 1280 720`.
2. Use native SVG elements; no `foreignObject`, no `<style>`, no CSS classes.
3. Use inline attributes only.
4. No `rgba()`. Use `fill-opacity` or `stroke-opacity`.
5. Text wraps via explicit `<tspan>`.
6. Avoid raster charts unless data quality is sufficient.
