# Ascend950 FAG v3 反向确定性特性 — 实现文档

> 目标分支：`integration/FAG-V3-A5`，目录 `csrc/ascend950/flash_attn_npu_3/`。
> 设计来源：《TriDao FA-NPU A5 v3 反向确定性方案详设》+ 评审结论。
> 本文档 §3 以一张进度总表列出全部需要修改/新增的文件及具体修改需求，
> 每条修改需求独占一行，末列 ✅ = 已完成，🟥 = 未完成。

## 1. 背景

浮点加法不满足结合律。非确定性方案中 C345/C5 对 dq/dk/dv 的 fp32 workspace 直接做
fixpipe 原子加，累加顺序由多核硬件仲裁决定，同一样本两次反向的梯度不保证逐 bit 一致。

确定性方案：每个 cube 核把梯度分块**非原子**写入自己的 GM 暂存槽（DetWorkSpace），
由 Vec 核在每轮末按**固定顺序**归约（ProcessVecDTM），保证任意两次运行的累加次序一致。

目标机器（Ascend950PR / 9579）：AIC = 28，AIV = 56（2:1），L2 = 128MB。

## 2. 已定稿的关键决策

### 2.1 任务枚举轴序（bn2s1gs2）

- 非确定性（现状，不动）：`b → n2 → s2 → g → s1`（kv 外 q 内）
- 确定性（新增分支）：**`b → n2 → s1 → g → s2`**（q 外 kv 内，s2 最内）

> 注：详设文档初稿为 `b → n2 → g → s1 → s2`，评审后互换 g/s1 定为
> `b → n2 → s1 → g → s2`，本 plan 全部按后者执行。

约束来源：

- `s2` 必须在 (g, s1) 之下最内层 —— 保证同一 dq tile（s1、head 固定）的贡献者
  blockId 连续，任一时刻最多 1 个未收尾 dq 块（dqPostAbsorb=1 的单 tile 缓冲区前提）；
- `n2` 在 g、s1 之外 —— 每个 kv head 的 dk/dv 在自身 n2 段内收齐，accumList 生命周期最短；
- `s1` 在 `g` 之外 —— 固定 s1 时 G 个 head 背靠背重扫同一份 K/V，L2 复用更好。

head 关系：GQA 下 `h1 = n2 × G + g`；dq 按 q head 独立（跨 head 无累加），
dk/dv 按 kv head 汇聚（同组 G 个 q head 的贡献全部累加到同一 kv head）。

### 2.2 DetWorkSpace 模型

每 cube 核一组私有槽位，槽数 = `continuousBlockNum`（轮内任务数），不是按流水深度 2：

```
槽地址 = detBase + (coreIdx × continuousBlockNum + issueLane) × slotBytes
```

- 生产者：C345/C5 在主流水内以 `enAtomic=false` 覆盖写；
- 消费者：VecDTM 在轮次末统一吸收，因此两次吸收之间本核要攒 `continuousBlockNum` 个任务；
- 轮内槽索引用 `issueLane`，**不是** `taskId % 2`。

### 2.3 dqPostAbsorb

host 侧决策的开关，仅影响 dq 的归约路径（det 槽三个梯度都需要）：

| | dqPostAbsorb=1 | dqPostAbsorb=0 |
|---|---|---|
| dqWorkSpaceGm_ | `BLOCK_S1 × Dqk`（单 tile 滚动累加器） | `S1 × N1 × Dqk`（全量 fp32） |
| dq 收尾 | VecDTM 内四分支（含 scale+cast 直写 dqGm） | 与 dk/dv 相同，入账后走 FagPost |
| FagPost 范围 | 仅 dk/dv | dq/dk/dv |

dqPostAbsorb=1 可行性的依据：轴序 bn2s1gs2 下 s2 最内，未收尾 dq 块任一时刻最多 1 个，
故 `BLOCK_S1 × Dqk` 容量足够（详设文档 §WorkSpace 说明）。

dq 四分支（按本组 accumList 是否含首块 s2Idx==0 / 尾块 s2Idx==lastValidS2(s1)）：

| 分支 | 条件 | 动作 |
|---|---|---|
| ① 尾非首 | last=1, first=0 | 合并值 Add 读回的旧值（**旧值最后加**，次序固定）→ Muls(scale) → Cast → 写 dqGm |
| ② 首+尾 | last=1, first=1 | 跳过读回 → Muls → Cast → 写 dqGm |
| ③ 首非尾 | first=1, last=0 | 覆盖写 dqWorkSpace |
| ④ 中间 | 0, 0 | atomic add 进 dqWorkSpace |

### 2.4 dk/dv 归约

每轮为每个 dk/dv 块维护 accumList（本轮贡献者按 blockId 升序；轴序定为
n2→s1→g→s2 后，同一 (n2,s2) 块的贡献者次序为 **s1 外层、g 内层**）：

```
① addTensor = accumList[0] 的 det 槽数据
② for i=1..n-1: addTensor = Add(addTensor, accumList[i])   # 组内定序合并
③ SetAtomicType<float> + DataCopyPad → dk/dvWorkSpace; SetAtomicNone()
```

入账用原子加是安全的：同一 dk 块的入账永远由同一 Vec 组按轮次先后执行，次序固定。

### 2.5 Vec 核三分组（可配置）

AIV 按 tiling 下发的 `dqVecNum/dkVecNum/dvVecNum` 分段：核号
`[0, dqVecNum)` → dQ 组，`[dqVecNum, +dkVecNum)` → dK 组，余下 → dV 组；
超出总和的 AIV 跳过 VecDTM（V12 主流水照跑）。

候选组合（AIV=56，按 (dq, dk, dv)）：`16+16+16` / `16+16+24` / `18+18+20`。
dqPostAbsorb=1 时 dq 组负载最重（合并+scale+cast+写输出），v1 统一定为 **16+16+16**，后续可据实测调整。

组内行均分：`ceil(128 / 组内核数)` 分段，末核为空直接跳过（沿用
`FagPost::ProcessRegion` 的 perCore/rangeBegin 模式）。同一组同一行段同一时刻
只有一个 Vector 核处理，组内不需要任何原子操作。

### 2.6 轮次末同步（信号量）

非确定性主流水是**连续任务流，无轮次边界同步**；确定性插入 VecDTM 需要新增握手。
分两步走：

- **v1（正确性优先）**：轮次末 `AscendC::SyncAll<false>()` **全核同步，所有核到齐后再做 VecDTM 定序累加**（910 同款），不加任何新协议；
- **v2（性能）**：GM 计数器协议，**VecDTM 与下一轮 C12 流水重叠**：

```
同步区（workspace 起始 64KB = MULTI_CORE_SYNC_BYTES）：
  readyCounter (8B): cube→vec，本轮 det 槽就绪计数（单调递增）
  doneCounter  (8B): vec→cube，本轮 VecDTM 完成计数（单调递增）

① 就绪: cube 完成本轮最后一个 C345/C5 → WaitFlag<FIX_M> 确认 fixpipe 落 GM
        → atomicAdd(readyCounter, 1)，不停，直接进发下一轮 C12
② 等齐: 参与 DTM 的 AIV 轮询 readyCounter ≥ 累计目标值(r)
③ 吸收: 三组 AIV 按分工归约
④ 完成: 每组 AIV atomicAdd(doneCounter, 1)
⑤ 放行: cube 下一轮首次写 det 槽前轮询 doneCounter ≥ 3 × issueRound
        （等待点在写槽前而非轮边界，C12 不被阻塞）
```

累计目标值：轮 r 参与核数 = `ceil(min(waveSize, totalBlocks − r×waveSize) / continuousBlockNum)`，
readyCounter 目标为各轮参与核数的**累计和**（单调免复位）。
轮询须用绕 cache 的原子/易变读（910 `MUL_CORE_SYNC_BUFFER` 同款机制）。

## 3. 文件修改进度总表

> 行序 = 实施顺序；末列 ✅ = 已完成，🟥 = 未完成。
> 阶段划分：**v1 = 全核同步（SyncAll）后再累加（VecDTM）**，先求正确性（条目 21-24、26-37）；
> **v2 = ready/done 计数器 + C12 与 VecDTM 流水重叠**（条目 25、31），v1 测试全绿后再做。

| # | 文件 | 具体修改需求 | 状态 |
|---|---|---|---|
| 1 | `fag_common.h` | `FAGInfo`/`FAGTilingData` 已含 `deterministic`、`dqPostAbsorb`、`dqVecNum/dkVecNum/dvVecNum`、`dq/dk/dvDetOffset`（:40-41、:55-57、:67-69、:97-99） | ✅ |
| 2 | `fag_common.h` | `FAGBlockInfo`（:103-129）不变；同步区固定放 workspace 起始（`syncOffset=0`，不下发字段） | ✅ |
| 3 | `fag_common.h` | 与 fag_tiling.cpp 透传逻辑联调核对 | ✅ |
| 4 | `fag_tiling.cpp` | 改写布局计算（:56-65，现为 `dqOffset=0` 起排、无同步区/无 det 三段）：`dqOffset = MULTI_CORE_SYNC_BYTES(64KB)`，布局 `[sync][dq][dk][dv][delta][dqDet][dkDet][dvDet]`，行步长按 `RoundUp(headDim, 8)` 对齐 | ✅ |
| 5 | `fag_tiling.cpp` | dq 累加区随 dqPostAbsorb 分档：=1 为 `qTile×dAlign`，=0 为 `totalQ×qHeadNum×dAlign` | ✅ |
| 6 | `fag_tiling.cpp` | det 三段 = `aicNum × continuousBlockNum × tile`（紧凑行主序）；`deterministic=0` 时不追加（workspaceSize 在 delta 截止） | ✅ |
| 7 | `fag_tiling.cpp` | 透传 `dqPostAbsorb/dqVecNum/dkVecNum/dvVecNum`（现仅透传 `deterministic`，:32） | ✅ |
| 8 | `mha_bwd.cpp` | `fag_info.deterministic` 接入（:128） | ✅ |
| 9 | `mha_bwd.cpp` | dispatch 宏按 deterministic 分流模板参（:265/:279）；workspace 分配跟随 `workspaceSize`，不改 | ✅ |
| 10 | `mha_bwd.cpp` | :128 附近填 `fag_info.dqPostAbsorb` 和三个 Vec 组核数（已固定 `dqPostAbsorb=1` + `16+16+16`（mha_bwd.cpp :129-136），后续换启发式） | ✅ |
| 11 | `mha_bwd.cpp` | workspace 对齐检查补 det 三段偏移 | ✅ |
| 12 | `fag_epilogue_pre.hpp` | dqPostAbsorb=1 时跳过 dq 全量区清零（单 tile 由 dq 分支③覆盖写） | 🟥 |
| 13 | `fag_epilogue_pre.hpp` | 新增清零 `readyCounter/doneCounter`（0 号 AIV 负责） | 🟥 |
| 14 | `fag_epilogue_pre.hpp` | det 槽不清零（VecDTM 只读有效行 s1Extend/s2Extend） | 🟥 |
| 15 | `fag_kernel.cpp` | 模板参 `IS_DTM` 已存在（:64） | ✅ |
| 16 | `fag_kernel.cpp` | `Init`：IS_DTM 时新增 `dq/dk/dvDetWorkspaceGm_` 三个 GlobalTensor + 同步计数器地址 | ✅ |
| 17 | `fag_kernel.cpp` | :302 附近新增 `LastValidS2Block(s1BlockIdx)`（`FirstValidS1Block` 镜像）与 `CountValidS2Blocks`（按 s1 累加 validS2 数） | ✅ |
| 18 | `fag_kernel.cpp` | `LoadDecoderBatch`（:344-362）：IS_DTM 分支改算 per-s1 统计，batch 段长公式不变 | ✅ |
| 19 | `fag_kernel.cpp` | `DecodeBlock`（:410-470）：IS_DTM 分支轴序改 bn2s1gs2——③变长扫描定 **s1**（游标换 `decoderS1BlockIdx_`，段长 = G × validS2Num(s1)）；④ `g = blockInS1 / validS2Num`，`s2 = blockInS1 % validS2Num` | ✅ |
| 20 | `fag_kernel.cpp` | :947 游标成员换 s1 版本 | ✅ |
| 21 | `fag_kernel.cpp` | `ProcessC5Stage/C34Stage` IS_DTM 分支：写目标改 det 槽（紧凑布局），`enAtomic=false`；槽索引 = `blockId % waveSize_`（= coreIdx×cbn+issueLane）；槽步长/行步长用 `RoundUp(headDim,8)`（`dq/dk/dvDetSlotElems_`），与 tiling 的 det 段大小一致；全部包在 `if constexpr (IS_DTM)` 内，非 det 路径不动 | ✅ |
| 22 | `fag_kernel.cpp` | `RunTasks` IS_DTM 分支：轮末 flush 本轮最后任务的 C5/C34/V1/V2；`pendingBackend` 显式跟踪上一任务后端是否已消费（防轮边界重复 flush）；`Init` 内 device 端预扫描 `totalBlockNum_/totalRounds_`（逐 batch `CountValidS2Blocks`），保证所有核参与相同轮数的 barrier；非 det 路径保持原流水 | ✅ |
| 23 | `fag_kernel.cpp` | 新增 `DecodeBlockCold(blockId)` 无状态冷解码，供 VecDTM 重建轮次任务清单；反解 `coreIdx = (blockId % waveSize) / cbn`，`lane = blockId % cbn` | 🟥 |
| 24 | `fag_kernel.cpp` | **v1 轮末同步**：轮次末 `AscendC::SyncAll<false>()` 全核同步 → `ProcessVecDTM` 定序累加 → 再一次 SyncAll 放行；按 `totalRounds_` 走满所有轮（无任务的核也参与 barrier），末轮后 WaitCube/VecEvents 收尾退出 | ✅ |
| 25 | `fag_kernel.cpp` | **v2 流水重叠**：SyncAll 换 ready/done GM 计数器协议（§2.6），cube 写完本轮 det 槽 + `WaitFlag<FIX_M>` 后 readyCounter+1 直接进发下一轮 C12；VecDTM 与下一轮 C12 重叠；写槽前轮询 doneCounter 放行。依赖 v1 测试全绿后再做 | 🟥 |
| 26 | `fag_epilogue_deterministic_add.hpp` | 新建 VecDTM 本体（当前为空文件占位）：结构对齐现有 epilogue（`Init(resource, workspace, tiling)` + `operator()(vectorBlockIdx, round, ...)`） | 🟥 |
| 27 | `fag_epilogue_deterministic_add.hpp` | Vec 三分组：`vectorBlockIdx < dqVecNum` → dq；`< +dkVecNum` → dk；余 → dv；超出总和的 AIV 跳过 | 🟥 |
| 28 | `fag_epilogue_deterministic_add.hpp` | dk/dv 组：本轮本组每块按 accumList（blockId 升序）组内 Add 合并 → `SetAtomicType<float>` + DataCopyPad 入账 | 🟥 |
| 29 | `fag_epilogue_deterministic_add.hpp` | dq 组：组内合并 → existFirst/existLast 四分支收尾（§2.3，含 Muls/Cast 直写 dqGm） | 🟥 |
| 30 | `fag_epilogue_deterministic_add.hpp` | UB 按 tileElements 分块流式处理；行均分沿用 FagPost 的 ceil 分段模式 | 🟥 |
| 31 | `fag_epilogue_deterministic_add.hpp` | 收尾每组 doneCounter+1（v2 需要；v1 用 SyncAll 时可空） | 🟥 |
| 32 | `fag_epilogue_post.hpp` | :65-67 加分支：`!(deterministic && dqPostAbsorb)` 时才对 dq 走 `ProcessRegion<true>`（否则 dq 已在 VecDTM 直写 dqGm）；FagPost 范围仅 dk/dv | 🟥 |
| 33 | `fag_mmad_dqkv.hpp` | 不改：`FixpipeTla` 已支持 `enAtomic=false` 普通覆盖写；GM 目标由调用方 tla tensor 传入 | ✅ |
| 34 | `tests/test_flash_attn_npu_v3_bwd.py` | 解除确定性用例 `skipif`（:206、:248、:291、:337 逐个评估） | 🟥 |
| 35 | `tests/test_flash_attn_npu_v3_bwd.py` | 新增 bit-exact 用例：同输入连跑两次，`torch.equal` 校验 dq/dk/dv 逐 bit 一致（v1 全核同步版即应通过） | 🟥 |
| 36 | `tests/test_flash_attn_npu_v3_bwd.py` | 精度对照走现有 `npu_precision_utils` 容差框架 | 🟥 |
| 37 | `tests/test_flash_attn_npu_v3_bwd.py` | 覆盖：MHA（G=1）、GQA（G>1）、causal/非 causal、TND/BSND、尾块 | 🟥 |

## 4. 开放项（TBD）

- dqPostAbsorb 的启发式启用条件（v1 固定 =1）；
- Vec 分组组合的选择策略（v1 已固定 16+16+16）；
- VecDTM 耗时预算：须 < （cube 周期 − V12 周期）× continuousBlockNum，
  否则挤压主流水 —— 实测后决定是否降为每两轮吸收一次；
- g/s1 轴序（已定 n2→s1→g→s2）与 L2 复用的实测对比。
