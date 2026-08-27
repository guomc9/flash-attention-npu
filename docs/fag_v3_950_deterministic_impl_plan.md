# Ascend950 FAG v3 反向确定性特性 — 实现文档

> 目标分支：`integration/FAG-V3-A5`，目录 `csrc/ascend950/flash_attn_npu_3/`。
> 设计来源：《TriDao FA-NPU A5 v3 反向确定性方案详设》+ 评审结论。
> 本文档列出全部需要修改/新增的文件、代码位置和关键实现要点。

## 1. 背景

浮点加法不满足结合律。非确定性方案中 C345/C5 对 dq/dk/dv 的 fp32 workspace 直接做
fixpipe 原子加，累加顺序由多核硬件仲裁决定，同一样本两次反向的梯度不保证逐 bit 一致。

确定性方案：每个 cube 核把梯度分块**非原子**写入自己的 GM 暂存槽（DetWorkSpace），
由 Vec 核在每轮末按**固定顺序**归约（ProcessVecDTM），保证任意两次运行的累加次序一致。

目标机器（Ascend950PR / 9579）：AIC = 28，AIV = 56（2:1），L2 = 128MB。

## 2. 已定稿的关键决策

### 2.1 任务枚举轴序

- 非确定性（现状，不动）：`b → n2 → s2 → g → s1`（kv 外 q 内）
- 确定性（新增分支）：**`b → n2 → s1 → g → s2`**（q 外 kv 内，s2 最内）

约束来源：

- `s2` 必须在 (g, s1) 之下最内层 —— 保证同一 dq tile（s1、head 固定）的贡献者
  blockId 连续，任一时刻最多 1 个未收尾 dq 块（dqPostAbsorb=1 的单 tile 缓冲区前提）；
- `n2` 在 g、s1 之外 —— 每个 kv head 的 dk/dv 在自身 n2 段内收齐，accumList 生命周期最短；
- `s1` 在 `g` 之外 —— 固定 s1 时 G 个 head 背靠背重扫同一份 K/V，L2 复用更好
  （相比文档初稿的 n2→g→s1→s2 互换 g/s1 所得）。

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
dqPostAbsorb=1 时 dq 组负载最重（合并+scale+cast+写输出），建议 16+16+24。

组内行均分：`ceil(128 / 组内核数)` 分段，末核为空直接跳过（沿用
`FagPost::ProcessRegion` 的 perCore/rangeBegin 模式）。

### 2.6 轮次末同步（信号量）

非确定性主流水是**连续任务流，无轮次边界同步**；确定性插入 VecDTM 需要新增握手。
分两步走：

- **v1（正确性优先）**：轮次末 `AscendC::SyncAll<false>()`（910 同款）；
- **v2（性能）**：GM 计数器协议，VecDTM 与下一轮 C12 重叠：

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

## 3. 文件改动清单

### 3.1 `fag_common.h`（改）— tiling ABI

`FAGInfo`（现 `:37-58`）新增：

```cpp
uint32_t dqPostAbsorb = 0;                 // host 决策
uint32_t dqVecNum = 0, dkVecNum = 0, dvVecNum = 0;  // Vec 三分组
```

`FAGTilingData`（现 `:60-92`）新增：

```cpp
uint32_t dqPostAbsorb = 0;                 // 可复用 reserved0（:68）
uint32_t dqVecNum = 0, dkVecNum = 0, dvVecNum = 0;
uint64_t dqDetOffset = 0, dkDetOffset = 0, dvDetOffset = 0;  // det 三段偏移
```

`FAGBlockInfo`（现 `:103-129`）不变。同步区固定放 workspace 起始（`syncOffset=0`，不下发字段）。

### 3.2 `fag_tiling.cpp`（改）— workspace 布局

改写现 `:55-64` 的布局计算：

```cpp
const uint64_t dAlign  = RoundUp(qkHeadDim, 8);   // fp32 32B 对齐
const uint64_t dvAlign = RoundUp(vHeadDim, 8);
// dq 累加区大小随 dqPostAbsorb
dqWsSize = dqPostAbsorb ? AlignUp(qTile × dAlign × 4B)
                        : AlignUp(totalQ × qHeadNum × dAlign × 4B);
// det 槽：aicNum × continuousBlockNum × tile（紧凑行主序，行步长 dAlign）
dqDetSize = AlignUp(aicNum × cbn × qTile  × dAlign  × 4B, GM_ALIGNMENT);
dkDetSize = AlignUp(aicNum × cbn × kvTile × dAlign  × 4B, GM_ALIGNMENT);
dvDetSize = AlignUp(aicNum × cbn × kvTile × dvAlign × 4B, GM_ALIGNMENT);

// 布局：[sync 64KB][dq][dk][dv][delta][dqDet][dkDet][dvDet]
dqOffset    = MULTI_CORE_SYNC_BYTES;
dkOffset    = dqOffset + dqWsSize;
dvOffset    = dkOffset + dkWsSize;
deltaOffset = dvOffset + dvWsSize;
// deterministic=1 时追加 det 三段；=0 时不追加（workspaceSize 在 delta 截止）
```

同时透传 `dqPostAbsorb/dqVecNum/dkVecNum/dvVecNum`。

### 3.3 `mha_bwd.cpp`（改）— host 决策

- `:128` 附近：填 `fag_info.dqPostAbsorb` 和三个 Vec 组核数（v1 可固定
  `dqPostAbsorb=1` + `16+16+24`，后续换启发式）；
- `:214-216`：对齐检查补上 det 三段偏移；
- workspace 分配（`:208-219`）跟随 `workspaceSize`，**不改**；dispatch 宏已透
  deterministic 模板参，**不改**。

### 3.4 `fag_epilogue_pre.hpp`（改）— 清零

- dqPostAbsorb=1 时跳过 dq 全量区清零（单 tile 由分支③覆盖写）；
- 新增：清零 `readyCounter/doneCounter`（0 号 AIV 负责）；
- det 槽不清零（VecDTM 只读有效行 s1Extend/s2Extend）。

### 3.5 `fag_kernel.cpp`（改）— 核心

| 位置 | 改动 |
|---|---|
| `Init`（`:96-101` 后） | IS_DTM 时新增 `dq/dk/dvDetWorkspaceGm_` 三个 GlobalTensor + 同步计数器地址 |
| `:302-323` | 新增 `LastValidS2Block(s1BlockIdx)`（FirstValidS1Block 的镜像） |
| `:325-341` | 新增 `CountValidS2Blocks`（按 s1 累加 validS2 数） |
| `LoadDecoderBatch`（`:344-362`） | IS_DTM 分支改算 per-s1 统计；batch 段长公式不变（有效对数相同） |
| `DecodeBlock`（`:410-470`） | IS_DTM 分支：①②段不动；③变长线性扫描定 **s1**（游标换 `decoderS1BlockIdx_/S1BlockBegin_`，段长 = G × validS2Num(s1)）；④ `g = blockInS1 / validS2Num`，`s2 = blockInS1 % validS2Num`（0 起，无偏移） |
| `:946-948` | 游标成员换成 s1 版本 |
| `ProcessC5Stage/C34Stage`（`:627-687`） | IS_DTM 分支：写目标改 det 槽（紧凑布局），`enAtomic=false`；轮 r>0 首次写槽前轮询放行（doneCounter）；本轮最后任务写完后 `WaitFlag<FIX_M>` + readyCounter+1 |
| `RunTasks`（`:524-544`） | Vec 侧在 `ProcessV2Stage(previousBlock_)` 后，若 `previousBlock_.issueLane == continuousBlockNum_-1` → 调 `ProcessVecDTM(issueRound)`；Cube 侧同样条件发 ready |
| drain（`:500-511`） | 补最后一轮的 ready + VecDTM |
| 新增 | `DecodeBlockCold(blockId)` 无状态冷解码，供 VecDTM 重建轮次任务清单（O(batch) 可接受） |

任务来源反解（VecDTM 用）：
`coreIdx = (blockId % waveSize) / continuousBlockNum`，`lane = blockId % continuousBlockNum`。

### 3.6 `fag_epilogue_deterministic_add.hpp`（新增）— VecDTM 本体

结构对齐现有 epilogue（`Init(resource, workspace, tiling)` +
`operator()(vectorBlockIdx, round, ...)`）：

```
分组: vectorBlockIdx < dqVecNum → dq; < +dkVecNum → dk; 余 → dv
dk/dv 组: 对本轮本组每个 dk/dv 块: accumList 按 blockId 升序合并 → atomic add 入账
dq 组:    对本轮本组每个 dq 块: 组内合并 → existFirst/existLast 四分支（§2.3）
收尾: 每组 doneCounter + 1
```

UB 按 tileElements 分块流式处理；行均分沿用 FagPost 的 ceil 分段模式。

### 3.7 `fag_epilogue_post.hpp`（改）— dq 分支

`:65-67` 处：

```cpp
if (!(tiling_->deterministic && tiling_->dqPostAbsorb)) {
    ProcessRegion<true>(dqGm_, dqWorkspace_, dqCount, ...);  // 否则 dq 已在 VecDTM 写完
}
```

### 3.8 `fag_mmad_dqkv.hpp`（不改）

`FixpipeTla`（`:400-434`）已支持 `enAtomic=false` 普通覆盖写；GM 目标由调用方
tla tensor 传入，无需改动。

### 3.9 `tests/test_flash_attn_npu_v3_bwd.py`（改）

- 解除确定性用例的 `skipif`；
- 新增 bit-exact 用例：同输入连跑两次，`torch.equal` 校验 dq/dk/dv 逐 bit 一致；
- 精度对照走现有 `npu_precision_utils` 容差框架；
- 覆盖形状：MHA（G=1）、GQA（G>1）、causal/非 causal、TND/BSND、尾块。

## 4. 实施顺序

| 步 | 内容 | 依赖 |
|---|---|---|
| 1 | fag_common.h + fag_tiling.cpp + mha_bwd.cpp（ABI 与布局） | — |
| 2 | fag_kernel.cpp 解码器轴序（IS_DTM 分支） | 1 |
| 3 | C345/C5 分槽写 + v1 SyncAll 轮次同步 | 2 |
| 4 | fag_epilogue_deterministic_add.hpp（VecDTM）+ pre/post 改动 | 3 |
| 5 | 测试：bit-exact + 精度 | 4 |
| 6 | v2：SyncAll 换 ready/done 计数器（VecDTM 与下一轮 C12 重叠） | 5 |

## 5. 开放项（TBD）

- dqPostAbsorb 的启发式启用条件（v1 固定 =1）；
- Vec 分组组合的选择策略（v1 固定 16+16+24）；
- VecDTM 耗时预算：须 < （cube 周期 − V12 周期）× continuousBlockNum，
  否则挤压主流水 —— 实测后决定是否降为每两轮吸收一次；
- g/s1 轴序（已定 n2→s1→g→s2）与 L2 复用的实测对比。
