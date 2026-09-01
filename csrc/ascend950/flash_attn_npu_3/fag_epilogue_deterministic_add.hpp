/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward deterministic reduction epilogue
 * (VecDTM).  At each round end, the vector cores reduce every cube core's
 * private det staging slots into the shared fp32 workspaces in a FIXED
 * blockId order, so repeated backward runs accumulate bit-exactly.
 *
 * Deterministic axis order: b -> n2 -> s1 -> g -> s2 (s2 innermost).
 *
 * Vec core assignment (tiling: dqVecNum/dkVecNum/dvVecNum, v1 = 16/16/16):
 *   [0, dqVecNum)                  -> dQ group
 *   [dqVecNum, dqVecNum+dkVecNum)  -> dK group
 *   [dqVecNum+dkVecNum, +dvVecNum) -> dV group
 *   the rest skip.  Within a group each core owns a fixed row range of
 *   every gradient block (ceil(tileRows / groupSize), FagPost-style).
 */
#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_DETERMINISTIC_ADD_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_DETERMINISTIC_ADD_HPP

#include "catlass/arch/resource.hpp"
#include "fag_common.h"
#include "kernel_operator.h"

namespace Catlass::Epilogue::Block {

namespace fag_det {

// ---------------------------------------------------------------------------
// Stateless decode helpers (bn2s1gs2).  The kernel main pipeline walks tasks
// with a stateful streaming decoder (FlashAttentionScoreGrad950::DecodeBlock);
// VecDTM must instead rebuild an arbitrary round's task list by blockId, so
// these helpers are pure functions of (blockId, shape) with no cursor state.
// ---------------------------------------------------------------------------

struct DecodeParams {
    uint32_t batchNum = 0;
    uint32_t qSeqlen = 0;     // BSND uniform length
    uint32_t kvSeqlen = 0;    // BSND uniform length
    uint32_t qHeadNum = 0;
    uint32_t kvHeadNum = 0;
    uint32_t groupNum = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t qBlockSize = 0;
    uint32_t kvBlockSize = 0;
    bool isTnd = false;
    bool isAttenMask = false;
    AscendC::GlobalTensor<int32_t> cuSeqQ;   // TND cumulative lens (B entries)
    AscendC::GlobalTensor<int32_t> cuSeqKv;
};

struct BatchShape {
    uint64_t qStart = 0;
    uint64_t kvStart = 0;
    uint32_t s1Len = 0;
    uint32_t s2Len = 0;
};

struct KvEntry {
    uint32_t n2Idx = 0;
    uint32_t s2BlockIdx = 0;
    uint32_t s2Extend = 0;
    uint32_t slotId = 0;
    uint64_t totalS2Start = 0;
};

CATLASS_DEVICE inline BatchShape GetBatchShape(
    const DecodeParams &p, uint32_t batchIdx)
{
    BatchShape bs;
    if (p.isTnd) {
        const uint64_t qEnd =
            static_cast<uint64_t>(p.cuSeqQ.GetValue(batchIdx));
        const uint64_t kvEnd =
            static_cast<uint64_t>(p.cuSeqKv.GetValue(batchIdx));
        bs.qStart = batchIdx == 0
            ? 0 : static_cast<uint64_t>(p.cuSeqQ.GetValue(batchIdx - 1));
        bs.kvStart = batchIdx == 0
            ? 0 : static_cast<uint64_t>(p.cuSeqKv.GetValue(batchIdx - 1));
        bs.s1Len = static_cast<uint32_t>(qEnd - bs.qStart);
        bs.s2Len = static_cast<uint32_t>(kvEnd - bs.kvStart);
    } else {
        bs.s1Len = p.qSeqlen;
        bs.s2Len = p.kvSeqlen;
        bs.qStart = static_cast<uint64_t>(batchIdx) * bs.s1Len;
        bs.kvStart = static_cast<uint64_t>(batchIdx) * bs.s2Len;
    }
    return bs;
}

// Valid s2-block count of one s1 block.  Causal: kv col j is visible to q
// row i iff j <= i + (s2Len - s1Len); the last visible col of the s1 block
// is (s1BlockIdx + 1) * qBlk - 1 + (s2Len - s1Len).
CATLASS_DEVICE inline uint32_t ValidS2BlockNum(
    uint32_t s1Len, uint32_t s2Len, uint32_t s1BlockIdx,
    uint32_t qBlk, uint32_t kvBlk, bool isAttenMask)
{
    const uint32_t s2BlockNum = (s2Len + kvBlk - 1) / kvBlk;
    if (!isAttenMask) {
        return s2BlockNum;
    }
    const int64_t lastCol =
        static_cast<int64_t>(s1BlockIdx + 1) * qBlk - 1 +
        (static_cast<int64_t>(s2Len) - s1Len);
    if (lastCol < 0) {
        return 0;
    }
    const uint32_t lastBlk = static_cast<uint32_t>(lastCol / kvBlk);
    return (lastBlk < s2BlockNum - 1 ? lastBlk : s2BlockNum - 1) + 1;
}

CATLASS_DEVICE inline uint32_t LastValidS2Block(
    uint32_t s1Len, uint32_t s2Len, uint32_t s1BlockIdx,
    uint32_t qBlk, uint32_t kvBlk, bool isAttenMask)
{
    // Requires ValidS2BlockNum > 0.
    return ValidS2BlockNum(
        s1Len, s2Len, s1BlockIdx, qBlk, kvBlk, isAttenMask) - 1;
}

CATLASS_DEVICE inline uint64_t CountValidS2Blocks(
    uint32_t s1Len, uint32_t s2Len, uint32_t s1BlockNum,
    uint32_t qBlk, uint32_t kvBlk, bool isAttenMask)
{
    uint64_t count = 0;
    for (uint32_t s1Blk = 0; s1Blk < s1BlockNum; ++s1Blk) {
        count += ValidS2BlockNum(
            s1Len, s2Len, s1Blk, qBlk, kvBlk, isAttenMask);
    }
    return count;
}

// Rebuild the full task coordinates of one blockId from scratch
// (b -> n2 -> s1 -> g -> s2).  Fills the same FAGBlockInfo the streaming
// decoder produces.  O(batchNum + s1BlockNum) scalar work, no data movement.
CATLASS_DEVICE inline bool DecodeBlockById(
    const DecodeParams &p, uint64_t blockId, FAGBlockInfo &out)
{
    // ---- batch: accumulate per-batch block counts ----
    uint32_t batchIdx = 0;
    uint64_t batchBegin = 0;
    BatchShape bs;
    uint64_t batchBlocks = 0;
    for (; batchIdx < p.batchNum; ++batchIdx) {
        bs = GetBatchShape(p, batchIdx);
        const uint32_t s1BlkNum = (bs.s1Len + p.qBlockSize - 1) / p.qBlockSize;
        batchBlocks = static_cast<uint64_t>(p.kvHeadNum) * p.groupNum *
            CountValidS2Blocks(bs.s1Len, bs.s2Len, s1BlkNum,
                p.qBlockSize, p.kvBlockSize, p.isAttenMask);
        if (blockId < batchBegin + batchBlocks) {
            break;
        }
        batchBegin += batchBlocks;
    }
    if (batchIdx >= p.batchNum) {
        return false;
    }
    // ---- n2 ----
    const uint32_t s1BlkNum = (bs.s1Len + p.qBlockSize - 1) / p.qBlockSize;
    const uint64_t validPerN2 = CountValidS2Blocks(
        bs.s1Len, bs.s2Len, s1BlkNum, p.qBlockSize, p.kvBlockSize,
        p.isAttenMask);
    const uint64_t blocksPerN2 = static_cast<uint64_t>(p.groupNum) * validPerN2;
    const uint64_t blockInBatch = blockId - batchBegin;
    const uint32_t n2Idx = static_cast<uint32_t>(blockInBatch / blocksPerN2);
    const uint64_t blockInN2 = blockInBatch % blocksPerN2;
    // ---- s1: variable-length linear scan, seg = groupNum * validS2Num(s1) ----
    uint64_t s1Begin = 0;
    uint32_t s1Blk = 0;
    uint32_t validS2Num = 0;
    for (; s1Blk < s1BlkNum; ++s1Blk) {
        validS2Num = ValidS2BlockNum(bs.s1Len, bs.s2Len, s1Blk,
            p.qBlockSize, p.kvBlockSize, p.isAttenMask);
        const uint64_t seg = static_cast<uint64_t>(p.groupNum) * validS2Num;
        if (blockInN2 < s1Begin + seg) {
            break;
        }
        s1Begin += seg;
    }
    if (s1Blk >= s1BlkNum) {
        return false;
    }
    // ---- g & s2 (s2 innermost, 0-based) ----
    const uint64_t blockInS1 = blockInN2 - s1Begin;
    const uint32_t groupIdx = static_cast<uint32_t>(blockInS1 / validS2Num);
    const uint32_t s2Blk = static_cast<uint32_t>(blockInS1 % validS2Num);

    const uint32_t s1Start = s1Blk * p.qBlockSize;
    const uint32_t s2Start = s2Blk * p.kvBlockSize;
    const uint64_t totalS1Start = bs.qStart + s1Start;
    const uint64_t totalS2Start = bs.kvStart + s2Start;
    const uint64_t qHeadIdx =
        static_cast<uint64_t>(n2Idx) * p.groupNum + groupIdx;

    out.blockId = blockId;
    out.batchIdx = batchIdx;
    out.n2Idx = n2Idx;
    out.groupIdx = groupIdx;
    out.s1BlockIdx = s1Blk;
    out.s2BlockIdx = s2Blk;
    out.s1Start = s1Start;
    out.s2Start = s2Start;
    out.curBatchS1 = bs.s1Len;
    out.curBatchS2 = bs.s2Len;
    out.s1Extend = (bs.s1Len - s1Start < p.qBlockSize)
        ? bs.s1Len - s1Start : p.qBlockSize;
    out.s2Extend = (bs.s2Len - s2Start < p.kvBlockSize)
        ? bs.s2Len - s2Start : p.kvBlockSize;
    out.totalS1Start = totalS1Start;
    out.totalS2Start = totalS2Start;
    out.qOffset = (totalS1Start * p.qHeadNum + qHeadIdx) * p.qkHeadDim;
    out.kOffset = (totalS2Start * p.kvHeadNum + n2Idx) * p.qkHeadDim;
    out.vOffset = (totalS2Start * p.kvHeadNum + n2Idx) * p.vHeadDim;
    out.doutOffset = (totalS1Start * p.qHeadNum + qHeadIdx) * p.vHeadDim;
    return true;
}

}  // namespace fag_det

// ---------------------------------------------------------------------------
// VecDTM epilogue.  One call per issue round, between the two round-end
// SyncAll barriers in RunTasks (v1).  UB is borrowed from the main pipeline's
// buffers: at this point the round's V1/V2 have all completed and the next
// round has not started, so the UB window is safe.
// ---------------------------------------------------------------------------
template <typename DataType, class ArchTag, class TilingData>
class FagDeterministicAdd {
public:
    CATLASS_DEVICE void Init(
        Catlass::Arch::Resource<ArchTag> &resource,
        GM_ADDR dq,        // final dq output (dq branches 1/2 write through)
        GM_ADDR cuSeqQ,    // TND cumulative lens; nullptr for BSND
        GM_ADDR cuSeqKv,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(tiling);
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dq));
        dqWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dqOffset));
        dkWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dkOffset));
        dvWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dvOffset));
        dqDetWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dqDetOffset));
        dkDetWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dkDetOffset));
        dvDetWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace + tiling_->dvDetOffset));
        cuSeqQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cuSeqQ));
        cuSeqKvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cuSeqKv));
        // Round-sync counters, v2 only (v1 uses SyncAll instead):
        // readyCounter @ workspace+0, doneCounter @ workspace+8.
        readyCounter_ = workspace;
        doneCounter_ = workspace + sizeof(int64_t);

        // Det slot geometry: slot = qTile/kvTile rows of RoundUp(dim, 8)
        // floats; must match fag_tiling.cpp and the C345/C5 write side.
        qkDimAlign_ = (tiling_->qkHeadDim + 7U) / 8U * 8U;
        dvDimAlign_ = (tiling_->vHeadDim + 7U) / 8U * 8U;
        dqDetSlotElems_ =
            static_cast<uint64_t>(tiling_->qTile) * qkDimAlign_;
        dkDetSlotElems_ =
            static_cast<uint64_t>(tiling_->kvTile) * qkDimAlign_;
        dvDetSlotElems_ =
            static_cast<uint64_t>(tiling_->kvTile) * dvDimAlign_;
        waveSize_ = static_cast<uint64_t>(tiling_->usedCoreNum) *
            tiling_->continuousBlockNum;
        scaleValue_ = tiling_->scaleValue;

        decodeParams_.batchNum = static_cast<uint32_t>(tiling_->batch);
        decodeParams_.qSeqlen = static_cast<uint32_t>(tiling_->qSeqlen);
        decodeParams_.kvSeqlen = static_cast<uint32_t>(tiling_->kvSeqlen);
        decodeParams_.qHeadNum = static_cast<uint32_t>(tiling_->qHeadNum);
        decodeParams_.kvHeadNum = static_cast<uint32_t>(tiling_->kvHeadNum);
        decodeParams_.groupNum = static_cast<uint32_t>(tiling_->groupSize);
        decodeParams_.qkHeadDim = static_cast<uint32_t>(tiling_->qkHeadDim);
        decodeParams_.vHeadDim = static_cast<uint32_t>(tiling_->vHeadDim);
        decodeParams_.qBlockSize = tiling_->qTile;
        decodeParams_.kvBlockSize = tiling_->kvTile;
        decodeParams_.isTnd =
            tiling_->layout ==
            static_cast<uint32_t>(FAGTiling950::Layout::TND);
        decodeParams_.isAttenMask =
            tiling_->maskType ==
            static_cast<uint32_t>(FAGTiling950::MaskType::CAUSAL);
        decodeParams_.cuSeqQ = cuSeqQGm_;
        decodeParams_.cuSeqKv = cuSeqKvGm_;

        // UB: accumulation buffer + incoming tile buffer (+ dq cast buffer).
        accUb_ = resource.ubBuf.template GetBufferByByte<float>(0);
        inUb_ = resource.ubBuf.template GetBufferByByte<float>(
            TILE_FLOATS * sizeof(float));
        castUb_ = resource.ubBuf.template GetBufferByByte<DataType>(
            2U * TILE_FLOATS * sizeof(float));

        eventAccUBMTE3ToMTE2 = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
        eventAccUBMTE2ToV = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        eventAccUBVToMTE3 = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        eventInUBMTE2ToV = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        eventInUBVToMTE2 = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());

    }

    CATLASS_DEVICE void operator()(
        uint32_t vectorBlockIdx,
        uint32_t issueRound,
        uint64_t totalBlockNum)
    {
        // ---- 1. group assignment: dq | dk | dv | idle ----
        const uint32_t dqN = tiling_->dqVecNum;
        const uint32_t dkN = tiling_->dkVecNum;
        const uint32_t dvN = tiling_->dvVecNum;
        Group group;
        uint32_t idxInGroup = 0;
        uint32_t groupSize = 0;
        if (vectorBlockIdx < dqN) {
            group = Group::DQ; idxInGroup = vectorBlockIdx; groupSize = dqN;
        } else if (vectorBlockIdx < dqN + dkN) {
            group = Group::DK; idxInGroup = vectorBlockIdx - dqN;
            groupSize = dkN;
        } else if (vectorBlockIdx < dqN + dkN + dvN) {
            group = Group::DV; idxInGroup = vectorBlockIdx - dqN - dkN;
            groupSize = dvN;
        } else {
            return;  // surplus AIVs skip VecDTM (V1/V2 main pipe unaffected)
        }
        if (groupSize == 0) {
            return;
        }

        // ---- 2. row range inside every block of this group ----
        const uint32_t rows =
            group == Group::DQ ? tiling_->qTile : tiling_->kvTile;
        const uint32_t perCore = (rows + groupSize - 1U) / groupSize;
        const uint32_t rowBegin = idxInGroup * perCore;
        const uint32_t rowEnd =
            (rowBegin + perCore < rows) ? rowBegin + perCore : rows;
        if (rowBegin >= rowEnd) {
            return;  // tail core of the group has no rows
        }

        // ---- 3. this round's blockId range ----
        const uint64_t blockBegin =
            static_cast<uint64_t>(issueRound) * waveSize_;
        uint64_t blockEnd = blockBegin + waveSize_;
        if (blockEnd > totalBlockNum) {
            blockEnd = totalBlockNum;
        }
        if (blockBegin >= blockEnd) {
            return;
        }

        // ---- 4. per-group reduction ----
        if (group == Group::DQ) {
            ProcessDq(rowBegin, rowEnd, blockBegin, blockEnd);
        } else if (group == Group::DK) {
            ProcessDkv(dkDetWorkspaceGm_, dkWorkspace_, dkDetSlotElems_,
                qkDimAlign_, tiling_->qkHeadDim,
                rowBegin, rowEnd, blockBegin, blockEnd);
        } else {
            ProcessDkv(dvDetWorkspaceGm_, dvWorkspace_, dvDetSlotElems_,
                dvDimAlign_, tiling_->vHeadDim,
                rowBegin, rowEnd, blockBegin, blockEnd);
        }
    }

private:
    enum class Group : uint32_t { DQ = 0, DK = 1, DV = 2 };

    static constexpr uint32_t TILE_FLOATS = 20U * 1024U;

    // Chunk size for the round scan: the on-stack task table and the 64-bit
    // consumed bitmap below cover one chunk; rounds larger than this are
    // processed in multiple chunks.
    static constexpr uint32_t MAX_DTM_CHUNK_TASKS = FAGTiling950::MAX_DTM_CHUNK_TASKS;

    // dk/dv shared path.  For every (n2, s2BlockIdx) block touched by this
    // round:
    //   1. collect its accumList: blockIds in [blockBegin, blockEnd) whose
    //      DecodeBlockById matches the key, ascending (scanning blockId in
    //      order gives the fixed merge order for free);
    //   2. acc = slot(accumList[0]) rows [rowBegin,rowEnd); for each further
    //      member: DataCopyPad into inUb_, Add(accUb_, accUb_, inUb_);
    //      slot addr = detBase + slotId * slotElems + rowBegin * dimAlign
    //      with slotId = blockId % waveSize_ (round-relative);
    //   3. SetAtomicType<float>() + DataCopyPad into ws at
    //      (totalS2Start * kvHeadNum + n2) * headDim
    //      + rowBegin * kvHeadNum * headDim, then SetAtomicNone().
    // Note: contributors of one (n2,s2) key are NOT consecutive under
    // bn2s1gs2 (s1 outer, g inner); buffer the chunk's (key, slotId) pairs
    // (at most MAX_DTM_CHUNK_TASKS entries) and group them in a second pass.
    CATLASS_DEVICE void ProcessDkv(
        AscendC::GlobalTensor<float> &detGm,
        AscendC::GlobalTensor<float> &wsGm,
        uint64_t slotElems,
        uint64_t dimAlign,
        uint64_t headDim,
        uint32_t rowBegin,
        uint32_t rowEnd,
        uint64_t blockBegin,
        uint64_t blockEnd)
    {
        const uint32_t headNum = tiling_->kvHeadNum;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInUBVToMTE2);
        // Chunk the round so the on-stack table and the 64-bit consumed bitmap always fit, however large the round gets.
        // A key split at a chunk boundary is reduced once per chunk; the fixed chunk order and disjoint per-core rows
        // keep the result deterministic.
        for (uint64_t chunkBegin = blockBegin; chunkBegin < blockEnd; chunkBegin += MAX_DTM_CHUNK_TASKS) {
            const uint64_t chunkEnd = (chunkBegin + MAX_DTM_CHUNK_TASKS < blockEnd) ? chunkBegin + MAX_DTM_CHUNK_TASKS : blockEnd;

            // 1. Collect this chunk's (n2, s2BlockIdx) keys and det slots.
            //    slotId stays round-relative (= blockId % waveSize_), NOT chunk-relative.
            KvEntry kvList[MAX_DTM_CHUNK_TASKS];
            uint32_t kvCount = 0;
            for (uint64_t blockId = chunkBegin; blockId < chunkEnd; ++blockId) {
                FAGBlockInfo info;
                if (!fag_det::DecodeBlockById(decodeParams_, blockId, info)) {
                    continue;
                }
                kvList[kvCount++] = {info.n2Idx, info.s2BlockIdx, info.s2Extend, static_cast<uint32_t>(blockId - blockBegin), info.totalS2Start};
            }

            // 2. Collect kv entries for each unique (n2, s2BlockIdx) pair.
            uint64_t consumed = 0;
            for (uint32_t i = 0; i < kvCount; ++i) {
                if (consumed & (1ULL << i)) {
                    continue;
                }
                KvEntry first = kvList[i];
                uint32_t rowNum = rowEnd - rowBegin;
                if (rowBegin >= first.s2Extend) {
                    for (uint32_t j = i + 1; j < kvCount; ++j) {
                        if (kvList[j].n2Idx == first.n2Idx && kvList[j].totalS2Start == first.totalS2Start && kvList[j].s2BlockIdx == first.s2BlockIdx) {
                            consumed |= (1ULL << j);
                        }
                    }
                    continue;
                }
                if (rowEnd > first.s2Extend) {
                    rowNum = first.s2Extend - rowBegin;
                }
                // 3. Copy valid rows of the first Dkv block from det-GM into acc-UB.
                //     rowNum: min(rowEnd, first.s2Extend) - rowBegin
                //     detOffset: slotId * slotElems + rowBegin * dimAlign
                //     src: detGm[detOffset]
                //     dst: accUb_
                uint64_t detOffset = kvList[i].slotId * slotElems + rowBegin * dimAlign;
                AscendC::DataCopyExtParams inParams{rowNum, headDim * sizeof(float), (dimAlign - headDim) * sizeof(float), 0, 0};
                AscendC::DataCopyPadExtParams<float> inPadParams{false, 0, 0, 0};
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
                AscendC::DataCopyPad(accUb_, detGm[detOffset], inParams, inPadParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventAccUBMTE2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventAccUBMTE2ToV);
                for (uint32_t j = i + 1; j < kvCount; ++j) {
                    if (kvList[j].n2Idx != first.n2Idx || kvList[j].totalS2Start != first.totalS2Start || kvList[j].s2BlockIdx != first.s2BlockIdx) {
                        continue;
                    }
                    consumed |= (1ULL << j);
                    // 4. Copy valid rows of the next Dkv block from det-GM into in-UB, then Add(accUb_, accUb_, inUb_).
                    uint64_t nextDetOffset = kvList[j].slotId * slotElems + rowBegin * dimAlign;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventInUBVToMTE2);
                    AscendC::DataCopyPad(inUb_, detGm[nextDetOffset], inParams, inPadParams);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventInUBMTE2ToV);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventInUBMTE2ToV);
                    AscendC::Add(accUb_, accUb_, inUb_, rowNum * headDim);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInUBVToMTE2);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventAccUBVToMTE3);
                // 5. Atomic add acc-UB into ws-GM.
                //     wsOffset: (first.totalS2Start * headNum + first.n2Idx) * headDim + rowBegin * headNum * headDim
                //     src: accUb_
                //     dst: wsGm[wsOffset]
                uint64_t wsOffset = (first.totalS2Start * headNum + first.n2Idx) * headDim + rowBegin * headNum * headDim;
                AscendC::DataCopyExtParams outParams{rowNum, headDim * sizeof(float), 0, (headNum - 1) * headDim * sizeof(float), 0};
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventAccUBVToMTE3);
                AscendC::SetAtomicType<float>();
                AscendC::DataCopyPad(wsGm[wsOffset], accUb_, outParams);
                AscendC::SetAtomicNone();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
            }
        }
    }

    // dq path (dqPostAbsorb=1).  Contributors of one (n1, s1) block are
    // CONSECUTIVE blockIds (s2 innermost), so a single ascending scan can
    // process run by run:
    //   1. merge the run's det slots in order (same as ProcessDkv step 2);
    //   2. existFirst = run contains s2BlockIdx == 0;
    //      existLast  = run contains s2BlockIdx == fag_det::LastValidS2Block(s1);
    //   3. four branches:
    //      last && !first: Add(acc, read-back dqWorkspace_) [old value LAST]
    //                      -> Muls(scaleValue_) -> Cast -> write dqGm_ at
    //                      qOffset + rowBegin * qHeadNum * qkHeadDim;
    //      last && first : skip read-back, Muls -> Cast -> write dqGm_;
    //      first && !last: overwrite dqWorkspace_ (plain DataCopyPad);
    //      otherwise     : atomic-add into dqWorkspace_.
    // dqWorkspace_ is a single rolling tile of qTile x qkDimAlign floats.
    CATLASS_DEVICE void ProcessDq(
        uint32_t rowBegin,
        uint32_t rowEnd,
        uint64_t blockBegin,
        uint64_t blockEnd)
    {
        const uint32_t headNum = tiling_->qHeadNum;
        uint32_t headBlockId = blockBegin;
        uint32_t rowNum = rowEnd - rowBegin;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventAccUBVToMTE3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
        while (headBlockId < blockEnd) {
            FAGBlockInfo headBlockInfo;
            if (!fag_det::DecodeBlockById(decodeParams_, headBlockId, headBlockInfo)) {
                ++headBlockId;
                continue;
            }
            uint32_t lastValidS2Block = fag_det::LastValidS2Block(
                headBlockInfo.curBatchS1, headBlockInfo.curBatchS2, 
                headBlockInfo.s1BlockIdx, decodeParams_.qBlockSize, decodeParams_.kvBlockSize, decodeParams_.isAttenMask);
            bool existFirst = (headBlockInfo.s2BlockIdx == 0);
            bool existLast = (headBlockInfo.s2BlockIdx == lastValidS2Block);
            uint32_t nextBlockId = headBlockId + 1;

            // 1. Copy valid rows of the head Dq block from det-GM into acc-UB.
            uint32_t slotId = static_cast<uint32_t>(headBlockId - blockBegin);
            uint64_t detOffset = slotId * dqDetSlotElems_ + rowBegin * qkDimAlign_;
            AscendC::DataCopyExtParams inParams{rowNum, qkHeadDim * sizeof(float), (qkDimAlign_ - qkHeadDim) * sizeof(float), 0, 0};
            AscendC::DataCopyPadExtParams<float> inPadParams{false, 0, 0, 0};
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
            AscendC::DataCopyPad(accUb_, dqDetWorkspaceGm_[detOffset], inParams, inPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventAccUBMTE2ToV);

            while (nextBlockId < blockEnd)
            {
                FAGBlockInfo nextBlockInfo;
                if (!fag_det::DecodeBlockById(decodeParams_, nextBlockId, nextBlockInfo)) {
                    ++nextBlockId;
                    continue;
                }
                if (nextBlockInfo.groupIdx != headBlockInfo.groupIdx || nextBlockInfo.totalS1Start != headBlockInfo.totalS1Start 
                     || nextBlockInfo.n2Idx != headBlockInfo.n2Idx) {
                    break;
                }
                // 2. Copy valid rows of the next Dq block from det-GM into acc-UB, then Add(accUb_, accUb_, inUb_).
                uint32_t nextSlotId = static_cast<uint32_t>(nextBlockId - blockBegin);
                uint64_t nextDetOffset = nextSlotId * dqDetSlotElems_ + rowBegin * qkDimAlign_;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventInUBVToMTE2);
                AscendC::DataCopyPad(inUb_, dqDetWorkspaceGm_[nextDetOffset], inParams, inPadParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventInUBMTE2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventInUBMTE2ToV);
                AscendC::Add(accUb_, accUb_, inUb_, rowNum * qkHeadDim);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInUBVToMTE2);
                existLast |= (nextBlockInfo.s2BlockIdx == lastValidS2Block);
                ++nextBlockId;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventAccUBVToMTE3);
            // 3. Write acc-UB into dqGm_ or dqWorkspace_ according to the four branches.
            uint64_t dqOffset = headBlockInfo.qOffset + rowBegin * headNum * qkHeadDim;
            if (existLast && !existFirst) {
                // Atomic add acc-UB into dqWorkspace_, then read-back, Muls(scaleValue_), Cast, write dqGm_.
                uint64_t wsOffset = rowBegin * qkDimAlign_;
                AscendC::DataCopyExtParams outParams{rowNum, qkHeadDim * sizeof(float), 0, (qkDimAlign_ - qkHeadDim) * sizeof(float), 0};
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventAccUBVToMTE3);
                AscendC::SetAtomicType<float>();
                AscendC::DataCopyPad(dqWorkspace_[wsOffset], accUb_, outParams);
                AscendC::SetAtomicNone();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
                // Todo: read-back dqWorkspace_, Muls(scaleValue_), Cast, write dqGm_ at dqOffset
                // ...
            }
            else if (existLast && existFirst) {
                // Directly acc-UB Muls(scaleValue_), Cast, write dqGm_.
                // ...
            }
            else if (!existLast && existFirst) {
                // Write acc-UB into dqWorkspace_.
                uint64_t wsOffset = rowBegin * qkDimAlign_;
                AscendC::DataCopyExtParams outParams{rowNum, qkHeadDim * sizeof(float), 0, (qkDimAlign_ - qkHeadDim) * sizeof(float), 0};
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventAccUBVToMTE3);
                AscendC::DataCopyPad(dqWorkspace_[wsOffset], accUb_, outParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
            }
            else {
                // Atomic add acc-UB into dqWorkspace_.
                uint64_t wsOffset = rowBegin * qkDimAlign_;
                AscendC::DataCopyExtParams outParams{rowNum, qkHeadDim * sizeof(float), 0, (qkDimAlign_ - qkHeadDim) * sizeof(float), 0};
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventAccUBVToMTE3);
                AscendC::SetAtomicType<float>();
                AscendC::DataCopyPad(dqWorkspace_[wsOffset], accUb_, outParams);
                AscendC::SetAtomicNone();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventAccUBMTE3ToMTE2);
            }
            headBlockId = nextBlockId;
        }
        return;
    }

    const __gm__ TilingData *tiling_ = nullptr;

    AscendC::GlobalTensor<DataType> dqGm_;
    AscendC::GlobalTensor<float> dqWorkspace_;
    AscendC::GlobalTensor<float> dkWorkspace_;
    AscendC::GlobalTensor<float> dvWorkspace_;
    AscendC::GlobalTensor<float> dqDetWorkspaceGm_;
    AscendC::GlobalTensor<float> dkDetWorkspaceGm_;
    AscendC::GlobalTensor<float> dvDetWorkspaceGm_;
    AscendC::GlobalTensor<int32_t> cuSeqQGm_;
    AscendC::GlobalTensor<int32_t> cuSeqKvGm_;
    GM_ADDR readyCounter_ = nullptr;   // v2
    GM_ADDR doneCounter_ = nullptr;    // v2

    // UB buffers borrowed from the main pipeline (round-end SyncAll makes the
    // window safe).  accUb_/inUb_ are shared by ProcessDq and ProcessDkv: a
    // core belongs to exactly one group per round, so the two paths are
    // mutually exclusive and never live at once; reuse ordering within each
    // path is guarded by the event_* flags below.  castUb_ is dq-only (Cast
    // result before writing dqGm_).
    AscendC::LocalTensor<float> accUb_;
    AscendC::LocalTensor<float> inUb_;
    AscendC::LocalTensor<DataType> castUb_;

    fag_det::DecodeParams decodeParams_;
    uint64_t qkDimAlign_ = 0;
    uint64_t dvDimAlign_ = 0;
    uint64_t dqDetSlotElems_ = 0;
    uint64_t dkDetSlotElems_ = 0;
    uint64_t dvDetSlotElems_ = 0;
    uint64_t waveSize_ = 0;
    float scaleValue_ = 1.0f;

    event_t eventAccUBMTE3ToMTE2 = 0;
    event_t eventAccUBMTE2ToV = 0;
    event_t eventAccUBVToMTE3 = 0;
    // event_t eventAccUBVToV = 0;
    event_t eventInUBVToMTE2 = 0;
    event_t eventInUBMTE2ToV = 0;
};

}  // namespace Catlass::Epilogue::Block

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_DETERMINISTIC_ADD_HPP
