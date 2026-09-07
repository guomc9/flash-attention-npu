/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950-specific host tiling and workspace calculation for FA v3 bwd.
 */

#include "fag_common.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace FAGTiling950 {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t FP32_ROW_ALIGN = 8;  // 32B alignment in float elements

uint64_t RoundUpU64(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

}  // namespace

int64_t GetFAGTilingParam(const FAGInfo &info, FAGTilingData &tiling)
{
    if (info.batch == 0 || info.qSeqlen == 0 || info.kvSeqlen == 0 ||
        info.totalQ == 0 || info.totalKv == 0 || info.qHeadNum == 0 ||
        info.kvHeadNum == 0 || info.qHeadNum % info.kvHeadNum != 0 ||
        info.qkHeadDim == 0 || info.vHeadDim == 0 || info.aicNum == 0 ||
        info.aivNum == 0 || info.continuousBlockNum == 0 || info.ubSize == 0) {
        return -1;
    }

    tiling = FAGTilingData{};
    tiling.layout = static_cast<uint32_t>(info.layout);
    tiling.maskType = static_cast<uint32_t>(info.maskType);
    tiling.deterministic = info.deterministic;
    tiling.aicNum = info.aicNum;
    tiling.aivNum = info.aivNum;
    // tiling.
    tiling.continuousBlockNum = info.continuousBlockNum;
    tiling.ubSize = info.ubSize;
    tiling.batch = info.batch;
    tiling.qSeqlen = info.qSeqlen;
    tiling.kvSeqlen = info.kvSeqlen;
    tiling.totalQ = info.totalQ;   // B * S
    tiling.totalKv = info.totalKv; // B * S
    tiling.qHeadNum = info.qHeadNum;
    tiling.kvHeadNum = info.kvHeadNum;
    tiling.groupSize = info.qHeadNum / info.kvHeadNum;
    tiling.qkHeadDim = info.qkHeadDim;
    tiling.vHeadDim = info.vHeadDim;
    tiling.scaleValue = info.scaleValue;
    tiling.softcapValue = info.softcapValue;

    tiling.qTile = 128;
    tiling.kvTile = 128;

    tiling.usedCoreNum = info.aicNum; // TODO: 是否需要和实际task取min

    // Deterministic knobs are only meaningful when the deterministic path is
    // enabled; keep them zeroed otherwise so the legacy layout is untouched.
    tiling.dqPostAbsorb = info.deterministic ? info.dqPostAbsorb : 0;
    tiling.dqVecNum = info.deterministic ? info.dqVecNum : 0;
    tiling.dkVecNum = info.deterministic ? info.dkVecNum : 0;
    tiling.dvVecNum = info.deterministic ? info.dvVecNum : 0;

    const uint64_t dAlign = RoundUpU64(info.qkHeadDim, FP32_ROW_ALIGN);
    const uint64_t dvAlign = RoundUpU64(info.vHeadDim, FP32_ROW_ALIGN);

    if (!info.deterministic) {
        // Legacy layout, byte-identical to the non-deterministic path.
        uint64_t dqWsSize = tiling.totalQ * tiling.qHeadNum * tiling.qkHeadDim * FP32_BYTES;
        uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * tiling.qkHeadDim * FP32_BYTES;
        uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * tiling.vHeadDim * FP32_BYTES;
        uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;

        tiling.dqOffset = 0;
        tiling.dkOffset = tiling.dqOffset + dqWsSize;
        tiling.dvOffset = tiling.dkOffset + dkWsSize;
        tiling.deltaOffset = tiling.dvOffset + dvWsSize;
        tiling.workspaceSize = tiling.deltaOffset + deltaWsSize;
        return 0;
    }

    // Deterministic layout:
    //   [sync 64KB][dq][dk][dv][delta][dqDet][dkDet][dvDet]
    // dq accumulates into a single rolling tile when dqPostAbsorb=1, into the
    // full S1*N1 region otherwise. det slots reserve two complete issue
    // rounds (aicNum * continuousBlockNum tiles per round) per gradient.
    // Alternating the round bank lets VecDTM(r) read one bank while C345(r+1)
    // writes the other, without a done-counter gate on slot reuse.

    // UB budget check.  The main pipeline owns two ping/pong halves starting
    // at UB+0 (formula mirrors fag_kernel.cpp Init); the VecDTM epilogue owns
    // a dedicated tail region (mirrors fag_epilogue_deterministic_add.hpp
    // Init).  Borrowing is NOT allowed: v2 overlaps VecDTM(r) with the next
    // round's C12 SPLIT_M fixpipe, which writes the ping/pong halves.
    // The VecDTM group mapping assumes the three vec groups all exist as
    // physical AIVs.
    if (info.dqVecNum == 0 || info.dkVecNum == 0 || info.dvVecNum == 0 ||
        info.dqVecNum + info.dkVecNum + info.dvVecNum > info.aivNum) {
        fprintf(stderr,
            "FAG950 det bwd: bad vec groups %u/%u/%u (aivNum %u)\n",
            info.dqVecNum, info.dkVecNum, info.dvVecNum, info.aivNum);
        return -1;
    }
    const uint64_t rowsPerSub = RoundUpU64(tiling.qTile, 16) / 2;
    const uint64_t mmResBytes = rowsPerSub * tiling.kvTile * FP32_BYTES;
    const uint64_t attenMaskBytes = rowsPerSub * tiling.kvTile;
    const uint64_t lseBytes =
        RoundUpU64(rowsPerSub, 8) * 8 * FP32_BYTES;
    const uint64_t pBytes = (rowsPerSub + 1) * tiling.kvTile * 2;
    const uint64_t deltaBytes = rowsPerSub * 8 * FP32_BYTES;
    const uint64_t lseOff = RoundUpU64(2 * mmResBytes + attenMaskBytes, 32);
    const uint64_t pOff = RoundUpU64(lseOff + lseBytes, 32);
    const uint64_t dSOff = RoundUpU64(pOff + pBytes, 32);
    const uint64_t deltaOff = RoundUpU64(dSOff + pBytes, 32);
    const uint64_t halfUb = RoundUpU64(deltaOff + deltaBytes, 32);
    const uint64_t pipelineUb = 2 * halfUb;  // TASK_PINGPONG
    const uint64_t rowCapDq = CeilDivU64(tiling.qTile, info.dqVecNum);
    const uint64_t rowCapDk = CeilDivU64(tiling.kvTile, info.dkVecNum);
    const uint64_t rowCapDv = CeilDivU64(tiling.kvTile, info.dvVecNum);
    uint64_t rowCap = rowCapDq;
    if (rowCapDk > rowCap) { rowCap = rowCapDk; }
    if (rowCapDv > rowCap) { rowCap = rowCapDv; }
    const uint64_t colCap = dAlign > dvAlign ? dAlign : dvAlign;
    const uint64_t accUbBytes =
        RoundUpU64(rowCap * colCap * FP32_BYTES, 32);
    const uint64_t castUbBytes =
        RoundUpU64(rowCap * info.qkHeadDim * 2, 32);
    const uint64_t detAddUb = 2 * accUbBytes + castUbBytes;
    if (pipelineUb + detAddUb > info.ubSize) {
        fprintf(stderr,
            "FAG950 det bwd: UB too small: pipeline %llu + VecDTM %llu > %llu\n",
            (unsigned long long)pipelineUb, (unsigned long long)detAddUb,
            (unsigned long long)info.ubSize);
        return -1;
    }

    const uint64_t dqWsSize = tiling.dqPostAbsorb
        ? static_cast<uint64_t>(tiling.qTile) * dAlign * FP32_BYTES
        : tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
    const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
    const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
    const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;

    constexpr uint64_t detSlotBanks = 2;
    const uint64_t slotNum = detSlotBanks *
        static_cast<uint64_t>(info.aicNum) * info.continuousBlockNum;
    const uint64_t dqDetSize = RoundUpU64(
        slotNum * tiling.qTile * dAlign * FP32_BYTES, GM_ALIGNMENT);
    const uint64_t dkDetSize = RoundUpU64(
        slotNum * tiling.kvTile * dAlign * FP32_BYTES, GM_ALIGNMENT);
    const uint64_t dvDetSize = RoundUpU64(
        slotNum * tiling.kvTile * dvAlign * FP32_BYTES, GM_ALIGNMENT);

    tiling.dqOffset = MULTI_CORE_SYNC_BYTES;
    tiling.dkOffset = tiling.dqOffset + RoundUpU64(dqWsSize, GM_ALIGNMENT);
    tiling.dvOffset = tiling.dkOffset + RoundUpU64(dkWsSize, GM_ALIGNMENT);
    tiling.deltaOffset = tiling.dvOffset + RoundUpU64(dvWsSize, GM_ALIGNMENT);
    tiling.dqDetOffset = tiling.deltaOffset + RoundUpU64(deltaWsSize, GM_ALIGNMENT);
    tiling.dkDetOffset = tiling.dqDetOffset + dqDetSize;
    tiling.dvDetOffset = tiling.dkDetOffset + dkDetSize;
    tiling.workspaceSize = tiling.dvDetOffset + dvDetSize;
    return 0;
}

}  // namespace FAGTiling950
