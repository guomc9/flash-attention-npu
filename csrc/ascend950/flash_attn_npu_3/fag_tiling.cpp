/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950-specific host tiling and workspace calculation for FA v3 bwd.
 */

#include "fag_common.h"

#include <algorithm>
#include <limits>

namespace FAGTiling950 {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t FP32_ROW_ALIGN = 8;  // 32B alignment in float elements

uint64_t RoundUpU64(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
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
    // full S1*N1 region otherwise.  det slots: aicNum * continuousBlockNum
    // tiles per gradient, compact row-major with an aligned row stride.
    const uint64_t dqWsSize = tiling.dqPostAbsorb
        ? static_cast<uint64_t>(tiling.qTile) * dAlign * FP32_BYTES
        : tiling.totalQ * tiling.qHeadNum * dAlign * FP32_BYTES;
    const uint64_t dkWsSize = tiling.totalKv * tiling.kvHeadNum * dAlign * FP32_BYTES;
    const uint64_t dvWsSize = tiling.totalKv * tiling.kvHeadNum * dvAlign * FP32_BYTES;
    const uint64_t deltaWsSize = tiling.totalQ * tiling.qHeadNum * 8;

    const uint64_t slotNum =
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
