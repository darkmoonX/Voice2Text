#include "runtime/runtime_update.h"

#include <cmath>

bool requiresCaptureRestart(const RuntimeSettings &previous, const RuntimeSettings &next) {
    const auto changedFloat = [](float lhs, float rhs) {
        return std::abs(lhs - rhs) > 0.0001F;
    };

    return previous.sourceMode != next.sourceMode || previous.loopbackDeviceId != next.loopbackDeviceId ||
           previous.sourceApps != next.sourceApps || previous.sourceLanguage != next.sourceLanguage ||
           previous.modelPath != next.modelPath ||
           changedFloat(previous.segmentSeconds, next.segmentSeconds) ||
           changedFloat(previous.hopSeconds, next.hopSeconds) ||
           previous.overlapMergeMethod != next.overlapMergeMethod ||
           previous.vadEnabled != next.vadEnabled ||
           previous.vadAdaptiveEnabled != next.vadAdaptiveEnabled ||
           changedFloat(previous.vadRmsThreshold, next.vadRmsThreshold) ||
           previous.translationEnabled != next.translationEnabled || previous.fromLang != next.fromLang ||
           previous.toLang != next.toLang;
}
