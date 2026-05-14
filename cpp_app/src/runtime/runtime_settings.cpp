#include "runtime/runtime_settings.h"

QString normalizeOverlapMergeMethod(const QString &rawMethod) {
    const QString method = rawMethod.trimmed().toLower();
    if (method == "stable-tail" || method == "replace-window" || method == "suffix-overlap" ||
        method == "fuzzy-overlap") {
        return "stable-tail";
    }
    if (method == "commit-on-break" || method == "append-only") {
        return "commit-on-break";
    }
    return "stable-tail";
}

QString effectiveModelLabel(const QString &modelPath) {
    const QString value = modelPath.trimmed();
    if (!value.isEmpty()) {
        return value;
    }
    return "unknown";
}

