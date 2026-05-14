#pragma once

#include "runtime/runtime_settings.h"

bool requiresCaptureRestart(const RuntimeSettings &previous, const RuntimeSettings &next);

