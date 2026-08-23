// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>

#include <QApplication>

#include <stdexcept>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    Skigen::Plot::PlotView view;

    bool rejectedZeroWindow = false;
    try {
        view.startTelemetry(0);
    } catch (const std::invalid_argument&) {
        rejectedZeroWindow = true;
    }
    if (!rejectedZeroWindow)
        return 1;

    view.startTelemetry(3);
    if (view.is2DView() || view.telemetryPointCount() != 0)
        return 2;
    view.appendTelemetry(1.0f);
    view.appendTelemetry(2.0f);
    view.appendTelemetry(10.0f, 3.0f);
    view.appendTelemetry(11.0f, 4.0f);
    if (view.telemetryPointCount() != 3)
        return 3;

    view.startTelemetry(2);
    if (view.telemetryPointCount() != 0)
        return 4;
    view.appendTelemetry(5.0f);
    if (view.telemetryPointCount() != 1)
        return 5;

    view.clearTelemetry();
    return view.telemetryPointCount() == 0 ? 0 : 6;
}
