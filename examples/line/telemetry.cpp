// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>

#include <QApplication>
#include <QTimer>

#include <cmath>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    Skigen::Plot::PlotView view;
    view.setTitle("Streaming Telemetry");
    view.setAxisLabels("sample", "signal");
    view.startTelemetry(300, {.label = "live signal"});
    view.resize(900, 500);
    view.show();

    float sample = 0.0f;
    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, [&] {
        const float signal = std::sin(sample * 0.08f)
            + 0.2f * std::sin(sample * 0.31f);
        view.appendTelemetry(signal);
        sample += 1.0f;
    });
    timer.start(16);

    return app.exec();
}