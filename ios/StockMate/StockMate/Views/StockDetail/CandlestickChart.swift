import SwiftUI
import Charts

/// Candlestick chart view using Swift Charts
struct CandlestickChart: View {
    let bars: [PriceBar]
    let supportLevels: [Double]
    let resistanceLevels: [Double]

    @State private var selectedBar: PriceBar?

    var body: some View {
        if bars.isEmpty {
            EmptyChartView()
        } else {
            VStack(spacing: 8) {
                // Selected bar info
                if let bar = selectedBar {
                    SelectedBarInfo(bar: bar)
                }

                // Chart
                Chart {
                    // Candlestick bodies and wicks
                    ForEach(Array(bars.enumerated()), id: \.element.id) { index, bar in
                        // Wick (high-low line)
                        RuleMark(
                            x: .value("Index", index),
                            yStart: .value("Low", bar.low),
                            yEnd: .value("High", bar.high)
                        )
                        .foregroundStyle(bar.isUp ? Color.green.opacity(0.8) : Color.red.opacity(0.8))
                        .lineStyle(StrokeStyle(lineWidth: 1))

                        // Body (open-close rectangle)
                        RectangleMark(
                            x: .value("Index", index),
                            yStart: .value("Open", bar.open),
                            yEnd: .value("Close", bar.close),
                            width: .fixed(max(2, min(8, 300 / Double(bars.count))))
                        )
                        .foregroundStyle(bar.isUp ? Color.green : Color.red)
                    }

                    // Support levels
                    ForEach(supportLevels, id: \.self) { level in
                        RuleMark(y: .value("Support", level))
                            .foregroundStyle(Color.green.opacity(0.4))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
                            .annotation(position: .leading, alignment: .leading) {
                                Text("S")
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundStyle(.green)
                            }
                    }

                    // Resistance levels
                    ForEach(resistanceLevels, id: \.self) { level in
                        RuleMark(y: .value("Resistance", level))
                            .foregroundStyle(Color.red.opacity(0.4))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
                            .annotation(position: .leading, alignment: .leading) {
                                Text("R")
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundStyle(.red)
                            }
                    }
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .trailing) { value in
                        AxisValueLabel {
                            if let price = value.as(Double.self) {
                                Text(String(format: "$%.0f", price))
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                            .foregroundStyle(Color(.systemGray4))
                    }
                }
                .chartYScale(domain: yAxisDomain)
                .chartOverlay { proxy in
                    GeometryReader { geometry in
                        Rectangle()
                            .fill(Color.clear)
                            .contentShape(Rectangle())
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onChanged { value in
                                        let xPosition = value.location.x
                                        if let index = proxy.value(atX: xPosition, as: Int.self),
                                           index >= 0 && index < bars.count {
                                            selectedBar = bars[index]
                                        }
                                    }
                                    .onEnded { _ in
                                        selectedBar = nil
                                    }
                            )
                    }
                }
                .frame(height: 250)
            }
        }
    }

    private var yAxisDomain: ClosedRange<Double> {
        guard !bars.isEmpty else { return 0...100 }

        let lows = bars.map { $0.low } + supportLevels
        let highs = bars.map { $0.high } + resistanceLevels

        let minPrice = lows.min() ?? 0
        let maxPrice = highs.max() ?? 100

        let padding = (maxPrice - minPrice) * 0.05
        return (minPrice - padding)...(maxPrice + padding)
    }
}

/// Selected bar information display
struct SelectedBarInfo: View {
    let bar: PriceBar

    var body: some View {
        HStack(spacing: 16) {
            PriceLabel(title: "O", value: bar.open, color: .secondary)
            PriceLabel(title: "H", value: bar.high, color: .green)
            PriceLabel(title: "L", value: bar.low, color: .red)
            PriceLabel(title: "C", value: bar.close, color: bar.isUp ? .green : .red)

            Spacer()

            Text(formatVolume(bar.volume))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }

    private func formatVolume(_ volume: Int) -> String {
        if volume >= 1_000_000 {
            return String(format: "%.1fM", Double(volume) / 1_000_000)
        } else if volume >= 1_000 {
            return String(format: "%.1fK", Double(volume) / 1_000)
        }
        return "\(volume)"
    }
}

/// Price label for OHLC display
struct PriceLabel: View {
    let title: String
    let value: Double
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.tertiary)

            Text(String(format: "%.2f", value))
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(color)
        }
    }
}

/// Empty state for chart
struct EmptyChartView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "chart.xyaxis.line")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)

            Text("No chart data available")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(height: 250)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

// MARK: - Preview

#Preview("Candlestick Chart") {
    CandlestickChart(
        bars: PriceBar.samples,
        supportLevels: [165.0],
        resistanceLevels: [180.0, 185.0]
    )
    .padding()
}

#Preview("Empty Chart") {
    CandlestickChart(
        bars: [],
        supportLevels: [],
        resistanceLevels: []
    )
    .padding()
}
