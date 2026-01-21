import SwiftUI
import Charts

/// Enhanced candlestick chart view using Swift Charts
/// Matches the quality and features of MinimalChart with proper OHLC visualization
struct CandlestickChart: View {
    let bars: [PriceBar]
    let isUp: Bool
    let supportLevels: [Double]
    let resistanceLevels: [Double]
    var timeframe: ChartTimeframe = .oneMonth

    @State private var selectedIndex: Int?
    @State private var selectedBar: PriceBar?

    private var accentColor: Color {
        isUp ? Color(.systemGreen) : Color(.systemRed)
    }

    /// Calculate candlestick body width based on bar count
    private var candleWidth: CGFloat {
        let count = bars.count
        if count <= 20 { return 10 }
        if count <= 50 { return 6 }
        if count <= 100 { return 4 }
        if count <= 200 { return 2.5 }
        return 1.5
    }

    /// Calculate wick width based on bar count
    private var wickWidth: CGFloat {
        candleWidth <= 2 ? 0.5 : 1
    }

    var body: some View {
        if bars.isEmpty {
            EmptyChartPlaceholder()
        } else {
            GeometryReader { geometry in
                let width = geometry.size.width
                let height = timeframe == .oneDay ? geometry.size.height - 24 : geometry.size.height

                VStack(spacing: 0) {
                    ZStack {
                        // Session background shading for 1D
                        if timeframe == .oneDay {
                            sessionBackgroundView(width: width, height: height)
                        }

                        // Main Chart
                        Chart {
                            // Candlestick bodies and wicks
                            ForEach(Array(bars.enumerated()), id: \.element.id) { index, bar in
                                // Wick (high-low line)
                                RuleMark(
                                    x: .value("Index", index),
                                    yStart: .value("Low", bar.low),
                                    yEnd: .value("High", bar.high)
                                )
                                .foregroundStyle(bar.isUp ? Color.green.opacity(0.9) : Color.red.opacity(0.9))
                                .lineStyle(StrokeStyle(lineWidth: wickWidth))

                                // Body (open-close rectangle)
                                RectangleMark(
                                    x: .value("Index", index),
                                    yStart: .value("Open", bar.open),
                                    yEnd: .value("Close", bar.close),
                                    width: .fixed(candleWidth)
                                )
                                .foregroundStyle(bar.isUp ? Color.green : Color.red)
                                .cornerRadius(candleWidth > 4 ? 1 : 0)
                            }

                            // Support levels
                            ForEach(supportLevels, id: \.self) { level in
                                RuleMark(y: .value("Support", level))
                                    .foregroundStyle(Color.green.opacity(0.5))
                                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                            }

                            // Resistance levels
                            ForEach(resistanceLevels, id: \.self) { level in
                                RuleMark(y: .value("Resistance", level))
                                    .foregroundStyle(Color.red.opacity(0.5))
                                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                            }

                            // Selection indicator
                            if let index = selectedIndex {
                                RuleMark(x: .value("Selected", index))
                                    .foregroundStyle(Color(.systemGray3))
                                    .lineStyle(StrokeStyle(lineWidth: 1))
                            }
                        }
                        .chartXAxis(.hidden)
                        .chartYAxis {
                            AxisMarks(position: .trailing, values: .automatic(desiredCount: 5)) { value in
                                AxisValueLabel {
                                    if let price = value.as(Double.self) {
                                        Text(formatAxisPrice(price))
                                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.3))
                                    .foregroundStyle(Color(.systemGray4))
                            }
                        }
                        .chartYScale(domain: yAxisDomain)
                        .chartXScale(domain: -1...(bars.count))
                        .chartOverlay { proxy in
                            GeometryReader { geo in
                                Rectangle()
                                    .fill(Color.clear)
                                    .contentShape(Rectangle())
                                    .gesture(
                                        DragGesture(minimumDistance: 0)
                                            .onChanged { value in
                                                let xPosition = value.location.x
                                                if let plotFrame = proxy.plotFrame {
                                                    let plotWidth = plotFrame.width
                                                    let xRatio = xPosition / plotWidth
                                                    let index = Int((xRatio * CGFloat(bars.count)).rounded())
                                                    let clampedIndex = max(0, min(index, bars.count - 1))
                                                    selectedIndex = clampedIndex
                                                    selectedBar = bars[clampedIndex]
                                                }
                                            }
                                            .onEnded { _ in
                                                withAnimation(.easeOut(duration: 0.2)) {
                                                    selectedIndex = nil
                                                    selectedBar = nil
                                                }
                                            }
                                    )
                            }
                        }
                        .frame(height: height)

                        // Selected bar tooltip overlay
                        if let bar = selectedBar, let index = selectedIndex {
                            selectedBarOverlay(bar: bar, index: index, width: width, height: height)
                        }
                    }
                    .frame(height: height)

                    // Session labels for 1D timeframe
                    if timeframe == .oneDay {
                        sessionLabelsView(width: width)
                    }
                }
            }
        }
    }

    // MARK: - Y-Axis Domain

    private var yAxisDomain: ClosedRange<Double> {
        guard !bars.isEmpty else { return 0...100 }

        let allLows = bars.map { $0.low } + supportLevels.filter { level in
            let minPrice = bars.map { $0.low }.min() ?? 0
            let maxPrice = bars.map { $0.high }.max() ?? 100
            return level >= minPrice * 0.95 && level <= maxPrice * 1.05
        }
        let allHighs = bars.map { $0.high } + resistanceLevels.filter { level in
            let minPrice = bars.map { $0.low }.min() ?? 0
            let maxPrice = bars.map { $0.high }.max() ?? 100
            return level >= minPrice * 0.95 && level <= maxPrice * 1.05
        }

        let minPrice = allLows.min() ?? 0
        let maxPrice = allHighs.max() ?? 100
        let padding = (maxPrice - minPrice) * 0.08

        return (minPrice - padding)...(maxPrice + padding)
    }

    // MARK: - Price Formatting

    private func formatAxisPrice(_ price: Double) -> String {
        if price >= 1000 {
            return String(format: "$%.0f", price)
        } else if price >= 100 {
            return String(format: "$%.1f", price)
        } else {
            return String(format: "$%.2f", price)
        }
    }

    // MARK: - Time Formatting

    private func formatTime(for bar: PriceBar) -> String {
        guard let date = bar.date else { return "" }
        let formatter = DateFormatter()
        formatter.timeZone = TimeZone(identifier: "America/New_York")

        switch timeframe {
        case .oneDay:
            formatter.dateFormat = "h:mm a"
            return formatter.string(from: date)
        case .oneWeek:
            formatter.dateFormat = "E h:mm a"
            return formatter.string(from: date)
        case .oneMonth, .threeMonths:
            formatter.dateFormat = "MMM d"
            return formatter.string(from: date)
        case .sixMonths, .oneYear, .yearToDate:
            formatter.dateFormat = "MMM d, yyyy"
            return formatter.string(from: date)
        case .fiveYears, .all:
            formatter.dateFormat = "MMM yyyy"
            return formatter.string(from: date)
        }
    }

    // MARK: - Session Detection (for 1D timeframe)

    private enum SessionType {
        case preMarket, marketHours, afterHours
    }

    private func sessionType(for bar: PriceBar) -> SessionType? {
        guard timeframe == .oneDay, let date = bar.date else { return nil }
        let calendar = Calendar.current
        guard let tz = TimeZone(identifier: "America/New_York") else { return nil }
        let components = calendar.dateComponents(in: tz, from: date)
        let hour = components.hour ?? 0
        let minute = components.minute ?? 0
        let timeInMinutes = hour * 60 + minute

        // Pre-market: 4:00 AM - 9:30 AM ET (240 - 570 minutes)
        // Market hours: 9:30 AM - 4:00 PM ET (570 - 960 minutes)
        // After-hours: 4:00 PM - 8:00 PM ET (960 - 1200 minutes)

        if timeInMinutes < 570 {
            return .preMarket
        } else if timeInMinutes < 960 {
            return .marketHours
        } else {
            return .afterHours
        }
    }

    private func sessionBoundaries() -> [(label: String, startIndex: Int, endIndex: Int)] {
        guard timeframe == .oneDay, !bars.isEmpty else { return [] }

        var boundaries: [(label: String, startIndex: Int, endIndex: Int)] = []

        var preMarketStart: Int?
        var preMarketEnd: Int?
        var marketStart: Int?
        var marketEnd: Int?
        var afterHoursStart: Int?
        var afterHoursEnd: Int?

        for (index, bar) in bars.enumerated() {
            if let session = sessionType(for: bar) {
                switch session {
                case .preMarket:
                    if preMarketStart == nil { preMarketStart = index }
                    preMarketEnd = index
                case .marketHours:
                    if marketStart == nil { marketStart = index }
                    marketEnd = index
                case .afterHours:
                    if afterHoursStart == nil { afterHoursStart = index }
                    afterHoursEnd = index
                }
            }
        }

        if let start = preMarketStart, let end = preMarketEnd {
            boundaries.append(("Pre", start, end))
        }
        if let start = marketStart, let end = marketEnd {
            boundaries.append(("Market", start, end))
        }
        if let start = afterHoursStart, let end = afterHoursEnd {
            boundaries.append(("After", start, end))
        }

        return boundaries
    }

    // MARK: - Session Background View

    @ViewBuilder
    private func sessionBackgroundView(width: CGFloat, height: CGFloat) -> some View {
        let boundaries = sessionBoundaries()
        let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width

        ForEach(Array(boundaries.enumerated()), id: \.offset) { _, boundary in
            let startX = CGFloat(boundary.startIndex) * stepX
            let endX = CGFloat(boundary.endIndex) * stepX
            let rectWidth = max(endX - startX, 1)

            Rectangle()
                .fill(boundary.label == "Market" ? Color(.systemGray6).opacity(0.6) : Color.clear)
                .frame(width: rectWidth, height: height)
                .position(x: (startX + endX) / 2, y: height / 2)
        }
    }

    // MARK: - Session Labels View

    @ViewBuilder
    private func sessionLabelsView(width: CGFloat) -> some View {
        let boundaries = sessionBoundaries()
        let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width

        HStack(spacing: 0) {
            ForEach(Array(boundaries.enumerated()), id: \.offset) { idx, boundary in
                let startX = CGFloat(boundary.startIndex) * stepX
                let endX = CGFloat(boundary.endIndex) * stepX
                let sectionWidth = max(endX - startX, 1)

                Text(boundary.label)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(boundary.label == "Market" ? .primary : .secondary)
                    .frame(width: sectionWidth)

                if idx < boundaries.count - 1 {
                    Rectangle()
                        .fill(Color(.systemGray4))
                        .frame(width: 1, height: 12)
                }
            }
        }
        .frame(height: 20)
        .padding(.top, 4)
    }

    // MARK: - Selected Bar Overlay

    @ViewBuilder
    private func selectedBarOverlay(bar: PriceBar, index: Int, width: CGFloat, height: CGFloat) -> some View {
        let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width
        let x = CGFloat(index) * stepX

        // Price tooltip
        VStack(spacing: 4) {
            // OHLC row
            HStack(spacing: 8) {
                OHLCLabel(title: "O", value: bar.open, color: .secondary)
                OHLCLabel(title: "H", value: bar.high, color: .green)
                OHLCLabel(title: "L", value: bar.low, color: .red)
                OHLCLabel(title: "C", value: bar.close, color: bar.isUp ? .green : .red)
            }

            // Time and volume
            HStack(spacing: 8) {
                Text(formatTime(for: bar))
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                Text(formatVolume(bar.volume))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .position(x: min(max(x, 70), width - 70), y: 36)
    }

    // MARK: - Volume Formatting

    private func formatVolume(_ volume: Int) -> String {
        if volume >= 1_000_000 {
            return String(format: "%.1fM", Double(volume) / 1_000_000)
        } else if volume >= 1_000 {
            return String(format: "%.1fK", Double(volume) / 1_000)
        }
        return "\(volume)"
    }
}

// MARK: - OHLC Label Component

private struct OHLCLabel: View {
    let title: String
    let value: Double
    let color: Color

    var body: some View {
        HStack(spacing: 2) {
            Text(title)
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.tertiary)

            Text(String(format: "%.2f", value))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(color)
        }
    }
}

// MARK: - Empty Chart Placeholder

private struct EmptyChartPlaceholder: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.bar.fill")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(.tertiary)

            Text("No data available")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Preview

#Preview("Candlestick Chart") {
    VStack {
        CandlestickChart(
            bars: PriceBar.samples,
            isUp: true,
            supportLevels: [165.0],
            resistanceLevels: [180.0, 185.0],
            timeframe: .oneMonth
        )
        .frame(height: 200)
        .padding()
    }
}

#Preview("Empty Candlestick Chart") {
    CandlestickChart(
        bars: [],
        isUp: true,
        supportLevels: [],
        resistanceLevels: [],
        timeframe: .oneMonth
    )
    .frame(height: 200)
    .padding()
}
