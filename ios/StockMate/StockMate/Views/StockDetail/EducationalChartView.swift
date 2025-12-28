import SwiftUI
import Charts

/// Educational chart view that displays price data with pattern overlays and annotations
/// Used in the "Learn More" section to visually explain trading setups
struct EducationalChartView: View {
    let bars: [PriceBar]
    let annotations: [ChartAnnotation]
    let pattern: PatternOverlay?
    let entryZone: (low: Double, high: Double)?
    let stopLoss: Double?
    let targets: [Double]

    @State private var selectedAnnotation: ChartAnnotation?
    @State private var showAnnotationDetail = false

    var body: some View {
        VStack(spacing: 12) {
            // Chart title
            if let pattern = pattern {
                PatternHeader(pattern: pattern)
            }

            // Main chart
            if bars.isEmpty {
                EmptyEducationalChartView()
            } else {
                chartContent
            }

            // Selected annotation detail
            if let annotation = selectedAnnotation {
                AnnotationDetailCard(annotation: annotation) {
                    withAnimation {
                        selectedAnnotation = nil
                    }
                }
            }

            // Legend
            ChartLegend()
        }
    }

    private var chartContent: some View {
        Chart {
            // Entry zone (blue shaded area)
            if let entryZone = entryZone {
                RectangleMark(
                    xStart: .value("Start", 0),
                    xEnd: .value("End", bars.count - 1),
                    yStart: .value("Low", entryZone.low),
                    yEnd: .value("High", entryZone.high)
                )
                .foregroundStyle(Color.blue.opacity(0.15))
            }

            // Stop loss line
            if let stopLoss = stopLoss {
                RuleMark(y: .value("Stop", stopLoss))
                    .foregroundStyle(Color.red.opacity(0.7))
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [8, 4]))
                    .annotation(position: .trailing, alignment: .trailing) {
                        Text("STOP")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.red)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                            .background(Color.red.opacity(0.2))
                            .cornerRadius(4)
                    }
            }

            // Target lines
            ForEach(Array(targets.enumerated()), id: \.offset) { index, target in
                RuleMark(y: .value("Target", target))
                    .foregroundStyle(Color.green.opacity(0.7))
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [8, 4]))
                    .annotation(position: .trailing, alignment: .trailing) {
                        Text("T\(index + 1)")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.green)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.2))
                            .cornerRadius(4)
                    }
            }

            // Chart annotations (support/resistance levels)
            ForEach(annotations) { annotation in
                if annotation.type == "level", let price = annotation.price {
                    RuleMark(y: .value("Level", price))
                        .foregroundStyle(annotationColor(annotation.color).opacity(0.6))
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
                        .annotation(position: .leading, alignment: .leading) {
                            Button {
                                withAnimation(.spring(response: 0.3)) {
                                    selectedAnnotation = annotation
                                }
                            } label: {
                                Text(annotation.label)
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(annotationColor(annotation.color))
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 2)
                                    .background(annotationColor(annotation.color).opacity(0.15))
                                    .cornerRadius(4)
                            }
                        }
                } else if annotation.type == "zone",
                          let priceHigh = annotation.priceHigh,
                          let priceLow = annotation.priceLow {
                    RectangleMark(
                        xStart: .value("Start", 0),
                        xEnd: .value("End", bars.count - 1),
                        yStart: .value("Low", priceLow),
                        yEnd: .value("High", priceHigh)
                    )
                    .foregroundStyle(annotationColor(annotation.color).opacity(0.1))
                }
            }

            // Candlestick bodies and wicks
            ForEach(Array(bars.enumerated()), id: \.element.id) { index, bar in
                // Highlight bars that are part of the pattern
                let isPatternBar = pattern?.highlightedBarIndices.contains(index) ?? false
                let opacity = isPatternBar ? 1.0 : 0.6

                // Wick
                RuleMark(
                    x: .value("Index", index),
                    yStart: .value("Low", bar.low),
                    yEnd: .value("High", bar.high)
                )
                .foregroundStyle(bar.isUp ? Color.green.opacity(opacity) : Color.red.opacity(opacity))
                .lineStyle(StrokeStyle(lineWidth: 1))

                // Body
                RectangleMark(
                    x: .value("Index", index),
                    yStart: .value("Open", bar.open),
                    yEnd: .value("Close", bar.close),
                    width: .fixed(max(3, min(10, 300 / Double(bars.count))))
                )
                .foregroundStyle(bar.isUp ? Color.green.opacity(opacity) : Color.red.opacity(opacity))
            }

            // Pattern overlay lines (trend lines, flag boundaries, etc.)
            if let pattern = pattern {
                ForEach(pattern.lines) { line in
                    LineMark(
                        x: .value("X1", line.startIndex),
                        y: .value("Y1", line.startPrice)
                    )
                    .foregroundStyle(patternLineColor(line.style))
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: line.style == .dashed ? [6, 3] : []))

                    LineMark(
                        x: .value("X2", line.endIndex),
                        y: .value("Y2", line.endPrice)
                    )
                    .foregroundStyle(patternLineColor(line.style))
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: line.style == .dashed ? [6, 3] : []))
                }
            }
        }
        .chartXAxis(.hidden)
        .chartYAxis {
            AxisMarks(position: .trailing) { value in
                AxisValueLabel {
                    if let price = value.as(Double.self) {
                        Text(String(format: "$%.2f", price))
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.3))
                    .foregroundStyle(Color(.systemGray4))
            }
        }
        .chartYScale(domain: yAxisDomain)
        .frame(height: 280)
        .padding(.horizontal, 8)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }

    private var yAxisDomain: ClosedRange<Double> {
        guard !bars.isEmpty else { return 0...100 }

        var allPrices = bars.flatMap { [$0.low, $0.high] }

        // Include annotations
        for annotation in annotations {
            if let price = annotation.price { allPrices.append(price) }
            if let high = annotation.priceHigh { allPrices.append(high) }
            if let low = annotation.priceLow { allPrices.append(low) }
        }

        // Include entry zone, stop, targets
        if let entryZone = entryZone {
            allPrices.append(contentsOf: [entryZone.low, entryZone.high])
        }
        if let stop = stopLoss { allPrices.append(stop) }
        allPrices.append(contentsOf: targets)

        let minPrice = allPrices.min() ?? 0
        let maxPrice = allPrices.max() ?? 100
        let padding = (maxPrice - minPrice) * 0.08

        return (minPrice - padding)...(maxPrice + padding)
    }

    private func annotationColor(_ color: String) -> Color {
        switch color.lowercased() {
        case "green": return .green
        case "red": return .red
        case "blue": return .blue
        case "yellow": return .yellow
        case "orange": return .orange
        case "purple": return .purple
        case "teal", "cyan": return .teal
        case "pink": return .pink
        default: return .gray
        }
    }

    private func patternLineColor(_ style: PatternLine.LineStyle) -> Color {
        switch style {
        case .solid: return .purple
        case .dashed: return .orange
        case .dotted: return .blue
        }
    }
}

// MARK: - Pattern Overlay Model

/// Represents a pattern to be drawn on the chart
struct PatternOverlay: Identifiable {
    let id = UUID()
    let name: String
    let type: PatternType
    let description: String
    let lines: [PatternLine]
    let highlightedBarIndices: Set<Int>

    enum PatternType: String {
        case bullFlag = "Bull Flag"
        case bearFlag = "Bear Flag"
        case triangle = "Triangle"
        case headAndShoulders = "Head & Shoulders"
        case doubleBottom = "Double Bottom"
        case doubleTop = "Double Top"
        case cup = "Cup & Handle"
        case wedge = "Wedge"
        case channel = "Channel"
        case custom = "Pattern"
    }
}

/// A line to be drawn as part of a pattern
struct PatternLine: Identifiable {
    let id = UUID()
    let startIndex: Int
    let startPrice: Double
    let endIndex: Int
    let endPrice: Double
    let style: LineStyle
    let label: String?

    enum LineStyle {
        case solid
        case dashed
        case dotted
    }
}

// MARK: - Supporting Views

private struct PatternHeader: View {
    let pattern: PatternOverlay

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: patternIcon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(.purple)

            VStack(alignment: .leading, spacing: 2) {
                Text(pattern.name)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(pattern.description)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.purple.opacity(0.1))
        )
    }

    private var patternIcon: String {
        switch pattern.type {
        case .bullFlag, .bearFlag: return "flag.fill"
        case .triangle, .wedge: return "triangle.fill"
        case .headAndShoulders: return "person.fill"
        case .doubleBottom, .doubleTop: return "w.square.fill"
        case .cup: return "cup.and.saucer.fill"
        case .channel: return "arrow.left.and.right"
        case .custom: return "chart.xyaxis.line"
        }
    }
}

private struct AnnotationDetailCard: View {
    let annotation: ChartAnnotation
    let onDismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(colorForAnnotation)
                .frame(width: 8, height: 8)
                .padding(.top, 6)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(annotation.label)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.primary)

                    if let price = annotation.price {
                        Text("$\(String(format: "%.2f", price))")
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundColor(colorForAnnotation)
                    }
                }

                Text(annotation.description)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()

            Button(action: onDismiss) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 18))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(.tertiarySystemGroupedBackground))
        )
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private var colorForAnnotation: Color {
        switch annotation.color.lowercased() {
        case "green": return .green
        case "red": return .red
        case "blue": return .blue
        case "yellow": return .yellow
        case "orange": return .orange
        case "purple": return .purple
        default: return .gray
        }
    }
}

private struct ChartLegend: View {
    var body: some View {
        HStack(spacing: 16) {
            LegendItem(color: .blue.opacity(0.3), label: "Entry Zone")
            LegendItem(color: .red, label: "Stop Loss", isDashed: true)
            LegendItem(color: .green, label: "Targets", isDashed: true)
            LegendItem(color: .purple, label: "Pattern")
        }
        .font(.system(size: 10))
        .foregroundStyle(.secondary)
    }
}

private struct LegendItem: View {
    let color: Color
    let label: String
    var isDashed: Bool = false

    var body: some View {
        HStack(spacing: 4) {
            if isDashed {
                Rectangle()
                    .fill(color)
                    .frame(width: 12, height: 2)
                    .overlay(
                        Rectangle()
                            .stroke(style: StrokeStyle(lineWidth: 2, dash: [3, 2]))
                            .foregroundColor(color)
                    )
            } else {
                Rectangle()
                    .fill(color)
                    .frame(width: 12, height: 8)
                    .cornerRadius(2)
            }

            Text(label)
        }
    }
}

private struct EmptyEducationalChartView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "chart.xyaxis.line")
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(.tertiary)

            Text("Chart data unavailable")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(height: 200)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

// MARK: - Preview

#Preview("Educational Chart with Bull Flag") {
    let bars = PriceBar.samples

    // Create a bull flag pattern overlay
    let pattern = PatternOverlay(
        name: "Bull Flag Pattern",
        type: .bullFlag,
        description: "Consolidation after strong upward move. Breakout above flag resistance confirms continuation.",
        lines: [
            // Flag upper boundary
            PatternLine(startIndex: 20, startPrice: 178, endIndex: 28, endPrice: 176, style: .dashed, label: "Resistance"),
            // Flag lower boundary
            PatternLine(startIndex: 20, startPrice: 174, endIndex: 28, endPrice: 172, style: .dashed, label: "Support"),
        ],
        highlightedBarIndices: Set(20...28)
    )

    let annotations = [
        ChartAnnotation(
            type: "level",
            price: 175,
            priceHigh: nil,
            priceLow: nil,
            label: "Support",
            color: "green",
            description: "Strong support - tested 3 times with buyers stepping in each time. Also aligns with 21 EMA."
        ),
        ChartAnnotation(
            type: "level",
            price: 180,
            priceHigh: nil,
            priceLow: nil,
            label: "Resistance",
            color: "red",
            description: "Key resistance - previous high. A close above confirms breakout."
        )
    ]

    return ScrollView {
        EducationalChartView(
            bars: bars,
            annotations: annotations,
            pattern: pattern,
            entryZone: (low: 173, high: 176),
            stopLoss: 170,
            targets: [182, 187]
        )
        .padding()
    }
}
