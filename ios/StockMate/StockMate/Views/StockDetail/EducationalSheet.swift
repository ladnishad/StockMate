import SwiftUI

/// Sheet displaying educational content from the smart analysis
struct EducationalSheet: View {
    let educational: EducationalContent
    let symbol: String
    var tradePlan: EnhancedTradePlan?
    var bars: [PriceBar]

    @Environment(\.dismiss) private var dismiss

    init(educational: EducationalContent, symbol: String, tradePlan: EnhancedTradePlan? = nil, bars: [PriceBar] = []) {
        self.educational = educational
        self.symbol = symbol
        self.tradePlan = tradePlan
        self.bars = bars
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // Interactive Chart (if bars available)
                    if !bars.isEmpty {
                        chartSection
                    }

                    // Setup Explanation
                    setupExplanationSection

                    // What to Watch
                    if !educational.whatToWatch.isEmpty {
                        whatToWatchSection
                    }

                    // Scenarios
                    if !educational.scenarios.isEmpty {
                        scenariosSection
                    }

                    // Level Explanations
                    if !educational.levelExplanations.isEmpty {
                        levelExplanationsSection
                    }

                    // Risk Warnings
                    if !educational.riskWarnings.isEmpty {
                        riskWarningsSection
                    }
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Understanding \(symbol)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }

    // MARK: - Chart Section

    private var chartSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Visual Analysis", systemImage: "chart.xyaxis.line")
                .font(.headline)
                .foregroundColor(.purple)

            EducationalChartView(
                bars: bars,
                annotations: educational.chartAnnotations,
                pattern: nil, // Pattern detection would be added later
                entryZone: entryZone,
                stopLoss: tradePlan?.stopLoss,
                targets: targetPrices
            )
        }
    }

    private var entryZone: (low: Double, high: Double)? {
        guard let plan = tradePlan,
              let low = plan.entryZoneLow,
              let high = plan.entryZoneHigh else { return nil }
        return (low: low, high: high)
    }

    private var targetPrices: [Double] {
        tradePlan?.targets.map { $0.price } ?? []
    }

    // MARK: - Setup Explanation

    private var setupExplanationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("The Setup", systemImage: "lightbulb.fill")
                .font(.headline)
                .foregroundColor(.yellow)

            Text(educational.setupExplanation)
                .font(.body)
                .foregroundColor(.primary)
                .padding()
                .background(Color(.secondarySystemGroupedBackground))
                .cornerRadius(12)
        }
    }

    // MARK: - What to Watch

    private var whatToWatchSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("What to Watch", systemImage: "eye.fill")
                .font(.headline)
                .foregroundColor(.blue)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(educational.whatToWatch.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "\(index + 1).circle.fill")
                            .foregroundColor(.blue)
                            .font(.subheadline)

                        Text(item)
                            .font(.subheadline)
                            .foregroundColor(.primary)
                    }
                }
            }
            .padding()
            .background(Color(.secondarySystemGroupedBackground))
            .cornerRadius(12)
        }
    }

    // MARK: - Scenarios

    private var scenariosSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Possible Outcomes", systemImage: "arrow.triangle.branch")
                .font(.headline)
                .foregroundColor(.purple)

            VStack(spacing: 12) {
                ForEach(educational.scenarios) { scenario in
                    ScenarioCard(scenario: scenario)
                }
            }
        }
    }

    // MARK: - Level Explanations

    private var levelExplanationsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Key Levels", systemImage: "ruler.fill")
                .font(.headline)
                .foregroundColor(.orange)

            VStack(spacing: 8) {
                ForEach(Array(educational.levelExplanations.sorted(by: { $0.key > $1.key })), id: \.key) { level, explanation in
                    LevelExplanationRow(level: level, explanation: explanation)
                }
            }
            .padding()
            .background(Color(.secondarySystemGroupedBackground))
            .cornerRadius(12)
        }
    }

    // MARK: - Risk Warnings

    private var riskWarningsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Risk Warnings", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundColor(.red)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(educational.riskWarnings.enumerated()), id: \.offset) { _, warning in
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "exclamationmark.circle.fill")
                            .foregroundColor(.red)
                            .font(.subheadline)

                        Text(warning)
                            .font(.subheadline)
                            .foregroundColor(.primary)
                    }
                }
            }
            .padding()
            .background(Color.red.opacity(0.1))
            .cornerRadius(12)
        }
    }
}


// MARK: - Scenario Card

struct ScenarioCard: View {
    let scenario: ScenarioPath

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with icon, name, and probability
            HStack {
                Image(systemName: scenario.icon)
                    .foregroundColor(iconColor)
                    .font(.title3)

                Text(scenario.displayName)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(iconColor)

                Spacer()

                // Probability badge
                Text("\(scenario.probability)% likely")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(iconColor)
                    .cornerRadius(8)
            }

            // Description
            Text(scenario.description)
                .font(.subheadline)
                .foregroundColor(.primary)

            // Target price if available
            if let target = scenario.priceTarget {
                HStack {
                    Text("Target:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("$\(String(format: "%.2f", target))")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(iconColor)
                }
            }

            // Key trigger
            HStack(alignment: .top) {
                Image(systemName: "key.fill")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Trigger: \(scenario.keyTrigger)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(iconColor.opacity(0.1))
        .cornerRadius(12)
    }

    private var iconColor: Color {
        switch scenario.scenario {
        case "bullish": return .green
        case "bearish": return .red
        default: return .gray
        }
    }
}


// MARK: - Level Explanation Row

struct LevelExplanationRow: View {
    let level: String
    let explanation: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "mappin.circle.fill")
                    .foregroundColor(.orange)
                    .font(.caption)
                Text("$\(level)")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.orange)
            }

            Text(explanation)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}


// MARK: - Preview

#Preview {
    let mockScenarios = [
        ScenarioPath(
            scenario: "bullish",
            probability: 60,
            description: "Price holds support and breaks above the flag pattern, running to the $185-190 range where previous resistance sits.",
            priceTarget: 187.50,
            keyTrigger: "Daily close above $180 with volume > 1.5x average"
        ),
        ScenarioPath(
            scenario: "bearish",
            probability: 25,
            description: "Support breaks and price falls to the next major support zone around $165-170.",
            priceTarget: 167.00,
            keyTrigger: "Close below $172 on elevated volume"
        ),
        ScenarioPath(
            scenario: "sideways",
            probability: 15,
            description: "Price consolidates between $172-180 for several days before making a decisive move.",
            priceTarget: nil,
            keyTrigger: "Decreasing volume with tightening range"
        )
    ]

    let mockEducational = EducationalContent(
        setupExplanation: "This stock is forming a classic bull flag pattern after a strong earnings-driven breakout. Think of a bull flag like a pause in an uptrend - the price shoots up (the 'pole'), then consolidates in a tight range (the 'flag'). When it breaks out of this consolidation, it often continues in the original direction. The pullback to support at $175 offers a lower-risk entry compared to chasing the initial breakout.",
        levelExplanations: [
            "175.00": "Strong support - this level has been tested 3 times in the past week with buyers stepping in each time. It also aligns with the 21 EMA, adding confluence.",
            "180.00": "Resistance - the top of the flag pattern. A close above this level would confirm the breakout.",
            "185.00": "Major resistance - previous all-time high from October. First target area.",
            "172.00": "Stop loss level - below the flag low and key moving averages. If this breaks, the pattern is invalidated."
        ],
        whatToWatch: [
            "Watch for a green candle close above $180 with above-average volume - this confirms the breakout",
            "RSI should stay above 50 to confirm bullish momentum is intact",
            "If price closes below $172, exit the trade - the pattern would be broken",
            "Monitor overall market direction (SPY/QQQ) - if they sell off hard, this trade may fail"
        ],
        scenarios: mockScenarios,
        riskWarnings: [
            "Overall market is showing some weakness - watch SPY for confirmation",
            "Tech sector has been rotating - AAPL may face headwinds if rotation continues",
            "Earnings season approaching - increased volatility possible"
        ],
        chartAnnotations: []
    )

    return EducationalSheet(educational: mockEducational, symbol: "AAPL")
}
