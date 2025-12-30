import SwiftUI

/// Displays the AI-generated trading plan with live update support
struct TradingPlanView: View {
    @StateObject private var viewModel: TradingPlanViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showingEducational = false

    init(symbol: String) {
        _viewModel = StateObject(wrappedValue: TradingPlanViewModel(symbol: symbol))
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Check for active draft session first
                if viewModel.isDraftMode || viewModel.draftPlan != nil {
                    // Interactive draft plan mode (Claude Code-style)
                    DraftPlanView(viewModel: viewModel)

                } else if viewModel.isLoading && viewModel.plan == nil {
                    PlanLoadingView()
                } else if viewModel.isStreaming {
                    // Show streaming view while AI generates the plan
                    StreamingPlanView(viewModel: viewModel)
                } else if let plan = viewModel.plan {
                    // Show "Continue Editing" banner if there's an approved session
                    if viewModel.sessionStatus == "approved" && viewModel.sessionId != nil {
                        ApprovedSessionBanner(viewModel: viewModel)
                    }

                    // Approved plan content
                    PlanHeaderCard(plan: plan, viewModel: viewModel)

                    ThesisCard(plan: plan, isUpdating: viewModel.isUpdating)

                    PriceLadderCard(plan: plan, viewModel: viewModel)

                    KeyLevelsCard(plan: plan, viewModel: viewModel)

                    // Latest Evaluation (moved before Learn More)
                    if let notes = plan.evaluationNotes, !notes.isEmpty {
                        EvaluationCard(plan: plan)
                    }

                    // News & Sentiment (from web search)
                    if plan.hasNewsSentiment {
                        NewsSentimentCard(plan: plan)
                    }

                    // Learn More Button (at the end)
                    LearnMoreButton(viewModel: viewModel) {
                        showingEducational = true
                    }

                    // Action buttons
                    ActionButtonsRow(viewModel: viewModel)

                } else {
                    // No plan yet - show option to start interactive session
                    NoPlanViewWithSession(viewModel: viewModel)
                }
            }
            .padding(20)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(viewModel.isDraftMode ? "Draft Plan" : "Trading Plan")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if viewModel.isUpdating || viewModel.isProcessingFeedback {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
        }
        .task {
            await viewModel.loadPlan()
        }
        .onAppear {
            viewModel.startPolling()
        }
        .onDisappear {
            viewModel.stopPolling()
            viewModel.cancelStreaming()
        }
        .sheet(isPresented: $showingEducational) {
            if let educational = viewModel.educationalContent {
                EducationalSheet(
                    educational: educational,
                    symbol: viewModel.symbol,
                    tradePlan: viewModel.smartAnalysis?.tradePlan,
                    bars: viewModel.educationalBars
                )
            }
        }
    }
}

// MARK: - No Plan View with Session Option

private struct NoPlanViewWithSession: View {
    @ObservedObject var viewModel: TradingPlanViewModel

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 50))
                .foregroundColor(.secondary)

            VStack(spacing: 8) {
                Text("No Trading Plan")
                    .font(.title2.bold())

                Text("AI will analyze the stock and create a plan you can review and adjust")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

            Button {
                Task { await viewModel.startPlanSession() }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "sparkles")
                    Text("Generate Plan")
                }
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color.blue)
                .cornerRadius(12)
            }
            .disabled(viewModel.isLoading)
            .padding(.top, 8)
        }
        .padding(32)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

// MARK: - Plan Header Card

private struct PlanHeaderCard: View {
    let plan: TradingPlanResponse
    @ObservedObject var viewModel: TradingPlanViewModel

    var body: some View {
        VStack(spacing: 16) {
            // Trade Style and Confidence row
            HStack(alignment: .top) {
                // Trade style badge
                TradeStyleBadge(plan: plan)

                Spacer()

                // Confidence ring
                if let confidence = plan.confidence, confidence > 0 {
                    ConfidenceRing(confidence: confidence)
                }
            }

            // Bias and Status row
            HStack {
                // Bias badge
                HStack(spacing: 6) {
                    Image(systemName: biasIcon)
                        .font(.system(size: 14, weight: .semibold))
                    Text(plan.bias.uppercased())
                        .font(.system(size: 13, weight: .bold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(viewModel.biasColor)
                .clipShape(Capsule())

                Spacer()

                // Status badge
                HStack(spacing: 4) {
                    Circle()
                        .fill(viewModel.planStatusColor)
                        .frame(width: 8, height: 8)
                    Text(plan.status.capitalized)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }

            // Position-based indicator
            if viewModel.hasActivePosition {
                HStack(spacing: 6) {
                    Image(systemName: "person.fill.checkmark")
                        .font(.system(size: 11, weight: .medium))
                    Text("Based on your position")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundColor(.blue)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.blue.opacity(0.1))
                .clipShape(Capsule())
            }

            // Update phase indicator
            if viewModel.isUpdating {
                UpdatePhaseIndicator(phase: viewModel.updatePhase)
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
            }

            // Risk/Reward
            if let rr = plan.riskReward {
                HStack {
                    Label("Risk/Reward", systemImage: "arrow.left.arrow.right")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)

                    Spacer()

                    Text("1 : \(String(format: "%.1f", rr))")
                        .font(.system(size: 15, weight: .semibold, design: .monospaced))
                        .foregroundColor(rr >= 2 ? .green : rr >= 1.5 ? .orange : .red)
                }
            }

            // Last updated
            if let lastEval = plan.lastEvaluation, !lastEval.isEmpty {
                HStack {
                    Image(systemName: "clock")
                        .font(.system(size: 11))
                    Text("Last evaluated: \(formatDate(lastEval))")
                        .font(.system(size: 11))
                }
                .foregroundStyle(.tertiary)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: viewModel.isUpdating)
    }

    private var biasIcon: String {
        switch plan.bias.lowercased() {
        case "bullish": return "arrow.up.right"
        case "bearish": return "arrow.down.right"
        default: return "arrow.left.and.right"
        }
    }

    private func formatDate(_ isoString: String) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: isoString) {
            let relFormatter = RelativeDateTimeFormatter()
            relFormatter.unitsStyle = .abbreviated
            return relFormatter.localizedString(for: date, relativeTo: Date())
        }
        return isoString
    }
}

// MARK: - Trade Style Badge

private struct TradeStyleBadge: View {
    let plan: TradingPlanResponse

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: plan.tradeStyleIcon)
                    .font(.subheadline)
                Text(plan.tradeStyleDisplay.uppercased())
                    .font(.system(size: 12, weight: .bold))
            }
            .foregroundColor(styleColor)

            if let holdingPeriod = plan.holdingPeriod, !holdingPeriod.isEmpty {
                Text(holdingPeriod)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(styleColor.opacity(0.1))
        .cornerRadius(10)
    }

    private var styleColor: Color {
        guard let style = plan.tradeStyle else { return .blue }
        switch style.lowercased() {
        case "day": return .orange
        case "swing": return .blue
        case "position": return .purple
        default: return .blue
        }
    }
}

// MARK: - Confidence Ring

private struct ConfidenceRing: View {
    let confidence: Int

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.gray.opacity(0.2), lineWidth: 4)
                .frame(width: 50, height: 50)

            Circle()
                .trim(from: 0, to: CGFloat(confidence) / 100)
                .stroke(ringColor, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .frame(width: 50, height: 50)
                .rotationEffect(.degrees(-90))

            VStack(spacing: 0) {
                Text("\(confidence)")
                    .font(.system(size: 14, weight: .bold))
                Text("%")
                    .font(.system(size: 8))
                    .foregroundColor(.secondary)
            }
        }
    }

    private var ringColor: Color {
        if confidence >= 70 { return .green }
        if confidence >= 50 { return .yellow }
        return .orange
    }
}

// MARK: - Update Phase Indicator

private struct UpdatePhaseIndicator: View {
    let phase: TradingPlanViewModel.UpdatePhase

    var body: some View {
        HStack(spacing: 12) {
            // Animated dots
            HStack(spacing: 4) {
                ForEach(0..<3) { index in
                    Circle()
                        .fill(Color.blue)
                        .frame(width: 6, height: 6)
                        .scaleEffect(pulseScale(for: index))
                        .animation(
                            .easeInOut(duration: 0.6)
                            .repeatForever()
                            .delay(Double(index) * 0.2),
                            value: phase
                        )
                }
            }

            Text(phaseText)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.blue)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.blue.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var phaseText: String {
        switch phase {
        case .idle: return ""
        case .gatheringData: return "Gathering market data..."
        case .analyzing: return "Analyzing technicals..."
        case .generatingPlan: return "Generating plan..."
        case .complete: return "Plan updated!"
        }
    }

    private func pulseScale(for index: Int) -> CGFloat {
        phase != .idle && phase != .complete ? 1.0 : 0.6
    }
}

// MARK: - Thesis Card (Accordion)

private struct ThesisCard: View {
    let plan: TradingPlanResponse
    let isUpdating: Bool
    @State private var isExpanded: Bool = false

    /// Preview text - first ~100 characters of thesis
    private var previewText: String {
        let thesis = plan.thesis
        if thesis.count <= 100 {
            return thesis
        }
        let endIndex = thesis.index(thesis.startIndex, offsetBy: 100)
        return String(thesis[..<endIndex]) + "..."
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with expand/collapse button
            Button {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack {
                    Label("Thesis", systemImage: "lightbulb.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)

            // Content - collapsed or expanded
            VStack(alignment: .leading, spacing: 12) {
                if isExpanded {
                    // Full thesis
                    Text(plan.thesis)
                        .font(.system(size: 15, weight: .regular))
                        .foregroundStyle(.primary)
                        .lineSpacing(4)

                    if !plan.technicalSummary.isEmpty {
                        Divider()

                        Text(plan.technicalSummary)
                            .font(.system(size: 13, weight: .regular))
                            .foregroundStyle(.secondary)
                            .lineSpacing(2)
                    }
                } else {
                    // Preview only
                    Text(previewText)
                        .font(.system(size: 14, weight: .regular))
                        .foregroundStyle(.secondary)
                        .lineSpacing(3)
                        .lineLimit(2)
                }
            }
            .opacity(isUpdating ? 0.6 : 1)
            .overlay {
                if isUpdating {
                    ShimmerOverlay()
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

// MARK: - Price Ladder Card

private struct PriceLadderCard: View {
    let plan: TradingPlanResponse
    @ObservedObject var viewModel: TradingPlanViewModel

    private var isBearish: Bool {
        plan.bias.lowercased() == "bearish"
    }

    /// Map field names to their backend keys for adjustment tracking
    private func fieldKey(for shortLabel: String) -> String {
        switch shortLabel {
        case "SL": return "stop_loss"
        case "E↑": return "entry_zone_high"
        case "E↓": return "entry_zone_low"
        case "T1": return "target_1"
        case "T2": return "target_2"
        case "T3": return "target_3"
        default: return ""
        }
    }

    private var levels: [(String, Double?, Color, String)] {
        var result: [(String, Double?, Color, String)] = []

        // For BEARISH trades: Stop is above entry, targets are below
        // For BULLISH trades: Stop is below entry, targets are above

        // Stop Loss
        if let stop = plan.stopLoss {
            result.append(("Stop Loss", stop, .red, "SL"))
        }

        // Entry zone
        if let high = plan.entryZoneHigh {
            let label = isBearish ? "Short Entry High" : "Entry High"
            result.append((label, high, .blue, "E↑"))
        }
        if let low = plan.entryZoneLow {
            let label = isBearish ? "Short Entry Low" : "Entry Low"
            result.append((label, low, .blue, "E↓"))
        }

        // Targets
        if let t1 = plan.target1 {
            result.append(("Target 1", t1, .green, "T1"))
        }
        if let t2 = plan.target2 {
            result.append(("Target 2", t2, .green, "T2"))
        }
        if let t3 = plan.target3 {
            result.append(("Target 3", t3, .green, "T3"))
        }

        // Sort by price descending (highest at top)
        return result.sorted { ($0.1 ?? 0) > ($1.1 ?? 0) }
    }

    /// Check if any price level was adjusted in the last evaluation
    private var hasAdjustedLevels: Bool {
        let levelFields = ["stop_loss", "entry_zone_high", "entry_zone_low", "target_1", "target_2", "target_3"]
        return levelFields.contains { viewModel.wasAdjusted($0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Price Levels", systemImage: "chart.bar.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)

                // Updated badge if any levels were adjusted
                if hasAdjustedLevels {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.triangle.2.circlepath.circle.fill")
                            .font(.system(size: 10))
                        Text("Updated")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .foregroundColor(.blue)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.blue.opacity(0.12))
                    .clipShape(Capsule())
                }

                Spacer()

                // Direction indicator
                if isBearish {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.down.right")
                            .font(.system(size: 10, weight: .bold))
                        Text("SHORT")
                            .font(.system(size: 10, weight: .bold))
                    }
                    .foregroundColor(.red)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.red.opacity(0.12))
                    .clipShape(Capsule())
                } else {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 10, weight: .bold))
                        Text("LONG")
                            .font(.system(size: 10, weight: .bold))
                    }
                    .foregroundColor(.green)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.green.opacity(0.12))
                    .clipShape(Capsule())
                }
            }

            // Direction explanation for clarity
            if isBearish {
                HStack(spacing: 6) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 11))
                    Text("Profit when price falls below entry")
                        .font(.system(size: 11))
                }
                .foregroundStyle(.secondary)
                .padding(.bottom, 4)
            }

            // Price ladder visualization
            VStack(spacing: 0) {
                ForEach(Array(levels.enumerated()), id: \.offset) { index, level in
                    let fieldKey = fieldKey(for: level.3)
                    let wasAdjusted = viewModel.wasAdjusted(fieldKey)
                    let previousValue = viewModel.previousValue(for: fieldKey)

                    PriceLevelRow(
                        label: level.0,
                        price: level.1,
                        color: level.2,
                        shortLabel: level.3,
                        showConnector: index < levels.count - 1,
                        wasAdjusted: wasAdjusted,
                        previousPrice: previousValue
                    )
                }
            }

            // Stop reasoning
            if !plan.stopReasoning.isEmpty {
                Divider()
                    .padding(.vertical, 4)

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.shield")
                        .font(.system(size: 12))
                        .foregroundColor(.red)

                    Text(plan.stopReasoning)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
            }

            // Target reasoning
            if !plan.targetReasoning.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "target")
                        .font(.system(size: 12))
                        .foregroundColor(.green)

                    Text(plan.targetReasoning)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

private struct PriceLevelRow: View {
    let label: String
    let price: Double?
    let color: Color
    let shortLabel: String
    let showConnector: Bool
    var wasAdjusted: Bool = false
    var previousPrice: Double? = nil

    var body: some View {
        HStack(spacing: 12) {
            // Left indicator
            VStack(spacing: 0) {
                ZStack {
                    Circle()
                        .fill(color.opacity(0.2))
                        .frame(width: 28, height: 28)

                    Text(shortLabel)
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(color)
                }

                if showConnector {
                    Rectangle()
                        .fill(Color(.systemGray4))
                        .frame(width: 2, height: 24)
                }
            }

            // Label and price
            HStack {
                Text(label)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.secondary)

                Spacer()

                if let price = price {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(String(format: "$%.2f", price))
                            .font(.system(size: 15, weight: .semibold, design: .monospaced))
                            .foregroundColor(color)

                        // Show previous value if this level was adjusted
                        if wasAdjusted, let prev = previousPrice {
                            HStack(spacing: 3) {
                                Image(systemName: price > prev ? "arrow.up.circle.fill" : "arrow.down.circle.fill")
                                    .font(.system(size: 9))
                                Text("from $\(String(format: "%.2f", prev))")
                                    .font(.system(size: 10))
                            }
                            .foregroundColor(.blue)
                        }
                    }
                } else {
                    Text("—")
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.vertical, showConnector ? 0 : 8)
        }
    }
}

// MARK: - Key Levels Card

private struct KeyLevelsCard: View {
    let plan: TradingPlanResponse
    @ObservedObject var viewModel: TradingPlanViewModel

    /// Check if key levels were adjusted in the last evaluation
    private var hasAdjustedKeyLevels: Bool {
        viewModel.wasAdjusted("key_supports") || viewModel.wasAdjusted("key_resistances")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Key Levels to Watch", systemImage: "eye.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)

                // Updated badge if key levels were adjusted
                if hasAdjustedKeyLevels {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.triangle.2.circlepath.circle.fill")
                            .font(.system(size: 10))
                        Text("Updated")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .foregroundColor(.blue)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.blue.opacity(0.12))
                    .clipShape(Capsule())
                }

                Spacer()
            }

            HStack(alignment: .top, spacing: 16) {
                // Resistances
                VStack(alignment: .leading, spacing: 8) {
                    Label("Resistance", systemImage: "arrow.up.to.line")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)

                    if plan.keyResistances.isEmpty {
                        Text("—")
                            .foregroundStyle(.tertiary)
                    } else {
                        ForEach(plan.keyResistances.prefix(3), id: \.self) { level in
                            Text(String(format: "$%.2f", level))
                                .font(.system(size: 13, weight: .medium, design: .monospaced))
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Divider()

                // Supports
                VStack(alignment: .leading, spacing: 8) {
                    Label("Support", systemImage: "arrow.down.to.line")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)

                    if plan.keySupports.isEmpty {
                        Text("—")
                            .foregroundStyle(.tertiary)
                    } else {
                        ForEach(plan.keySupports.prefix(3), id: \.self) { level in
                            Text(String(format: "$%.2f", level))
                                .font(.system(size: 13, weight: .medium, design: .monospaced))
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            // Invalidation criteria
            if !plan.invalidationCriteria.isEmpty {
                Divider()
                    .padding(.vertical, 4)

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "xmark.octagon")
                        .font(.system(size: 12))
                        .foregroundColor(.orange)

                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text("Invalidation")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.secondary)

                            // Show if invalidation criteria was updated
                            if viewModel.wasAdjusted("invalidation_criteria") {
                                Text("Updated")
                                    .font(.system(size: 9, weight: .medium))
                                    .foregroundColor(.blue)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.blue.opacity(0.12))
                                    .clipShape(Capsule())
                            }
                        }
                        Text(plan.invalidationCriteria)
                            .font(.system(size: 12))
                            .foregroundStyle(.primary)
                    }
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

// MARK: - News & Sentiment Card

private struct NewsSentimentCard: View {
    let plan: TradingPlanResponse

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Market Sentiment", systemImage: "newspaper.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)

            // Reddit sentiment badge
            if !plan.redditSentimentDisplay.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "bubble.left.and.bubble.right.fill")
                        .font(.system(size: 14))
                        .foregroundColor(sentimentColor)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Reddit Buzz")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)

                        HStack(spacing: 6) {
                            Text(plan.redditSentimentDisplay)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(sentimentColor)

                            if let buzz = plan.redditBuzz, !buzz.isEmpty {
                                Text("•")
                                    .foregroundStyle(.tertiary)
                                Text(buzz)
                                    .font(.system(size: 12))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                        }
                    }

                    Spacer()
                }
                .padding(12)
                .background(sentimentColor.opacity(0.1))
                .cornerRadius(10)
            }

            // News summary
            if let news = plan.newsSummary, !news.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 12))
                        .foregroundColor(.blue)
                        .padding(.top, 2)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Recent News")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text(news)
                            .font(.system(size: 13))
                            .foregroundStyle(.primary)
                            .lineSpacing(2)
                    }
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private var sentimentColor: Color {
        switch plan.redditSentiment?.lowercased() {
        case "bullish": return .green
        case "bearish": return .red
        case "mixed": return .orange
        default: return .gray
        }
    }
}

// MARK: - Evaluation Card

private struct EvaluationCard: View {
    let plan: TradingPlanResponse

    /// Parse evaluation notes and render with proper formatting
    private var formattedNotes: Text {
        guard let notes = plan.evaluationNotes else { return Text("") }
        return parseMarkdown(notes)
    }

    /// Simple markdown parser for bold (**text**) and sections
    private func parseMarkdown(_ text: String) -> Text {
        var result = Text("")
        var remaining = text

        while !remaining.isEmpty {
            // Look for **bold** patterns
            if let boldStart = remaining.range(of: "**") {
                // Add text before the bold marker
                let beforeBold = String(remaining[..<boldStart.lowerBound])
                if !beforeBold.isEmpty {
                    result = result + Text(beforeBold)
                }

                // Find the closing **
                let afterStart = remaining[boldStart.upperBound...]
                if let boldEnd = afterStart.range(of: "**") {
                    let boldText = String(afterStart[..<boldEnd.lowerBound])
                    result = result + Text(boldText).bold()
                    remaining = String(afterStart[boldEnd.upperBound...])
                } else {
                    // No closing **, just add the rest as-is
                    result = result + Text(String(remaining[boldStart.lowerBound...]))
                    remaining = ""
                }
            } else {
                // No more bold markers, add the rest
                result = result + Text(remaining)
                remaining = ""
            }
        }

        return result
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Latest Evaluation", systemImage: "checkmark.circle.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)

                Spacer()

                if plan.status.lowercased() == "active" {
                    Text("VALID")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.green)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.green.opacity(0.15))
                        .clipShape(Capsule())
                } else if plan.status.lowercased() == "invalidated" {
                    Text("INVALIDATED")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.orange)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.orange.opacity(0.15))
                        .clipShape(Capsule())
                }
            }

            formattedNotes
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(.primary)
                .lineSpacing(4)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

// MARK: - Learn More Button

private struct LearnMoreButton: View {
    @ObservedObject var viewModel: TradingPlanViewModel
    let onTap: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Button {
                Task {
                    await viewModel.loadEducationalContent()
                    // Only show sheet after content is loaded
                    if viewModel.educationalContent != nil {
                        onTap()
                    }
                }
            } label: {
                HStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Color.purple.opacity(0.8), Color.blue.opacity(0.8)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 40, height: 40)

                        Image(systemName: "graduationcap.fill")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(.white)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Learn More")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.primary)

                        Text("Understand this setup like a pro")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if viewModel.isLoadingEducational {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(.tertiary)
                    }
                }
                .padding(16)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color(.secondarySystemBackground))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.purple.opacity(0.3), Color.blue.opacity(0.3)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isLoadingEducational)

            // Error message
            if let error = viewModel.educationalError {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 12))
                        .foregroundColor(.orange)

                    Text(error)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .lineLimit(2)

                    Spacer()

                    Button {
                        viewModel.clearEducationalError()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 14))
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }
        }
    }
}

// MARK: - Action Buttons

private struct ActionButtonsRow: View {
    @ObservedObject var viewModel: TradingPlanViewModel

    var body: some View {
        VStack(spacing: 12) {
            // Primary row: Evaluate and Modify
            HStack(spacing: 12) {
                // Evaluate button
                Button {
                    Task {
                        await viewModel.evaluatePlan()
                    }
                } label: {
                    Label("Evaluate", systemImage: "arrow.triangle.2.circlepath")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isUpdating || viewModel.isStreaming)

                // Modify button - starts interactive session from existing plan
                Button {
                    Task {
                        await viewModel.startSessionFromExisting()
                    }
                } label: {
                    Label("Modify", systemImage: "slider.horizontal.3")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .disabled(viewModel.isUpdating || viewModel.isStreaming || viewModel.isLoading)
            }

            // Secondary row: Start fresh with new plan
            Button {
                Task {
                    viewModel.clearSession()  // Clear any existing session state
                    await viewModel.startPlanSession()  // Start fresh session
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.system(size: 12))
                    Text("Start Fresh")
                        .font(.system(size: 13, weight: .medium))
                }
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(Color(.tertiarySystemBackground))
                .cornerRadius(10)
            }
            .disabled(viewModel.isUpdating || viewModel.isLoading)
        }
        .padding(.top, 8)
    }
}

// MARK: - Approved Session Banner

private struct ApprovedSessionBanner: View {
    @ObservedObject var viewModel: TradingPlanViewModel

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 18))
                .foregroundColor(.green)

            VStack(alignment: .leading, spacing: 2) {
                Text("Plan Approved")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.primary)

                Text("You can continue editing if needed")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }

            Spacer()

            Button {
                Task {
                    await viewModel.reopenSession()
                }
            } label: {
                Text("Continue")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(8)
            }
            .disabled(viewModel.isUpdating)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.green.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.green.opacity(0.3), lineWidth: 1)
        )
    }
}

// MARK: - Plan Loading View

private struct PlanLoadingView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)

            Text("Loading plan...")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 80)
    }
}

// MARK: - Shimmer Overlay

private struct ShimmerOverlay: View {
    @State private var phase: CGFloat = 0

    var body: some View {
        GeometryReader { geometry in
            LinearGradient(
                gradient: Gradient(colors: [
                    .clear,
                    Color.white.opacity(0.3),
                    .clear
                ]),
                startPoint: .leading,
                endPoint: .trailing
            )
            .frame(width: geometry.size.width * 0.5)
            .offset(x: -geometry.size.width * 0.25 + (geometry.size.width * 1.5) * phase)
            .onAppear {
                withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                    phase = 1
                }
            }
        }
        .clipped()
    }
}

// MARK: - Preview

#Preview("Trading Plan") {
    NavigationStack {
        TradingPlanView(symbol: "AAPL")
    }
}
