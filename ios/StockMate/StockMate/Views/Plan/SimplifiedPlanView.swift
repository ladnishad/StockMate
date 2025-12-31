import SwiftUI

// MARK: - Main View

struct SimplifiedPlanView: View {
    @StateObject private var viewModel: TradingPlanViewModel
    @State private var feedbackText: String = ""
    @State private var showSavedConfirmation: Bool = false
    @FocusState private var isInputFocused: Bool

    let symbol: String

    init(symbol: String) {
        self.symbol = symbol
        _viewModel = StateObject(wrappedValue: TradingPlanViewModel(symbol: symbol))
    }

    /// The plan to display - either the approved plan or the draft plan
    private var displayPlan: TradingPlanResponse? {
        viewModel.plan ?? viewModel.draftPlan
    }

    /// Whether we're in draft mode (session with unapproved plan)
    private var isDraftMode: Bool {
        viewModel.draftPlan != nil && viewModel.plan == nil
    }

    /// Plan is saved when we have an approved plan (not draft)
    private var isPlanSaved: Bool {
        viewModel.plan != nil && viewModel.draftPlan == nil
    }

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()

            if viewModel.isLoading && displayPlan == nil {
                AgentGeneratingView(symbol: symbol, steps: viewModel.analysisSteps)
            } else if let plan = displayPlan {
                planContent(plan)
            } else {
                EmptyPlanView {
                    Task { await viewModel.startPlanSession() }
                }
            }
        }
        .task { await viewModel.loadPlan() }
    }

    // MARK: - Plan Content

    @ViewBuilder
    private func planContent(_ plan: TradingPlanResponse) -> some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    PlanHeaderView(
                        symbol: plan.symbol,
                        tradeStyle: plan.tradeStyle ?? "swing",
                        holdingPeriod: plan.holdingPeriod ?? "2-5 days"
                    )

                    // Bias Indicator
                    BiasIndicatorView(
                        bias: plan.bias,
                        confidence: plan.confidence ?? 70
                    )

                    // Thesis
                    ThesisSectionView(text: plan.thesis)

                    // Price Ladder
                    PriceLadderView(
                        stopLoss: plan.stopLoss,
                        entryLow: plan.entryZoneLow,
                        entryHigh: plan.entryZoneHigh,
                        target1: plan.target1,
                        target2: plan.target2,
                        target3: plan.target3,
                        bias: plan.bias
                    )

                    // Risk/Reward
                    if let rr = plan.riskReward {
                        RiskRewardBadgeView(ratio: rr)
                    }

                    // Key Levels (Supports & Resistances)
                    if !plan.keySupports.isEmpty || !plan.keyResistances.isEmpty {
                        KeyLevelsSectionView(
                            supports: plan.keySupports,
                            resistances: plan.keyResistances,
                            invalidation: plan.invalidationCriteria
                        )
                    }

                    // Market Sentiment (News + Reddit)
                    if plan.hasNewsSentiment {
                        MarketSentimentSectionView(
                            newsSummary: plan.newsSummary,
                            redditSentiment: plan.redditSentiment,
                            redditBuzz: plan.redditBuzz
                        )
                    }

                    // Evaluation Status Section (enhanced)
                    EvaluationStatusSectionView(
                        viewModel: viewModel,
                        plan: plan
                    )
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 140)
            }

            // Bottom Action Bar - conditional based on saved state
            if isDraftMode {
                // Draft mode: show Accept Plan button
                PlanActionBar(
                    feedbackText: $feedbackText,
                    isInputFocused: $isInputFocused,
                    isProcessing: viewModel.isUpdating || viewModel.isProcessingFeedback,
                    isDraftMode: isDraftMode,
                    onSubmit: {
                        guard !feedbackText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                        Task {
                            await viewModel.submitFeedback(feedbackText)
                            feedbackText = ""
                        }
                    },
                    onAccept: {
                        Task {
                            let success = await viewModel.acceptPlan()
                            if success {
                                // Show saved confirmation
                                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                                    showSavedConfirmation = true
                                }
                                // Auto-dismiss after 2.5 seconds
                                try? await Task.sleep(nanoseconds: 2_500_000_000)
                                withAnimation(.easeOut(duration: 0.3)) {
                                    showSavedConfirmation = false
                                }
                            }
                        }
                    },
                    onStartOver: {
                        Task { await viewModel.startOver() }
                    }
                )
            } else {
                // Saved mode: show saved confirmation and regenerate button
                SavedPlanBar(
                    feedbackText: $feedbackText,
                    isInputFocused: $isInputFocused,
                    isProcessing: viewModel.isUpdating || viewModel.isProcessingFeedback,
                    showSavedConfirmation: $showSavedConfirmation,
                    onSubmit: {
                        guard !feedbackText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                        Task {
                            await viewModel.submitFeedback(feedbackText)
                            feedbackText = ""
                        }
                    },
                    onRegenerate: {
                        Task { await viewModel.startOver() }
                    }
                )
            }
        }
    }
}

// MARK: - Plan Header

private struct PlanHeaderView: View {
    let symbol: String
    let tradeStyle: String
    let holdingPeriod: String

    var body: some View {
        VStack(spacing: 6) {
            Text(symbol)
                .font(.system(size: 32, weight: .bold, design: .rounded))
                .foregroundColor(.primary)
                .tracking(-0.5)

            Text("\(tradeStyle.capitalized) Trade \u{2022} \(holdingPeriod)")
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.secondary)
                .textCase(.uppercase)
                .tracking(1.0)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
    }
}

// MARK: - Bias Indicator

private struct BiasIndicatorView: View {
    let bias: String
    let confidence: Int

    private var isBullish: Bool { bias.lowercased() == "bullish" }
    private var accentColor: Color {
        isBullish ? Color(hex: "10B981") : Color(hex: "F87171")
    }

    var body: some View {
        HStack(spacing: 16) {
            // Bias pill
            Text(bias.uppercased())
                .font(.system(size: 12, weight: .bold))
                .tracking(1.5)
                .foregroundColor(accentColor)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(
                    Capsule()
                        .fill(accentColor.opacity(0.15))
                )

            // Confidence bar
            HStack(spacing: 10) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        // Track
                        Capsule()
                            .fill(Color(.tertiarySystemFill))

                        // Fill
                        Capsule()
                            .fill(accentColor)
                            .frame(width: geo.size.width * CGFloat(confidence) / 100)
                    }
                }
                .frame(width: 80, height: 4)

                Text("\(confidence)%")
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundColor(.secondary)
            }
        }
    }
}

// MARK: - Thesis Section

private struct ThesisSectionView: View {
    let text: String

    var body: some View {
        MarkdownText(text, size: 15, opacity: 0.9)
            .lineSpacing(5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(.secondarySystemBackground))
            )
    }
}

// MARK: - Price Ladder

private struct PriceLadderView: View {
    let stopLoss: Double?
    let entryLow: Double?
    let entryHigh: Double?
    let target1: Double?
    let target2: Double?
    let target3: Double?
    let bias: String

    private var isBullish: Bool { bias.lowercased() == "bullish" }
    private var accentColor: Color {
        isBullish ? Color(hex: "10B981") : Color(hex: "F87171")
    }

    var body: some View {
        VStack(spacing: 0) {
            // Build price levels from top to bottom (highest first)
            if let t3 = target3 {
                PriceLevelRow(
                    label: "T3",
                    price: t3,
                    style: .target,
                    accentColor: accentColor
                )
            }

            if let t2 = target2 {
                PriceLevelRow(
                    label: "T2",
                    price: t2,
                    style: .target,
                    accentColor: accentColor
                )
            }

            if let t1 = target1 {
                PriceLevelRow(
                    label: "T1",
                    price: t1,
                    style: .target,
                    accentColor: accentColor
                )
            }

            // Entry zone
            if let high = entryHigh, let low = entryLow {
                EntryZoneRowView(
                    high: high,
                    low: low,
                    accentColor: accentColor
                )
            }

            if let sl = stopLoss {
                PriceLevelRow(
                    label: "SL",
                    price: sl,
                    style: .stop,
                    accentColor: Color(hex: "F87171")
                )
            }
        }
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color(.separator).opacity(0.5), lineWidth: 1)
        )
    }
}

private struct PriceLevelRow: View {
    let label: String
    let price: Double
    let style: PriceLevelStyle
    let accentColor: Color

    enum PriceLevelStyle {
        case target, stop
    }

    private var lineColor: Color {
        style == .stop ? Color(hex: "F87171").opacity(0.4) : accentColor.opacity(0.3)
    }

    var body: some View {
        HStack(spacing: 12) {
            // Label
            Text(label)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(style == .stop ? Color(hex: "F87171").opacity(0.8) : accentColor.opacity(0.8))
                .frame(width: 28, alignment: .leading)

            // Line
            if style == .stop {
                // Dashed line for stop
                LineShape()
                    .stroke(style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .foregroundColor(lineColor)
                    .frame(height: 1)
            } else {
                Rectangle()
                    .fill(lineColor)
                    .frame(height: 2)
            }

            // Price
            Text(formatPrice(price))
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundColor(.primary)
                .frame(width: 80, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func formatPrice(_ price: Double) -> String {
        if price >= 1000 {
            return String(format: "$%.0f", price)
        } else if price >= 100 {
            return String(format: "$%.1f", price)
        } else {
            return String(format: "$%.2f", price)
        }
    }
}

private struct LineShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: 0, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.width, y: rect.midY))
        return path
    }
}

private struct EntryZoneRowView: View {
    let high: Double
    let low: Double
    let accentColor: Color

    var body: some View {
        HStack(spacing: 12) {
            Text("ENTRY")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(accentColor)
                .frame(width: 40, alignment: .leading)

            // Entry zone bar
            RoundedRectangle(cornerRadius: 4)
                .fill(accentColor.opacity(0.2))
                .frame(height: 28)
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .strokeBorder(accentColor.opacity(0.4), lineWidth: 1)
                )

            // Price range
            VStack(alignment: .trailing, spacing: 2) {
                Text(formatPrice(high))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
                Text(formatPrice(low))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            .frame(width: 68)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(accentColor.opacity(0.08))
    }

    private func formatPrice(_ price: Double) -> String {
        if price >= 1000 {
            return String(format: "$%.0f", price)
        } else if price >= 100 {
            return String(format: "$%.1f", price)
        } else {
            return String(format: "$%.2f", price)
        }
    }
}

// MARK: - Risk/Reward Badge

private struct RiskRewardBadgeView: View {
    let ratio: Double

    var body: some View {
        HStack(spacing: 8) {
            Text("R:R")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)

            Text(String(format: "%.1f:1", ratio))
                .font(.system(size: 17, weight: .bold, design: .monospaced))
                .foregroundColor(ratio >= 2 ? Color(hex: "10B981") : .primary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(
            Capsule()
                .fill(Color(.tertiarySystemBackground))
        )
    }
}

// MARK: - Action Bar

private struct PlanActionBar: View {
    @Binding var feedbackText: String
    var isInputFocused: FocusState<Bool>.Binding
    let isProcessing: Bool
    let isDraftMode: Bool
    let onSubmit: () -> Void
    let onAccept: () -> Void
    let onStartOver: () -> Void

    private var acceptButtonText: String {
        isDraftMode ? "Accept Plan" : "Done"
    }

    private var showSendButton: Bool {
        !feedbackText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 12) {
            // Divider
            Divider()

            // Text input row
            HStack(spacing: 10) {
                TextField("Ask something or suggest changes...", text: $feedbackText, axis: .vertical)
                    .font(.system(size: 15))
                    .foregroundColor(.primary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Color(.secondarySystemBackground))
                    )
                    .focused(isInputFocused)
                    .lineLimit(1...3)
                    .disabled(isProcessing)
                    .submitLabel(.send)
                    .onSubmit {
                        onSubmit()
                    }

                // Send button - always in layout, just hidden when empty
                Button(action: onSubmit) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 30))
                        .foregroundColor(Color(hex: "10B981"))
                }
                .disabled(isProcessing || !showSendButton)
                .opacity(showSendButton ? 1 : 0)
                .scaleEffect(showSendButton ? 1 : 0.5)
                .animation(.easeInOut(duration: 0.15), value: showSendButton)
            }
            .padding(.horizontal, 16)

            // Action buttons
            HStack(spacing: 12) {
                // Start Over
                Button(action: onStartOver) {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 13, weight: .semibold))
                        Text("Start Over")
                            .font(.system(size: 14, weight: .semibold))
                    }
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(Color(.secondarySystemBackground))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(Color(.separator).opacity(0.5), lineWidth: 1)
                    )
                }
                .disabled(isProcessing)

                // Accept / Done
                Button(action: onAccept) {
                    HStack(spacing: 6) {
                        if isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: isDraftMode ? "checkmark.circle.fill" : "checkmark")
                                .font(.system(size: isDraftMode ? 16 : 13, weight: .bold))
                        }
                        Text(acceptButtonText)
                            .font(.system(size: 15, weight: .bold))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [Color(hex: "10B981"), Color(hex: "059669")],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                    )
                    .shadow(color: Color(hex: "10B981").opacity(0.3), radius: 4, x: 0, y: 2)
                }
                .disabled(isProcessing)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .background(Color(.systemBackground))
    }
}

// MARK: - Saved Plan Bar (for approved plans)

private struct SavedPlanBar: View {
    @Binding var feedbackText: String
    var isInputFocused: FocusState<Bool>.Binding
    let isProcessing: Bool
    @Binding var showSavedConfirmation: Bool
    let onSubmit: () -> Void
    let onRegenerate: () -> Void

    private var showSendButton: Bool {
        !feedbackText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            // Success confirmation banner
            if showSavedConfirmation {
                SavedConfirmationBanner(onDismiss: {
                    withAnimation(.easeOut(duration: 0.2)) {
                        showSavedConfirmation = false
                    }
                })
                .transition(.asymmetric(
                    insertion: .move(edge: .top).combined(with: .opacity),
                    removal: .opacity
                ))
            }

            VStack(spacing: 12) {
                // Divider
                Divider()

                // Text input row
                HStack(spacing: 10) {
                    TextField("Ask something or suggest changes...", text: $feedbackText, axis: .vertical)
                        .font(.system(size: 15))
                        .foregroundColor(.primary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(Color(.secondarySystemBackground))
                        )
                        .focused(isInputFocused)
                        .lineLimit(1...3)
                        .disabled(isProcessing)
                        .submitLabel(.send)
                        .onSubmit {
                            onSubmit()
                        }

                    // Send button
                    Button(action: onSubmit) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 30))
                            .foregroundColor(Color(hex: "10B981"))
                    }
                    .disabled(isProcessing || !showSendButton)
                    .opacity(showSendButton ? 1 : 0)
                    .scaleEffect(showSendButton ? 1 : 0.5)
                    .animation(.easeInOut(duration: 0.15), value: showSendButton)
                }
                .padding(.horizontal, 16)

                // Regenerate button only (no Accept/Done since plan is saved)
                Button(action: onRegenerate) {
                    HStack(spacing: 8) {
                        if isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .secondary))
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "arrow.counterclockwise")
                                .font(.system(size: 13, weight: .semibold))
                        }
                        Text("Regenerate Plan")
                            .font(.system(size: 14, weight: .semibold))
                    }
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(Color(.secondarySystemBackground))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(Color(.separator).opacity(0.5), lineWidth: 1)
                    )
                }
                .disabled(isProcessing)
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
            }
        }
        .background(Color(.systemBackground))
    }
}

// MARK: - Saved Confirmation Banner

private struct SavedConfirmationBanner: View {
    let onDismiss: () -> Void

    @State private var checkmarkScale: CGFloat = 0
    @State private var contentOpacity: Double = 0

    var body: some View {
        HStack(spacing: 12) {
            // Animated checkmark
            ZStack {
                Circle()
                    .fill(Color(hex: "10B981").opacity(0.15))
                    .frame(width: 32, height: 32)

                Image(systemName: "checkmark")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(Color(hex: "10B981"))
                    .scaleEffect(checkmarkScale)
            }

            Text("Plan saved")
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(.primary)

            Spacer()

            // Dismiss button
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Color(.tertiaryLabel))
                    .padding(6)
                    .background(
                        Circle()
                            .fill(Color(.tertiarySystemFill))
                    )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(hex: "10B981").opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(Color(hex: "10B981").opacity(0.2), lineWidth: 1)
                )
        )
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .opacity(contentOpacity)
        .onAppear {
            // Staggered animation for polish
            withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                contentOpacity = 1
            }
            withAnimation(.spring(response: 0.5, dampingFraction: 0.5).delay(0.1)) {
                checkmarkScale = 1
            }
        }
    }
}

// MARK: - Agent Generating View (Claude Code Style)

private struct AgentGeneratingView: View {
    let symbol: String
    let steps: [AnalysisStep]

    @State private var expandedSteps: Set<UUID> = []

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 8) {
                Text(symbol)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.primary)

                HStack(spacing: 6) {
                    PulsingDot()
                    Text("Analyzing stock...")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }
            }
            .padding(.top, 32)
            .padding(.bottom, 28)

            // Steps list
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(steps.enumerated()), id: \.element.id) { index, step in
                        if step.type != .complete {
                            AgentStepRow(
                                step: step,
                                isLast: index == steps.count - 2,
                                isExpanded: isStepExpanded(step),
                                onToggle: { toggleStepExpansion(step.id) }
                            )
                        }
                    }
                }
                .padding(.horizontal, 24)
            }
        }
    }

    /// Determines if a step should show its findings expanded
    private func isStepExpanded(_ step: AnalysisStep) -> Bool {
        switch step.status {
        case .active:
            return true  // Always show findings for active step
        case .completed:
            return expandedSteps.contains(step.id)  // User-controlled
        case .pending:
            return false
        }
    }

    /// Toggle expansion state for a completed step
    private func toggleStepExpansion(_ stepId: UUID) {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            if expandedSteps.contains(stepId) {
                expandedSteps.remove(stepId)
            } else {
                expandedSteps.insert(stepId)
            }
        }
    }
}

// MARK: - Agent Step Row

private struct AgentStepRow: View {
    let step: AnalysisStep
    let isLast: Bool
    let isExpanded: Bool
    let onToggle: () -> Void

    /// Whether this step can be tapped to expand/collapse
    private var canToggle: Bool {
        step.status == .completed && !step.findings.isEmpty
    }

    private var stepLabel: String {
        switch step.type {
        case .gatheringData: return "Market data"
        case .technicalIndicators: return "Technicals"
        case .supportResistance: return "Key levels"
        case .chartPatterns: return "Patterns"
        case .generatingChart: return "Chart"
        case .visionAnalysis: return "Vision"
        case .generatingPlan: return "Plan"
        case .complete: return "Complete"
        }
    }

    private var stepSummary: String? {
        // Only show summary when collapsed
        guard step.status == .completed, !step.findings.isEmpty, !isExpanded else { return nil }
        // Create a short summary from findings
        let summaryParts = step.findings.prefix(2).compactMap { finding -> String? in
            // Extract short key info
            if finding.contains(":") {
                let parts = finding.split(separator: ":", maxSplits: 1)
                if parts.count >= 2 {
                    let value = String(parts[1]).trimmingCharacters(in: .whitespaces)
                    // Truncate long values
                    return value.count > 12 ? String(value.prefix(10)) + "…" : value
                }
            }
            return finding.count > 15 ? String(finding.prefix(12)) + "…" : finding
        }
        return summaryParts.joined(separator: " · ")
    }

    /// Whether to show the findings section
    private var showFindings: Bool {
        if step.status == .active && !step.findings.isEmpty {
            return true
        }
        if step.status == .completed && isExpanded && !step.findings.isEmpty {
            return true
        }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Main row (tappable for completed steps)
            HStack(alignment: .top, spacing: 12) {
                // Status indicator
                statusIndicator
                    .frame(width: 18, height: 18)

                VStack(alignment: .leading, spacing: 4) {
                    // Label row
                    HStack {
                        Text(stepLabel)
                            .font(.system(size: 14, weight: step.status == .active ? .semibold : .regular))
                            .foregroundColor(step.status == .pending ? Color(.tertiaryLabel) : .primary)

                        Spacer()

                        // Summary for completed steps (when collapsed)
                        if let summary = stepSummary {
                            Text(summary)
                                .font(.system(size: 12, weight: .regular, design: .monospaced))
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                        }

                        // Chevron for expandable steps
                        if canToggle {
                            Image(systemName: "chevron.right")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(Color(.tertiaryLabel))
                                .rotationEffect(.degrees(isExpanded ? 90 : 0))
                                .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isExpanded)
                        }
                    }

                    // Expanded findings
                    if showFindings {
                        if step.type == .visionAnalysis {
                            VisionAnalysisCard(findings: step.findings)
                                .padding(.top, 8)
                        } else {
                            FindingsTreeView(findings: step.findings)
                                .padding(.top, 4)
                        }
                    }
                }
            }
            .padding(.vertical, 10)
            .contentShape(Rectangle())
            .onTapGesture {
                if canToggle {
                    onToggle()
                }
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isExpanded)
    }

    @ViewBuilder
    private var statusIndicator: some View {
        switch step.status {
        case .completed:
            Image(systemName: "checkmark")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(Color(hex: "10B981"))
        case .active:
            PulsingDot()
        case .pending:
            Circle()
                .stroke(Color(.tertiaryLabel), lineWidth: 1.5)
                .frame(width: 8, height: 8)
        }
    }
}

// MARK: - Findings Tree View

private struct FindingsTreeView: View {
    let findings: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(findings.enumerated()), id: \.offset) { index, finding in
                HStack(alignment: .top, spacing: 8) {
                    // Tree connector
                    Text(index == findings.count - 1 ? "└─" : "├─")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(Color(.separator))

                    TypewriterText(text: finding, isActive: index == findings.count - 1)
                        .font(.system(size: 13))
                        .foregroundColor(.primary.opacity(0.85))
                }
            }
        }
        .padding(.leading, 4)
    }
}

// MARK: - Vision Analysis Card (Special Treatment)

private struct VisionAnalysisCard: View {
    let findings: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "eye")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(Color(hex: "10B981"))

                Text("Claude Vision analyzing...")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.primary)
            }

            if !findings.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(findings.enumerated()), id: \.offset) { index, finding in
                        HStack(alignment: .top, spacing: 8) {
                            Text(index == findings.count - 1 ? "└─" : "├─")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(Color(.separator))

                            TypewriterText(text: finding, isActive: index == findings.count - 1)
                                .font(.system(size: 13))
                                .foregroundColor(.primary.opacity(0.85))
                        }
                    }
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }
}

// MARK: - Typewriter Text Animation

private struct TypewriterText: View {
    let text: String
    let isActive: Bool

    @State private var displayedText: String = ""
    @State private var showCursor: Bool = true

    var body: some View {
        HStack(spacing: 0) {
            Text(displayedText)

            if isActive && displayedText.count < text.count {
                Text("▌")
                    .foregroundColor(Color(hex: "10B981"))
                    .opacity(showCursor ? 1 : 0)
            }
        }
        .onAppear {
            startTypewriter()
        }
        .onChange(of: text) { _ in
            // Reset and restart if text changes
            displayedText = ""
            startTypewriter()
        }
    }

    private func startTypewriter() {
        guard displayedText.count < text.count else { return }

        // Animate cursor blink
        withAnimation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true)) {
            showCursor.toggle()
        }

        // Type out characters
        typeNextCharacter()
    }

    private func typeNextCharacter() {
        guard displayedText.count < text.count else { return }

        let delay = Double.random(in: 0.02...0.05) // Natural typing speed
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
            let index = text.index(text.startIndex, offsetBy: displayedText.count)
            displayedText.append(text[index])
            typeNextCharacter()
        }
    }
}

// MARK: - Pulsing Dot Animation

private struct PulsingDot: View {
    @State private var isPulsing = false

    var body: some View {
        Circle()
            .fill(Color(hex: "10B981"))
            .frame(width: 8, height: 8)
            .scaleEffect(isPulsing ? 1.3 : 1.0)
            .opacity(isPulsing ? 0.7 : 1.0)
            .animation(
                .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                value: isPulsing
            )
            .onAppear { isPulsing = true }
    }
}

// Legacy support for existing code
enum GenerationPhase: String, CaseIterable {
    case gathering = "Gathering market data"
    case analyzing = "Analyzing technicals"
    case generating = "Generating plan"

    var order: Int {
        switch self {
        case .gathering: return 0
        case .analyzing: return 1
        case .generating: return 2
        }
    }
}

// MARK: - Empty Plan View

private struct EmptyPlanView: View {
    let onGenerate: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 44, weight: .thin))
                .foregroundColor(Color(.tertiaryLabel))

            VStack(spacing: 8) {
                Text("No Trading Plan")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.primary)

                Text("Generate an AI-powered plan\ntailored to your position")
                    .font(.system(size: 14))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
            }

            Button(action: onGenerate) {
                Text("Generate Plan")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 28)
                    .padding(.vertical, 12)
                    .background(
                        Capsule()
                            .fill(Color(hex: "10B981"))
                    )
            }
            .padding(.top, 4)
        }
    }
}

// MARK: - Key Levels Section (Refined)

private struct KeyLevelsSectionView: View {
    let supports: [Double]
    let resistances: [Double]
    let invalidation: String

    @State private var showInvalidation = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Two-column price grid - prices as heroes
            HStack(alignment: .top, spacing: 0) {
                // Resistances column
                VStack(alignment: .leading, spacing: 10) {
                    Text("RESISTANCE")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(Color(hex: "F87171").opacity(0.7))
                        .tracking(0.8)

                    if resistances.isEmpty {
                        Text("—")
                            .font(.system(size: 14, design: .monospaced))
                            .foregroundColor(Color(.tertiaryLabel))
                    } else {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(Array(resistances.prefix(3).enumerated()), id: \.offset) { idx, level in
                                Text(formatPrice(level))
                                    .font(.system(size: 14, weight: idx == 0 ? .semibold : .regular, design: .monospaced))
                                    .foregroundColor(idx == 0 ? .primary : .secondary)
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                // Subtle vertical divider
                Rectangle()
                    .fill(Color(.separator).opacity(0.3))
                    .frame(width: 1)
                    .padding(.vertical, 4)

                // Supports column
                VStack(alignment: .trailing, spacing: 10) {
                    Text("SUPPORT")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(Color(hex: "10B981").opacity(0.7))
                        .tracking(0.8)

                    if supports.isEmpty {
                        Text("—")
                            .font(.system(size: 14, design: .monospaced))
                            .foregroundColor(Color(.tertiaryLabel))
                    } else {
                        VStack(alignment: .trailing, spacing: 6) {
                            ForEach(Array(supports.prefix(3).enumerated()), id: \.offset) { idx, level in
                                Text(formatPrice(level))
                                    .font(.system(size: 14, weight: idx == 0 ? .semibold : .regular, design: .monospaced))
                                    .foregroundColor(idx == 0 ? .primary : .secondary)
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .trailing)
            }

            // Collapsible invalidation criteria
            if !invalidation.isEmpty {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showInvalidation.toggle()
                    }
                } label: {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.orange.opacity(0.6))
                            .frame(width: 5, height: 5)

                        Text("Invalidation")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.secondary)

                        Spacer()

                        Image(systemName: "chevron.down")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(Color(.tertiaryLabel))
                            .rotationEffect(.degrees(showInvalidation ? 180 : 0))
                    }
                    .padding(.top, 8)
                }
                .buttonStyle(.plain)

                if showInvalidation {
                    MarkdownText(invalidation, size: 13, opacity: 0.8)
                        .lineSpacing(3)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private func formatPrice(_ price: Double) -> String {
        if price >= 1000 {
            return String(format: "$%.0f", price)
        } else if price >= 100 {
            return String(format: "$%.1f", price)
        } else {
            return String(format: "$%.2f", price)
        }
    }
}

// MARK: - Market Sentiment Section (Refined)

private struct MarketSentimentSectionView: View {
    let newsSummary: String?
    let redditSentiment: String?
    let redditBuzz: String?

    @State private var isExpanded = false

    private var sentimentColor: Color {
        switch redditSentiment?.lowercased() {
        case "bullish": return Color(hex: "10B981")
        case "bearish": return Color(hex: "F87171")
        case "mixed": return .orange
        default: return .secondary
        }
    }

    private var hasSentiment: Bool {
        if let s = redditSentiment, !s.isEmpty, s.lowercased() != "none" {
            return true
        }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Tap-to-expand header
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 12) {
                    // Sentiment indicator dot
                    if hasSentiment {
                        Circle()
                            .fill(sentimentColor)
                            .frame(width: 8, height: 8)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("SENTIMENT")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.secondary)
                            .tracking(0.8)

                        if hasSentiment, let sentiment = redditSentiment {
                            Text(sentiment.capitalized)
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundColor(sentimentColor)
                        } else {
                            Text("Neutral")
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundColor(.secondary)
                        }
                    }

                    Spacer()

                    Image(systemName: "chevron.down")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            // Expanded content
            if isExpanded {
                VStack(alignment: .leading, spacing: 14) {
                    Divider()
                        .padding(.horizontal, 16)

                    VStack(alignment: .leading, spacing: 12) {
                        // Reddit buzz
                        if let buzz = redditBuzz, !buzz.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Reddit")
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundColor(.secondary)
                                    .tracking(0.5)

                                MarkdownText(buzz, size: 13, opacity: 0.85)
                                    .lineSpacing(3)
                            }
                        }

                        // News
                        if let news = newsSummary, !news.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("News")
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundColor(.secondary)
                                    .tracking(0.5)

                                MarkdownText(news, size: 13, opacity: 0.85)
                                    .lineSpacing(3)
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 16)
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

// MARK: - Evaluation Status Section (Enhanced)

private struct EvaluationStatusSectionView: View {
    @ObservedObject var viewModel: TradingPlanViewModel
    let plan: TradingPlanResponse

    @State private var isExpanded = true

    private var isEvaluating: Bool {
        viewModel.updatePhase == .analyzing
    }

    private var hasAdjustments: Bool {
        viewModel.hasRecentAdjustments
    }

    private var statusColor: Color {
        if isEvaluating {
            return .blue
        }
        switch plan.status.lowercased() {
        case "active": return Color(hex: "10B981")
        case "invalidated": return .orange
        default: return .secondary
        }
    }

    private var statusText: String {
        if isEvaluating {
            return "Evaluating..."
        }
        if hasAdjustments {
            return "Adjusted"
        }
        return plan.status.lowercased() == "active" ? "Valid" : "Invalidated"
    }

    private var hasContent: Bool {
        isEvaluating || hasAdjustments || (plan.evaluationNotes != nil && !plan.evaluationNotes!.isEmpty)
    }

    var body: some View {
        if hasContent || viewModel.hasActivePosition {
            VStack(alignment: .leading, spacing: 0) {
                // Header
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: 10) {
                        // Status indicator
                        if isEvaluating {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .blue))
                                .scaleEffect(0.7)
                                .frame(width: 14, height: 14)
                        } else {
                            Circle()
                                .fill(statusColor)
                                .frame(width: 6, height: 6)
                        }

                        Text("PLAN STATUS")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.secondary)
                            .tracking(0.8)

                        Spacer()

                        // Status badge
                        HStack(spacing: 4) {
                            if hasAdjustments {
                                Image(systemName: "arrow.triangle.2.circlepath")
                                    .font(.system(size: 9, weight: .semibold))
                            }
                            Text(statusText)
                                .font(.system(size: 11, weight: .medium))
                        }
                        .foregroundColor(statusColor)

                        if !isEvaluating {
                            Image(systemName: "chevron.down")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundColor(Color(.tertiaryLabel))
                                .rotationEffect(.degrees(isExpanded ? 180 : 0))
                        }
                    }
                    .padding(16)
                }
                .buttonStyle(.plain)
                .disabled(isEvaluating)

                // Expanded content
                if isExpanded && !isEvaluating {
                    VStack(alignment: .leading, spacing: 12) {
                        Divider()
                            .padding(.horizontal, 16)

                        // Adjustments section (if any)
                        if let evaluation = viewModel.lastEvaluation, !evaluation.adjustmentsMade.isEmpty {
                            AdjustmentsView(
                                adjustments: evaluation.adjustmentsMade,
                                previousValues: evaluation.previousValues
                            )
                            .padding(.horizontal, 16)
                        }

                        // Evaluation notes
                        if let notes = plan.evaluationNotes, !notes.isEmpty {
                            MarkdownText(notes, size: 14, opacity: 0.85)
                                .lineSpacing(4)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 16)
                        }

                        // Last evaluated time
                        if let lastEval = plan.lastEvaluation {
                            HStack {
                                Text("Last evaluated")
                                    .font(.system(size: 11))
                                    .foregroundColor(Color(.tertiaryLabel))
                                Spacer()
                                Text(formatRelativeTime(lastEval))
                                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal, 16)
                        }

                        // Refresh button
                        Button {
                            Task { await viewModel.evaluatePlan() }
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.clockwise")
                                    .font(.system(size: 12, weight: .medium))
                                Text("Refresh Evaluation")
                                    .font(.system(size: 13, weight: .medium))
                            }
                            .foregroundColor(Color(hex: "10B981"))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(Color(hex: "10B981").opacity(0.1))
                            )
                        }
                        .padding(.horizontal, 16)
                        .padding(.bottom, 16)
                        .disabled(viewModel.isUpdating)
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(.secondarySystemBackground))
            )
        }
    }

    private func formatRelativeTime(_ dateString: String) -> String {
        // Try multiple date formats
        let date: Date? = {
            // ISO8601 with fractional seconds
            let iso8601Fractional = ISO8601DateFormatter()
            iso8601Fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let d = iso8601Fractional.date(from: dateString) { return d }

            // ISO8601 without fractional seconds
            let iso8601 = ISO8601DateFormatter()
            iso8601.formatOptions = [.withInternetDateTime]
            if let d = iso8601.date(from: dateString) { return d }

            // Try DateFormatter with common server formats
            let dateFormatter = DateFormatter()
            dateFormatter.locale = Locale(identifier: "en_US_POSIX")

            // Format: 2025-01-15T10:30:00Z
            dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
            if let d = dateFormatter.date(from: dateString) { return d }

            // Format: 2025-01-15T10:30:00.000Z
            dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"
            if let d = dateFormatter.date(from: dateString) { return d }

            // Format with timezone offset: 2025-01-15T10:30:00+00:00
            dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssXXXXX"
            if let d = dateFormatter.date(from: dateString) { return d }

            return nil
        }()

        guard let parsedDate = date else {
            // Fallback: try to show at least something readable
            // Remove timezone info and 'T' separator if present
            let cleaned = dateString
                .replacingOccurrences(of: "T", with: " ")
                .replacingOccurrences(of: "Z", with: "")
                .components(separatedBy: ".").first ?? dateString
            return cleaned.count > 16 ? String(cleaned.prefix(16)) : cleaned
        }

        let interval = Date().timeIntervalSince(parsedDate)

        // Handle future dates (clock skew)
        if interval < 0 {
            return "just now"
        }

        if interval < 60 {
            return "just now"
        } else if interval < 3600 {
            let minutes = Int(interval / 60)
            return "\(minutes)m ago"
        } else if interval < 86400 {
            let hours = Int(interval / 3600)
            return "\(hours)h ago"
        } else if interval < 604800 { // Less than 7 days
            let days = Int(interval / 86400)
            return "\(days)d ago"
        } else {
            // Show actual date for older timestamps
            let displayFormatter = DateFormatter()
            displayFormatter.dateStyle = .short
            displayFormatter.timeStyle = .short
            return displayFormatter.string(from: parsedDate)
        }
    }
}

// MARK: - Adjustments View

private struct AdjustmentsView: View {
    let adjustments: [String]
    let previousValues: [String: Double]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "arrow.triangle.2.circlepath")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.orange)

                Text("Adjustments Made")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.primary)
            }

            VStack(alignment: .leading, spacing: 6) {
                ForEach(adjustments, id: \.self) { field in
                    AdjustmentRow(field: field, previousValue: previousValues[field])
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.orange.opacity(0.08))
        )
    }
}

private struct AdjustmentRow: View {
    let field: String
    let previousValue: Double?

    private var fieldLabel: String {
        switch field {
        case "stop_loss": return "Stop Loss"
        case "entry_zone_low": return "Entry Low"
        case "entry_zone_high": return "Entry High"
        case "target_1": return "Target 1"
        case "target_2": return "Target 2"
        case "target_3": return "Target 3"
        default: return field.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    var body: some View {
        HStack(spacing: 8) {
            Text("•")
                .font(.system(size: 12))
                .foregroundColor(.orange)

            Text(fieldLabel)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.primary)

            if let prev = previousValue {
                Text(formatPrice(prev))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(Color(.tertiaryLabel))
                    .strikethrough()

                Image(systemName: "arrow.right")
                    .font(.system(size: 10))
                    .foregroundColor(Color(.tertiaryLabel))

                Text("updated")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.orange)
            }
        }
    }

    private func formatPrice(_ price: Double) -> String {
        if price >= 1000 {
            return String(format: "$%.0f", price)
        } else if price >= 100 {
            return String(format: "$%.1f", price)
        } else {
            return String(format: "$%.2f", price)
        }
    }
}

// MARK: - Legacy Evaluation Notes Section

private struct EvaluationNotesSectionView: View {
    let notes: String
    let status: String

    @State private var isExpanded = true

    private var statusColor: Color {
        switch status.lowercased() {
        case "active": return Color(hex: "10B981")
        case "invalidated": return .orange
        default: return .secondary
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 10) {
                    // Status dot
                    Circle()
                        .fill(statusColor)
                        .frame(width: 6, height: 6)

                    Text("EVALUATION")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.secondary)
                        .tracking(0.8)

                    Spacer()

                    // Status badge - minimal
                    Text(status.lowercased() == "active" ? "Valid" : "Invalidated")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(statusColor)

                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            // Notes content
            if isExpanded {
                VStack(alignment: .leading, spacing: 0) {
                    Divider()
                        .padding(.horizontal, 16)

                    MarkdownText(notes, size: 14, opacity: 0.85)
                        .lineSpacing(4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(16)
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

// MARK: - Markdown Text Helper

private struct MarkdownText: View {
    let text: String
    let fontSize: CGFloat
    let opacity: Double

    init(_ text: String, size: CGFloat = 14, opacity: Double = 0.85) {
        self.text = text
        self.fontSize = size
        self.opacity = opacity
    }

    var body: some View {
        parsedText
            .font(.system(size: fontSize))
            .foregroundColor(.primary.opacity(opacity))
            .lineSpacing(4)
    }

    private var parsedText: Text {
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
}

// MARK: - Preview

#Preview {
    SimplifiedPlanView(symbol: "AAPL")
}
