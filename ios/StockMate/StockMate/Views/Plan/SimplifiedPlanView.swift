import SwiftUI

// MARK: - Main View

struct SimplifiedPlanView: View {
    @StateObject private var viewModel: TradingPlanViewModel
    @ObservedObject private var generationManager = PlanGenerationManager.shared
    @State private var feedbackText: String = ""
    @State private var showSavedConfirmation: Bool = false
    @State private var showPlanDetail: Bool = false  // Controls whether to show plan detail view
    @FocusState private var isInputFocused: Bool

    let symbol: String

    init(symbol: String) {
        self.symbol = symbol
        _viewModel = StateObject(wrappedValue: TradingPlanViewModel(symbol: symbol))
    }

    /// The plan to display - from manager (if completed), ViewModel, or draft
    private var displayPlan: TradingPlanResponse? {
        // Check if manager has a completed plan for this symbol
        if generationManager.hasCompletedPlan(for: symbol) {
            return generationManager.generatedPlan
        }
        return viewModel.plan ?? viewModel.draftPlan
    }

    /// Whether generation is in progress (either ViewModel or manager)
    private var isGenerating: Bool {
        viewModel.isLoading || generationManager.hasActiveGeneration(for: symbol)
    }

    /// Whether we're in draft mode (session with unapproved plan)
    private var isDraftMode: Bool {
        // Manager's plan counts as draft until accepted
        if generationManager.hasCompletedPlan(for: symbol) && viewModel.plan == nil {
            return true
        }
        return viewModel.draftPlan != nil && viewModel.plan == nil
    }

    /// Plan is saved when we have an approved plan (not draft)
    private var isPlanSaved: Bool {
        viewModel.plan != nil && viewModel.draftPlan == nil && !generationManager.hasCompletedPlan(for: symbol)
    }

    /// Whether analysis just completed and we should show the completion UI
    private var showAnalysisComplete: Bool {
        // Show completion UI when manager has completed plan but user hasn't clicked "View Plan" yet
        generationManager.hasCompletedPlan(for: symbol) && !showPlanDetail && !isGenerating
    }

    /// Whether to show back button in plan detail view (only for newly generated plans, not saved ones)
    private var canGoBackToAnalysis: Bool {
        showPlanDetail && generationManager.hasCompletedPlan(for: symbol) && viewModel.plan == nil
    }

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()

            if isGenerating && displayPlan == nil {
                // Show generation progress from either ViewModel or manager
                if generationManager.hasActiveGeneration(for: symbol) {
                    ManagerGeneratingView(symbol: symbol, manager: generationManager)
                } else {
                    AgentGeneratingView(symbol: symbol, steps: viewModel.analysisSteps, viewModel: viewModel)
                }
            } else if showAnalysisComplete, let plan = displayPlan {
                // Analysis just completed - show completion UI with "View Selected Plan" button
                AnalysisCompleteView(
                    symbol: symbol,
                    plan: plan,
                    manager: generationManager,
                    onViewPlan: {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                            showPlanDetail = true
                        }
                    },
                    onStartOver: {
                        showPlanDetail = false
                        generationManager.clearCompletedGeneration()
                        generationManager.startGeneration(for: symbol)
                    }
                )
            } else if let plan = displayPlan, (showPlanDetail || viewModel.plan != nil) {
                // User clicked "View Plan" or has an existing saved plan
                planContent(plan)
            } else if displayPlan == nil {
                EmptyPlanView {
                    // Use the manager for background-safe generation
                    showPlanDetail = false
                    generationManager.startGeneration(for: symbol)
                }
            }
        }
        .task { await viewModel.loadPlan() }
        .onAppear {
            // If manager has a completed plan, sync it to the ViewModel
            if let managerPlan = generationManager.generatedPlan,
               generationManager.hasCompletedPlan(for: symbol) {
                viewModel.setDraftPlan(managerPlan)
            }
            // If we already have a saved plan, show detail directly
            if viewModel.plan != nil {
                showPlanDetail = true
            }
        }
        .onChange(of: generationManager.generatedPlan) { newPlan in
            // When manager completes, sync to ViewModel
            if let plan = newPlan, generationManager.activeSymbol?.uppercased() == symbol.uppercased() {
                viewModel.setDraftPlan(plan)
            }
        }
        .onChange(of: viewModel.plan) { newPlan in
            // If we load an existing saved plan, show detail directly
            if newPlan != nil {
                showPlanDetail = true
            }
        }
    }

    // MARK: - Plan Content

    @ViewBuilder
    private func planContent(_ plan: TradingPlanResponse) -> some View {
        VStack(spacing: 0) {
            // Back button row (only shown when navigating from analysis complete view)
            if canGoBackToAnalysis {
                HStack {
                    Button(action: {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                            showPlanDetail = false
                        }
                    }) {
                        HStack(spacing: 6) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 14, weight: .semibold))
                            Text("Analysis")
                                .font(.system(size: 15, weight: .medium))
                        }
                        .foregroundColor(Color(hex: "10B981"))
                    }

                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                .padding(.bottom, 4)
            }

            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    PlanHeaderView(
                        symbol: plan.symbol,
                        tradeStyle: plan.tradeStyle ?? "swing",
                        holdingPeriod: plan.holdingPeriod ?? "2-5 days"
                    )

                    // V2: Position Action Card (shown when user has position)
                    if plan.hasPositionRecommendation {
                        PositionActionCardView(recommendation: plan.positionRecommendationDisplay)
                    }

                    // Bias Indicator
                    BiasIndicatorView(
                        bias: plan.bias,
                        confidence: plan.confidence ?? 70
                    )

                    // Thesis
                    ThesisSectionView(text: plan.thesis, originalThesis: plan.originalThesis)

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

                    // V2: What to Watch Section
                    if plan.hasWatchItems {
                        WhatToWatchSectionView(items: plan.whatToWatch)
                    }

                    // V2: Risk Warnings Section
                    if plan.hasRiskWarnings {
                        RiskWarningsSectionView(warnings: plan.riskWarnings)
                    }

                    // V2: Alternative Analyses Section
                    if plan.hasAlternatives {
                        AlternativeAnalysesSectionView(alternatives: plan.alternatives)
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
                            var success = false

                            // Check if this is a V2 plan from the generation manager
                            if generationManager.hasCompletedPlan(for: symbol) && generationManager.analysisId != nil {
                                // V2 flow: Use manager's approval endpoint
                                if let approvedPlan = await generationManager.approveAnalysis() {
                                    viewModel.setApprovedPlan(approvedPlan)
                                    success = true
                                }
                            } else {
                                // V1/session flow: Use viewModel's approval
                                success = await viewModel.acceptPlan()
                            }

                            if success {
                                // Clear manager's completed generation state
                                generationManager.clearCompletedGeneration()

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
                        Task {
                            // Clear manager state before starting over
                            generationManager.clearCompletedGeneration()
                            await viewModel.startOver()
                        }
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
    let originalThesis: String?

    @State private var showOriginalThesis = false

    // Check if thesis has been updated (differs from original)
    private var hasUpdatedThesis: Bool {
        guard let original = originalThesis, !original.isEmpty else { return false }
        return original != text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Current thesis
            MarkdownText(text, size: 15, opacity: 0.9)
                .lineSpacing(5)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Show original thesis toggle if thesis was updated
            if hasUpdatedThesis {
                Divider()
                    .padding(.vertical, 4)

                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showOriginalThesis.toggle()
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: showOriginalThesis ? "chevron.down" : "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                        Text("View Original Thesis")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)

                if showOriginalThesis, let original = originalThesis {
                    MarkdownText(original, size: 14, opacity: 0.7)
                        .lineSpacing(4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(Color(.tertiarySystemBackground))
                        )
                }
            }
        }
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
    @ObservedObject var viewModel: TradingPlanViewModel

    @State private var expandedSteps: Set<UUID> = []
    @State private var pulseAnimation = false

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 8) {
                Text(symbol)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.primary)

                HStack(spacing: 6) {
                    PulsingDot()
                    Text(viewModel.isV2Mode ? "Running parallel analysis..." : "Analyzing stock...")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }
            }
            .padding(.top, 32)
            .padding(.bottom, 28)

            // Content - V2 or V1 mode
            ScrollView {
                if viewModel.isV2Mode && !viewModel.subagentProgress.isEmpty {
                    // V2: Parallel sub-agent view
                    V2SubAgentsView(
                        orchestratorSteps: viewModel.orchestratorSteps,
                        subagents: viewModel.sortedSubagents,
                        expandedAgents: viewModel.expandedSubagents,
                        isAnalyzersSectionExpanded: viewModel.isAnalyzersSectionExpanded,
                        allSubagentsComplete: viewModel.allSubagentsComplete,
                        completedCount: viewModel.completedSubagentCount,
                        onToggle: { viewModel.toggleSubagentExpansion($0) },
                        onToggleAnalyzersSection: { viewModel.toggleAnalyzersSection() },
                        pulseAnimation: pulseAnimation
                    )
                    .padding(.horizontal, 20)
                } else {
                    // V1: Linear steps list
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
        .onAppear {
            withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                pulseAnimation = true
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

// MARK: - V2 Sub-Agents View

private struct V2SubAgentsView: View {
    let orchestratorSteps: [OrchestratorStep]
    let subagents: [SubAgentProgress]
    let expandedAgents: Set<String>
    let isAnalyzersSectionExpanded: Bool
    let allSubagentsComplete: Bool
    let completedCount: Int
    let onToggle: (String) -> Void
    let onToggleAnalyzersSection: () -> Void
    let pulseAnimation: Bool
    var plan: TradingPlanResponse? = nil  // Optional - for showing individual agent plan details

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(orchestratorSteps) { step in
                if step.stepType == "spawning_subagents" {
                    // Special expandable section for Starting Analyzers with sub-agents nested inside
                    ExpandableAnalyzersSectionRow(
                        step: step,
                        subagents: subagents,
                        expandedAgents: expandedAgents,
                        isExpanded: isAnalyzersSectionExpanded,
                        allComplete: allSubagentsComplete,
                        completedCount: completedCount,
                        onToggleSection: onToggleAnalyzersSection,
                        onToggleAgent: onToggle,
                        pulseAnimation: pulseAnimation,
                        plan: plan
                    )
                } else {
                    // Regular orchestrator step row
                    OrchestratorStepRow(step: step, pulseAnimation: pulseAnimation)
                }
            }
        }
    }
}

// MARK: - Expandable Analyzers Section Row

private struct ExpandableAnalyzersSectionRow: View {
    let step: OrchestratorStep
    let subagents: [SubAgentProgress]
    let expandedAgents: Set<String>
    let isExpanded: Bool
    let allComplete: Bool
    let completedCount: Int
    let onToggleSection: () -> Void
    let onToggleAgent: (String) -> Void
    let pulseAnimation: Bool
    var plan: TradingPlanResponse? = nil  // Optional - for showing individual agent plan details

    private var statusColor: Color {
        allComplete ? Color(hex: "10B981") : Color(hex: "3B82F6")
    }

    private var displayName: String {
        allComplete ? "Analysis Complete" : "Running Analyzers"
    }

    private var statusText: String {
        if allComplete {
            return "DONE"
        } else if step.status == .active {
            return "\(completedCount)/\(subagents.count)"
        } else {
            return "PENDING"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header row (tappable to expand/collapse)
            Button(action: onToggleSection) {
                HStack(spacing: 12) {
                    // Status indicator
                    ZStack {
                        Circle()
                            .fill(statusColor.opacity(0.15))
                            .frame(width: 32, height: 32)

                        if step.status == .active && !allComplete {
                            Circle()
                                .fill(statusColor.opacity(0.2))
                                .frame(width: 32, height: 32)
                                .scaleEffect(pulseAnimation ? 1.3 : 1.0)
                                .opacity(pulseAnimation ? 0 : 0.5)

                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: statusColor))
                                .scaleEffect(0.55)
                        } else if allComplete {
                            Image(systemName: "checkmark")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(statusColor)
                        } else {
                            Image(systemName: "arrow.triangle.branch")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(statusColor)
                        }
                    }

                    // Section info
                    VStack(alignment: .leading, spacing: 2) {
                        Text(displayName)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(.primary)

                        Text(allComplete ? "All analyzers finished" : "Day, Swing, Position")
                            .font(.system(size: 11, weight: .regular))
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    // Status badge
                    Text(statusText)
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundColor(statusColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule()
                                .fill(statusColor.opacity(0.12))
                        )

                    // Expand indicator
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(.vertical, 12)
                .padding(.horizontal, 14)
            }
            .buttonStyle(.plain)

            // Expanded content - sub-agent cards
            if isExpanded {
                VStack(alignment: .leading, spacing: 10) {
                    Divider()
                        .padding(.horizontal, 14)

                    // Sub-agent cards
                    ForEach(subagents) { agent in
                        V2SubAgentCard(
                            agent: agent,
                            isExpanded: expandedAgents.contains(agent.agentName),
                            onToggle: { onToggleAgent(agent.agentName) },
                            pulseAnimation: pulseAnimation,
                            plan: plan
                        )
                        .padding(.horizontal, 10)
                    }
                }
                .padding(.bottom, 12)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(
                            step.status == .active && !allComplete ? statusColor.opacity(0.3) :
                            allComplete ? Color(hex: "10B981").opacity(0.2) : Color.clear,
                            lineWidth: 1
                        )
                )
        )
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isExpanded)
        .animation(.easeInOut(duration: 0.2), value: allComplete)
    }
}

// MARK: - Orchestrator Step Row

private struct OrchestratorStepRow: View {
    let step: OrchestratorStep
    let pulseAnimation: Bool

    private var statusColor: Color {
        step.status == .completed ? Color(hex: "10B981") : Color(hex: "3B82F6")
    }

    var body: some View {
        HStack(spacing: 12) {
            // Status indicator
            ZStack {
                Circle()
                    .fill(statusColor.opacity(0.15))
                    .frame(width: 28, height: 28)

                if step.status == .active {
                    Circle()
                        .fill(statusColor.opacity(0.2))
                        .frame(width: 28, height: 28)
                        .scaleEffect(pulseAnimation ? 1.3 : 1.0)
                        .opacity(pulseAnimation ? 0 : 0.5)

                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: statusColor))
                        .scaleEffect(0.5)
                } else {
                    Image(systemName: step.status == .completed ? "checkmark" : step.icon)
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(statusColor)
                }
            }

            // Step info
            VStack(alignment: .leading, spacing: 2) {
                Text(step.displayName)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.primary)

                if !step.findings.isEmpty {
                    Text(step.findings.joined(separator: " • "))
                        .font(.system(size: 11, weight: .regular))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()

            // Status badge
            Text(step.status == .completed ? "DONE" : "ACTIVE")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(statusColor)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    Capsule()
                        .fill(statusColor.opacity(0.12))
                )
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color(.secondarySystemBackground).opacity(0.5))
        )
    }
}

// MARK: - Agent Plan Detail Sheet

private struct AgentPlanDetailSheet: View {
    let agent: SubAgentProgress
    let plan: TradingPlanResponse
    @Environment(\.dismiss) private var dismiss

    /// Check if this agent is the selected one (matches the main plan's trade style)
    private var isSelectedAgent: Bool {
        guard let selectedStyle = plan.tradeStyle?.lowercased() else { return false }
        switch agent.agentName {
        case "day-trade-analyzer": return selectedStyle == "day"
        case "swing-trade-analyzer": return selectedStyle == "swing"
        case "position-trade-analyzer": return selectedStyle == "position"
        default: return false
        }
    }

    /// Get the alternative plan for this agent (if not selected)
    private var alternativePlan: AlternativePlan? {
        let style: String
        switch agent.agentName {
        case "day-trade-analyzer": style = "day"
        case "swing-trade-analyzer": style = "swing"
        case "position-trade-analyzer": style = "position"
        default: return nil
        }
        return plan.alternatives.first { $0.tradeStyle.lowercased() == style }
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header
                    HStack {
                        // Agent icon
                        ZStack {
                            Circle()
                                .fill(agent.accentColor.opacity(0.15))
                                .frame(width: 48, height: 48)

                            Image(systemName: agent.icon)
                                .font(.system(size: 20, weight: .medium))
                                .foregroundColor(agent.accentColor)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text(agent.displayName)
                                .font(.system(size: 20, weight: .bold))

                            if isSelectedAgent {
                                HStack(spacing: 4) {
                                    Image(systemName: "star.fill")
                                        .font(.system(size: 10))
                                        .foregroundColor(Color(hex: "F59E0B"))
                                    Text("SELECTED PLAN")
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundColor(Color(hex: "F59E0B"))
                                }
                            } else {
                                Text("Alternative Analysis")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(.secondary)
                            }
                        }

                        Spacer()
                    }
                    .padding(.bottom, 8)

                    if isSelectedAgent {
                        // Show full selected plan details
                        selectedPlanContent
                    } else if let alt = alternativePlan {
                        // Show alternative plan summary
                        alternativePlanContent(alt)
                    } else {
                        // Fallback - show findings from agent progress
                        findingsContent
                    }
                }
                .padding(20)
            }
            .background(Color(.systemBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(Color(hex: "10B981"))
                }
            }
        }
    }

    @ViewBuilder
    private var selectedPlanContent: some View {
        // Bias and confidence
        HStack(spacing: 12) {
            Text(plan.bias.uppercased())
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(plan.isBullish ? Color(hex: "10B981") : Color(hex: "EF4444"))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill((plan.isBullish ? Color(hex: "10B981") : Color(hex: "EF4444")).opacity(0.15))
                )

            if let confidence = plan.confidence {
                Text("\(confidence)% Confidence")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.secondary)
            }

            Spacer()
        }

        // Thesis
        VStack(alignment: .leading, spacing: 8) {
            Text("THESIS")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.secondary)
                .tracking(1.0)

            Text(plan.thesis)
                .font(.system(size: 14, weight: .regular))
                .foregroundColor(.primary)
        }

        // Entry/Exit levels
        VStack(alignment: .leading, spacing: 12) {
            Text("PRICE LEVELS")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.secondary)
                .tracking(1.0)

            VStack(spacing: 8) {
                if let low = plan.entryZoneLow, let high = plan.entryZoneHigh {
                    priceLevelRow("Entry Zone", value: "$\(formatPrice(low)) - $\(formatPrice(high))", color: agent.accentColor)
                }
                if let stop = plan.stopLoss {
                    priceLevelRow("Stop Loss", value: "$\(formatPrice(stop))", color: Color(hex: "EF4444"))
                }
                if let t1 = plan.target1 {
                    priceLevelRow("Target 1", value: "$\(formatPrice(t1))", color: Color(hex: "10B981"))
                }
                if let t2 = plan.target2 {
                    priceLevelRow("Target 2", value: "$\(formatPrice(t2))", color: Color(hex: "10B981"))
                }
                if let t3 = plan.target3 {
                    priceLevelRow("Target 3", value: "$\(formatPrice(t3))", color: Color(hex: "10B981"))
                }
            }
        }

        // Risk warnings
        if !plan.riskWarnings.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("RISK WARNINGS")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.secondary)
                    .tracking(1.0)

                ForEach(plan.riskWarnings, id: \.self) { warning in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 12))
                            .foregroundColor(Color(hex: "F59E0B"))
                        Text(warning)
                            .font(.system(size: 13))
                            .foregroundColor(.primary)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func alternativePlanContent(_ alt: AlternativePlan) -> some View {
        // Bias and suitability
        HStack(spacing: 12) {
            Text(alt.bias.uppercased())
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(alt.bias.lowercased() == "bullish" ? Color(hex: "10B981") : Color(hex: "EF4444"))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill((alt.bias.lowercased() == "bullish" ? Color(hex: "10B981") : Color(hex: "EF4444")).opacity(0.15))
                )

            if !alt.suitable {
                Text("NOT SUITABLE")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Color(hex: "F59E0B"))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(hex: "F59E0B").opacity(0.12))
                    .clipShape(Capsule())
            }

            Spacer()

            Text("\(alt.confidence)%")
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundColor(.secondary)
        }

        // Brief thesis
        VStack(alignment: .leading, spacing: 8) {
            Text("ANALYSIS")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.secondary)
                .tracking(1.0)

            Text(alt.briefThesis)
                .font(.system(size: 14, weight: .regular))
                .foregroundColor(.primary)
        }

        // Why not selected
        if !alt.whyNotSelected.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("WHY NOT SELECTED")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.secondary)
                    .tracking(1.0)

                Text(alt.whyNotSelected)
                    .font(.system(size: 13, weight: .regular))
                    .foregroundColor(.secondary)
            }
        }

        // Risk warnings
        if !alt.riskWarnings.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("RISK WARNINGS")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.secondary)
                    .tracking(1.0)

                ForEach(alt.riskWarnings, id: \.self) { warning in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 12))
                            .foregroundColor(Color(hex: "F59E0B"))
                        Text(warning)
                            .font(.system(size: 13))
                            .foregroundColor(.primary)
                    }
                }
            }
        }

        // Holding period
        if !alt.holdingPeriod.isEmpty {
            HStack {
                Image(systemName: "clock")
                    .foregroundColor(.secondary)
                Text("Holding Period: \(alt.holdingPeriod)")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.secondary)
            }
        }
    }

    @ViewBuilder
    private var findingsContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("FINDINGS")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.secondary)
                .tracking(1.0)

            ForEach(agent.findings, id: \.self) { finding in
                HStack(alignment: .top, spacing: 8) {
                    Circle()
                        .fill(agent.accentColor)
                        .frame(width: 6, height: 6)
                        .padding(.top, 6)
                    Text(finding)
                        .font(.system(size: 13))
                        .foregroundColor(.primary)
                }
            }
        }
    }

    @ViewBuilder
    private func priceLevelRow(_ label: String, value: String, color: Color) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundColor(color)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(color.opacity(0.08))
        .cornerRadius(8)
    }

    private func formatPrice(_ value: Double) -> String {
        if value >= 1000 {
            return String(format: "%.0f", value)
        } else if value >= 100 {
            return String(format: "%.1f", value)
        } else {
            return String(format: "%.2f", value)
        }
    }
}

// MARK: - V2 Sub-Agent Card (V1-style hierarchical)

private struct V2SubAgentCard: View {
    let agent: SubAgentProgress
    let isExpanded: Bool
    let onToggle: () -> Void
    let pulseAnimation: Bool
    var plan: TradingPlanResponse? = nil  // Optional - for showing individual agent plan details

    @State private var expandedSteps: Set<String> = []
    @State private var showAgentPlanSheet: Bool = false

    private var statusColor: Color {
        switch agent.status {
        case .completed: return Color(hex: "10B981")
        case .failed: return Color(hex: "EF4444")
        case .pending: return .secondary
        default: return agent.accentColor
        }
    }

    private var statusText: String {
        switch agent.status {
        case .completed: return "DONE"
        case .failed: return "FAILED"
        case .pending: return "WAITING"
        default: return "RUNNING"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header (always visible, tappable)
            Button(action: onToggle) {
                HStack(spacing: 12) {
                    // Status indicator
                    ZStack {
                        Circle()
                            .fill(statusColor.opacity(0.15))
                            .frame(width: 36, height: 36)

                        if agent.status.isActive {
                            Circle()
                                .fill(statusColor.opacity(0.2))
                                .frame(width: 36, height: 36)
                                .scaleEffect(pulseAnimation ? 1.3 : 1.0)
                                .opacity(pulseAnimation ? 0 : 0.5)

                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: statusColor))
                                .scaleEffect(0.6)
                        } else if agent.status == .completed {
                            Image(systemName: "checkmark")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(statusColor)
                        } else if agent.status == .failed {
                            Image(systemName: "xmark")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(statusColor)
                        } else {
                            Image(systemName: agent.icon)
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(statusColor.opacity(0.5))
                        }
                    }

                    // Agent info
                    VStack(alignment: .leading, spacing: 2) {
                        Text(agent.displayName)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(.primary)

                        Text(agent.currentStep ?? agent.status.displayText)
                            .font(.system(size: 12, weight: .regular))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }

                    Spacer()

                    // Status badge
                    Text(statusText)
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(statusColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule()
                                .fill(statusColor.opacity(0.12))
                        )

                    // Expand indicator
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(14)
            }
            .buttonStyle(.plain)

            // Expanded content - V1-style hierarchical steps
            if isExpanded {
                VStack(alignment: .leading, spacing: 0) {
                    Divider()
                        .padding(.horizontal, 14)

                    // V1-style step rows with tree connectors - progressive reveal
                    ForEach(Array(agent.structuredSteps.enumerated()), id: \.element.id) { index, step in
                        V1StyleStepRow(
                            step: step,
                            isLast: index == agent.structuredSteps.count - 1,
                            isExpanded: expandedSteps.contains(step.id) || step.status == .active,
                            accentColor: agent.accentColor,
                            pulseAnimation: pulseAnimation,
                            onToggle: {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                    if expandedSteps.contains(step.id) {
                                        expandedSteps.remove(step.id)
                                    } else {
                                        expandedSteps.insert(step.id)
                                    }
                                }
                            },
                            onShowPlan: plan != nil ? { showAgentPlanSheet = true } : nil
                        )
                        .transition(.asymmetric(
                            insertion: .opacity.combined(with: .move(edge: .top)).animation(.spring(response: 0.4, dampingFraction: 0.8)),
                            removal: .opacity.animation(.easeOut(duration: 0.2))
                        ))
                    }
                    .animation(.spring(response: 0.4, dampingFraction: 0.8), value: agent.structuredSteps.count)
                }
                .padding(.bottom, 10)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(
                            agent.status.isActive ? statusColor.opacity(0.3) :
                            agent.status == .completed ? Color(hex: "10B981").opacity(0.2) : Color.clear,
                            lineWidth: 1
                        )
                )
        )
        .animation(.easeInOut(duration: 0.2), value: isExpanded)
        .sheet(isPresented: $showAgentPlanSheet) {
            if let plan = plan {
                AgentPlanDetailSheet(agent: agent, plan: plan)
            }
        }
    }
}

// MARK: - V1-Style Step Row (within sub-agent)

private struct V1StyleStepRow: View {
    let step: SubAgentStepProgress
    let isLast: Bool
    let isExpanded: Bool
    let accentColor: Color
    let pulseAnimation: Bool
    let onToggle: () -> Void
    var onShowPlan: (() -> Void)? = nil  // Optional action for "Plan" step

    private let greenColor = Color(hex: "10B981")
    private let cyanColor = Color(hex: "00DEDE")

    private var stepStatusColor: Color {
        switch step.status {
        case .pending: return .secondary.opacity(0.4)
        case .active: return accentColor
        case .completed: return greenColor
        }
    }

    /// Whether this step can be tapped to expand
    private var canToggle: Bool {
        step.status == .completed && !step.findings.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Step header row
            HStack(alignment: .center, spacing: 10) {
                // Tree connector + status indicator
                HStack(spacing: 0) {
                    // Tree line (vertical connector)
                    VStack(spacing: 0) {
                        Rectangle()
                            .fill(step.status == .pending ? Color.clear : stepStatusColor.opacity(0.3))
                            .frame(width: 2, height: 10)

                        // Status dot
                        ZStack {
                            Circle()
                                .stroke(stepStatusColor, lineWidth: 2)
                                .frame(width: 16, height: 16)

                            if step.status == .active {
                                Circle()
                                    .fill(accentColor.opacity(0.2))
                                    .frame(width: 16, height: 16)
                                    .scaleEffect(pulseAnimation ? 1.4 : 1.0)
                                    .opacity(pulseAnimation ? 0 : 0.5)

                                Circle()
                                    .fill(accentColor)
                                    .frame(width: 6, height: 6)
                            } else if step.status == .completed {
                                Image(systemName: "checkmark")
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(greenColor)
                            }
                        }

                        if !isLast {
                            Rectangle()
                                .fill(step.status == .completed ? greenColor.opacity(0.3) : Color.clear)
                                .frame(width: 2, height: 10)
                        } else {
                            Spacer().frame(height: 10)
                        }
                    }
                    .frame(width: 20)
                }

                // Step icon
                Image(systemName: step.type.icon)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(stepStatusColor)
                    .frame(width: 16)

                // Step name
                Text(step.type.displayName)
                    .font(.system(size: 12, weight: step.status == .active ? .semibold : .medium))
                    .foregroundColor(step.status == .pending ? .secondary.opacity(0.5) : .primary)

                Spacer()

                // Findings count / status
                if step.status == .active {
                    Text("RUNNING")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(accentColor)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(accentColor.opacity(0.12))
                        .clipShape(Capsule())
                } else if step.status == .completed && !step.findings.isEmpty && !isExpanded {
                    Text("\(step.findings.count)")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color(.tertiarySystemFill))
                        .clipShape(Capsule())
                }

                // Chevron for expandable steps or "View" badge for Plan step
                if step.type == .plan && step.status == .completed && onShowPlan != nil {
                    Text("VIEW")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(greenColor)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(greenColor.opacity(0.12))
                        .clipShape(Capsule())
                } else if canToggle {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
            }
            .padding(.horizontal, 14)
            .contentShape(Rectangle())
            .onTapGesture {
                // Special handling for Plan step - show plan detail sheet
                if step.type == .plan && step.status == .completed, let showPlan = onShowPlan {
                    showPlan()
                } else if canToggle {
                    onToggle()
                }
            }

            // Findings (when expanded or active)
            if (isExpanded || step.status == .active) && !step.findings.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(step.findings, id: \.self) { finding in
                        HStack(alignment: .top, spacing: 6) {
                            // Tree connector
                            Text("└─")
                                .font(.system(size: 10, weight: .regular, design: .monospaced))
                                .foregroundColor(accentColor.opacity(0.4))

                            // Finding text with smart coloring
                            V1FindingText(text: finding, accentColor: accentColor)
                        }
                        .padding(.leading, 34)
                    }
                }
                .padding(.top, 4)
                .padding(.bottom, 6)
                .padding(.horizontal, 14)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}

// MARK: - V1-Style Finding Text

private struct V1FindingText: View {
    let text: String
    let accentColor: Color

    private let greenColor = Color(hex: "10B981")
    private let redColor = Color(hex: "EF4444")
    private let amberColor = Color(hex: "F59E0B")

    var body: some View {
        if text.contains(":") {
            let parts = text.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                HStack(spacing: 4) {
                    Text(String(parts[0]) + ":")
                        .font(.system(size: 11, weight: .regular))
                        .foregroundColor(.secondary)

                    Text(String(parts[1]).trimmingCharacters(in: .whitespaces))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(colorForValue(String(parts[1])))
                }
            } else {
                plainText
            }
        } else {
            plainText
        }
    }

    private var plainText: some View {
        Text(text)
            .font(.system(size: 11, weight: .regular))
            .foregroundColor(.primary.opacity(0.8))
    }

    private func colorForValue(_ value: String) -> Color {
        let lower = value.lowercased()
        if lower.contains("bullish") || lower.contains("strong") || lower.contains("above") || lower.contains("breakout") {
            return greenColor
        } else if lower.contains("bearish") || lower.contains("weak") || lower.contains("below") || lower.contains("breakdown") {
            return redColor
        } else if lower.contains("neutral") || lower.contains("mixed") || lower.contains("consolidat") {
            return amberColor
        }
        return accentColor
    }
}

// MARK: - Manager Generating View (Background-safe)

/// Generating view that uses PlanGenerationManager for background-safe generation.
/// Shows the same UI as AgentGeneratingView but reads from the shared manager.
private struct ManagerGeneratingView: View {
    let symbol: String
    @ObservedObject var manager: PlanGenerationManager

    @State private var pulseAnimation = false

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 8) {
                Text(symbol)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.primary)

                HStack(spacing: 6) {
                    PulsingDot()
                    Text("Running parallel analysis...")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }
            }
            .padding(.top, 32)
            .padding(.bottom, 28)

            // Sub-agent progress
            ScrollView {
                V2SubAgentsView(
                    orchestratorSteps: manager.orchestratorSteps,
                    subagents: manager.sortedSubagents,
                    expandedAgents: manager.expandedSubagents,
                    isAnalyzersSectionExpanded: manager.isAnalyzersSectionExpanded,
                    allSubagentsComplete: manager.allSubagentsComplete,
                    completedCount: manager.completedSubagentsCount,
                    onToggle: { manager.toggleSubagentExpansion($0) },
                    onToggleAnalyzersSection: { manager.toggleAnalyzersSection() },
                    pulseAnimation: pulseAnimation
                )
                .padding(.horizontal, 20)
            }

            // Error display
            if let error = manager.error {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(10)
                .padding()
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                pulseAnimation = true
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
        // V2 cases
        case .gatheringCommonData: return "Common data"
        case .spawningSubagents: return "Sub-agents"
        case .waitingForSubagents: return "Analyzing"
        case .selectingBest: return "Selecting"
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

// MARK: - Analysis Complete View

private struct AnalysisCompleteView: View {
    let symbol: String
    let plan: TradingPlanResponse
    @ObservedObject var manager: PlanGenerationManager
    let onViewPlan: () -> Void
    let onStartOver: () -> Void

    @State private var showConfetti = false

    private var tradeStyleColor: Color {
        switch plan.tradeStyle?.lowercased() ?? "swing" {
        case "day": return Color(red: 1.0, green: 0.6, blue: 0.0)  // Orange
        case "swing": return Color(red: 0.4, green: 0.7, blue: 1.0)  // Blue
        case "position": return Color(red: 0.7, green: 0.5, blue: 1.0)  // Purple
        default: return Color(hex: "10B981")
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 8) {
                Text(symbol)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.primary)

                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(Color(hex: "10B981"))
                    Text("Analysis Complete")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }
            }
            .padding(.top, 32)
            .padding(.bottom, 24)

            // Completed orchestrator steps summary
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Summary of completed analysis
                    V2SubAgentsView(
                        orchestratorSteps: manager.orchestratorSteps,
                        subagents: manager.sortedSubagents,
                        expandedAgents: manager.expandedSubagents,
                        isAnalyzersSectionExpanded: manager.isAnalyzersSectionExpanded,
                        allSubagentsComplete: manager.allSubagentsComplete,
                        completedCount: manager.completedSubagentsCount,
                        onToggle: { manager.toggleSubagentExpansion($0) },
                        onToggleAnalyzersSection: { manager.toggleAnalyzersSection() },
                        pulseAnimation: false,
                        plan: plan
                    )

                    // Selected plan summary card
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "star.fill")
                                .foregroundColor(Color(hex: "F59E0B"))
                            Text("SELECTED PLAN")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.secondary)
                                .tracking(1.0)
                            Spacer()
                        }

                        HStack(spacing: 12) {
                            // Trade style badge
                            Text((plan.tradeStyle ?? "Swing").uppercased())
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(tradeStyleColor)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(
                                    Capsule()
                                        .fill(tradeStyleColor.opacity(0.15))
                                )

                            // Bias indicator
                            Text(plan.bias.uppercased())
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(plan.bias.lowercased() == "bullish" ? Color(hex: "10B981") : Color(hex: "EF4444"))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(
                                    Capsule()
                                        .fill((plan.bias.lowercased() == "bullish" ? Color(hex: "10B981") : Color(hex: "EF4444")).opacity(0.15))
                                )

                            Spacer()

                            // Confidence
                            if let confidence = plan.confidence {
                                Text("\(confidence)%")
                                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                                    .foregroundColor(.primary)
                            }
                        }

                        // Brief thesis preview
                        Text(plan.thesis)
                            .font(.system(size: 13, weight: .regular))
                            .foregroundColor(.secondary)
                            .lineLimit(3)
                    }
                    .padding(16)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(Color(.secondarySystemBackground))
                            .overlay(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .stroke(tradeStyleColor.opacity(0.3), lineWidth: 1)
                            )
                    )
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 120)
            }

            Spacer()

            // Bottom action buttons
            VStack(spacing: 12) {
                Button(action: onViewPlan) {
                    HStack(spacing: 8) {
                        Text("View Selected Plan")
                            .font(.system(size: 16, weight: .semibold))
                        Image(systemName: "arrow.right")
                            .font(.system(size: 14, weight: .semibold))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(
                        Capsule()
                            .fill(Color(hex: "10B981"))
                    )
                }

                Button(action: onStartOver) {
                    Text("Start Over")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 32)
            .background(
                LinearGradient(
                    colors: [Color(.systemBackground).opacity(0), Color(.systemBackground)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 60)
                .offset(y: -60)
            )
        }
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7).delay(0.2)) {
                showConfetti = true
            }
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
        // Always show evaluation section when there's a saved plan
        // (this view is only instantiated when plan exists)
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

// MARK: - V2 Position Action Card

private struct PositionActionCardView: View {
    let recommendation: String

    private var actionColor: Color {
        switch recommendation.uppercased() {
        case "HOLD": return Color(hex: "10B981")  // Green
        case "ADD": return Color(hex: "3B82F6")   // Blue
        case "TRIM", "REDUCE": return Color(hex: "F59E0B")  // Amber/Orange
        case "EXIT": return Color(hex: "EF4444")  // Red
        default: return .secondary
        }
    }

    private var actionIcon: String {
        switch recommendation.uppercased() {
        case "HOLD": return "hand.raised.fill"
        case "ADD": return "plus.circle.fill"
        case "TRIM", "REDUCE": return "arrow.down.right.circle.fill"
        case "EXIT": return "xmark.circle.fill"
        default: return "questionmark.circle.fill"
        }
    }

    private var actionSubtitle: String {
        switch recommendation.uppercased() {
        case "HOLD": return "Maintain current position"
        case "ADD": return "Consider adding to position"
        case "TRIM": return "Consider partial profit-taking"
        case "REDUCE": return "Consider reducing exposure"
        case "EXIT": return "Consider closing position"
        default: return ""
        }
    }

    var body: some View {
        HStack(spacing: 16) {
            // Icon container with glow effect
            ZStack {
                Circle()
                    .fill(actionColor.opacity(0.15))
                    .frame(width: 56, height: 56)

                Circle()
                    .fill(actionColor.opacity(0.08))
                    .frame(width: 72, height: 72)
                    .blur(radius: 8)

                Image(systemName: actionIcon)
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundColor(actionColor)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("POSITION ACTION")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(.secondary)
                    .tracking(1.2)

                Text(recommendation)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(actionColor)

                Text(actionSubtitle)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(actionColor.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(actionColor.opacity(0.25), lineWidth: 1)
                )
        )
    }
}

// MARK: - V2 What to Watch Section

private struct WhatToWatchSectionView: View {
    let items: [String]

    @State private var isExpanded = true

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "eye.fill")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(Color(hex: "3B82F6"))

                    Text("WHAT TO WATCH")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .tracking(0.8)

                    Spacer()

                    Text("\(items.count)")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(Color(hex: "3B82F6"))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(Color(hex: "3B82F6").opacity(0.12))
                        )

                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            // Items
            if isExpanded {
                VStack(alignment: .leading, spacing: 0) {
                    Divider()
                        .padding(.horizontal, 16)

                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(Array(items.enumerated()), id: \.offset) { idx, item in
                            HStack(alignment: .top, spacing: 12) {
                                ZStack {
                                    Circle()
                                        .fill(Color(hex: "3B82F6").opacity(0.12))
                                        .frame(width: 24, height: 24)

                                    Text("\(idx + 1)")
                                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                                        .foregroundColor(Color(hex: "3B82F6"))
                                }

                                Text(item)
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(.primary.opacity(0.85))
                                    .lineSpacing(3)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
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

// MARK: - V2 Risk Warnings Section

private struct RiskWarningsSectionView: View {
    let warnings: [String]

    @State private var isExpanded = true

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(Color(hex: "F59E0B"))

                    Text("RISK WARNINGS")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .tracking(0.8)

                    Spacer()

                    Text("\(warnings.count)")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(Color(hex: "F59E0B"))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(Color(hex: "F59E0B").opacity(0.12))
                        )

                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            // Warnings
            if isExpanded {
                VStack(alignment: .leading, spacing: 0) {
                    Divider()
                        .padding(.horizontal, 16)

                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(warnings.enumerated()), id: \.offset) { _, warning in
                            HStack(alignment: .top, spacing: 10) {
                                Circle()
                                    .fill(Color(hex: "F59E0B"))
                                    .frame(width: 6, height: 6)
                                    .padding(.top, 6)

                                Text(warning)
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(Color(hex: "F59E0B").opacity(0.9))
                                    .lineSpacing(3)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                    .padding(16)
                    .background(Color(hex: "F59E0B").opacity(0.06))
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(Color(hex: "F59E0B").opacity(0.2), lineWidth: 1)
                )
        )
    }
}

// MARK: - V2 Alternative Analyses Section

private struct AlternativeAnalysesSectionView: View {
    let alternatives: [AlternativePlan]

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "arrow.triangle.branch")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.secondary)

                    Text("ALTERNATIVE ANALYSES")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .tracking(0.8)

                    Spacer()

                    Text("\(alternatives.count) styles")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)

                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            // Alternative cards
            if isExpanded {
                VStack(alignment: .leading, spacing: 0) {
                    Divider()
                        .padding(.horizontal, 16)

                    VStack(spacing: 12) {
                        ForEach(alternatives) { alt in
                            AlternativeCardView(alternative: alt)
                        }
                    }
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

// MARK: - Alternative Card View

private struct AlternativeCardView: View {
    let alternative: AlternativePlan

    private var biasColor: Color {
        switch alternative.bias.lowercased() {
        case "bullish": return Color(hex: "10B981")
        case "bearish": return Color(hex: "EF4444")
        default: return .secondary
        }
    }

    private var recommendationColor: Color {
        guard let rec = alternative.positionRecommendation?.lowercased() else { return .secondary }
        switch rec {
        case "hold": return Color(hex: "10B981")
        case "add": return Color(hex: "3B82F6")
        case "trim", "reduce": return Color(hex: "F59E0B")
        case "exit": return Color(hex: "EF4444")
        default: return .secondary
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header row
            HStack(spacing: 12) {
                // Trade style icon
                ZStack {
                    Circle()
                        .fill(Color(.tertiarySystemFill))
                        .frame(width: 36, height: 36)

                    Image(systemName: alternative.tradeStyleIcon)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(alternative.tradeStyleDisplay)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.primary)

                    Text(alternative.holdingPeriod)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Bias & Confidence
                VStack(alignment: .trailing, spacing: 2) {
                    Text(alternative.bias.uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(biasColor)

                    Text("\(alternative.confidence)%")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }

            // Position recommendation (if exists)
            if alternative.hasPositionRecommendation {
                HStack(spacing: 8) {
                    Text("Recommends:")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)

                    Text(alternative.positionRecommendationDisplay)
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(recommendationColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(recommendationColor.opacity(0.12))
                        )
                }
            }

            // Risk warnings (if any)
            if !alternative.riskWarnings.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(alternative.riskWarnings.prefix(2).enumerated()), id: \.offset) { _, warning in
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 9))
                                .foregroundColor(Color(hex: "F59E0B").opacity(0.7))

                            Text(warning)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(Color(hex: "F59E0B").opacity(0.8))
                                .lineLimit(2)
                        }
                    }
                }
            }

            // Why not selected
            if !alternative.whyNotSelected.isEmpty {
                Text(alternative.whyNotSelected)
                    .font(.system(size: 11, weight: .regular))
                    .foregroundColor(Color(.tertiaryLabel))
                    .lineSpacing(2)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(.tertiarySystemBackground))
        )
    }
}

// MARK: - Previews

#Preview("Default") {
    SimplifiedPlanView(symbol: "AAPL")
        .preferredColorScheme(.dark)
}

#Preview("V2 Position Action - HOLD") {
    ScrollView {
        VStack(spacing: 24) {
            PositionActionCardView(recommendation: "HOLD")
            PositionActionCardView(recommendation: "TRIM")
            PositionActionCardView(recommendation: "REDUCE")
            PositionActionCardView(recommendation: "EXIT")
            PositionActionCardView(recommendation: "ADD")
        }
        .padding(20)
    }
    .background(Color.black)
    .preferredColorScheme(.dark)
}

#Preview("V2 What to Watch") {
    VStack(spacing: 20) {
        WhatToWatchSectionView(items: [
            "Volume confirmation on breakout above $275",
            "Tighten stop to breakeven or better",
            "Watch for rejection at $280 resistance"
        ])
    }
    .padding(20)
    .background(Color.black)
    .preferredColorScheme(.dark)
}

#Preview("V2 Risk Warnings") {
    VStack(spacing: 20) {
        RiskWarningsSectionView(warnings: [
            "Technicals turning bearish - protect profits",
            "Analysis bias conflicts with long position",
            "High volume distribution day observed"
        ])
    }
    .padding(20)
    .background(Color.black)
    .preferredColorScheme(.dark)
}

#Preview("V2 Alternative Analyses") {
    ScrollView {
        VStack(spacing: 20) {
            AlternativeAnalysesSectionView_Preview(alternatives: [
                PreviewAlternativePlan(
                    tradeStyle: "day",
                    bias: "bearish",
                    suitable: false,
                    confidence: 45,
                    holdingPeriod: "1-4 hours",
                    briefThesis: "Day trade shows bearish intraday momentum with weak volume.",
                    whyNotSelected: "Conflicts with existing position. No valid setup for this trade style",
                    riskReward: 1.5,
                    positionRecommendation: "trim",
                    riskWarnings: ["Technicals turning bearish - protect profits", "Analysis bias conflicts with long position"]
                ),
                PreviewAlternativePlan(
                    tradeStyle: "swing",
                    bias: "bearish",
                    suitable: false,
                    confidence: 42,
                    holdingPeriod: "3-7 days",
                    briefThesis: "Swing trade analysis shows weakening trend structure.",
                    whyNotSelected: "Lower confidence (42% vs 55%)",
                    riskReward: 2.0,
                    positionRecommendation: "reduce",
                    riskWarnings: ["Double top pattern forming", "EMAs converging bearishly"]
                )
            ])
        }
        .padding(20)
    }
    .background(Color.black)
    .preferredColorScheme(.dark)
}

#Preview("V2 Full Plan View") {
    PreviewFullPlanView()
}

#Preview("V2 Saved Plan View") {
    PreviewSavedPlanView()
}

/// Full plan preview with all sections including action bar
private struct PreviewFullPlanView: View {
    @State private var feedbackText = ""
    @FocusState private var isInputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    PlanHeaderView(
                        symbol: "ARBK",
                        tradeStyle: "swing",
                        holdingPeriod: "N/A - NO TRADE RECOMMENDED"
                    )

                    // Bias
                    BiasIndicatorView(bias: "neutral", confidence: 85)

                    // Thesis - detailed like screenshot
                    ThesisSectionView(text: "ARBK just completed a major court-sanctioned restructuring on December 15, 2025, resulting in massive dilution (existing shareholders went from 10:1 to 2160:1 ADS ratio). Growler Mining now owns approximately 87.5% of the company. The stock dropped 79% after the restructuring was announced. While the company claims enhanced hashrate capacity (1.8 to 2.4 EH/s) and expansion into AI/HPC, the technicals are messy with virtually no support/resistance levels established post-restructuring, and volume is nearly non-existent at 0.04x average. This is a highly speculative restructuring play with extreme risk - not suitable for a technical swing trade.", originalThesis: nil)

                    // Risk/Reward
                    RiskRewardBadgeView(ratio: 0.0)

                    // Key Levels
                    KeyLevelsSectionView(
                        supports: [3.25],
                        resistances: [4.00, 5.00],
                        invalidation: "Close below $3.00"
                    )

                    // Market Sentiment - like screenshot
                    MarketSentimentSectionView(
                        newsSummary: "Major restructuring completed in mid-December 2025 transferred 87.5% stake to Growler, causing 79% stock crash. Company pivoting toward AI/HPC alongside crypto mining. New CEO Justin Nolan appointed March 2025. 2024 results showed revenue down 7%, $55.1M net loss, Bitcoin production cut in half due to halving. Company faces class action lawsuits from 2023.",
                        redditSentiment: "Neutral",
                        redditBuzz: nil
                    )

                    // V2: Position Action (when user has position)
                    PositionActionCardView(recommendation: "HOLD")

                    // V2: What to Watch
                    WhatToWatchSectionView(items: [
                        "Volume confirmation on any breakout",
                        "Watch for support at $3.25",
                        "Monitor restructuring news flow"
                    ])

                    // V2: Risk Warnings
                    RiskWarningsSectionView(warnings: [
                        "Extreme dilution risk - 2160:1 ADS ratio",
                        "Post-restructuring volatility expected"
                    ])

                    // V2: Alternatives
                    AlternativeAnalysesSectionView_Preview(alternatives: [
                        PreviewAlternativePlan(
                            tradeStyle: "day",
                            bias: "bearish",
                            suitable: false,
                            confidence: 45,
                            holdingPeriod: "1-4 hours",
                            briefThesis: "Day trade shows no clear intraday setup due to extremely low volume.",
                            whyNotSelected: "No valid setup - volume too thin",
                            riskReward: 1.0,
                            positionRecommendation: nil,
                            riskWarnings: ["Volume nearly non-existent"]
                        ),
                        PreviewAlternativePlan(
                            tradeStyle: "position",
                            bias: "neutral",
                            suitable: false,
                            confidence: 30,
                            holdingPeriod: "2-6 weeks",
                            briefThesis: "Position trade not recommended - too early post-restructuring.",
                            whyNotSelected: "Lower confidence (30% vs 85%)",
                            riskReward: 0.5,
                            positionRecommendation: nil,
                            riskWarnings: ["No established trend structure"]
                        )
                    ])
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 140)
            }

            // Bottom Action Bar (Draft Mode)
            VStack(spacing: 12) {
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
                        .focused($isInputFocused)
                        .lineLimit(1...3)
                }
                .padding(.horizontal, 16)

                // Action buttons
                HStack(spacing: 12) {
                    // Start Over
                    Button(action: {}) {
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

                    // Accept Plan
                    Button(action: {}) {
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 16, weight: .bold))
                            Text("Accept Plan")
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
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
            }
            .background(Color(.systemBackground))
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
    }
}

/// Saved plan preview with Regenerate button (like second screenshot)
private struct PreviewSavedPlanView: View {
    @State private var feedbackText = ""
    @FocusState private var isInputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    PlanHeaderView(
                        symbol: "ARBK",
                        tradeStyle: "swing",
                        holdingPeriod: "N/A - NO TRADE RECOMMENDED"
                    )

                    // Bias
                    BiasIndicatorView(bias: "neutral", confidence: 85)

                    // Thesis
                    ThesisSectionView(text: "ARBK just completed a major court-sanctioned restructuring on December 15, 2025, resulting in massive dilution. The technicals are messy with virtually no support/resistance levels established post-restructuring, and volume is nearly non-existent. This is a highly speculative restructuring play with extreme risk.", originalThesis: nil)

                    // Risk/Reward
                    RiskRewardBadgeView(ratio: 0.0)

                    // Key Levels
                    KeyLevelsSectionView(
                        supports: [3.25],
                        resistances: [4.00, 5.00],
                        invalidation: "Close below $3.00"
                    )

                    // Market Sentiment
                    MarketSentimentSectionView(
                        newsSummary: "Major restructuring completed in mid-December 2025 transferred 87.5% stake to Growler, causing 79% stock crash. Company pivoting toward AI/HPC alongside crypto mining. New CEO Justin Nolan appointed March 2025. 2024 results showed revenue down 7%, $55.1M net loss, Bitcoin production cut in half due to halving. Company faces class action lawsuits from 2023.",
                        redditSentiment: "Neutral",
                        redditBuzz: nil
                    )
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 100)
            }

            // Bottom Action Bar (Saved Mode - Regenerate only)
            VStack(spacing: 12) {
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
                        .focused($isInputFocused)
                        .lineLimit(1...3)
                }
                .padding(.horizontal, 16)

                // Regenerate button only
                Button(action: {}) {
                    HStack(spacing: 8) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 13, weight: .semibold))
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
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
            }
            .background(Color(.systemBackground))
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
    }
}

// MARK: - Preview Helper

/// Preview-only wrapper for AlternativePlan since it requires Decoder init
private struct PreviewAlternativePlan: Identifiable {
    var id: String { tradeStyle }
    let tradeStyle: String
    let bias: String
    let suitable: Bool
    let confidence: Int
    let holdingPeriod: String
    let briefThesis: String
    let whyNotSelected: String
    let riskReward: Double?
    let positionRecommendation: String?
    let riskWarnings: [String]

    var tradeStyleDisplay: String {
        switch tradeStyle.lowercased() {
        case "day": return "Day Trade"
        case "swing": return "Swing Trade"
        case "position": return "Position Trade"
        default: return tradeStyle.capitalized
        }
    }

    var tradeStyleIcon: String {
        switch tradeStyle.lowercased() {
        case "day": return "bolt.fill"
        case "swing": return "chart.line.uptrend.xyaxis"
        case "position": return "calendar"
        default: return "chart.bar.fill"
        }
    }

    var hasPositionRecommendation: Bool {
        positionRecommendation != nil && !positionRecommendation!.isEmpty
    }

    var positionRecommendationDisplay: String {
        guard let rec = positionRecommendation?.lowercased() else { return "" }
        switch rec {
        case "hold": return "HOLD"
        case "trim": return "TRIM"
        case "reduce": return "REDUCE"
        case "exit": return "EXIT"
        case "add": return "ADD"
        default: return rec.uppercased()
        }
    }
}

// Extend AlternativeAnalysesSectionView to accept preview data
private struct AlternativeAnalysesSectionView_Preview: View {
    let alternatives: [PreviewAlternativePlan]

    @State private var isExpanded = true

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "arrow.triangle.branch")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.secondary)

                    Text("ALTERNATIVE ANALYSES")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .tracking(0.8)

                    Spacer()

                    Text("\(alternatives.count) styles")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)

                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Color(.tertiaryLabel))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .padding(16)
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 0) {
                    Divider()
                        .padding(.horizontal, 16)

                    VStack(spacing: 12) {
                        ForEach(alternatives) { alt in
                            AlternativeCardView_Preview(alternative: alt)
                        }
                    }
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

private struct AlternativeCardView_Preview: View {
    let alternative: PreviewAlternativePlan

    private var biasColor: Color {
        switch alternative.bias.lowercased() {
        case "bullish": return Color(hex: "10B981")
        case "bearish": return Color(hex: "EF4444")
        default: return .secondary
        }
    }

    private var recommendationColor: Color {
        guard let rec = alternative.positionRecommendation?.lowercased() else { return .secondary }
        switch rec {
        case "hold": return Color(hex: "10B981")
        case "add": return Color(hex: "3B82F6")
        case "trim", "reduce": return Color(hex: "F59E0B")
        case "exit": return Color(hex: "EF4444")
        default: return .secondary
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color(.tertiarySystemFill))
                        .frame(width: 36, height: 36)

                    Image(systemName: alternative.tradeStyleIcon)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(alternative.tradeStyleDisplay)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.primary)

                    Text(alternative.holdingPeriod)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text(alternative.bias.uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(biasColor)

                    Text("\(alternative.confidence)%")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }

            if alternative.hasPositionRecommendation {
                HStack(spacing: 8) {
                    Text("Recommends:")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)

                    Text(alternative.positionRecommendationDisplay)
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(recommendationColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(recommendationColor.opacity(0.12))
                        )
                }
            }

            if !alternative.riskWarnings.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(alternative.riskWarnings.prefix(2).enumerated()), id: \.offset) { _, warning in
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 9))
                                .foregroundColor(Color(hex: "F59E0B").opacity(0.7))

                            Text(warning)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(Color(hex: "F59E0B").opacity(0.8))
                                .lineLimit(2)
                        }
                    }
                }
            }

            if !alternative.whyNotSelected.isEmpty {
                Text(alternative.whyNotSelected)
                    .font(.system(size: 11, weight: .regular))
                    .foregroundColor(Color(.tertiaryLabel))
                    .lineSpacing(2)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(.tertiarySystemBackground))
        )
    }
}

// MARK: - V2 Sub-Agent Card Previews (V1-Style Hierarchical)

#Preview("V2 SubAgent - Running") {
    ScrollView {
        VStack(spacing: 16) {
            V2SubAgentCard(
                agent: SubAgentProgress(
                    agentName: "day-trade-analyzer",
                    displayName: "Day Trade",
                    status: .analyzingChart,
                    currentStep: "Vision analysis (5-min chart)",
                    stepsCompleted: [
                        "Gathered 5-min bars",
                        "Calculated EMAs (5, 9, 20)",
                        "Found support/resistance levels",
                        "Detected patterns",
                        "Generated 5-min chart"
                    ],
                    findings: [
                        "Price: $189.45",
                        "RSI: 58.3 (neutral)",
                        "MACD: Bullish crossover",
                        "Support: $187.50",
                        "Resistance: $192.00",
                        "Pattern: Opening range breakout",
                        "Chart: 5-min candlestick rendered"
                    ],
                    elapsedMs: 2500,
                    errorMessage: nil
                ),
                isExpanded: true,
                onToggle: {},
                pulseAnimation: true
            )

            V2SubAgentCard(
                agent: SubAgentProgress(
                    agentName: "swing-trade-analyzer",
                    displayName: "Swing Trade",
                    status: .generatingChart,
                    currentStep: "Generating daily chart",
                    stepsCompleted: [
                        "Gathered daily bars",
                        "Calculated EMAs (9, 21, 50)",
                        "Calculated support levels"
                    ],
                    findings: [
                        "Price: $189.45",
                        "RSI: 52.1 (neutral)",
                        "Support: $185.00"
                    ],
                    elapsedMs: 1800,
                    errorMessage: nil
                ),
                isExpanded: true,
                onToggle: {},
                pulseAnimation: true
            )

            V2SubAgentCard(
                agent: SubAgentProgress(
                    agentName: "position-trade-analyzer",
                    displayName: "Position Trade",
                    status: .gatheringData,
                    currentStep: "Gathering weekly bars",
                    stepsCompleted: [],
                    findings: [],
                    elapsedMs: 800,
                    errorMessage: nil
                ),
                isExpanded: true,
                onToggle: {},
                pulseAnimation: true
            )
        }
        .padding(20)
    }
    .background(Color(.systemBackground))
}

#Preview("V2 SubAgent - Complete") {
    ScrollView {
        VStack(spacing: 16) {
            V2SubAgentCard(
                agent: SubAgentProgress(
                    agentName: "swing-trade-analyzer",
                    displayName: "Swing Trade",
                    status: .completed,
                    currentStep: nil,
                    stepsCompleted: [
                        "Gathered daily bars",
                        "Calculated EMAs (9, 21, 50)",
                        "Found support/resistance levels",
                        "Detected bull flag pattern",
                        "Generated daily chart",
                        "Vision analysis complete",
                        "Generated trading plan"
                    ],
                    findings: [
                        "Price: $189.45",
                        "RSI: 52.1 (neutral momentum)",
                        "MACD: Bullish crossover forming",
                        "EMA: Price above 9, 21, testing 50",
                        "Support: $185.00, $180.50",
                        "Resistance: $192.00, $198.50",
                        "Pattern: Bull flag forming",
                        "Chart: Daily candlestick rendered (1050x700)",
                        "Trend: Clean uptrend with consolidation",
                        "Visual: Support holding well",
                        "Bias: Bullish",
                        "Confidence: 78%",
                        "Entry: $188.50 - $190.00",
                        "Target 1: $195.00",
                        "Stop: $184.00"
                    ],
                    elapsedMs: 4200,
                    errorMessage: nil
                ),
                isExpanded: true,
                onToggle: {},
                pulseAnimation: false
            )
        }
        .padding(20)
    }
    .background(Color(.systemBackground))
}

#Preview("V2 Generating View") {
    let manager = PlanGenerationManager.shared

    return ManagerGeneratingView(symbol: "AAPL", manager: manager)
        .onAppear {
            // Simulate some progress
            manager.startGeneration(for: "AAPL")
        }
}
