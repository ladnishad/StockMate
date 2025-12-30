import SwiftUI

/// Displays a draft trading plan with Approve/Adjust/Ask actions
/// This is the Claude Code-style interactive planning view
struct DraftPlanView: View {
    @ObservedObject var viewModel: TradingPlanViewModel
    @State private var showingConversation = false
    @State private var feedbackMode: FeedbackMode?

    enum FeedbackMode: Identifiable {
        case question
        case adjust

        var id: String {
            switch self {
            case .question: return "question"
            case .adjust: return "adjust"
            }
        }
    }

    var body: some View {
        VStack(spacing: 20) {
            // Draft badge header
            DraftBadgeHeader()

            if let plan = viewModel.draftPlan {
                // Plan content (reusing existing card styles)
                DraftPlanContent(plan: plan, viewModel: viewModel)

                // Action buttons
                DraftActionButtons(
                    viewModel: viewModel,
                    onApprove: {
                        Task { await viewModel.approveDraftPlan() }
                    },
                    onAdjust: {
                        feedbackMode = .adjust
                    },
                    onAskQuestion: {
                        feedbackMode = .question
                    }
                )

                // Conversation history (if any)
                if !viewModel.conversationMessages.isEmpty {
                    ConversationHistorySection(messages: viewModel.conversationMessages)
                }

            } else if viewModel.isLoading || viewModel.updatePhase != .idle {
                // Loading state
                DraftLoadingView(phase: viewModel.updatePhase)
            }
        }
        .sheet(item: $feedbackMode) { mode in
            FeedbackSheet(
                viewModel: viewModel,
                mode: mode,
                onDismiss: { feedbackMode = nil }
            )
        }
    }
}

// MARK: - Draft Badge Header

private struct DraftBadgeHeader: View {
    var body: some View {
        HStack {
            HStack(spacing: 8) {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.system(size: 14, weight: .semibold))
                Text("DRAFT PLAN")
                    .font(.system(size: 13, weight: .bold))
            }
            .foregroundColor(.orange)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color.orange.opacity(0.15))
            .clipShape(Capsule())

            Spacer()

            Text("Review & Approve")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Draft Plan Content

private struct DraftPlanContent: View {
    let plan: TradingPlanResponse
    @ObservedObject var viewModel: TradingPlanViewModel

    var body: some View {
        VStack(spacing: 16) {
            // Trade Style and Confidence
            HStack(alignment: .top) {
                // Trade style
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

                Spacer()

                // Confidence ring
                if let confidence = plan.confidence, confidence > 0 {
                    ZStack {
                        Circle()
                            .stroke(Color.gray.opacity(0.2), lineWidth: 4)
                            .frame(width: 50, height: 50)

                        Circle()
                            .trim(from: 0, to: CGFloat(confidence) / 100)
                            .stroke(confidenceColor(confidence), style: StrokeStyle(lineWidth: 4, lineCap: .round))
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
            }

            // Bias badge
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: biasIcon)
                        .font(.system(size: 14, weight: .semibold))
                    Text(plan.bias.uppercased())
                        .font(.system(size: 13, weight: .bold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(biasColor)
                .clipShape(Capsule())

                Spacer()
            }

            // Thesis
            VStack(alignment: .leading, spacing: 8) {
                Text("THESIS")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.secondary)

                Text(plan.thesis)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundColor(.primary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(Color(.tertiarySystemBackground))
            .cornerRadius(12)

            // Price Levels
            DraftPriceLevels(plan: plan)

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

            // Warnings/Notes
            if plan.hasNewsSentiment {
                DraftWarningsSection(plan: plan)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.orange.opacity(0.3), lineWidth: 2)
        )
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

    private var biasIcon: String {
        switch plan.bias.lowercased() {
        case "bullish": return "arrow.up.right"
        case "bearish": return "arrow.down.right"
        default: return "arrow.left.and.right"
        }
    }

    private var biasColor: Color {
        switch plan.bias.lowercased() {
        case "bullish": return .green
        case "bearish": return .red
        default: return .gray
        }
    }

    private func confidenceColor(_ confidence: Int) -> Color {
        if confidence >= 70 { return .green }
        if confidence >= 50 { return .yellow }
        return .orange
    }
}

// MARK: - Draft Price Levels

private struct DraftPriceLevels: View {
    let plan: TradingPlanResponse

    var body: some View {
        VStack(spacing: 12) {
            // Entry Zone
            if let entryLow = plan.entryZoneLow, let entryHigh = plan.entryZoneHigh {
                LevelRow(
                    label: "Entry Zone",
                    value: "$\(formatPrice(entryLow)) - $\(formatPrice(entryHigh))",
                    color: .blue,
                    icon: "arrow.right.circle.fill"
                )
            }

            // Stop Loss
            if let stop = plan.stopLoss {
                LevelRow(
                    label: "Stop Loss",
                    value: "$\(formatPrice(stop))",
                    color: .red,
                    icon: "xmark.circle.fill",
                    subtitle: plan.stopReasoning.isEmpty ? nil : plan.stopReasoning
                )
            }

            Divider()

            // Targets
            if let t1 = plan.target1 {
                LevelRow(
                    label: "Target 1",
                    value: "$\(formatPrice(t1))",
                    color: .green,
                    icon: "target"
                )
            }
            if let t2 = plan.target2 {
                LevelRow(
                    label: "Target 2",
                    value: "$\(formatPrice(t2))",
                    color: .green.opacity(0.8),
                    icon: "target"
                )
            }
            if let t3 = plan.target3 {
                LevelRow(
                    label: "Target 3",
                    value: "$\(formatPrice(t3))",
                    color: .green.opacity(0.6),
                    icon: "target"
                )
            }
        }
        .padding(14)
        .background(Color(.tertiarySystemBackground))
        .cornerRadius(12)
    }

    private func formatPrice(_ price: Double) -> String {
        String(format: "%.2f", price)
    }
}

private struct LevelRow: View {
    let label: String
    let value: String
    let color: Color
    let icon: String
    var subtitle: String? = nil

    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(color)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.secondary)

                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }

            Spacer()

            Text(value)
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundColor(.primary)
        }
    }
}

// MARK: - Draft Warnings Section

private struct DraftWarningsSection: View {
    let plan: TradingPlanResponse

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Things to Know", systemImage: "exclamationmark.triangle.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.orange)

            if let news = plan.newsSummary, !news.isEmpty {
                WarningItem(text: news)
            }

            if let sentiment = plan.redditSentiment, sentiment.lowercased() != "none" {
                WarningItem(text: "Reddit sentiment: \(plan.redditSentimentDisplay)")
            }
        }
        .padding(12)
        .background(Color.orange.opacity(0.1))
        .cornerRadius(10)
    }
}

private struct WarningItem: View {
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .foregroundColor(.orange)
            Text(text)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Draft Action Buttons

private struct DraftActionButtons: View {
    @ObservedObject var viewModel: TradingPlanViewModel
    let onApprove: () -> Void
    let onAdjust: () -> Void
    let onAskQuestion: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            // Primary approve button
            Button(action: onApprove) {
                HStack(spacing: 10) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 18))
                    Text("Approve Plan")
                        .font(.system(size: 16, weight: .semibold))
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color.green)
                .cornerRadius(12)
            }
            .disabled(viewModel.isUpdating || viewModel.isProcessingFeedback)

            // Secondary buttons row
            HStack(spacing: 12) {
                // Adjust button
                Button(action: onAdjust) {
                    HStack(spacing: 6) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 14))
                        Text("Adjust")
                            .font(.system(size: 14, weight: .medium))
                    }
                    .foregroundColor(.blue)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                }
                .disabled(viewModel.isProcessingFeedback)

                // Ask question button
                Button(action: onAskQuestion) {
                    HStack(spacing: 6) {
                        Image(systemName: "questionmark.circle")
                            .font(.system(size: 14))
                        Text("Ask Question")
                            .font(.system(size: 14, weight: .medium))
                    }
                    .foregroundColor(.purple)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(Color.purple.opacity(0.1))
                    .cornerRadius(10)
                }
                .disabled(viewModel.isProcessingFeedback)
            }
        }
    }
}

// MARK: - Conversation History Section

private struct ConversationHistorySection: View {
    let messages: [PlanMessage]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Conversation", systemImage: "bubble.left.and.bubble.right.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.secondary)

            VStack(spacing: 10) {
                ForEach(messages.suffix(4)) { message in
                    ConversationBubble(message: message)
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

private struct ConversationBubble: View {
    let message: PlanMessage

    var body: some View {
        HStack {
            if message.isUser { Spacer(minLength: 40) }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .font(.system(size: 13))
                    .foregroundColor(message.isUser ? .white : .primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(message.isUser ? Color.blue : Color(.tertiarySystemBackground))
                    .cornerRadius(16)
            }

            if !message.isUser { Spacer(minLength: 40) }
        }
    }
}

// MARK: - Draft Loading View

private struct DraftLoadingView: View {
    let phase: TradingPlanViewModel.UpdatePhase

    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.2)

            Text(phaseText)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.secondary)

            // Progress indicator
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 6)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.blue)
                        .frame(width: geo.size.width * progress, height: 6)
                        .animation(.easeInOut(duration: 0.3), value: phase)
                }
            }
            .frame(height: 6)
            .padding(.horizontal, 40)
        }
        .padding(40)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private var phaseText: String {
        switch phase {
        case .idle: return "Preparing..."
        case .gatheringData: return "Gathering market data..."
        case .analyzing: return "Analyzing technicals..."
        case .generatingPlan: return "AI generating plan..."
        case .complete: return "Plan ready!"
        }
    }

    private var progress: Double {
        switch phase {
        case .idle: return 0.1
        case .gatheringData: return 0.3
        case .analyzing: return 0.5
        case .generatingPlan: return 0.8
        case .complete: return 1.0
        }
    }
}

// MARK: - Feedback Sheet

private struct FeedbackSheet: View {
    @ObservedObject var viewModel: TradingPlanViewModel
    let mode: DraftPlanView.FeedbackMode
    let onDismiss: () -> Void

    @State private var inputText: String = ""
    @FocusState private var isInputFocused: Bool

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // AI Response (if any)
                if let response = viewModel.lastAIResponse {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            Label("AI Response", systemImage: "sparkles")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(.purple)

                            Text(response)
                                .font(.system(size: 14))
                                .foregroundColor(.primary)
                        }
                        .padding()
                    }
                    .frame(maxHeight: 300)
                    .background(Color(.secondarySystemBackground))
                }

                Spacer()

                // Input area
                VStack(spacing: 12) {
                    Text(mode == .question ? "What would you like to know?" : "What would you like to adjust?")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)

                    TextEditor(text: $inputText)
                        .focused($isInputFocused)
                        .frame(height: 100)
                        .padding(12)
                        .background(Color(.tertiarySystemBackground))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                        )

                    Button(action: submitFeedback) {
                        HStack {
                            if viewModel.isProcessingFeedback {
                                ProgressView()
                                    .scaleEffect(0.8)
                                    .tint(.white)
                            } else {
                                Image(systemName: mode == .question ? "paperplane.fill" : "arrow.triangle.2.circlepath")
                            }
                            Text(mode == .question ? "Ask" : "Request Adjustment")
                        }
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(inputText.isEmpty ? Color.gray : (mode == .question ? Color.purple : Color.blue))
                        .cornerRadius(12)
                    }
                    .disabled(inputText.isEmpty || viewModel.isProcessingFeedback)
                }
                .padding()
                .background(Color(.systemBackground))
            }
            .navigationTitle(mode == .question ? "Ask a Question" : "Request Adjustment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        onDismiss()
                    }
                }
            }
        }
        .onAppear {
            isInputFocused = true
        }
    }

    private func submitFeedback() {
        let text = inputText
        inputText = ""

        Task {
            if mode == .question {
                await viewModel.askQuestion(text)
            } else {
                await viewModel.requestAdjustment(text)
            }
        }
    }
}
