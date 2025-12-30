import SwiftUI

// MARK: - Analysis Step Model

enum AnalysisStepType: String, CaseIterable {
    case gatheringData = "gathering_data"
    case technicalIndicators = "technical_indicators"
    case supportResistance = "support_resistance"
    case chartPatterns = "chart_patterns"
    case generatingChart = "generating_chart"
    case visionAnalysis = "vision_analysis"
    case generatingPlan = "generating_plan"
    case complete = "complete"

    var displayTitle: String {
        switch self {
        case .gatheringData: return "Gathering market data"
        case .technicalIndicators: return "Calculating technical indicators"
        case .supportResistance: return "Analyzing support & resistance"
        case .chartPatterns: return "Detecting chart patterns"
        case .generatingChart: return "Generating candlestick chart"
        case .visionAnalysis: return "Analyzing chart with Vision"
        case .generatingPlan: return "Generating trading plan"
        case .complete: return "Analysis complete"
        }
    }

    var icon: String {
        switch self {
        case .gatheringData: return "antenna.radiowaves.left.and.right"
        case .technicalIndicators: return "function"
        case .supportResistance: return "arrow.up.arrow.down"
        case .chartPatterns: return "chart.xyaxis.line"
        case .generatingChart: return "chart.bar.xaxis"
        case .visionAnalysis: return "eye"
        case .generatingPlan: return "doc.text"
        case .complete: return "checkmark.seal.fill"
        }
    }
}

struct AnalysisStep: Identifiable, Equatable {
    let id = UUID()
    let type: AnalysisStepType
    var status: StepStatus
    var findings: [String]
    var timestamp: Date?

    enum StepStatus: Equatable {
        case pending
        case active
        case completed
    }
}

// MARK: - Agent Streaming View

struct AgentStreamingView: View {
    let symbol: String
    @Binding var steps: [AnalysisStep]
    @Binding var isComplete: Bool

    @State private var visibleFindings: [UUID: [String]] = [:]
    @State private var typingStates: [UUID: String] = [:]
    @State private var pulseAnimation = false
    @State private var typewriterTimers: [UUID: Timer] = [:]  // Track timers for cleanup

    private let accentCyan = Color(red: 0.0, green: 0.87, blue: 0.87)
    private let accentGreen = Color(red: 0.0, green: 0.85, blue: 0.45)
    private let accentAmber = Color(red: 1.0, green: 0.76, blue: 0.03)
    private let terminalBg = Color(red: 0.06, green: 0.07, blue: 0.09)
    private let cardBg = Color(red: 0.09, green: 0.10, blue: 0.12)

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // Terminal Header
                    TerminalHeader(symbol: symbol, accentCyan: accentCyan)
                        .padding(.bottom, 20)

                    // Analysis Steps
                    ForEach(Array(steps.enumerated()), id: \.element.id) { index, step in
                        AnalysisStepRow(
                            step: step,
                            stepNumber: index + 1,
                            visibleFindings: visibleFindings[step.id] ?? [],
                            typingText: typingStates[step.id] ?? "",
                            accentCyan: accentCyan,
                            accentGreen: accentGreen,
                            pulseAnimation: pulseAnimation
                        )
                        .id(step.id)
                        .transition(.asymmetric(
                            insertion: .opacity.combined(with: .move(edge: .leading)),
                            removal: .opacity
                        ))
                    }

                    // Completion indicator
                    if isComplete {
                        CompletionBanner(accentGreen: accentGreen)
                            .padding(.top, 24)
                            .transition(.opacity.combined(with: .scale(scale: 0.95)))
                    }

                    Spacer(minLength: 100)
                }
                .padding(20)
            }
            .onChange(of: steps) { _, newSteps in
                if let lastActive = newSteps.last(where: { $0.status == .active }) {
                    withAnimation(.easeOut(duration: 0.3)) {
                        proxy.scrollTo(lastActive.id, anchor: .center)
                    }
                }
            }
        }
        .background(terminalBg)
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                pulseAnimation = true
            }
        }
        .onDisappear {
            // Cleanup: invalidate all typewriter timers to prevent memory leaks
            for timer in typewriterTimers.values {
                timer.invalidate()
            }
            typewriterTimers.removeAll()
        }
        .onChange(of: steps) { oldSteps, newSteps in
            animateFindingsForChangedSteps(oldSteps: oldSteps, newSteps: newSteps)
        }
    }

    private func animateFindingsForChangedSteps(oldSteps: [AnalysisStep], newSteps: [AnalysisStep]) {
        for newStep in newSteps {
            let oldStep = oldSteps.first { $0.id == newStep.id }
            let oldFindings = oldStep?.findings ?? []
            let newFindings = newStep.findings

            // Find new findings that weren't in old
            let addedFindings = newFindings.filter { !oldFindings.contains($0) }

            for finding in addedFindings {
                animateTypewriter(stepId: newStep.id, text: finding)
            }
        }
    }

    private func animateTypewriter(stepId: UUID, text: String) {
        // Cancel any existing timer for this step to avoid duplicates
        typewriterTimers[stepId]?.invalidate()

        var currentIndex = 0
        let characters = Array(text)

        let timer = Timer.scheduledTimer(withTimeInterval: 0.02, repeats: true) { timer in
            if currentIndex < characters.count {
                typingStates[stepId] = String(characters.prefix(currentIndex + 1))
                currentIndex += 1
            } else {
                timer.invalidate()
                typewriterTimers[stepId] = nil
                typingStates[stepId] = nil

                // Add to visible findings
                withAnimation(.easeOut(duration: 0.2)) {
                    if visibleFindings[stepId] == nil {
                        visibleFindings[stepId] = []
                    }
                    visibleFindings[stepId]?.append(text)
                }
            }
        }

        // Track timer for cleanup
        typewriterTimers[stepId] = timer
    }
}

// MARK: - Terminal Header

private struct TerminalHeader: View {
    let symbol: String
    let accentCyan: Color

    @State private var cursorVisible = true
    @State private var cursorTimer: Timer?  // Track timer for cleanup

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Terminal title bar
            HStack(spacing: 8) {
                Circle().fill(Color.red.opacity(0.8)).frame(width: 12, height: 12)
                Circle().fill(Color.yellow.opacity(0.8)).frame(width: 12, height: 12)
                Circle().fill(Color.green.opacity(0.8)).frame(width: 12, height: 12)

                Spacer()

                Text("stockmate-agent v2.0")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.4))

                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color.white.opacity(0.03))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

            // Command line
            HStack(spacing: 0) {
                Text("$ ")
                    .foregroundStyle(accentCyan)

                Text("analyze ")
                    .foregroundStyle(.white.opacity(0.6))

                Text("--symbol ")
                    .foregroundStyle(Color.purple.opacity(0.8))

                Text(symbol)
                    .foregroundStyle(accentCyan)
                    .fontWeight(.semibold)

                Text(" --deep")
                    .foregroundStyle(.white.opacity(0.4))

                // Blinking cursor
                Rectangle()
                    .fill(accentCyan)
                    .frame(width: 8, height: 16)
                    .opacity(cursorVisible ? 1 : 0)
                    .padding(.leading, 2)
            }
            .font(.system(size: 14, weight: .regular, design: .monospaced))
            .onAppear {
                cursorTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
                    cursorVisible.toggle()
                }
            }
            .onDisappear {
                cursorTimer?.invalidate()
                cursorTimer = nil
            }
        }
    }
}

// MARK: - Analysis Step Row

private struct AnalysisStepRow: View {
    let step: AnalysisStep
    let stepNumber: Int
    let visibleFindings: [String]
    let typingText: String
    let accentCyan: Color
    let accentGreen: Color
    let pulseAnimation: Bool

    private let cardBg = Color(red: 0.09, green: 0.10, blue: 0.12)

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Step header
            HStack(spacing: 12) {
                // Status indicator
                ZStack {
                    Circle()
                        .stroke(statusColor.opacity(0.3), lineWidth: 2)
                        .frame(width: 32, height: 32)

                    if step.status == .active {
                        Circle()
                            .fill(statusColor.opacity(0.2))
                            .frame(width: 32, height: 32)
                            .scaleEffect(pulseAnimation ? 1.3 : 1.0)
                            .opacity(pulseAnimation ? 0 : 0.5)

                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: statusColor))
                            .scaleEffect(0.7)
                    } else if step.status == .completed {
                        Image(systemName: "checkmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(accentGreen)
                    } else {
                        Text("\(stepNumber)")
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.3))
                    }
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Image(systemName: step.type.icon)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(statusColor)

                        Text(step.type.displayTitle)
                            .font(.system(size: 14, weight: .medium, design: .monospaced))
                            .foregroundStyle(step.status == .pending ? .white.opacity(0.4) : .white)
                    }

                    if let timestamp = step.timestamp, step.status == .completed {
                        Text(formatDuration(from: timestamp))
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.3))
                    }
                }

                Spacer()

                if step.status == .active {
                    Text("RUNNING")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(accentCyan)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(accentCyan.opacity(0.15))
                        .clipShape(Capsule())
                }
            }
            .padding(16)

            // Findings section
            if !visibleFindings.isEmpty || !typingText.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    // Completed findings
                    ForEach(visibleFindings, id: \.self) { finding in
                        FindingRow(text: finding, accentCyan: accentCyan)
                    }

                    // Currently typing finding
                    if !typingText.isEmpty {
                        FindingRow(text: typingText, accentCyan: accentCyan, isTyping: true)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
                .padding(.leading, 44) // Align with text
            }

            // Connector line
            if step.type != .complete && step.status != .pending {
                Rectangle()
                    .fill(
                        LinearGradient(
                            colors: [statusColor.opacity(0.3), statusColor.opacity(0.1)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: 2, height: 20)
                    .padding(.leading, 31)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(step.status == .active ? cardBg : Color.clear)
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(step.status == .active ? accentCyan.opacity(0.2) : Color.clear, lineWidth: 1)
                )
        )
        .animation(.easeOut(duration: 0.3), value: step.status)
    }

    private var statusColor: Color {
        switch step.status {
        case .pending: return .white.opacity(0.3)
        case .active: return accentCyan
        case .completed: return accentGreen
        }
    }

    private func formatDuration(from date: Date) -> String {
        let duration = Date().timeIntervalSince(date)
        return String(format: "%.1fs", duration)
    }
}

// MARK: - Finding Row

private struct FindingRow: View {
    let text: String
    let accentCyan: Color
    var isTyping: Bool = false

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text("→")
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(accentCyan.opacity(0.6))

            // Parse and style the finding text
            styledText
                .font(.system(size: 12, weight: .regular, design: .monospaced))

            if isTyping {
                Rectangle()
                    .fill(accentCyan)
                    .frame(width: 6, height: 14)
                    .opacity(0.8)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 12)
        .background(Color.white.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }

    @ViewBuilder
    private var styledText: some View {
        // Check for different patterns and style accordingly
        if text.contains(":") {
            let parts = text.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                HStack(spacing: 4) {
                    Text(String(parts[0]) + ":")
                        .foregroundStyle(.white.opacity(0.5))
                    Text(String(parts[1]).trimmingCharacters(in: .whitespaces))
                        .foregroundStyle(colorForValue(String(parts[1])))
                }
            } else {
                Text(text)
                    .foregroundStyle(.white.opacity(0.8))
            }
        } else {
            Text(text)
                .foregroundStyle(.white.opacity(0.8))
        }
    }

    private func colorForValue(_ value: String) -> Color {
        let lower = value.lowercased()
        if lower.contains("bullish") || lower.contains("strong") || lower.contains("above") {
            return Color(red: 0.0, green: 0.85, blue: 0.45)
        } else if lower.contains("bearish") || lower.contains("weak") || lower.contains("below") {
            return Color(red: 1.0, green: 0.4, blue: 0.4)
        } else if lower.contains("neutral") || lower.contains("mixed") {
            return Color(red: 1.0, green: 0.76, blue: 0.03)
        }
        return accentCyan
    }
}

// MARK: - Completion Banner

private struct CompletionBanner: View {
    let accentGreen: Color

    @State private var showCheckmark = false
    @State private var expandBanner = false

    var body: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(accentGreen.opacity(0.2))
                    .frame(width: 48, height: 48)

                Image(systemName: "checkmark.seal.fill")
                    .font(.system(size: 24, weight: .medium))
                    .foregroundStyle(accentGreen)
                    .scaleEffect(showCheckmark ? 1 : 0)
                    .rotationEffect(.degrees(showCheckmark ? 0 : -180))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("ANALYSIS COMPLETE")
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white)

                Text("Trading plan ready for review")
                    .font(.system(size: 12, weight: .regular, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(accentGreen)
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(accentGreen.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(accentGreen.opacity(0.3), lineWidth: 1)
                )
        )
        .scaleEffect(expandBanner ? 1 : 0.95)
        .opacity(expandBanner ? 1 : 0)
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7)) {
                expandBanner = true
            }
            withAnimation(.spring(response: 0.4, dampingFraction: 0.6).delay(0.2)) {
                showCheckmark = true
            }
        }
    }
}

// MARK: - Preview

#Preview("Agent Streaming - Active") {
    AgentStreamingView(
        symbol: "AAPL",
        steps: .constant([
            AnalysisStep(type: .gatheringData, status: .completed, findings: ["Price: $189.45", "Bid: $189.43 | Ask: $189.47"], timestamp: Date().addingTimeInterval(-3)),
            AnalysisStep(type: .technicalIndicators, status: .completed, findings: ["RSI: 52.3 (neutral)", "MACD: Bullish crossover", "EMA: Price above 9, 21, 50"], timestamp: Date().addingTimeInterval(-2)),
            AnalysisStep(type: .supportResistance, status: .completed, findings: ["Support: $185.50, $182.00", "Resistance: $192.00, $198.00"], timestamp: Date().addingTimeInterval(-1)),
            AnalysisStep(type: .chartPatterns, status: .active, findings: ["Scanning for patterns..."], timestamp: nil),
            AnalysisStep(type: .generatingChart, status: .pending, findings: [], timestamp: nil),
            AnalysisStep(type: .visionAnalysis, status: .pending, findings: [], timestamp: nil),
            AnalysisStep(type: .generatingPlan, status: .pending, findings: [], timestamp: nil),
        ]),
        isComplete: .constant(false)
    )
}

#Preview("Agent Streaming - Complete") {
    AgentStreamingView(
        symbol: "TSLA",
        steps: .constant([
            AnalysisStep(type: .gatheringData, status: .completed, findings: ["Price: $248.50"], timestamp: Date().addingTimeInterval(-8)),
            AnalysisStep(type: .technicalIndicators, status: .completed, findings: ["RSI: 68.2 (approaching overbought)", "MACD: Strong bullish momentum"], timestamp: Date().addingTimeInterval(-6)),
            AnalysisStep(type: .supportResistance, status: .completed, findings: ["Key support at $240"], timestamp: Date().addingTimeInterval(-4)),
            AnalysisStep(type: .chartPatterns, status: .completed, findings: ["Found: Bull flag forming"], timestamp: Date().addingTimeInterval(-3)),
            AnalysisStep(type: .generatingChart, status: .completed, findings: ["Chart rendered (1050x700)"], timestamp: Date().addingTimeInterval(-2)),
            AnalysisStep(type: .visionAnalysis, status: .completed, findings: ["Trend: Clean uptrend", "Visual confidence: +12"], timestamp: Date().addingTimeInterval(-1)),
            AnalysisStep(type: .generatingPlan, status: .completed, findings: ["Bias: Bullish", "Confidence: 78%"], timestamp: Date()),
        ]),
        isComplete: .constant(true)
    )
}
