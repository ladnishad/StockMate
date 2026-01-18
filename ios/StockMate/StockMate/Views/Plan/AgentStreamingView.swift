import SwiftUI

// MARK: - Analysis Step Model

enum AnalysisStepType: String, CaseIterable {
    // V1 steps (existing)
    case gatheringData = "gathering_data"
    case technicalIndicators = "technical_indicators"
    case supportResistance = "support_resistance"
    case chartPatterns = "chart_patterns"
    case generatingChart = "generating_chart"
    case visionAnalysis = "vision_analysis"
    case generatingPlan = "generating_plan"
    case complete = "complete"

    // V2 steps (sub-agent orchestration)
    case gatheringCommonData = "gathering_common_data"
    case spawningSubagents = "spawning_subagents"
    case waitingForSubagents = "waiting_for_subagents"
    case selectingBest = "selecting_best"

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
        // V2 titles
        case .gatheringCommonData: return "Gathering common data"
        case .spawningSubagents: return "Spawning trade analyzers"
        case .waitingForSubagents: return "Analyzing in parallel"
        case .selectingBest: return "Selecting best plan"
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
        // V2 icons
        case .gatheringCommonData: return "antenna.radiowaves.left.and.right"
        case .spawningSubagents: return "arrow.triangle.branch"
        case .waitingForSubagents: return "arrow.triangle.merge"
        case .selectingBest: return "star.fill"
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

// MARK: - V2 Sub-Agent Models

/// Status for individual sub-agents in the v2 parallel flow
enum SubAgentStatus: String, Codable {
    case pending
    case running
    case gatheringData = "gathering_data"
    case calculatingTechnicals = "calculating_technicals"
    case generatingChart = "generating_chart"
    case analyzingChart = "analyzing_chart"
    case generatingPlan = "generating_plan"
    case completed
    case failed

    var displayText: String {
        switch self {
        case .pending: return "Waiting"
        case .running: return "Starting"
        case .gatheringData: return "Gathering data"
        case .calculatingTechnicals: return "Calculating indicators"
        case .generatingChart: return "Generating chart"
        case .analyzingChart: return "Vision analysis"
        case .generatingPlan: return "Creating plan"
        case .completed: return "Complete"
        case .failed: return "Failed"
        }
    }

    var isActive: Bool {
        switch self {
        case .pending, .completed, .failed: return false
        default: return true
        }
    }
}

// MARK: - Sub-Agent Step Types (V1-style hierarchical)

/// Step types for V1-style hierarchical display within each sub-agent
enum SubAgentStepType: String, CaseIterable, Codable {
    case data = "data"
    case technicals = "technicals"
    case levels = "levels"
    case patterns = "patterns"
    case chart = "chart"
    case vision = "vision"
    case plan = "plan"

    var displayName: String {
        switch self {
        case .data: return "Market Data"
        case .technicals: return "Technicals"
        case .levels: return "Key Levels"
        case .patterns: return "Patterns"
        case .chart: return "Chart"
        case .vision: return "Vision"
        case .plan: return "Plan"
        }
    }

    var icon: String {
        switch self {
        case .data: return "antenna.radiowaves.left.and.right"
        case .technicals: return "function"
        case .levels: return "arrow.up.arrow.down"
        case .patterns: return "chart.xyaxis.line"
        case .chart: return "chart.bar.xaxis"
        case .vision: return "eye"
        case .plan: return "doc.text"
        }
    }

    /// Order index for sorting
    var order: Int {
        switch self {
        case .data: return 0
        case .technicals: return 1
        case .levels: return 2
        case .patterns: return 3
        case .chart: return 4
        case .vision: return 5
        case .plan: return 6
        }
    }
}

/// Progress for a single step within a sub-agent (V1-style)
struct SubAgentStepProgress: Identifiable, Equatable {
    let id: String
    let type: SubAgentStepType
    var status: AnalysisStep.StepStatus
    var findings: [String]
    var timestamp: Date?

    init(type: SubAgentStepType, status: AnalysisStep.StepStatus = .pending, findings: [String] = [], timestamp: Date? = nil) {
        self.id = type.rawValue
        self.type = type
        self.status = status
        self.findings = findings
        self.timestamp = timestamp
    }
}

/// Progress for a single sub-agent
struct SubAgentProgress: Identifiable, Equatable, Codable {
    var id: String { agentName }
    let agentName: String
    let displayName: String
    var status: SubAgentStatus
    var currentStep: String?
    var stepsCompleted: [String]
    var findings: [String]
    var elapsedMs: Int
    var errorMessage: String?

    /// V1-style structured steps (derived from stepsCompleted/findings)
    var structuredSteps: [SubAgentStepProgress] {
        parseStructuredSteps()
    }

    enum CodingKeys: String, CodingKey {
        case agentName = "agent_name"
        case displayName = "display_name"
        case status
        case currentStep = "current_step"
        case stepsCompleted = "steps_completed"
        case findings
        case elapsedMs = "elapsed_ms"
        case errorMessage = "error_message"
    }

    /// Memberwise initializer for creating instances programmatically
    init(
        agentName: String,
        displayName: String,
        status: SubAgentStatus,
        currentStep: String?,
        stepsCompleted: [String],
        findings: [String],
        elapsedMs: Int,
        errorMessage: String?
    ) {
        self.agentName = agentName
        self.displayName = displayName
        self.status = status
        self.currentStep = currentStep
        self.stepsCompleted = stepsCompleted
        self.findings = findings
        self.elapsedMs = elapsedMs
        self.errorMessage = errorMessage
    }

    var icon: String {
        switch agentName {
        case "day-trade-analyzer": return "clock"
        case "swing-trade-analyzer": return "chart.line.uptrend.xyaxis"
        case "position-trade-analyzer": return "chart.bar.fill"
        default: return "cpu"
        }
    }

    var accentColor: Color {
        switch agentName {
        case "day-trade-analyzer": return Color(red: 1.0, green: 0.6, blue: 0.0)  // Orange
        case "swing-trade-analyzer": return Color(red: 0.4, green: 0.7, blue: 1.0)  // Blue
        case "position-trade-analyzer": return Color(red: 0.7, green: 0.5, blue: 1.0)  // Purple
        default: return Color.white
        }
    }

    // MARK: - Parse Structured Steps

    /// Parses flat stepsCompleted/findings into V1-style hierarchical steps
    private func parseStructuredSteps() -> [SubAgentStepProgress] {
        var steps: [SubAgentStepProgress] = []

        // Determine which steps are completed based on status and currentStep
        let completedTypes = determineCompletedStepTypes()
        let activeType = determineActiveStepType()

        for stepType in SubAgentStepType.allCases {
            var stepStatus: AnalysisStep.StepStatus = .pending
            var stepFindings: [String] = []

            if completedTypes.contains(stepType) {
                stepStatus = .completed
                stepFindings = findingsForStep(stepType)
            } else if stepType == activeType {
                stepStatus = .active
                stepFindings = findingsForStep(stepType)
            }

            // Only include steps that have started or are relevant
            if stepStatus != .pending || shouldShowPendingStep(stepType, activeType: activeType) {
                steps.append(SubAgentStepProgress(
                    type: stepType,
                    status: stepStatus,
                    findings: stepFindings
                ))
            }
        }

        return steps.sorted { $0.type.order < $1.type.order }
    }

    /// Determines which step types are completed based on current progress
    private func determineCompletedStepTypes() -> Set<SubAgentStepType> {
        var completed: Set<SubAgentStepType> = []

        // Check status-based completion
        switch status {
        case .completed:
            // All steps completed
            return Set(SubAgentStepType.allCases)
        case .generatingPlan:
            completed = [.data, .technicals, .levels, .patterns, .chart, .vision]
        case .analyzingChart:
            completed = [.data, .technicals, .levels, .patterns, .chart]
        case .generatingChart:
            completed = [.data, .technicals, .levels, .patterns]
        case .calculatingTechnicals:
            completed = [.data]
        case .gatheringData:
            completed = []
        default:
            break
        }

        // Also check stepsCompleted strings for more granular completion
        for step in stepsCompleted {
            let lower = step.lowercased()
            if lower.contains("bar") || lower.contains("data") || lower.contains("price") {
                completed.insert(.data)
            }
            if lower.contains("ema") || lower.contains("rsi") || lower.contains("macd") || lower.contains("indicator") || lower.contains("technical") {
                completed.insert(.technicals)
            }
            if lower.contains("level") || lower.contains("support") || lower.contains("resistance") {
                completed.insert(.levels)
            }
            if lower.contains("pattern") {
                completed.insert(.patterns)
            }
            if lower.contains("chart") && !lower.contains("vision") && !lower.contains("analy") {
                completed.insert(.chart)
            }
            if lower.contains("vision") || lower.contains("image") {
                completed.insert(.vision)
            }
            if lower.contains("plan") || lower.contains("recommendation") {
                completed.insert(.plan)
            }
        }

        return completed
    }

    /// Determines the currently active step type
    private func determineActiveStepType() -> SubAgentStepType? {
        switch status {
        case .gatheringData, .running:
            return .data
        case .calculatingTechnicals:
            return .technicals
        case .generatingChart:
            return .chart
        case .analyzingChart:
            return .vision
        case .generatingPlan:
            return .plan
        default:
            // Check currentStep string
            if let current = currentStep?.lowercased() {
                if current.contains("bar") || current.contains("data") || current.contains("price") {
                    return .data
                }
                if current.contains("ema") || current.contains("indicator") || current.contains("technical") {
                    return .technicals
                }
                if current.contains("level") || current.contains("support") {
                    return .levels
                }
                if current.contains("pattern") {
                    return .patterns
                }
                if current.contains("chart") && !current.contains("vision") {
                    return .chart
                }
                if current.contains("vision") || current.contains("image") || current.contains("analy") {
                    return .vision
                }
                if current.contains("plan") {
                    return .plan
                }
            }
            return nil
        }
    }

    /// Gets findings relevant to a specific step type
    private func findingsForStep(_ stepType: SubAgentStepType) -> [String] {
        return findings.filter { finding in
            let lower = finding.lowercased()
            switch stepType {
            case .data:
                return lower.contains("price") || lower.contains("bid") || lower.contains("ask") ||
                       lower.contains("volume") || lower.contains("bar")
            case .technicals:
                return lower.contains("rsi") || lower.contains("macd") || lower.contains("ema") ||
                       lower.contains("indicator") || lower.contains("momentum")
            case .levels:
                return lower.contains("support") || lower.contains("resistance") || lower.contains("level")
            case .patterns:
                return lower.contains("pattern") || lower.contains("flag") || lower.contains("triangle") ||
                       lower.contains("breakout") || lower.contains("wedge")
            case .chart:
                return lower.contains("chart") || lower.contains("render") || lower.contains("candlestick")
            case .vision:
                return lower.contains("vision") || lower.contains("trend") || lower.contains("visual") ||
                       lower.contains("analysis")
            case .plan:
                return lower.contains("bias") || lower.contains("confidence") || lower.contains("entry") ||
                       lower.contains("target") || lower.contains("stop") || lower.contains("recommendation")
            }
        }
    }

    /// Determines if a pending step should be shown (shows only the next pending step for progressive reveal)
    private func shouldShowPendingStep(_ stepType: SubAgentStepType, activeType: SubAgentStepType?) -> Bool {
        guard let active = activeType else {
            // If no active step, show only the first pending step
            return stepType.order == 0
        }
        // Only show the immediately next pending step (1 ahead) for progressive reveal
        return stepType.order == active.order + 1
    }
}

// MARK: - Sub-Agent Progress Row (V2)

struct SubAgentProgressRow: View {
    let agent: SubAgentProgress
    let pulseAnimation: Bool

    private let cardBg = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let accentGreen = Color(red: 0.0, green: 0.85, blue: 0.45)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with agent name and status
            HStack(spacing: 10) {
                // Agent icon
                ZStack {
                    Circle()
                        .fill(agent.accentColor.opacity(0.15))
                        .frame(width: 36, height: 36)

                    if agent.status.isActive {
                        Circle()
                            .fill(agent.accentColor.opacity(0.2))
                            .frame(width: 36, height: 36)
                            .scaleEffect(pulseAnimation ? 1.3 : 1.0)
                            .opacity(pulseAnimation ? 0 : 0.5)

                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: agent.accentColor))
                            .scaleEffect(0.6)
                    } else if agent.status == .completed {
                        Image(systemName: "checkmark")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(accentGreen)
                    } else if agent.status == .failed {
                        Image(systemName: "xmark")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(.red)
                    } else {
                        Image(systemName: agent.icon)
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(agent.accentColor.opacity(0.5))
                    }
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(agent.displayName)
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.white)

                    Text(agent.currentStep ?? agent.status.displayText)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                        .lineLimit(1)
                }

                Spacer()

                // Status badge
                if agent.status.isActive {
                    Text("RUNNING")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(agent.accentColor)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(agent.accentColor.opacity(0.15))
                        .clipShape(Capsule())
                } else if agent.status == .completed {
                    Text("DONE")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(accentGreen)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(accentGreen.opacity(0.15))
                        .clipShape(Capsule())
                }
            }

            // Findings (when completed)
            if agent.status == .completed && !agent.findings.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(agent.findings, id: \.self) { finding in
                        HStack(spacing: 6) {
                            Text("→")
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundStyle(agent.accentColor.opacity(0.6))

                            Text(finding)
                                .font(.system(size: 10, weight: .regular, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.7))
                        }
                    }
                }
                .padding(.leading, 46)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(agent.status.isActive ? cardBg : cardBg.opacity(0.5))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(
                            agent.status.isActive ? agent.accentColor.opacity(0.3) :
                            agent.status == .completed ? accentGreen.opacity(0.2) : Color.clear,
                            lineWidth: 1
                        )
                )
        )
    }
}

// MARK: - Parallel Sub-Agents Container (V2)

struct ParallelSubAgentsView: View {
    let subagents: [SubAgentProgress]
    let pulseAnimation: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: "arrow.triangle.branch")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))

                Text("Parallel Analysis")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.6))

                Spacer()

                let completed = subagents.filter { $0.status == .completed }.count
                Text("\(completed)/\(subagents.count)")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.4))
            }
            .padding(.horizontal, 4)

            // Sub-agent rows
            ForEach(subagents) { agent in
                SubAgentProgressRow(agent: agent, pulseAnimation: pulseAnimation)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.02))
        )
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
    @State private var expandedSteps: Set<UUID> = []  // Track expanded accordion steps

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
                            isExpanded: isStepExpanded(step),
                            canToggle: step.status == .completed && !step.findings.isEmpty,
                            onToggle: { toggleStepExpansion(step.id) },
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

    // MARK: - Accordion Helpers

    /// Determines if a step should show its findings expanded
    /// - Active steps are always expanded (streaming)
    /// - Completed steps are collapsed by default, but can be manually expanded
    /// - Pending steps are always collapsed
    private func isStepExpanded(_ step: AnalysisStep) -> Bool {
        switch step.status {
        case .active:
            return true  // Always show findings for active step
        case .completed:
            return expandedSteps.contains(step.id)  // User-controlled
        case .pending:
            return false  // Never expanded
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
    let isExpanded: Bool
    let canToggle: Bool
    let onToggle: () -> Void
    let accentCyan: Color
    let accentGreen: Color
    let pulseAnimation: Bool

    private let cardBg = Color(red: 0.09, green: 0.10, blue: 0.12)

    /// Whether to show findings section (expanded or has active typing)
    private var showFindings: Bool {
        (isExpanded && !visibleFindings.isEmpty) || !typingText.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Step header (tappable for completed steps)
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

                    // Show findings count when collapsed
                    if step.status == .completed {
                        HStack(spacing: 6) {
                            if let timestamp = step.timestamp {
                                Text(formatDuration(from: timestamp))
                                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.3))
                            }

                            if !visibleFindings.isEmpty && !isExpanded {
                                Text("•")
                                    .foregroundStyle(.white.opacity(0.2))
                                Text("\(visibleFindings.count) finding\(visibleFindings.count == 1 ? "" : "s")")
                                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                                    .foregroundStyle(accentCyan.opacity(0.6))
                            }
                        }
                    }
                }

                Spacer()

                // Status badge or chevron
                if step.status == .active {
                    Text("RUNNING")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(accentCyan)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(accentCyan.opacity(0.15))
                        .clipShape(Capsule())
                } else if canToggle {
                    // Chevron for expandable completed steps
                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.4))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isExpanded)
                }
            }
            .padding(16)
            .contentShape(Rectangle())
            .onTapGesture {
                if canToggle {
                    onToggle()
                }
            }

            // Findings section (animated)
            if showFindings {
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
                .transition(.asymmetric(
                    insertion: .opacity.combined(with: .move(edge: .top)),
                    removal: .opacity
                ))
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
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isExpanded)
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

// MARK: - V2 Sub-Agent Previews

#Preview("V2 Parallel Sub-Agents - Running") {
    let terminalBg = Color(red: 0.06, green: 0.07, blue: 0.09)

    return ScrollView {
        VStack(spacing: 20) {
            ParallelSubAgentsView(
                subagents: [
                    SubAgentProgress(
                        agentName: "day-trade-analyzer",
                        displayName: "Day Trade",
                        status: .analyzingChart,
                        currentStep: "Vision analysis (5-min chart)",
                        stepsCompleted: ["Gathered 5-min bars", "Calculated EMAs", "Generated chart"],
                        findings: [],
                        elapsedMs: 2500,
                        errorMessage: nil
                    ),
                    SubAgentProgress(
                        agentName: "swing-trade-analyzer",
                        displayName: "Swing Trade",
                        status: .generatingChart,
                        currentStep: "Generating daily chart",
                        stepsCompleted: ["Gathered daily bars", "Calculated EMAs"],
                        findings: [],
                        elapsedMs: 1800,
                        errorMessage: nil
                    ),
                    SubAgentProgress(
                        agentName: "position-trade-analyzer",
                        displayName: "Position Trade",
                        status: .gatheringData,
                        currentStep: "Gathering weekly bars",
                        stepsCompleted: [],
                        findings: [],
                        elapsedMs: 800,
                        errorMessage: nil
                    ),
                ],
                pulseAnimation: true
            )
        }
        .padding(20)
    }
    .background(terminalBg)
}

#Preview("V2 Parallel Sub-Agents - Complete") {
    let terminalBg = Color(red: 0.06, green: 0.07, blue: 0.09)

    return ScrollView {
        VStack(spacing: 20) {
            ParallelSubAgentsView(
                subagents: [
                    SubAgentProgress(
                        agentName: "day-trade-analyzer",
                        displayName: "Day Trade",
                        status: .completed,
                        currentStep: nil,
                        stepsCompleted: ["All steps"],
                        findings: ["Bullish", "65% confidence", "ATR: 3.2%"],
                        elapsedMs: 4200,
                        errorMessage: nil
                    ),
                    SubAgentProgress(
                        agentName: "swing-trade-analyzer",
                        displayName: "Swing Trade",
                        status: .completed,
                        currentStep: nil,
                        stepsCompleted: ["All steps"],
                        findings: ["Bullish", "78% confidence", "Bull flag forming"],
                        elapsedMs: 3800,
                        errorMessage: nil
                    ),
                    SubAgentProgress(
                        agentName: "position-trade-analyzer",
                        displayName: "Position Trade",
                        status: .completed,
                        currentStep: nil,
                        stepsCompleted: ["All steps"],
                        findings: ["Neutral", "No clear weekly trend"],
                        elapsedMs: 5100,
                        errorMessage: nil
                    ),
                ],
                pulseAnimation: false
            )
        }
        .padding(20)
    }
    .background(terminalBg)
}

// MARK: - Agentic Mode Models

/// Represents AI thinking/reasoning during investigation
struct AgentThinking: Identifiable {
    let id = UUID()
    let text: String
    let iteration: Int
    let timestamp: Date
}

/// Represents a tool being called by the AI
struct ToolCall: Identifiable {
    let id = UUID()
    let name: String
    let arguments: [String: Any]
    let iteration: Int
    let timestamp: Date

    var displayName: String {
        switch name {
        case "get_price": return "Checking Price"
        case "get_market_context": return "Checking Market"
        case "get_position": return "Checking Position"
        case "get_and_analyze_chart": return "Analyzing Chart"
        case "get_technicals": return "Getting Technicals"
        case "get_support_resistance": return "Finding Key Levels"
        case "get_fibonacci": return "Calculating Fibonacci"
        case "get_fundamentals": return "Getting Fundamentals"
        case "get_news": return "Checking News"
        case "search_x": return "Searching X"
        case "get_atr": return "Calculating ATR"
        default: return name.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    var icon: String {
        switch name {
        case "get_price": return "dollarsign.circle"
        case "get_market_context": return "chart.bar.xaxis"
        case "get_position": return "person.crop.square"
        case "get_and_analyze_chart": return "eye"
        case "get_technicals": return "function"
        case "get_support_resistance": return "arrow.up.arrow.down"
        case "get_fibonacci": return "ruler"
        case "get_fundamentals": return "building.columns"
        case "get_news": return "newspaper"
        case "search_x": return "at.badge.plus"
        case "get_atr": return "waveform.path.ecg"
        default: return "wrench"
        }
    }
}

/// Represents a tool result
struct ToolResult: Identifiable {
    let id = UUID()
    let toolName: String
    let result: [String: Any]
    let iteration: Int
    let timestamp: Date

    /// Summary of the result for collapsed view
    var summary: String {
        // Extract key value from result based on tool type
        if let price = result["price"] as? Double {
            return String(format: "$%.2f", price)
        }
        if let direction = result["market_direction"] as? String {
            return direction.capitalized
        }
        if let hasPosition = result["has_position"] as? Bool {
            return hasPosition ? "Has Position" : "No Position"
        }
        if let trendQuality = result["trend_quality"] as? String {
            return trendQuality.capitalized
        }
        if let sentiment = result["sentiment"] as? String {
            return sentiment.capitalized
        }
        if let atrPct = result["atr_pct"] as? Double {
            return String(format: "%.2f%%", atrPct)
        }
        if let valuation = result["valuation"] as? [String: Any],
           let assessment = valuation["assessment"] as? String {
            return assessment
        }
        return "View Details"
    }
}

/// An item in the agentic conversation stream
enum AgenticStreamItem: Identifiable {
    case thinking(AgentThinking)
    case toolCall(ToolCall)
    case toolResult(ToolResult)

    var id: UUID {
        switch self {
        case .thinking(let t): return t.id
        case .toolCall(let c): return c.id
        case .toolResult(let r): return r.id
        }
    }
}

// MARK: - Agentic Thinking Bubble (ChatGPT-style)

struct AgentThinkingBubble: View {
    let thinking: AgentThinking

    private let accentCyan = Color(red: 0.0, green: 0.87, blue: 0.87)

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(accentCyan)

                Text("AI Reasoning")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(accentCyan)

                Spacer()

                // Iteration badge
                Text("Step \(thinking.iteration)")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.white.opacity(0.1))
                    .clipShape(Capsule())
            }

            // Thinking text
            Text(thinking.text)
                .font(.system(size: 12, weight: .regular, design: .monospaced))
                .foregroundStyle(.white.opacity(0.85))
                .lineSpacing(4)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(accentCyan.opacity(0.15), lineWidth: 1)
                )
        )
    }
}

// MARK: - Tool Call Row

struct ToolCallRow: View {
    let toolCall: ToolCall
    let isActive: Bool
    @State private var pulseAnimation = false

    private let accentAmber = Color(red: 1.0, green: 0.76, blue: 0.03)

    var body: some View {
        HStack(spacing: 12) {
            // Tool icon with animation
            ZStack {
                Circle()
                    .fill(accentAmber.opacity(0.15))
                    .frame(width: 32, height: 32)

                if isActive {
                    Circle()
                        .fill(accentAmber.opacity(0.2))
                        .frame(width: 32, height: 32)
                        .scaleEffect(pulseAnimation ? 1.4 : 1.0)
                        .opacity(pulseAnimation ? 0 : 0.5)
                        .onAppear {
                            withAnimation(.easeInOut(duration: 1.0).repeatForever(autoreverses: false)) {
                                pulseAnimation = true
                            }
                        }

                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: accentAmber))
                        .scaleEffect(0.6)
                } else {
                    Image(systemName: toolCall.icon)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(accentAmber)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(toolCall.displayName)
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white)

                // Show key argument if any
                if let timeframe = toolCall.arguments["timeframe"] as? String {
                    Text("Timeframe: \(timeframe)")
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                } else if let tradeStyle = toolCall.arguments["trade_style"] as? String {
                    Text("Style: \(tradeStyle)")
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }

            Spacer()

            if isActive {
                Text("CALLING")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundStyle(accentAmber)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(accentAmber.opacity(0.15))
                    .clipShape(Capsule())
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(isActive ? Color(red: 0.09, green: 0.10, blue: 0.12) : Color.clear)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(isActive ? accentAmber.opacity(0.2) : Color.clear, lineWidth: 1)
                )
        )
    }
}

// MARK: - Tool Result Row (Collapsible)

struct ToolResultRow: View {
    let result: ToolResult
    @Binding var isExpanded: Bool

    private let accentGreen = Color(red: 0.0, green: 0.85, blue: 0.45)

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header (always visible)
            HStack(spacing: 12) {
                // Checkmark
                ZStack {
                    Circle()
                        .fill(accentGreen.opacity(0.15))
                        .frame(width: 32, height: 32)

                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(accentGreen)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(ToolCall(name: result.toolName, arguments: [:], iteration: 0, timestamp: Date()).displayName)
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white)

                    Text(result.summary)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(accentGreen)
                }

                Spacer()

                // Expand/collapse chevron
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.4))
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
            }
            .padding(12)
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                    isExpanded.toggle()
                }
            }

            // Expanded content
            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(result.result.keys.sorted()), id: \.self) { key in
                        if let value = result.result[key] {
                            HStack(alignment: .top, spacing: 8) {
                                Text("\(key):")
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.5))

                                Text(formatValue(value))
                                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.8))
                                    .lineLimit(3)
                            }
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 12)
                .padding(.leading, 44)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.02))
        )
    }

    private func formatValue(_ value: Any) -> String {
        if let dict = value as? [String: Any] {
            // Truncate long dicts
            let items = dict.prefix(3).map { "\($0.key): \($0.value)" }
            let result = items.joined(separator: ", ")
            return dict.count > 3 ? "\(result), ..." : result
        }
        if let array = value as? [Any] {
            // Truncate long arrays
            let items = array.prefix(3).map { "\($0)" }
            let result = items.joined(separator: ", ")
            return array.count > 3 ? "\(result), ..." : result
        }
        return "\(value)"
    }
}

// MARK: - Agentic Streaming View (Main View)

struct AgenticStreamingView: View {
    let symbol: String
    @Binding var streamItems: [AgenticStreamItem]
    @Binding var isComplete: Bool
    @Binding var finalPlan: [String: Any]?
    @Binding var expandedToolResults: Set<UUID>
    var onAcceptPlan: (() async -> Bool)?  // Callback when user accepts the plan

    @State private var activeToolCallId: UUID? = nil

    private let terminalBg = Color(red: 0.06, green: 0.07, blue: 0.09)
    private let accentCyan = Color(red: 0.0, green: 0.87, blue: 0.87)

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    // Terminal header
                    AgenticTerminalHeader(symbol: symbol)
                        .padding(.bottom, 12)

                    // Stream items
                    ForEach(streamItems) { item in
                        streamItemView(for: item)
                            .id(item.id)
                            .transition(.asymmetric(
                                insertion: .opacity.combined(with: .move(edge: .bottom)),
                                removal: .opacity
                            ))
                    }

                    // Final result
                    if isComplete, let plan = finalPlan {
                        FinalRecommendationCard(plan: plan, symbol: symbol, onAcceptPlan: onAcceptPlan)
                            .padding(.top, 16)
                            .transition(.opacity.combined(with: .scale(scale: 0.95)))
                    }

                    Spacer(minLength: 100)
                }
                .padding(20)
            }
            .onChange(of: streamItems.count) { _, _ in
                if let lastItem = streamItems.last {
                    withAnimation(.easeOut(duration: 0.3)) {
                        proxy.scrollTo(lastItem.id, anchor: .bottom)
                    }
                }
            }
        }
        .background(terminalBg)
    }

    @ViewBuilder
    private func streamItemView(for item: AgenticStreamItem) -> some View {
        switch item {
        case .thinking(let thinking):
            AgentThinkingBubble(thinking: thinking)

        case .toolCall(let toolCall):
            ToolCallRow(
                toolCall: toolCall,
                isActive: activeToolCallId == toolCall.id
            )
            .onAppear { activeToolCallId = toolCall.id }

        case .toolResult(let result):
            ToolResultRow(
                result: result,
                isExpanded: Binding(
                    get: { expandedToolResults.contains(result.id) },
                    set: { if $0 { expandedToolResults.insert(result.id) } else { expandedToolResults.remove(result.id) } }
                )
            )
            .onAppear { activeToolCallId = nil }
        }
    }
}

// MARK: - Agentic Terminal Header

struct AgenticTerminalHeader: View {
    let symbol: String
    private let accentCyan = Color(red: 0.0, green: 0.87, blue: 0.87)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Terminal title bar
            HStack(spacing: 8) {
                Circle().fill(Color.red.opacity(0.8)).frame(width: 12, height: 12)
                Circle().fill(Color.yellow.opacity(0.8)).frame(width: 12, height: 12)
                Circle().fill(Color.green.opacity(0.8)).frame(width: 12, height: 12)
                Spacer()
                Text("stockmate-agent v3.0 (agentic)")
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
                Text("investigate ")
                    .foregroundStyle(.white.opacity(0.6))
                Text("--stock ")
                    .foregroundStyle(Color.purple.opacity(0.8))
                Text(symbol)
                    .foregroundStyle(accentCyan)
                    .fontWeight(.semibold)
                Text(" --mode agentic")
                    .foregroundStyle(.white.opacity(0.4))
            }
            .font(.system(size: 14, weight: .regular, design: .monospaced))
        }
    }
}

// MARK: - Final Recommendation Card

struct FinalRecommendationCard: View {
    let plan: [String: Any]
    let symbol: String
    var onAcceptPlan: (() async -> Bool)?  // Returns true on success

    // Track which trade style is expanded (recommended style auto-expands)
    @State private var expandedStyle: String?
    @State private var showFullPlanSheet: Bool = false
    @State private var isAccepting: Bool = false
    @State private var isAccepted: Bool = false

    private let accentGreen = Color(red: 0.0, green: 0.85, blue: 0.45)
    private let accentAmber = Color(red: 1.0, green: 0.76, blue: 0.03)
    private let accentRed = Color(red: 1.0, green: 0.4, blue: 0.4)
    private let accentCyan = Color(red: 0.0, green: 0.87, blue: 0.87)

    private var recommendedStyle: String? {
        plan["recommended_style"] as? String
    }

    private var recommendedPlan: [String: Any]? {
        guard let style = recommendedStyle else { return nil }
        return plan["\(style)_trade_plan"] as? [String: Any]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(accentGreen.opacity(0.2))
                        .frame(width: 48, height: 48)
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 24, weight: .medium))
                        .foregroundStyle(accentGreen)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("ANALYSIS COMPLETE")
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)

                    if let style = recommendedStyle, style != "none" {
                        Text("Recommended: \(style.uppercased()) TRADE")
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(accentGreen)
                    } else {
                        Text("No Trade Recommended")
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(accentAmber)
                    }
                }

                Spacer()
            }

            // Reasoning
            if let reasoning = plan["recommendation_reasoning"] as? String {
                Text(reasoning)
                    .font(.system(size: 11, weight: .regular, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
                    .lineSpacing(3)
            }

            Divider().background(Color.white.opacity(0.1))

            // Expandable trade plan sections
            tradePlanSection("day", displayName: "Day Trade", plan: plan["day_trade_plan"] as? [String: Any])
            tradePlanSection("swing", displayName: "Swing Trade", plan: plan["swing_trade_plan"] as? [String: Any])
            tradePlanSection("position", displayName: "Position Trade", plan: plan["position_trade_plan"] as? [String: Any])

            // Risk Warnings
            if let warnings = plan["risk_warnings"] as? [String], !warnings.isEmpty {
                Divider().background(Color.white.opacity(0.1))
                riskWarningsSection(warnings)
            }

            // What to Watch
            if let watchItems = plan["what_to_watch"] as? [String], !watchItems.isEmpty {
                whatToWatchSection(watchItems)
            }

            // View Full Plan Button
            if let style = recommendedStyle, style != "none", recommendedPlan != nil {
                Divider().background(Color.white.opacity(0.1))

                Button(action: { showFullPlanSheet = true }) {
                    HStack {
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 14, weight: .medium))
                        Text("View Full \(style.capitalized) Trade Plan")
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                        Spacer()
                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(accentCyan)
                    .padding(.vertical, 12)
                    .padding(.horizontal, 16)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(accentCyan.opacity(0.1))
                    )
                }
                .buttonStyle(.plain)
            }

            // Accept Plan Button
            if let onAccept = onAcceptPlan, !isAccepted {
                Button(action: {
                    Task {
                        isAccepting = true
                        let success = await onAccept()
                        isAccepting = false
                        if success {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                isAccepted = true
                            }
                        }
                    }
                }) {
                    HStack {
                        if isAccepting {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 16, weight: .medium))
                        }
                        Text(isAccepting ? "Saving Plan..." : "Accept Plan")
                            .font(.system(size: 14, weight: .bold, design: .monospaced))
                        Spacer()
                    }
                    .foregroundStyle(.white)
                    .padding(.vertical, 14)
                    .padding(.horizontal, 16)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(accentGreen)
                    )
                }
                .buttonStyle(.plain)
                .disabled(isAccepting)
            }

            // Accepted confirmation
            if isAccepted {
                HStack {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 16, weight: .medium))
                    Text("Plan Saved for Evaluation")
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    Spacer()
                }
                .foregroundStyle(accentGreen)
                .padding(.vertical, 12)
                .padding(.horizontal, 16)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(accentGreen.opacity(0.15))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .stroke(accentGreen.opacity(0.3), lineWidth: 1)
                        )
                )
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(accentGreen.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(accentGreen.opacity(0.25), lineWidth: 1)
                )
        )
        .onAppear {
            // Auto-expand recommended style
            if let style = recommendedStyle, style != "none" {
                expandedStyle = style
            }
        }
        .sheet(isPresented: $showFullPlanSheet) {
            if let style = recommendedStyle, let tradePlan = recommendedPlan {
                AgenticPlanDetailSheet(
                    symbol: symbol,
                    tradeStyle: style,
                    plan: tradePlan,
                    riskWarnings: plan["risk_warnings"] as? [String] ?? [],
                    whatToWatch: plan["what_to_watch"] as? [String] ?? []
                )
            }
        }
    }

    // MARK: - Trade Plan Section (Expandable)

    @ViewBuilder
    private func tradePlanSection(_ styleKey: String, displayName: String, plan: [String: Any]?) -> some View {
        if let plan = plan {
            let isExpanded = expandedStyle == styleKey
            let isRecommended = recommendedStyle == styleKey
            let conviction = plan["conviction"] as? String ?? "low"
            let suitable = plan["suitable"] as? Bool ?? false

            VStack(alignment: .leading, spacing: 0) {
                // Header row (tappable)
                Button(action: {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                        expandedStyle = isExpanded ? nil : styleKey
                    }
                }) {
                    HStack(spacing: 8) {
                        // Recommended indicator
                        if isRecommended {
                            Image(systemName: "star.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(accentGreen)
                        }

                        Text(displayName)
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .foregroundStyle(isRecommended ? .white : .white.opacity(0.7))

                        convictionBadge(conviction)

                        if !suitable {
                            Text("Not Suitable")
                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.4))
                        }

                        Spacer()

                        Image(systemName: "chevron.right")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.4))
                            .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    }
                    .padding(.vertical, 10)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                // Expanded details
                if isExpanded {
                    VStack(alignment: .leading, spacing: 12) {
                        // Conviction reasoning
                        if let reasoning = plan["conviction_reasoning"] as? String {
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "lightbulb.fill")
                                    .font(.system(size: 10))
                                    .foregroundStyle(accentAmber)
                                    .frame(width: 16)
                                Text(reasoning)
                                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.7))
                                    .lineSpacing(2)
                            }
                        }

                        // Entry, Stop, Targets (if suitable)
                        if suitable {
                            levelDetailsView(plan)
                        }

                        // Full thesis
                        if let thesis = plan["thesis"] as? String {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("THESIS")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.5))
                                Text(thesis)
                                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.8))
                                    .lineSpacing(2)
                            }
                            .padding(.top, 4)
                        }

                        // Holding period
                        if let holdingPeriod = plan["holding_period"] as? String {
                            HStack(spacing: 6) {
                                Image(systemName: "clock")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.white.opacity(0.5))
                                Text("Hold: \(holdingPeriod)")
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.6))
                            }
                        }
                    }
                    .padding(.leading, 16)
                    .padding(.bottom, 12)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                Divider().background(Color.white.opacity(0.08))
            }
        }
    }

    // MARK: - Level Details (Entry, Stop, Targets)

    @ViewBuilder
    private func levelDetailsView(_ plan: [String: Any]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            // Entry Zone
            if let entryZone = plan["entry_zone"] as? [Double], entryZone.count >= 2 {
                levelRow(
                    icon: "arrow.right.circle.fill",
                    label: "Entry",
                    value: String(format: "$%.2f - $%.2f", entryZone[0], entryZone[1]),
                    color: accentGreen
                )
            } else if let entryZone = plan["entry_zone"] as? [Any], entryZone.count >= 2 {
                // Handle mixed types
                let low = (entryZone[0] as? Double) ?? (entryZone[0] as? Int).map(Double.init) ?? 0
                let high = (entryZone[1] as? Double) ?? (entryZone[1] as? Int).map(Double.init) ?? 0
                if low > 0 && high > 0 {
                    levelRow(
                        icon: "arrow.right.circle.fill",
                        label: "Entry",
                        value: String(format: "$%.2f - $%.2f", low, high),
                        color: accentGreen
                    )
                }
            }

            // Stop Loss
            if let stopLoss = plan["stop_loss"] as? Double {
                levelRow(
                    icon: "xmark.octagon.fill",
                    label: "Stop",
                    value: String(format: "$%.2f", stopLoss),
                    color: accentRed
                )
            } else if let stopLoss = plan["stop_loss"] as? Int {
                levelRow(
                    icon: "xmark.octagon.fill",
                    label: "Stop",
                    value: String(format: "$%.2f", Double(stopLoss)),
                    color: accentRed
                )
            }

            // Targets
            if let targets = plan["targets"] as? [Double], !targets.isEmpty {
                let targetStrings = targets.prefix(3).map { String(format: "$%.2f", $0) }
                levelRow(
                    icon: "target",
                    label: "Targets",
                    value: targetStrings.joined(separator: " → "),
                    color: accentGreen
                )
            } else if let targets = plan["targets"] as? [Any], !targets.isEmpty {
                let targetStrings = targets.prefix(3).compactMap { t -> String? in
                    if let d = t as? Double { return String(format: "$%.2f", d) }
                    if let i = t as? Int { return String(format: "$%.2f", Double(i)) }
                    return nil
                }
                if !targetStrings.isEmpty {
                    levelRow(
                        icon: "target",
                        label: "Targets",
                        value: targetStrings.joined(separator: " → "),
                        color: accentGreen
                    )
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.white.opacity(0.03))
        )
    }

    @ViewBuilder
    private func levelRow(icon: String, label: String, value: String, color: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundStyle(color)
                .frame(width: 16)

            Text(label)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
                .frame(width: 50, alignment: .leading)

            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white)
        }
    }

    // MARK: - Risk Warnings

    @ViewBuilder
    private func riskWarningsSection(_ warnings: [String]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(accentAmber)
                Text("RISK WARNINGS")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(accentAmber)
            }

            VStack(alignment: .leading, spacing: 4) {
                ForEach(warnings.prefix(5), id: \.self) { warning in
                    HStack(alignment: .top, spacing: 6) {
                        Text("•")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(accentAmber.opacity(0.7))
                        Text(warning)
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.7))
                            .lineSpacing(2)
                    }
                }
            }
        }
    }

    // MARK: - What to Watch

    @ViewBuilder
    private func whatToWatchSection(_ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "eye.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(accentGreen)
                Text("WHAT TO WATCH")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(accentGreen)
            }

            VStack(alignment: .leading, spacing: 4) {
                ForEach(items.prefix(5), id: \.self) { item in
                    HStack(alignment: .top, spacing: 6) {
                        Text("→")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(accentGreen.opacity(0.7))
                        Text(item)
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.7))
                            .lineSpacing(2)
                    }
                }
            }
        }
    }

    // MARK: - Conviction Badge

    @ViewBuilder
    private func convictionBadge(_ conviction: String) -> some View {
        let color: Color = {
            switch conviction.lowercased() {
            case "high": return accentGreen
            case "medium": return accentAmber
            case "low": return accentRed
            default: return .white.opacity(0.5)
            }
        }()

        Text(conviction.uppercased())
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .clipShape(Capsule())
    }
}

// MARK: - Agentic Plan Detail Sheet

struct AgenticPlanDetailSheet: View {
    @Environment(\.dismiss) private var dismiss

    let symbol: String
    let tradeStyle: String
    let plan: [String: Any]
    let riskWarnings: [String]
    let whatToWatch: [String]

    private let terminalBg = Color(red: 0.06, green: 0.07, blue: 0.09)
    private let cardBg = Color(red: 0.10, green: 0.11, blue: 0.13)
    private let accentGreen = Color(red: 0.0, green: 0.85, blue: 0.45)
    private let accentAmber = Color(red: 1.0, green: 0.76, blue: 0.03)
    private let accentRed = Color(red: 1.0, green: 0.4, blue: 0.4)
    private let accentCyan = Color(red: 0.0, green: 0.87, blue: 0.87)

    private var conviction: String { plan["conviction"] as? String ?? "low" }
    private var suitable: Bool { plan["suitable"] as? Bool ?? false }
    private var bias: String { plan["bias"] as? String ?? "neutral" }

    private var convictionColor: Color {
        switch conviction.lowercased() {
        case "high": return accentGreen
        case "medium": return accentAmber
        case "low": return accentRed
        default: return .white.opacity(0.5)
        }
    }

    private var biasColor: Color {
        switch bias.lowercased() {
        case "bullish": return accentGreen
        case "bearish": return accentRed
        default: return accentAmber
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header Card
                    headerCard

                    // Price Levels Card
                    priceLevelsCard

                    // Thesis Card
                    thesisCard

                    // Risk Warnings
                    if !riskWarnings.isEmpty {
                        riskWarningsCard
                    }

                    // What to Watch
                    if !whatToWatch.isEmpty {
                        whatToWatchCard
                    }

                    Spacer(minLength: 40)
                }
                .padding(20)
            }
            .background(terminalBg)
            .navigationTitle("\(symbol) \(tradeStyle.capitalized) Trade")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(accentCyan)
                }
            }
            .toolbarBackground(terminalBg, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
        }
        .presentationDetents([.large])
        .presentationDragIndicator(.visible)
    }

    // MARK: - Header Card

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Symbol and Style
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(symbol)
                        .font(.system(size: 28, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)

                    Text("\(tradeStyle.uppercased()) TRADE PLAN")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(accentCyan)
                }

                Spacer()

                // Suitable badge
                if suitable {
                    VStack(spacing: 4) {
                        Image(systemName: "checkmark.seal.fill")
                            .font(.system(size: 24))
                            .foregroundStyle(accentGreen)
                        Text("SUITABLE")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundStyle(accentGreen)
                    }
                } else {
                    VStack(spacing: 4) {
                        Image(systemName: "xmark.seal.fill")
                            .font(.system(size: 24))
                            .foregroundStyle(accentRed.opacity(0.7))
                        Text("NOT IDEAL")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundStyle(accentRed.opacity(0.7))
                    }
                }
            }

            Divider().background(Color.white.opacity(0.1))

            // Conviction & Bias Row
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("CONVICTION")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                    Text(conviction.uppercased())
                        .font(.system(size: 16, weight: .bold, design: .monospaced))
                        .foregroundStyle(convictionColor)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("BIAS")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                    Text(bias.uppercased())
                        .font(.system(size: 16, weight: .bold, design: .monospaced))
                        .foregroundStyle(biasColor)
                }

                if let holdingPeriod = plan["holding_period"] as? String {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("HOLD TIME")
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.5))
                        Text(holdingPeriod)
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white)
                    }
                }

                Spacer()
            }

            // Conviction Reasoning
            if let reasoning = plan["conviction_reasoning"] as? String {
                Text(reasoning)
                    .font(.system(size: 11, weight: .regular, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
                    .lineSpacing(3)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(cardBg)
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(convictionColor.opacity(0.3), lineWidth: 1)
                )
        )
    }

    // MARK: - Price Levels Card

    private var priceLevelsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 8) {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(accentCyan)
                Text("PRICE LEVELS")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(accentCyan)
            }

            // Entry Zone
            if let entryZone = plan["entry_zone"] as? [Double], entryZone.count >= 2 {
                priceLevelRow(
                    label: "ENTRY ZONE",
                    value: String(format: "$%.2f - $%.2f", entryZone[0], entryZone[1]),
                    icon: "arrow.right.circle.fill",
                    color: accentCyan
                )
            }

            // Stop Loss
            if let stopLoss = plan["stop_loss"] as? Double {
                priceLevelRow(
                    label: "STOP LOSS",
                    value: String(format: "$%.2f", stopLoss),
                    icon: "xmark.octagon.fill",
                    color: accentRed
                )
            }

            // Targets
            if let targets = plan["targets"] as? [Double], !targets.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: "target")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(accentGreen)
                        Text("TARGETS")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.5))
                    }

                    HStack(spacing: 12) {
                        ForEach(Array(targets.enumerated()), id: \.offset) { index, target in
                            VStack(spacing: 4) {
                                Text("T\(index + 1)")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(accentGreen.opacity(0.7))
                                Text(String(format: "$%.2f", target))
                                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                                    .foregroundStyle(accentGreen)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(accentGreen.opacity(0.1))
                            )

                            if index < targets.count - 1 {
                                Image(systemName: "chevron.right")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.white.opacity(0.3))
                            }
                        }
                    }
                }
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(cardBg)
        )
    }

    private func priceLevelRow(label: String, value: String, icon: String, color: Color) -> some View {
        HStack {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(color)
                Text(label)
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
            Spacer()
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .monospaced))
                .foregroundStyle(color)
        }
    }

    // MARK: - Thesis Card

    private var thesisCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "text.quote")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(accentAmber)
                Text("THESIS")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(accentAmber)
            }

            if let thesis = plan["thesis"] as? String {
                Text(thesis)
                    .font(.system(size: 12, weight: .regular, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.85))
                    .lineSpacing(4)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(cardBg)
        )
    }

    // MARK: - Risk Warnings Card

    private var riskWarningsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(accentAmber)
                Text("RISK WARNINGS")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(accentAmber)
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(riskWarnings, id: \.self) { warning in
                    HStack(alignment: .top, spacing: 8) {
                        Text("•")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(accentAmber.opacity(0.7))
                        Text(warning)
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.7))
                            .lineSpacing(2)
                    }
                }
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(cardBg)
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(accentAmber.opacity(0.2), lineWidth: 1)
                )
        )
    }

    // MARK: - What to Watch Card

    private var whatToWatchCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "eye.fill")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(accentGreen)
                Text("WHAT TO WATCH")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(accentGreen)
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(whatToWatch, id: \.self) { item in
                    HStack(alignment: .top, spacing: 8) {
                        Text("→")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(accentGreen.opacity(0.7))
                        Text(item)
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.7))
                            .lineSpacing(2)
                    }
                }
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(cardBg)
        )
    }
}

// MARK: - Agentic View Preview

#Preview("Agentic Streaming") {
    AgenticStreamingView(
        symbol: "NVDA",
        streamItems: .constant(PreviewData.agenticStreamItems),
        isComplete: .constant(true),
        finalPlan: .constant(PreviewData.agenticFinalPlan),
        expandedToolResults: .constant([])
    )
}

#Preview("Agentic In Progress") {
    AgenticStreamingView(
        symbol: "AAPL",
        streamItems: .constant(PreviewData.agenticInProgressItems),
        isComplete: .constant(false),
        finalPlan: .constant(nil),
        expandedToolResults: .constant([])
    )
}

// MARK: - Preview Data

private enum PreviewData {
    static let agenticStreamItems: [AgenticStreamItem] = [
        .thinking(AgentThinking(
            text: "Let me start by checking the current price and market context to understand where NVDA is trading today.",
            iteration: 1,
            timestamp: Date()
        )),
        .toolCall(ToolCall(
            name: "get_price",
            arguments: ["symbol": "NVDA"],
            iteration: 1,
            timestamp: Date()
        )),
        .toolResult(ToolResult(
            toolName: "get_price",
            result: ["price": 148.50, "change_pct": 2.3, "bid": 148.45, "ask": 148.55],
            iteration: 1,
            timestamp: Date()
        )),
        .thinking(AgentThinking(
            text: "Price is $148.50, up 2.3% today. Good momentum. Let me analyze the daily chart to understand the current pattern and trend structure.",
            iteration: 2,
            timestamp: Date()
        )),
        .toolCall(ToolCall(
            name: "get_and_analyze_chart",
            arguments: ["symbol": "NVDA", "timeframe": "1d", "trade_style": "swing"],
            iteration: 2,
            timestamp: Date()
        )),
        .toolResult(ToolResult(
            toolName: "get_and_analyze_chart",
            result: ["trend_quality": "strong_uptrend", "pattern": "bull_flag", "ema_alignment": "bullish", "key_observation": "Clean consolidation after breakout"],
            iteration: 2,
            timestamp: Date()
        )),
        .thinking(AgentThinking(
            text: "Beautiful bull flag pattern on the daily! EMAs are stacked bullish and volume is decreasing in the consolidation - classic accumulation. Let me check fundamentals to confirm the setup has strong backing.",
            iteration: 3,
            timestamp: Date()
        )),
        .toolCall(ToolCall(
            name: "get_fundamentals",
            arguments: ["symbol": "NVDA"],
            iteration: 3,
            timestamp: Date()
        )),
        .toolResult(ToolResult(
            toolName: "get_fundamentals",
            result: ["pe_ratio": 45.2, "eps_growth": 30.5, "revenue_growth": 25.3, "profit_margin": 55.2],
            iteration: 3,
            timestamp: Date()
        )),
        .thinking(AgentThinking(
            text: "Strong fundamentals with 30% EPS growth and healthy margins. Now let me check news sentiment and X/Twitter for real-time trader sentiment.",
            iteration: 4,
            timestamp: Date()
        )),
        .toolCall(ToolCall(
            name: "get_news",
            arguments: ["symbol": "NVDA", "days_back": 7],
            iteration: 4,
            timestamp: Date()
        )),
        .toolResult(ToolResult(
            toolName: "get_news",
            result: ["sentiment": "bullish", "headline_count": 12, "key_theme": "AI chip demand surging"],
            iteration: 4,
            timestamp: Date()
        )),
        .toolCall(ToolCall(
            name: "search_x",
            arguments: ["query": "$NVDA sentiment traders"],
            iteration: 5,
            timestamp: Date()
        )),
        .toolResult(ToolResult(
            toolName: "search_x",
            result: ["sentiment_analysis": "bullish", "volume": "high", "key_themes": ["breakout", "accumulation", "AI demand"]],
            iteration: 5,
            timestamp: Date()
        ))
    ]

    static let agenticFinalPlan: [String: Any] = [
        "symbol": "NVDA",
        "recommended_style": "swing",
        "recommendation_reasoning": "Clean bull flag at 61.8% Fibonacci retracement with decreasing volume in consolidation. Strong fundamentals (30% EPS growth, 55% margins) and bullish sentiment across news and X/Twitter support the setup. Daily chart shows textbook accumulation pattern.",
        "day_trade_plan": [
            "conviction": "low",
            "conviction_reasoning": "ATR of 1.8% provides limited intraday range for day trading. Better opportunities in swing timeframe.",
            "suitable": false,
            "bias": "neutral",
            "thesis": "Insufficient volatility for day trading. Wait for earnings or catalyst for increased range. Current consolidation pattern suggests accumulation but intraday swings are too narrow for reliable day trade setups.",
            "entry_zone": [148.20, 148.80],
            "stop_loss": 147.50,
            "targets": [149.50, 150.20],
            "holding_period": "1-3 hours"
        ] as [String: Any],
        "swing_trade_plan": [
            "conviction": "high",
            "conviction_reasoning": "Perfect bull flag setup with all signals aligned - technical pattern, fundamentals, and sentiment all bullish. Volume declining in consolidation shows healthy accumulation. 61.8% Fib holding as support.",
            "suitable": true,
            "bias": "bullish",
            "thesis": "Enter on breakout above $150 flag resistance with stop at $144.50 below flag low. The bull flag measured move projects to $175, with intermediate resistance at $162 (1.618 Fib extension). Strong fundamentals with 30% EPS growth and 55% margins support higher prices. News catalyst potential with AI chip demand narrative.",
            "entry_zone": [147.00, 148.50],
            "stop_loss": 144.50,
            "targets": [155.00, 162.00, 175.00],
            "holding_period": "5-10 days"
        ] as [String: Any],
        "position_trade_plan": [
            "conviction": "medium",
            "conviction_reasoning": "Weekly uptrend intact but approaching 52-week high resistance at $175. Better entry on pullback to 50 EMA around $140. Long-term AI thesis remains compelling.",
            "suitable": true,
            "bias": "bullish",
            "thesis": "NVDA remains the premier AI chip play with unmatched market position. Weekly chart shows healthy uptrend with higher lows. Consider scaling in on pullbacks to key moving averages. Risk is elevated P/E at 45x but growth justifies premium. Position for multi-quarter AI infrastructure buildout.",
            "entry_zone": [140.00, 145.00],
            "stop_loss": 132.00,
            "targets": [175.00, 200.00, 225.00],
            "holding_period": "1-3 months"
        ] as [String: Any],
        "risk_warnings": ["Approaching 52-week high resistance at $175", "Elevated P/E ratio of 45x limits upside", "Chip sector rotation risk on macro shifts", "Earnings in 3 weeks - high volatility expected"],
        "what_to_watch": ["Break above $150 confirms bull flag breakout", "Volume expansion on breakout validates move", "NVDA earnings in 3 weeks - key catalyst", "Watch SMH (semiconductor ETF) for sector health"]
    ]

    static let agenticInProgressItems: [AgenticStreamItem] = [
        .thinking(AgentThinking(
            text: "Let me start by checking the current price and market context for AAPL.",
            iteration: 1,
            timestamp: Date()
        )),
        .toolCall(ToolCall(
            name: "get_price",
            arguments: ["symbol": "AAPL"],
            iteration: 1,
            timestamp: Date()
        )),
        .toolResult(ToolResult(
            toolName: "get_price",
            result: ["price": 178.25, "change_pct": -0.5],
            iteration: 1,
            timestamp: Date()
        )),
        .thinking(AgentThinking(
            text: "AAPL is at $178.25, down slightly today. Let me check the chart patterns...",
            iteration: 2,
            timestamp: Date()
        )),
        .toolCall(ToolCall(
            name: "get_and_analyze_chart",
            arguments: ["symbol": "AAPL", "timeframe": "1d", "trade_style": "swing"],
            iteration: 2,
            timestamp: Date()
        ))
    ]
}

// MARK: - Final Recommendation Card Preview

#Preview("Final Recommendation Card") {
    ScrollView {
        FinalRecommendationCard(plan: PreviewData.agenticFinalPlan, symbol: "NVDA")
            .padding()
    }
    .background(Color(red: 0.06, green: 0.07, blue: 0.09))
}

#Preview("Plan Detail Sheet") {
    AgenticPlanDetailSheet(
        symbol: "NVDA",
        tradeStyle: "swing",
        plan: PreviewData.agenticFinalPlan["swing_trade_plan"] as! [String: Any],
        riskWarnings: PreviewData.agenticFinalPlan["risk_warnings"] as! [String],
        whatToWatch: PreviewData.agenticFinalPlan["what_to_watch"] as! [String]
    )
}
