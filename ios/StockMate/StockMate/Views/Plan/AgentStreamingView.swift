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

    /// Determines if a pending step should be shown (shows next few pending steps)
    private func shouldShowPendingStep(_ stepType: SubAgentStepType, activeType: SubAgentStepType?) -> Bool {
        guard let active = activeType else {
            // If no active step, show first 3 pending steps
            return stepType.order < 3
        }
        // Show pending steps that come after the active one (up to 2 ahead)
        return stepType.order > active.order && stepType.order <= active.order + 2
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
