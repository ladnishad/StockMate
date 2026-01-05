import Foundation
import SwiftUI
import Combine

/// Singleton manager for plan generation tasks.
/// Keeps generation running in background when user navigates away.
@MainActor
final class PlanGenerationManager: ObservableObject {

    static let shared = PlanGenerationManager()

    // MARK: - Published State

    /// Currently active generation (nil if none)
    @Published private(set) var activeSymbol: String?

    /// Generation progress state
    @Published private(set) var isGenerating: Bool = false
    @Published private(set) var subagentProgress: [String: SubAgentProgress] = [:]
    @Published var expandedSubagents: Set<String> = []
    @Published private(set) var generatedPlan: TradingPlanResponse?
    @Published private(set) var analysisId: String?  // V2: ID for approving the analysis
    @Published private(set) var error: String?

    /// Orchestrator-level progress (shown before sub-agents)
    @Published private(set) var orchestratorSteps: [OrchestratorStep] = []

    // MARK: - Agentic Mode State

    /// Whether we're in agentic mode (AI-driven tool-use loop) vs legacy V2 mode
    @Published private(set) var isAgenticMode: Bool = true

    /// Stream of agentic events (thinking, tool calls, results)
    @Published private(set) var agenticStreamItems: [AgenticStreamItem] = []

    /// Final plan from agentic mode (raw dictionary before conversion)
    @Published private(set) var agenticFinalPlan: [String: Any]?

    /// Expanded tool results in agentic view
    @Published var expandedToolResults: Set<UUID> = []

    /// Whether the "Starting Analyzers" section is expanded to show sub-agents
    @Published var isAnalyzersSectionExpanded: Bool = true

    /// For reconnecting views
    @Published private(set) var lastUpdated: Date?

    // MARK: - Private State

    private var generationTask: Task<Void, Never>?

    private init() {}

    // MARK: - Public Methods

    /// Check if there's an active generation for a symbol
    func hasActiveGeneration(for symbol: String) -> Bool {
        activeSymbol?.uppercased() == symbol.uppercased() && isGenerating
    }

    /// Check if generation completed for a symbol (not still generating)
    func hasCompletedPlan(for symbol: String) -> Bool {
        guard activeSymbol?.uppercased() == symbol.uppercased() && !isGenerating else {
            return false
        }
        // Check either V2 plan or agentic plan
        return generatedPlan != nil || agenticFinalPlan != nil
    }

    /// Check if agentic mode completed (has agentic plan, not V2 plan)
    func hasCompletedAgenticPlan(for symbol: String) -> Bool {
        activeSymbol?.uppercased() == symbol.uppercased() && agenticFinalPlan != nil && !isGenerating
    }

    /// Start plan generation for a symbol
    func startGeneration(for symbol: String) {
        // Cancel any existing generation
        cancelGeneration()

        activeSymbol = symbol.uppercased()
        isGenerating = true
        error = nil
        generatedPlan = nil
        analysisId = nil
        subagentProgress = [:]
        orchestratorSteps = []
        isAnalyzersSectionExpanded = true

        // Reset agentic mode state
        isAgenticMode = true  // Assume agentic until we see V2 events
        agenticStreamItems = []
        agenticFinalPlan = nil
        expandedToolResults = []

        // Initialize sub-agents with pending state (for V2 fallback)
        initializeSubagents()

        // Start the generation task
        generationTask = Task {
            await runGeneration(symbol: symbol)
        }
    }

    /// Cancel current generation and clear state
    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        // Clear partial state to avoid stale data
        activeSymbol = nil
        generatedPlan = nil
        analysisId = nil
        subagentProgress = [:]
        orchestratorSteps = []
        error = nil
        // Clear agentic state
        agenticStreamItems = []
        agenticFinalPlan = nil
        expandedToolResults = []
    }

    /// Clear completed generation data
    func clearCompletedGeneration() {
        guard !isGenerating else { return }
        activeSymbol = nil
        generatedPlan = nil
        analysisId = nil
        subagentProgress = [:]
        orchestratorSteps = []
        error = nil
        // Clear agentic state
        agenticStreamItems = []
        agenticFinalPlan = nil
        expandedToolResults = []
    }

    /// Approve the analysis and create a trading plan from it
    /// Returns the approved plan on success, nil on failure
    func approveAnalysis() async -> TradingPlanResponse? {
        guard let symbol = activeSymbol,
              let analysisId = analysisId else {
            error = "No analysis to approve"
            return nil
        }

        do {
            let approvedPlan = try await APIService.shared.approveV2Analysis(
                symbol: symbol,
                analysisId: analysisId
            )
            withAnimation {
                self.generatedPlan = approvedPlan
            }
            return approvedPlan
        } catch {
            self.error = "Failed to approve plan: \(error.localizedDescription)"
            return nil
        }
    }

    /// Toggle sub-agent expansion
    func toggleSubagentExpansion(_ agentName: String) {
        withAnimation(.easeInOut(duration: 0.2)) {
            if expandedSubagents.contains(agentName) {
                expandedSubagents.remove(agentName)
            } else {
                expandedSubagents.insert(agentName)
            }
        }
    }

    /// Toggle the "Starting Analyzers" section expansion
    func toggleAnalyzersSection() {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            isAnalyzersSectionExpanded.toggle()
        }
    }

    /// Get sorted sub-agents
    var sortedSubagents: [SubAgentProgress] {
        let order = ["day-trade-analyzer", "swing-trade-analyzer", "position-trade-analyzer"]
        return order.compactMap { subagentProgress[$0] }
    }

    /// Check if all sub-agents have completed
    var allSubagentsComplete: Bool {
        let sorted = sortedSubagents
        guard !sorted.isEmpty else { return false }
        return sorted.allSatisfy { $0.status == .completed || $0.status == .failed }
    }

    /// Count of completed sub-agents
    var completedSubagentsCount: Int {
        sortedSubagents.filter { $0.status == .completed }.count
    }

    // MARK: - Private Methods

    private func initializeSubagents() {
        let agents = ["day-trade-analyzer", "swing-trade-analyzer", "position-trade-analyzer"]
        let displayNames = [
            "day-trade-analyzer": "Day Trade",
            "swing-trade-analyzer": "Swing Trade",
            "position-trade-analyzer": "Position Trade"
        ]

        withAnimation {
            subagentProgress = [:]
            for agentName in agents {
                subagentProgress[agentName] = SubAgentProgress(
                    agentName: agentName,
                    displayName: displayNames[agentName] ?? agentName,
                    status: .pending,
                    currentStep: nil,
                    stepsCompleted: [],
                    findings: [],
                    elapsedMs: 0,
                    errorMessage: nil
                )
            }
            expandedSubagents = Set(agents)
        }
    }

    private func runGeneration(symbol: String) async {
        do {
            let stream = APIService.shared.generateTradingPlanStream(
                symbol: symbol,
                forceNew: false
            )

            for try await event in stream {
                // Check for cancellation
                try Task.checkCancellation()

                handleStreamEvent(event)
            }

            // Stream completed but no plan received
            if generatedPlan == nil {
                // Try to fetch existing plan
                if let existingPlan = try? await APIService.shared.getTradingPlan(symbol: symbol) {
                    withAnimation {
                        self.generatedPlan = existingPlan
                        self.lastUpdated = Date()
                    }
                }
            }

        } catch is CancellationError {
            // Generation was cancelled - do nothing
        } catch {
            withAnimation {
                self.error = error.localizedDescription
            }
        }

        withAnimation {
            isGenerating = false
            lastUpdated = Date()
        }
    }

    private func handleStreamEvent(_ event: APIService.PlanStreamEvent) {
        switch event.type {
        // MARK: - Agentic Mode Events
        case "agent_thinking":
            // AI is reasoning about what to do next
            if let thinking = event.thinking {
                let item = AgentThinking(
                    text: thinking,
                    iteration: event.iteration ?? 0,
                    timestamp: Date()
                )
                withAnimation(.easeInOut(duration: 0.2)) {
                    agenticStreamItems.append(.thinking(item))
                }
            }

        case "tool_call":
            // AI is calling a tool
            if let toolName = event.toolName {
                let arguments = event.toolArguments?.mapValues { $0.value } ?? [:]
                let item = ToolCall(
                    name: toolName,
                    arguments: arguments,
                    iteration: event.iteration ?? 0,
                    timestamp: Date()
                )
                withAnimation(.easeInOut(duration: 0.2)) {
                    agenticStreamItems.append(.toolCall(item))
                }
            }

        case "tool_result":
            // Tool returned a result
            if let toolName = event.toolName,
               let result = event.toolResult {
                let resultDict = result.mapValues { $0.value }
                let item = ToolResult(
                    toolName: toolName,
                    result: resultDict,
                    iteration: event.iteration ?? 0,
                    timestamp: Date()
                )
                withAnimation(.easeInOut(duration: 0.2)) {
                    agenticStreamItems.append(.toolResult(item))
                }
            }

        // MARK: - V2/Legacy Mode Events
        case "orchestrator_step":
            // V2: Orchestrator-level progress - switch to V2 mode
            isAgenticMode = false
            handleOrchestratorStep(
                stepType: event.stepType,
                status: event.stepStatus,
                findings: event.stepFindings ?? []
            )

        case "subagent_progress":
            isAgenticMode = false
            if let subagentsData = event.subagents {
                handleSubagentProgress(subagentsData)
            }

        case "subagent_complete":
            isAgenticMode = false
            if let agentName = event.agentName {
                handleSubagentComplete(agentName, findings: event.agentFindings ?? [])
            }

        case "plan_complete", "plan", "final_result":
            // Handle agentic mode vs V2 mode differently
            if isAgenticMode, let rawPlan = event.agenticPlan {
                // Agentic mode: use raw agentic_plan dictionary
                let planDict = rawPlan.mapValues { $0.value }
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    self.agenticFinalPlan = planDict
                    self.analysisId = event.analysisId
                    self.lastUpdated = Date()
                    // Don't set generatedPlan for agentic - it has different structure
                }
            } else if let newPlan = event.plan {
                // V2/Legacy mode: use TradingPlanResponse
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    self.generatedPlan = newPlan
                    self.analysisId = event.analysisId
                    self.lastUpdated = Date()
                    self.markAllSubagentsComplete()
                }
            }

        case "existing_plan":
            if let existingPlan = event.plan {
                withAnimation {
                    self.generatedPlan = existingPlan
                    self.lastUpdated = Date()
                }
            }

        case "error":
            withAnimation {
                self.error = event.effectiveErrorMessage ?? "Unknown error"
            }

        case "step":
            // V1 step events - just log for now (V1 mode uses ViewModel directly)
            print("[PlanGenerationManager] V1 step event: \(event.stepType ?? "unknown")")

        default:
            break
        }
    }

    private func handleOrchestratorStep(stepType: String?, status: String?, findings: [String]) {
        guard let stepType = stepType else { return }

        withAnimation(.easeInOut(duration: 0.15)) {
            // Find existing step or create new one
            if let existingIndex = orchestratorSteps.firstIndex(where: { $0.stepType == stepType }) {
                // Update existing step
                orchestratorSteps[existingIndex].status = status == "completed" ? .completed : .active
                orchestratorSteps[existingIndex].findings = findings
            } else {
                // Add new step
                let step = OrchestratorStep(
                    stepType: stepType,
                    status: status == "completed" ? .completed : .active,
                    findings: findings
                )
                orchestratorSteps.append(step)
            }
        }
    }

    private func handleSubagentProgress(_ data: [String: APIService.SubAgentProgressData]) {
        withAnimation(.easeInOut(duration: 0.15)) {
            for (agentName, progressData) in data {
                let progress = SubAgentProgress(
                    agentName: progressData.agentName,
                    displayName: progressData.displayName,
                    status: SubAgentStatus(rawValue: progressData.status) ?? .pending,
                    currentStep: progressData.currentStep,
                    stepsCompleted: progressData.stepsCompleted,
                    findings: progressData.findings,
                    elapsedMs: progressData.elapsedMs,
                    errorMessage: progressData.errorMessage
                )
                self.subagentProgress[agentName] = progress
            }
        }
    }

    private func handleSubagentComplete(_ agentName: String, findings: [String]) {
        withAnimation(.easeInOut(duration: 0.2)) {
            if var progress = self.subagentProgress[agentName] {
                progress.status = .completed
                progress.findings = findings
                self.subagentProgress[agentName] = progress
            }
        }
    }

    private func markAllSubagentsComplete() {
        for agentName in subagentProgress.keys {
            if var progress = subagentProgress[agentName] {
                if progress.status != .completed && progress.status != .failed {
                    progress.status = .completed
                    subagentProgress[agentName] = progress
                }
            }
        }
    }

    /// Convert TradingPlanResponse to dictionary for agentic view
    private func convertPlanToDictionary(_ plan: TradingPlanResponse) -> [String: Any] {
        let selectedStyle = plan.tradeStyle ?? "swing"

        var dict: [String: Any] = [
            "symbol": plan.symbol,
            "recommended_style": selectedStyle,
            "recommendation_reasoning": plan.tradeStyleReasoning ?? plan.thesis
        ]

        // Build the selected plan
        let selectedPlan: [String: Any] = [
            "conviction": convictionFromInt(plan.confidence),
            "conviction_reasoning": plan.tradeStyleReasoning ?? "",
            "suitable": true,
            "bias": plan.bias,
            "thesis": plan.thesis
        ]
        dict["\(selectedStyle)_trade_plan"] = selectedPlan

        // Build alternative plans
        for alt in plan.alternatives {
            let altPlan: [String: Any] = [
                "conviction": convictionFromInt(alt.confidence),
                "conviction_reasoning": alt.whyNotSelected,
                "suitable": alt.suitable,
                "bias": alt.bias,
                "thesis": alt.briefThesis
            ]
            dict["\(alt.tradeStyle)_trade_plan"] = altPlan
        }

        // Add risk warnings and what to watch
        dict["risk_warnings"] = plan.riskWarnings
        dict["what_to_watch"] = plan.whatToWatch

        return dict
    }

    /// Convert integer confidence (0-100) to string conviction level
    private func convictionFromInt(_ confidence: Int?) -> String {
        guard let conf = confidence else { return "medium" }
        if conf >= 70 { return "high" }
        if conf >= 40 { return "medium" }
        return "low"
    }
}

// MARK: - Orchestrator Step Model

/// Represents a single orchestrator-level step in the plan generation flow
struct OrchestratorStep: Identifiable, Equatable {
    let id = UUID()
    let stepType: String
    var status: Status
    var findings: [String]

    enum Status: Equatable {
        case active
        case completed
    }

    var displayName: String {
        switch stepType {
        case "gathering_common_data": return "Gathering Market Data"
        case "spawning_subagents": return "Starting Analyzers"
        case "selecting_best": return "Selecting Best Plan"
        case "complete": return "Complete"
        default: return stepType.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    var icon: String {
        switch stepType {
        case "gathering_common_data": return "antenna.radiowaves.left.and.right"
        case "spawning_subagents": return "arrow.triangle.branch"
        case "selecting_best": return "star.fill"
        case "complete": return "checkmark.seal.fill"
        default: return "gearshape"
        }
    }
}
