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
    @Published private(set) var error: String?

    /// Orchestrator-level progress (shown before sub-agents)
    @Published private(set) var orchestratorSteps: [OrchestratorStep] = []

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
        activeSymbol?.uppercased() == symbol.uppercased() && generatedPlan != nil && !isGenerating
    }

    /// Start plan generation for a symbol
    func startGeneration(for symbol: String) {
        // Cancel any existing generation
        cancelGeneration()

        activeSymbol = symbol.uppercased()
        isGenerating = true
        error = nil
        generatedPlan = nil
        subagentProgress = [:]
        orchestratorSteps = []

        // Initialize sub-agents with pending state
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
        subagentProgress = [:]
        orchestratorSteps = []
        error = nil
    }

    /// Clear completed generation data
    func clearCompletedGeneration() {
        guard !isGenerating else { return }
        activeSymbol = nil
        generatedPlan = nil
        subagentProgress = [:]
        orchestratorSteps = []
        error = nil
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

    /// Get sorted sub-agents
    var sortedSubagents: [SubAgentProgress] {
        let order = ["day-trade-analyzer", "swing-trade-analyzer", "position-trade-analyzer"]
        return order.compactMap { subagentProgress[$0] }
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
        case "orchestrator_step":
            // V2: Orchestrator-level progress (gathering common data, spawning subagents, etc.)
            handleOrchestratorStep(
                stepType: event.stepType,
                status: event.stepStatus,
                findings: event.stepFindings ?? []
            )

        case "subagent_progress":
            if let subagentsData = event.subagents {
                handleSubagentProgress(subagentsData)
            }

        case "subagent_complete":
            if let agentName = event.agentName {
                handleSubagentComplete(agentName, findings: event.agentFindings ?? [])
            }

        case "plan_complete", "plan", "final_result":
            if let newPlan = event.plan {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    self.generatedPlan = newPlan
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
