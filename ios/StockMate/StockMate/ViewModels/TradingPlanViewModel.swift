import Foundation
import SwiftUI
import Combine

/// ViewModel for the trading plan view with live update support
@MainActor
final class TradingPlanViewModel: ObservableObject {
    // MARK: - Published State

    @Published private(set) var plan: TradingPlanResponse?
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var isUpdating: Bool = false
    @Published private(set) var error: String?
    @Published private(set) var lastUpdated: Date?

    // Educational content (fetched on-demand)
    @Published private(set) var educationalContent: EducationalContent?
    @Published private(set) var smartAnalysis: SmartAnalysisResponse?
    @Published private(set) var educationalBars: [PriceBar] = []
    @Published private(set) var isLoadingEducational: Bool = false
    @Published private(set) var educationalError: String?

    // Streaming support
    @Published private(set) var streamingText: String = ""
    @Published private(set) var isStreaming: Bool = false

    // Agent-style streaming steps (Claude Code style)
    @Published var analysisSteps: [AnalysisStep] = []
    @Published var isAnalysisComplete: Bool = false

    // For live update animation
    @Published private(set) var updatePhase: UpdatePhase = .idle

    // Track recent adjustments from evaluation (persists until next evaluation/new plan)
    @Published private(set) var lastEvaluation: EvaluationResponse?

    enum UpdatePhase: Equatable {
        case idle
        case gatheringData
        case analyzing
        case generatingPlan
        case complete
    }

    // MARK: - Properties

    let symbol: String
    private var pollTimer: Timer?
    private var updateTask: Task<Void, Never>?
    private var streamingTask: Task<Void, Never>?
    private var evaluationObserver: NSObjectProtocol?

    /// Whether the plan has been re-evaluated based on user's position
    @Published private(set) var hasActivePosition: Bool = false

    // MARK: - Computed Properties

    var hasPlan: Bool { plan != nil }

    /// Check if any adjustments were made in the last evaluation
    var hasRecentAdjustments: Bool {
        guard let eval = lastEvaluation else { return false }
        return !eval.adjustmentsMade.isEmpty
    }

    /// Check if a specific field was adjusted in the last evaluation
    func wasAdjusted(_ field: String) -> Bool {
        lastEvaluation?.wasAdjusted(field) ?? false
    }

    /// Get the previous value for a field if it was adjusted
    func previousValue(for field: String) -> Double? {
        lastEvaluation?.previousValue(for: field)
    }

    var planStatusColor: Color {
        guard let status = plan?.status.lowercased() else { return .gray }
        switch status {
        case "active": return .blue
        case "invalidated": return .orange
        case "completed": return .green
        case "stopped_out": return .red
        default: return .gray
        }
    }

    var biasColor: Color {
        guard let bias = plan?.bias.lowercased() else { return .gray }
        switch bias {
        case "bullish": return .green
        case "bearish": return .red
        default: return .gray
        }
    }

    // MARK: - Initialization

    init(symbol: String) {
        self.symbol = symbol.uppercased()

        // Listen for evaluation triggers (from position entry/exit)
        evaluationObserver = NotificationCenter.default.addObserver(
            forName: .planEvaluationTriggered,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self,
                  let userInfo = notification.userInfo,
                  let triggerSymbol = userInfo["symbol"] as? String,
                  triggerSymbol.uppercased() == self.symbol else { return }

            // Mark that plan is now position-based and trigger re-evaluation
            Task { @MainActor in
                self.hasActivePosition = true

                // Actually trigger a re-evaluation of the plan
                await self.evaluatePlan()
            }
        }
    }

    deinit {
        pollTimer?.invalidate()
        streamingTask?.cancel()
        if let observer = evaluationObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    // MARK: - Public Methods

    func loadPlan() async {
        guard !isLoading else { return }

        isLoading = true
        error = nil

        do {
            plan = try await APIService.shared.getTradingPlan(symbol: symbol)
            lastUpdated = Date()
        } catch {
            // Plan might not exist yet - that's okay
            if let apiError = error as? APIServiceError {
                switch apiError {
                case .httpError(404):
                    // No plan exists yet
                    self.plan = nil
                default:
                    self.error = apiError.localizedDescription
                }
            }
        }

        isLoading = false
    }

    func generateNewPlan() async {
        guard !isUpdating else { return }

        isUpdating = true
        error = nil
        lastEvaluation = nil  // Clear previous adjustments when generating new plan

        // Start animation in parallel (don't block the API call)
        let animationTask = Task {
            await animateUpdatePhases()
        }

        do {
            // API call runs immediately, in parallel with animation
            let newPlan = try await APIService.shared.generateTradingPlan(
                symbol: symbol,
                forceNew: true
            )

            // Cancel animation if still running and update with the plan
            animationTask.cancel()

            // Update plan with animation for smooth transition
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                self.plan = newPlan
                self.lastUpdated = Date()
                self.updatePhase = .complete
            }

            // Reset to idle after showing completion
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            withAnimation {
                self.updatePhase = .idle
                self.isUpdating = false
            }

        } catch {
            animationTask.cancel()
            withAnimation {
                self.error = error.localizedDescription
                self.updatePhase = .idle
                self.isUpdating = false
            }
        }
    }

    func evaluatePlan() async {
        guard !isUpdating, plan != nil else { return }

        isUpdating = true
        updatePhase = .analyzing
        lastEvaluation = nil  // Clear previous evaluation before new one

        do {
            // Evaluate and capture the response with previous values
            let evalResponse = try await APIService.shared.evaluateTradingPlan(symbol: symbol)

            // Store the evaluation response (includes previous_values and adjustments_made)
            withAnimation {
                self.lastEvaluation = evalResponse
            }

            // Refresh the plan to get updated notes and values
            plan = try await APIService.shared.getTradingPlan(symbol: symbol)
            lastUpdated = Date()
            updatePhase = .complete

            try? await Task.sleep(nanoseconds: 1_000_000_000)
            updatePhase = .idle
        } catch {
            self.error = error.localizedDescription
            updatePhase = .idle
        }

        isUpdating = false
    }

    /// Fetch educational content on-demand (from smart analysis)
    func loadEducationalContent() async {
        guard !isLoadingEducational else { return }
        guard educationalContent == nil else { return } // Already loaded

        isLoadingEducational = true
        educationalError = nil

        do {
            // Fetch smart analysis and bars in parallel
            async let analysisTask = APIService.shared.smartAnalyzeStock(symbol: symbol)
            async let barsTask = APIService.shared.fetchBars(symbol: symbol, timeframe: .oneMonth)

            let (analysis, bars) = try await (analysisTask, barsTask)

            withAnimation {
                self.smartAnalysis = analysis
                self.educationalContent = analysis.tradePlan.educational
                self.educationalBars = bars
            }
        } catch {
            print("Failed to load educational content: \(error)")
            withAnimation {
                self.educationalError = error.localizedDescription
            }
        }

        isLoadingEducational = false
    }

    /// Clear educational error (call when user dismisses)
    func clearEducationalError() {
        educationalError = nil
    }

    /// Generate plan with real-time streaming from the AI (Claude Code agent style)
    func generateNewPlanWithStreaming() async {
        guard !isUpdating else { return }

        // Cancel any existing streaming task
        streamingTask?.cancel()
        streamingTask = nil

        isUpdating = true
        isStreaming = true
        error = nil
        streamingText = ""
        updatePhase = .gatheringData
        lastEvaluation = nil  // Clear previous adjustments when generating new plan

        // Initialize analysis steps for agent-style display
        initializeAnalysisSteps()
        isAnalysisComplete = false

        // Store task reference so it can be cancelled
        streamingTask = Task {
            do {
                let stream = APIService.shared.generateTradingPlanStream(
                    symbol: symbol,
                    forceNew: true
                )

                for try await event in stream {
                    // Check for cancellation
                    try Task.checkCancellation()

                    switch event.type {
                    case "step":
                        // Handle new detailed step events
                        await handleStepEvent(event)

                    case "phase":
                        await handlePhaseEvent(event.phase)

                    case "text":
                        if let content = event.content {
                            withAnimation(.linear(duration: 0.05)) {
                                streamingText += content
                            }
                        }

                    case "plan_complete", "plan":
                        if let newPlan = event.plan {
                            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                                self.plan = newPlan
                                self.lastUpdated = Date()
                                self.updatePhase = .complete
                                self.isAnalysisComplete = true
                            }
                        }

                    case "existing_plan":
                        if let existingPlan = event.plan {
                            withAnimation {
                                self.plan = existingPlan
                                self.lastUpdated = Date()
                                self.updatePhase = .complete
                                self.isAnalysisComplete = true
                            }
                        }

                    case "error":
                        throw NSError(
                            domain: "PlanGeneration",
                            code: -1,
                            userInfo: [NSLocalizedDescriptionKey: event.message ?? "Unknown error"]
                        )

                    default:
                        break
                    }
                }

                // Cleanup after stream completes
                try await Task.sleep(nanoseconds: 1_500_000_000)
                withAnimation {
                    updatePhase = .idle
                    isStreaming = false
                    isUpdating = false
                    streamingText = ""
                }

            } catch is CancellationError {
                // Task was cancelled - clean up silently
                withAnimation {
                    updatePhase = .idle
                    isStreaming = false
                    isUpdating = false
                    streamingText = ""
                    analysisSteps = []
                }
            } catch {
                withAnimation {
                    self.error = error.localizedDescription
                    updatePhase = .idle
                    isStreaming = false
                    isUpdating = false
                    streamingText = ""
                    analysisSteps = []  // Clear stale steps on error
                }
            }
        }

        // Wait for task to complete
        await streamingTask?.value
    }

    /// Initialize analysis steps for agent-style streaming display
    private func initializeAnalysisSteps() {
        analysisSteps = [
            AnalysisStep(type: .gatheringData, status: .pending, findings: []),
            AnalysisStep(type: .technicalIndicators, status: .pending, findings: []),
            AnalysisStep(type: .supportResistance, status: .pending, findings: []),
            AnalysisStep(type: .chartPatterns, status: .pending, findings: []),
            AnalysisStep(type: .generatingChart, status: .pending, findings: []),
            AnalysisStep(type: .visionAnalysis, status: .pending, findings: []),
            AnalysisStep(type: .generatingPlan, status: .pending, findings: []),
            AnalysisStep(type: .complete, status: .pending, findings: []),
        ]
    }

    /// Handle detailed step events from the streaming API
    private func handleStepEvent(_ event: APIService.PlanStreamEvent) async {
        guard let stepTypeStr = event.stepType,
              let stepType = AnalysisStepType(rawValue: stepTypeStr),
              let statusStr = event.status else { return }

        let status: AnalysisStep.StepStatus = statusStr == "completed" ? .completed : .active
        let newFindings = event.findings ?? []

        withAnimation(.easeOut(duration: 0.3)) {
            // Find and update the matching step
            if let index = analysisSteps.firstIndex(where: { $0.type == stepType }) {
                analysisSteps[index].status = status

                // Append new findings instead of replacing (avoid data loss on duplicate events)
                for finding in newFindings {
                    if !analysisSteps[index].findings.contains(finding) {
                        analysisSteps[index].findings.append(finding)
                    }
                }

                if status == .completed {
                    analysisSteps[index].timestamp = Date()
                }
            }

            // Handle special "complete" step
            if stepType == .complete && status == .completed {
                isAnalysisComplete = true
            }
        }
    }

    /// Cancel any ongoing streaming operation
    func cancelStreaming() {
        streamingTask?.cancel()
        streamingTask = nil
        withAnimation {
            updatePhase = .idle
            isStreaming = false
            isUpdating = false
            streamingText = ""
        }
    }

    private func handlePhaseEvent(_ phase: String?) async {
        guard let phase = phase else { return }

        withAnimation(.easeInOut(duration: 0.3)) {
            switch phase {
            case "gathering_data":
                updatePhase = .gatheringData
            case "analyzing":
                updatePhase = .analyzing
            case "generating":
                updatePhase = .generatingPlan
            case "complete":
                updatePhase = .complete
            default:
                break
            }
        }
    }

    /// Start polling for plan updates (when viewing the plan)
    func startPolling(interval: TimeInterval = 30) {
        stopPolling()
        pollTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.refreshPlanSilently()
            }
        }
    }

    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    // MARK: - Private Methods

    private func refreshPlanSilently() async {
        do {
            let newPlan = try await APIService.shared.getTradingPlan(symbol: symbol)

            // Check if plan changed
            if let newPlan = newPlan, planChanged(newPlan) {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    self.plan = newPlan
                    self.lastUpdated = Date()
                }
            }
        } catch {
            // Silent refresh - don't show errors
        }
    }

    private func planChanged(_ newPlan: TradingPlanResponse) -> Bool {
        guard let current = plan else { return true }

        // Check key fields for changes
        return current.status != newPlan.status ||
               current.evaluationNotes != newPlan.evaluationNotes ||
               current.lastEvaluation != newPlan.lastEvaluation
    }

    private func animateUpdatePhases() async {
        let phases: [(UpdatePhase, UInt64)] = [
            (.gatheringData, 800_000_000),
            (.analyzing, 1_200_000_000),
            (.generatingPlan, 0) // Will stay here until complete
        ]

        for (phase, delay) in phases {
            withAnimation(.easeInOut(duration: 0.3)) {
                updatePhase = phase
            }
            if delay > 0 {
                try? await Task.sleep(nanoseconds: delay)
            }
        }
    }

    // MARK: - Interactive Plan Session (Claude Code-style planning)

    @Published private(set) var sessionId: String?
    @Published private(set) var sessionStatus: String = "idle"
    @Published private(set) var draftPlan: TradingPlanResponse?
    @Published private(set) var conversationMessages: [PlanMessage] = []
    @Published private(set) var isProcessingFeedback: Bool = false
    @Published private(set) var lastAIResponse: String?

    var hasActiveSession: Bool { sessionId != nil && (sessionStatus == "draft" || sessionStatus == "refining") }
    var isDraftMode: Bool { sessionStatus == "draft" || sessionStatus == "refining" }

    /// Start an interactive planning session
    func startPlanSession() async {
        guard !isLoading else { return }

        isLoading = true
        error = nil
        updatePhase = .gatheringData

        do {
            let session = try await APIService.shared.startPlanSession(symbol: symbol)
            withAnimation {
                self.sessionId = session.sessionId
                self.sessionStatus = session.status
                self.draftPlan = session.draftPlan
                self.conversationMessages = session.messages
            }

            // If session is still generating, poll for updates
            if session.isGenerating {
                await pollSessionUntilDraft()
            } else {
                updatePhase = .idle
            }
        } catch {
            self.error = error.localizedDescription
            updatePhase = .idle
        }

        isLoading = false
    }

    /// Poll session until draft plan is ready
    private func pollSessionUntilDraft() async {
        guard let sessionId = sessionId else { return }

        updatePhase = .generatingPlan

        for _ in 0..<60 { // Max 60 attempts (2 minutes)
            do {
                try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

                let session = try await APIService.shared.getPlanSession(symbol: symbol, sessionId: sessionId)

                withAnimation {
                    self.sessionStatus = session.status
                    self.draftPlan = session.draftPlan
                    self.conversationMessages = session.messages
                }

                if session.hasDraftPlan {
                    withAnimation {
                        self.updatePhase = .complete
                    }
                    try await Task.sleep(nanoseconds: 1_000_000_000)
                    withAnimation {
                        self.updatePhase = .idle
                    }
                    return
                }
            } catch {
                // Continue polling on error
            }
        }

        // Timeout
        self.error = "Plan generation timed out"
        updatePhase = .idle
    }

    /// Submit a question about the draft plan
    func askQuestion(_ question: String) async {
        guard let sessionId = sessionId, !isProcessingFeedback else { return }

        isProcessingFeedback = true

        do {
            let response = try await APIService.shared.submitPlanFeedback(
                symbol: symbol,
                sessionId: sessionId,
                feedbackType: "question",
                content: question
            )

            withAnimation {
                self.lastAIResponse = response.aiResponse
                self.sessionStatus = response.sessionStatus
            }

            // Refresh session to get updated messages
            await refreshSession()
        } catch {
            self.error = error.localizedDescription
        }

        isProcessingFeedback = false
    }

    /// Request an adjustment to the draft plan
    func requestAdjustment(_ request: String) async {
        guard let sessionId = sessionId, !isProcessingFeedback else { return }

        isProcessingFeedback = true

        do {
            let response = try await APIService.shared.submitPlanFeedback(
                symbol: symbol,
                sessionId: sessionId,
                feedbackType: "adjust",
                content: request
            )

            withAnimation {
                self.lastAIResponse = response.aiResponse
                self.sessionStatus = response.sessionStatus
                if let updatedPlan = response.updatedPlan {
                    self.draftPlan = updatedPlan
                }
            }

            // Refresh session to get updated messages
            await refreshSession()
        } catch {
            self.error = error.localizedDescription
        }

        isProcessingFeedback = false
    }

    /// Approve the draft plan
    func approveDraftPlan() async {
        guard let sessionId = sessionId else { return }

        isUpdating = true

        do {
            let approvedPlan = try await APIService.shared.approvePlanSession(symbol: symbol, sessionId: sessionId)

            withAnimation {
                self.plan = approvedPlan
                self.sessionStatus = "approved"
                self.draftPlan = nil
                self.conversationMessages = []
                self.lastUpdated = Date()
            }
        } catch {
            self.error = error.localizedDescription
        }

        isUpdating = false
    }

    /// Refresh session state
    private func refreshSession() async {
        guard let sessionId = sessionId else { return }

        do {
            let session = try await APIService.shared.getPlanSession(symbol: symbol, sessionId: sessionId)

            withAnimation {
                self.sessionStatus = session.status
                self.draftPlan = session.draftPlan
                self.conversationMessages = session.messages
            }
        } catch {
            // Silent refresh
        }
    }

    /// Start a session from an existing plan (for modifying existing plans)
    func startSessionFromExisting() async {
        guard !isLoading, plan != nil else { return }

        isLoading = true
        error = nil

        do {
            let session = try await APIService.shared.startSessionFromExisting(symbol: symbol)
            withAnimation {
                self.sessionId = session.sessionId
                self.sessionStatus = session.status
                self.draftPlan = session.draftPlan
                self.conversationMessages = session.messages
            }
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    /// Reopen an approved session to continue making adjustments
    func reopenSession() async {
        guard let sessionId = sessionId, sessionStatus == "approved" else { return }

        isUpdating = true
        error = nil

        do {
            let session = try await APIService.shared.reopenPlanSession(symbol: symbol, sessionId: sessionId)
            withAnimation {
                self.sessionStatus = session.status
                self.draftPlan = session.draftPlan
                self.conversationMessages = session.messages
            }
        } catch {
            self.error = error.localizedDescription
        }

        isUpdating = false
    }

    /// Clear the current session (start fresh)
    func clearSession() {
        withAnimation {
            sessionId = nil
            sessionStatus = "idle"
            draftPlan = nil
            conversationMessages = []
            lastAIResponse = nil
        }
    }
}
