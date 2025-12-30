import SwiftUI

/// Shows the AI's analysis as it streams in during plan generation
/// Uses the terminal-style AgentStreamingView for Claude Code-like experience
struct StreamingPlanView: View {
    @ObservedObject var viewModel: TradingPlanViewModel

    var body: some View {
        AgentStreamingView(
            symbol: viewModel.symbol,
            steps: $viewModel.analysisSteps,
            isComplete: $viewModel.isAnalysisComplete
        )
    }
}

// MARK: - Legacy Components (kept for backwards compatibility)

/// Enhanced phase indicator for simpler streaming displays
struct EnhancedPhaseIndicator: View {
    let phase: TradingPlanViewModel.UpdatePhase

    private var progress: CGFloat {
        switch phase {
        case .idle: return 0
        case .gatheringData: return 0.25
        case .analyzing: return 0.5
        case .generatingPlan: return 0.75
        case .complete: return 1.0
        }
    }

    private var phaseText: String {
        switch phase {
        case .idle: return "Starting..."
        case .gatheringData: return "Gathering market data..."
        case .analyzing: return "Running technical analysis..."
        case .generatingPlan: return "AI is writing your plan..."
        case .complete: return "Plan ready!"
        }
    }

    private var phaseIcon: String {
        switch phase {
        case .idle: return "hourglass"
        case .gatheringData: return "chart.line.uptrend.xyaxis"
        case .analyzing: return "waveform.path.ecg"
        case .generatingPlan: return "brain.head.profile"
        case .complete: return "checkmark.circle.fill"
        }
    }

    var body: some View {
        VStack(spacing: 12) {
            // Step indicators
            HStack(spacing: 0) {
                ForEach(0..<4, id: \.self) { index in
                    let stepProgress = CGFloat(index + 1) * 0.25

                    HStack(spacing: 0) {
                        // Step circle
                        ZStack {
                            Circle()
                                .stroke(progress >= stepProgress ? Color.blue : Color.gray.opacity(0.3), lineWidth: 2)
                                .frame(width: 24, height: 24)

                            if progress >= stepProgress {
                                Circle()
                                    .fill(Color.blue)
                                    .frame(width: 16, height: 16)
                            }
                        }

                        // Connector line (except for last)
                        if index < 3 {
                            Rectangle()
                                .fill(progress > stepProgress ? Color.blue : Color.gray.opacity(0.3))
                                .frame(height: 2)
                        }
                    }
                }
            }
            .padding(.horizontal, 8)

            // Phase info
            HStack(spacing: 8) {
                Image(systemName: phaseIcon)
                    .font(.system(size: 14))
                    .foregroundColor(.blue)
                    .symbolEffect(.pulse, isActive: phase != .complete && phase != .idle)

                Text(phaseText)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.primary)

                Spacer()

                Text("\(Int(progress * 100))%")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundColor(.blue)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: phase)
    }
}

// MARK: - Streaming Dots Animation

struct StreamingDotsView: View {
    @State private var animating = false

    var body: some View {
        HStack(spacing: 8) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 12, height: 12)
                    .scaleEffect(animating ? 1.0 : 0.5)
                    .opacity(animating ? 1.0 : 0.3)
                    .animation(
                        .easeInOut(duration: 0.6)
                            .repeatForever()
                            .delay(Double(index) * 0.2),
                        value: animating
                    )
            }
        }
        .onAppear {
            animating = true
        }
    }
}

// MARK: - Streaming Cursor

struct StreamingCursor: View {
    @State private var visible = true

    var body: some View {
        Rectangle()
            .fill(Color.blue)
            .frame(width: 2, height: 12)
            .opacity(visible ? 1 : 0)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.5).repeatForever()) {
                    visible.toggle()
                }
            }
    }
}

// MARK: - Preview

#Preview("Streaming Plan") {
    StreamingPlanView(viewModel: {
        let vm = TradingPlanViewModel(symbol: "AAPL")
        return vm
    }())
}
