import SwiftUI

/// A beautiful loading indicator with customizable style
struct LoadingView: View {
    let message: String?
    let style: LoadingStyle

    init(_ message: String? = nil, style: LoadingStyle = .default) {
        self.message = message
        self.style = style
    }

    var body: some View {
        VStack(spacing: 16) {
            switch style {
            case .default:
                ProgressView()
                    .scaleEffect(1.2)
                    .tint(.accentColor)

            case .pulse:
                PulsingDots()

            case .circular:
                CircularLoader()
            }

            if let message {
                Text(message)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
    }

    enum LoadingStyle {
        case `default`
        case pulse
        case circular
    }
}

/// Three pulsing dots animation
struct PulsingDots: View {
    @State private var isAnimating = false

    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<3) { index in
                Circle()
                    .fill(Color.accentColor)
                    .frame(width: 10, height: 10)
                    .scaleEffect(isAnimating ? 1.0 : 0.5)
                    .opacity(isAnimating ? 1.0 : 0.3)
                    .animation(
                        .easeInOut(duration: 0.6)
                        .repeatForever()
                        .delay(Double(index) * 0.15),
                        value: isAnimating
                    )
            }
        }
        .onAppear { isAnimating = true }
    }
}

/// Circular progress loader
struct CircularLoader: View {
    @State private var rotation: Double = 0

    var body: some View {
        Circle()
            .trim(from: 0, to: 0.7)
            .stroke(
                AngularGradient(
                    colors: [.accentColor.opacity(0), .accentColor],
                    center: .center
                ),
                style: StrokeStyle(lineWidth: 3, lineCap: .round)
            )
            .frame(width: 30, height: 30)
            .rotationEffect(.degrees(rotation))
            .onAppear {
                withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
    }
}

/// Empty state view with icon and message
struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    let actionTitle: String?
    let action: (() -> Void)?

    init(
        icon: String,
        title: String,
        message: String,
        actionTitle: String? = nil,
        action: (() -> Void)? = nil
    ) {
        self.icon = icon
        self.title = title
        self.message = message
        self.actionTitle = actionTitle
        self.action = action
    }

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(.secondary)
                .symbolEffect(.pulse.byLayer, options: .repeating)

            VStack(spacing: 6) {
                Text(title)
                    .font(.system(size: 18, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(message)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }

            if let actionTitle, let action {
                Button(action: action) {
                    Text(actionTitle)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(Color.accentColor)
                        )
                }
                .padding(.top, 8)
            }
        }
        .padding(32)
    }
}

/// Error state view with retry option
struct ErrorStateView: View {
    let message: String
    let retryAction: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.orange)

            VStack(spacing: 6) {
                Text("Something went wrong")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(message)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
            }

            Button(action: retryAction) {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 14, weight: .semibold))

                    Text("Retry")
                        .font(.system(size: 15, weight: .semibold))
                }
                .foregroundStyle(Color.accentColor)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color.accentColor, lineWidth: 1.5)
                )
            }
            .padding(.top, 4)
        }
        .padding(24)
    }
}

// MARK: - Preview

#Preview("Loading States") {
    VStack(spacing: 40) {
        LoadingView("Loading market data...")
        LoadingView("Fetching...", style: .pulse)
        LoadingView(style: .circular)
    }
    .padding()
}

#Preview("Empty State") {
    EmptyStateView(
        icon: "chart.line.uptrend.xyaxis",
        title: "No Stocks Found",
        message: "No stocks match your current profile criteria. Try adjusting your settings.",
        actionTitle: "Refresh",
        action: {}
    )
}

#Preview("Error State") {
    ErrorStateView(
        message: "Unable to connect to the server. Please check your connection.",
        retryAction: {}
    )
}
