import SwiftUI

/// Elegant message bubble with glass morphism and smooth animations
struct MessageBubbleView: View {
    let message: ChatMessage
    @State private var appeared = false

    private var isUser: Bool { message.isUser }

    /// Parse message content as markdown, fallback to plain text
    private var markdownContent: AttributedString {
        do {
            return try AttributedString(markdown: message.content)
        } catch {
            return AttributedString(message.content)
        }
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isUser { Spacer(minLength: 48) }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
                // Stock context badge (for AI responses with stock data)
                if let context = message.stockContext, !isUser {
                    StockContextBadge(context: context)
                        .transition(.opacity.combined(with: .scale(scale: 0.9)))
                }

                // Message bubble with markdown support
                Text(markdownContent)
                    .font(.system(size: 16, weight: .regular, design: .default))
                    .foregroundColor(isUser ? .white : Color(white: 0.92))
                    .lineSpacing(2)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        Group {
                            if isUser {
                                userBubbleBackground
                            } else {
                                aiBubbleBackground
                            }
                        }
                    )
                    .clipShape(BubbleShape(isUser: isUser))
                    .shadow(
                        color: isUser
                            ? Color.blue.opacity(0.25)
                            : Color.black.opacity(0.2),
                        radius: isUser ? 12 : 8,
                        x: 0,
                        y: 4
                    )

                // Timestamp
                Text(message.timestamp, style: .time)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Color(white: 0.5))
                    .padding(.horizontal, 4)
            }

            if !isUser { Spacer(minLength: 48) }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
        .scaleEffect(appeared ? 1 : 0.85)
        .opacity(appeared ? 1 : 0)
        .offset(y: appeared ? 0 : 20)
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.75)) {
                appeared = true
            }
        }
    }

    // MARK: - Backgrounds

    private var userBubbleBackground: some View {
        LinearGradient(
            colors: [
                Color(red: 0.0, green: 0.48, blue: 1.0),
                Color(red: 0.0, green: 0.38, blue: 0.9)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    private var aiBubbleBackground: some View {
        ZStack {
            // Base glass effect
            Color(red: 0.12, green: 0.12, blue: 0.16)

            // Subtle gradient overlay
            LinearGradient(
                colors: [
                    Color.white.opacity(0.08),
                    Color.clear
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            // Inner glow border
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(
                    LinearGradient(
                        colors: [
                            Color.white.opacity(0.12),
                            Color.white.opacity(0.04)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        }
    }
}

/// Custom bubble shape with tail
struct BubbleShape: Shape {
    let isUser: Bool

    func path(in rect: CGRect) -> Path {
        let radius: CGFloat = 18
        let tailWidth: CGFloat = 6
        let tailHeight: CGFloat = 8

        var path = Path()

        if isUser {
            // User bubble - tail on right
            path.move(to: CGPoint(x: 0, y: radius))

            // Top left corner
            path.addQuadCurve(
                to: CGPoint(x: radius, y: 0),
                control: CGPoint(x: 0, y: 0)
            )

            // Top edge
            path.addLine(to: CGPoint(x: rect.maxX - radius, y: 0))

            // Top right corner
            path.addQuadCurve(
                to: CGPoint(x: rect.maxX, y: radius),
                control: CGPoint(x: rect.maxX, y: 0)
            )

            // Right edge to tail
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - tailHeight - radius))

            // Tail
            path.addQuadCurve(
                to: CGPoint(x: rect.maxX + tailWidth, y: rect.maxY),
                control: CGPoint(x: rect.maxX, y: rect.maxY - 4)
            )

            path.addQuadCurve(
                to: CGPoint(x: rect.maxX - radius, y: rect.maxY),
                control: CGPoint(x: rect.maxX - 2, y: rect.maxY - 2)
            )

            // Bottom edge
            path.addLine(to: CGPoint(x: radius, y: rect.maxY))

            // Bottom left corner
            path.addQuadCurve(
                to: CGPoint(x: 0, y: rect.maxY - radius),
                control: CGPoint(x: 0, y: rect.maxY)
            )

            // Left edge
            path.addLine(to: CGPoint(x: 0, y: radius))

        } else {
            // AI bubble - tail on left
            path.move(to: CGPoint(x: tailWidth, y: rect.maxY - radius))

            // Bottom left with tail
            path.addQuadCurve(
                to: CGPoint(x: 0, y: rect.maxY),
                control: CGPoint(x: tailWidth, y: rect.maxY - 4)
            )

            path.addQuadCurve(
                to: CGPoint(x: tailWidth + radius, y: rect.maxY),
                control: CGPoint(x: 2, y: rect.maxY - 2)
            )

            // Bottom edge
            path.addLine(to: CGPoint(x: rect.maxX - radius, y: rect.maxY))

            // Bottom right corner
            path.addQuadCurve(
                to: CGPoint(x: rect.maxX, y: rect.maxY - radius),
                control: CGPoint(x: rect.maxX, y: rect.maxY)
            )

            // Right edge
            path.addLine(to: CGPoint(x: rect.maxX, y: radius))

            // Top right corner
            path.addQuadCurve(
                to: CGPoint(x: rect.maxX - radius, y: 0),
                control: CGPoint(x: rect.maxX, y: 0)
            )

            // Top edge
            path.addLine(to: CGPoint(x: tailWidth + radius, y: 0))

            // Top left corner
            path.addQuadCurve(
                to: CGPoint(x: tailWidth, y: radius),
                control: CGPoint(x: tailWidth, y: 0)
            )

            // Left edge
            path.addLine(to: CGPoint(x: tailWidth, y: rect.maxY - radius))
        }

        return path
    }
}

/// Stock context badge shown above AI messages
struct StockContextBadge: View {
    let context: StockContext

    var body: some View {
        HStack(spacing: 12) {
            // Symbol
            Text(context.symbol)
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundColor(.white)

            // Price
            Text(context.priceFormatted)
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundColor(Color(white: 0.85))

            // Change
            if !context.changeFormatted.isEmpty {
                Text(context.changeFormatted)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundColor(context.isPositive ? Color.green : Color.red)
            }

            // RSI if available
            if let rsi = context.rsi {
                HStack(spacing: 3) {
                    Text("RSI")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(Color(white: 0.5))
                    Text(String(format: "%.0f", rsi))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundColor(rsiColor(rsi))
                }
            }

            // Position indicator
            if context.hasPosition {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 12))
                    .foregroundColor(.green)
            }

            // Plan status
            if context.hasPlan {
                HStack(spacing: 3) {
                    Image(systemName: "doc.text.fill")
                        .font(.system(size: 10))
                    Text(context.planStatusFormatted)
                        .font(.system(size: 10, weight: .medium))
                }
                .foregroundColor(planStatusColor)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            ZStack {
                Color(red: 0.08, green: 0.08, blue: 0.1)

                LinearGradient(
                    colors: [
                        Color.white.opacity(0.05),
                        Color.clear
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            }
        )
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
        )
    }

    private func rsiColor(_ value: Double) -> Color {
        if value >= 70 { return .red }
        if value <= 30 { return .green }
        return Color(white: 0.75)
    }

    private var planStatusColor: Color {
        switch context.planStatus?.lowercased() {
        case "active": return Color.blue
        case "invalidated": return Color.orange
        case "completed": return Color.green
        case "stopped_out": return Color.red
        default: return Color(white: 0.6)
        }
    }
}

// MARK: - Preview

#Preview("Message Bubbles") {
    ZStack {
        Color.black.ignoresSafeArea()

        ScrollView {
            VStack(spacing: 8) {
                MessageBubbleView(message: ChatMessage(
                    content: "I'm your AI trading assistant. How can I help you today?",
                    isUser: false
                ))

                MessageBubbleView(message: ChatMessage(
                    content: "What's the support level for AAPL?",
                    isUser: true
                ))

                MessageBubbleView(message: ChatMessage(
                    content: "AAPL has strong support at $174.50, which aligns with the 21 EMA. The RSI is at 45, neutral territory. Volume is average.",
                    isUser: false,
                    stockContext: StockContext(
                        symbol: "AAPL",
                        price: 175.84,
                        changePercent: 1.24,
                        rsi: 45.0,
                        hasPosition: false
                    )
                ))

                MessageBubbleView(message: ChatMessage(
                    content: "Should I enter here?",
                    isUser: true
                ))
            }
            .padding(.vertical, 20)
        }
    }
}
