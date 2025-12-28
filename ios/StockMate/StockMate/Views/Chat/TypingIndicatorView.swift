import SwiftUI

/// Elegant pulsing typing indicator with staggered dot animation
struct TypingIndicatorView: View {
    @State private var animating = false

    private let dotSize: CGFloat = 8
    private let spacing: CGFloat = 4

    var body: some View {
        HStack(spacing: spacing) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(Color.white.opacity(0.7))
                    .frame(width: dotSize, height: dotSize)
                    .scaleEffect(animating ? 1.0 : 0.5)
                    .opacity(animating ? 1.0 : 0.3)
                    .animation(
                        .easeInOut(duration: 0.6)
                        .repeatForever(autoreverses: true)
                        .delay(Double(index) * 0.15),
                        value: animating
                    )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            ZStack {
                // Gradient base
                LinearGradient(
                    colors: [
                        Color(red: 0.15, green: 0.15, blue: 0.2),
                        Color(red: 0.12, green: 0.12, blue: 0.16)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )

                // Subtle shimmer overlay
                LinearGradient(
                    colors: [
                        Color.white.opacity(0.05),
                        Color.clear,
                        Color.white.opacity(0.02)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
        )
        .clipShape(TypingBubbleShape())
        .shadow(color: Color.black.opacity(0.2), radius: 8, x: 0, y: 4)
        .onAppear {
            animating = true
        }
    }
}

/// Custom bubble shape with tail on the left
struct TypingBubbleShape: Shape {
    func path(in rect: CGRect) -> Path {
        let radius: CGFloat = 16
        let tailWidth: CGFloat = 8
        let tailHeight: CGFloat = 10

        var path = Path()

        // Start from bottom left (after tail)
        path.move(to: CGPoint(x: tailWidth, y: rect.maxY - radius))

        // Bottom left corner
        path.addQuadCurve(
            to: CGPoint(x: tailWidth + radius, y: rect.maxY),
            control: CGPoint(x: tailWidth, y: rect.maxY)
        )

        // Bottom edge to right
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

        // Left edge to tail
        path.addLine(to: CGPoint(x: tailWidth, y: rect.maxY - tailHeight - radius))

        // Tail curve
        path.addQuadCurve(
            to: CGPoint(x: 0, y: rect.maxY),
            control: CGPoint(x: tailWidth, y: rect.maxY - 4)
        )

        // Back up to start
        path.addQuadCurve(
            to: CGPoint(x: tailWidth, y: rect.maxY - radius),
            control: CGPoint(x: 2, y: rect.maxY - 2)
        )

        return path
    }
}

/// Wrapper view for typing indicator in message list
struct TypingIndicatorBubble: View {
    var body: some View {
        HStack {
            TypingIndicatorView()
                .transition(.asymmetric(
                    insertion: .scale(scale: 0.8).combined(with: .opacity),
                    removal: .scale(scale: 0.9).combined(with: .opacity)
                ))

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
    }
}

// MARK: - Preview

#Preview("Typing Indicator") {
    ZStack {
        Color.black.ignoresSafeArea()

        VStack(spacing: 20) {
            TypingIndicatorView()

            TypingIndicatorBubble()
        }
    }
}
