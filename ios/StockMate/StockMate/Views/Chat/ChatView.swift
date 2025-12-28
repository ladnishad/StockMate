import SwiftUI

/// Elegant real-time chat interface with Bloomberg-meets-iMessage aesthetics
struct ChatView: View {
    @StateObject private var viewModel: ChatViewModel
    @FocusState private var isInputFocused: Bool
    @Environment(\.dismiss) private var dismiss

    init(symbol: String? = nil) {
        _viewModel = StateObject(wrappedValue: ChatViewModel(symbol: symbol))
    }

    var body: some View {
        ZStack {
            // Background gradient
            backgroundGradient
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                headerView

                // Messages
                messagesScrollView

                // Input bar
                inputBar
            }
        }
        .navigationBarHidden(true)
        .onAppear {
            // Auto-focus the input field after a small delay
            // This helps with fullScreenCover presentation
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                isInputFocused = true
            }
        }
    }

    // MARK: - Background

    private var backgroundGradient: some View {
        ZStack {
            // Deep dark base
            Color(red: 0.04, green: 0.04, blue: 0.06)

            // Subtle radial gradient from top
            RadialGradient(
                colors: [
                    Color(red: 0.08, green: 0.08, blue: 0.12),
                    Color.clear
                ],
                center: .top,
                startRadius: 100,
                endRadius: 500
            )

            // Noise texture overlay
            NoiseTexture()
                .opacity(0.02)
        }
    }

    // MARK: - Header

    private var headerView: some View {
        HStack(spacing: 16) {
            // Back button
            Button(action: { dismiss() }) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 36, height: 36)
                    .background(
                        Circle()
                            .fill(Color.white.opacity(0.08))
                    )
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.headerTitle)
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)

                Text(viewModel.headerSubtitle)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Color(white: 0.5))
            }

            Spacer()

            // Clear chat button
            if viewModel.hasMessages {
                Button(action: { viewModel.clearChat() }) {
                    Image(systemName: "trash")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(Color(white: 0.6))
                        .frame(width: 36, height: 36)
                        .background(
                            Circle()
                                .fill(Color.white.opacity(0.05))
                        )
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
        .background(
            // Glass header
            ZStack {
                Color(red: 0.06, green: 0.06, blue: 0.08).opacity(0.9)

                LinearGradient(
                    colors: [
                        Color.white.opacity(0.03),
                        Color.clear
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            }
        )
        .overlay(
            Divider()
                .frame(height: 0.5)
                .background(Color.white.opacity(0.08)),
            alignment: .bottom
        )
    }

    // MARK: - Messages

    private var messagesScrollView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(viewModel.messages) { message in
                        MessageBubbleView(message: message)
                            .id(message.id)
                    }

                    // Typing indicator
                    if viewModel.isTyping {
                        TypingIndicatorBubble()
                            .id("typing")
                    }
                }
                .padding(.top, 16)
                .padding(.bottom, 8)
            }
            .onChange(of: viewModel.messages.count) { _ in
                withAnimation(.easeOut(duration: 0.3)) {
                    if viewModel.isTyping {
                        proxy.scrollTo("typing", anchor: .bottom)
                    } else if let lastMessage = viewModel.messages.last {
                        proxy.scrollTo(lastMessage.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: viewModel.isTyping) { isTyping in
                if isTyping {
                    withAnimation(.easeOut(duration: 0.3)) {
                        proxy.scrollTo("typing", anchor: .bottom)
                    }
                }
            }
        }
        .scrollDismissesKeyboard(.interactively)
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 0) {
            // Error banner
            if let error = viewModel.error {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error)
                        .font(.system(size: 13))
                        .foregroundColor(Color(white: 0.8))
                    Spacer()
                    Button("Dismiss") {
                        viewModel.clearError()
                    }
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.blue)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.orange.opacity(0.15))
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            // Input area
            HStack(alignment: .bottom, spacing: 12) {
                // Text input
                HStack(spacing: 0) {
                    TextField("Ask about stocks...", text: $viewModel.inputText, axis: .vertical)
                        .font(.system(size: 16))
                        .foregroundColor(.white)
                        .lineLimit(1...5)
                        .focused($isInputFocused)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .submitLabel(.send)
                        .onSubmit {
                            if viewModel.canSend {
                                viewModel.sendMessage()
                            }
                        }
                }
                .background(
                    ZStack {
                        Color(red: 0.1, green: 0.1, blue: 0.12)

                        // Focus highlight
                        if isInputFocused {
                            RoundedRectangle(cornerRadius: 22)
                                .strokeBorder(
                                    LinearGradient(
                                        colors: [
                                            Color.blue.opacity(0.5),
                                            Color.blue.opacity(0.2)
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    lineWidth: 1.5
                                )
                        }
                    }
                )
                .clipShape(RoundedRectangle(cornerRadius: 22))
                .overlay(
                    RoundedRectangle(cornerRadius: 22)
                        .strokeBorder(Color.white.opacity(0.06), lineWidth: 1)
                )

                // Send button
                Button(action: { viewModel.sendMessage() }) {
                    ZStack {
                        Circle()
                            .fill(
                                viewModel.canSend
                                    ? LinearGradient(
                                        colors: [
                                            Color(red: 0.0, green: 0.48, blue: 1.0),
                                            Color(red: 0.0, green: 0.38, blue: 0.9)
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                    : LinearGradient(
                                        colors: [
                                            Color(white: 0.2),
                                            Color(white: 0.15)
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                            )
                            .frame(width: 44, height: 44)
                            .shadow(
                                color: viewModel.canSend ? Color.blue.opacity(0.4) : Color.clear,
                                radius: 8,
                                x: 0,
                                y: 4
                            )

                        Image(systemName: "arrow.up")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(viewModel.canSend ? .white : Color(white: 0.4))
                    }
                }
                .disabled(!viewModel.canSend)
                .animation(.spring(response: 0.3, dampingFraction: 0.7), value: viewModel.canSend)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                // Glass input bar
                ZStack {
                    Color(red: 0.06, green: 0.06, blue: 0.08).opacity(0.95)

                    LinearGradient(
                        colors: [
                            Color.white.opacity(0.02),
                            Color.clear
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                }
            )
            .overlay(
                Divider()
                    .frame(height: 0.5)
                    .background(Color.white.opacity(0.08)),
                alignment: .top
            )
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: viewModel.error != nil)
    }
}

// MARK: - Noise Texture

struct NoiseTexture: View {
    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                for _ in 0..<Int(size.width * size.height / 50) {
                    let x = CGFloat.random(in: 0..<size.width)
                    let y = CGFloat.random(in: 0..<size.height)
                    let opacity = Double.random(in: 0.1...0.3)

                    context.fill(
                        Path(ellipseIn: CGRect(x: x, y: y, width: 1, height: 1)),
                        with: .color(Color.white.opacity(opacity))
                    )
                }
            }
        }
    }
}

// MARK: - Preview

#Preview("Chat View") {
    ChatView(symbol: "AAPL")
}

#Preview("General Chat") {
    ChatView()
}

#Preview("Chat with Messages") {
    NavigationStack {
        ChatViewPreviewContainer()
    }
}

struct ChatViewPreviewContainer: View {
    var body: some View {
        ChatView(symbol: "NVDA")
    }
}
