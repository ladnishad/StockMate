import Foundation
import SwiftUI
import Combine

/// ViewModel for the AI chat interface
@MainActor
final class ChatViewModel: ObservableObject {
    // MARK: - Published State

    @Published private(set) var messages: [ChatMessage] = []
    @Published private(set) var isTyping: Bool = false
    @Published private(set) var error: String?
    @Published var inputText: String = ""

    // MARK: - Properties

    let symbol: String?
    private let userId: String
    private let chatKey: String
    private var sendTask: Task<Void, Never>?

    // MARK: - Computed Properties

    var hasMessages: Bool { !messages.isEmpty }
    var canSend: Bool { !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isTyping }

    var headerTitle: String {
        if let symbol = symbol {
            return symbol
        }
        return "Portfolio AI"
    }

    var headerSubtitle: String {
        if symbol != nil {
            return "Ask about this stock"
        }
        return "Ask about your portfolio"
    }

    // MARK: - Initialization

    init(symbol: String? = nil, userId: String = "default") {
        self.symbol = symbol
        self.userId = userId
        self.chatKey = ChatService.chatKey(for: symbol)

        // Load persisted messages or show welcome
        Task {
            await loadPersistedMessages()
        }
    }

    // MARK: - Persistence

    private func loadPersistedMessages() async {
        // Try to load from server first (for persistent chat history)
        do {
            let serverMessages = try await ChatService.shared.loadFromServer(for: chatKey)
            if !serverMessages.isEmpty {
                messages = serverMessages
                return
            }
        } catch {
            // Server load failed, fall back to local cache
            print("Failed to load from server: \(error)")
        }

        // Fall back to local cache
        let persisted = await ChatService.shared.getMessages(for: chatKey)
        if persisted.isEmpty {
            addWelcomeMessage()
        } else {
            messages = persisted
        }
    }

    private func persistMessages() {
        Task {
            await ChatService.shared.saveMessages(messages, for: chatKey)
        }
    }

    // MARK: - Public Methods

    func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isTyping else { return }

        // Clear input immediately for snappy feel
        inputText = ""

        // Add user message with animation
        let userMessage = ChatMessage(content: text, isUser: true)
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
            messages.append(userMessage)
        }
        persistMessages()

        // Show typing indicator
        withAnimation(.easeInOut(duration: 0.2)) {
            isTyping = true
        }

        // Cancel any pending send
        sendTask?.cancel()

        // Send to API
        sendTask = Task {
            await performSend(text: text)
        }
    }

    func clearError() {
        withAnimation {
            error = nil
        }
    }

    func clearChat() {
        withAnimation(.spring(response: 0.4, dampingFraction: 0.75)) {
            messages.removeAll()
            addWelcomeMessage()
        }
        // Clear from server and local cache
        Task {
            do {
                try await ChatService.shared.clearOnServer(for: chatKey)
            } catch {
                print("Failed to clear on server: \(error)")
                // Still clear locally even if server fails
                await ChatService.shared.clearMessages(for: chatKey)
            }
        }
        persistMessages()
    }

    // MARK: - Private Methods

    private func addWelcomeMessage() {
        let welcomeText: String
        if let symbol = symbol {
            welcomeText = "I'm your AI trading assistant for \(symbol). Ask me about price levels, technical analysis, trade setups, or anything else about this stock."
        } else {
            welcomeText = "I'm your Portfolio AI assistant. I can help you with:\n\n\u{2022} How is my portfolio doing?\n\u{2022} Analyze any stock in your watchlist\n\u{2022} What's the market direction?\n\u{2022} Which stocks are near support?\n\nAsk me anything about your stocks!"
        }

        let welcomeMessage = ChatMessage(content: welcomeText, isUser: false)
        messages.append(welcomeMessage)
    }

    private func performSend(text: String) async {
        do {
            // Track time for minimum typing indicator
            let startTime = Date()

            let responseText: String
            var stockContext: StockContext?

            if let symbol = symbol {
                // Stock-specific chat - use existing ChatResponse
                let response = try await APIService.shared.sendChatMessage(
                    symbol: symbol,
                    message: text,
                    userId: userId
                )
                responseText = response.response

                // Build stock context
                stockContext = StockContext(
                    symbol: symbol,
                    price: response.context?.currentPrice,
                    changePercent: nil,
                    rsi: nil,
                    hasPosition: response.context?.hasPosition ?? false,
                    hasPlan: response.hasPlan ?? false,
                    planStatus: response.planStatus
                )
            } else {
                // Portfolio chat - use new PortfolioChatResponse
                let response = try await APIService.shared.sendPortfolioChatMessage(
                    message: text,
                    userId: userId
                )
                responseText = response.response

                // Build portfolio context if we have positions
                if let summary = response.portfolioSummary, summary.totalPositions > 0 {
                    stockContext = StockContext(
                        symbol: "Portfolio",
                        price: nil,
                        changePercent: nil,
                        rsi: nil,
                        hasPosition: true,
                        hasPlan: false,
                        planStatus: nil
                    )
                }
            }

            // Ensure minimum 0.5s typing indicator for natural feel
            let elapsed = Date().timeIntervalSince(startTime)
            if elapsed < 0.5 {
                try? await Task.sleep(nanoseconds: UInt64((0.5 - elapsed) * 1_000_000_000))
            }

            // Check for cancellation
            guard !Task.isCancelled else { return }

            // Add AI response with animation
            let aiMessage = ChatMessage(
                content: responseText,
                isUser: false,
                stockContext: stockContext
            )

            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                isTyping = false
                messages.append(aiMessage)
            }
            persistMessages()

        } catch {
            guard !Task.isCancelled else { return }

            withAnimation {
                isTyping = false
                self.error = error.localizedDescription
            }

            // Add error message to chat
            let errorMessage = ChatMessage(
                content: "Sorry, I couldn't process that request. Please try again.",
                isUser: false
            )
            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                messages.append(errorMessage)
            }
            persistMessages()
        }
    }
}

// MARK: - Preview Helpers

extension ChatViewModel {
    static var preview: ChatViewModel {
        let vm = ChatViewModel(symbol: "AAPL")
        vm.messages = [
            ChatMessage(content: "I'm your AI trading assistant for AAPL.", isUser: false),
            ChatMessage(content: "What's the current support level?", isUser: true),
            ChatMessage(
                content: "AAPL has strong support at $174.50, which aligns with the 21 EMA. The next support below is at $171.20. RSI is currently at 45, neutral territory.",
                isUser: false,
                stockContext: StockContext(
                    symbol: "AAPL",
                    price: 175.84,
                    changePercent: 1.24,
                    rsi: 45.0,
                    hasPosition: false
                )
            )
        ]
        return vm
    }

    static var previewTyping: ChatViewModel {
        let vm = ChatViewModel(symbol: "NVDA")
        vm.messages = [
            ChatMessage(content: "I'm your AI trading assistant for NVDA.", isUser: false),
            ChatMessage(content: "Should I buy here?", isUser: true)
        ]
        vm.isTyping = true
        return vm
    }
}
