import Foundation

/// Service that manages chat messages with local caching and server persistence
actor ChatService {
    static let shared = ChatService()

    /// Local cache of messages by chat key (symbol or "portfolio")
    private var messageStore: [String: [ChatMessage]] = [:]

    private init() {}

    // MARK: - Local Cache Operations

    /// Get messages from local cache
    func getMessages(for key: String) -> [ChatMessage] {
        return messageStore[key] ?? []
    }

    /// Save messages to local cache
    func saveMessages(_ messages: [ChatMessage], for key: String) {
        messageStore[key] = messages
    }

    /// Add a single message to local cache
    func addMessage(_ message: ChatMessage, for key: String) {
        if messageStore[key] == nil {
            messageStore[key] = []
        }
        messageStore[key]?.append(message)
    }

    /// Clear messages from local cache
    func clearMessages(for key: String) {
        messageStore[key] = nil
    }

    /// Clear all local messages
    func clearAllMessages() {
        messageStore.removeAll()
    }

    /// Check if a chat has existing local messages
    func hasMessages(for key: String) -> Bool {
        guard let messages = messageStore[key] else { return false }
        return !messages.isEmpty
    }

    /// Get the chat key for a symbol (or portfolio chat)
    /// - Note: "portfolio" is used for home page chat, symbols are uppercased for stock-specific chats
    static func chatKey(for symbol: String?) -> String {
        return symbol?.uppercased() ?? "portfolio"
    }

    // MARK: - Server Sync Operations

    /// Load messages from server into local cache
    /// - Parameter key: Chat key ("portfolio" or stock symbol)
    /// - Returns: Array of messages loaded from server
    func loadFromServer(for key: String) async throws -> [ChatMessage] {
        let symbol = key == "portfolio" ? nil : key
        let history = try await APIService.shared.getChatHistory(symbol: symbol)

        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        let messages = history.messages.map { msg in
            let timestamp = dateFormatter.date(from: msg.timestamp ?? "") ?? Date()
            return ChatMessage(
                content: msg.content,
                isUser: msg.role == "user",
                timestamp: timestamp
            )
        }

        messageStore[key] = messages
        return messages
    }

    /// Clear messages on server
    /// - Parameter key: Chat key ("portfolio" or stock symbol)
    func clearOnServer(for key: String) async throws {
        let symbol = key == "portfolio" ? nil : key
        try await APIService.shared.clearChatHistory(symbol: symbol)
        messageStore[key] = nil
    }
}
