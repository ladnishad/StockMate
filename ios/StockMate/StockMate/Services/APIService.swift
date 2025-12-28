import Foundation

// MARK: - Date Decoding Helper

/// Creates a date decoding strategy that handles ISO 8601 dates with fractional seconds
private func makeISO8601DateDecodingStrategy() -> JSONDecoder.DateDecodingStrategy {
    .custom { decoder in
        let container = try decoder.singleValueContainer()
        let dateString = try container.decode(String.self)

        // Try ISO 8601 with fractional seconds first
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: dateString) {
            return date
        }

        // Fall back to standard ISO 8601
        formatter.formatOptions = [.withInternetDateTime]
        if let date = formatter.date(from: dateString) {
            return date
        }

        // Try a custom format for Python's default datetime format
        let customFormatter = DateFormatter()
        customFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        customFormatter.timeZone = TimeZone(identifier: "UTC")
        if let date = customFormatter.date(from: dateString) {
            return date
        }

        // Try without microseconds
        customFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        if let date = customFormatter.date(from: dateString) {
            return date
        }

        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "Cannot decode date: \(dateString)"
        )
    }
}

/// Handles all API communication with the StockMate backend
actor APIService {
    static let shared = APIService()

    private let baseURL = "http://localhost:8000"
    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
    }

    // MARK: - Market Data

    /// Fetch quick market overview (indices)
    func fetchMarketQuick() async throws -> [MarketIndex] {
        let url = URL(string: "\(baseURL)/market/quick")!
        let response: MarketQuickResponse = try await fetch(url: url)
        return response.marketIndices
    }

    /// Fetch full market overview with signals
    func fetchMarketOverview(daysBack: Int = 30) async throws -> MarketQuickResponse {
        var components = URLComponents(string: "\(baseURL)/market")!
        components.queryItems = [URLQueryItem(name: "days_back", value: "\(daysBack)")]
        let response: MarketQuickResponse = try await fetch(url: components.url!)
        return response
    }

    // MARK: - Watchlist / Market Scan

    /// Fetch smart watchlist based on market scan and profile
    func fetchWatchlist(profile: TraderProfile, minScore: Int? = nil, topSectors: Int = 3, stocksPerSector: Int = 5) async throws -> [Stock] {
        var components = URLComponents(string: "\(baseURL)/market/scan")!
        let scoreThreshold = minScore ?? profile.confidenceThreshold
        components.queryItems = [
            URLQueryItem(name: "min_stock_score", value: "\(scoreThreshold)"),
            URLQueryItem(name: "top_sectors", value: "\(topSectors)"),
            URLQueryItem(name: "stocks_per_sector", value: "\(stocksPerSector)")
        ]

        let response: MarketScanResponse = try await fetch(url: components.url!)
        return response.allStocks
    }

    /// Fetch full market scan data
    func fetchMarketScan(profile: TraderProfile) async throws -> MarketScanResponse {
        var components = URLComponents(string: "\(baseURL)/market/scan")!
        components.queryItems = [
            URLQueryItem(name: "min_stock_score", value: "\(profile.confidenceThreshold)"),
            URLQueryItem(name: "top_sectors", value: "3"),
            URLQueryItem(name: "stocks_per_sector", value: "5")
        ]

        return try await fetch(url: components.url!)
    }

    // MARK: - Stock Analysis

    /// Analyze a specific stock with given profile
    func analyzeStock(symbol: String, accountSize: Double = 10000, profile: TraderProfile) async throws -> AnalysisResponse {
        let url = URL(string: "\(baseURL)/analyze")!

        struct AnalysisRequest: Encodable {
            let symbol: String
            let account_size: Double
            let use_ai: Bool
            let trader_profile: String
        }

        let request = AnalysisRequest(
            symbol: symbol.uppercased(),
            account_size: accountSize,
            use_ai: false,
            trader_profile: profile.rawValue
        )

        return try await post(url: url, body: request)
    }

    /// Smart stock analysis - AI agent determines optimal trade style and provides educational content
    /// No profile selection needed - the agent analyzes the stock and recommends the best approach
    /// Note: This endpoint can take 60-90 seconds due to comprehensive data gathering and AI analysis
    func smartAnalyzeStock(symbol: String) async throws -> SmartAnalysisResponse {
        let url = URL(string: "\(baseURL)/analyze/smart/\(symbol.uppercased())")!
        return try await fetchWithTimeout(url: url, timeout: 120)
    }

    /// Fetch with custom timeout for long-running requests
    private func fetchWithTimeout<T: Decodable>(url: URL, timeout: TimeInterval) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = timeout

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorResponse = try? JSONDecoder().decode(APIError.self, from: data) {
                throw APIServiceError.serverError(errorResponse.detail)
            }
            throw APIServiceError.httpError(httpResponse.statusCode)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = makeISO8601DateDecodingStrategy()
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            print("Decoding error: \(error)")
            print("Raw JSON: \(String(data: data, encoding: .utf8) ?? "nil")")
            throw APIServiceError.decodingError(error)
        }
    }

    // MARK: - Profiles

    /// Fetch available profiles
    func fetchProfiles() async throws -> ProfilesResponse {
        let url = URL(string: "\(baseURL)/profiles")!
        return try await fetch(url: url)
    }

    // MARK: - Real-Time Data

    /// Fetch real-time quote for a symbol
    func fetchQuote(symbol: String) async throws -> [String: Any] {
        let url = URL(string: "\(baseURL)/quote/\(symbol.uppercased())")!
        let (data, _) = try await session.data(from: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return json ?? [:]
    }

    // MARK: - User Watchlist

    /// Fetch user's watchlist with live prices
    func fetchUserWatchlist(userId: String = "default") async throws -> [WatchlistItem] {
        var components = URLComponents(string: "\(baseURL)/watchlist")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]
        let response: WatchlistResponse = try await fetch(url: components.url!)
        return response.items
    }

    /// Add a symbol to user's watchlist
    func addToWatchlist(symbol: String, userId: String = "default") async throws -> WatchlistItem {
        var components = URLComponents(string: "\(baseURL)/watchlist/\(symbol.uppercased())")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        struct EmptyBody: Encodable {}
        return try await post(url: components.url!, body: EmptyBody())
    }

    /// Remove a symbol from user's watchlist
    func removeFromWatchlist(symbol: String, userId: String = "default") async throws {
        var components = URLComponents(string: "\(baseURL)/watchlist/\(symbol.uppercased())")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]
        try await delete(url: components.url!)
    }

    /// Search for tickers
    func searchTickers(query: String, limit: Int = 10) async throws -> [SearchResult] {
        guard !query.isEmpty else { return [] }

        var components = URLComponents(string: "\(baseURL)/watchlist/search")!
        components.queryItems = [
            URLQueryItem(name: "query", value: query),
            URLQueryItem(name: "limit", value: "\(limit)")
        ]
        return try await fetch(url: components.url!)
    }

    /// Fetch comprehensive stock detail with charts
    func fetchStockDetail(symbol: String) async throws -> StockDetail {
        let url = URL(string: "\(baseURL)/stock/\(symbol.uppercased())/detail")!
        return try await fetch(url: url)
    }

    /// Fetch chart bars for a specific timeframe
    func fetchBars(symbol: String, timeframe: ChartTimeframe) async throws -> [PriceBar] {
        var components = URLComponents(string: "\(baseURL)/stock/\(symbol.uppercased())/bars")!
        components.queryItems = [
            URLQueryItem(name: "timeframe", value: timeframe.apiValue)
        ]
        return try await fetch(url: components.url!)
    }

    // MARK: - Chat

    /// Send a chat message about a specific stock
    func sendChatMessage(symbol: String, message: String, userId: String = "default") async throws -> ChatResponse {
        let url = URL(string: "\(baseURL)/chat/\(symbol.uppercased())")!

        struct ChatRequest: Encodable {
            let message: String
            let user_id: String
        }

        let request = ChatRequest(message: message, user_id: userId)
        return try await post(url: url, body: request)
    }

    /// Send a portfolio chat message (home page chat)
    /// Connects to the Portfolio Agent which can analyze all watchlist stocks
    func sendPortfolioChatMessage(message: String, userId: String = "default") async throws -> PortfolioChatResponse {
        var components = URLComponents(string: "\(baseURL)/chat")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        struct ChatRequest: Encodable {
            let message: String
        }

        let request = ChatRequest(message: message)
        return try await post(url: components.url!, body: request)
    }

    // MARK: - Chat History

    /// Get chat history from server
    /// - Parameters:
    ///   - symbol: Stock symbol for stock-specific chat, or nil for portfolio chat
    ///   - userId: User identifier
    ///   - limit: Maximum number of messages to return
    /// - Returns: Chat history response
    func getChatHistory(symbol: String? = nil, userId: String = "default", limit: Int = 50) async throws -> ChatHistoryResponse {
        var components = URLComponents(string: "\(baseURL)/chat/history")!
        var queryItems = [URLQueryItem(name: "user_id", value: userId)]
        queryItems.append(URLQueryItem(name: "limit", value: "\(limit)"))
        if let symbol = symbol {
            queryItems.append(URLQueryItem(name: "symbol", value: symbol.uppercased()))
        }
        components.queryItems = queryItems

        return try await fetch(url: components.url!)
    }

    /// Clear chat history on server
    /// - Parameters:
    ///   - symbol: Stock symbol for stock-specific chat, or nil for portfolio chat
    ///   - userId: User identifier
    func clearChatHistory(symbol: String? = nil, userId: String = "default") async throws {
        var components = URLComponents(string: "\(baseURL)/chat/history")!
        var queryItems = [URLQueryItem(name: "user_id", value: userId)]
        if let symbol = symbol {
            queryItems.append(URLQueryItem(name: "symbol", value: symbol.uppercased()))
        }
        components.queryItems = queryItems

        try await delete(url: components.url!)
    }

    // MARK: - Trading Plan

    /// Get trading plan for a symbol
    func getTradingPlan(symbol: String, userId: String = "default") async throws -> TradingPlanResponse? {
        var components = URLComponents(string: "\(baseURL)/chat/\(symbol.uppercased())/plan")!
        components.queryItems = [
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "force_new", value: "false")
        ]

        do {
            return try await fetch(url: components.url!)
        } catch APIServiceError.httpError(404) {
            return nil
        }
    }

    /// Generate a new trading plan
    func generateTradingPlan(symbol: String, forceNew: Bool = false, userId: String = "default") async throws -> TradingPlanResponse {
        var components = URLComponents(string: "\(baseURL)/chat/\(symbol.uppercased())/plan")!
        components.queryItems = [
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "force_new", value: String(forceNew))
        ]

        struct EmptyBody: Encodable {}
        return try await post(url: components.url!, body: EmptyBody())
    }

    // MARK: - Streaming Trading Plan

    /// Stream event types from plan generation
    struct PlanStreamEvent: Decodable {
        let type: String
        let phase: String?
        let content: String?
        let plan: TradingPlanResponse?
        let message: String?
    }

    /// Generate trading plan with real-time streaming
    /// Returns an AsyncThrowingStream of events as the AI generates the plan
    nonisolated func generateTradingPlanStream(
        symbol: String,
        forceNew: Bool = true,
        userId: String = "default"
    ) -> AsyncThrowingStream<PlanStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var components = URLComponents(string: "http://localhost:8000/chat/\(symbol.uppercased())/plan/stream")!
                components.queryItems = [
                    URLQueryItem(name: "user_id", value: userId),
                    URLQueryItem(name: "force_new", value: String(forceNew))
                ]

                var request = URLRequest(url: components.url!)
                request.httpMethod = "POST"
                request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = "{}".data(using: .utf8)

                // Increase timeout for streaming
                request.timeoutInterval = 120

                do {
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)

                    guard let httpResponse = response as? HTTPURLResponse,
                          (200...299).contains(httpResponse.statusCode) else {
                        continuation.finish(throwing: APIServiceError.invalidResponse)
                        return
                    }

                    var buffer = ""

                    for try await byte in bytes {
                        guard let char = String(bytes: [byte], encoding: .utf8) else { continue }
                        buffer += char

                        // SSE format: "data: {json}\n\n"
                        while let range = buffer.range(of: "\n\n") {
                            let line = String(buffer[..<range.lowerBound])
                            buffer = String(buffer[range.upperBound...])

                            if line.hasPrefix("data: ") {
                                let jsonString = String(line.dropFirst(6))

                                // Check for stream end
                                if jsonString == "[DONE]" {
                                    continuation.finish()
                                    return
                                }

                                // Parse JSON event
                                if let data = jsonString.data(using: .utf8) {
                                    let decoder = JSONDecoder()
                                    do {
                                        let event = try decoder.decode(PlanStreamEvent.self, from: data)
                                        continuation.yield(event)
                                    } catch {
                                        // Log decode error but continue streaming
                                        print("SSE decode error: \(error), JSON: \(jsonString.prefix(200))")
                                    }
                                }
                            }
                        }
                    }

                    continuation.finish()

                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Evaluate a trading plan
    func evaluateTradingPlan(symbol: String, userId: String = "default") async throws -> EvaluationResponse {
        var components = URLComponents(string: "\(baseURL)/chat/\(symbol.uppercased())/evaluate")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        struct EmptyBody: Encodable {}

        var request = URLRequest(url: components.url!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONEncoder().encode(EmptyBody())

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }

        let decoder = JSONDecoder()
        return try decoder.decode(EvaluationResponse.self, from: data)
    }

    // MARK: - Position Tracking

    /// Get position for a symbol
    func getPosition(symbol: String, userId: String = "default") async throws -> Position? {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        do {
            return try await fetch(url: components.url!)
        } catch APIServiceError.httpError(404) {
            return nil
        }
    }

    /// Get position with live P&L calculated
    func getPositionWithPnl(symbol: String, userId: String = "default") async throws -> Position? {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())/pnl")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        do {
            return try await fetch(url: components.url!)
        } catch APIServiceError.httpError(404) {
            return nil
        }
    }

    /// Create a new position (watching status)
    func createPosition(
        symbol: String,
        stopLoss: Double,
        tradeType: String = "swing",
        target1: Double? = nil,
        target2: Double? = nil,
        target3: Double? = nil,
        notes: String? = nil,
        userId: String = "default"
    ) async throws -> Position {
        var components = URLComponents(string: "\(baseURL)/positions")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        let body = CreatePositionRequest(
            symbol: symbol.uppercased(),
            stopLoss: stopLoss,
            tradeType: tradeType,
            target1: target1,
            target2: target2,
            target3: target3,
            notes: notes
        )

        return try await post(url: components.url!, body: body)
    }

    /// Add an entry to a position (scale in or initial entry)
    func addPositionEntry(
        symbol: String,
        price: Double,
        shares: Int,
        date: String? = nil,
        userId: String = "default"
    ) async throws -> Position {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())/entries")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        let body = AddEntryRequest(price: price, shares: shares, date: date)
        return try await post(url: components.url!, body: body)
    }

    /// Add an exit from a position (partial or full)
    func addPositionExit(
        symbol: String,
        price: Double,
        shares: Int,
        reason: String = "manual",
        date: String? = nil,
        userId: String = "default"
    ) async throws -> Position {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())/exits")!
        components.queryItems = [URLQueryItem(name: "user_id", value: userId)]

        let body = AddExitRequest(price: price, shares: shares, reason: reason, date: date)
        return try await post(url: components.url!, body: body)
    }

    /// Delete a position
    func deletePosition(symbol: String, reason: String = "manual", userId: String = "default") async throws {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())")!
        components.queryItems = [
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "reason", value: reason)
        ]

        var request = URLRequest(url: components.url!)
        request.httpMethod = "DELETE"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (_, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }
    }

    // MARK: - Private Helpers

    private func fetch<T: Decodable>(url: URL) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorResponse = try? JSONDecoder().decode(APIError.self, from: data) {
                throw APIServiceError.serverError(errorResponse.detail)
            }
            throw APIServiceError.httpError(httpResponse.statusCode)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = makeISO8601DateDecodingStrategy()
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            print("Decoding error: \(error)")
            print("Raw JSON: \(String(data: data, encoding: .utf8) ?? "nil")")
            throw APIServiceError.decodingError(error)
        }
    }

    private func post<T: Decodable, B: Encodable>(url: URL, body: B) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let encoder = JSONEncoder()
        request.httpBody = try encoder.encode(body)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorResponse = try? JSONDecoder().decode(APIError.self, from: data) {
                throw APIServiceError.serverError(errorResponse.detail)
            }
            throw APIServiceError.httpError(httpResponse.statusCode)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = makeISO8601DateDecodingStrategy()
        return try decoder.decode(T.self, from: data)
    }

    private func delete(url: URL) async throws {
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorResponse = try? JSONDecoder().decode(APIError.self, from: data) {
                throw APIServiceError.serverError(errorResponse.detail)
            }
            throw APIServiceError.httpError(httpResponse.statusCode)
        }
    }
}

// MARK: - Error Types

enum APIServiceError: LocalizedError {
    case invalidResponse
    case httpError(Int)
    case serverError(String)
    case decodingError(Error)
    case networkError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "Server returned error code \(code)"
        case .serverError(let message):
            return message
        case .decodingError(let error):
            return "Failed to parse response: \(error.localizedDescription)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}
