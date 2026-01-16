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

    private let baseURL = AppConfiguration.apiBaseURL
    private let session: URLSession

    /// Access keychain lazily to avoid I/O on main thread during init
    private var keychain: KeychainHelper {
        KeychainHelper.shared
    }

    /// Get the current authenticated user's ID, or fallback to "default" if not authenticated
    private var currentUserId: String {
        keychain.userId ?? "default"
    }

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
    }

    // MARK: - Auth Header Helper

    private func addAuthHeader(to request: inout URLRequest) {
        if let token = keychain.accessToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }

    // MARK: - Token Refresh Coordination

    /// Actor-isolated state for token refresh coordination
    /// Using instance properties instead of static to leverage actor isolation
    private var isRefreshing = false
    private var refreshTask: Task<Void, Error>?
    private var pendingRefreshContinuations: [CheckedContinuation<Void, Error>] = []

    /// Coordinate token refresh to avoid race conditions
    /// Multiple concurrent 401 responses will wait for a single refresh operation
    private func coordinatedRefresh() async throws {
        if isRefreshing {
            // Another refresh is in progress - wait for it
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                pendingRefreshContinuations.append(continuation)
            }
            return
        }

        // Start the refresh
        isRefreshing = true

        do {
            try await AuthenticationManager.shared.refreshAccessToken()

            // Resume all waiting requests
            let continuations = pendingRefreshContinuations
            pendingRefreshContinuations = []
            isRefreshing = false

            for continuation in continuations {
                continuation.resume()
            }
        } catch {
            // Resume all waiting requests with error
            let continuations = pendingRefreshContinuations
            pendingRefreshContinuations = []
            isRefreshing = false

            for continuation in continuations {
                continuation.resume(throwing: error)
            }
            throw error
        }
    }

    /// Attempt to refresh token and retry the request on 401 error
    private func handleUnauthorizedAndRetry<T: Decodable>(
        request: URLRequest,
        decoder: JSONDecoder
    ) async throws -> T {
        // Coordinate refresh to avoid race conditions
        try await coordinatedRefresh()

        // Retry the request with new token
        var retryRequest = request
        addAuthHeader(to: &retryRequest)

        let (data, response) = try await session.data(for: retryRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        // If we still get 401 after refresh, the refresh token is also invalid
        if httpResponse.statusCode == 401 {
            // Don't retry again - logout will be handled by AuthenticationManager
            throw APIServiceError.httpError(401)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorResponse = try? decoder.decode(APIError.self, from: data) {
                throw APIServiceError.serverError(errorResponse.detail)
            }
            throw APIServiceError.httpError(httpResponse.statusCode)
        }

        return try decoder.decode(T.self, from: data)
    }

    // MARK: - Market Data

    /// Fetch quick market overview (indices)
    func fetchMarketQuick() async throws -> [MarketIndex] {
        let response = try await fetchMarketQuickFull()
        return response.marketIndices
    }

    /// Fetch quick market data with full response including market status
    func fetchMarketQuickFull() async throws -> MarketQuickResponse {
        let url = URL(string: "\(baseURL)/market/quick")!
        return try await fetch(url: url)
    }

    /// Fetch full market overview with signals
    func fetchMarketOverview(daysBack: Int = 30) async throws -> MarketQuickResponse {
        var components = URLComponents(string: "\(baseURL)/market")!
        components.queryItems = [URLQueryItem(name: "days_back", value: "\(daysBack)")]
        let response: MarketQuickResponse = try await fetch(url: components.url!)
        return response
    }

    // MARK: - Watchlist / Market Scan

    /// Fetch smart watchlist based on market scan
    /// The agent automatically determines optimal trade style for each stock
    func fetchWatchlist(minScore: Int = 65, topSectors: Int = 3, stocksPerSector: Int = 5) async throws -> [Stock] {
        var components = URLComponents(string: "\(baseURL)/market/scan")!
        components.queryItems = [
            URLQueryItem(name: "min_stock_score", value: "\(minScore)"),
            URLQueryItem(name: "top_sectors", value: "\(topSectors)"),
            URLQueryItem(name: "stocks_per_sector", value: "\(stocksPerSector)")
        ]

        let response: MarketScanResponse = try await fetch(url: components.url!)
        return response.allStocks
    }

    /// Fetch full market scan data
    /// The agent automatically determines optimal trade style for each stock
    func fetchMarketScan(minScore: Int = 65) async throws -> MarketScanResponse {
        var components = URLComponents(string: "\(baseURL)/market/scan")!
        components.queryItems = [
            URLQueryItem(name: "min_stock_score", value: "\(minScore)"),
            URLQueryItem(name: "top_sectors", value: "3"),
            URLQueryItem(name: "stocks_per_sector", value: "5")
        ]

        return try await fetch(url: components.url!)
    }

    // MARK: - Stock Analysis

    /// Analyze a specific stock - agent determines optimal trade style
    func analyzeStock(symbol: String, accountSize: Double = 10000) async throws -> AnalysisResponse {
        let url = URL(string: "\(baseURL)/analyze")!

        struct AnalysisRequest: Encodable {
            let symbol: String
            let account_size: Double
            let use_ai: Bool
        }

        let request = AnalysisRequest(
            symbol: symbol.uppercased(),
            account_size: accountSize,
            use_ai: false
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
        addAuthHeader(to: &request)

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

    // MARK: - Real-Time Data

    /// Fetch real-time quote for a symbol
    func fetchQuote(symbol: String) async throws -> [String: Any] {
        let url = URL(string: "\(baseURL)/quote/\(symbol.uppercased())")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        addAuthHeader(to: &request)
        let (data, _) = try await session.data(for: request)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return json ?? [:]
    }

    // MARK: - User Watchlist

    /// Fetch user's watchlist with live prices
    func fetchUserWatchlist() async throws -> [WatchlistItem] {
        var components = URLComponents(string: "\(baseURL)/watchlist")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]
        let response: WatchlistResponse = try await fetch(url: components.url!)
        return response.items
    }

    /// Add a symbol to user's watchlist
    func addToWatchlist(symbol: String) async throws -> WatchlistItem {
        var components = URLComponents(string: "\(baseURL)/watchlist/\(symbol.uppercased())")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

        struct EmptyBody: Encodable {}
        return try await post(url: components.url!, body: EmptyBody())
    }

    /// Remove a symbol from user's watchlist
    func removeFromWatchlist(symbol: String) async throws {
        var components = URLComponents(string: "\(baseURL)/watchlist/\(symbol.uppercased())")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]
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
    func sendChatMessage(symbol: String, message: String) async throws -> ChatResponse {
        let url = URL(string: "\(baseURL)/chat/\(symbol.uppercased())")!

        struct ChatRequest: Encodable {
            let message: String
            let user_id: String
        }

        let request = ChatRequest(message: message, user_id: currentUserId)
        return try await post(url: url, body: request)
    }

    /// Send a portfolio chat message (home page chat)
    /// Connects to the Portfolio Agent which can analyze all watchlist stocks
    func sendPortfolioChatMessage(message: String) async throws -> PortfolioChatResponse {
        var components = URLComponents(string: "\(baseURL)/chat")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

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
    ///   - limit: Maximum number of messages to return
    /// - Returns: Chat history response
    func getChatHistory(symbol: String? = nil, limit: Int = 50) async throws -> ChatHistoryResponse {
        var components = URLComponents(string: "\(baseURL)/chat/history")!
        var queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]
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
    func clearChatHistory(symbol: String? = nil) async throws {
        var components = URLComponents(string: "\(baseURL)/chat/history")!
        var queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]
        if let symbol = symbol {
            queryItems.append(URLQueryItem(name: "symbol", value: symbol.uppercased()))
        }
        components.queryItems = queryItems

        try await delete(url: components.url!)
    }

    // MARK: - Trading Plan

    /// Get trading plan for a symbol
    func getTradingPlan(symbol: String) async throws -> TradingPlanResponse? {
        var components = URLComponents(string: "\(baseURL)/chat/\(symbol.uppercased())/plan")!
        components.queryItems = [
            URLQueryItem(name: "user_id", value: currentUserId),
            URLQueryItem(name: "force_new", value: "false")
        ]

        do {
            return try await fetch(url: components.url!)
        } catch APIServiceError.httpError(404) {
            return nil
        }
    }

    /// Generate a new trading plan
    func generateTradingPlan(symbol: String, forceNew: Bool = false) async throws -> TradingPlanResponse {
        var components = URLComponents(string: "\(baseURL)/chat/\(symbol.uppercased())/plan")!
        components.queryItems = [
            URLQueryItem(name: "user_id", value: currentUserId),
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

        // Analysis step events (type == "step")
        let stepType: String?
        let status: String?
        let findings: [String]?
        let timestamp: Double?

        // V2: Orchestrator step events (type == "orchestrator_step")
        let stepStatus: String?
        let stepFindings: [String]?

        // V2: Sub-agent progress (type == "subagent_progress")
        let subagents: [String: SubAgentProgressData]?

        // V2: Sub-agent completion (type == "subagent_complete")
        let agentName: String?
        let agentFindings: [String]?

        // V2: Final result metadata
        let selectedStyle: String?
        let selectionReasoning: String?
        let alternatives: [[String: AnyCodable]]?
        let analysisId: String?  // ID of the saved analysis (for approval)

        // V2: Error event (type == "error")
        let errorMessage: String?

        // Agentic mode events
        let thinking: String?                    // agent_thinking event: AI's reasoning
        let toolName: String?                    // tool_call/tool_result: name of tool
        let toolArguments: [String: AnyCodable]? // tool_call: arguments passed to tool
        let toolResult: [String: AnyCodable]?    // tool_result: result from tool
        let iteration: Int?                      // Agentic events: which iteration

        // Agentic mode final result - raw plan dict (different structure than TradingPlanResponse)
        let agenticPlan: [String: AnyCodable]?   // Raw agentic plan with day/swing/position_trade_plan
        let agenticData: [String: AnyCodable]?   // Additional agentic data (iterations, tool_history, etc.)

        enum CodingKeys: String, CodingKey {
            case type, phase, content, plan, message
            case stepType = "step_type"
            case status, findings, timestamp
            case stepStatus = "step_status"
            case stepFindings = "step_findings"
            case subagents
            case agentName = "agent_name"
            case agentFindings = "agent_findings"
            case selectedStyle = "selected_style"
            case selectionReasoning = "selection_reasoning"
            case alternatives
            case analysisId = "analysis_id"
            case errorMessage = "error_message"
            // Agentic mode
            case thinking
            case toolName = "tool_name"
            case toolArguments = "tool_arguments"
            case toolResult = "tool_result"
            case iteration
            case agenticPlan = "agentic_plan"  // Agentic final result plan
            case agenticData = "data"          // Agentic metadata (iterations, tool_history)
        }

        /// Get the error message from either V1 (message) or V2 (error_message) format
        var effectiveErrorMessage: String? {
            errorMessage ?? message
        }
    }

    /// Sub-agent progress data from the streaming API
    struct SubAgentProgressData: Decodable {
        let agentName: String
        let displayName: String
        let status: String
        let currentStep: String?
        let stepsCompleted: [String]
        let findings: [String]
        let elapsedMs: Int
        let errorMessage: String?

        enum CodingKeys: String, CodingKey {
            case agentName = "agent_name"
            case displayName = "display_name"
            case status
            case currentStep = "current_step"
            case stepsCompleted = "steps_completed"
            case findings
            case elapsedMs = "elapsed_ms"
            case errorMessage = "error_message"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            agentName = try container.decode(String.self, forKey: .agentName)
            displayName = try container.decodeIfPresent(String.self, forKey: .displayName) ?? agentName
            status = try container.decode(String.self, forKey: .status)
            currentStep = try container.decodeIfPresent(String.self, forKey: .currentStep)
            stepsCompleted = try container.decodeIfPresent([String].self, forKey: .stepsCompleted) ?? []
            findings = try container.decodeIfPresent([String].self, forKey: .findings) ?? []
            elapsedMs = try container.decodeIfPresent(Int.self, forKey: .elapsedMs) ?? 0
            errorMessage = try container.decodeIfPresent(String.self, forKey: .errorMessage)
        }
    }

    /// Helper for decoding arbitrary JSON values
    struct AnyCodable: Decodable {
        let value: Any

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let string = try? container.decode(String.self) {
                value = string
            } else if let int = try? container.decode(Int.self) {
                value = int
            } else if let double = try? container.decode(Double.self) {
                value = double
            } else if let bool = try? container.decode(Bool.self) {
                value = bool
            } else if let array = try? container.decode([AnyCodable].self) {
                value = array.map { $0.value }
            } else if let dict = try? container.decode([String: AnyCodable].self) {
                value = dict.mapValues { $0.value }
            } else {
                value = NSNull()
            }
        }
    }

    /// Generate trading plan with real-time streaming
    /// Returns an AsyncThrowingStream of events as the AI generates the plan
    nonisolated func generateTradingPlanStream(
        symbol: String,
        forceNew: Bool = true
    ) -> AsyncThrowingStream<PlanStreamEvent, Error> {
        let keychain = KeychainHelper.shared
        let userId = keychain.userId ?? "default"
        return AsyncThrowingStream { continuation in
            Task {
                // V2 endpoint with agentic mode (iterative AI tool-calling)
                var components = URLComponents(string: "\(AppConfiguration.apiBaseURL)/plan/\(symbol.uppercased())/generate/v2")!
                components.queryItems = [
                    URLQueryItem(name: "user_id", value: userId),
                    URLQueryItem(name: "force_new", value: String(forceNew)),
                    URLQueryItem(name: "agentic_mode", value: "true")  // Enable agentic AI mode
                ]

                var request = URLRequest(url: components.url!)
                request.httpMethod = "POST"
                request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let token = keychain.accessToken {
                    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                }
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
    func evaluateTradingPlan(symbol: String) async throws -> EvaluationResponse {
        var components = URLComponents(string: "\(baseURL)/chat/\(symbol.uppercased())/evaluate")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

        struct EmptyBody: Encodable {}

        var request = URLRequest(url: components.url!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(EmptyBody())

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }

        let decoder = JSONDecoder()
        return try decoder.decode(EvaluationResponse.self, from: data)
    }

    // MARK: - Interactive Plan Sessions

    /// Start a new interactive planning session
    func startPlanSession(symbol: String) async throws -> PlanSessionResponse {
        let url = URL(string: "\(baseURL)/plan/\(symbol.uppercased())/session")!

        struct EmptyBody: Encodable {}

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(EmptyBody())

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }

        return try JSONDecoder().decode(PlanSessionResponse.self, from: data)
    }

    /// Get plan session state
    func getPlanSession(symbol: String, sessionId: String) async throws -> PlanSessionResponse {
        let url = URL(string: "\(baseURL)/plan/\(symbol.uppercased())/session/\(sessionId)")!

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }

        return try JSONDecoder().decode(PlanSessionResponse.self, from: data)
    }

    /// Submit feedback (question or adjustment) on a draft plan
    func submitPlanFeedback(
        symbol: String,
        sessionId: String,
        feedbackType: String,
        content: String
    ) async throws -> PlanFeedbackResponse {
        let url = URL(string: "\(baseURL)/plan/\(symbol.uppercased())/session/\(sessionId)/feedback")!

        struct FeedbackRequest: Encodable {
            let feedbackType: String
            let content: String

            enum CodingKeys: String, CodingKey {
                case feedbackType = "feedback_type"
                case content
            }
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(FeedbackRequest(feedbackType: feedbackType, content: content))

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }

        return try JSONDecoder().decode(PlanFeedbackResponse.self, from: data)
    }

    /// Approve the draft plan
    func approvePlanSession(symbol: String, sessionId: String) async throws -> TradingPlanResponse {
        let url = URL(string: "\(baseURL)/plan/\(symbol.uppercased())/session/\(sessionId)/approve")!

        struct EmptyBody: Encodable {}

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(EmptyBody())

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }

        return try JSONDecoder().decode(TradingPlanResponse.self, from: data)
    }

    /// Approve a V2 analysis and create a trading plan from it
    func approveV2Analysis(symbol: String, analysisId: String) async throws -> TradingPlanResponse {
        let url = URL(string: "\(baseURL)/plan/\(symbol.uppercased())/analysis/\(analysisId)/approve")!

        struct EmptyBody: Encodable {}

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(EmptyBody())

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        // Handle 401 - token expired, try refresh
        if httpResponse.statusCode == 401 {
            return try await handleUnauthorizedAndRetry(request: request, decoder: JSONDecoder())
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }

        return try JSONDecoder().decode(TradingPlanResponse.self, from: data)
    }

    /// Start a session from an existing plan (for modifying existing plans)
    func startSessionFromExisting(symbol: String) async throws -> PlanSessionResponse {
        let url = URL(string: "\(baseURL)/plan/\(symbol.uppercased())/session/from-existing")!

        struct EmptyBody: Encodable {}

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(EmptyBody())

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            if let httpResponse = response as? HTTPURLResponse {
                throw APIServiceError.httpError(httpResponse.statusCode)
            }
            throw APIServiceError.invalidResponse
        }

        return try JSONDecoder().decode(PlanSessionResponse.self, from: data)
    }

    /// Reopen an approved session to continue making adjustments
    func reopenPlanSession(symbol: String, sessionId: String) async throws -> PlanSessionResponse {
        let url = URL(string: "\(baseURL)/plan/\(symbol.uppercased())/session/\(sessionId)/reopen")!

        struct EmptyBody: Encodable {}

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(EmptyBody())

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            if let httpResponse = response as? HTTPURLResponse {
                throw APIServiceError.httpError(httpResponse.statusCode)
            }
            throw APIServiceError.invalidResponse
        }

        return try JSONDecoder().decode(PlanSessionResponse.self, from: data)
    }

    // MARK: - Position Tracking

    /// Get position for a symbol
    func getPosition(symbol: String) async throws -> Position? {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

        do {
            return try await fetch(url: components.url!)
        } catch APIServiceError.httpError(404) {
            return nil
        }
    }

    /// Get position with live P&L calculated
    func getPositionWithPnl(symbol: String) async throws -> Position? {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())/pnl")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

        do {
            return try await fetch(url: components.url!)
        } catch APIServiceError.httpError(404) {
            return nil
        }
    }

    /// Create a new position (watching status)
    func createPosition(
        symbol: String,
        tradeType: String = "swing",
        stopLoss: Double? = nil,
        target1: Double? = nil,
        target2: Double? = nil,
        target3: Double? = nil,
        notes: String? = nil
    ) async throws -> Position {
        var components = URLComponents(string: "\(baseURL)/positions")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

        let body = CreatePositionRequest(
            symbol: symbol.uppercased(),
            tradeType: tradeType,
            stopLoss: stopLoss,
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
        date: String? = nil
    ) async throws -> Position {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())/entries")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

        let body = AddEntryRequest(price: price, shares: shares, date: date)
        return try await post(url: components.url!, body: body)
    }

    /// Add an exit from a position (partial or full)
    func addPositionExit(
        symbol: String,
        price: Double,
        shares: Int,
        reason: String = "manual",
        date: String? = nil
    ) async throws -> Position {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())/exits")!
        components.queryItems = [URLQueryItem(name: "user_id", value: currentUserId)]

        let body = AddExitRequest(price: price, shares: shares, reason: reason, date: date)
        return try await post(url: components.url!, body: body)
    }

    /// Delete a position
    func deletePosition(symbol: String, reason: String = "manual") async throws {
        var components = URLComponents(string: "\(baseURL)/positions/\(symbol.uppercased())")!
        components.queryItems = [
            URLQueryItem(name: "user_id", value: currentUserId),
            URLQueryItem(name: "reason", value: reason)
        ]

        var request = URLRequest(url: components.url!)
        request.httpMethod = "DELETE"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)

        let (_, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIServiceError.invalidResponse
        }
    }

    // MARK: - Settings

    struct UserSettingsResponse: Decodable {
        let modelProvider: String
        let availableProviders: [String]

        enum CodingKeys: String, CodingKey {
            case modelProvider = "model_provider"
            case availableProviders = "available_providers"
        }
    }

    func getUserSettings() async throws -> UserSettingsResponse {
        let url = URL(string: "\(baseURL)/settings")!
        return try await fetch(url: url)
    }

    func updateProvider(provider: String) async throws {
        let url = URL(string: "\(baseURL)/settings/provider")!

        struct UpdateRequest: Encodable {
            let provider: String
        }

        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeader(to: &request)
        request.httpBody = try JSONEncoder().encode(UpdateRequest(provider: provider))

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
        addAuthHeader(to: &request)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = makeISO8601DateDecodingStrategy()

        // Handle 401 with automatic token refresh
        if httpResponse.statusCode == 401 {
            return try await handleUnauthorizedAndRetry(request: request, decoder: decoder)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorResponse = try? decoder.decode(APIError.self, from: data) {
                throw APIServiceError.serverError(errorResponse.detail)
            }
            throw APIServiceError.httpError(httpResponse.statusCode)
        }

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
        addAuthHeader(to: &request)

        let encoder = JSONEncoder()
        request.httpBody = try encoder.encode(body)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = makeISO8601DateDecodingStrategy()

        // Handle 401 with automatic token refresh
        if httpResponse.statusCode == 401 {
            return try await handleUnauthorizedAndRetry(request: request, decoder: decoder)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorResponse = try? decoder.decode(APIError.self, from: data) {
                throw APIServiceError.serverError(errorResponse.detail)
            }
            throw APIServiceError.httpError(httpResponse.statusCode)
        }

        return try decoder.decode(T.self, from: data)
    }

    private func delete(url: URL) async throws {
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        addAuthHeader(to: &request)

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
