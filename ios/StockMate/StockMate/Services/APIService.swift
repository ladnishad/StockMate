import Foundation

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
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            print("Decoding error: \(error)")
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
        return try decoder.decode(T.self, from: data)
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
