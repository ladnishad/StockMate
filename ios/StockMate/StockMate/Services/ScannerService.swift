import Foundation

/// Manages scanner operations with backend sync
actor ScannerService {
    static let shared = ScannerService()

    private let baseURL = AppConfiguration.apiBaseURL
    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
    }

    // MARK: - Scanner Operations

    /// Get all scanner results for all trading styles
    func getAllResults() async throws -> AllScannersResponse {
        let url = URL(string: "\(baseURL)/scanner")!
        return try await fetchJSON(url: url)
    }

    /// Get scanner results for a specific trading style
    func getResults(for style: TradingStyle) async throws -> ScannerResponse {
        let url = URL(string: "\(baseURL)/scanner/\(style.rawValue)")!
        return try await fetchJSON(url: url)
    }

    /// Trigger an on-demand scan refresh
    func refresh() async throws -> AllScannersResponse {
        let url = URL(string: "\(baseURL)/scanner/refresh")!
        return try await postJSON(url: url, body: EmptyBody())
    }

    /// Get scanner status (last scan time, next scheduled, etc.)
    func getStatus() async throws -> ScannerStatusResponse {
        let url = URL(string: "\(baseURL)/scanner/status")!
        return try await fetchJSON(url: url)
    }

    /// Add a scanned stock to watchlist with scanner metadata
    func addToWatchlist(
        symbol: String,
        scannerSource: String,
        scannerReason: String
    ) async throws -> WatchlistItem {
        let url = URL(string: "\(baseURL)/scanner/\(symbol.uppercased())/add")!

        let body = AddFromScannerBody(
            scannerSource: scannerSource,
            scannerReason: scannerReason
        )

        return try await postJSON(url: url, body: body)
    }

    // MARK: - Private Helpers

    private struct EmptyBody: Encodable {}

    private struct AddFromScannerBody: Encodable {
        let scannerSource: String
        let scannerReason: String

        enum CodingKeys: String, CodingKey {
            case scannerSource = "scanner_source"
            case scannerReason = "scanner_reason"
        }
    }

    private func fetchJSON<T: Decodable>(url: URL) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        if let token = KeychainHelper.shared.accessToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw ScannerServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw ScannerServiceError.httpError(httpResponse.statusCode)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = makeDateDecodingStrategy()

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            print("Scanner decoding error: \(error)")
            throw ScannerServiceError.decodingError(error)
        }
    }

    private func postJSON<T: Decodable, B: Encodable>(url: URL, body: B) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        if let token = KeychainHelper.shared.accessToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw ScannerServiceError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw ScannerServiceError.httpError(httpResponse.statusCode)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = makeDateDecodingStrategy()

        return try decoder.decode(T.self, from: data)
    }

    private func makeDateDecodingStrategy() -> JSONDecoder.DateDecodingStrategy {
        .custom { decoder in
            let container = try decoder.singleValueContainer()
            let dateString = try container.decode(String.self)

            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = formatter.date(from: dateString) {
                return date
            }

            formatter.formatOptions = [.withInternetDateTime]
            if let date = formatter.date(from: dateString) {
                return date
            }

            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Cannot decode date: \(dateString)"
            )
        }
    }
}

// MARK: - Scanner Service Errors

enum ScannerServiceError: LocalizedError {
    case invalidResponse
    case httpError(Int)
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "Server returned error code \(code)"
        case .decodingError(let error):
            return "Failed to parse response: \(error.localizedDescription)"
        }
    }
}
