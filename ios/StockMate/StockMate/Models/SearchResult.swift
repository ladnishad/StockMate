import Foundation

/// Ticker search result from the API
struct SearchResult: Identifiable, Codable, Equatable {
    var id: String { symbol }

    let symbol: String
    let name: String
    let exchange: String
    let assetType: String

    var isETF: Bool {
        assetType.lowercased() == "etf"
    }

    enum CodingKeys: String, CodingKey {
        case symbol
        case name
        case exchange
        case assetType = "asset_type"
    }

    // MARK: - Sample Data

    static let samples: [SearchResult] = [
        SearchResult(symbol: "AAPL", name: "Apple Inc.", exchange: "NASDAQ", assetType: "stock"),
        SearchResult(symbol: "MSFT", name: "Microsoft Corporation", exchange: "NASDAQ", assetType: "stock"),
        SearchResult(symbol: "GOOGL", name: "Alphabet Inc.", exchange: "NASDAQ", assetType: "stock"),
        SearchResult(symbol: "AMZN", name: "Amazon.com Inc.", exchange: "NASDAQ", assetType: "stock"),
        SearchResult(symbol: "SPY", name: "SPDR S&P 500 ETF Trust", exchange: "NYSE", assetType: "etf"),
        SearchResult(symbol: "QQQ", name: "Invesco QQQ Trust", exchange: "NASDAQ", assetType: "etf"),
    ]
}
