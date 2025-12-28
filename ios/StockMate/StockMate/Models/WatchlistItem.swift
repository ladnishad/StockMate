import Foundation

/// Represents a single item in the user's watchlist
struct WatchlistItem: Identifiable, Codable, Equatable {
    var id: String { symbol }

    let symbol: String
    let addedAt: Date
    var notes: String?
    var alertsEnabled: Bool

    // Live data (populated from API)
    var currentPrice: Double?
    var change: Double?
    var changePct: Double?
    var score: Double?
    var recommendation: String?

    // MARK: - Computed Properties

    var isUp: Bool {
        (change ?? 0) >= 0
    }

    var formattedPrice: String {
        guard let price = currentPrice else { return "--" }
        return String(format: "$%.2f", price)
    }

    var formattedChange: String {
        guard let change = change else { return "--" }
        let sign = change >= 0 ? "+" : ""
        return "\(sign)\(String(format: "%.2f", change))"
    }

    var formattedChangePct: String {
        guard let pct = changePct else { return "--" }
        let sign = pct >= 0 ? "+" : ""
        return "\(sign)\(String(format: "%.2f", pct))%"
    }

    var isRecommendedBuy: Bool {
        recommendation?.uppercased() == "BUY"
    }

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case symbol
        case addedAt = "added_at"
        case notes
        case alertsEnabled = "alerts_enabled"
        case currentPrice = "current_price"
        case change
        case changePct = "change_pct"
        case score
        case recommendation
    }

    // MARK: - Sample Data

    static let sample = WatchlistItem(
        symbol: "AAPL",
        addedAt: Date(),
        notes: "Strong fundamentals",
        alertsEnabled: true,
        currentPrice: 175.50,
        change: 2.35,
        changePct: 1.36,
        score: 78.5,
        recommendation: "BUY"
    )

    static let samples: [WatchlistItem] = [
        WatchlistItem(symbol: "AAPL", addedAt: Date(), notes: nil, alertsEnabled: false, currentPrice: 248.92, change: 3.45, changePct: 1.41, score: 76.3, recommendation: "BUY"),
        WatchlistItem(symbol: "NVDA", addedAt: Date().addingTimeInterval(-86400), notes: nil, alertsEnabled: true, currentPrice: 142.65, change: -2.10, changePct: -1.45, score: 82.5, recommendation: "BUY"),
        WatchlistItem(symbol: "MSFT", addedAt: Date().addingTimeInterval(-172800), notes: nil, alertsEnabled: false, currentPrice: 432.15, change: 5.20, changePct: 1.22, score: 71.8, recommendation: "BUY"),
        WatchlistItem(symbol: "META", addedAt: Date().addingTimeInterval(-259200), notes: nil, alertsEnabled: false, currentPrice: 612.30, change: -8.50, changePct: -1.37, score: 68.2, recommendation: "NO_BUY"),
    ]
}

/// Response from the watchlist API
struct WatchlistResponse: Codable {
    let userId: String
    let items: [WatchlistItem]
    let count: Int
    let lastUpdated: String

    enum CodingKeys: String, CodingKey {
        case userId = "user_id"
        case items
        case count
        case lastUpdated = "last_updated"
    }
}
