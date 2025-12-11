import Foundation

/// Represents a market index (SPY, QQQ, DIA, IWM)
struct MarketIndex: Identifiable, Codable, Equatable {
    var id: String { symbol }
    let symbol: String
    let name: String
    let price: Double
    let change: Double
    let changePct: Double
    let dayHigh: Double?
    let dayLow: Double?
    let volume: Int?

    var isUp: Bool { change >= 0 }

    var formattedPrice: String {
        String(format: "$%.2f", price)
    }

    var formattedChange: String {
        let sign = change >= 0 ? "+" : ""
        return String(format: "%@%.2f", sign, change)
    }

    var formattedChangePct: String {
        let sign = changePct >= 0 ? "+" : ""
        return String(format: "%@%.2f%%", sign, changePct)
    }

    // Coding keys for API response mapping
    enum CodingKeys: String, CodingKey {
        case symbol
        case name
        case price
        case change
        case changePct = "change_pct"
        case dayHigh = "day_high"
        case dayLow = "day_low"
        case volume = "day_volume"
    }

    // Sample data for previews
    static let sample = MarketIndex(
        symbol: "SPY",
        name: "S&P 500",
        price: 598.42,
        change: 4.23,
        changePct: 0.71,
        dayHigh: 599.50,
        dayLow: 594.20,
        volume: 45_000_000
    )

    static let samples: [MarketIndex] = [
        MarketIndex(symbol: "SPY", name: "S&P 500", price: 598.42, change: 4.23, changePct: 0.71, dayHigh: 599.50, dayLow: 594.20, volume: 45_000_000),
        MarketIndex(symbol: "QQQ", name: "Nasdaq 100", price: 518.73, change: -2.15, changePct: -0.41, dayHigh: 521.00, dayLow: 516.50, volume: 32_000_000),
        MarketIndex(symbol: "DIA", name: "Dow Jones", price: 438.91, change: 3.67, changePct: 0.84, dayHigh: 440.20, dayLow: 435.80, volume: 18_000_000),
        MarketIndex(symbol: "IWM", name: "Russell 2000", price: 234.56, change: 1.89, changePct: 0.81, dayHigh: 235.40, dayLow: 232.10, volume: 28_000_000)
    ]
}
