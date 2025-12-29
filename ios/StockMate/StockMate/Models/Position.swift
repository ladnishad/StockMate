import Foundation

// MARK: - Position Status

enum PositionStatus: String, Codable {
    case watching = "watching"
    case entered = "entered"
    case partial = "partial"
    case stoppedOut = "stopped_out"
    case closed = "closed"

    var displayName: String {
        switch self {
        case .watching: return "Watching"
        case .entered: return "Entered"
        case .partial: return "Partial"
        case .stoppedOut: return "Stopped Out"
        case .closed: return "Closed"
        }
    }

    var color: String {
        switch self {
        case .watching: return "gray"
        case .entered: return "blue"
        case .partial: return "orange"
        case .stoppedOut: return "red"
        case .closed: return "green"
        }
    }
}

// MARK: - Position Entry

struct PositionEntry: Codable, Identifiable, Equatable {
    var id: String { "\(price)-\(shares)-\(date)" }
    let price: Double
    let shares: Int
    let date: String

    var formattedPrice: String {
        String(format: "$%.2f", price)
    }

    var formattedDate: String {
        // Parse ISO date and format nicely
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let parsedDate = formatter.date(from: date) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateStyle = .short
            return displayFormatter.string(from: parsedDate)
        }
        return date
    }
}

// MARK: - Position Exit

struct PositionExit: Codable, Identifiable, Equatable {
    var id: String { "\(price)-\(shares)-\(date)" }
    let price: Double
    let shares: Int
    let date: String
    let reason: String

    var formattedPrice: String {
        String(format: "$%.2f", price)
    }

    var reasonDisplay: String {
        switch reason {
        case "target_1": return "Target 1"
        case "target_2": return "Target 2"
        case "target_3": return "Target 3"
        case "stop_loss": return "Stop Loss"
        case "manual": return "Manual"
        default: return reason.capitalized
        }
    }
}

// MARK: - Position

struct Position: Codable, Identifiable, Equatable {
    let id: String
    let userId: String
    let symbol: String
    let status: PositionStatus

    // Legacy single entry (backwards compatibility)
    let entryPrice: Double?
    let entryDate: String?

    // Multiple entries support
    let entries: [PositionEntry]
    let avgEntryPrice: Double?

    // Multiple exits support
    let exits: [PositionExit]
    let avgExitPrice: Double?

    // Size tracking
    let currentSize: Int
    let originalSize: Int

    // Risk management
    let stopLoss: Double?
    let target1: Double?
    let target2: Double?
    let target3: Double?
    let targetsHit: [Int]

    // P&L tracking
    let costBasis: Double?
    let realizedPnl: Double?
    let realizedPnlPct: Double?
    let unrealizedPnl: Double?
    let unrealizedPnlPct: Double?

    let tradeType: String
    let notes: String?
    let createdAt: String
    let updatedAt: String

    // Current price (populated by /pnl endpoint)
    let currentPrice: Double?

    enum CodingKeys: String, CodingKey {
        case id, symbol, status, entries, exits, notes
        case userId = "user_id"
        case entryPrice = "entry_price"
        case entryDate = "entry_date"
        case avgEntryPrice = "avg_entry_price"
        case avgExitPrice = "avg_exit_price"
        case currentSize = "current_size"
        case originalSize = "original_size"
        case stopLoss = "stop_loss"
        case target1 = "target_1"
        case target2 = "target_2"
        case target3 = "target_3"
        case targetsHit = "targets_hit"
        case costBasis = "cost_basis"
        case realizedPnl = "realized_pnl"
        case realizedPnlPct = "realized_pnl_pct"
        case unrealizedPnl = "unrealized_pnl"
        case unrealizedPnlPct = "unrealized_pnl_pct"
        case tradeType = "trade_type"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case currentPrice = "current_price"
    }

    // MARK: - Computed Properties

    var hasEntries: Bool { !entries.isEmpty }
    var hasExits: Bool { !exits.isEmpty }
    var hasPosition: Bool { currentSize > 0 }

    var totalPnl: Double? {
        guard let realized = realizedPnl else {
            return unrealizedPnl
        }
        return realized + (unrealizedPnl ?? 0)
    }

    var totalPnlPct: Double? {
        guard let totalCost = costBasis, totalCost > 0, let total = totalPnl else {
            return unrealizedPnlPct
        }
        return (total / totalCost) * 100
    }

    var isProfit: Bool {
        (totalPnl ?? 0) >= 0
    }

    var avgEntryFormatted: String {
        guard let avg = avgEntryPrice else { return "—" }
        return String(format: "$%.2f", avg)
    }

    var currentValueFormatted: String {
        guard let price = currentPrice else { return "—" }
        return String(format: "$%.2f", price * Double(currentSize))
    }

    var costBasisFormatted: String {
        guard let cost = costBasis else { return "—" }
        return String(format: "$%.2f", cost)
    }

    var unrealizedPnlFormatted: String {
        guard let pnl = unrealizedPnl else { return "—" }
        let sign = pnl >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, pnl)
    }

    var unrealizedPnlPctFormatted: String {
        guard let pct = unrealizedPnlPct else { return "" }
        let sign = pct >= 0 ? "+" : ""
        return String(format: "%@%.2f%%", sign, pct)
    }

    var realizedPnlFormatted: String {
        guard let pnl = realizedPnl else { return "$0.00" }
        let sign = pnl >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, pnl)
    }

    var realizedPnlPctFormatted: String {
        guard let pct = realizedPnlPct else { return "" }
        let sign = pct >= 0 ? "+" : ""
        return String(format: "%@%.2f%%", sign, pct)
    }

    var stopLossFormatted: String {
        guard let stop = stopLoss else { return "—" }
        return String(format: "$%.2f", stop)
    }

    var stopLossDistance: Double? {
        guard let entry = avgEntryPrice, entry > 0, let stop = stopLoss else { return nil }
        return ((stop - entry) / entry) * 100
    }

    var stopLossDistanceFormatted: String {
        guard let distance = stopLossDistance else { return "" }
        return String(format: "%.1f%%", distance)
    }

    var targets: [(number: Int, price: Double, hit: Bool)] {
        var result: [(number: Int, price: Double, hit: Bool)] = []
        if let t1 = target1 {
            result.append((1, t1, targetsHit.contains(1)))
        }
        if let t2 = target2 {
            result.append((2, t2, targetsHit.contains(2)))
        }
        if let t3 = target3 {
            result.append((3, t3, targetsHit.contains(3)))
        }
        return result
    }

    var tradeStyleDisplay: String {
        switch tradeType.lowercased() {
        case "day": return "Day Trade"
        case "swing": return "Swing Trade"
        case "long", "position": return "Position Trade"
        default: return tradeType.capitalized
        }
    }
}

// MARK: - Request/Response Models

struct CreatePositionRequest: Codable {
    let symbol: String
    let tradeType: String
    let stopLoss: Double?
    let target1: Double?
    let target2: Double?
    let target3: Double?
    let notes: String?

    enum CodingKeys: String, CodingKey {
        case symbol
        case tradeType = "trade_type"
        case stopLoss = "stop_loss"
        case target1 = "target_1"
        case target2 = "target_2"
        case target3 = "target_3"
        case notes
    }
}

struct AddEntryRequest: Codable {
    let price: Double
    let shares: Int
    let date: String?
}

struct AddExitRequest: Codable {
    let price: Double
    let shares: Int
    let reason: String
    let date: String?
}
