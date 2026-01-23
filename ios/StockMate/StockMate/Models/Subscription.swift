import Foundation

/// Represents a subscription tier
enum SubscriptionTier: String, Codable, CaseIterable {
    case base
    case premium
    case pro
    case unlimited

    /// Display name for the tier
    var displayName: String {
        switch self {
        case .base: return "Base"
        case .premium: return "Premium"
        case .pro: return "Pro"
        case .unlimited: return "Unlimited"
        }
    }

    /// Short description for the tier
    var shortDescription: String {
        switch self {
        case .base: return "Free"
        case .premium: return "$20/month"
        case .pro: return "$50/month"
        case .unlimited: return "$200/month"
        }
    }

    /// Icon name for the tier
    var iconName: String {
        switch self {
        case .base: return "star"
        case .premium: return "star.fill"
        case .pro: return "crown"
        case .unlimited: return "crown.fill"
        }
    }

    /// Accent color for the tier
    var accentColorName: String {
        switch self {
        case .base: return "gray"
        case .premium: return "blue"
        case .pro: return "purple"
        case .unlimited: return "gold"
        }
    }
}

/// Detailed information about a subscription tier
struct SubscriptionTierInfo: Codable, Equatable {
    let tier: String
    let name: String
    let description: String
    let pricePerMonth: Int
    let watchlistLimit: Int  // -1 for unlimited
    let multiModelAccess: Bool
    let features: [String]

    enum CodingKeys: String, CodingKey {
        case tier
        case name
        case description
        case pricePerMonth = "price_per_month"
        case watchlistLimit = "watchlist_limit"
        case multiModelAccess = "multi_model_access"
        case features
    }

    /// Whether this tier has unlimited watchlist
    var isUnlimited: Bool {
        watchlistLimit == -1
    }

    /// Display string for price
    var priceDisplay: String {
        if pricePerMonth == 0 {
            return "Free"
        }
        return "$\(pricePerMonth)/month"
    }

    /// Display string for watchlist limit
    var watchlistLimitDisplay: String {
        if watchlistLimit == -1 {
            return "Unlimited"
        }
        return "\(watchlistLimit) stocks"
    }
}

/// User's current subscription status
struct UserSubscription: Codable, Equatable {
    let tier: String
    let tierInfo: SubscriptionTierInfo
    let watchlistCount: Int
    let watchlistRemaining: Int  // -1 for unlimited
    let canAddToWatchlist: Bool

    enum CodingKeys: String, CodingKey {
        case tier
        case tierInfo = "tier_info"
        case watchlistCount = "watchlist_count"
        case watchlistRemaining = "watchlist_remaining"
        case canAddToWatchlist = "can_add_to_watchlist"
    }

    /// Get the subscription tier enum
    var subscriptionTier: SubscriptionTier {
        SubscriptionTier(rawValue: tier) ?? .base
    }

    /// Display string for remaining watchlist slots
    var remainingDisplay: String {
        if watchlistRemaining == -1 {
            return "Unlimited"
        }
        return "\(watchlistRemaining) remaining"
    }

    /// Display string for watchlist usage
    var usageDisplay: String {
        if tierInfo.isUnlimited {
            return "\(watchlistCount) stocks"
        }
        return "\(watchlistCount) of \(tierInfo.watchlistLimit)"
    }
}
