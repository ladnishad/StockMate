import SwiftUI
import Combine

/// ViewModel for the Home screen managing market data and user watchlist
@MainActor
class HomeViewModel: ObservableObject {
    // MARK: - Published State

    @Published var indices: [MarketIndex] = []

    // Market status
    @Published var isMarketOpen: Bool = true
    @Published var marketNextEvent: String = ""
    @Published var marketNextEventType: String = "close"

    // User-managed watchlist (not auto-populated)
    @Published var watchlistItems: [WatchlistItem] = []
    @Published var marketDirection: MarketDirection = .mixed

    @Published var isLoadingIndices = false
    @Published var isLoadingWatchlist = false
    @Published var isRefreshing = false
    @Published var indicesError: String?
    @Published var watchlistError: String?
    @Published var error: String?

    @Published var lastUpdated: Date?

    // MARK: - Services

    private let watchlistService = WatchlistService.shared

    // MARK: - Computed Properties

    var isLoading: Bool {
        isLoadingIndices || isLoadingWatchlist
    }

    var hasError: Bool {
        indicesError != nil || watchlistError != nil || error != nil
    }

    var formattedLastUpdated: String {
        guard let lastUpdated else { return "Never" }
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: lastUpdated, relativeTo: Date())
    }

    var watchlistCount: Int {
        watchlistItems.count
    }

    var hasWatchlistItems: Bool {
        !watchlistItems.isEmpty
    }

    // MARK: - Initialization

    init() {
        // Load cached watchlist immediately for fast UI
        watchlistItems = watchlistService.getCachedItems()
    }

    // MARK: - Data Loading

    /// Load initial data on app launch
    func loadInitialData() async {
        await loadAllData()
    }

    /// Load all data (indices + watchlist)
    func loadAllData() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await self.loadMarketIndices() }
            group.addTask { await self.loadWatchlist() }
        }
        lastUpdated = Date()
    }

    /// Refresh all data
    func refresh() async {
        isRefreshing = true
        error = nil
        await loadAllData()
        isRefreshing = false
    }

    /// Dismiss current error
    func dismissError() {
        error = nil
        indicesError = nil
        watchlistError = nil
    }

    /// Load market indices
    func loadMarketIndices() async {
        isLoadingIndices = true
        indicesError = nil

        do {
            let response = try await APIService.shared.fetchMarketQuickFull()
            withAnimation(.easeInOut(duration: 0.3)) {
                self.indices = response.marketIndices
                self.updateMarketDirection()

                // Update market status
                if let status = response.marketStatus {
                    self.isMarketOpen = status.isOpen
                    self.marketNextEvent = status.nextEvent ?? ""
                    self.marketNextEventType = status.nextEventType
                }
            }
        } catch {
            indicesError = error.localizedDescription
            // Use sample data for development/preview
            if indices.isEmpty {
                indices = MarketIndex.samples
                updateMarketDirection()
            }
        }

        isLoadingIndices = false
    }

    /// Load user's watchlist
    func loadWatchlist() async {
        isLoadingWatchlist = true
        watchlistError = nil

        do {
            let items = try await watchlistService.getWatchlist()
            withAnimation(.easeInOut(duration: 0.3)) {
                self.watchlistItems = items
            }
        } catch {
            watchlistError = error.localizedDescription
            // Use sample data for development/preview
            if watchlistItems.isEmpty {
                watchlistItems = WatchlistItem.samples
            }
        }

        isLoadingWatchlist = false
    }

    // MARK: - Watchlist Management

    /// Add a symbol to the watchlist
    func addToWatchlist(_ symbol: String) async -> Bool {
        do {
            let item = try await watchlistService.addSymbol(symbol)
            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                // Add at the beginning
                watchlistItems.insert(item, at: 0)
            }

            // Haptic feedback
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)

            return true
        } catch {
            self.error = error.localizedDescription

            // Haptic feedback for error
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.error)

            return false
        }
    }

    /// Remove a symbol from the watchlist
    func removeFromWatchlist(_ symbol: String) async {
        // Optimistic removal
        let removedIndex = watchlistItems.firstIndex { $0.symbol == symbol }
        let removedItem = watchlistItems.first { $0.symbol == symbol }

        withAnimation(.easeOut(duration: 0.2)) {
            watchlistItems.removeAll { $0.symbol == symbol }
        }

        do {
            try await watchlistService.removeSymbol(symbol)

            // Haptic feedback
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)
        } catch {
            // Rollback on failure
            if let item = removedItem, let index = removedIndex {
                withAnimation {
                    watchlistItems.insert(item, at: min(index, watchlistItems.count))
                }
            }

            self.error = error.localizedDescription

            // Haptic feedback for error
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.error)
        }
    }

    /// Check if a symbol is in the watchlist
    func isInWatchlist(_ symbol: String) -> Bool {
        watchlistItems.contains { $0.symbol.uppercased() == symbol.uppercased() }
    }

    // MARK: - Private Helpers

    private func updateMarketDirection() {
        let upCount = indices.filter { $0.isUp }.count
        let total = indices.count

        if total == 0 {
            marketDirection = .mixed
        } else if upCount >= (total * 3 / 4) {
            marketDirection = .bullish
        } else if upCount <= (total / 4) {
            marketDirection = .bearish
        } else {
            marketDirection = .mixed
        }
    }
}

// MARK: - Supporting Types

enum MarketDirection: String {
    case bullish = "Bullish"
    case bearish = "Bearish"
    case mixed = "Mixed"

    var icon: String {
        switch self {
        case .bullish: return "arrow.up.right"
        case .bearish: return "arrow.down.right"
        case .mixed: return "arrow.left.arrow.right"
        }
    }

    var color: Color {
        switch self {
        case .bullish: return .green
        case .bearish: return .red
        case .mixed: return .orange
        }
    }
}
