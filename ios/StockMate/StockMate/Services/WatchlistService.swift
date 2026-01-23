import Foundation

/// Manages user watchlist with backend sync and local caching
actor WatchlistService {
    static let shared = WatchlistService()

    private let apiService = APIService.shared
    private let cache = WatchlistCache()

    // MARK: - CRUD Operations

    /// Get the user's watchlist
    /// Returns cached data immediately, then syncs with backend
    func getWatchlist(forceRefresh: Bool = false) async throws -> [WatchlistItem] {
        // Return cached data if available and not forcing refresh
        let cached = cache.load()
        if !cached.isEmpty && !forceRefresh && !cache.needsSync {
            return cached
        }

        // Fetch from backend
        do {
            let items = try await apiService.fetchUserWatchlist()
            cache.save(items)
            return items
        } catch {
            // Return cached data on error
            if !cached.isEmpty {
                return cached
            }
            throw error
        }
    }

    /// Add a symbol to the watchlist
    func addSymbol(_ symbol: String) async throws -> WatchlistItem {
        let upperSymbol = symbol.uppercased()

        // If already in watchlist, return the existing item (not an error)
        if let existingItem = cache.load().first(where: { $0.symbol.uppercased() == upperSymbol }) {
            return existingItem
        }

        // Optimistically add to cache
        let tempItem = WatchlistItem(
            symbol: upperSymbol,
            addedAt: Date(),
            notes: nil,
            alertsEnabled: false,
            currentPrice: nil,
            change: nil,
            changePct: nil,
            score: nil,
            recommendation: nil
        )
        cache.addItem(tempItem)

        // POST to backend
        do {
            let item = try await apiService.addToWatchlist(symbol: upperSymbol)
            cache.updateItem(item)
            return item
        } catch {
            // Rollback on failure
            cache.removeItem(symbol: upperSymbol)
            throw error
        }
    }

    /// Remove a symbol from the watchlist
    func removeSymbol(_ symbol: String) async throws {
        let upperSymbol = symbol.uppercased()

        // Get current item for rollback
        let items = cache.load()
        guard let itemToRemove = items.first(where: { $0.symbol == upperSymbol }) else {
            throw WatchlistError.notFound
        }

        // Optimistically remove from cache
        cache.removeItem(symbol: upperSymbol)

        // DELETE from backend
        do {
            try await apiService.removeFromWatchlist(symbol: upperSymbol)
        } catch {
            // Rollback on failure
            cache.addItem(itemToRemove)
            throw error
        }
    }

    /// Check if a symbol is in the watchlist (nonisolated for synchronous access)
    nonisolated func hasSymbol(_ symbol: String) -> Bool {
        cache.hasSymbol(symbol)
    }

    /// Search for tickers
    func searchTickers(query: String) async throws -> [SearchResult] {
        guard !query.isEmpty else { return [] }
        return try await apiService.searchTickers(query: query)
    }

    /// Get stock detail for a symbol
    func getStockDetail(symbol: String) async throws -> StockDetail {
        try await apiService.fetchStockDetail(symbol: symbol)
    }

    /// Refresh quotes for all items in the watchlist
    func refreshQuotes() async throws -> [WatchlistItem] {
        try await getWatchlist(forceRefresh: true)
    }

    /// Get cached items (synchronous, for quick UI updates)
    nonisolated func getCachedItems() -> [WatchlistItem] {
        cache.load()
    }

    /// Get count of items in watchlist
    nonisolated var count: Int {
        cache.count
    }
}

// MARK: - Errors

enum WatchlistError: LocalizedError {
    case alreadyExists
    case notFound
    case syncFailed

    var errorDescription: String? {
        switch self {
        case .alreadyExists:
            return "This symbol is already in your watchlist"
        case .notFound:
            return "Symbol not found in watchlist"
        case .syncFailed:
            return "Failed to sync with server"
        }
    }
}
