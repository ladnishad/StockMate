import Foundation

/// Local cache for watchlist with UserDefaults persistence
class WatchlistCache {
    private let defaults = UserDefaults.standard
    private let watchlistKey = "user_watchlist_items"
    private let lastSyncKey = "watchlist_last_sync"

    // In-memory cache
    private var cachedItems: [WatchlistItem]?

    // MARK: - Public Methods

    /// Save items to cache
    func save(_ items: [WatchlistItem]) {
        cachedItems = items

        // Persist to UserDefaults
        if let encoded = try? JSONEncoder().encode(items) {
            defaults.set(encoded, forKey: watchlistKey)
            defaults.set(Date(), forKey: lastSyncKey)
        }
    }

    /// Load items from cache
    func load() -> [WatchlistItem] {
        // Return in-memory cache if available
        if let cached = cachedItems {
            return cached
        }

        // Load from UserDefaults
        guard let data = defaults.data(forKey: watchlistKey),
              let items = try? JSONDecoder().decode([WatchlistItem].self, from: data) else {
            return []
        }

        cachedItems = items
        return items
    }

    /// Add an item to the cache (at the beginning)
    func addItem(_ item: WatchlistItem) {
        var items = load()

        // Remove if already exists (to avoid duplicates)
        items.removeAll { $0.symbol == item.symbol }

        // Add at beginning
        items.insert(item, at: 0)

        save(items)
    }

    /// Remove an item from the cache
    func removeItem(symbol: String) {
        var items = load()
        items.removeAll { $0.symbol == symbol }
        save(items)
    }

    /// Update an item in the cache
    func updateItem(_ item: WatchlistItem) {
        var items = load()
        if let index = items.firstIndex(where: { $0.symbol == item.symbol }) {
            items[index] = item
            save(items)
        }
    }

    /// Check if a symbol is in the watchlist
    func hasSymbol(_ symbol: String) -> Bool {
        load().contains { $0.symbol.uppercased() == symbol.uppercased() }
    }

    /// Clear the cache
    func clear() {
        cachedItems = nil
        defaults.removeObject(forKey: watchlistKey)
        defaults.removeObject(forKey: lastSyncKey)
    }

    /// Get the last sync date
    var lastSyncDate: Date? {
        defaults.object(forKey: lastSyncKey) as? Date
    }

    /// Check if cache needs sync (older than 5 minutes)
    var needsSync: Bool {
        guard let lastSync = lastSyncDate else { return true }
        return Date().timeIntervalSince(lastSync) > 300 // 5 minutes
    }

    /// Get symbol list only (for quick checks)
    var symbols: [String] {
        load().map { $0.symbol }
    }

    /// Get count of items
    var count: Int {
        load().count
    }
}
