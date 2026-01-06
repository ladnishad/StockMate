import SwiftUI
import Combine

/// ViewModel for watchlist management (search and add)
@MainActor
class WatchlistViewModel: ObservableObject {
    // MARK: - Published State

    @Published var searchQuery = ""
    @Published var searchResults: [SearchResult] = []
    @Published var isSearching = false
    @Published var searchError: String?

    @Published var isAdding = false
    @Published var addError: String?

    // MARK: - Private

    private let service = WatchlistService.shared
    private var searchTask: Task<Void, Never>?

    // Debounce search
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init() {
        // Setup debounced search
        $searchQuery
            .debounce(for: .milliseconds(300), scheduler: DispatchQueue.main)
            .removeDuplicates()
            .sink { [weak self] query in
                Task {
                    await self?.performSearch(query: query)
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Search

    /// Perform search for tickers
    func performSearch(query: String) async {
        // Cancel any existing search
        searchTask?.cancel()

        guard !query.isEmpty else {
            searchResults = []
            return
        }

        searchTask = Task {
            isSearching = true
            searchError = nil

            do {
                let results = try await service.searchTickers(query: query)
                if !Task.isCancelled {
                    searchResults = results
                }
            } catch {
                if !Task.isCancelled {
                    searchError = error.localizedDescription
                    searchResults = []
                }
            }

            isSearching = false
        }
    }

    /// Clear search state
    func clearSearch() {
        searchQuery = ""
        searchResults = []
        searchError = nil
    }

    // MARK: - Add Symbol

    /// Error type for watchlist operations
    enum WatchlistError: LocalizedError {
        case limitReached(message: String)
        case networkError(String)
        case unknown(String)

        var errorDescription: String? {
            switch self {
            case .limitReached(let message):
                return message
            case .networkError(let message):
                return message
            case .unknown(let message):
                return message
            }
        }

        var isLimitError: Bool {
            if case .limitReached = self { return true }
            return false
        }
    }

    @Published var lastError: WatchlistError?

    /// Add a symbol to the watchlist
    func addSymbol(_ symbol: String) async -> Bool {
        isAdding = true
        addError = nil
        lastError = nil

        do {
            _ = try await service.addSymbol(symbol)
            isAdding = false
            return true
        } catch let error as APIServiceError {
            isAdding = false

            switch error {
            case .serverError(let detail):
                // Check if it's a tier limit error (403)
                if detail.contains("limit reached") || detail.contains("Watchlist limit") {
                    lastError = .limitReached(message: detail)
                    addError = detail
                } else {
                    lastError = .networkError(detail)
                    addError = detail
                }
            case .httpError(let code):
                if code == 403 {
                    lastError = .limitReached(message: "Watchlist limit reached. Upgrade your subscription to add more stocks.")
                    addError = "Watchlist limit reached. Upgrade your subscription to add more stocks."
                } else {
                    lastError = .networkError("Server error (code: \(code))")
                    addError = "Server error (code: \(code))"
                }
            default:
                lastError = .unknown(error.localizedDescription)
                addError = error.localizedDescription
            }
            return false
        } catch {
            lastError = .unknown(error.localizedDescription)
            addError = error.localizedDescription
            isAdding = false
            return false
        }
    }

    /// Check if symbol is already in watchlist
    func isInWatchlist(_ symbol: String) -> Bool {
        service.hasSymbol(symbol)
    }
}
