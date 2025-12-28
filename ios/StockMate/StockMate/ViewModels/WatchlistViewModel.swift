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

    /// Add a symbol to the watchlist
    func addSymbol(_ symbol: String) async -> Bool {
        isAdding = true
        addError = nil

        do {
            _ = try await service.addSymbol(symbol)
            isAdding = false
            return true
        } catch {
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
