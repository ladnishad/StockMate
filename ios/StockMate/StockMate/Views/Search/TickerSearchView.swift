import SwiftUI

/// Modal sheet for searching and adding tickers
struct TickerSearchView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel = WatchlistViewModel()
    @FocusState private var isSearchFocused: Bool

    let onAdd: ((String) -> Void)?

    init(onAdd: ((String) -> Void)? = nil) {
        self.onAdd = onAdd
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Limit reached banner
                if viewModel.isAtLimit {
                    LimitReachedBanner(tierName: viewModel.tierName, limit: viewModel.watchlistLimit)
                }

                // Search bar
                SearchInputField(
                    text: $viewModel.searchQuery,
                    isFocused: $isSearchFocused,
                    placeholder: "Search ticker or company"
                )
                .padding()

                Divider()

                // Results
                if viewModel.isSearching {
                    Spacer()
                    LoadingView("Searching...", style: .pulse)
                    Spacer()
                } else if !viewModel.searchQuery.isEmpty && viewModel.searchResults.isEmpty {
                    Spacer()
                    EmptySearchView(query: viewModel.searchQuery)
                    Spacer()
                } else if viewModel.searchResults.isEmpty {
                    Spacer()
                    SearchPromptView()
                    Spacer()
                } else {
                    List(viewModel.searchResults) { result in
                        SearchResultRow(
                            result: result,
                            isInWatchlist: viewModel.isInWatchlist(result.symbol),
                            isAtLimit: viewModel.isAtLimit
                        ) {
                            addTicker(result.symbol)
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Add Ticker")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .alert(
                viewModel.lastError?.isLimitError == true ? "Watchlist Full" : "Unable to Add",
                isPresented: .constant(viewModel.addError != nil)
            ) {
                Button("OK") {
                    viewModel.addError = nil
                    viewModel.lastError = nil
                }
                if viewModel.lastError?.isLimitError == true {
                    Button("View Plans") {
                        viewModel.addError = nil
                        viewModel.lastError = nil
                        // TODO: Navigate to subscription plans
                    }
                }
            } message: {
                if let error = viewModel.lastError {
                    if error.isLimitError {
                        Text("Your \(viewModel.tierName) plan allows \(viewModel.watchlistLimit) stocks. Upgrade to add more.")
                    } else {
                        Text(error.localizedDescription)
                    }
                }
            }
        }
        .task {
            await viewModel.loadSubscriptionInfo()
        }
        .onAppear {
            isSearchFocused = true
        }
    }

    private func addTicker(_ symbol: String) {
        Task {
            let success = await viewModel.addSymbol(symbol)
            if success {
                onAdd?(symbol)
                dismiss()
            }
        }
    }
}

// MARK: - Search Input Field

struct SearchInputField: View {
    @Binding var text: String
    var isFocused: FocusState<Bool>.Binding
    let placeholder: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.secondary)

            TextField(placeholder, text: $text)
                .font(.system(size: 16, weight: .regular))
                .textInputAutocapitalization(.characters)
                .autocorrectionDisabled()
                .focused(isFocused)

            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

// MARK: - Search Result Row

struct SearchResultRow: View {
    let result: SearchResult
    let isInWatchlist: Bool
    var isAtLimit: Bool = false
    let onAdd: () -> Void

    private var isDisabled: Bool {
        isInWatchlist || (isAtLimit && !isInWatchlist)
    }

    var body: some View {
        Button(action: onAdd) {
            HStack(spacing: 12) {
                // Symbol badge
                Text(result.symbol)
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .frame(width: 70, alignment: .leading)

                // Company name and exchange
                VStack(alignment: .leading, spacing: 2) {
                    Text(result.name)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        Text(result.exchange)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)

                        if result.isETF {
                            Text("ETF")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(.blue)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(
                                    Capsule()
                                        .fill(Color.blue.opacity(0.12))
                                )
                        }
                    }
                }

                Spacer()

                // Add indicator
                if isInWatchlist {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(.green)
                } else if isAtLimit {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(.secondary)
                } else {
                    Image(systemName: "plus.circle")
                        .font(.system(size: 20))
                        .foregroundStyle(Color.accentColor)
                }
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .opacity(isDisabled ? 0.6 : 1)
    }
}

// MARK: - Empty States

struct EmptySearchView: View {
    let query: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)

            Text("No results for \"\(query)\"")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.primary)

            Text("Try searching by ticker symbol or company name")
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(32)
    }
}

struct SearchPromptView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)

            Text("Search for stocks")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.primary)

            Text("Enter a ticker symbol like AAPL or company name like Apple")
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(32)
    }
}

// MARK: - Limit Reached Banner

struct LimitReachedBanner: View {
    let tierName: String
    let limit: Int

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.circle.fill")
                .font(.system(size: 20))
                .foregroundStyle(.orange)

            VStack(alignment: .leading, spacing: 2) {
                Text("Watchlist Full")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.primary)

                Text("Your \(tierName) plan allows \(limit) stocks. Upgrade to add more.")
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.orange.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
        .padding(.horizontal)
        .padding(.top, 8)
    }
}

// MARK: - Preview

#Preview("Ticker Search") {
    TickerSearchView()
}

#Preview("Limit Banner") {
    LimitReachedBanner(tierName: "Base", limit: 2)
        .padding()
}
