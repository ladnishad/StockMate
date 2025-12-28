import SwiftUI

/// Main home screen combining market overview, profile selection, and user watchlist
struct HomeView: View {
    @StateObject private var viewModel = HomeViewModel()
    @State private var showingAddTicker = false
    @State private var showingChat = false

    var body: some View {
        NavigationStack {
            ZStack {
                ScrollView {
                    VStack(spacing: 24) {
                        // Search bar (opens add ticker sheet)
                        SearchBarView {
                            showingAddTicker = true
                        }
                        .padding(.horizontal, 20)

                        // Market Overview Section
                        MarketIndicesView(
                            indices: viewModel.indices,
                            isLoading: viewModel.isLoadingIndices,
                            marketDirection: viewModel.marketDirection
                        )

                        // User Watchlist Section
                        UserWatchlistView(
                            items: viewModel.watchlistItems,
                            isLoading: viewModel.isLoadingWatchlist
                        ) { symbol in
                            Task {
                                await viewModel.removeFromWatchlist(symbol)
                            }
                        }

                        // Bottom padding for FAB
                        Spacer()
                            .frame(height: 80)
                    }
                    .padding(.top, 8)
                }
                .scrollIndicators(.hidden)

                // Floating Add Button
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        FloatingAddButton {
                            showingAddTicker = true
                        }
                        .padding(.trailing, 20)
                        .padding(.bottom, 20)
                    }
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("StockMate")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 16) {
                        // AI Chat button
                        Button {
                            showingChat = true
                        } label: {
                            Image(systemName: "bubble.left.and.bubble.right.fill")
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(Color.blue)
                        }

                        // Refresh button
                        RefreshButton(isRefreshing: viewModel.isRefreshing) {
                            Task {
                                await viewModel.refresh()
                            }
                        }
                    }
                }
            }
            .refreshable {
                await viewModel.refresh()
            }
            .navigationDestination(for: String.self) { symbol in
                StockDetailView(symbol: symbol)
            }
            .sheet(isPresented: $showingAddTicker) {
                TickerSearchView { symbol in
                    Task {
                        _ = await viewModel.addToWatchlist(symbol)
                    }
                }
            }
            .fullScreenCover(isPresented: $showingChat) {
                NavigationStack {
                    ChatView()
                }
            }
            .alert("Error", isPresented: .constant(viewModel.error != nil)) {
                Button("Retry") {
                    Task {
                        await viewModel.refresh()
                    }
                }
                Button("Dismiss", role: .cancel) {
                    viewModel.dismissError()
                }
            } message: {
                if let error = viewModel.error {
                    Text(error)
                }
            }
        }
        .task {
            await viewModel.loadInitialData()
        }
    }
}

/// Toolbar refresh button with loading state
struct RefreshButton: View {
    let isRefreshing: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            if isRefreshing {
                ProgressView()
                    .scaleEffect(0.8)
            } else {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 15, weight: .semibold))
            }
        }
        .disabled(isRefreshing)
    }
}

// MARK: - Preview

#Preview("Home View") {
    HomeView()
}
