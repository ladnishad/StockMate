import SwiftUI

/// Main home screen combining market overview and user watchlist
struct HomeView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @StateObject private var viewModel = HomeViewModel()
    @State private var showingAddTicker = false
    @State private var showingChat = false
    @State private var showingLogoutConfirmation = false
    @State private var showingSettings = false
    @State private var showingScanner = false

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
                            marketDirection: viewModel.marketDirection,
                            isMarketOpen: viewModel.isMarketOpen
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
                ToolbarItem(placement: .topBarLeading) {
                    Menu {
                        if let user = authManager.currentUser {
                            Text(user.email)
                        }
                        Divider()
                        Button {
                            showingSettings = true
                        } label: {
                            Label("Settings", systemImage: "gearshape")
                        }
                        Button(role: .destructive) {
                            showingLogoutConfirmation = true
                        } label: {
                            Label("Sign Out", systemImage: "rectangle.portrait.and.arrow.right")
                        }
                    } label: {
                        Image(systemName: "person.circle.fill")
                            .font(.system(size: 20))
                            .foregroundStyle(.blue)
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 16) {
                        // Scanner button
                        Button {
                            showingScanner = true
                        } label: {
                            Image(systemName: "waveform.badge.magnifyingglass")
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(Color.blue)
                        }

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
            .confirmationDialog("Sign Out", isPresented: $showingLogoutConfirmation) {
                Button("Sign Out", role: .destructive) {
                    Task {
                        await authManager.logout()
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Are you sure you want to sign out?")
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
            .sheet(isPresented: $showingSettings) {
                SettingsView()
                    .environmentObject(authManager)
            }
            .fullScreenCover(isPresented: $showingScanner) {
                ScannerView()
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
        .environmentObject(AuthenticationManager.shared)
}
