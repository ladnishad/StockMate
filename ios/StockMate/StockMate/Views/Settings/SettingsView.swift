import SwiftUI

/// Settings screen for managing AI provider, subscription, and account preferences
struct SettingsView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @StateObject private var viewModel = SettingsViewModel()
    @Environment(\.dismiss) private var dismiss

    @State private var showingLogoutConfirmation = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 28) {
                    // Subscription Section
                    subscriptionSection

                    // AI Provider Section
                    aiProviderSection

                    // Account Section
                    accountSection
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 40)
            }
            .scrollIndicators(.hidden)
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
            .task {
                await viewModel.loadSettings()
            }
            .overlay {
                if viewModel.isLoading {
                    loadingOverlay
                }
            }
            .confirmationDialog(
                "Sign Out",
                isPresented: $showingLogoutConfirmation,
                titleVisibility: .visible
            ) {
                Button("Sign Out", role: .destructive) {
                    Task {
                        await authManager.logout()
                        dismiss()
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Are you sure you want to sign out?")
            }
        }
    }

    // MARK: - Subscription Section

    private var subscriptionSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Section Header
            HStack(spacing: 8) {
                Image(systemName: "crown.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.secondary)

                Text("Subscription")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)
            }
            .padding(.leading, 4)

            // Subscription Card
            VStack(spacing: 0) {
                if let subscription = viewModel.subscription {
                    // Current Plan
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(subscription.tierInfo.name)
                                    .font(.system(size: 20, weight: .bold, design: .rounded))
                                    .foregroundStyle(.primary)

                                Text(subscription.tierInfo.priceDisplay)
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 4)
                                    .background(
                                        Capsule()
                                            .fill(tierAccentColor(for: subscription.tier))
                                    )
                            }

                            Text(subscription.tierInfo.description)
                                .font(.system(size: 13))
                                .foregroundStyle(.secondary)
                        }

                        Spacer()

                        Image(systemName: tierIcon(for: subscription.tier))
                            .font(.system(size: 28, weight: .medium))
                            .foregroundStyle(tierAccentColor(for: subscription.tier))
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 16)

                    Divider()
                        .padding(.leading, 16)

                    // Watchlist Usage
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Watchlist")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.secondary)

                            Text(subscription.usageDisplay)
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundStyle(.primary)
                        }

                        Spacer()

                        if subscription.tierInfo.isUnlimited {
                            Text("Unlimited")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.green)
                        } else {
                            Text(subscription.remainingDisplay)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(subscription.canAddToWatchlist ? .green : .orange)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)

                    Divider()
                        .padding(.leading, 16)

                    // Features List
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Your Benefits")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.secondary)
                            .padding(.bottom, 4)

                        ForEach(subscription.tierInfo.features, id: \.self) { feature in
                            HStack(spacing: 10) {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.system(size: 14))
                                    .foregroundStyle(.green)

                                Text(feature)
                                    .font(.system(size: 14))
                                    .foregroundStyle(.primary)
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)

                    Divider()
                        .padding(.leading, 16)

                    // Manage Subscription Button
                    Button {
                        // Coming soon
                    } label: {
                        HStack {
                            Image(systemName: "creditcard")
                                .font(.system(size: 16, weight: .medium))

                            Text("Manage Subscription")
                                .font(.system(size: 16, weight: .medium))

                            Spacer()

                            Text("Coming Soon")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(
                                    Capsule()
                                        .fill(Color.gray)
                                )
                        }
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 14)
                    }
                    .disabled(true)
                } else {
                    // Loading state
                    HStack {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("Loading subscription...")
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 20)
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
    }

    // MARK: - AI Provider Section

    private var aiProviderSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Section Header
            HStack(spacing: 8) {
                Image(systemName: "cpu")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.secondary)

                Text("AI Provider")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)
            }
            .padding(.leading, 4)

            // Provider Cards - Show all providers, not just available ones
            VStack(spacing: 12) {
                // Always show both Claude and Grok
                ForEach(["claude", "grok"], id: \.self) { provider in
                    ProviderCard(
                        provider: provider,
                        isSelected: viewModel.selectedProvider == provider,
                        isEnabled: viewModel.isProviderEnabled(provider),
                        isSaving: viewModel.isSaving,
                        badge: viewModel.providerBadge(for: provider),
                        viewModel: viewModel
                    ) {
                        Task {
                            await viewModel.updateProvider(provider)
                        }
                    }
                }
            }

            // Provider Description
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Image(systemName: "info.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(.tertiary)

                    Text(viewModel.providerDescription)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                // Show upgrade hint if on base tier
                if !viewModel.isGrokAvailable {
                    HStack(spacing: 10) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 14))
                            .foregroundStyle(.blue)

                        Text("Upgrade to Premium to unlock Grok with real-time X/Twitter sentiment")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.blue)
                    }
                    .padding(.top, 4)
                }
            }
            .padding(.horizontal, 4)
            .padding(.top, 4)
            .animation(.easeInOut(duration: 0.25), value: viewModel.selectedProvider)
        }
    }

    // MARK: - Account Section

    private var accountSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Section Header
            HStack(spacing: 8) {
                Image(systemName: "person.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.secondary)

                Text("Account")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)
            }
            .padding(.leading, 4)

            // Account Card
            VStack(spacing: 0) {
                // Email Row
                if let user = authManager.currentUser {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Email")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.secondary)

                            Text(user.email)
                                .font(.system(size: 16, weight: .medium))
                                .foregroundStyle(.primary)
                        }

                        Spacer()

                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 20))
                            .foregroundStyle(.green)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)

                    Divider()
                        .padding(.leading, 16)
                }

                // Sign Out Button
                Button {
                    showingLogoutConfirmation = true
                } label: {
                    HStack {
                        Image(systemName: "rectangle.portrait.and.arrow.right")
                            .font(.system(size: 16, weight: .medium))

                        Text("Sign Out")
                            .font(.system(size: 16, weight: .medium))

                        Spacer()

                        Image(systemName: "chevron.right")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.tertiary)
                    }
                    .foregroundStyle(.red)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
    }

    // MARK: - Loading Overlay

    private var loadingOverlay: some View {
        ZStack {
            Color.black.opacity(0.3)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                ProgressView()
                    .scaleEffect(1.2)
                    .tint(.white)

                Text("Loading...")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white)
            }
            .padding(24)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.ultraThinMaterial)
            )
        }
    }

    // MARK: - Helpers

    private func tierAccentColor(for tier: String) -> Color {
        switch tier {
        case "base":
            return Color.gray
        case "premium":
            return Color.blue
        case "pro":
            return Color.purple
        case "unlimited":
            return Color(red: 0.85, green: 0.65, blue: 0.13) // Gold
        default:
            return Color.gray
        }
    }

    private func tierIcon(for tier: String) -> String {
        switch tier {
        case "base":
            return "star"
        case "premium":
            return "star.fill"
        case "pro":
            return "crown"
        case "unlimited":
            return "crown.fill"
        default:
            return "star"
        }
    }
}

// MARK: - Claude Logo View

struct ClaudeLogoView: View {
    let size: CGFloat
    let isSelected: Bool
    var isEnabled: Bool = true

    // Claude brand color: #D97757 (warm terracotta)
    private let claudeColor = Color(red: 0.851, green: 0.467, blue: 0.341)

    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .fill(
                    isSelected
                        ? claudeColor.opacity(0.15)
                        : Color(.systemGray5)
                )
                .frame(width: size, height: size)

            // Claude logo from asset
            Image("claude-logo")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: size * 0.55, height: size * 0.55)
                .opacity(isEnabled ? 1 : 0.4)
        }
    }
}

// MARK: - Grok Logo View

struct GrokLogoView: View {
    let size: CGFloat
    let isSelected: Bool
    var isEnabled: Bool = true

    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .fill(
                    isSelected
                        ? Color.white.opacity(0.12)
                        : Color(.systemGray5)
                )
                .frame(width: size, height: size)

            // Grok logo from asset (template mode - takes foreground color)
            Image("grok-logo")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: size * 0.55, height: size * 0.55)
                .foregroundStyle(isSelected ? Color.white : Color.secondary)
                .opacity(isEnabled ? 1 : 0.4)
        }
    }
}

// MARK: - Provider Card Component

struct ProviderCard: View {
    let provider: String
    let isSelected: Bool
    let isEnabled: Bool
    let isSaving: Bool
    let badge: String?
    @ObservedObject var viewModel: SettingsViewModel
    let onSelect: () -> Void

    @State private var isPressed = false

    var body: some View {
        Button(action: {
            if isEnabled {
                onSelect()
            }
        }) {
            HStack(spacing: 14) {
                // Provider Logo
                Group {
                    if provider == "claude" {
                        ClaudeLogoView(size: 44, isSelected: isSelected, isEnabled: isEnabled)
                    } else if provider == "grok" {
                        GrokLogoView(size: 44, isSelected: isSelected, isEnabled: isEnabled)
                    } else {
                        // Fallback for unknown providers
                        ZStack {
                            Circle()
                                .fill(Color(.systemGray5))
                                .frame(width: 44, height: 44)

                            Image(systemName: "cpu")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundStyle(Color.secondary)
                        }
                    }
                }

                // Provider Info
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(viewModel.displayName(for: provider))
                            .font(.system(size: 17, weight: .semibold, design: .rounded))
                            .foregroundStyle(isEnabled ? .primary : .secondary)

                        Text(viewModel.companyName(for: provider))
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.tertiary)

                        // Badge for disabled providers
                        if let badge = badge {
                            Text(badge)
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(
                                    Capsule()
                                        .fill(Color.blue)
                                )
                        }
                    }

                    Text(viewModel.featureTagline(for: provider))
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(isEnabled ? .secondary : .tertiary)
                }

                Spacer()

                // Selection Indicator or Lock
                ZStack {
                    if !isEnabled {
                        // Locked state
                        Circle()
                            .fill(Color(.systemGray5))
                            .frame(width: 24, height: 24)

                        Image(systemName: "lock.fill")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(.secondary)
                    } else if isSelected {
                        Circle()
                            .fill(viewModel.accentColor(for: provider))
                            .frame(width: 24, height: 24)

                        Image(systemName: "checkmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(provider == "grok" ? .black : .white)
                    } else {
                        Circle()
                            .strokeBorder(Color(.systemGray3), lineWidth: 2)
                            .frame(width: 24, height: 24)
                    }
                }
                .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isSelected)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(
                ZStack {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(isEnabled ? viewModel.cardGradient(for: provider, isSelected: isSelected) : LinearGradient(colors: [Color(.systemGray6)], startPoint: .topLeading, endPoint: .bottomTrailing))

                    if isSelected && isEnabled {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(
                                viewModel.accentColor(for: provider).opacity(0.4),
                                lineWidth: 1.5
                            )
                    }
                }
            )
            .scaleEffect(isPressed && isEnabled ? 0.98 : 1.0)
            .animation(.spring(response: 0.25, dampingFraction: 0.7), value: isPressed)
        }
        .buttonStyle(.plain)
        .disabled(isSaving || !isEnabled)
        .opacity(isSaving && !isSelected ? 0.6 : 1.0)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in if isEnabled { isPressed = true } }
                .onEnded { _ in isPressed = false }
        )
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
        .environmentObject(AuthenticationManager.shared)
}
