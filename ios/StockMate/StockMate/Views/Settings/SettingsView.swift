import SwiftUI

/// Settings screen for managing AI provider and account preferences
struct SettingsView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @StateObject private var viewModel = SettingsViewModel()
    @Environment(\.dismiss) private var dismiss

    @State private var showingLogoutConfirmation = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 28) {
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

            // Provider Cards
            VStack(spacing: 12) {
                ForEach(viewModel.availableProviders, id: \.self) { provider in
                    ProviderCard(
                        provider: provider,
                        isSelected: viewModel.selectedProvider == provider,
                        isSaving: viewModel.isSaving,
                        viewModel: viewModel
                    ) {
                        Task {
                            await viewModel.updateProvider(provider)
                        }
                    }
                }
            }

            // Provider Description
            HStack(spacing: 10) {
                Image(systemName: "info.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.tertiary)

                Text(viewModel.providerDescription)
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
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
}

// MARK: - Claude Logo View

struct ClaudeLogoView: View {
    let size: CGFloat
    let isSelected: Bool

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
        }
    }
}

// MARK: - Grok Logo View

struct GrokLogoView: View {
    let size: CGFloat
    let isSelected: Bool

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
        }
    }
}

// MARK: - Provider Card Component

struct ProviderCard: View {
    let provider: String
    let isSelected: Bool
    let isSaving: Bool
    @ObservedObject var viewModel: SettingsViewModel
    let onSelect: () -> Void

    @State private var isPressed = false

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 14) {
                // Provider Logo
                Group {
                    if provider == "claude" {
                        ClaudeLogoView(size: 44, isSelected: isSelected)
                    } else if provider == "grok" {
                        GrokLogoView(size: 44, isSelected: isSelected)
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
                            .foregroundStyle(.primary)

                        Text(viewModel.companyName(for: provider))
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.tertiary)
                    }

                    Text(viewModel.featureTagline(for: provider))
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                // Selection Indicator
                ZStack {
                    if isSelected {
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
                        .fill(viewModel.cardGradient(for: provider, isSelected: isSelected))

                    if isSelected {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(
                                viewModel.accentColor(for: provider).opacity(0.4),
                                lineWidth: 1.5
                            )
                    }
                }
            )
            .scaleEffect(isPressed ? 0.98 : 1.0)
            .animation(.spring(response: 0.25, dampingFraction: 0.7), value: isPressed)
        }
        .buttonStyle(.plain)
        .disabled(isSaving)
        .opacity(isSaving && !isSelected ? 0.6 : 1.0)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in isPressed = true }
                .onEnded { _ in isPressed = false }
        )
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
        .environmentObject(AuthenticationManager.shared)
}
