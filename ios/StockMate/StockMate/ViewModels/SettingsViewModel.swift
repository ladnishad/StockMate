import Foundation
import SwiftUI
import Combine

/// ViewModel for managing user settings, primarily AI provider selection
@MainActor
class SettingsViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var selectedProvider: String = "grok"  // Grok is the default
    @Published var availableProviders: [String] = ["claude", "grok"]
    @Published var isLoading: Bool = false
    @Published var isSaving: Bool = false
    @Published var error: String?
    @Published var showSuccessAnimation: Bool = false

    // Subscription
    @Published var subscription: UserSubscription?
    @Published var allTiers: [SubscriptionTierInfo] = []

    // MARK: - Computed Properties

    /// Returns a description for the currently selected AI provider
    var providerDescription: String {
        switch selectedProvider {
        case "claude":
            return "Advanced reasoning with comprehensive web search for news, earnings, and market analysis."
        case "grok":
            return "Real-time X/Twitter sentiment analysis for market buzz and retail investor discussions."
        default:
            return "Select an AI provider to power your trading insights."
        }
    }

    /// Whether Grok is available based on subscription tier
    var isGrokAvailable: Bool {
        subscription?.tierInfo.multiModelAccess ?? false
    }

    /// Current tier display name
    var currentTierName: String {
        subscription?.tierInfo.name ?? "Base"
    }

    /// Current tier price display
    var currentTierPrice: String {
        subscription?.tierInfo.priceDisplay ?? "Free"
    }

    /// Returns the display name for a provider
    func displayName(for provider: String) -> String {
        switch provider {
        case "claude":
            return "Claude"
        case "grok":
            return "Grok"
        default:
            return provider.capitalized
        }
    }

    /// Returns the company name for a provider
    func companyName(for provider: String) -> String {
        switch provider {
        case "claude":
            return "by Anthropic"
        case "grok":
            return "by xAI"
        default:
            return ""
        }
    }

    /// Returns the feature tagline for a provider
    func featureTagline(for provider: String) -> String {
        switch provider {
        case "claude":
            return "Deep Analysis"
        case "grok":
            return "Real-time Sentiment"
        default:
            return ""
        }
    }

    /// Returns the icon name for a provider
    func iconName(for provider: String) -> String {
        switch provider {
        case "claude":
            return "brain.head.profile"
        case "grok":
            return "sparkles"
        default:
            return "cpu"
        }
    }

    /// Returns the accent color for a provider
    func accentColor(for provider: String) -> Color {
        switch provider {
        case "claude":
            return Color(red: 0.85, green: 0.45, blue: 0.25) // Anthropic orange
        case "grok":
            return Color.white // X/Grok white on dark
        default:
            return Color.blue
        }
    }

    /// Returns the gradient for a provider card
    func cardGradient(for provider: String, isSelected: Bool) -> LinearGradient {
        if !isSelected {
            return LinearGradient(
                colors: [Color(.secondarySystemGroupedBackground)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }

        switch provider {
        case "claude":
            return LinearGradient(
                colors: [
                    Color(red: 0.85, green: 0.45, blue: 0.25).opacity(0.15),
                    Color(red: 0.75, green: 0.35, blue: 0.20).opacity(0.08)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        case "grok":
            return LinearGradient(
                colors: [
                    Color.white.opacity(0.12),
                    Color.gray.opacity(0.06)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        default:
            return LinearGradient(
                colors: [Color.blue.opacity(0.15)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
    }

    /// Check if a provider is enabled for selection
    func isProviderEnabled(_ provider: String) -> Bool {
        // Grok requires multi-model access (Premium+ subscription)
        if provider == "grok" {
            return isGrokAvailable
        }
        return true
    }

    /// Get badge text for a disabled provider
    func providerBadge(for provider: String) -> String? {
        if provider == "grok" && !isGrokAvailable {
            return "Premium"
        }
        return nil
    }

    // MARK: - API Methods

    /// Load user settings from the server
    func loadSettings() async {
        isLoading = true
        error = nil

        do {
            let settings = try await APIService.shared.getUserSettings()
            selectedProvider = settings.modelProvider
            availableProviders = settings.availableProviders
            subscription = settings.subscription
        } catch {
            self.error = "Failed to load settings"
            print("SettingsViewModel: Error loading settings - \(error)")
        }

        isLoading = false
    }

    /// Load all subscription tiers
    func loadAllTiers() async {
        do {
            allTiers = try await APIService.shared.getSubscriptionTiers()
        } catch {
            print("SettingsViewModel: Error loading tiers - \(error)")
        }
    }

    /// Update the user's AI provider preference
    func updateProvider(_ provider: String) async {
        guard provider != selectedProvider else { return }

        // Check if provider is available for this subscription
        if provider == "grok" && !isGrokAvailable {
            self.error = "Upgrade to Premium to access Grok"
            return
        }

        let previousProvider = selectedProvider
        selectedProvider = provider // Optimistic update
        isSaving = true
        error = nil

        do {
            try await APIService.shared.updateProvider(provider: provider)

            // Show success animation
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                showSuccessAnimation = true
            }

            // Hide success animation after delay
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            withAnimation {
                showSuccessAnimation = false
            }

        } catch {
            // Revert on failure
            selectedProvider = previousProvider
            self.error = "Failed to update provider"
            print("SettingsViewModel: Error updating provider - \(error)")
        }

        isSaving = false
    }
}
