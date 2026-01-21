//
//  StockMateApp.swift
//  StockMate
//
//  Created by Nishad Lad on 12/10/25.
//

import SwiftUI

@main
struct StockMateApp: App {
    @StateObject private var authManager = AuthenticationManager.shared
    @Environment(\.scenePhase) private var scenePhase

    init() {
        configureAppearance()
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if authManager.isAuthenticated {
                    HomeView()
                        .environmentObject(authManager)
                } else {
                    LoginView()
                        .environmentObject(authManager)
                }
            }
            .task {
                await authManager.checkSession()
            }
            .onChange(of: scenePhase) { oldPhase, newPhase in
                handleScenePhaseChange(oldPhase: oldPhase, newPhase: newPhase)
            }
        }
    }

    /// Handle scene phase changes for token management
    private func handleScenePhaseChange(oldPhase: ScenePhase, newPhase: ScenePhase) {
        switch newPhase {
        case .active:
            // Re-check session when app becomes active (from background OR inactive)
            // This covers both returning from background and cold launches
            if oldPhase != .active {
                Task {
                    await authManager.checkSession()
                }
            }
        case .background, .inactive:
            break
        @unknown default:
            break
        }
    }

    /// Configure global app appearance
    private func configureAppearance() {
        // Navigation bar appearance
        let navAppearance = UINavigationBarAppearance()
        navAppearance.configureWithDefaultBackground()
        navAppearance.largeTitleTextAttributes = [
            .font: UIFont.systemFont(ofSize: 34, weight: .bold)
        ]
        navAppearance.titleTextAttributes = [
            .font: UIFont.systemFont(ofSize: 17, weight: .semibold)
        ]

        UINavigationBar.appearance().standardAppearance = navAppearance
        UINavigationBar.appearance().scrollEdgeAppearance = navAppearance
        UINavigationBar.appearance().compactAppearance = navAppearance

        // Tab bar appearance (for future use)
        let tabAppearance = UITabBarAppearance()
        tabAppearance.configureWithDefaultBackground()
        UITabBar.appearance().standardAppearance = tabAppearance
        UITabBar.appearance().scrollEdgeAppearance = tabAppearance
    }
}
