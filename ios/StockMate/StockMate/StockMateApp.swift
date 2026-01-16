//
//  StockMateApp.swift
//  StockMate
//
//  Created by Nishad Lad on 12/10/25.
//

import SwiftUI
import BackgroundTasks

@main
struct StockMateApp: App {
    @StateObject private var authManager = AuthenticationManager.shared
    @Environment(\.scenePhase) private var scenePhase

    /// Background task identifier for token refresh
    private static let backgroundRefreshTaskId = "com.stockmate.tokenRefresh"

    init() {
        configureAppearance()
        registerBackgroundTasks()
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
        case .background:
            // Schedule background refresh when entering background
            scheduleBackgroundTokenRefresh()
        case .inactive:
            break
        @unknown default:
            break
        }
    }

    /// Register background task handlers
    private func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.backgroundRefreshTaskId,
            using: nil
        ) { task in
            guard let refreshTask = task as? BGAppRefreshTask else { return }
            handleBackgroundTokenRefresh(task: refreshTask)
        }
    }

    /// Schedule a background task to refresh tokens
    private func scheduleBackgroundTokenRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: Self.backgroundRefreshTaskId)
        // Schedule to run within the next hour, but let the system decide the exact time
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15 minutes minimum

        do {
            try BGTaskScheduler.shared.submit(request)
            print("Background token refresh scheduled")
        } catch {
            print("Failed to schedule background token refresh: \(error)")
        }
    }

    /// Handle the background token refresh task
    private static func handleBackgroundTokenRefresh(task: BGAppRefreshTask) {
        // Schedule the next refresh
        let app = StockMateApp()
        app.scheduleBackgroundTokenRefresh()

        // Create a task to refresh tokens
        let refreshOperation = Task {
            do {
                try await AuthenticationManager.shared.refreshAccessToken()
                print("Background token refresh successful")
                task.setTaskCompleted(success: true)
            } catch {
                print("Background token refresh failed: \(error)")
                task.setTaskCompleted(success: false)
            }
        }

        // Handle task expiration
        task.expirationHandler = {
            refreshOperation.cancel()
            task.setTaskCompleted(success: false)
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
