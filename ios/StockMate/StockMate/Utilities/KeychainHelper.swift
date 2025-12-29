//
//  KeychainHelper.swift
//  StockMate
//
//  Secure storage helper using iOS Keychain for auth tokens.
//

import Foundation
import Security

final class KeychainHelper {
    static let shared = KeychainHelper()
    private init() {}

    // MARK: - Keychain Keys

    private enum Keys {
        static let accessToken = "com.stockmate.accessToken"
        static let refreshToken = "com.stockmate.refreshToken"
        static let userEmail = "com.stockmate.userEmail"
        static let userId = "com.stockmate.userId"
    }

    // MARK: - Token Management

    var accessToken: String? {
        get { read(key: Keys.accessToken) }
        set {
            if let value = newValue {
                save(key: Keys.accessToken, value: value)
            } else {
                delete(key: Keys.accessToken)
            }
        }
    }

    var refreshToken: String? {
        get { read(key: Keys.refreshToken) }
        set {
            if let value = newValue {
                save(key: Keys.refreshToken, value: value)
            } else {
                delete(key: Keys.refreshToken)
            }
        }
    }

    var userEmail: String? {
        get { read(key: Keys.userEmail) }
        set {
            if let value = newValue {
                save(key: Keys.userEmail, value: value)
            } else {
                delete(key: Keys.userEmail)
            }
        }
    }

    var userId: String? {
        get { read(key: Keys.userId) }
        set {
            if let value = newValue {
                save(key: Keys.userId, value: value)
            } else {
                delete(key: Keys.userId)
            }
        }
    }

    // MARK: - Save Auth Session

    func saveAuthSession(response: AuthResponse) {
        accessToken = response.accessToken
        refreshToken = response.refreshToken
        userEmail = response.user.email
        userId = response.user.id
    }

    // MARK: - Clear All

    func clearAll() {
        accessToken = nil
        refreshToken = nil
        userEmail = nil
        userId = nil
    }

    // MARK: - Check if Logged In

    var hasValidSession: Bool {
        return accessToken != nil && userId != nil
    }

    // MARK: - Private Keychain Operations

    private func save(key: String, value: String) {
        guard let data = value.data(using: .utf8) else { return }

        // Delete any existing item first
        delete(key: key)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]

        let status = SecItemAdd(query as CFDictionary, nil)
        if status != errSecSuccess {
            print("Keychain save error for \(key): \(status)")
        }
    }

    private func read(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            return nil
        }

        return value
    }

    private func delete(key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        SecItemDelete(query as CFDictionary)
    }
}
