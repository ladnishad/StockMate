//
//  SignupView.swift
//  StockMate
//
//  Registration screen for new users.
//

import SwiftUI

struct SignupView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @Environment(\.dismiss) private var dismiss

    @State private var email = ""
    @State private var password = ""
    @State private var confirmPassword = ""
    @State private var errorMessage: String?
    @State private var successMessage: String?
    @State private var isLoading = false

    var body: some View {
        ZStack {
            Color(.systemBackground)
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 32) {
                    // Header
                    VStack(spacing: 8) {
                        Image(systemName: "person.badge.plus")
                            .font(.system(size: 50, weight: .light))
                            .foregroundStyle(.blue)

                        Text("Create Account")
                            .font(.title)
                            .fontWeight(.bold)

                        Text("Start your trading journey")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 40)

                    // Signup Form
                    VStack(spacing: 16) {
                        // Email Field
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Email")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundStyle(.secondary)

                            TextField("Enter your email", text: $email)
                                .textFieldStyle(.plain)
                                .textContentType(.emailAddress)
                                .keyboardType(.emailAddress)
                                .autocapitalization(.none)
                                .autocorrectionDisabled()
                                .padding()
                                .background(Color(.secondarySystemBackground))
                                .cornerRadius(12)
                        }

                        // Password Field
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Password")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundStyle(.secondary)

                            SecureField("Create a password", text: $password)
                                .textFieldStyle(.plain)
                                .textContentType(.newPassword)
                                .padding()
                                .background(Color(.secondarySystemBackground))
                                .cornerRadius(12)

                            Text("At least 6 characters")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }

                        // Confirm Password Field
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Confirm Password")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundStyle(.secondary)

                            SecureField("Confirm your password", text: $confirmPassword)
                                .textFieldStyle(.plain)
                                .textContentType(.newPassword)
                                .padding()
                                .background(Color(.secondarySystemBackground))
                                .cornerRadius(12)
                        }
                    }
                    .padding(.horizontal, 24)

                    // Password Mismatch Warning
                    if !confirmPassword.isEmpty && password != confirmPassword {
                        Text("Passwords don't match")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                            .padding(.horizontal, 24)
                    }

                    // Error Message
                    if let error = errorMessage {
                        Text(error)
                            .font(.footnote)
                            .foregroundStyle(.red)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }

                    // Success Message
                    if let success = successMessage {
                        VStack(spacing: 8) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.title)
                                .foregroundStyle(.green)

                            Text(success)
                                .font(.footnote)
                                .foregroundStyle(.green)
                                .multilineTextAlignment(.center)
                        }
                        .padding(.horizontal, 24)
                    }

                    // Create Account Button
                    Button {
                        Task {
                            await signup()
                        }
                    } label: {
                        HStack {
                            if isLoading {
                                ProgressView()
                                    .tint(.white)
                            } else {
                                Text("Create Account")
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isFormValid ? Color.blue : Color.blue.opacity(0.5))
                        .foregroundStyle(.white)
                        .cornerRadius(12)
                    }
                    .disabled(!isFormValid || isLoading)
                    .padding(.horizontal, 24)

                    // Back to Login Link
                    HStack(spacing: 4) {
                        Text("Already have an account?")
                            .foregroundStyle(.secondary)
                        Button("Sign In") {
                            dismiss()
                        }
                        .foregroundStyle(.blue)
                        .fontWeight(.medium)
                    }
                    .font(.subheadline)

                    Spacer()
                }
            }
        }
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Computed Properties

    private var isFormValid: Bool {
        !email.isEmpty &&
        !password.isEmpty &&
        !confirmPassword.isEmpty &&
        email.contains("@") &&
        password.count >= 6 &&
        password == confirmPassword
    }

    // MARK: - Actions

    private func signup() async {
        guard isFormValid else { return }

        isLoading = true
        errorMessage = nil
        successMessage = nil

        do {
            try await authManager.signup(email: email, password: password)
            // If signup succeeds without email confirmation, user is logged in
            // If email confirmation is required, we show success message
        } catch AuthError.emailConfirmationRequired {
            successMessage = "Check your email to confirm your account, then sign in."
        } catch let error as AuthError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = "An unexpected error occurred"
        }

        isLoading = false
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        SignupView()
            .environmentObject(AuthenticationManager.shared)
    }
}
