// Package userservice provides user management functionality
package userservice

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

// User represents a user in the system
type User struct {
	ID        string    `json:"id" db:"id"`
	Username  string    `json:"username" db:"username"`
	Email     string    `json:"email" db:"email"`
	Password  string    `json:"-" db:"password_hash"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// UserRepository defines the interface for user data operations
type UserRepository interface {
	Create(ctx context.Context, user *User) error
	GetByID(ctx context.Context, id string) (*User, error)
	GetByEmail(ctx context.Context, email string) (*User, error)
	Update(ctx context.Context, user *User) error
	Delete(ctx context.Context, id string) error
}

// UserService provides user business logic
type UserService struct {
	repo   UserRepository
	logger *log.Logger
}

// NewUserService creates a new user service instance
func NewUserService(repo UserRepository, logger *log.Logger) *UserService {
	return &UserService{
		repo:   repo,
		logger: logger,
	}
}

// CreateUser creates a new user with encrypted password
func (s *UserService) CreateUser(ctx context.Context, username, email, password string) (*User, error) {
	// Validate input
	if username == "" || email == "" || password == "" {
		return nil, fmt.Errorf("username, email, and password are required")
	}

	// Check if user already exists
	existingUser, err := s.repo.GetByEmail(ctx, email)
	if err != nil && err != sql.ErrNoRows {
		s.logger.Printf("Error checking existing user: %v", err)
		return nil, fmt.Errorf("failed to check existing user: %w", err)
	}
	if existingUser != nil {
		return nil, fmt.Errorf("user with email %s already exists", email)
	}

	// Hash password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		s.logger.Printf("Error hashing password: %v", err)
		return nil, fmt.Errorf("failed to hash password: %w", err)
	}

	// Create user
	user := &User{
		ID:        uuid.New().String(),
		Username:  username,
		Email:     email,
		Password:  string(hashedPassword),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.repo.Create(ctx, user); err != nil {
		s.logger.Printf("Error creating user: %v", err)
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	// Don't return password hash
	user.Password = ""
	return user, nil
}

// AuthenticateUser verifies user credentials
func (s *UserService) AuthenticateUser(ctx context.Context, email, password string) (*User, error) {
	user, err := s.repo.GetByEmail(ctx, email)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("invalid credentials")
		}
		s.logger.Printf("Error getting user by email: %v", err)
		return nil, fmt.Errorf("authentication failed: %w", err)
	}

	// Check password
	err = bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password))
	if err != nil {
		return nil, fmt.Errorf("invalid credentials")
	}

	// Don't return password hash
	user.Password = ""
	return user, nil
}

// GetUserByID retrieves a user by ID
func (s *UserService) GetUserByID(ctx context.Context, id string) (*User, error) {
	user, err := s.repo.GetByID(ctx, id)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("user not found")
		}
		s.logger.Printf("Error getting user by ID: %v", err)
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	// Don't return password hash
	user.Password = ""
	return user, nil
}

// UpdateUser updates user information
func (s *UserService) UpdateUser(ctx context.Context, id string, updates UserUpdateRequest) (*User, error) {
	user, err := s.repo.GetByID(ctx, id)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	// Apply updates
	if updates.Username != nil {
		user.Username = *updates.Username
	}
	if updates.Email != nil {
		user.Email = *updates.Email
	}
	user.UpdatedAt = time.Now()

	if err := s.repo.Update(ctx, user); err != nil {
		s.logger.Printf("Error updating user: %v", err)
		return nil, fmt.Errorf("failed to update user: %w", err)
	}

	// Don't return password hash
	user.Password = ""
	return user, nil
}

// DeleteUser removes a user from the system
func (s *UserService) DeleteUser(ctx context.Context, id string) error {
	if err := s.repo.Delete(ctx, id); err != nil {
		s.logger.Printf("Error deleting user: %v", err)
		return fmt.Errorf("failed to delete user: %w", err)
	}
	return nil
}

// UserUpdateRequest represents the fields that can be updated
type UserUpdateRequest struct {
	Username *string `json:"username,omitempty"`
	Email    *string `json:"email,omitempty"`
}

// Constants for user validation
const (
	MinUsernameLength = 3
	MaxUsernameLength = 50
	MinPasswordLength = 8
)

// ValidateUsername checks if username meets requirements
func ValidateUsername(username string) error {
	if len(username) < MinUsernameLength {
		return fmt.Errorf("username must be at least %d characters", MinUsernameLength)
	}
	if len(username) > MaxUsernameLength {
		return fmt.Errorf("username must be at most %d characters", MaxUsernameLength)
	}
	return nil
}

// ValidatePassword checks if password meets requirements
func ValidatePassword(password string) error {
	if len(password) < MinPasswordLength {
		return fmt.Errorf("password must be at least %d characters", MinPasswordLength)
	}
	return nil
}

// Helper functions

// generateID creates a new UUID string
func generateID() string {
	return uuid.New().String()
}

// hashPassword hashes a password using bcrypt
func hashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}
