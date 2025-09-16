package com.example.userservice;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.time.LocalDateTime;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Email;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

/**
 * User service for managing user operations.
 *
 * This service provides comprehensive user management functionality including
 * creation, authentication, profile management, and user lifecycle operations.
 *
 * @author Development Team
 * @version 1.0
 * @since 2024-01-01
 */
@Service
public class UserService {

    private static final String DEFAULT_ROLE = "USER";
    private static final int MAX_LOGIN_ATTEMPTS = 3;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    private final Map<String, Integer> loginAttempts = new ConcurrentHashMap<>();

    /**
     * Default constructor for UserService.
     */
    public UserService() {
        // Initialize service
    }

    /**
     * Constructor with repository injection.
     *
     * @param userRepository the user repository
     * @param passwordEncoder the password encoder
     */
    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Creates a new user with the provided information.
     *
     * @param userData the user data containing username, email, and password
     * @return the created user entity
     * @throws UserAlreadyExistsException if user with email already exists
     * @throws InvalidUserDataException if user data is invalid
     */
    @Transactional
    public User createUser(@NotNull UserCreationRequest userData) throws UserAlreadyExistsException {
        validateUserData(userData);

        if (userRepository.existsByEmail(userData.getEmail())) {
            throw new UserAlreadyExistsException("User with email " + userData.getEmail() + " already exists");
        }

        User user = new User();
        user.setUsername(userData.getUsername());
        user.setEmail(userData.getEmail());
        user.setPasswordHash(passwordEncoder.encode(userData.getPassword()));
        user.setRole(DEFAULT_ROLE);
        user.setCreatedAt(LocalDateTime.now());
        user.setUpdatedAt(LocalDateTime.now());
        user.setActive(true);

        return userRepository.save(user);
    }

    /**
     * Authenticates a user with email and password.
     *
     * @param email the user's email
     * @param password the user's password
     * @return the authenticated user if successful
     * @throws AuthenticationException if authentication fails
     */
    public User authenticateUser(@Email String email, @NotNull String password) throws AuthenticationException {
        if (isAccountLocked(email)) {
            throw new AuthenticationException("Account is temporarily locked due to too many failed attempts");
        }

        Optional<User> userOptional = userRepository.findByEmail(email);
        if (!userOptional.isPresent()) {
            recordFailedAttempt(email);
            throw new AuthenticationException("Invalid credentials");
        }

        User user = userOptional.get();
        if (!user.isActive()) {
            throw new AuthenticationException("Account is disabled");
        }

        if (!passwordEncoder.matches(password, user.getPasswordHash())) {
            recordFailedAttempt(email);
            throw new AuthenticationException("Invalid credentials");
        }

        clearFailedAttempts(email);
        user.setLastLoginAt(LocalDateTime.now());
        return userRepository.save(user);
    }

    /**
     * Updates user profile information.
     *
     * @param userId the user ID
     * @param updateRequest the update request containing new information
     * @return the updated user
     * @throws UserNotFoundException if user is not found
     */
    @Transactional
    public User updateUserProfile(Long userId, UserUpdateRequest updateRequest) throws UserNotFoundException {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));

        if (updateRequest.getUsername() != null) {
            user.setUsername(updateRequest.getUsername());
        }

        if (updateRequest.getEmail() != null) {
            if (!user.getEmail().equals(updateRequest.getEmail()) &&
                userRepository.existsByEmail(updateRequest.getEmail())) {
                throw new UserAlreadyExistsException("Email already in use");
            }
            user.setEmail(updateRequest.getEmail());
        }

        if (updateRequest.getFirstName() != null) {
            user.setFirstName(updateRequest.getFirstName());
        }

        if (updateRequest.getLastName() != null) {
            user.setLastName(updateRequest.getLastName());
        }

        user.setUpdatedAt(LocalDateTime.now());
        return userRepository.save(user);
    }

    /**
     * Deactivates a user account.
     *
     * @param userId the user ID to deactivate
     * @throws UserNotFoundException if user is not found
     */
    @Transactional
    public void deactivateUser(Long userId) throws UserNotFoundException {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));

        user.setActive(false);
        user.setUpdatedAt(LocalDateTime.now());
        userRepository.save(user);
    }

    /**
     * Retrieves all active users with pagination.
     *
     * @param pageNumber the page number (0-based)
     * @param pageSize the page size
     * @return a page of active users
     */
    public Page<User> getActiveUsers(int pageNumber, int pageSize) {
        Pageable pageable = PageRequest.of(pageNumber, pageSize, Sort.by("createdAt").descending());
        return userRepository.findByActiveTrue(pageable);
    }

    /**
     * Changes user password.
     *
     * @param userId the user ID
     * @param oldPassword the current password
     * @param newPassword the new password
     * @throws UserNotFoundException if user is not found
     * @throws AuthenticationException if old password is incorrect
     */
    @Transactional
    public void changePassword(Long userId, String oldPassword, String newPassword)
            throws UserNotFoundException, AuthenticationException {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));

        if (!passwordEncoder.matches(oldPassword, user.getPasswordHash())) {
            throw new AuthenticationException("Current password is incorrect");
        }

        validatePassword(newPassword);
        user.setPasswordHash(passwordEncoder.encode(newPassword));
        user.setUpdatedAt(LocalDateTime.now());
        userRepository.save(user);
    }

    // Private helper methods

    private void validateUserData(UserCreationRequest userData) throws InvalidUserDataException {
        if (userData.getUsername() == null || userData.getUsername().trim().isEmpty()) {
            throw new InvalidUserDataException("Username cannot be empty");
        }

        if (userData.getUsername().length() < 3 || userData.getUsername().length() > 50) {
            throw new InvalidUserDataException("Username must be between 3 and 50 characters");
        }

        if (userData.getEmail() == null || !isValidEmail(userData.getEmail())) {
            throw new InvalidUserDataException("Invalid email format");
        }

        validatePassword(userData.getPassword());
    }

    private void validatePassword(String password) throws InvalidUserDataException {
        if (password == null || password.length() < 8) {
            throw new InvalidUserDataException("Password must be at least 8 characters long");
        }

        if (!password.matches(".*[A-Z].*")) {
            throw new InvalidUserDataException("Password must contain at least one uppercase letter");
        }

        if (!password.matches(".*[a-z].*")) {
            throw new InvalidUserDataException("Password must contain at least one lowercase letter");
        }

        if (!password.matches(".*\\d.*")) {
            throw new InvalidUserDataException("Password must contain at least one digit");
        }
    }

    private boolean isValidEmail(String email) {
        return email.matches("^[A-Za-z0-9+_.-]+@([A-Za-z0-9.-]+\\.[A-Za-z]{2,})$");
    }

    private boolean isAccountLocked(String email) {
        return loginAttempts.getOrDefault(email, 0) >= MAX_LOGIN_ATTEMPTS;
    }

    private void recordFailedAttempt(String email) {
        loginAttempts.put(email, loginAttempts.getOrDefault(email, 0) + 1);
    }

    private void clearFailedAttempts(String email) {
        loginAttempts.remove(email);
    }

    // Inner class for user statistics
    public static class UserStatistics {
        private final int totalUsers;
        private final int activeUsers;
        private final int inactiveUsers;

        public UserStatistics(int totalUsers, int activeUsers, int inactiveUsers) {
            this.totalUsers = totalUsers;
            this.activeUsers = activeUsers;
            this.inactiveUsers = inactiveUsers;
        }

        public int getTotalUsers() { return totalUsers; }
        public int getActiveUsers() { return activeUsers; }
        public int getInactiveUsers() { return inactiveUsers; }
    }

    /**
     * Gets user statistics.
     *
     * @return user statistics object
     */
    public UserStatistics getUserStatistics() {
        int total = (int) userRepository.count();
        int active = userRepository.countByActiveTrue();
        int inactive = total - active;

        return new UserStatistics(total, active, inactive);
    }
}

/**
 * Enumeration for user roles.
 */
enum UserRole {
    ADMIN("Administrator"),
    MODERATOR("Moderator"),
    USER("Regular User"),
    GUEST("Guest User");

    private final String displayName;

    UserRole(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}

/**
 * Interface for user operations.
 */
interface UserOperations {
    User createUser(UserCreationRequest request) throws UserAlreadyExistsException;
    User authenticateUser(String email, String password) throws AuthenticationException;
    User updateUserProfile(Long userId, UserUpdateRequest request) throws UserNotFoundException;
    void deactivateUser(Long userId) throws UserNotFoundException;
}
