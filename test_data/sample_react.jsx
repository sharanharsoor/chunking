// React JSX components for testing JSX-aware chunking
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import PropTypes from 'prop-types';

// Functional component with hooks
const UserProfile = ({ userId, onUserUpdate }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Custom hook usage
    const { permissions, hasPermission } = useUserPermissions(userId);

    useEffect(() => {
        const fetchUser = async () => {
            try {
                setLoading(true);
                const response = await fetch(`/api/users/${userId}`);
                if (!response.ok) {
                    throw new Error('Failed to fetch user');
                }
                const userData = await response.json();
                setUser(userData);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        if (userId) {
            fetchUser();
        }
    }, [userId]);

    const handleUpdateUser = useCallback((updates) => {
        setUser(prevUser => ({
            ...prevUser,
            ...updates
        }));
        onUserUpdate?.(updates);
    }, [onUserUpdate]);

    const formattedUser = useMemo(() => {
        if (!user) return null;

        return {
            ...user,
            fullName: `${user.firstName} ${user.lastName}`,
            initials: `${user.firstName[0]}${user.lastName[0]}`.toUpperCase()
        };
    }, [user]);

    if (loading) {
        return (
            <div className="loading-container">
                <Spinner size="large" />
                <p>Loading user profile...</p>
            </div>
        );
    }

    if (error) {
        return (
            <ErrorMessage
                message={error}
                onRetry={() => window.location.reload()}
            />
        );
    }

    if (!formattedUser) {
        return <div>User not found</div>;
    }

    return (
        <div className="user-profile">
            <UserAvatar
                src={formattedUser.avatar}
                alt={formattedUser.fullName}
                size="large"
                initials={formattedUser.initials}
            />

            <div className="user-info">
                <h1>{formattedUser.fullName}</h1>
                <p className="user-email">{formattedUser.email}</p>
                <UserRoleBadge role={formattedUser.role} />
            </div>

            {hasPermission('edit_profile') && (
                <UserEditForm
                    user={formattedUser}
                    onUpdate={handleUpdateUser}
                />
            )}

            <UserStats userId={userId} />
        </div>
    );
};

UserProfile.propTypes = {
    userId: PropTypes.string.isRequired,
    onUserUpdate: PropTypes.func
};

// Class component with lifecycle methods
class UserEditForm extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            formData: {
                firstName: props.user.firstName,
                lastName: props.user.lastName,
                email: props.user.email,
                bio: props.user.bio || ''
            },
            isDirty: false,
            isSubmitting: false
        };

        this.handleInputChange = this.handleInputChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    componentDidUpdate(prevProps) {
        if (prevProps.user.id !== this.props.user.id) {
            this.setState({
                formData: {
                    firstName: this.props.user.firstName,
                    lastName: this.props.user.lastName,
                    email: this.props.user.email,
                    bio: this.props.user.bio || ''
                },
                isDirty: false
            });
        }
    }

    handleInputChange(event) {
        const { name, value } = event.target;

        this.setState(prevState => ({
            formData: {
                ...prevState.formData,
                [name]: value
            },
            isDirty: true
        }));
    }

    async handleSubmit(event) {
        event.preventDefault();

        if (!this.state.isDirty) return;

        this.setState({ isSubmitting: true });

        try {
            const response = await fetch(`/api/users/${this.props.user.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.state.formData)
            });

            if (!response.ok) {
                throw new Error('Failed to update user');
            }

            const updatedUser = await response.json();
            this.props.onUpdate(updatedUser);
            this.setState({ isDirty: false });
        } catch (error) {
            console.error('Error updating user:', error);
        } finally {
            this.setState({ isSubmitting: false });
        }
    }

    render() {
        const { formData, isDirty, isSubmitting } = this.state;

        return (
            <form onSubmit={this.handleSubmit} className="user-edit-form">
                <div className="form-group">
                    <label htmlFor="firstName">First Name</label>
                    <input
                        type="text"
                        id="firstName"
                        name="firstName"
                        value={formData.firstName}
                        onChange={this.handleInputChange}
                        required
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="lastName">Last Name</label>
                    <input
                        type="text"
                        id="lastName"
                        name="lastName"
                        value={formData.lastName}
                        onChange={this.handleInputChange}
                        required
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="email">Email</label>
                    <input
                        type="email"
                        id="email"
                        name="email"
                        value={formData.email}
                        onChange={this.handleInputChange}
                        required
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="bio">Bio</label>
                    <textarea
                        id="bio"
                        name="bio"
                        value={formData.bio}
                        onChange={this.handleInputChange}
                        rows={4}
                        placeholder="Tell us about yourself..."
                    />
                </div>

                <div className="form-actions">
                    <button
                        type="submit"
                        disabled={!isDirty || isSubmitting}
                        className="btn btn-primary"
                    >
                        {isSubmitting ? 'Updating...' : 'Update Profile'}
                    </button>
                </div>
            </form>
        );
    }
}

UserEditForm.propTypes = {
    user: PropTypes.shape({
        id: PropTypes.string.isRequired,
        firstName: PropTypes.string.isRequired,
        lastName: PropTypes.string.isRequired,
        email: PropTypes.string.isRequired,
        bio: PropTypes.string
    }).isRequired,
    onUpdate: PropTypes.func.isRequired
};

// Higher-order component
const withLoadingState = (WrappedComponent) => {
    return function WithLoadingStateComponent(props) {
        const [isLoading, setIsLoading] = useState(false);

        const withLoading = useCallback((asyncFn) => {
            return async (...args) => {
                setIsLoading(true);
                try {
                    return await asyncFn(...args);
                } finally {
                    setIsLoading(false);
                }
            };
        }, []);

        return (
            <>
                {isLoading && <LoadingOverlay />}
                <WrappedComponent
                    {...props}
                    isLoading={isLoading}
                    withLoading={withLoading}
                />
            </>
        );
    };
};

// Custom hook
const useUserPermissions = (userId) => {
    const [permissions, setPermissions] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchPermissions = async () => {
            try {
                const response = await fetch(`/api/users/${userId}/permissions`);
                const perms = await response.json();
                setPermissions(perms);
            } catch (error) {
                console.error('Failed to fetch permissions:', error);
                setPermissions([]);
            } finally {
                setLoading(false);
            }
        };

        if (userId) {
            fetchPermissions();
        }
    }, [userId]);

    const hasPermission = useCallback((permission) => {
        return permissions.includes(permission);
    }, [permissions]);

    return { permissions, hasPermission, loading };
};

// Portal component
const Modal = ({ isOpen, onClose, children }) => {
    useEffect(() => {
        const handleEscape = (event) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleEscape);
            document.body.style.overflow = 'hidden';
        }

        return () => {
            document.removeEventListener('keydown', handleEscape);
            document.body.style.overflow = 'unset';
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return createPortal(
        <div className="modal-overlay" onClick={onClose}>
            <div
                className="modal-content"
                onClick={(e) => e.stopPropagation()}
            >
                <button
                    className="modal-close"
                    onClick={onClose}
                    aria-label="Close modal"
                >
                    ×
                </button>
                {children}
            </div>
        </div>,
        document.body
    );
};

Modal.propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    children: PropTypes.node.isRequired
};

// Context and Provider
const UserContext = React.createContext();

export const UserProvider = ({ children }) => {
    const [currentUser, setCurrentUser] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    const login = useCallback(async (credentials) => {
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(credentials)
            });

            if (response.ok) {
                const user = await response.json();
                setCurrentUser(user);
                setIsAuthenticated(true);
                return { success: true };
            } else {
                return { success: false, error: 'Invalid credentials' };
            }
        } catch (error) {
            return { success: false, error: error.message };
        }
    }, []);

    const logout = useCallback(() => {
        setCurrentUser(null);
        setIsAuthenticated(false);
    }, []);

    const value = {
        currentUser,
        isAuthenticated,
        login,
        logout
    };

    return (
        <UserContext.Provider value={value}>
            {children}
        </UserContext.Provider>
    );
};

// Export components
export { UserProfile, UserEditForm, Modal, withLoadingState, useUserPermissions };
export default UserProfile;
