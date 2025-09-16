// Modern JavaScript features for testing advanced chunking
import { useState, useEffect } from 'react';
import axios from 'axios';
import { debounce } from 'lodash';

// ES6+ class with modern features
export class UserManager {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.cache = new Map();
        this.subscribers = new Set();
    }

    // Async method with error handling
    async fetchUser(userId) {
        if (this.cache.has(userId)) {
            return this.cache.get(userId);
        }

        try {
            const response = await axios.get(`${this.apiBaseUrl}/users/${userId}`);
            const user = response.data;
            this.cache.set(userId, user);
            this.notifySubscribers('userFetched', user);
            return user;
        } catch (error) {
            console.error('Failed to fetch user:', error);
            throw new Error(`User ${userId} not found`);
        }
    }

    // Method with destructuring and default parameters
    subscribe(callback, events = ['userFetched', 'userUpdated']) {
        this.subscribers.add({ callback, events });

        // Return unsubscribe function
        return () => {
            this.subscribers.delete({ callback, events });
        };
    }

    // Private method (using naming convention)
    notifySubscribers(event, data) {
        this.subscribers.forEach(({ callback, events }) => {
            if (events.includes(event)) {
                callback(event, data);
            }
        });
    }
}

// Arrow function with template literals
const createUserCard = (user) => {
    return `
        <div class="user-card" data-user-id="${user.id}">
            <h3>${user.name}</h3>
            <p>${user.email}</p>
            <span class="role">${user.role || 'User'}</span>
        </div>
    `;
};

// Higher-order function
const withLogging = (fn) => {
    return (...args) => {
        console.log(`Calling ${fn.name} with args:`, args);
        const result = fn(...args);
        console.log(`${fn.name} returned:`, result);
        return result;
    };
};

// Async arrow function with destructuring
const processUserData = async ({ users, transformFn, filterFn }) => {
    const validUsers = users.filter(filterFn);

    const processedUsers = await Promise.all(
        validUsers.map(async (user) => {
            const transformed = transformFn(user);
            const enriched = await enrichUserData(transformed);
            return enriched;
        })
    );

    return processedUsers.sort((a, b) => a.name.localeCompare(b.name));
};

// Function with complex parameter destructuring
function createApiClient({
    baseURL,
    timeout = 5000,
    retries = 3,
    headers = {},
    interceptors = {}
}) {
    const client = axios.create({ baseURL, timeout, headers });

    // Request interceptor
    if (interceptors.request) {
        client.interceptors.request.use(interceptors.request);
    }

    // Response interceptor with retry logic
    client.interceptors.response.use(
        (response) => response,
        async (error) => {
            const { config } = error;

            if (!config || !config.retry) {
                config.retry = 0;
            }

            if (config.retry < retries) {
                config.retry += 1;
                return client.request(config);
            }

            return Promise.reject(error);
        }
    );

    return client;
}

// Generator function
function* generateUserIds(start = 1, end = 100) {
    for (let id = start; id <= end; id++) {
        yield `user_${id.toString().padStart(3, '0')}`;
    }
}

// Object with computed properties and methods
const userUtils = {
    // Method shorthand
    validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    },

    // Computed property
    [`format${Date.now()}Name`]: (firstName, lastName) => {
        return `${lastName}, ${firstName}`;
    },

    // Arrow function as property
    capitalizeRole: (role) => role.charAt(0).toUpperCase() + role.slice(1),

    // Async method
    async enrichUserData(user) {
        const geoData = await this.fetchGeoData(user.ip);
        const preferences = await this.fetchUserPreferences(user.id);

        return {
            ...user,
            location: geoData,
            preferences,
            lastSeen: new Date().toISOString()
        };
    }
};

// Complex destructuring and spread operators
const { validateEmail, capitalizeRole, ...restUtils } = userUtils;

// IIFE with complex logic
const userModule = (() => {
    let privateCounter = 0;
    const privateCache = new WeakMap();

    return {
        createUser(userData) {
            const user = {
                id: ++privateCounter,
                ...userData,
                createdAt: Date.now()
            };

            privateCache.set(user, {
                access_count: 0,
                last_accessed: null
            });

            return user;
        },

        trackAccess(user) {
            if (privateCache.has(user)) {
                const metadata = privateCache.get(user);
                metadata.access_count++;
                metadata.last_accessed = Date.now();
            }
        }
    };
})();

// Default export with re-exports
export { createUserCard, withLogging, processUserData };
export default UserManager;
