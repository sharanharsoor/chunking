// TypeScript features for testing TypeScript-aware chunking
import { Observable, BehaviorSubject } from 'rxjs';
import { map, filter, debounceTime } from 'rxjs/operators';

// Interface definitions
interface User {
    id: string;
    name: string;
    email: string;
    role: UserRole;
    metadata?: UserMetadata;
}

interface UserMetadata {
    lastLogin: Date;
    preferences: {
        theme: 'light' | 'dark';
        language: string;
        notifications: boolean;
    };
}

// Type aliases
type UserRole = 'admin' | 'moderator' | 'user' | 'guest';
type ApiResponse<T> = {
    data: T;
    status: 'success' | 'error';
    message?: string;
};

type EventHandler<T> = (event: T) => void;

// Generic interface
interface Repository<T> {
    findById(id: string): Promise<T | null>;
    findAll(): Promise<T[]>;
    create(entity: Omit<T, 'id'>): Promise<T>;
    update(id: string, updates: Partial<T>): Promise<T>;
    delete(id: string): Promise<boolean>;
}

// Abstract class with generics
abstract class BaseService<T, K extends keyof T> {
    protected repository: Repository<T>;
    protected cache = new Map<string, T>();

    constructor(repository: Repository<T>) {
        this.repository = repository;
    }

    abstract validate(entity: T): boolean;
    abstract getKey(entity: T): T[K];

    async findById(id: string): Promise<T | null> {
        if (this.cache.has(id)) {
            return this.cache.get(id)!;
        }

        const entity = await this.repository.findById(id);
        if (entity) {
            this.cache.set(id, entity);
        }
        return entity;
    }

    protected invalidateCache(id: string): void {
        this.cache.delete(id);
    }
}

// Concrete class implementing the abstract class
class UserService extends BaseService<User, 'id'> {
    private eventSubject = new BehaviorSubject<User | null>(null);

    constructor(repository: Repository<User>) {
        super(repository);
    }

    validate(user: User): boolean {
        return (
            typeof user.name === 'string' &&
            user.name.length > 0 &&
            this.isValidEmail(user.email) &&
            ['admin', 'moderator', 'user', 'guest'].includes(user.role)
        );
    }

    getKey(user: User): string {
        return user.id;
    }

    // Method with complex type annotations
    async createUser(
        userData: Omit<User, 'id' | 'metadata'>,
        options: {
            sendWelcomeEmail?: boolean;
            assignDefaultRole?: boolean;
        } = {}
    ): Promise<ApiResponse<User>> {
        try {
            const newUser: User = {
                ...userData,
                id: this.generateId(),
                metadata: {
                    lastLogin: new Date(),
                    preferences: {
                        theme: 'light',
                        language: 'en',
                        notifications: true
                    }
                }
            };

            if (!this.validate(newUser)) {
                return {
                    data: newUser,
                    status: 'error',
                    message: 'Invalid user data'
                };
            }

            const created = await this.repository.create(newUser);
            this.eventSubject.next(created);

            if (options.sendWelcomeEmail) {
                await this.sendWelcomeEmail(created);
            }

            return {
                data: created,
                status: 'success'
            };
        } catch (error) {
            return {
                data: {} as User,
                status: 'error',
                message: error instanceof Error ? error.message : 'Unknown error'
            };
        }
    }

    // Generic method
    async processUsers<R>(
        processor: (user: User) => R,
        filter?: (user: User) => boolean
    ): Promise<R[]> {
        const users = await this.repository.findAll();
        const filteredUsers = filter ? users.filter(filter) : users;
        return filteredUsers.map(processor);
    }

    // Observable stream
    getUserStream(): Observable<User> {
        return this.eventSubject.asObservable().pipe(
            filter((user): user is User => user !== null),
            debounceTime(300),
            map(user => ({
                ...user,
                lastAccessed: new Date()
            }))
        );
    }

    private isValidEmail(email: string): boolean {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    private generateId(): string {
        return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private async sendWelcomeEmail(user: User): Promise<void> {
        // Implementation would go here
        console.log(`Sending welcome email to ${user.email}`);
    }
}

// Utility type manipulations
type RequiredUser = Required<User>;
type PartialUser = Partial<User>;
type UserEmail = Pick<User, 'email'>;
type UserWithoutId = Omit<User, 'id'>;

// Conditional types
type NonNullable<T> = T extends null | undefined ? never : T;
type Flatten<T> = T extends (infer U)[] ? U : T;

// Mapped types
type Readonly<T> = {
    readonly [P in keyof T]: T[P];
};

type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Decorator (experimental)
function logged(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = function (...args: any[]) {
        console.log(`Calling ${propertyName} with arguments:`, args);
        const result = method.apply(this, args);
        console.log(`${propertyName} returned:`, result);
        return result;
    };
}

// Class with decorators and access modifiers
class AdvancedUserManager {
    private readonly userService: UserService;
    private readonly logger: Console;

    constructor(
        userService: UserService,
        private readonly config: {
            enableLogging: boolean;
            cacheTimeout: number;
        }
    ) {
        this.userService = userService;
        this.logger = console;
    }

    @logged
    public async getUserWithLogging(id: string): Promise<User | null> {
        return this.userService.findById(id);
    }

    protected validateConfig(): boolean {
        return (
            typeof this.config.enableLogging === 'boolean' &&
            typeof this.config.cacheTimeout === 'number' &&
            this.config.cacheTimeout > 0
        );
    }
}

// Namespace (module)
namespace UserUtils {
    export function formatUserName(user: User): string {
        return `${user.name} (${user.role})`;
    }

    export function isAdmin(user: User): boolean {
        return user.role === 'admin';
    }

    export const DEFAULT_PREFERENCES: UserMetadata['preferences'] = {
        theme: 'light',
        language: 'en',
        notifications: true
    };
}

// Module augmentation
declare global {
    interface Window {
        userManager: AdvancedUserManager;
    }
}

// Export types and classes
export type { User, UserRole, ApiResponse, Repository };
export { UserService, AdvancedUserManager, UserUtils };
export default UserService;
