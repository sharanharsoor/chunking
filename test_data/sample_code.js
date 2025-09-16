// Sample JavaScript code for testing universal code chunking
class TaskManager {
    constructor() {
        this.tasks = [];
        this.nextId = 1;
        console.log('TaskManager initialized');
    }

    addTask(title, description = '') {
        const task = {
            id: this.nextId++,
            title: title,
            description: description,
            completed: false,
            createdAt: new Date()
        };

        this.tasks.push(task);
        return task;
    }

    completeTask(taskId) {
        const task = this.tasks.find(t => t.id === taskId);
        if (task) {
            task.completed = true;
            task.completedAt = new Date();
            return true;
        }
        return false;
    }

    getTasks(filter = 'all') {
        switch(filter) {
            case 'completed':
                return this.tasks.filter(task => task.completed);
            case 'pending':
                return this.tasks.filter(task => !task.completed);
            default:
                return this.tasks;
        }
    }

    removeTask(taskId) {
        const index = this.tasks.findIndex(t => t.id === taskId);
        if (index !== -1) {
            return this.tasks.splice(index, 1)[0];
        }
        return null;
    }
}

// Utility functions
const formatDate = (date) => {
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
};

const validateTask = (task) => {
    if (!task.title || task.title.trim() === '') {
        throw new Error('Task title is required');
    }

    if (task.title.length > 100) {
        throw new Error('Task title too long');
    }

    return true;
};

// Arrow function for processing tasks
const processTaskBatch = (tasks) => {
    return tasks.map(task => ({
        ...task,
        displayTitle: task.title.toUpperCase(),
        formattedDate: formatDate(task.createdAt),
        status: task.completed ? 'DONE' : 'PENDING'
    }));
};

// Main execution
function main() {
    const taskManager = new TaskManager();

    // Add some sample tasks
    taskManager.addTask('Learn JavaScript', 'Study modern JS features');
    taskManager.addTask('Build a web app', 'Create a todo application');
    taskManager.addTask('Write tests', 'Add unit tests for all functions');

    // Complete one task
    taskManager.completeTask(1);

    // Get and process tasks
    const allTasks = taskManager.getTasks();
    const processedTasks = processTaskBatch(allTasks);

    console.log('Processed tasks:', processedTasks);

    // Test filtering
    const completedTasks = taskManager.getTasks('completed');
    const pendingTasks = taskManager.getTasks('pending');

    console.log(`Total: ${allTasks.length}, Completed: ${completedTasks.length}, Pending: ${pendingTasks.length}`);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TaskManager,
        formatDate,
        validateTask,
        processTaskBatch
    };
} else {
    // Browser environment
    window.TaskManager = TaskManager;
    main();
}