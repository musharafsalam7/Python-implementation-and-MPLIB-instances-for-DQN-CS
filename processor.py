import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import logging
from memory_profiler import profile
from functools import lru_cache
from collections import defaultdict, deque, namedtuple
from typing import List, Dict, Set, Optional, Tuple
import cProfile
import pstats

def profile_execution():
    profiler = cProfile.Profile()
    profiler.enable()
    # Run your solver
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Define experience tuple for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])
class Project:
    def __init__(self, project_id, num_activities, release_date, resource_usage):
        self.project_id = project_id  # Add project_id
        self.num_activities = num_activities
        self.release_date = release_date
        self.resource_usage = resource_usage
        self.activities = []
        self.critical_path_length = 0
        self.earliest_start = {}
        self.earliest_finish = {}
        self.latest_start = {}
        self.latest_finish = {}
        self.slack = {}
        self.available_resources = {}# Will be populated during parsing

class Task:
    def __init__(self, project_id, task_id, duration, resources, predecessors=None, successors=None):
        self.project_id = project_id
        self.task_id = task_id
        self.duration = duration
        self.resources = resources
        self.predecessors = predecessors or []
        self.successors = successors or []

class Solution:
    def __init__(self, task_sequence, projects):
        self.task_sequence = task_sequence
        self.projects = projects
        self.schedule = {}  # Dictionary to store start times
        self.makespan = float('inf')
        self.fitness = float('-inf')
        self.project_makespans = {}
        
    def is_valid(self):
        """Check if solution respects precedence constraints"""
        scheduled = set()
        for task in self.task_sequence:
            for pred in task.predecessors:
                if pred not in scheduled:
                    return False
            scheduled.add(task)
        return True

class Activity:
    def __init__(self, project_id, activity_id, duration, resource_requirements, successors):
        self.project_id = project_id
        self.activity_id = activity_id
        self.duration = duration
        self.resource_requirements = resource_requirements
        self.successors = successors
        self.predecessors = []
        self.task_id = f"{project_id}.{activity_id}"   # Unique identifier

def parse_activity_line(line, num_resources):
    """Parse a single activity line and return its components."""
    parts = line.strip().split()
    if len(parts) < num_resources + 2:  # Duration + resources + num_successors
        raise ValueError(f"Invalid activity line format: {line}")
    
    duration = int(parts[0])
    resource_requirements = [int(x) for x in parts[1:num_resources+1]]
    num_successors = int(parts[num_resources+1])
    
    
    successors = []
    if num_successors > 0:
        successor_parts = parts[num_resources+2:num_resources+2+num_successors]
        for succ in successor_parts:
            proj_id, act_id = map(int, succ.split(':'))
            successors.append((proj_id, act_id))
    
    return duration, resource_requirements, successors

class SchedulingDQN(nn.Module):
    """Enhanced DQN for scheduling decisions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(SchedulingDQN, self).__init__()
        
        # Extended network architecture
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        return self.network(x)

class ImprovedSchedulingStrategy:
    def __init__(self, projects: List[Project]):
        self.projects = projects
        self.resource_profiles = self._initialize_resource_profiles()
        
    def _initialize_resource_profiles(self) -> Dict[int, List[float]]:
        """Create resource usage profiles for better allocation"""
        profiles = {}
        max_time = 0
        
        # Find maximum possible timeline
        for project in self.projects:
            project_duration = sum(task.duration for task in project.tasks)
            max_time = max(max_time, project_duration + project.release_date)
            
        # Initialize profiles for each resource
        all_resources = set()
        for project in self.projects:
            all_resources.update(project.available_resources.keys())
            
        return {r: [0.0] * max_time for r in all_resources}
        
    def find_earliest_slot(self, task: Task, project: Project, current_time: int) -> int:
        """Find earliest feasible time slot considering resource constraints"""
        earliest = max(current_time, project.release_date)  # Changed from arrival_date
        
        while True:
            can_schedule = True
            for t in range(earliest, earliest + task.duration):
                for resource_id, amount in task.resources.items():
                    if t >= len(self.resource_profiles[resource_id]):
                        # Extend profile if needed
                        self.resource_profiles[resource_id].extend([0.0] * task.duration)
                    
                    if (self.resource_profiles[resource_id][t] + amount > 
                        project.available_resources[resource_id]):
                        can_schedule = False
                        break
                if not can_schedule:
                    break
                    
            if can_schedule:
                return earliest
            earliest += 1
            
    def update_resource_profile(self, task: Task, start_time: int):
        """Update resource usage profile after scheduling a task"""
        for t in range(start_time, start_time + task.duration):
            for resource_id, amount in task.resources.items():
                self.resource_profiles[resource_id][t] += amount
                
    def schedule_with_resource_leveling(self, solution: Solution) -> None:
        """Schedule tasks with improved resource leveling"""
        solution.schedule = {}
        scheduled_tasks = set()
        
        # Calculate critical paths for prioritization
        critical_paths = self._calculate_critical_paths()
        
        # Process projects in order of release
        sorted_projects = sorted(self.projects, key=lambda p: p.release_date)
        
        for project in sorted_projects:
            current_time = project.release_date
            project_tasks = set(task for task in project.tasks)
            
            while project_tasks:
                # Find available tasks
                available = [
                    task for task in project_tasks
                    if all(pred in scheduled_tasks for pred in task.predecessors)
                ]
                
                if not available:
                    break
                    
                # Sort by priority
                available.sort(key=lambda t: (
                    -critical_paths.get(t, 0),  # Critical path length
                    -sum(t.resources.values()),  # Resource demand
                    t.duration  # Duration as tiebreaker
                ))
                
                # Schedule highest priority task
                task = available[0]
                start_time = self.find_earliest_slot(task, project, current_time)
                
                # Update schedule and tracking
                solution.schedule[task.task_id] = start_time
                self.update_resource_profile(task, start_time)
                scheduled_tasks.add(task)
                project_tasks.remove(task)
                
                # Update current time if needed
                current_time = max(current_time, start_time + task.duration)
        
        # Calculate makespans
        self._calculate_solution_metrics(solution)
        
    def _calculate_critical_paths(self) -> Dict[Task, int]:
        """Calculate critical path lengths for all tasks"""
        critical_paths = {}
        
        def get_path_length(task: Task, memo=None):
            if memo is None:
                memo = {}
            if task in memo:
                return memo[task]
                
            if not task.successors:
                path_length = task.duration
            else:
                path_length = task.duration + max(
                    get_path_length(succ, memo) 
                    for succ in task.successors
                )
            memo[task] = path_length
            return path_length
        
        # Calculate for all tasks
        for project in self.projects:
            for task in project.tasks:
                if task not in critical_paths:
                    critical_paths[task] = get_path_length(task)
                    
        return critical_paths
        
    def _calculate_solution_metrics(self, solution: Solution) -> None:
        """Calculate and update solution metrics"""
        if not solution.schedule:
            solution.makespan = float('inf')
            solution.fitness = float('-inf')
            return
            
        # Calculate project completion times and makespans
        project_makespans = {}
        completion_times = {}
        
        for project in self.projects:
            project_tasks = [t for t in project.tasks if t.task_id in solution.schedule]
            if not project_tasks:
                continue
                
            completion_time = max(
                solution.schedule[t.task_id] + t.duration
                for t in project_tasks
            )
            completion_times[project.project_id] = completion_time
            
            # Calculate project makespan: Mi = cdi - release_date
            project_makespans[project.project_id] = completion_time - project.release_date
        
        # Store metrics
        solution.project_makespans = project_makespans
        solution.completion_times = completion_times
        
        # Calculate average project makespan
        if project_makespans:
            solution.makespan = sum(project_makespans.values()) / len(project_makespans)
            solution.fitness = -solution.makespan
        else:
            solution.makespan = float('inf')
            solution.fitness = float('-inf')

class Experience:
    """Modified Experience class to ensure hashable state storage"""
    def __init__(self, state, action, reward, next_state):
        # Convert state and next_state to tuples if they're tensors
        self.state = tuple(state.tolist()) if isinstance(state, torch.Tensor) else state
        self.action = action
        self.reward = reward
        self.next_state = tuple(next_state.tolist()) if isinstance(next_state, torch.Tensor) else next_state


class EnhancedCuckooSearch:
    """Enhanced Cuckoo Search with multiple scheduling strategies"""
    
    def __init__(self, max_moves: int, n_strategies: int = 4):
        self.max_moves = max_moves
        self.n_strategies = n_strategies
        
    def levy_flight(self) -> float:
        """Mantegna algorithm for Levy flight"""
        beta = 1.5
        sigma_u = ((math.gamma(1 + beta) * math.sin(math.pi * beta / 2)) / 
                  (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1
        
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)
        
        step = u / abs(v) ** (1 / beta)
        return step
        
    def apply_scheduling_strategy(self, 
                                solution: Solution, 
                                strategy: int,
                                levy_step: float) -> Solution:
        """Apply different scheduling strategies based on strategy index"""
        try:
            new_sequence = solution.task_sequence.copy()
            
            if strategy == 0:
                # Resource-based grouping
                resource_groups = defaultdict(list)
                for task in new_sequence:
                    key = tuple(sorted(task.resources.items()))
                    resource_groups[key].append(task)
                
                # Reorder within groups
                new_tasks = []
                for group in resource_groups.values():
                    if random.random() < 0.5:
                        group.sort(key=lambda t: sum(t.resources.values()))
                    new_tasks.extend(group)
                
                new_sequence = new_tasks
                
            elif strategy == 1:
                # Critical path priority
                tasks_with_priority = [(task, len(task.successors)) 
                                     for task in new_sequence]
                tasks_with_priority.sort(key=lambda x: -x[1])
                new_sequence = [t[0] for t in tasks_with_priority]
                
            elif strategy == 2:
                # Resource balancing
                n = len(new_sequence)
                n_swaps = int(levy_step * n * 0.2)
                
                for _ in range(n_swaps):
                    i, j = random.sample(range(n), 2)
                    if (self.is_valid_position(new_sequence[i], j, new_sequence) and
                        self.is_valid_position(new_sequence[j], i, new_sequence)):
                        new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
                        
            else:
                # Project-based grouping
                project_groups = defaultdict(list)
                for task in new_sequence:
                    project_groups[task.project_id].append(task)
                
                # Random project priority
                project_order = list(project_groups.keys())
                random.shuffle(project_order)
                
                new_sequence = []
                for proj_id in project_order:
                    new_sequence.extend(project_groups[proj_id])
                    
            return Solution(new_sequence, solution.projects)
            
        except Exception as e:
            logger.error(f"Error in apply_scheduling_strategy: {str(e)}")
            return solution
            
    def is_valid_position(self, task: Task, position: int, sequence: List[Task]) -> bool:
        """Check if position is valid for task regarding precedence constraints"""
        for pred in task.predecessors:
            if pred in sequence[position:]:
                return False
        for succ in task.successors:
            if succ in sequence[:position]:
                return False
        return True
        
    def apply_cs(self, solution: Solution, strategy: int) -> Solution:
        """Apply Cuckoo Search with specified strategy"""
        try:
            levy_step = self.levy_flight()
            
            # Apply selected strategy
            new_solution = self.apply_scheduling_strategy(
                solution, 
                strategy,
                levy_step
            )
            
            # Ensure solution validity
            if not new_solution.is_valid():
                return solution
                
            return new_solution
            
        except Exception as e:
            logger.error(f"Error in apply_cs: {str(e)}")
            return solution


class DQNCSHybrid:
    def __init__(self, 
                pop_size: int = 80,
                max_fes_num: int = 100000,
                n_iter: int = 100,
                start_num: int = 40,
                update_num: int = 10,
                max_moves: int = 6,
                n_strategies: int = 4,
                hidden_size: int = 128,
                memory_size: int = 10000,
                batch_size: int = 32,
                gamma: float = 0.99,
                learning_rate: float = 0.001):
                
        # Core parameters
        self.pop_size = pop_size
        self.max_fes_num = max_fes_num
        self.n_iter = n_iter
        self.start_num = start_num
        self.update_num = update_num
        self.batch_size = batch_size
        
        # Initialize CS
        self.cs = EnhancedCuckooSearch(max_moves, n_strategies)
        
        # DQN state and action spaces
        self.state_size = 6
        self.action_size = n_strategies
        
        # Initialize memory and parameters
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        
        # Create networks
        self.dqn = SchedulingDQN(self.state_size, self.action_size, hidden_size)
        self.target_dqn = SchedulingDQN(self.state_size, self.action_size, hidden_size)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        
        self.projects = []
        self.valid_tasks = []
        
    def _get_path_length_safe(self, task_id: str, memo: Optional[Dict[str, int]] = None) -> int:
        """Calculate path length using task IDs with proper memoization"""
        if memo is None:
            memo = {}
        
        # Return memoized result if available
        if task_id in memo:
            return memo[task_id]
            
        # Find the task object
        current_task = None
        for project in self.projects:
            for task in project.tasks:
                if task.task_id == task_id:
                    current_task = task
                    break
            if current_task:
                break
                
        if not current_task:
            return 0
            
        # Calculate path length
        successors = getattr(current_task, 'successors', [])
        if not successors:
            path_length = current_task.duration
        else:
            # Calculate lengths for all successors
            max_successor_length = 0
            for successor in successors:
                # Ensure we have a valid task_id
                succ_id = getattr(successor, 'task_id', None)
                if succ_id:
                    successor_length = self._get_path_length_safe(succ_id, memo)
                    max_successor_length = max(max_successor_length, successor_length)
            
            path_length = current_task.duration + max_successor_length
            
        # Store result in memo before returning
        memo[task_id] = path_length
        return path_length

    def get_extended_state(self, solution: Solution) -> torch.Tensor:
        """Calculate extended state features with improved path length calculation"""
        try:
            # Initialize default values
            state_values = [0.0] * self.state_size
            
            if solution.makespan != float('inf'):
                # Get previous states from memory
                similar_solutions = []
                for exp in self.memory:
                    if isinstance(exp, Experience):
                        if isinstance(exp.state, tuple):
                            state_tensor = torch.tensor(exp.state)
                            similar_solutions.append(state_tensor)
                        elif isinstance(exp.state, torch.Tensor):
                            similar_solutions.append(exp.state)
                
                # Calculate diversity
                if similar_solutions:
                    makespans = [s[1].item() for s in similar_solutions]
                    mean_makespan = np.mean(makespans)
                    if mean_makespan > 0:
                        state_values[0] = min(abs(solution.makespan - mean_makespan) / mean_makespan, 1.0)
                
                # Calculate quality
                total_duration = sum(task.duration for project in self.projects
                                for task in project.tasks)
                if total_duration > 0:
                    state_values[1] = min(solution.makespan / total_duration, 1.0)
                
                # Calculate resource usage
                if solution.schedule:
                    total_resources = 0
                    used_resources = 0
                    for project in self.projects:
                        for task in project.tasks:
                            if task.task_id in solution.schedule:
                                used_resources += sum(task.resources.values()) * task.duration
                            total_resources += sum(task.resources.values()) * task.duration
                    if total_resources > 0:
                        state_values[2] = min(used_resources / total_resources, 1.0)
                
                # Calculate critical path ratio using shared memo dictionary
                memo = {}  # Create a single memo dictionary for all path calculations
                max_path_length = 0
                for project in self.projects:
                    for task in project.tasks:
                        # Use the same memo dictionary for all calculations
                        path_length = self._get_path_length_safe(task.task_id, memo)
                        max_path_length = max(max_path_length, path_length)
                
                if solution.makespan > 0:
                    state_values[3] = min(max_path_length / solution.makespan, 1.0)
                
                # Calculate project balance
                if solution.schedule:
                    completion_times = {}
                    for project in self.projects:
                        proj_tasks = [t for t in project.tasks if t.task_id in solution.schedule]
                        if proj_tasks:
                            completion_time = max(solution.schedule[t.task_id] + t.duration 
                                            for t in proj_tasks)
                            completion_times[project.project_id] = completion_time
                    
                    if completion_times:
                        times = list(completion_times.values())
                        mean_time = np.mean(times)
                        if mean_time > 0:
                            state_values[4] = min(np.std(times) / mean_time, 1.0)
                
                # Calculate progress
                if self.max_fes_num > 0:
                    state_values[5] = min(len(self.memory) / self.max_fes_num, 1.0)
            
            return torch.tensor(state_values, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Error in get_extended_state: {str(e)}")
            logger.exception("Full traceback:")
            return torch.zeros(self.state_size, dtype=torch.float32)

        
    def select_strategy(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Select scheduling strategy using epsilon-greedy policy"""
        try:
            if random.random() < epsilon:
                return random.randint(0, self.action_size - 1)
                
            with torch.no_grad():
                q_values = self.dqn(state)
                return q_values.argmax().item()
                
        except Exception as e:
            logger.error(f"Error in select_strategy: {str(e)}")
            return random.randint(0, self.action_size - 1)
            
    def calculate_reward(self, 
                        old_solution: Solution,
                        new_solution: Solution,
                        strategy: int) -> float:
        """Calculate reward based on multiple objectives"""
        try:
            if old_solution.makespan == float('inf') or new_solution.makespan == float('inf'):
                return 0.0
                
            # Improvement in makespan
            makespan_improvement = ((old_solution.makespan - new_solution.makespan) / 
                                  old_solution.makespan)
            
            # Resource utilization improvement
            old_util = self._calculate_resource_utilization(old_solution)
            new_util = self._calculate_resource_utilization(new_solution)
            resource_improvement = new_util - old_util
            
            # Project balance improvement
            old_balance = self._calculate_project_balance(old_solution)
            new_balance = self._calculate_project_balance(new_solution)
            balance_improvement = new_balance - old_balance
            
            # Weight the components
            reward = (0.6 * makespan_improvement + 
                     0.2 * resource_improvement +
                     0.2 * balance_improvement)
            
            return float(np.clip(reward, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error in calculate_reward: {str(e)}")
            return 0.0
            
    def _calculate_resource_utilization(self, solution: Solution) -> float:
        """Calculate resource utilization ratio"""
        if not solution.schedule:
            return 0.0
            
        total_available = 0
        total_used = 0
        
        for project in self.projects:
            for resource_id, capacity in project.available_resources.items():
                total_available += capacity * solution.makespan
                
                for task in project.tasks:
                    if task.task_id in solution.schedule:
                        start = solution.schedule[task.task_id]
                        if resource_id in task.resources:
                            total_used += task.resources[resource_id] * task.duration
                            
        return total_used / total_available if total_available > 0 else 0.0
        
    def _calculate_project_balance(self, solution: Solution) -> float:
        """Calculate project completion balance"""
        if not solution.schedule:
            return 0.0
            
        project_makespans = []
        for project in self.projects:
            proj_tasks = [t for t in project.tasks if t.task_id in solution.schedule]
            if proj_tasks:
                proj_end = max(solution.schedule[t.task_id] + t.duration 
                             for t in proj_tasks)
                project_makespans.append(proj_end)
                
        if not project_makespans:
            return 0.0
            
        mean_makespan = np.mean(project_makespans)
        variance = np.mean([(m - mean_makespan) ** 2 for m in project_makespans])
        
        return 1.0 / (1.0 + variance)  # Higher value means better balance
            
    def train_dqn(self) -> float:
        """Train DQN with improved state handling"""
        try:
            if len(self.memory) < self.batch_size:
                return 0.0
                
            batch = random.sample(self.memory, self.batch_size)
            
            # Convert states back to tensors if stored as tuples
            states = torch.stack([
                torch.tensor(exp.state) if isinstance(exp.state, tuple)
                else exp.state
                for exp in batch
            ])
            
            actions = torch.tensor([exp.action for exp in batch])
            rewards = torch.tensor([exp.reward for exp in batch])
            next_states = torch.stack([
                torch.tensor(exp.next_state) if isinstance(exp.next_state, tuple)
                else exp.next_state
                for exp in batch
            ])
            
            # Rest of training logic remains the same
            current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.dqn(next_states).argmax(1)
                next_q = self.target_dqn(next_states)
                max_next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze()
                target_q = rewards + self.gamma * max_next_q
            
            loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in train_dqn: {str(e)}")
            return 0.0
            
    
    def schedule_solution(self, solution: Solution) -> None:
        try:
            import numpy as np
            
            # Pre-calculate project and task mappings
            project_tasks = {p.project_id: p.tasks for p in self.projects}
            task_durations = {t.task_id: t.duration for p in self.projects for t in p.tasks}
            
            # Create resource availability matrix
            max_time = sum(task_durations.values())
            n_resources = max(res_id for p in self.projects 
                            for t in p.tasks 
                            for res_id in t.resources.keys()) + 1
            
            resource_usage = np.zeros((max_time, n_resources), dtype=np.int32)
            resource_limits = np.array([max(p.available_resources.get(i, 0) 
                                        for p in self.projects) 
                                    for i in range(n_resources)])
            
            # Initialize schedule
            solution.schedule = {}
            scheduled_tasks = set()
            
            # Process tasks in sequence
            for task in solution.task_sequence:
                project = next(p for p in self.projects if p.project_id == task.project_id)
                
                # Find earliest feasible start time
                earliest_start = max(
                    project.release_date,
                    max((solution.schedule.get(pred.task_id, 0) + pred.duration 
                        for pred in task.predecessors), default=0)
                )
                
                # Find first viable time slot using vectorized operations
                start_time = earliest_start
                duration = task.duration
                resources = task.resources
                
                while True:
                    end_time = start_time + duration
                    window = resource_usage[start_time:end_time]
                    
                    # Check resource constraints vectorized
                    can_schedule = True
                    for res_id, amount in resources.items():
                        if np.any(window[:, res_id] + amount > resource_limits[res_id]):
                            can_schedule = False
                            break
                    
                    if can_schedule:
                        break
                        
                    start_time += 1
                    if start_time > earliest_start + max_time:
                        solution.makespan = float('inf')
                        solution.fitness = float('-inf')
                        return
                
                # Update schedule and resource usage
                solution.schedule[task.task_id] = start_time
                for res_id, amount in resources.items():
                    resource_usage[start_time:end_time, res_id] += amount
                
                scheduled_tasks.add(task)
            
            # Calculate metrics
            project_makespans = {}
            for project in self.projects:
                proj_tasks = [t for t in project.tasks if t.task_id in solution.schedule]
                if proj_tasks:
                    completion_time = max(solution.schedule[t.task_id] + t.duration 
                                    for t in proj_tasks)
                    project_makespans[project.project_id] = completion_time - project.release_date
            
            if project_makespans:
                solution.makespan = sum(project_makespans.values()) / len(project_makespans)
                solution.fitness = -solution.makespan
                solution.project_makespans = project_makespans
            else:
                solution.makespan = float('inf')
                solution.fitness = float('-inf')
                
        except Exception as e:
            logger.error(f"Error in scheduling: {str(e)}")
            solution.makespan = float('inf')
            solution.fitness = float('-inf')

# Add debug logging to verify calculations
    def solve(self, projects: List[Project]) -> Solution:
        """
        Main solving method implementing the hybrid DQN-CS approach with 100k generation limit
        """
        try:
            self.projects = projects
            total_generations = 0
            best_solution = None
            best_makespan = float('inf')
            
            # Initialize population
            logger.info("Starting population initialization...")
            population = self.initialize_population()
            if not population:
                raise ValueError("Failed to initialize valid population")
                
            # Update generation count and find initial best
            total_generations += len(population)
            best_solution = min(population, key=lambda x: x.makespan)
            best_makespan = best_solution.makespan
            logger.info(f"Initial population size: {len(population)}")
            logger.info(f"Initial best makespan: {best_makespan}")
            logger.info(f"Generations used in initialization: {total_generations}/{self.max_fes_num}")
            
            # Learning phase
            learning_count = 0
            epsilon = 0.3  # Initial exploration rate
            
            while learning_count < self.start_num and total_generations < self.max_fes_num:
                logger.info(f"Learning iteration {learning_count + 1}/{self.start_num}")
                logger.info(f"Current generations: {total_generations}/{self.max_fes_num}")
                
                new_population = []
                improvements = 0
                
                # Process each solution in population
                for solution in population:
                    if total_generations >= self.max_fes_num:
                        break
                        
                    # Get state and select strategy
                    state = self.get_extended_state(solution)
                    strategy = self.select_strategy(state, epsilon=epsilon)
                    
                    # Apply selected strategy using CS
                    new_solution = self.cs.apply_cs(solution, strategy)
                    self.schedule_solution(new_solution)
                    total_generations += 1
                    
                    if new_solution.is_valid():
                        # Calculate reward and store experience
                        reward = self.calculate_reward(solution, new_solution, strategy)
                        next_state = self.get_extended_state(new_solution)
                        self.memory.append(Experience(state, strategy, reward, next_state))
                        
                        # Update best solution if improved
                        if new_solution.makespan < best_makespan:
                            best_solution = new_solution
                            best_makespan = new_solution.makespan
                            improvements += 1
                            logger.info(f"New best makespan: {best_makespan}")
                        
                        new_population.append(new_solution)
                
                # Update population
                if new_population:
                    population = new_population
                
                # Train DQN if enough experiences
                if len(self.memory) >= self.batch_size:
                    loss = self.train_dqn()
                    logger.debug(f"Training loss: {loss:.4f}")
                
                # Update epsilon (exploration rate)
                epsilon = max(0.1, epsilon * 0.95)
                
                # Log progress
                logger.info(f"Improvements in iteration: {improvements}")
                learning_count += 1
            
            # Optimization phase
            logger.info("Starting optimization phase...")
            epsilon = 0.1  # Lower exploration for optimization
            
            for iteration in range(self.n_iter):
                if total_generations >= self.max_fes_num:
                    logger.info("Reached maximum generations during optimization")
                    break
                
                if iteration % self.update_num == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())
                    logger.info(f"Optimization iteration {iteration + 1}/{self.n_iter}")
                    logger.info(f"Current generations: {total_generations}/{self.max_fes_num}")
                
                new_population = []
                improvements = 0
                
                for solution in population:
                    if total_generations >= self.max_fes_num:
                        break
                    
                    state = self.get_extended_state(solution)
                    strategy = self.select_strategy(state, epsilon=epsilon)
                    
                    new_solution = self.cs.apply_cs(solution, strategy)
                    self.schedule_solution(new_solution)
                    total_generations += 1
                    
                    if new_solution.is_valid():
                        if new_solution.makespan < best_makespan:
                            best_solution = new_solution
                            best_makespan = new_solution.makespan
                            improvements += 1
                            logger.info(f"New best makespan: {best_makespan}")
                        
                        new_population.append(new_solution)
                
                # Update population
                if new_population:
                    population = new_population
                
                # Early stopping if no improvements
                if improvements == 0 and iteration > self.n_iter // 2:
                    logger.info("Early stopping due to no improvements")
                    break
            
            logger.info(f"Final generations used: {total_generations}/{self.max_fes_num}")
            logger.info(f"Final best makespan: {best_makespan}")
            
            return best_solution
            
        except Exception as e:
            logger.error(f"Error in solve: {str(e)}")
            raise e

    def initialize_population(self) -> List[Solution]:
        """Initialize diverse population with improved generation management"""
        try:
            population = []
            generation_budget = min(self.pop_size * 2, self.max_fes_num // 20)
            attempts = 0
            makespans = set()
            
            # Debug log
            logger.info(f"Starting initialization with budget: {generation_budget}")
            
            # Get list of all tasks
            all_tasks = []
            for project in self.projects:
                all_tasks.extend(project.tasks)
            self.valid_tasks = all_tasks
            
            # Debug log
            logger.info(f"Total tasks to schedule: {len(all_tasks)}")
            
            # First phase: Generate basic solutions
            while len(population) < self.pop_size // 2 and attempts < generation_budget:
                # Debug log
                if attempts % 10 == 0:
                    logger.info(f"Generation attempt {attempts}, population size: {len(population)}")
                
                sequence = generate_valid_sequence(self.valid_tasks)
                if not sequence:
                    logger.warning("Failed to generate valid sequence")
                    continue
                
                solution = Solution(sequence, self.projects)
                self.schedule_solution(solution)
                attempts += 1
                
                # Debug log solution validity and makespan
                logger.debug(f"Solution valid: {solution.is_valid()}, makespan: {solution.makespan}")
                
                if (solution.makespan != float('inf') and 
                    solution.is_valid() and 
                    (solution.makespan not in makespans or len(makespans) < 5)):
                    
                    population.append(solution)
                    makespans.add(solution.makespan)
                    logger.info(f"Generated base solution {len(population)}/{self.pop_size} "
                            f"with makespan {solution.makespan}")
            
            # Add debug check for empty population
            if not population:
                logger.error("Failed to generate any valid solutions in first phase")
                return []
                
            # Debug log before second phase
            logger.info(f"Completed first phase with {len(population)} solutions")
            
            return population
            
        except Exception as e:
            logger.error(f"Error in initialize_population: {str(e)}")
            logger.exception("Full traceback:")  # This will log the full stack trace
            return []
class RCMPSPMetrics:
    def __init__(self, projects, solution):
        self.projects = projects
        self.solution = solution
        self.metrics = {}

    def calculate_cp1(self):
        def get_cp_length(task, memo=None):
            if memo is None:
                memo = {}
            if task in memo:
                return memo[task]
                
            # Calculate path including cross-project dependencies
            if not task.successors:
                cp_length = task.duration
            else:
                cp_length = task.duration + max(
                    get_cp_length(succ, memo) 
                    for succ in task.successors
                )
            memo[task] = cp_length
            return cp_length
        
        # Find the maximum critical path across all starting tasks
        max_cp = 0
        for project in self.projects:
            start_tasks = [t for t in project.tasks if not t.predecessors]
            if start_tasks:
                project_cp = max(get_cp_length(task) for task in start_tasks)
                max_cp = max(max_cp, project_cp)
        
        self.metrics['CP1'] = max_cp
        return self.metrics['CP1']

    def calculate_rlb1(self):
        """Calculate Resource Load Balance 1 (RLB1) metric"""
        resource_types = set()
        for project in self.projects:
            resource_types.update(project.available_resources.keys())
        
        max_resource_load = 0
        
        for resource_id in resource_types:
            total_work = 0
            total_capacity = 0
            
            # Calculate total work and capacity across all projects
            for project in self.projects:
                # Sum resource requirements * duration for each task
                for task in project.tasks:
                    if resource_id in task.resources:
                        total_work += task.resources[resource_id] * task.duration
                
                # Sum resource capacity if project uses this resource
                if resource_id in project.available_resources:
                    total_capacity += project.available_resources[resource_id]
            
            # Calculate load for this resource type
            if total_capacity > 0:
                resource_load = total_work / total_capacity
                max_resource_load = max(max_resource_load, resource_load)
        
        # RLB1 is the ceiling of the maximum resource load
        self.metrics['RLB1'] = math.ceil(max_resource_load)
        return self.metrics['RLB1']

    def calculate_project_end_times(self):
        """Helper to calculate project completion times"""
        project_ends = {}
        for project in self.projects:
            project_tasks = [t for t in project.tasks if t.task_id in self.solution.schedule]
            if project_tasks:
                end_times = [
                    self.solution.schedule[t.task_id] + t.duration 
                    for t in project_tasks
                ]
                project_ends[project.project_id] = max(end_times)
                logger.debug(f"Project {project.project_id} end times: {end_times}")
                logger.debug(f"Project {project.project_id} makespan: {project_ends[project.project_id]}")
        return project_ends

    def calculate_tpm(self):
        """Total Portfolio Makespan"""
        project_ends = self.calculate_project_end_times()
        self.metrics['TPM'] = max(project_ends.values()) if project_ends else 0
        return self.metrics['TPM']

    def calculate_apm(self):
        """Average Project Makespan"""
        project_ends = self.calculate_project_end_times()
        self.metrics['APM'] = sum(project_ends.values()) / len(project_ends) if project_ends else 0
        return self.metrics['APM']

    def calculate_delays_and_gaps(self, reference_value, metric_prefix):
        """Calculate delays and gaps using the provided reference value"""
        if not reference_value or metric_prefix not in ['CP1', 'RLB1']:
            return
        
        project_ends = self.calculate_project_end_times()
        if not project_ends:
            return

        project_delays = []
        for project_id, end_time in project_ends.items():
            project = next(p for p in self.projects if p.project_id == project_id)
            project_makespan = end_time - project.release_date
            
            if metric_prefix == 'RLB1':
                # For RLB1, normalize the delay calculation
                reference = reference_value * project.critical_path_length
                delay = max(0, (project_makespan - reference) / reference_value)
            else:
                # For CP1, keep original calculation
                delay = max(0, project_makespan - reference_value)
                
            project_delays.append(delay)

        # Calculate metrics
        n_projects = len(project_delays)
        
        if metric_prefix == 'RLB1':
            # Adjusted calculations for RLB1
            self.metrics[f'APD_{metric_prefix}'] = sum(project_delays) / n_projects
            self.metrics[f'MaxPD_{metric_prefix}'] = max(project_delays)
            self.metrics[f'SPD_{metric_prefix}'] = sum(d * d for d in project_delays) / n_projects
            self.metrics[f'ARG_{metric_prefix}'] = sum(project_delays) / n_projects
            self.metrics[f'MaxRG_{metric_prefix}'] = max(project_delays)
        else:
            # Original calculations for CP1
            self.metrics[f'APD_{metric_prefix}'] = sum(project_delays) / n_projects
            self.metrics[f'MaxPD_{metric_prefix}'] = max(project_delays)
            self.metrics[f'SPD_{metric_prefix}'] = sum(d * d for d in project_delays)
            self.metrics[f'ARG_{metric_prefix}'] = (sum(d / reference_value for d in project_delays) / 
                                                n_projects if reference_value > 0 else 0)
            self.metrics[f'MaxRG_{metric_prefix}'] = (max(d / reference_value for d in project_delays) 
                                                    if reference_value > 0 else 0)

    def calculate_all_metrics(self):
        """Calculate all metrics with both CP1 and RLB1 references"""
        # Calculate base metrics
        cp1 = self.calculate_cp1()
        rlb1 = self.calculate_rlb1()
        self.calculate_tpm()
        self.calculate_apm()
        
        # Calculate metrics using CP1 as reference
        self.calculate_delays_and_gaps(cp1, 'CP1')
        
        # Calculate metrics using RLB1 as reference
        self.calculate_delays_and_gaps(rlb1, 'RLB1')
        
        return self.metrics

    def analyze_results(self):
        metrics = self.calculate_all_metrics()
        analysis = {'metrics': metrics, 'insights': [], 'recommendations': []}
        
        if metrics['APD_CP1'] > 0.2 * metrics['CP1']:
            analysis['insights'].append("High project delays relative to critical path")
            
        if metrics['ARG_RLB1'] > 0.3:
            analysis['insights'].append("Significant resource utilization gap")
            
        if metrics['MaxPD_CP1'] > 2 * metrics['APD_CP1']:
            analysis['insights'].append("Large variation in project delays")
            
        return analysis
    
def parse_rcmp_file(file_path):
    """Parse RCMP file and return projects"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    current_line = 0
    num_projects = int(lines[current_line])
    current_line += 1
    
    num_resources = int(lines[current_line])
    current_line += 1
    
    resource_availability = [int(x) for x in lines[current_line].split()]
    current_line += 1
    
    projects = []
    for project_id in range(1, num_projects + 1):
        # Parse project header
        project_info = lines[current_line].split()
        num_activities = int(project_info[0])
        release_date = int(project_info[1])
        current_line += 1
        
        # Parse resource usage
        resource_usage = [int(x) for x in lines[current_line].split()]
        current_line += 1
        
        # Create project with project_id
        project = Project(project_id, num_activities, release_date, resource_usage)
        
        # Set available resources
        for i, usage in enumerate(resource_usage):
            if usage == 1:
                project.available_resources[i] = resource_availability[i]
        
        # Parse activities
        for activity_id in range(1, num_activities + 1):
            parts = lines[current_line].split()
            duration = int(parts[0])
            resource_requirements = {
                i: int(amount) 
                for i, amount in enumerate(parts[1:num_resources+1])
                if int(amount) > 0
            }
            
            num_successors = int(parts[num_resources+1])
            successors = []
            if num_successors > 0:
                successor_parts = parts[num_resources+2:num_resources+2+num_successors]
                for succ in successor_parts:
                    proj_id, act_id = map(int, succ.split(':'))
                    successors.append((proj_id, act_id))
            
            activity = Activity(project_id, activity_id, duration, 
                             resource_requirements, successors)
            project.activities.append(activity)
            current_line += 1
        
        projects.append(project)
    
    # Build predecessor relationships
    activity_map = {}
    for project in projects:
        for activity in project.activities:
            activity_map[(activity.project_id, activity.activity_id)] = activity
    
    for project in projects:
        for activity in project.activities:
            for succ_proj_id, succ_act_id in activity.successors:
                successor = activity_map.get((succ_proj_id, succ_act_id))
                if successor:
                    successor.predecessors.append((activity.project_id, activity.activity_id))
    
    return projects

def build_predecessors(projects):
    """Build predecessors relationships for all activities."""
    # Create a mapping of (project_id, activity_id) to Activity object
    activity_map = {}
    for project in projects:
        for activity in project.activities:
            activity_map[(activity.project_id, activity.activity_id)] = activity
    
    # Build predecessors
    for project in projects:
        for activity in project.activities:
            for succ_proj_id, succ_act_id in activity.successors:
                successor = activity_map.get((succ_proj_id, succ_act_id))
                if successor:
                    successor.predecessors.append((activity.project_id, activity.activity_id))

def calculate_critical_path(project):
    """Calculate the critical path for a project using forward and backward pass."""
    # Initialize dictionaries
    project.earliest_start = defaultdict(int)
    project.earliest_finish = defaultdict(int)
    project.latest_start = defaultdict(int)
    project.latest_finish = defaultdict(int)
    project.slack = defaultdict(int)
    
    # Forward pass
    for activity in project.activities:
        act_id = activity.activity_id
        if not activity.predecessors:  # Start activity
            project.earliest_start[act_id] = 0
        else:
            # Find maximum earliest finish time of predecessors
            max_pred_finish = 0
            for pred_proj_id, pred_act_id in activity.predecessors:
                if pred_proj_id == project.activities[0].project_id:  # Same project
                    max_pred_finish = max(max_pred_finish, 
                                       project.earliest_finish[pred_act_id])
            project.earliest_start[act_id] = max_pred_finish
        
        project.earliest_finish[act_id] = (project.earliest_start[act_id] + 
                                         activity.duration)
    
    # Find project completion time
    project_completion = max(project.earliest_finish.values())
    project.critical_path_length = project_completion
    
    # Backward pass
    for activity in reversed(project.activities):
        act_id = activity.activity_id
        if not activity.successors:  # End activity
            project.latest_finish[act_id] = project_completion
        else:
            # Find minimum latest start time of successors
            min_succ_start = float('inf')
            for succ_proj_id, succ_act_id in activity.successors:
                if succ_proj_id == project.activities[0].project_id:  # Same project
                    min_succ_start = min(min_succ_start, 
                                       project.latest_start[succ_act_id])
            project.latest_finish[act_id] = min_succ_start if min_succ_start != float('inf') else project_completion
        
        project.latest_start[act_id] = (project.latest_finish[act_id] - 
                                      activity.duration)
        
        # Calculate slack
        project.slack[act_id] = (project.latest_start[act_id] - 
                               project.earliest_start[act_id])

def calculate_overall_critical_path(projects):
    """Calculate the critical path length for the entire multi-project problem."""
    # Create a combined earliest finish time dictionary for all projects
    all_earliest_finish = {}
    for project in projects:
        for activity in project.activities:
            key = (activity.project_id, activity.activity_id)
            all_earliest_finish[key] = project.earliest_finish[activity.activity_id]
    
    # Find the maximum finish time considering all projects
    overall_completion_time = max(all_earliest_finish.values())
    return overall_completion_time

def print_project_summary(num_projects, num_resources, resource_availability, projects):
    """Print a summary of the parsed RCMP data including critical path information."""
    print(f"Number of projects: {num_projects}")
    print(f"Number of resources: {num_resources}")
    print(f"Resource availability: {resource_availability}")
    print("\nProject details:")
    
    for idx, project in enumerate(projects, 1):
        print(f"\nProject {idx}:")
        print(f"  Number of activities: {project.num_activities}")
        print(f"  Release date: {project.release_date}")
        print(f"  Critical Path Length: {project.critical_path_length}")
        print(f"  Resource usage: {project.resource_usage}")
        print("\n  Activities:")
        
        for activity in project.activities:
            print(f"\n    Activity {activity.project_id}.{activity.activity_id}:")
            print(f"      Duration: {activity.duration}")
            print(f"      Resource requirements: {activity.resource_requirements}")
            print(f"      Successors: {[f'{p}:{a}' for p, a in activity.successors]}")
            print(f"      Earliest Start: {project.earliest_start[activity.activity_id]}")
            print(f"      Latest Start: {project.latest_start[activity.activity_id]}")
            print(f"      Slack: {project.slack[activity.activity_id]}")
            print(f"      On Critical Path: {'Yes' if project.slack[activity.activity_id] == 0 else 'No'}")

def calculate_total_critical_path(projects):
    """Calculate the maximum critical path length considering all dependencies."""
    # Initialize a dictionary to store the earliest completion time for each activity
    earliest_completion = {}
    
    # Process all activities in topological order
    def process_activity(project_id, activity_id, visited):
        key = (project_id, activity_id)
        if key in visited:
            return earliest_completion[key]
        
        visited.add(key)
        
        # Find the activity object
        activity = None
        project = None
        for p in projects:
            if p.activities[0].project_id == project_id:
                project = p
                for a in p.activities:
                    if a.activity_id == activity_id:
                        activity = a
                        break
                break
        
        if not activity:
            return 0
        
        # Calculate the earliest time this activity can start
        max_pred_time = 0
        for pred_proj_id, pred_act_id in activity.predecessors:
            pred_time = process_activity(pred_proj_id, pred_act_id, visited)
            max_pred_time = max(max_pred_time, pred_time)
        
        # Calculate and store the earliest completion time for this activity
        completion_time = max_pred_time + activity.duration
        earliest_completion[key] = completion_time
        
        return completion_time
    
    # Process all activities
    visited = set()
    max_completion_time = 0
    for project in projects:
        for activity in project.activities:
            completion_time = process_activity(activity.project_id, activity.activity_id, visited)
            max_completion_time = max(max_completion_time, completion_time)
    
    return max_completion_time


def activities_to_tasks(projects):
    """Convert project activities to tasks and establish relationships"""
    for project in projects:
        project.tasks = []
        for activity in project.activities:
            # Create task from activity
            task = Task(
                project_id=activity.project_id,
                task_id=activity.task_id,
                duration=activity.duration,
                resources=activity.resource_requirements,
            )
            project.tasks.append(task)

    # Build task relationships after all tasks are created
    activity_to_task = {}
    for project in projects:
        for task, activity in zip(project.tasks, project.activities):
            activity_to_task[(activity.project_id, activity.activity_id)] = task

    # Set up predecessors and successors
    for project in projects:
        for task, activity in zip(project.tasks, project.activities):
            # Set successors
            task.successors = []
            for succ_proj_id, succ_act_id in activity.successors:
                successor = activity_to_task.get((succ_proj_id, succ_act_id))
                if successor:
                    task.successors.append(successor)

            # Set predecessors
            task.predecessors = []
            for pred_proj_id, pred_act_id in activity.predecessors:
                predecessor = activity_to_task.get((pred_proj_id, pred_act_id))
                if predecessor:
                    task.predecessors.append(predecessor)
    
    return projects


def generate_valid_sequence(tasks):
    """Generate a valid task sequence respecting precedence constraints"""
    if not tasks:
        logger.warning("No tasks provided to generate_valid_sequence")
        return []
        
    available_tasks = set(tasks)
    sequence = []
    scheduled = set()
    
    # Debug log
    logger.debug(f"Starting sequence generation with {len(tasks)} tasks")
    
    stuck_count = 0  # Add counter to detect if we're stuck
    while available_tasks:
        # Find tasks with no unscheduled predecessors
        ready_tasks = [
            task for task in available_tasks
            if all(pred in scheduled for pred in task.predecessors)
        ]
        
        # Debug log
        logger.debug(f"Available tasks: {len(available_tasks)}, Ready tasks: {len(ready_tasks)}")
        
        if not ready_tasks:
            logger.warning("No ready tasks found, breaking sequence generation")
            break
        
        # Select a task (randomly or using some heuristic)
        selected_task = random.choice(ready_tasks)
        sequence.append(selected_task)
        scheduled.add(selected_task)
        available_tasks.remove(selected_task)
        
        # Check if we're potentially stuck
        stuck_count += 1
        if stuck_count > len(tasks) * 2:  # If we've been iterating too long
            logger.error("Possible infinite loop detected in sequence generation")
            return None
    
    # Debug completion log
    success = len(sequence) == len(tasks)
    logger.debug(f"Sequence generation {'successful' if success else 'failed'}: "
                f"{len(sequence)}/{len(tasks)} tasks scheduled")
    
    return sequence if success else None


class RCMPSPSolver:
    def __init__(self, file_path: str):
        self.projects = parse_rcmp_file(file_path)
        # Convert activities to tasks
        self.projects = activities_to_tasks(self.projects)
        self.solver = DQNCSHybrid()
        self.metrics_calculator = RCMPSPMetrics(self.projects, None)
    @profile
    def solve(self):
        """Solve the RCMPSP instance"""
        # Solve using DQN-CS hybrid
        solution = self.solver.solve(self.projects)
        
        # Calculate metrics
        self.metrics_calculator.solution = solution
        metrics = self.metrics_calculator.calculate_all_metrics()
        analysis = self.metrics_calculator.analyze_results()
        
        return {
            'solution': solution,
            'metrics': metrics,
            'analysis': analysis
        }
    
    def print_results(self, results):
        """Print comprehensive solution and analysis results"""
        print("\nSolution Summary:")
        print("-" * 50)
        print(f"Total Portfolio Makespan (TPM): {results['metrics']['TPM']:.2f}")
        print(f"Average Project Makespan (APM): {results['metrics']['APM']:.2f}")
        print(f"Critical Path Length (CP1): {results['metrics']['CP1']}")
        print(f"Resource Load Balance (RLB1): {results['metrics']['RLB1']}")
        
        print("\nDelay Analysis (CP1-based):")
        print("-" * 50)
        print(f"Average Project Delay (APD_CP1): {results['metrics']['APD_CP1']:.2f}")
        print(f"Maximum Project Delay (MaxPD_CP1): {results['metrics']['MaxPD_CP1']:.2f}")
        print(f"Squared Project Delay (SPD_CP1): {results['metrics']['SPD_CP1']:.2f}")
        print(f"Average Relative Gap (ARG_CP1): {results['metrics']['ARG_CP1']:.2f}")
        print(f"Maximum Relative Gap (MaxRG_CP1): {results['metrics']['MaxRG_CP1']:.2f}")
    
        print("\nDelay Analysis (RLB1-based):")
        print("-" * 50)
        print(f"Average Project Delay (APD_RLB1): {results['metrics']['APD_RLB1']:.2f}")
        print(f"Maximum Project Delay (MaxPD_RLB1): {results['metrics']['MaxPD_RLB1']:.2f}")
        print(f"Squared Project Delay (SPD_RLB1): {results['metrics']['SPD_RLB1']:.2f}")
        print(f"Average Relative Gap (ARG_RLB1): {results['metrics']['ARG_RLB1']:.2f}")
        print(f"Maximum Relative Gap (MaxRG_RLB1): {results['metrics']['MaxRG_RLB1']:.2f}")
        
        print("\nResource Utilization:")
        print("-" * 50)
        print(f"Average Resource Utilization: {results['metrics'].get('AvgResourceUtilization', 0):.2%}")
        print(f"Peak Resource Utilization: {results['metrics'].get('PeakResourceUtilization', 0):.2%}")
        print(f"Project Overlap Ratio: {results['metrics'].get('ProjectOverlapRatio', 0):.2%}")
        
        print("\nInsights:")
        print("-" * 50)
        for insight in results['analysis']['insights']:
            print(f"- {insight}")
        
        if results['analysis']['recommendations']:
            print("\nRecommendations:")
            print("-" * 50)
            for rec in results['analysis']['recommendations']:
                print(f"- {rec}")

import os
import pandas as pd
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
import traceback

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rcmpsp_solver_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_instance(file_path):
    """Process a single RCMP instance and return its results"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Processing instance: {file_path}")
        
        solver = RCMPSPSolver(file_path)
        results = solver.solve()
        
        # Extract relevant metrics
        metrics = results['metrics']
        instance_name = os.path.basename(file_path)
        
        result_dict = {
            'Instance': instance_name,
            'TPM': metrics['TPM'],
            'APM': metrics['APM'],
            'CP1': metrics['CP1'],
            'RLB1': metrics['RLB1'],
            'APD_CP1': metrics['APD_CP1'],
            'MaxPD_CP1': metrics['MaxPD_CP1'],
            'SPD_CP1': metrics['SPD_CP1'],
            'ARG_CP1': metrics['ARG_CP1'],
            'MaxRG_CP1': metrics['MaxRG_CP1'],
            'APD_RLB1': metrics['APD_RLB1'],
            'MaxPD_RLB1': metrics['MaxPD_RLB1'],
            'SPD_RLB1': metrics['SPD_RLB1'],
            'ARG_RLB1': metrics['ARG_RLB1'],
            'MaxRG_RLB1': metrics['MaxRG_RLB1']
        }
        
        # Add any insights
        insights = '; '.join(results['analysis']['insights'])
        result_dict['Insights'] = insights
        
        logger.info(f"Successfully processed {instance_name}")
        return result_dict
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'Instance': os.path.basename(file_path),
            'Error': str(e)
        }

def batch_solve_instances(instances_dir, output_dir="results"):
    """Process all RCMP instances in the directory and save results"""
    # Setup logging
    logger = setup_logging()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all RCMP files
    rcmp_files = [
        os.path.join(instances_dir, f) 
        for f in os.listdir(instances_dir) 
        if f.endswith('.rcmp')
    ]
    
    if not rcmp_files:
        logger.error(f"No RCMP files found in {instances_dir}")
        return
    
    logger.info(f"Found {len(rcmp_files)} RCMP instances")
    
    # Process all instances
    results = []
    for file_path in rcmp_files:
        try:
            result = process_instance(file_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_dir, f"rcmpsp_results_{timestamp}.xlsx")
    
    # Create Excel writer
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Write main results
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Results']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D9E1F2',
            'border': 1
        })
        
        # Format headers
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Auto-adjust columns width
        for i, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(str(col))
            )
            worksheet.set_column(i, i, max_length + 2)
    
    logger.info(f"Results saved to {excel_path}")
    return excel_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch RCMPSP Solver')
    parser.add_argument('instances_dir', help='Directory containing RCMP files')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    args = parser.parse_args()
    
    batch_solve_instances(args.instances_dir, args.output_dir)

if __name__ == "__main__":
    main()