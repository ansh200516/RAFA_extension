import re
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import colorsys
from PIL import Image, ImageDraw, ImageFont
import random
import os
from termcolor import colored
import builtins
import numpy as np
import matplotlib.pyplot as plt

def get_stats_from_logs(log_data: Dict[str, dict]):
    stats_per_method = {}
    for method, data in log_data.items():
        summary = data.get('summary')
        if not summary:
            continue

        accuracy = summary.get('accuracy')
        cost = summary.get('cost', {}).get('cost')

        if accuracy is not None and cost is not None:
            if method not in stats_per_method:
                stats_per_method[method] = {}
            
            stats_per_method[method]['game24'] = {
                'accuracy': accuracy,
                'accuracy_std': 0,
                'cost': cost,
                'cost_std': 0,
            }
    return stats_per_method

def load_game24_logs(log_dir='log/lmmmpc/game24'):
    log_data = {}
    if not os.path.isdir(log_dir):
        print(colored(f"Log directory not found: {log_dir}", "red"))
        return {}

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith('.json'):
            continue

        match = re.search(r'_(summary_\d+|prevk_\d+|tot)_', filename)
        if match:
            method_name = match.group(1)
            if method_name == 'tot':
                method_name = 'base'
        else:
            method_name = filename.replace('.json', '')

        try:
            with open(os.path.join(log_dir, filename), 'r') as f:
                data = json.load(f)
            log_data[method_name] = data
            print(colored(f"Loaded log for method: {method_name}", "cyan"))
        except (json.JSONDecodeError, IOError) as e:
            print(colored(f"Error loading {filename}: {e}", "red"))
    return log_data

def draw_bar_graph(stats_per_method: Dict[str, dict], task_name: str):
    labels = list(stats_per_method.keys())
    
    all_stats_have_data = all(task_name in stats for stats in stats_per_method.values())
    if not all_stats_have_data:
        print(colored(f"Skipping graph for task '{task_name}' because some methods lack data.", "yellow"))
        return

    accuracies = [stats_per_method[method][task_name]['accuracy'] for method in labels]
    costs = [stats_per_method[method][task_name]['cost'] for method in labels]
    acc_errors = [stats_per_method[method][task_name]['accuracy_std'] for method in labels]
    cost_errors = [stats_per_method[method][task_name]['cost_std'] for method in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_ylabel('Accuracy', color='blue')
    ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='blue', yerr=acc_errors, capsize=5)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost', color='red')
    
    max_cost = max(costs) if costs else 1
    if max_cost == 0:
        max_cost = 1
    
    normalized_costs = [c / max_cost for c in costs]
    normalized_cost_errors = [err / max_cost for err in cost_errors]

    ax2.bar(x + width/2, normalized_costs, width, label='Cost (Normalized)', color='red', yerr=normalized_cost_errors, capsize=5)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, max(normalized_costs) * 1.2 if normalized_costs else 1)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Method')
    fig.suptitle(f'Environment: {task_name}')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs('tmp/graphs', exist_ok=True)
    save_path = f'tmp/graphs/{task_name}.png'
    plt.savefig(save_path)
    plt.close(fig)
    print(colored(f"Graph for task '{task_name}' saved to {save_path}", "green"))

def draw_cvq_graph(stats_per_method: Dict[str, dict]):
    all_tasks = set()
    for stats in stats_per_method.values():
        all_tasks.update(stats.keys())
    all_tasks = sorted(list(all_tasks))
    
    methods = list(stats_per_method.keys())
    
    ratios = {}
    has_inf = False
    max_finite_ratio = 0

    for method in methods:
        ratios[method] = []
        for task in all_tasks:
            if task in stats_per_method.get(method, {}):
                stats = stats_per_method[method][task]
                accuracy = stats['accuracy']
                cost = stats['cost']
                
                if cost > 1e-9:
                    ratio = accuracy / cost
                    if ratio > max_finite_ratio:
                        max_finite_ratio = ratio
                else:
                    if accuracy > 1e-9:
                        ratio = float('inf')
                        has_inf = True
                    else:
                        ratio = 0.0
                ratios[method].append(ratio)
            else:
                ratios[method].append(0.0)

    if has_inf:
        if max_finite_ratio == 0:
             max_finite_ratio = 1
        for method in methods:
            for i in range(len(ratios[method])):
                if ratios[method][i] == float('inf'):
                    ratios[method][i] = max_finite_ratio * 1.2

    x = np.arange(len(all_tasks))
    width = 0.8 / len(methods) if methods else 0.8
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, method in enumerate(methods):
        offset = width * (i - (len(methods) - 1) / 2)
        ax.bar(x + offset, ratios[method], width, label=method)

    ax.set_ylabel('Accuracy / Cost Ratio')
    ax.set_title('Task-wise Efficiency (Accuracy/Cost)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()

    os.makedirs('tmp/graphs', exist_ok=True)
    save_path = f'tmp/graphs/cvq_comparison.png'
    plt.savefig(save_path)
    plt.close(fig)
    print(colored(f"Cost vs Quality graph saved to {save_path}", "green"))

# ------------------------------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------------------------------

log_data = load_game24_logs()

if not log_data:
    print(colored("No valid log files found. Exiting.", "red"))
    exit()

current_method = None
current_puzzle_idx = None

while True:
    prompt_str = '>>> '
    if current_method:
        prompt_str = f'({current_method})'
        if current_puzzle_idx is not None:
            prompt_str += f'/puzzle-{current_puzzle_idx}'
    prompt_str += ' >>> '

    try:
        cmd = input(prompt_str)
    except EOFError:
        break

    if cmd == 'q':
        break

    if cmd == 'cd ..':
        if current_puzzle_idx is not None:
            current_puzzle_idx = None
        elif current_method is not None:
            current_method = None
        continue

    if cmd == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        continue

    if cmd.startswith('cd '):
        target = cmd.split(' ', 1)[1]
        if not current_method:
            if target in log_data:
                current_method = target
                print(colored(f'Switched to method {current_method}.', 'green'))
            else:
                print(colored(f'Method {target} not found.', 'red'))
            continue

        try:
            puzzle_idx = int(target)
            results = log_data[current_method].get('results', [])
            if 0 <= puzzle_idx < len(results):
                current_puzzle_idx = puzzle_idx
                print(colored(f'Selected puzzle {puzzle_idx}.', 'green'))
            else:
                print(colored(f'Puzzle index {puzzle_idx} is out of range.', 'red'))
        except ValueError:
            print(colored('Invalid puzzle index. Must be an integer.', 'red'))
        continue

    if cmd == 'cvq':
        stats_per_method = get_stats_from_logs(log_data)
        
        if not stats_per_method:
            print(colored("No stats found in logs to generate graph.", "red"))
            continue

        draw_cvq_graph(stats_per_method)
        continue

    if cmd.startswith('img'):
        parts = cmd.split()
        methods_to_plot = parts[1:]

        data_to_plot = log_data
        if methods_to_plot:
            valid_methods = [m for m in methods_to_plot if m in log_data]
            invalid_methods = [m for m in methods_to_plot if m not in log_data]
            
            if invalid_methods:
                print(colored(f"Methods not found: {', '.join(invalid_methods)}", "yellow"))

            if not valid_methods:
                print(colored("None of the specified methods were found. Aborting graph generation.", "red"))
                continue
            
            data_to_plot = {m: log_data[m] for m in valid_methods}

        stats_per_method = get_stats_from_logs(data_to_plot)
        
        if not stats_per_method:
            print(colored("No stats found for the selected methods to generate graphs.", "red"))
            continue

        all_tasks = set()
        for stats in stats_per_method.values():
            all_tasks.update(stats.keys())
        
        for task_name in sorted(list(all_tasks)):
            draw_bar_graph(stats_per_method, task_name)
        
        continue

    if cmd == 'ls':
        if not current_method:
            print(colored("Available methods:", "cyan"))
            for method_name in log_data:
                print(f'- {method_name}')
            continue

        results = log_data[current_method].get('results', [])
        if not results:
            print(colored(f"No puzzle results found for method '{current_method}'.", "yellow"))
            continue
        
        for i, result in enumerate(results):
            status = colored('Correct', 'green') if result.get('correct') else colored('Incorrect', 'red')
            print(f"Puzzle {i}: {status}")
        continue

    if cmd.startswith('show '):
        if not current_method:
            print(colored("No method selected. Use 'cd <method_name>' first.", "red"))
            continue
        try:
            idx_str = cmd.split(' ', 1)[1]
            puzzle_idx_to_show = int(idx_str)
            
            results = log_data[current_method].get('results', [])
            if 0 <= puzzle_idx_to_show < len(results):
                puzzle_data = results[puzzle_idx_to_show]
                print(colored(f"\n--- Puzzle {puzzle_idx_to_show} Details ---", "cyan"))
                print(f"Puzzle: {puzzle_data.get('puzzle')}")
                print(f"Answer: {puzzle_data.get('answer')}")
                status = colored('Correct', 'green') if puzzle_data.get('correct') else colored('Incorrect', 'red')
                print(f"Status: {status}")
                
                thoughts = puzzle_data.get("thoughts", "")
                if isinstance(thoughts, list) and thoughts:
                    thought = thoughts[0] # Taking the first thought block
                    if isinstance(thought, list) and thought:
                         # It's a list of steps, format them
                        print(colored("\nThoughts:", "cyan"))
                        for step in thought:
                            if isinstance(step, dict):
                                for k, v in step.items():
                                    print(f"  {k}: {v}")
                            else:
                                print(f"  - {step}")
                    else:
                        print(f"\nThoughts:\n{thoughts}")
                else:
                    print(f"\nThoughts:\n{thoughts}")

            else:
                print(colored(f"Puzzle index {puzzle_idx_to_show} is out of range.", "red"))
        except (ValueError, IndexError):
            print(colored("Invalid command. Use 'show <puzzle_index>'.", "red"))
        continue

    print(colored('Unknown command. Type "help" for a list of commands.', 'yellow'))


