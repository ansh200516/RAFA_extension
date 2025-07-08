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

logs = ''
state_names = {}
states_done_in_puzzle = {}
state_colors = {}

class State(BaseModel):
    name: str
    color: str
    num_thoughts: int
    serial_data: dict
    value: Optional[float] = None
    terminal_data: str = ''

class Timestep(BaseModel):
    timestep: int
    input_states: list[State]
    agent_output_states: list[State]
    state_wins: list[bool]
    state_fails: list[bool]
    replacement_states: list[State]
    values: Optional[list[float]] = None
    
def generate_distinct_hex_colors(n):
    """
    Generate `n` distinct hex colors that are as different as possible and not close to black.
    
    Returns:
        List of hex color strings (e.g., '#FF5733').
    """
    colors = []
    for i in range(n):
        # Evenly space hues around the color wheel
        hue = i / n
        saturation = 0.65  # Keep saturation high to avoid washed-out colors
        value = 0.8        # Avoid dark (black-ish) colors by setting high brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def load_all_logs() -> Dict[str, str]:
    """
    Loads all logs from both log files for interactive analysis.
    """
    log_contents = {}
    log_files = {
        'reflect_summary': 'logs/reflect_summary.log',
        'reflect_prevk': 'logs/reflect_prevk.log'
    }
    
    for name, file_path in log_files.items():
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if content:
                    print(colored(f"Loading {file_path} for interactive analysis.", "cyan"))
                    log_contents[name] = content
        except FileNotFoundError:
            # This is not an error, one of the files might not exist yet
            pass
            
    return log_contents

def get_puzzle_idx(log):
    res = re.search(r"reflect_(?:summary|prevk)_logs-(?:[a-zA-Z0-9_]+)-(\d+)-", log)
    assert res is not None, f'Puzzle index not found in log: {log}'
    return int(res.group(1))

def get_timestep(log):
    res = re.search(r"reflect_(?:summary|prevk)_logs-(?:[a-zA-Z0-9_]+)-(\d+)-(\d+)", log)
    assert res is not None, f'Timestep not found in log: {log}'
    return int(res.group(2))

def get_task_name_from_log(log: str) -> Optional[str]:
    res = re.search(r"reflect_(?:summary|prevk)_logs-([a-zA-Z0-9_]+)-", log)
    if res:
        return res.group(1)
    return None

def get_py_list(string, type):
    l = eval(string)
    assert isinstance(l, list), f'Expected a list, got {type(l)}: {l}'

    for i, item in enumerate(l):
        l[i] = type(item)

    assert all(isinstance(item, type) for item in l), f'Expected all items to be {type.__name__}, got {l}'
    return l

def get_fleet(log):
    log = log.replace('ValueFunctionWrapped', '').replace('EnvWrapped', '')
    isolated_list = log.split('fleet: ')[-1].strip()
    return get_py_list(isolated_list, str)

def state_name(current_state: str, index):
    if hash(current_state) in state_names:
        return state_names[hash(current_state)]
    
    if index not in states_done_in_puzzle:
        states_done_in_puzzle[index] = 0
    states_done_in_puzzle[index] += 1
    
    idx = states_done_in_puzzle[index]
    state_names[hash(current_state)] = f's{idx}'
    return state_names[hash(current_state)]

def get_state_color(state_name: str):
    if state_name in state_colors:
        return state_colors[state_name]
    
    idx = len(state_colors)
    state_colors[state_name] = f'color{idx}'
    return state_colors[state_name]

def get_states_from_log(log):
    index = get_puzzle_idx(log)
    isolated_list = log[log.find('['):]
    states_str = get_py_list(isolated_list, str)
    
    parsed_states = []
    for s_str in states_str:
        try:
            parsed_states.append(json.loads(s_str))
        except json.JSONDecodeError:
            raise ValueError(f'Invalid JSON in state: {s_str}')

    states_out = []
    for state_data in parsed_states:
        s_name = state_name(state_data['current_state'], index)
        states_out.append(State(
            name=s_name,
            color=get_state_color(s_name),
            num_thoughts=len(state_data['reflections']),
            value=state_data.get('value'),
            serial_data=state_data
        ))

    return states_out

def get_timestep_object(logs, timestep=0):
    log_map = {}
    for log in logs:
        match = re.search(r'-(agentinputs|agentouts|statewins|statefails|reflections|summaries):', log)
        if match:
            log_map[match.group(1)] = log

    input_states = get_states_from_log(log_map['agentinputs'])
    output_states = get_states_from_log(log_map['agentouts'])
    state_wins = get_py_list(log_map['statewins'].split('statewins: ')[-1].strip(), bool)
    state_fails = get_py_list(log_map['statefails'].split('statefails: ')[-1].strip(), bool)

    if 'reflections' in log_map:
        reflections_list = get_py_list(log_map['reflections'].split('reflections: ')[-1].strip(), int)
        for i, num_reflections in enumerate(reflections_list):
            if i < len(output_states):
                output_states[i].num_thoughts = num_reflections
    
    if 'summaries' in log_map:
        summaries_list = get_py_list(log_map['summaries'].split('summaries: ')[-1].strip(), list)
        for i, summary in enumerate(summaries_list):
            if i < len(output_states):
                output_states[i].num_thoughts = len(summary)

    return Timestep(
        timestep=timestep,
        input_states=input_states,
        agent_output_states=output_states,
        state_wins=state_wins,
        state_fails=state_fails,
        replacement_states=[],
        values=None,
    )

def get_stats_from_file(log_path):
    try:
        with open(log_path, 'r') as f:
            logs_content = f.read()
        if not logs_content:
            return {}
    except FileNotFoundError:
        return {}

    processed_data = process_log_bundle(logs_content)
    task_stats = {}

    for task_name, data in processed_data['tasks'].items():
        graph = data['graph']
        puzzle_outcomes = []
        puzzle_costs = []
        
        if not graph:
            continue

        for puzzle_idx, timesteps in graph.items():
            if timesteps:
                puzzle_outcomes.append(1 if any(timesteps[-1].state_wins) else 0)
            else:
                puzzle_outcomes.append(0)

            puzzle_cost = 0
            for timestep in timesteps:
                for state in timestep.input_states + timestep.agent_output_states:
                    puzzle_cost += state.num_thoughts
            puzzle_costs.append(puzzle_cost)
        
        if not puzzle_outcomes:
            continue

        accuracy = np.mean(puzzle_outcomes)
        accuracy_std = np.std(puzzle_outcomes) / np.sqrt(len(puzzle_outcomes)) if len(puzzle_outcomes) > 0 else 0

        cost = np.mean(puzzle_costs)
        cost_std = np.std(puzzle_costs)

        task_stats[task_name] = {
            'accuracy': accuracy,
            'accuracy_std': accuracy_std,
            'cost': cost,
            'cost_std': cost_std,
        }
        
    return task_stats

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

    # Plot accuracy
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='blue', yerr=acc_errors, capsize=5)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))


    # Plot cost
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost', color='red')
    
    max_cost = max(costs) if costs else 1
    if max_cost == 0:
        max_cost = 1
    
    normalized_costs = [c / max_cost for c in costs]
    normalized_cost_errors = [err / max_cost for err in cost_errors]

    ax2.bar(x + width/2, normalized_costs, width, label='Cost', color='red', yerr=normalized_cost_errors, capsize=5)
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

def get_stats_from_correctness_logs(log_dir='logs/correctness'):
    stats = {}
    model_log_dir = os.path.join(log_dir, 'gpt-4.1-nano')

    if not os.path.isdir(model_log_dir):
        print(colored(f"Log directory not found: {model_log_dir}", "red"))
        return {}

    for task_name in os.listdir(model_log_dir):
        task_dir = os.path.join(model_log_dir, task_name)
        if os.path.isdir(task_dir):
            for file in os.listdir(task_dir):
                if file.endswith('.log'):
                    log_path = os.path.join(task_dir, file)
                    method = file.replace('.log', '')

                    with open(log_path, 'r') as f:
                        content = f.read()

                    costs = re.findall(r"Costs: \{'total': (\d+\.?\d*e?-?\d*)", content)
                    detailed_correct_str = re.findall(r"Correct \(deailed\): (\[.*?\])", content)
                    
                    if not costs:
                        continue

                    cost = float(costs[-1])
                    acc = -1
                    acc_std = 0
                    
                    if detailed_correct_str:
                        try:
                            detailed_correct = eval(detailed_correct_str[-1])
                            if isinstance(detailed_correct, list) and len(detailed_correct) > 0:
                                acc = np.mean(detailed_correct)
                                acc_std = np.std(detailed_correct) / np.sqrt(len(detailed_correct))
                            else:
                                accuracies = re.findall(r"Correct: (\d+\.?\d*)", content)
                                if accuracies:
                                    acc = float(accuracies[-1])
                        except (SyntaxError, NameError):
                            accuracies = re.findall(r"Correct: (\d+\.?\d*)", content)
                            if accuracies:
                                acc = float(accuracies[-1])
                    else:
                        accuracies = re.findall(r"Correct: (\d+\.?\d*)", content)
                        if accuracies:
                            acc = float(accuracies[-1])

                    if acc == -1:
                        continue

                    if method not in stats:
                        stats[method] = {}
                    
                    stats[method][task_name] = {
                        'accuracy': acc,
                        'accuracy_std': acc_std,
                        'cost': cost,
                        'cost_std': 0,
                    }
    return stats

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

def process_log_bundle(logs_str: str):
    global state_names, states_done_in_puzzle, state_colors
    state_names = {}
    states_done_in_puzzle = {}
    state_colors = {}

    logs = logs_str.split('\n')
    
    tasks_log_data = {}

    log_prefix = ''
    if 'reflect_summary_logs' in logs_str:
        log_prefix = 'reflect_summary_logs'
    elif 'reflect_prevk_logs' in logs_str:
        log_prefix = 'reflect_prevk_logs'

    for log in logs:
        if log_prefix in log:
            task_name = get_task_name_from_log(log)
            if task_name:
                if task_name not in tasks_log_data:
                    tasks_log_data[task_name] = {'fleet': [], 'puzzle_logs': []}
                
                if '-fleet:' in log:
                    if len(tasks_log_data[task_name]['fleet']) == 0:
                        tasks_log_data[task_name]['fleet'] = get_fleet(log)
                else:
                    tasks_log_data[task_name]['puzzle_logs'].append(log)

    tasks_processed_data = {}

    for task_name, task_data in tasks_log_data.items():
        state_names = {}
        states_done_in_puzzle = {}
        state_colors = {}

        puzzles_dict = {}
        log_order = ['agentinputs', 'agentouts', 'statewins', 'statefails', 'reflections', 'summaries']

        def get_log_type(log_line):
            match = re.search(r'-([a-zA-Z]+):', log_line)
            return match.group(1) if match else ""

        for log in task_data['puzzle_logs']:
            try:
                puzzle_idx = get_puzzle_idx(log)
                timestep_idx = get_timestep(log)

                if puzzle_idx not in puzzles_dict:
                    puzzles_dict[puzzle_idx] = {}
                if timestep_idx not in puzzles_dict[puzzle_idx]:
                    puzzles_dict[puzzle_idx][timestep_idx] = []
                
                puzzles_dict[puzzle_idx][timestep_idx].append(log)
            except (AssertionError, IndexError):
                pass

        graph: Dict[int, List[Timestep]] = {}
        flows = {}
        for puzzle_idx, timesteps_dict in puzzles_dict.items():
            graph[puzzle_idx] = []
            
            sorted_timesteps = sorted(timesteps_dict.items())

            for timestep_idx, logs_for_timestep in sorted_timesteps:
                sorted_logs_for_timestep = sorted(logs_for_timestep, key=lambda l: log_order.index(get_log_type(l)) if get_log_type(l) in log_order else -1)
                
                if len(sorted_logs_for_timestep) < 4:
                    continue

                timestep = get_timestep_object(sorted_logs_for_timestep, timestep_idx)
                graph[puzzle_idx].append(timestep)

            num_colors = len(state_colors)
            colors = generate_distinct_hex_colors(num_colors)
            random.shuffle(colors)

            for k in state_colors:
                state_colors[k] = colors.pop(0)

            for timestep in graph[puzzle_idx]:
                for state in timestep.input_states + timestep.agent_output_states:
                    state.color = get_state_color(state.name)

            for timestep in graph[puzzle_idx]:
                for i in range(len(timestep.agent_output_states)):
                    if i < len(timestep.state_wins) and timestep.state_wins[i]:
                        timestep.agent_output_states[i].terminal_data = 'Winning'
                    elif i < len(timestep.state_fails) and timestep.state_fails[i]:
                        timestep.agent_output_states[i].terminal_data = 'Failed'

            fleet = task_data['fleet']
            if len(fleet) > 0:
                flows[puzzle_idx] = [{
                    'agent_name': fleet[0],
                    'input_states': [t.input_states[i] for t in graph[puzzle_idx] if len(t.input_states) > i],
                    'output_states': [t.agent_output_states[i] for t in graph[puzzle_idx] if len(t.agent_output_states) > i],
                } for i in range(1)]
        
        tasks_processed_data[task_name] = {
            'graph': graph,
            'flows': flows,
            'state_names': state_names,
        }
            
    return {
        'tasks': tasks_processed_data
    }

def get_puzzle_statuses_from_file(log_path):
    try:
        with open(log_path, 'r') as f:
            logs_content = f.read()
        if not logs_content:
            return {}
    except FileNotFoundError:
        return {}
    
    processed_data = process_log_bundle(logs_content)
    task_statuses = {}
    
    for task_name, data in processed_data['tasks'].items():
        graph = data['graph']
        statuses = {}
        for puzzle_idx, timesteps in graph.items():
            if timesteps:
                statuses[puzzle_idx] = 'Won' if any(timesteps[-1].state_wins) else 'Failed'
            else:
                statuses[puzzle_idx] = 'Failed'
        task_statuses[task_name] = statuses

    return task_statuses

def draw_agent_diagram(agent_name: str, input_states: List[State], output_states: List[State], 
                      x_offset: int = 0, font_size: int = 14) -> tuple[Image.Image, int]:
    padding = 20
    state_width = 200
    state_padding = 10
    arrow_height = 30
    spacing_between_pairs = 40
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        bold_font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        font = ImageFont.load_default()
        bold_font = font
    
    max_pairs = max(len(input_states), len(output_states))
    
    agent_name_height = 40
    state_height = 100
    total_height = (padding * 2 + 
                   agent_name_height + 
                   max_pairs * (state_height * 2 + arrow_height + spacing_between_pairs))
    
    diagram_width = state_width + padding * 2
    
    img = Image.new('RGB', (diagram_width, total_height), 'white')
    draw = ImageDraw.Draw(img)
    
    current_y = padding
    
    agent_rect = (x_offset + padding, current_y, 
                  x_offset + padding + state_width, current_y + agent_name_height)
    draw.rectangle(agent_rect, fill='black')
    
    agent_text_bbox = draw.textbbox((0, 0), agent_name, font=bold_font)
    agent_text_width = agent_text_bbox[2] - agent_text_bbox[0]
    agent_text_height = agent_text_bbox[3] - agent_text_bbox[1]
    agent_text_x = x_offset + padding + (state_width - agent_text_width) // 2
    agent_text_y = current_y + (agent_name_height - agent_text_height) // 2
    draw.text((agent_text_x, agent_text_y), agent_name, fill='white', font=bold_font)
    
    current_y += agent_name_height + padding
    
    for i in range(max_pairs):
        if i < len(input_states):
            current_y = draw_state(draw, input_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        arrow_start_x = x_offset + padding + state_width // 2
        arrow_start_y = current_y + 5
        arrow_end_y = current_y + arrow_height - 5
        
        draw.line([(arrow_start_x, arrow_start_y), (arrow_start_x, arrow_end_y)], 
                 fill='black', width=2)
        
        arrow_head_size = 5
        draw.polygon([(arrow_start_x, arrow_end_y),
                     (arrow_start_x - arrow_head_size, arrow_end_y - arrow_head_size),
                     (arrow_start_x + arrow_head_size, arrow_end_y - arrow_head_size)],
                    fill='black')
        
        current_y += arrow_height
        
        if i < len(output_states):
            current_y = draw_state(draw, output_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        current_y += spacing_between_pairs
    
    return img.crop((0,0,diagram_width, current_y)), diagram_width

def draw_state(draw: ImageDraw.Draw, state: State, x: int, y: int, width: int, 
               font: ImageFont.ImageFont, bold_font: ImageFont.ImageFont, padding: int) -> int:
    lines = [state.name]
    
    if state.value is not None:
        lines.append(f"Value: {state.value}")
    
    if state.num_thoughts > 0:
        lines.append(f"Thoughts: {state.num_thoughts}")

    if len(state.terminal_data) > 0:
        lines.append(f"{state.terminal_data} State")
    
    line_height = 20
    text_height = 4 * line_height
    total_height = text_height + padding * 2
    
    state_rect = (x, y, x + width, y + total_height)
    draw.rectangle(state_rect, fill=state.color, outline='black', width=1)
    
    text_y = y + padding
    for i, line in enumerate(lines):
        current_font = bold_font if i == 0 else font
        draw.text((x + padding, text_y), line, fill='black', font=current_font)
        text_y += line_height
    
    return y + total_height

def create_agent_diagrams(diagrams_data: List[dict], spacing: int = 50) -> Image.Image:
    if not diagrams_data:
        return Image.new('RGB', (100, 100), 'white')
    
    diagram_images = []
    diagram_widths = []
    max_height = 0
    
    for data in diagrams_data:
        img, width = draw_agent_diagram(
            data['agent_name'], 
            data['input_states'], 
            data['output_states']
        )
        diagram_images.append(img)
        diagram_widths.append(width)
        max_height = max(max_height, img.height)
    
    total_width = sum(diagram_widths) + spacing * (len(diagrams_data) - 1)
    
    final_image = Image.new('RGB', (total_width, max_height), 'white')
    
    current_x = 0
    for i, img in enumerate(diagram_images):
        final_image.paste(img, (current_x, 0))
        current_x += diagram_widths[i] + spacing
    
    return final_image

# ------------------------------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------------------------------

log_contents = load_all_logs()
log_data = {}

if 'reflect_summary' in log_contents and log_contents['reflect_summary']:
    log_data['reflect_summary'] = process_log_bundle(log_contents['reflect_summary'])
    print(colored("Processed 'reflect_summary' logs.", "green"))
if 'reflect_prevk' in log_contents and log_contents['reflect_prevk']:
    log_data['reflect_prevk'] = process_log_bundle(log_contents['reflect_prevk'])
    print(colored("Processed 'reflect_prevk' logs.", "green"))

if 'reflect_summary' in log_data:
    current_context_name = 'reflect_summary'
elif 'reflect_prevk' in log_data:
    current_context_name = 'reflect_prevk'
else:
    current_context_name = None

current_task = None
current_puzzle = None
while True:
    prompt_str = '>>> '
    if current_context_name:
        prompt_str = f'({current_context_name})'
        if current_task:
            prompt_str += f'/{current_task}'
    prompt_str += ' >>> '

    try:
        cmd = input(prompt_str)
    except EOFError:
        break

    if cmd == 'q':
        break

    if cmd == 'cd ..':
        if current_puzzle is not None:
            current_puzzle = None
        elif current_task is not None:
            current_task = None
        continue

    if cmd == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        continue

    if cmd == 'switch':
        if current_context_name == 'reflect_summary' and 'reflect_prevk' in log_data:
            current_context_name = 'reflect_prevk'
            current_puzzle = None
            current_task = None
            print(colored("Switched to 'reflect_prevk' context.", "cyan"))
        elif current_context_name == 'reflect_prevk' and 'reflect_summary' in log_data:
            current_context_name = 'reflect_summary'
            current_puzzle = None
            current_task = None
            print(colored("Switched to 'reflect_summary' context.", "cyan"))
        else:
            print(colored("Cannot switch context. Only one log file loaded.", "yellow"))
        continue

    if not current_context_name:
        print(colored("No logs loaded.", "red"))
        if cmd == 'q':
            break
        continue
    
    tasks = log_data[current_context_name]['tasks']

    if cmd == 'compare':
        statuses_summary = get_puzzle_statuses_from_file('logs/reflect_summary.log')
        statuses_prevk = get_puzzle_statuses_from_file('logs/reflect_prevk.log')
        
        all_task_names = sorted(list(set(statuses_summary.keys()) | set(statuses_prevk.keys())))

        for task_name in all_task_names:
            print(colored(f"\nTask: {task_name}", "cyan", attrs=['bold']))
            
            task_summary = statuses_summary.get(task_name, {})
            task_prevk = statuses_prevk.get(task_name, {})
            all_puzzle_ids = sorted(list(set(task_summary.keys()) & set(task_prevk.keys())))
            
            print(f"{'Puzzle':<10}{'reflect_summary':<20}{'reflect_prevk':<20}")
            print(f"{'-'*8:<10}{'-'*15:<20}{'-'*15:<20}")

            for puzzle_idx in all_puzzle_ids:
                status_summary = task_summary.get(puzzle_idx, 'Not found')
                status_prevk = task_prevk.get(puzzle_idx, 'Not found')
                
                status_summary_colored = colored(status_summary, 'green') if status_summary == 'Won' else colored(status_summary, 'red')
                if status_summary == 'Not found':
                    status_summary_colored = colored(status_summary, 'yellow')

                status_prevk_colored = colored(status_prevk, 'green') if status_prevk == 'Won' else colored(status_prevk, 'red')
                if status_prevk == 'Not found':
                    status_prevk_colored = colored(status_prevk, 'yellow')

                print(f"{puzzle_idx:<10}{status_summary_colored:<28}{status_prevk_colored:<28}")
        continue

    if cmd.startswith('cd '):
        target = cmd.split(' ')[1]
        if not current_task:
            if target in tasks:
                current_task = target
                print(colored(f'Opened task {current_task}.', 'green'))
            else:
                print(colored(f'Task {target} not found.', 'red'))
            continue

        try:
            puzzle_idx = int(target)
            if puzzle_idx not in tasks[current_task]['flows']:
                print(colored(f'Puzzle {puzzle_idx} not found in task {current_task}.', 'red'))
                continue
            
            current_puzzle = puzzle_idx
            print(colored(f'Opened puzzle {puzzle_idx}.', 'green'))
        except (ValueError, IndexError):
            print(colored('Invalid command. Use "cd <puzzle_idx>"', 'red'))
        continue

    if cmd == 'cvq':
        stats_per_method = get_stats_from_correctness_logs()
        
        if not stats_per_method:
            print(colored("No log data found to generate graphs.", "red"))
            continue

        draw_cvq_graph(stats_per_method)
        continue

    if cmd == 'img':
        stats_per_method = get_stats_from_correctness_logs()
        
        if not stats_per_method:
            print(colored("No log data found to generate graphs.", "red"))
            continue

        all_tasks = set()
        for stats in stats_per_method.values():
            all_tasks.update(stats.keys())
        
        for task_name in sorted(list(all_tasks)):
            draw_bar_graph(stats_per_method, task_name)
        
        continue

    if cmd.startswith('img'):
        print(colored("Did you mean 'img'?", 'yellow'))
        continue

    if cmd == 'ls':
        if not current_task:
            print(colored("Available tasks:", "cyan"))
            for task_name in tasks:
                print(f'- {task_name}')
            continue

        graph = tasks[current_task]['graph']
        for puzzle_idx in tasks[current_task]['flows']:
            print(f'Puzzle {puzzle_idx}: ', colored('Won', 'green') if any(graph[puzzle_idx][-1].state_wins) else colored('Failed', 'red'))
        continue

    res = re.search(r'^s(\d+.*$)', cmd)
    if res:
        if current_puzzle is None:
            print(colored('No puzzle selected. Use "cd <puzzle_idx>" to select a puzzle.', 'red'))
            continue
        if not current_task:
            print(colored('No task selected. Use "cd <task_name>" to select a task.', 'red'))
            continue
        
        cmd_parts = res.group(1).split('.')
        state_id = "s" + cmd_parts[0]

        state_names_map = tasks[current_task]['state_names']
        if state_id not in state_names_map.values():
            print(colored(f'State {state_id} not found.', 'red'))
            continue

        graph = tasks[current_task]['graph']
        found_state = None
        for timestep in reversed(graph[current_puzzle]):
            for s in timestep.agent_output_states + timestep.input_states:
                if s.name == state_id:
                    found_state = s
                    break
            if found_state:
                break
        
        if not found_state:
             print(colored(f'State {state_id} not found in puzzle {current_puzzle}.', 'red'))
             continue

        if len(cmd_parts) > 1:
            attr = cmd_parts[1].strip()
            attr = attr.replace('cs', 'current_state') 
            attr = attr.replace('sd', 'serial_data')

            try:
                if hasattr(found_state, attr):
                    print(getattr(found_state, attr))
                elif attr in found_state.serial_data:
                    print(found_state.serial_data[attr])
                else:
                    print(colored(f'Attribute {attr} not found in state {state_id}.', 'red'))
            except Exception as e:
                print(colored(f'Error accessing attribute {attr}: {e}', 'red'))
        else:
            print(found_state)

        continue

    print(colored('Unknown command. Type "help" for a list of commands.', 'yellow'))


