from . import *
import itertools
import numpy as np

from functools import partial


def get_value(env, history, x, y, n_evaluate_sample, cache_value=True):
    # validation_prompt = env.validation_prompt_wrap(x, y)
    # if validation_prompt:
    #     validation_outputs = gpt_with_history(validation_prompt, history, n=1, stop=None)
    #     validation = env.validation_outputs_unwrap(x, y, validation_outputs)
    #     if validation == 0:
    #         return 0
    value_prompt = env.value_prompt_wrap(x, y)
    if cache_value and value_prompt in env.value_cache:
        return env.value_cache[value_prompt]
    value_outputs = gpt_with_history(value_prompt, history, temperature=0.3, n=n_evaluate_sample, stop=None)
    value = env.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        env.value_cache[value_prompt] = value
    return value


def get_values(env, history, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache and cache_value:  # avoid duplicate candidates
            value = local_value_cache[y]
        else:
            value = get_value(env, history, x, y, n_evaluate_sample, cache_value=cache_value)
            if cache_value:
                local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(env, history, x, ys, n_evaluate_sample):
    vote_prompt = env.vote_prompt_wrap(x, ys)
    vote_outputs = gpt_with_history(vote_prompt, history, n=n_evaluate_sample, stop=None)
    values = env.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(env, history, x, y, n_propose_sample=10):
    propose_prompt = env.propose_prompt_wrap(x, y)
    proposal_list = [x.split('\n') for x in gpt_with_history(propose_prompt, history, n=1, stop=["\n\n"])]
    proposals = []
    for p in proposal_list:
        proposals.extend(p)
    proposals = proposals[:min(len(proposals), n_propose_sample)]
    return [y + _ + '\n' for _ in proposals]


def get_samples(env, history, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = env.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = env.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


class SummarizingTreeOfThoughtAgent(Agent):
    def __init__(self, backend, temperature, prompt_sample, method_generate, method_evaluate, method_select,
                 n_generate_sample,
                 n_evaluate_sample, n_select_sample, summary_size_percentage=20):
        super().__init__()

        global gpt
        global gpt_with_history
        gpt = partial(gpt, model=backend)
        gpt_with_history = partial(gpt_with_history, model=backend)

        self.backend = backend
        self.prompt_sample = prompt_sample
        self.method_generate = method_generate
        self.method_evaluate = method_evaluate
        self.method_select = method_select
        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.n_select_sample = n_select_sample
        self.summary_size_percentage = summary_size_percentage
        self.reflects = []
        self.value_reflects = []

    def plan(self, env, to_print=True):
        print(gpt)
        print(gpt_with_history)
        x = env.puzzle  # input
        history = env.history  # history
        ys = ["\n".join(history) + "\n"] if len(history) else [""]  # current output candidates
        infos = []
        prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use "
        "them with arithmetic operations (+ - * /) to get 24. "
        obs = [{"feedback":prompt},
               {"feedback": "What you have learned about the puzzle are summarized below.\n" + "\n".join(
                   self.reflects)}]
        value_obs = [prompt,
                     dict(feedback="What you have learned about the puzzle are summarized below.\n" + "\n".join(
                         self.value_reflects))]
        for step in range(4 - len(history)):
            # generation
            if self.method_generate == 'sample':
                new_ys = [
                    get_samples(env, obs, x, y, self.n_generate_sample, prompt_sample=self.prompt_sample,
                                stop=env.stops[step])
                    for y in ys]
            elif self.method_generate == 'propose':
                new_ys = [get_proposals(env, obs, x, y, self.n_generate_sample) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            # evaluation
            if self.method_evaluate == 'vote':
                values = get_votes(env, value_obs, x, new_ys, self.n_evaluate_sample)
            elif self.method_evaluate == 'value':
                values = get_values(env, value_obs, x, new_ys, self.n_evaluate_sample, cache_value=False)

            # selection
            if self.method_select == 'sample':
                array_values = np.array(values) + 1e-10
                ps = array_values / sum(array_values)
                select_ids = np.random.choice(ids, size=self.n_select_sample, p=ps).tolist()
            elif self.method_select == 'greedy':
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            # log
            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(
                    f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

            infos.append(
                {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            ys = select_new_ys

        if to_print:
            print(ys)
        # if len(ys):
        #     return ys[0], {'steps': infos}
        ys_list = [y.split('\n')[len(history):] for y in ys]
        res_ys = ["\n".join(ys) for ys in ys_list][0]
        return res_ys, {'steps': infos}

    def reflect(self, env, obs):
        y = obs['answer']
        feedback = obs['feedback']
        reflect_prompt, value_reflect_prompt = env.reflect_prompt_wrap(env.puzzle, y, feedback)

        new_reflection = gpt(reflect_prompt, stop=None)[0]
        new_value_reflection = gpt(value_reflect_prompt, stop=None)[0]

        old_summary = self.reflects[0] if self.reflects else "You are an expert at playing the game 24. You are great at summarizing learnings from past attempts."
        
        input_text = old_summary + "\n" + new_reflection
        input_word_count = len(input_text.split())
        summary_target_size = max(1, int(input_word_count * (self.summary_size_percentage / 100.0)))
        puzzle = env.puzzle

        summary_prompt = (
            f"You are an expert Game of 24 strategist. Your goal is to synthesize learnings from past attempts to solve the current puzzle. "
            f"The current puzzle involves the numbers: {puzzle}.\n\n"
            "Combine the 'Previous Summary' with the 'New Learning' to create a refined, concise strategy. Focus only on insights that will help solve this specific puzzle. Discard generic advice.\n\n"
            f"Previous Summary:\n{old_summary}\n\n"
            f"New Learning from the most recent attempt:\n{new_reflection}\n\n"
            f"Provide an updated summary of actionable strategies for the numbers {puzzle}. The summary should be approximately {summary_target_size} words."
        )
        updated_summary = gpt(summary_prompt, max_tokens=summary_target_size, stop=None)[0]
        self.reflects = [updated_summary]

        old_value_summary = self.value_reflects[0] if self.value_reflects else "You are an expert at evaluating trajectories in the game 24. You are great at summarizing value learnings from past attempts."
        
        value_input_text = old_value_summary + "\n" + new_value_reflection
        value_input_word_count = len(value_input_text.split())
        value_summary_target_size = max(1, int(value_input_word_count * (self.summary_size_percentage / 100.0)))

        value_summary_prompt = (
            "You are an expert evaluator for the Game of 24. Your goal is to refine your judgment on what makes a promising or unpromising step towards solving the puzzle. "
            f"The current puzzle involves the numbers: {puzzle}.\n\n"
            "Synthesize the 'Previous Value Summary' and the 'New Value Learning' into an updated, concise evaluation model. Focus on what these learnings tell you about the value of different moves for this specific puzzle.\n\n"
            f"Previous Value Summary:\n{old_value_summary}\n\n"
            f"New Value Learning from the most recent attempt:\n{new_value_reflection}\n\n"
            f"Provide an updated summary of how to evaluate moves for the numbers {puzzle}. The summary should be approximately {value_summary_target_size} words."
        )

        updated_value_summary = gpt(value_summary_prompt, max_tokens=value_summary_target_size, stop=None)[0]
        self.value_reflects = [updated_value_summary]
        return

    def act(self, env, obs):
        if len(obs['feedback']) >= 1:
            self.reflect(env, obs)
        action, info = self.plan(env)
        if self.reflects:
            info['summary'] = self.reflects[0]
        if self.value_reflects:
            info['value_summary'] = self.value_reflects[0]
        return action,info

    def update(self, obs, reward, done, info):
        if done:
            self.reflects = []
            self.value_reflects = []
