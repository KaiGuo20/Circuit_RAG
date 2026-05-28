
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils_ROG as utils
import random
from typing import Callable, Optional, List, Dict, Any

import re
import string

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


class PromptBuilderLlama3(object):
    """
    Prompt builder for Llama-3/3.1 using messages format.
    Compatible with original Llama-2 logic: add_rule / use_true / use_random / cand context enhancement,
    supports token length truncation.
    """

    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""

    def __init__(
        self,
        tokenizer,
        *,
        # Keep parameters consistent and backward compatible with original Llama2 version
        prompt_path: Optional[str] = None,          # Optional: used as system prompt
        system_prompt: Optional[str] = None,        # Can also pass system prompt directly
        encrypt: bool = False,
        add_rule: bool = False,
        use_true: bool = False,
        use_random: bool = False,
        cot: bool = False,
        explain: bool = False,
        each_line: bool = False,
        maximun_token: int = 4096,
        tokenize: Callable[[str], int] = lambda x: len(x),
    ):
        """
        tokenizer: must have tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_path/system_prompt: if provided, injected as system message (equivalent to original template file).
        """
        self.tokenizer = tokenizer
        self.encrypt = encrypt
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.each_line = each_line
        self.maximun_token = maximun_token
        self.tokenize = tokenize

        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif prompt_path is not None:
            self.system_prompt = self._read_prompt_template(prompt_path)
        else:
            # No forced system prompt; leave empty
            self.system_prompt = None

    # ---------- Basic utilities ----------
    def _read_prompt_template(self, template_file: str) -> str:
        with open(template_file, "r", encoding="utf-8") as fin:
            return fin.read()

    def _build_graph(self, graph_obj, skip_ents: Optional[List[Any]] = None):
        """Unified wrapper for utils.build_graph, compatible with encrypt / skip_ents signature."""
        if skip_ents is None:
            skip_ents = []
        try:
            # In Llama2 version utils.build_graph signature: build_graph(graph, skip_ents, encrypt)
            return utils.build_graph(graph_obj, skip_ents, self.encrypt)
        except TypeError:
            # If utils version is build_graph(graph) or build_graph(graph, skip_ents)
            try:
                return utils.build_graph(graph_obj, skip_ents)
            except TypeError:
                return utils.build_graph(graph_obj)

    def apply_rules(self, graph, rules, source_entities):
        results = []
        for entity in source_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results

    # ---------- Directly answer based on rules ----------
    def direct_answer(self, question_dict: Dict[str, Any]):
        entities = question_dict["q_entity"]
        skip_ents: List[Any] = []
        graph = self._build_graph(question_dict["graph"], skip_ents)

        rules = question_dict.get("predicted_paths", [])
        prediction = []
        if rules:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction

    # ---------- Main entry: construct Llama-3 messages and apply chat template ----------
    def process_input(self, question_dict: Dict[str, Any]) -> str:
        """
        Input: question_dict (same as original)
          - question, choices(list/[]), graph, q_entity, predicted_paths / ground_paths, cand(optional)
        Output: string generated by tokenizer.apply_chat_template (can be fed directly to model)
        """
        question = question_dict["question"]
        if not question.endswith("?"):
            question += "?"

        # 1) Build instruction (includes MCQ/SAQ & rule & COT/EXPLAIN/each_line)
        has_choices = len(question_dict.get("choices", [])) > 0
        if has_choices:
            if self.add_rule or (question_dict.get("cand") is not None):
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        else:
            if self.add_rule or (question_dict.get("cand") is not None):
                instruction = self.SAQ_RULE_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION

        if self.cot:
            instruction += self.COT
        if self.explain:
            instruction += self.EXPLAIN
        if self.each_line:
            instruction += self.EACH_LINE

        # 2) Choice text
        if has_choices:
            choices_text = "\n".join(question_dict["choices"])
            choices_text = self.CHOICES.format(choices=choices_text)
        else:
            choices_text = ""

        # 3) Build reasoning paths context
        lists_of_paths: List[str] = []
        context_text = ""
        graph = None  # also used later for cand

        if self.add_rule:
            entities = question_dict["q_entity"]
            graph = self._build_graph(question_dict["graph"])

            if self.use_true:
                rules = question_dict.get("ground_paths", [])
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict.get("predicted_paths", [])

            if rules:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]

        # 4) If cand is provided, append "ground truth paths" to lists_of_paths
        if question_dict.get("cand") is not None:
            if graph is None:
                # build_graph not yet called when add_rule is False
                graph = self._build_graph(question_dict["graph"])

            # Compute real paths for cand, deduplicate and merge into lists_of_paths
            truth_paths = utils.get_truth_paths(question_dict["q_entity"], question_dict["cand"], graph)
            seen = set(lists_of_paths)
            for p in truth_paths:
                ps = utils.path_to_string(p)
                if ps not in seen:
                    lists_of_paths.append(ps)
                    seen.add(ps)

        # 5) Truncate paths by token limit and build context
        if lists_of_paths:
            # messages rest not yet assembled; conservatively use prompt="" to estimate, only truncate paths
            context = self.check_prompt_length("", lists_of_paths, self.maximun_token, self.tokenizer)
            context_text = self.GRAPH_CONTEXT.format(context=context)
        else:
            context_text = self.GRAPH_CONTEXT.format(context="") if (self.add_rule or question_dict.get("cand") is not None) else ""

        # 6) Assemble user content
        user_content = instruction + "\n\n" + context_text + self.QUESTION.format(question=question) + choices_text

        # 7) Build messages (optional system)
        messages = []
        if self.system_prompt is not None and self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})

        # 8) Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return input_text




    def check_prompt_length(
        self,
        prompt: str,
        list_of_paths: List[str],
        maximum_token: int,
        tokenizer
    ) -> str:
        """
        If too long, shuffle then accumulate incrementally, return when limit exceeded.
        tokenizer: Hugging Face/TransformerLens tokenizer object
        """

        # Check all at once
        all_paths = "\n".join(list_of_paths)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        all_paths_len = len(tokenizer.encode(all_paths, add_special_tokens=False))
        if prompt_len + all_paths_len <= maximum_token:
            return all_paths

        # Shuffle and accumulate incrementally
        paths = list(list_of_paths)
        random.shuffle(paths)

        acc: List[str] = []
        cur_len = prompt_len  # only count special tokens for prompt once

        for p in paths:
            delta = ("\n" if acc else "") + p
            delta_len = len(tokenizer.encode(delta, add_special_tokens=False))
            if cur_len + delta_len > maximum_token:
                return "\n".join(acc)
            acc.append(p)
            cur_len += delta_len

        return "\n".join(acc)
