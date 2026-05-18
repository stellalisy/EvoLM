from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from open_instruct.search_rewards.utils.rubric_chat_templates import (
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT,
    V0_RUBRIC_GENERATION_SYSTEM_PROMPT,
)

import utils.prompts as prompts
import utils.models as models

# run judge on pairs
# judges should take a question and two responses, and return a decision (e.g., A>B or B>A)
# Note B>A and not A<B
# Judge.get_judgment must be async!
# For a new judge, add a corresponding entry to get_judge_from_judge_name_and_model


class Judge(ABC):
    @abstractmethod
    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        pass


class ArenaHard(Judge):
    # Implementation follows 
    # https://github.com/lmarena/arena-hard-auto/blob/4ce0f0087776158a4461162cbef1d9bb5464bb57/gen_judgment.py

    def __init__(self, model_name):
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.number_of_judgment_attempts = 2

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        system_message = prompts.render_template(
            "arena_hard_judge_system")
        user_message = prompts.render_template("arena_hard_judge_prompt",
                                                    prompt=question, answer_a=answer_A, answer_b=answer_B)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        judgment = ""
        for _ in range(self.number_of_judgment_attempts):
            new_judgment = await self.api.chat(
                messages=messages,
                temperature=0,
                max_tokens=4096,
            )
            judgment += ("\n" + new_judgment)
            score, try_again = self.get_score(
                judgment, re.compile("\[\[([AB<>=]+)\]\]"))
            messages.append({"role": "assistant", "content": new_judgment})
            if not try_again:
                break
            messages.append(
                {"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": messages[1]["content"],
                "response": judgment
            },
            "decision": score.replace(">>", ">").strip() if score else None
        }

    def get_score(cls, judgment: str, pattern: str, pairwise: bool = True) -> Tuple[Union[int, str], Optional[bool]]:
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None, True
        elif len(set(matches)) == 1:
            if pairwise:
                return matches[0].strip("\n"), False
            return int(matches[0])
        else:
            return None, False
        
class Vanilla(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    def extract_pairwise_result(self, raw_output):
        print("raw:", raw_output)
        if raw_output == "Output (a)":
            return "A>B"
        elif raw_output == "Output (b)":
            return "B>A"
        raise Exception("Cannot parse output:", raw_output)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "vanilla_prompt", question=question, answer_a=answer_A, answer_b=answer_B)
        print("prompt:", prompt)
        output = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )

        pred_label = self.extract_pairwise_result(output.strip())

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": pred_label
        }

class PandaLM(Judge):

    def __init__(self, model_name) -> None:
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.pattern = re.compile(
            r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, use_fast=False)
        
    def truncate_responses(self, question, answer_A, answer_B, context_limit, max_new_tokens, truncation_side):
        template_with_question = prompts.render_template("pandalm_prompt", instruction=question, resp1="", resp2="")
        len_template = len(self.tokenizer(template_with_question).input_ids) # includes special BOS token <s>
        tokens_per_response = (context_limit - max_new_tokens - len_template) // 2 - 2 # each response should be truncated to a length of tokens_per_response
        
        answer_A_tokenized = self.tokenizer(
            answer_A,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_A_tokenized_truncated = answer_A_tokenized[:tokens_per_response] if truncation_side == "right" else answer_A_tokenized[-tokens_per_response:] # left
        answer_A_truncated = self.tokenizer.decode(answer_A_tokenized_truncated) # should not be any special tokens anyways
        
        answer_B_tokenized = self.tokenizer(
            answer_B,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_B_tokenized_truncated = answer_B_tokenized[:tokens_per_response] if truncation_side == "right" else answer_B_tokenized[-tokens_per_response:] # left
        answer_B_truncated = self.tokenizer.decode(answer_B_tokenized_truncated) # should not be any special tokens anyways
        
        return answer_A_truncated, answer_B_truncated

    def build_pandalm_prompt(self, instruction, resp1, resp2):
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        input_sequence = prompts.render_template(
            "pandalm_prompt", instruction=instruction, resp1=resp1, resp2=resp2)
        return input_sequence + "\n" # why does jinja strip the training new line?

    def parse_pandalm_response(self, text):
        sp = text.strip().split("\n")
        if sp[0] in ["1", "2"]:
            return int(sp[0])
        elif sp[0].lower() == "tie":
            return 0
        else:
            return 0

    def postprocess_output(self, text):
        text = text.strip()
        self.pattern.sub("", text.strip()).strip()
        return text

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        
        answer_A, answer_B = self.truncate_responses(
            question,
            answer_A,
            answer_B,
            context_limit=2048,
            max_new_tokens=150, # we only need the first few tokens to determine decision
            truncation_side="left"
        )

        prompt = self.build_pandalm_prompt(
            instruction=question,
            resp1=answer_A,
            resp2=answer_B,
        )

        output = await self.api.complete(
            prompt=prompt,
            temperature=0,
            top_p=1,
            max_tokens=150,
            extra_body={
                "use_beam_search": True,
                "best_of": 4,
                "early_stopping": True,
                "repetition_penalty": 1.2,
            },
        )

        resp = self.postprocess_output(output)
        out = self.parse_pandalm_response(resp)
        if out == 1:
            decision = "A>B"
        elif out == 2:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": resp,
            },
            "decision": decision
        }


class JudgeLM(Judge):
    from transformers import AutoTokenizer
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, use_fast=False)
        
    def truncate_responses(self, question, answer_A, answer_B, context_limit, max_new_tokens, truncation_side):
        template_with_question = prompts.render_template("judgelm_prompt", question=question, answer_1="", answer_2="")
        len_template = len(self.tokenizer(template_with_question).input_ids) # includes special BOS token <s>
        tokens_per_response = (context_limit - max_new_tokens - len_template) // 2 - 2 # each response should be truncated to a length of tokens_per_response
        
        answer_A_tokenized = self.tokenizer(
            answer_A,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_A_tokenized_truncated = answer_A_tokenized[:tokens_per_response] if truncation_side == "right" else answer_A_tokenized[-tokens_per_response:] # left
        answer_A_truncated = self.tokenizer.decode(answer_A_tokenized_truncated) # should not be any special tokens anyways
        
        answer_B_tokenized = self.tokenizer(
            answer_B,
            add_special_tokens=False, # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_B_tokenized_truncated = answer_B_tokenized[:tokens_per_response] if truncation_side == "right" else answer_B_tokenized[-tokens_per_response:] # left
        answer_B_truncated = self.tokenizer.decode(answer_B_tokenized_truncated) # should not be any special tokens anyways
        
        return answer_A_truncated, answer_B_truncated

    def parse_score(self, review):
        try:
            score_pair = review.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            if len(sp) == 2:
                return [float(sp[0]), float(sp[1])]
            else:
                raise Exception()
        except Exception:
            return [-1, -1]

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        
        answer_A, answer_B = self.truncate_responses(
            question,
            answer_A,
            answer_B,
            context_limit=2048,
            max_new_tokens=16,
            truncation_side="right"
        )
        
        prompt = prompts.render_template(
            "judgelm_prompt", question=question, answer_1=answer_A, answer_2=answer_B)

        output = await self.api.complete(
            prompt=prompt,
            temperature=0.0, # https://github.com/baaivision/JudgeLM/blob/ce12b12779764fe06e28c797cecee86018a298e4/judgelm/llm_judge/gen_model_judgement_multi.py#L235
            max_tokens=16,
        )

        scores = self.parse_score(output)

        if scores[0] > scores[1]:
            decision = "A>B"
        elif scores[0] < scores[1]:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision
        }


class AutoJ(Judge):

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    def extract_pariwise_result(self, raw_output):
        raw_output = raw_output.strip()
        pos = raw_output.rfind('final decision is ')
        pred_label = None
        if pos != -1:
            pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
            if pred_rest.startswith('response 1'):
                pred_label = "A>B"
            elif pred_rest.startswith('response 2'):
                pred_label = "B>A"
            elif pred_rest.startswith('tie'):
                pred_label = "A=B"
        return pred_label

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "autoj_prompt", question=question, response=answer_A, response_another=answer_B)

        output = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )  # SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024) https://github.com/GAIR-NLP/auto-j

        pred_label = self.extract_pariwise_result(output)

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": pred_label
        }


class Prometheus2(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.rubric = "[Are the model's responses factually correct and well-supported by evidence?]" # https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/prompts.py
        self.REL_SYSTEM_PROMPT = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

    def _parse_output_relative(self, output):
        explicit_pattern = r"""
            (?:                                # Start of non-capturing group
                \[RESULT\]|\[RESULT:\s*|        # Match [RESULT] or [RESULT:
                \[Response\s+|                  # Match [Response
                # Match [Result] or [Result] Response
                \[Result\](?:\s+Response)?|
                \[Result:\s*|                   # Match [Result:
                # Match Result: at the start of a line
                (?:^|\n)Result:?\s*
            )                                   # End of non-capturing group
            \s*                                 # Allow any whitespace
            (A|B)                               # Capture A or B
            (?:\]|\s|$)                         # Allow closing bracket, whitespace, or end of string
        """
        match = re.search(
            explicit_pattern, output, re.IGNORECASE | re.VERBOSE | re.MULTILINE
        )

        if match:
            result = match.group(1).upper()
            feedback = output[: match.start()].strip()
            return output, result

        return None, None

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "prometheus2_prompt",
            instruction=question,
            response_A=answer_A,
            response_B=answer_B,
            rubric=self.rubric,
        )

        messages = [
            {"role": "system", "content": self.REL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        ) # https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/utils.py
        
        _, scores = self._parse_output_relative(output)
        
        decision = None # no tie option 
        if scores == "A":
            decision = "A>B"
        elif scores == "B":
            decision = "B>A"
        
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision
        }
        

class SkyworkCritic(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "skywork_critic_prompt",
            input=question,
            response_a=answer_A,
            response_b=answer_B,
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        
        if "A" in output:
            decision = "A>B"
        elif "B" in output:
            decision = "B>A"
        else:
            decision = None
        
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision
        }
        
        
class InternLM2Reward(Judge):
    def __init__(self, model_name="internlm/internlm2-20b-reward", device="cuda:0"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_name = model_name
        self.device = device
        self.rm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        ).to(self.device)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        conv1 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_A}]
        conv2 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_B}]

        score1 = self.rm.get_score(self.rm_tokenizer, conv1)
        score2 = self.rm.get_score(self.rm_tokenizer, conv2)
        
        judgement = 'A>B' if score1 > score2 else 'B>A'

        return {
            "judgment": {
                "judge_model": self.model_name,
                "scores": [score1, score2]
            },
            "decision": judgement
        }
        

class GRMReward(Judge):
    def __init__(self, model_name="Ray2333/GRM-Gemma-2B-rewardmodel-ft", device="cuda:0"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
        ).to(self.device)

    def get_reward(self, message):
        import torch
        message_template = self.tokenizer.apply_chat_template(message, tokenize=False)
        
        kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer.encode_plus(message_template, **kwargs)
        
        with torch.no_grad():
            reward_tensor = self.reward_model(
                tokens["input_ids"][0].view(1,-1).to(self.device), 
                attention_mask=tokens["attention_mask"][0].view(1,-1).to(self.device)
            )[0]
            reward = reward_tensor.cpu().detach().item()
        
        return reward

    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        message_A = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer_A}
        ]
        message_B = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer_B}
        ]

        score_A = self.get_reward(message_A)
        score_B = self.get_reward(message_B)

        judgement = 'A>B' if score_A > score_B else 'B>A'

        return {
            "judgment": {
                "judge_model": self.model_name,
                "scores": [score_A, score_B]
            },
            "decision": judgement
        }


class SkyworkReward(Judge):
    def __init__(self, model_name, device="cuda:0"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.model_name = model_name
        self.device = device
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def get_judgment(self, question: str, answer_A: str, answer_B: str, **kwargs) -> Dict[str, Any]:
        import torch
        conv1 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_A}]
        conv2 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_B}]

        conv1_formatted = self.rm_tokenizer.apply_chat_template(conv1, tokenize=False)
        conv2_formatted = self.rm_tokenizer.apply_chat_template(conv2, tokenize=False)
        conv1_tokenized = self.rm_tokenizer(conv1_formatted, return_tensors="pt").to(self.device)
        conv2_tokenized = self.rm_tokenizer(conv2_formatted, return_tensors="pt").to(self.device)

        # Get the reward scores
        with torch.no_grad():
            score1 = self.rm(**conv1_tokenized).logits[0][0].item()
            score2 = self.rm(**conv2_tokenized).logits[0][0].item()

        judgement = 'A>B' if score1 > score2 else 'B>A'

        return {
            "judgment": {
                "judge_model": self.model_name,
                "scores": [score1, score2]
            },
            "decision": judgement
        }
        

class CompassJudger(Judge):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        
    def get_score(cls, judgment: str, pattern: str, pairwise: bool = True) -> Tuple[Union[int, str], Optional[bool]]:
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None, True
        elif len(set(matches)) == 1:
            if pairwise:
                return matches[0].strip("\n"), False
            return int(matches[0])
        else:
            return None, False
        
    async def get_judgment(self, question: str, answer_A: str, answer_B: str) -> Dict[str, Any]:
        system_message = prompts.render_template(
            "arena_hard_judge_system")
        user_message = prompts.render_template("arena_hard_judge_prompt",
                                                    prompt=question, answer_a=answer_A, answer_b=answer_B)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        
        score, _ = self.get_score(output, re.compile("\[\[([AB<>=]+)\]\]"))
        
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": messages[1]["content"],
                "response": output,
            },
            "decision": score.replace(">>", ">").strip() if score else None
        }


class RubricJudge(Judge):
    """Rubric-based judge: generate a rubric, then score each answer independently."""

    _RUBRIC_GEN_SYS = DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT

    _SCORE_SYS = DEFAULT_JUDGE_SYSTEM_PROMPT

    _SCORE_USER_TPL = """Question: {question}

Rubric: {rubric}

Answer to evaluate: {response}

Evaluate the answer against the rubric. For each criterion, decide how well the answer satisfies it (0.0 = not at all, 1.0 = fully), then multiply by the criterion's weight. Sum the weighted scores to get the total (must be between 0.0 and 1.0).

Output ONLY valid JSON:
{{"reasoning": "<evaluate each criterion, give satisfaction * weight, then sum>", "score": <float 0.0-1.0>}}

Example 1 (rubric: Factual Accuracy 0.4, Completeness 0.35, Clarity 0.25):
{{"reasoning": "Factual Accuracy (weight 0.4): answer is fully correct, 1.0 * 0.4 = 0.4. Completeness (weight 0.35): covers main points but misses edge cases, 0.6 * 0.35 = 0.21. Clarity (weight 0.25): well organized and easy to follow, 1.0 * 0.25 = 0.25. Total = 0.86", "score": 0.86}}
Example 2 (rubric: Correctness of Solution 0.5, Use of Examples 0.3, Appropriate Detail 0.2):
{{"reasoning": "Correctness of Solution (weight 0.5): correct approach but has an arithmetic error in the final step, 0.8 * 0.5 = 0.4. Use of Examples (weight 0.3): no examples provided, 0.0 * 0.3 = 0.0. Appropriate Detail (weight 0.2): gives a brief answer without elaboration, 0.2 * 0.2 = 0.04. Total = 0.44", "score": 0.44}}

Your evaluation:"""

    _API_PREFIXES = ("gpt-", "o1-", "claude-", "gemini-")

    def __init__(self, rubric_model_name: str, judge_model_name: str,
                 rubric_port: int = 8000, judge_port: int = 8001,
                 rubric_prompt_key: str | None = None):
        from rewardbench.generative_v2_rubric import strip_thinking_tokens
        from transformers import AutoTokenizer

        self.rubric_model_name = rubric_model_name
        self.judge_model_name = judge_model_name
        self._rubric_is_api = any(rubric_model_name.startswith(p) for p in self._API_PREFIXES)
        self._judge_is_api = any(judge_model_name.startswith(p) for p in self._API_PREFIXES)
        self.rubric_api = models.get_chat_api_from_model(rubric_model_name, port=rubric_port)
        self.judge_api = models.get_chat_api_from_model(judge_model_name, port=judge_port)
        if rubric_prompt_key is not None:
            from open_instruct.search_rewards.utils.rubric_chat_templates import get_rubric_system_prompt
            self._RUBRIC_GEN_SYS = get_rubric_system_prompt(rubric_prompt_key)
            print(f"RubricJudge: using rubric prompt key '{rubric_prompt_key}'")
        if not self._rubric_is_api:
            self._rubric_tokenizer = AutoTokenizer.from_pretrained(
                rubric_model_name, trust_remote_code=True
            )
            if not getattr(self._rubric_tokenizer, "chat_template", None):
                _fallback = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
                self._rubric_tokenizer.chat_template = _fallback.chat_template
        if not self._judge_is_api:
            self._judge_tokenizer = AutoTokenizer.from_pretrained(
                judge_model_name, trust_remote_code=True
            )
            if not getattr(self._judge_tokenizer, "chat_template", None):
                _fallback = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
                self._judge_tokenizer.chat_template = _fallback.chat_template
        self._strip = strip_thinking_tokens

    @staticmethod
    def _is_valid_rubric(rubric: str) -> bool:
        return bool(rubric) and bool(re.search(r'\{\s*"criteria"\s*:', rubric))

    _RUBRIC_MAX_RETRIES = 3
    _RUBRIC_RETRY_TEMPS = (0.6, 0.3, 0.1)

    async def generate_rubric(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self._RUBRIC_GEN_SYS},
            {"role": "user", "content": question},
        ]

        rubric = ""
        for attempt in range(self._RUBRIC_MAX_RETRIES):
            temp = self._RUBRIC_RETRY_TEMPS[min(attempt, len(self._RUBRIC_RETRY_TEMPS) - 1)]
            if self._rubric_is_api:
                raw = await self.rubric_api.chat(
                    messages, temperature=temp, top_p=0.95, max_tokens=16384,
                )
            else:
                prompt = self._rubric_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt += "\n" + self._RUBRIC_GEN_SYS + "\n"
                raw = await self.rubric_api.complete(
                    prompt=prompt, temperature=temp, top_p=0.95, max_tokens=16384,
                )
            rubric = self._strip(raw)
            if self._is_valid_rubric(rubric):
                return rubric
            if attempt < self._RUBRIC_MAX_RETRIES - 1:
                print(f"RubricJudge: invalid rubric (attempt {attempt+1}, "
                      f"len={len(rubric)}), retrying with temp={self._RUBRIC_RETRY_TEMPS[min(attempt+1, len(self._RUBRIC_RETRY_TEMPS)-1)]}")

        print(f"RubricJudge: all {self._RUBRIC_MAX_RETRIES} rubric attempts failed "
              f"(len={len(rubric)})")
        return rubric

    @staticmethod
    def _parse_score(text: str) -> float:
        if not text or not text.strip():
            return -1.0
        m = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text.strip())
        if m:
            return float(m.group(1))
        return -1.0

    async def _score_response(self, question: str, rubric: str,
                               response_text: str) -> Tuple[float, str]:
        user_prompt = self._SCORE_USER_TPL.format(
            question=question, rubric=rubric, response=response_text,
        )
        messages = [
            {"role": "system", "content": self._SCORE_SYS},
            {"role": "user", "content": user_prompt},
        ]
        try:
            if self._judge_is_api:
                raw = await self.judge_api.chat(
                    messages, temperature=0.6, top_p=0.95, max_tokens=16384,
                )
            else:
                prompt = self._judge_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                raw = await self.judge_api.complete(
                    prompt=prompt, temperature=0.6, top_p=0.95, max_tokens=16384,
                )
            cleaned = self._strip(raw)
            score = self._parse_score(cleaned)
            return score, cleaned
        except Exception as e:
            print(f"RubricJudge: scoring failed: {e}")
            return -1.0, ""

    async def get_judgment(self, question: str, answer_A: str, answer_B: str,
                           rubric: str | None = None) -> Dict[str, Any]:
        if rubric is None:
            rubric = await self.generate_rubric(question)

        (score_A, raw_A), (score_B, raw_B) = await asyncio.gather(
            self._score_response(question, rubric, answer_A),
            self._score_response(question, rubric, answer_B),
        )

        if score_A < 0 and score_B < 0:
            decision = None
        elif score_A > score_B:
            decision = "A>B"
        elif score_B > score_A:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.judge_model_name,
                "rubric_model": self.rubric_model_name,
                "rubric": rubric,
                "score_A": score_A,
                "score_B": score_B,
                "raw_judge_A": raw_A,
                "raw_judge_B": raw_B,
            },
            "decision": decision,
        }


class NoRubricJudge(Judge):
    """Baseline: score responses without rubric generation."""

    _SCORE_SYS = DEFAULT_JUDGE_SYSTEM_PROMPT

    _SCORE_USER_TPL = """Question: {question}

Answer to evaluate: {response}

Evaluate the answer based on correctness, completeness, and helpfulness. Go through each aspect and assign a score. Then add up the scores to get the total score.

Output ONLY valid JSON in the following format:
{{
  "reasoning": "<brief explanation of your evaluation>",
  "score": <total score (a float between 0.0 to 1.0)>
}}

Example outputs:
{{"reasoning": "Answer addresses all key points with accurate details", "score": 0.9}}
{{"reasoning": "Answer partially satisfies the criteria but missing some critical elements", "score": 0.5}}
{{"reasoning": "Answer does not meet the criteria", "score": 0.1}}

Your evaluation:"""

    def __init__(self, judge_model_name: str, judge_port: int = 8001):
        from rewardbench.generative_v2_rubric import strip_thinking_tokens
        from transformers import AutoTokenizer

        self.judge_model_name = judge_model_name
        self.judge_api = models.get_chat_api_from_model(judge_model_name, port=judge_port)
        self._judge_tokenizer = AutoTokenizer.from_pretrained(
            judge_model_name, trust_remote_code=True
        )
        self._strip = strip_thinking_tokens

    @staticmethod
    def _parse_score(text: str) -> float:
        if not text or not text.strip():
            return -1.0
        m = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text.strip())
        if m:
            return float(m.group(1))
        return -1.0

    async def _score_response(self, question: str, response_text: str) -> Tuple[float, str]:
        user_prompt = self._SCORE_USER_TPL.format(
            question=question, response=response_text,
        )
        messages = [
            {"role": "system", "content": self._SCORE_SYS},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            raw = await self.judge_api.complete(
                prompt=prompt, temperature=0.6, top_p=0.95, max_tokens=16384,
            )
            cleaned = self._strip(raw)
            score = self._parse_score(cleaned)
            return score, cleaned
        except Exception as e:
            print(f"NoRubricJudge: scoring failed: {e}")
            return -1.0, ""

    async def get_judgment(self, question: str, answer_A: str, answer_B: str,
                           rubric: str | None = None) -> Dict[str, Any]:
        (score_A, raw_A), (score_B, raw_B) = await asyncio.gather(
            self._score_response(question, answer_A),
            self._score_response(question, answer_B),
        )

        if score_A < 0 and score_B < 0:
            decision = None
        elif score_A > score_B:
            decision = "A>B"
        elif score_B > score_A:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.judge_model_name,
                "score_A": score_A,
                "score_B": score_B,
                "raw_judge_A": raw_A,
                "raw_judge_B": raw_B,
            },
            "decision": decision,
        }


def get_judge_from_judge_name_and_model(judge_name: str, judge_model: str, **kwargs) -> Judge:
    if judge_name == "arena_hard":
        return ArenaHard(judge_model)
    elif judge_name == "vanilla":
        return Vanilla(judge_model)
    elif judge_name == "panda_lm":
        return PandaLM(judge_model) 
    elif judge_name == "judge_lm":
        return JudgeLM(judge_model)
    elif judge_name == "auto_j":
        return AutoJ(judge_model)
    elif judge_name == "prometheus_2":
        return Prometheus2(judge_model)
    elif judge_name == "skywork_critic":
        return SkyworkCritic(judge_model)
    elif judge_name == "compass_judger":
        return CompassJudger(judge_model)
    elif judge_name == "rubric":
        return RubricJudge(
            rubric_model_name=kwargs["rubric_model_name"],
            judge_model_name=judge_model,
            rubric_port=kwargs.get("rubric_port", 8000),
            judge_port=kwargs.get("judge_port", 8001),
            rubric_prompt_key=kwargs.get("rubric_prompt_key"),
        )
    elif judge_name == "no_rubric":
        return NoRubricJudge(
            judge_model_name=judge_model,
            judge_port=kwargs.get("judge_port", 8001),
        )
    elif judge_name == "reward_model":
        if judge_model in ["internlm/internlm2-7b-reward", "internlm/internlm2-20b-reward"]:
            return InternLM2Reward(judge_model)
        elif judge_model in ["Ray2333/GRM-Gemma-2B-rewardmodel-ft"]:
            return GRMReward(judge_model)
        elif judge_model in [
            "Skywork/Skywork-Reward-Gemma-2-27B",
            "Skywork/Skywork-Reward-Llama-3.1-8B",
            "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
        ]:
            return SkyworkReward(judge_model)
        else:
            raise NotImplementedError(
                f"Judge with name {judge_name} for model with name {judge_model} is not yet implemented.")
    else:
        raise NotImplementedError(
            f"Judge with name {judge_name} for model with name {judge_model} is not yet implemented.")