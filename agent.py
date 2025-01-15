from .utils.utils import save_detections_video, get_object_intervals, tic, toc, extract_timeframe, CustomException, get_video_duration, trim_video
from .utils.llm import VLLM, LLM, remove_files, flush_files
import google.generativeai as genai
from .utils.logger import Logger
import re
import os

VLLM_PROMPT_1 = """\
Based on the provided video, select or provide the correct answer for the user \
question. Break down your reasoning into clear, logical steps, and arrive at \
the most accurate answer.

To ensure accuracy, follow this step-by-step reasoning process:
1. Restate or reframe the question for clarity.
2. Consider key events, actions, or objects relevant to the question.
3. If answer options are provided, assess each option in relation to the \
video's content. If no options are given, logically derive an answer.
4. Provide a clear and concise response based on your reasoning.

You must provide the index of the selected answer or the answer itself, and a \
brief explanation of your reasoning.

""" # + dataset_subsinstruction

VLLM_SCHEMA_1 = {
    "type": "object",
    "properties": {
        "reasoning": { "type": "string" },
        "answer": { "type": "string" }
    }
}

VLLM_PROMPT_2 = """\
Based on the provided video and the given question (and answer options if \
available), capture a list of the main timeframes in the video in the format \
<<mm0:ss0,mm1:ss1>>: {description}, where 'description' is a detailed \
description of what is happening in that particular timeframe.

Follow these steps to generate your response:
1. Carefully analyze the question and the video content to identify the key \
events or actions that are relevant to the question.
2. Identify key events, actions, or transitions that represent meaningful \
changes or notable moments in the video.
3. Break the video into distinct timeframes where these events occur.
4. For each identified timeframe, provide a clear, detailed description of the \
action or scene in that segment.
5. Ensure that each description is specific, concise, and accurately reflects \
the action within the timeframe.
"""

VLLM_SCHEMA_2 = {
    "type": "object",
    "properties": {
        "timeframes": {
            "type": "array",
            "items": { "type": "string" }
        }
    }
}

VLLM_PROMPT_3 = """\
Based on the provided video and the given question (and answer options if \
available), your task is to capture a **list of objects/targets** that are \
involved in the video and are relevant to the question. These targets will \
be used for object detection and grounding via a YOLO model. Please follow \
these steps:

1. Understand the question and its context within the video, along with any \
answer options provided.
2. Focus on the most relevant objects or targets that are involved in the \
video's key actions or scenes. Ensure that these targets directly relate to \
the question.
3. Choose no more than 4 targets, ideally 3 or fewer. Consider only the \
objects that are clearly present and essential to answering the question, \
and that are not too complex to identify (not too large as well), but not \
too general for the particular video.
4. Ensure that the targets are also directly related to the answer options, \
if provided.
5. Provide a short list of targets, ensuring each description is clear and \
relevant (e.g., 'player in white outfit', 'spoon', etc.).
"""

VLLM_SCHEMA_3 = {
    "type": "object",
    "properties": {
        "targets": {
            "type": "array",
            "items": { "type": "string" }
        }
    }
}

VLLM_PROMPT_4 = """\
Based on the provided video, answer the user question in the VERY SPECIFIC \
given timeframe.

Only provide the final, concise answer, directly related to the question. 
Base your answer ONLY on the information in the video, and do not add any \
information. If the answer is not present in the video, state 'unanswerable'. \
For example, if the question is 'What color is the car?', and the car is not \
shown in the video timeframe, the answer should be 'unanswerable'.
"""

VLLM_SCHEMA_4 = {
    "type": "object",
    "properties": {
        "answer": { "type": "string" }
    }
}

LLM_PROMPT_1 = """\
You will be provided with reasoning for an answer to a question, along with \
two grounding pieces of information:
1. **VideoLLM-extracted grounding captions**: These describe the key events \
and timeframes within the video (e.g., <<mm0:ss0,mm1:ss1>>: {description}).
2. **YOLO object grounding**: This identifies the specific objects/targets \
and their appearances in different video timeframes.

Your task is to analyze if there is any disagreement between the grounding \
information (both the captions and object grounding) and the reasoning for \
the answer. Disagreements may occur if the reasoning implies events or objects \
appearing in timeframes that are inconsistent with the grounding.

Please output a "disagree" boolean indicating if there is any disagreement at \
all, and a detailed but concise explanation of the specific timeframes where \
the grounding information does not align with the reasoning. Only include \
timeframes where discrepancies occur, and keep the explanation short but clear. \
If no disagreement is found, simply explain that there is no disagreement.

Disagreements should be highlighted by timeframe (<<mm0:ss0,mm1:ss1>>) and why \
the reasoning conflicts with the provided grounding information.
"""

LLM_SCHEMA_1 = {
    "type": "object",
    "properties": {
        "reasoning": { "type": "string" },
        "disagree": { "type": "boolean" }
    }
}

LLM_CALL_1 = """\
- **Reasoning**: {reasoning1}
- **VideoLLM-extracted grounding captions**: 
{captions}
- **YOLO object grounding**: 
{yolo_grounding}
"""

LLM_PROMPT_2 = """\
You will be provided the following:
1. A question (and answer options if available) related to a video.
2. A text explaining the set of discrepancies found in previous studies of the \
video. These indicate specific timeframes in the video where the grounding \
information does not align with the reasoning. These timeframes and the reasons \
for the discrepancies are provided.

Your task is to generate a set of up to 3 concise questions to ask a VideoLLM \
to clarify and provide a more grounded, precise answer. The goal is to resolve \
the discrepancies and improve the grounding for the question at hand.

- Each question should focus on a specific timeframe where a discrepancy was \
found.
- Each question should be concise and relevant to the timeframe, and \
particularly relevant to answer the question.
- Ensure that each question includes the timeframe where the clarification \
is needed, formatted as <<mm0:ss0,mm1:ss1>>.
- The timeframe must be very precise in time, covering only the specific \
segment where the discrepancy occurred.
- Do not include any unnecessary details, just the specific query for \
clarification.
- If there are not CONSIDERABLE discrepancies, you may return an empty list!

Generate between 0 and up to 3 questions based on the discrepancies identified.
"""

LLM_SCHEMA_2 = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": { "type": "string" }
        }
    }
}

LLM_CALL_2 = """\
{prompt}
- **Discrepancies**: 
{discrepancies}
- **Video duration**: {video_duration}
"""

LLM_PROMPT_3 = """\
You will be provided the following:
1. A question (and answer options if available) related to a video.
2. An initial reasoning made for a possible answer, along with an \
explanation of why it was chosen. This reasoning was done BEFORE \
knowing the grounding information, and clarification questions.
3. The **grounding information**:
    - **VideoLLM grounding**: Timeframes and event descriptions from the video.
    - **YOLO object grounding**: Objects/targets identified in the video and their corresponding appearing timeframes.
4. A set of clarification questions asked about discrepancies in the \
grounding, and their responses.

Your task is to:
1. Analyze all the provided information and reasoning.
2. Select or provide the correct answer for the user question, based on the \
new clarifications from the questions and grounding data.
3. Provide the final, most accurate specific answer, as well as a reasoning \
for it.

Remember to stick to the information provided, and ensure that your answer \
is accurate and well-supported by the grounding information and reasoning \
provided. If none of the answer options are correct, select the most \
appropiate based on the new information and reasoning.

"""# + dataset_subsinstruction

LLM_SCHEMA_3 = {
    "type": "object",
    "properties": {
        "reasoning": { "type": "string" },
        "answer": { "type": "string" }
    }
}

LLM_CALL_3 = """\
{prompt}
- **Reasoning**: {reasoning1}
- **VideoLLM-extracted grounding captions**: 
{captions}
- **YOLO object grounding**: 
{yolo_grounding}
- **Clarification Questions already answered**: 
{qa_str}
"""

LLM_CALL_3_NOQA = """\
{prompt}
- **Reasoning**: {reasoning1}
- **VideoLLM-extracted grounding captions**: 
{captions}
- **YOLO object grounding**: 
{yolo_grounding}
"""


class ViQAgent():
    def __init__(self, model_name, api_key, dataset_subinstruction="", log_config={}, yolo_params={}, llm_params={}):
        self.log = (
            log_config if isinstance(log_config, Logger) else Logger(**log_config)
        ).log
        genai.configure(api_key=api_key)
        try: from .utils.yolo import YOLO
        except: raise ImportError("YOLO-World requires GPU, but no GPU was found")
        self.yolo = YOLO("yolo_world/l", **yolo_params)
        self.videollm1 = VLLM(model_name, VLLM_PROMPT_1+dataset_subinstruction, VLLM_SCHEMA_1, log=self.log, **llm_params)
        self.videollm2 = VLLM(model_name, VLLM_PROMPT_2, VLLM_SCHEMA_2, log=self.log, **llm_params)
        self.videollm3 = VLLM(model_name, VLLM_PROMPT_3, VLLM_SCHEMA_3, log=self.log, **llm_params)
        self.videollm4 = VLLM(model_name, VLLM_PROMPT_4, VLLM_SCHEMA_4, log=self.log, **llm_params)
        self.llm1 = LLM(model_name, LLM_PROMPT_1, LLM_SCHEMA_1, log=self.log, **llm_params)
        self.llm2 = LLM(model_name, LLM_PROMPT_2, LLM_SCHEMA_2, log=self.log, **llm_params)
        self.llm3 = LLM(model_name, LLM_PROMPT_3+dataset_subinstruction, LLM_SCHEMA_3, log=self.log, **llm_params)

    def rm_cache(self):
        remove_files(self.videollm1.last_execution_files)
        self.log(f"Removed from cache {len(self.videollm1.last_execution_files)} files")

    def invoke(self, video, query, answer_options=[]):
        opts_str = "\n".join([f"{i}. {v}" for i,v in enumerate(answer_options)])
        if opts_str:
            prompt = f"- **Question**: {query}\n- **Possible answers**:\n{opts_str}"
        else:
            prompt = f"- **Question**: {query}"
        responses = {}
        usages = {}
        delays = {}

        responses['metadata'] = {
            'video_duration': get_video_duration(video),
        }

        flush_files()

        self.m1(video, prompt, responses, usages, delays)
        self.og(video, responses, delays)
        self.m2(video, prompt, responses, usages, delays)

        return responses['vllm1']['answer'], responses['llm3']['answer']
    
    def m1(self, video, prompt, responses, usages, delays):
        #self.log(f"VLLM Prompt:\n{prompt}")

        tic()
        r1, usages['vllm1'] = self.videollm1(video, prompt)
        responses['vllm1'] = r1
        delays['vllm1'] = toc()
        self.log(f"VLLM1 response:\n{r1}\n")

        tic()
        r2, usages['vllm2'] = self.videollm2(video, prompt)
        responses['vllm2'] = r2
        delays['vllm2'] = toc()
        self.log(f"VLLM2 response:\n{r2}\n")

        tic()
        r3, usages['vllm3'] = self.videollm3(video, prompt)
        responses['vllm3'] = r3
        delays['vllm3'] = toc()
        self.log(f"VLLM3 response:\n{r3}\n")

    def m1_qa(self, video, questions, responses, usages, delays, trim=False):
        answers = []
        for i, _q in enumerate(questions):
            i += 1
            tic()
            if trim:
                match = re.search(r"<<(\d{2}:\d{2},\d{2}:\d{2})>>", _q)
                timeframe = match.group(1) if match else None
                if timeframe:
                    _q = re.sub(r"<<\d{2}:\d{2},\d{2}:\d{2}>>", "", _q).strip()
                    _video = trim_video(video, timeframe)
                else:
                    _video = video

                try:

                    r, tk = self.videollm4(_video, _q)
                    if timeframe: os.remove(_video)
                except Exception as e: 
                    if timeframe: os.remove(_video)
                    raise e
            else:
                r, tk = self.videollm4(video, _q)

            usages[f'vllm4_{i}'] = tk
            responses[f'vllm4_{i}'] = r
            delays[f'vllm4_{i}'] = toc()
            _r = r['answer']
            answers.append(_r)
            self.log(f"VLLM4 question {i}: {_q}\nVLLM4 response {i}: {_r}\n")

        return answers

    def og(self, video, responses, delays):
        classes = responses['vllm3']['targets']         # T

        tic()
        detections = self.yolo.process_video(classes, video)
        save_detections_video(detections, video, f"{video[:-4]}_yolo.{video[-3:]}")
        object_intervals = get_object_intervals(classes, detections, video)
        responses['yw'] = object_intervals
        delays['yw'] = toc()
        self.log(f"YOLO detections:\n{object_intervals}\n")

    def m2(self, video, prompt, responses, usages, delays):
        video_duration = responses['metadata']['video_duration']
        reasoning1 = responses['vllm1']['reasoning']    # R1
        captions = responses['vllm2']['timeframes']     # TC
        object_intervals = responses['yw']              
        yolo_grounding = '\n'.join([                    # TG
            f"""- {cls}: {', '.join([
                f'[{start} - {end}]' for start, end in intervals
            ])}""" for cls, intervals in object_intervals.items()
        ])

        llm1_prompt = LLM_CALL_1.format(reasoning1=reasoning1, captions=captions, yolo_grounding=yolo_grounding)
        #self.log(f"LLM1 prompt:\n{llm1_prompt}\n")
        tic()
        r1, usages['llm1'] = self.llm1(llm1_prompt)
        responses['llm1'] = r1
        delays['llm1'] = toc()
        self.log(f"LLM1 response:\n{r1}\n")

        disagreement = r1['disagree']
        discrepancies = r1['reasoning']

        if disagreement:
            llm2_prompt = LLM_CALL_2.format(prompt=prompt, discrepancies=discrepancies, video_duration=video_duration)
            #self.log(f"LLM2 prompt:\n{llm2_prompt}\n")
            tic()
            r2, usages['llm2'] = self.llm2(llm2_prompt)
            responses['llm2'] = r2
            delays['llm2'] = toc()
            self.log(f"LLM2 response:\n{r2}\n")

            questions = r2['questions']
            answers = self.m1_qa(video, questions, responses, usages, delays)
            qa_str = "\n".join([f"- {q} - {a}" for i,(q,a) in enumerate(list(zip(questions, answers)))])

            llm3_prompt = LLM_CALL_3.format(prompt=prompt, reasoning1=reasoning1, captions=captions, yolo_grounding=yolo_grounding, qa_str=qa_str)
            #self.log(f"LLM3 prompt:\n{llm3_prompt}\n")
            tic()
            responses['llm3'], usages['llm3'] = self.llm3(llm3_prompt)
            delays['llm3'] = toc()
            self.log(f"LLM3 response:\n{responses['llm3']}\n")
        else:
            llm3_prompt = LLM_CALL_3_NOQA.format(prompt=prompt, reasoning1=reasoning1, captions=captions, yolo_grounding=yolo_grounding)
            #self.log(f"LLM3 prompt:\n{llm3_prompt}\n")
            tic()
            responses['llm3'], usages['llm3'] = self.llm3(llm3_prompt)
            delays['llm3'] = toc()
            self.log(f"LLM3 response:\n{responses['llm3']}\n")
