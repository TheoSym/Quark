p = "/root/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py"
s = open(p).read()
old = """        if is_multimodal:
            request_prompts = [processed_messages.prompt]
            engine_prompts = [processed_messages.prompt]
        else:
            request_prompts = [processed_messages.prompt_ids]
            engine_prompts = [processed_messages.prompt_ids]
"""
new = """        # Backport of the _engine_prompt helper from SGLang main: token-first
        # multimodal encoders (DeepSeek-V4.1) pre-render input_ids and leave the
        # text prompt empty; sending that empty text made every /v1/responses
        # call fail with "texts cannot be empty". Pass the ids through instead.
        if is_multimodal and processed_messages.prompt:
            engine_prompt = processed_messages.prompt
        else:
            engine_prompt = processed_messages.prompt_ids
        request_prompts = [engine_prompt]
        engine_prompts = [engine_prompt]
"""
n = s.count(old)
assert n == 1, f"expected one match, found {n}"
open(p, "w").write(s.replace(old, new))
print("patched")
