import os
import sys
import json
import traceback

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    print("Error: jinja2 is required to run tests. Please install it using 'pip install jinja2'")
    sys.exit(1)

TEMPLATE_FILE = 'chat_template.jinja'
TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    lstrip_blocks=True,
    trim_blocks=True
)

def raise_exception(msg):
    raise Exception(msg)

env.globals['raise_exception'] = raise_exception

try:
    template = env.get_template(TEMPLATE_FILE)
except Exception as e:
    print(f"Error loading template: {e}")
    sys.exit(1)

def run_test(name, messages, tools=None, kwargs=None, expected_in=None, expected_not_in=None, expect_error=False):
    if kwargs is None:
        kwargs = {}
    
    print(f"\n--- Running Test: {name} ---")
    
    try:
        render_kwargs = {'messages': messages, 'add_generation_prompt': True}
        if tools is not None:
            render_kwargs['tools'] = tools
        render_kwargs.update(kwargs)
        
        rendered = template.render(**render_kwargs)
        
        if expect_error:
            print("❌ FAILED: Expected an exception but got none.")
            return False
            
        success = True
        
        if expected_in:
            for ex in expected_in:
                if ex not in rendered:
                    print(f"❌ FAILED: Missing expected string:\n'''{ex}'''")
                    print(f"Rendered:\n{rendered}")
                    success = False
        
        if expected_not_in:
            for n_ex in expected_not_in:
                if n_ex in rendered:
                    print(f"❌ FAILED: Found string that should NOT be present:\n'''{n_ex}'''")
                    print(f"Rendered:\n{rendered}")
                    success = False
                    
        if success:
            print("✅ PASSED")
            return True
        return False
        
    except Exception as e:
        if expect_error:
            print(f"✅ PASSED (Caught expected error: {e})")
            return True
        print(f"❌ FAILED with exception:\n{traceback.format_exc()}")
        return False

# Tests
tests_passed = 0
tests_total = 0

def execute_test(*args, **kwargs):
    global tests_passed, tests_total
    tests_total += 1
    if run_test(*args, **kwargs):
        tests_passed += 1

# ==========================================
# 1. Qwen 3.8 Reasoning Effort Controls
# ==========================================

# 1. Default reasoning_effort="xhigh" (no system message -> synthesized system message)
execute_test(
    "1. reasoning_effort='xhigh' (default, no system message)",
    messages=[{"role": "user", "content": "Hello!"}],
    expected_in=[
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n",
        "<|im_start|>user\nHello!<|im_end|>\n",
        "<|im_start|>assistant\n<think>\n"
    ]
)

# 2. reasoning_effort="high" (OpenAI alias mapped to xhigh)
execute_test(
    "2. reasoning_effort='high' (OpenAI alias)",
    messages=[{"role": "user", "content": "Hello!"}],
    kwargs={"reasoning_effort": "high"},
    expected_in=[
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
    ]
)

# 3. reasoning_effort="low"
execute_test(
    "3. reasoning_effort='low'",
    messages=[{"role": "user", "content": "Hello!"}],
    kwargs={"reasoning_effort": "low"},
    expected_in=[
        "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n"
    ]
)

# 4. reasoning_effort="medium" (No extra system instruction injected)
execute_test(
    "4. reasoning_effort='medium'",
    messages=[{"role": "user", "content": "Hello!"}],
    kwargs={"reasoning_effort": "medium"},
    expected_not_in=["<|im_start|>system"],
    expected_in=[
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ]
)

# 5. reasoning_effort="xhigh" + User System Prompt
execute_test(
    "5. reasoning_effort='xhigh' with user system prompt",
    messages=[
        {"role": "system", "content": "You are a Rust programmer."},
        {"role": "user", "content": "Write a red-black tree."}
    ],
    expected_in=[
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are a Rust programmer.<|im_end|>\n"
    ]
)

# 6. reasoning_effort="xhigh" + Tools (Reasoning instructions before # Tools)
execute_test(
    "6. reasoning_effort='xhigh' with tools",
    messages=[{"role": "user", "content": "Calculate 2+2"}],
    tools=[{"type": "function", "function": {"name": "calc", "parameters": {}}}],
    expected_in=[
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n# Tools\n\nYou have access to the following functions:"
    ]
)

# 7. reasoning_effort="xhigh" + Tools + User System Prompt
execute_test(
    "7. reasoning_effort='xhigh' with tools and user system prompt",
    messages=[
        {"role": "system", "content": "Custom instructions."},
        {"role": "user", "content": "Do it."}
    ],
    tools=[{"type": "function", "function": {"name": "calc", "parameters": {}}}],
    expected_in=[
        "Reasoning effort is set to xhigh.",
        "# Tools",
        "Custom instructions.<|im_end|>"
    ]
)

# ==========================================
# 2. Thinking Control & Pre-Scan Gating
# ==========================================

# 8. reasoning_effort with <|think_off|> in initial user message (string)
execute_test(
    "8. reasoning_effort suppressed when <|think_off|> in user string",
    messages=[{"role": "user", "content": "Fast answer: 2+2 <|think_off|>"}],
    expected_not_in=["<|im_start|>system", "Reasoning effort is set to"],
    expected_in=[
        "<|im_start|>user\nFast answer: 2+2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ]
)

# 9. reasoning_effort with <|think_off|> in multi-part content list
execute_test(
    "9. reasoning_effort suppressed when <|think_off|> in multi-part list",
    messages=[{"role": "user", "content": [{"type": "text", "text": "Fast answer: 2+2 <|think_off|>"}]}],
    expected_not_in=["<|im_start|>system", "Reasoning effort is set to"],
    expected_in=[
        "<|im_start|>user\nFast answer: 2+2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ]
)

# 10. enable_thinking=false kwarg (Disabling thinking does not crash and suppresses instructions)
execute_test(
    "10. enable_thinking=false kwarg",
    messages=[{"role": "user", "content": "Fast 2+2"}],
    kwargs={"enable_thinking": False},
    expected_not_in=["Reasoning effort is set to"],
    expected_in=[
        "<|im_start|>user\nFast 2+2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ]
)

# 11. auto_disable_thinking_with_tools=true kwarg
execute_test(
    "11. auto_disable_thinking_with_tools=true",
    messages=[{"role": "user", "content": "Call tool"}],
    tools=[{"type": "function", "function": {"name": "test"}}],
    kwargs={"auto_disable_thinking_with_tools": True},
    expected_not_in=["Reasoning effort is set to"],
    expected_in=[
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ]
)

# ==========================================
# 3. KV Cache & preserve_reasoning Aliasing
# ==========================================

# 12. preserve_reasoning=True (llama.cpp CLI flag compatibility)
execute_test(
    "12. preserve_reasoning=True preserves thinking",
    messages=[
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Answer 1", "reasoning_content": "Thought 1"},
        {"role": "user", "content": "Turn 2"}
    ],
    kwargs={"preserve_reasoning": True, "reasoning_effort": "medium"},
    expected_in=[
        "<|im_start|>assistant\n<think>\nThought 1\n</think>\n\nAnswer 1<|im_end|>"
    ]
)

# 13. preserve_reasoning=False (strips past thinking)
execute_test(
    "13. preserve_reasoning=False strips past thinking",
    messages=[
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Answer 1", "reasoning_content": "Thought 1"},
        {"role": "user", "content": "Turn 2"}
    ],
    kwargs={"preserve_reasoning": False, "reasoning_effort": "medium"},
    expected_not_in=["Thought 1", "<think>\nThought 1\n</think>"],
    expected_in=[
        "<|im_start|>assistant\nAnswer 1<|im_end|>"
    ]
)

# 14. preserve_thinking=True (backwards compatibility)
execute_test(
    "14. preserve_thinking=True preserves thinking",
    messages=[
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Answer 1", "reasoning_content": "Thought 1"},
        {"role": "user", "content": "Turn 2"}
    ],
    kwargs={"preserve_thinking": True, "reasoning_effort": "medium"},
    expected_in=[
        "<|im_start|>assistant\n<think>\nThought 1\n</think>\n\nAnswer 1<|im_end|>"
    ]
)

# 15. preserve_thinking=False (backwards compatibility)
execute_test(
    "15. preserve_thinking=False strips past thinking",
    messages=[
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Answer 1", "reasoning_content": "Thought 1"},
        {"role": "user", "content": "Turn 2"}
    ],
    kwargs={"preserve_thinking": False, "reasoning_effort": "medium"},
    expected_not_in=["Thought 1", "<think>\nThought 1\n</think>"],
    expected_in=[
        "<|im_start|>assistant\nAnswer 1<|im_end|>"
    ]
)

# ==========================================
# 4. Multi-Turn Thinking History Robustness
# ==========================================

# 16. Assistant history with embedded <think> in content (No empty think duplication)
execute_test(
    "16. In-content <think> parsing (Curing official 3.8 empty think poisoning)",
    messages=[
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>\nThinking deeply.\n</think>\n\nFinal response."},
        {"role": "user", "content": "Next question"}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_not_in=["<think>\n\n</think>\n\n<think>"],
    expected_in=[
        "<|im_start|>assistant\n<think>\nThinking deeply.\n</think>\n\nFinal response.<|im_end|>"
    ]
)

# 17. Assistant history with OpenAI reasoning_content field
execute_test(
    "17. OpenAI reasoning_content field",
    messages=[
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Final response.", "reasoning_content": "Thinking deeply."},
        {"role": "user", "content": "Next question"}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "<|im_start|>assistant\n<think>\nThinking deeply.\n</think>\n\nFinal response.<|im_end|>"
    ]
)

# 18. Assistant history with Anthropic thinking field
execute_test(
    "18. Anthropic message.thinking field",
    messages=[
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Final response.", "thinking": "Thinking deeply."},
        {"role": "user", "content": "Next question"}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "<|im_start|>assistant\n<think>\nThinking deeply.\n</think>\n\nFinal response.<|im_end|>"
    ]
)

# ==========================================
# 5. Universal Tool Calling & Serialization
# ==========================================

# 19. Tool calling with dictionary arguments (XML format)
execute_test(
    "19. Tool calling with dict arguments (XML)",
    messages=[
        {"role": "user", "content": "Search for news"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": {"query": "Qwen 3.8", "limit": 5}}}
        ]}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "<tool_call>\n<function=search>\n<parameter=query>\nQwen 3.8\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>"
    ]
)

# 20. Tool calling with JSON string arguments (XML format - prevents 3.8 crash)
execute_test(
    "20. Tool calling with JSON string arguments (XML)",
    messages=[
        {"role": "user", "content": "Search for news"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": '{"query": "Qwen 3.8"}'}}
        ]}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "<tool_call>\n<function=search>\n{\"query\": \"Qwen 3.8\"}</function>\n</tool_call>"
    ]
)

# 21. Tool calling with dictionary arguments (Hermes JSON format)
execute_test(
    "21. Tool calling with dict arguments (JSON format)",
    messages=[
        {"role": "user", "content": "Search for news"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": {"query": "Qwen 3.8"}}}
        ]}
    ],
    kwargs={"tool_call_format": "json", "reasoning_effort": "medium"},
    expected_in=[
        '<tool_call>\n{"name": "search", "arguments": {"query": "Qwen 3.8"}}\n</tool_call>'
    ]
)

# 22. Tool calling with JSON string arguments (Hermes JSON format)
execute_test(
    "22. Tool calling with JSON string arguments (JSON format)",
    messages=[
        {"role": "user", "content": "Search for news"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "search", "arguments": '{"query": "Qwen 3.8"}'}}
        ]}
    ],
    kwargs={"tool_call_format": "json", "reasoning_effort": "medium"},
    expected_in=[
        '<tool_call>\n{"name": "search", "arguments": {"query": "Qwen 3.8"}}\n</tool_call>'
    ]
)

# 23. Empty string tool arguments ""
execute_test(
    "23. Tool calling with empty arguments string",
    messages=[
        {"role": "user", "content": "Get time"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_time", "arguments": ""}}
        ]}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
    ]
)

# 24. Dynamic parameter truncation (max_tool_arg_chars)
execute_test(
    "24. Dynamic parameter truncation (max_tool_arg_chars)",
    messages=[
        {"role": "user", "content": "Long input"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "process", "arguments": {"text": "A" * 100}}}
        ]}
    ],
    kwargs={"max_tool_arg_chars": 20, "reasoning_effort": "medium"},
    expected_in=[
        "[TRUNCATED — original length 100 chars]"
    ]
)

# 25. Dynamic response truncation (max_tool_response_chars)
execute_test(
    "25. Dynamic response truncation (max_tool_response_chars)",
    messages=[
        {"role": "user", "content": "Run tool"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "run"}}]},
        {"role": "tool", "content": "B" * 200}
    ],
    kwargs={"max_tool_response_chars": 50, "reasoning_effort": "medium"},
    expected_in=[
        "[TRUNCATED — original length 200 chars]"
    ]
)

# 26. Consecutive tool error escalation (1 failure -> Warning 1)
execute_test(
    "26. Consecutive tool error warning 1",
    messages=[
        {"role": "user", "content": "Run tool"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "run"}}]},
        {"role": "tool", "content": "error: file not found"}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "⚠️ SYSTEM WARNING: The previous tool call returned an error. Diagnose the failure and retry with completely corrected arguments."
    ]
)

# 27. Consecutive tool error escalation (2 failures -> Warning 2 with reasoning preserved)
execute_test(
    "27. Consecutive tool error warning 2 (retaining reasoning for error correction)",
    messages=[
        {"role": "user", "content": "Run tool"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "run"}}]},
        {"role": "tool", "content": "error: file not found"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "run"}}]},
        {"role": "tool", "content": "fatal: permission denied"}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "⚠️ SYSTEM WARNING: 2 consecutive tool errors detected. Your previous approach is incorrect. You MUST use a fundamentally different approach or corrected arguments.",
        "<|im_start|>assistant\n<think>\n"
    ]
)

# 28. Mid-conversation system and developer messages
execute_test(
    "28. Mid-conversation system & developer messages",
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "developer", "content": "Be super concise."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    kwargs={"reasoning_effort": "medium"},
    expected_in=[
        "<|im_start|>system\nBe super concise.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>"
    ]
)

# ==========================================
# Summary
# ==========================================
print(f"\n==========================================")
print(f"Results: {tests_passed} / {tests_total} tests passed ({(tests_passed/tests_total)*100:.1f}%)")
print(f"==========================================")

if tests_passed != tests_total:
    sys.exit(1)
