#!/usr/bin/env python3
"""Reconstruct distinct historical requests, offline, from mini-swe-agent trajectories.

The selected response supplies usage only: its content is never part of its
own prompt. This exports API-compatible messages, not a verified wire capture.
No padding, truncation, tokenization, network access or shell execution occurs.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics


MESSAGE_FIELDS = {
    "role", "content", "name", "tool_calls", "tool_call_id", "function_call",
    "reasoning_content", "refusal", "audio",
}
REQUEST_FIELDS = {
    "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
    "frequency_penalty", "presence_penalty", "seed", "stop", "parallel_tool_calls",
    "tool_choice", "tools", "functions", "function_call", "response_format",
    "chat_template_kwargs", "reasoning_effort", "extra_body",
}
CAVEATS = [
    "Distinct task IDs and exact message deduplication do not establish semantic independence.",
    "Tasks share instructions, repositories and possibly code; disable prefix caching for an independent-context benchmark.",
    "Observed token counts belong to the historical server; no local retokenization or 256k padding was performed.",
    "API messages are reconstructed from trajectory history; original HTTP bodies, tokenizer and server chat template are not captured.",
    "Historical assistant messages remain in context; the selected call's response and all subsequent messages are excluded.",
    "Only the request object is candidate API input; task/source/invocation metadata must never be sent as messages. Select the target model explicitly when replaying.",
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def tool_schema(path):
    """Read the known client's literal schema without importing/executing it."""
    raw = path.read_bytes()
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BASH_TOOL" for target in node.targets
        ):
            tool = ast.literal_eval(node.value)
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("BASH_TOOL must be a function tool definition")
            return [tool], {
                "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                "status": "inferred from supplied client source, not captured in original request",
            }
    raise ValueError(f"No literal BASH_TOOL assignment in {path}")


def prepare_messages(messages):
    """Keep semantic API fields; reject incomplete tool exchanges."""
    output, pending, stripped = [], set(), set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "developer", "user", "assistant", "tool", "function"
        }:
            raise ValueError("Invalid message role or message structure")
        role = message["role"]
        if role != "tool" and pending:
            raise ValueError("Unanswered tool call in historical prompt")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("Tool result has no preceding matching call")
            pending.remove(call_id)
        prepared = {k: v for k, v in message.items() if k in MESSAGE_FIELDS}
        stripped.update(message.keys() - MESSAGE_FIELDS)
        calls = prepared.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        if calls:
            if role != "assistant":
                raise ValueError("Tool calls must belong to assistant messages")
            clean_calls = []
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    raise ValueError("Tool call requires an ID")
                if call["id"] in pending:
                    raise ValueError("Duplicate tool call ID")
                pending.add(call["id"])
                # Streaming response indices are not part of request tool calls.
                clean_calls.append({k: v for k, v in call.items() if k != "index"})
            prepared["tool_calls"] = clean_calls
        if "content" not in prepared and not calls:
            raise ValueError("Message has neither content nor tool calls")
        if prepared.get("content") is not None and not isinstance(prepared["content"], (str, list)):
            raise ValueError("Invalid message content")
        output.append(prepared)
    if pending or not output:
        raise ValueError("Empty prompt or unanswered tool calls")
    return output, sorted(stripped)


def extract(path, inferred_tools=None, tools_provenance=None):
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("trajectory_format") != "mini-swe-agent-1.1":
        raise ValueError("Only mini-swe-agent-1.1 trajectories are supported")
    task = data.get("instance_id")
    if not isinstance(task, str) or not task:
        raise ValueError("Missing instance_id; cannot establish distinct tasks")
    info = data.get("info", {})
    config = info.get("config", {})
    model = config.get("model", {})
    if config.get("model_type") != "minisweagent.models.litellm_model.LitellmModel":
        raise ValueError("Unsupported model adapter; request reconstruction is not known")
    if model.get("set_cache_control") or model.get("multimodal_regex"):
        raise ValueError("Adapter message transforms require an original request capture")
    kwargs = model.get("model_kwargs", {})
    if not isinstance(kwargs, dict) or not isinstance(data.get("messages"), list):
        raise ValueError("Invalid model kwargs or messages")
    candidates = []
    request_index = 0
    for index, response_message in enumerate(data["messages"]):
        if not isinstance(response_message, dict):
            raise ValueError("Invalid message structure")
        extra = response_message.get("extra") or {}
        response = extra.get("response")
        if response is None:
            continue
        current_request = request_index
        request_index += 1
        if response_message.get("role") != "assistant" or not isinstance(response, dict):
            continue
        usage = response.get("usage") or {}
        tokens = usage.get("prompt_tokens")
        if type(tokens) is not int or tokens <= 0 or not response.get("choices") or response.get("error"):
            continue
        candidates.append((tokens, -index, index, current_request, usage, response))
    if not candidates:
        raise ValueError("No successful historical call with original prompt-token usage")
    tokens, _, index, request_index, usage, response = max(candidates)
    messages, stripped = prepare_messages(data["messages"][:index])
    request = {k: v for k, v in kwargs.items() if k in REQUEST_FIELDS}
    request["messages"] = messages
    tool_origin = "saved model_kwargs" if "tools" in request else None
    if "tools" not in request and inferred_tools is not None:
        request["tools"] = inferred_tools
        tool_origin = tools_provenance
    missing = [field for field in ("temperature", "top_p", "top_k", "max_tokens", "seed", "stop", "chat_template_kwargs", "tool_choice") if field not in request]
    if "max_completion_tokens" in request:
        missing = [field for field in missing if field != "max_tokens"]
    if "tools" not in request:
        missing.append("tools")
    return {
        "schema_version": 1, "task_id": task,
        "request": request,
        "request_sha256": digest(request), "messages_sha256": digest(messages),
        "observed_prompt_tokens": tokens,
        "observed_completion_tokens": usage.get("completion_tokens"),
        "source": {
            "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "response_message_index": index, "request_index": request_index,
            "index_base": 0, "mini_version": info.get("mini_version"),
        },
        "invocation_metadata": {
            "configured_model": model.get("model_name"), "response_model": response.get("model"),
            "tool_schema_source": tool_origin, "unknown_original_parameters": missing,
            "unknown_transport_details": ["server defaults", "server chat template", "tokenizer revision", "LiteLLM HTTP serialization"],
            "stripped_message_fields": stripped,
            "non_payload_model_kwarg_names": sorted(kwargs.keys() - REQUEST_FIELDS),
            "agent_templates": {k: v for k, v in config.get("agent", {}).items() if k.endswith("_template")},
            "model_templates": {k: v for k, v in model.items() if k.endswith("_template")},
        },
    }


def build(source, output, summary_path, tools_source=None):
    paths = sorted(source.rglob("*.traj.json"))
    tools, provenance = tool_schema(tools_source) if tools_source else (None, None)
    records, skipped, seen_tasks, seen_messages = [], [], set(), set()
    for path in paths:
        try:
            record = extract(path, tools, provenance)
            if record["task_id"] in seen_tasks:
                raise ValueError("Duplicate task ID")
            if record["messages_sha256"] in seen_messages:
                raise ValueError("Duplicate canonical request messages")
            seen_tasks.add(record["task_id"])
            seen_messages.add(record["messages_sha256"])
            records.append(record)
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            skipped.append({"path": str(path.resolve()), "reason": str(error)})
    lengths = [record["observed_prompt_tokens"] for record in records]
    summary = {
        "schema_version": 1, "source": str(source.resolve()),
        "selection": "Longest recorded prompt per trajectory; earliest response on equal lengths; first sorted source for duplicate task/message history",
        "files_examined": len(paths), "requests_selected": len(records), "skipped": skipped,
        "observed_prompt_tokens": {
            "min": min(lengths, default=None), "median": statistics.median(lengths) if lengths else None,
            "max": max(lengths, default=None), "total": sum(lengths),
            "counts_at_least": {str(n): sum(x >= n for x in lengths) for n in (4096, 16384, 32768, 65536, 131072, 262144)},
        },
        "unique_first_messages": len({digest(r["request"]["messages"][0]) for r in records}),
        "missing_parameter_counts": {key: sum(key in r["invocation_metadata"]["unknown_original_parameters"] for r in records) for key in sorted({k for r in records for k in r["invocation_metadata"]["unknown_original_parameters"]})},
        "caveats": CAVEATS,
    }
    if not records:
        raise ValueError(f"No usable requests; examined {len(paths)} files. First errors: {skipped[:3]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical(record) + "\n" for record in records).encode()
    output.write_bytes(payload)
    summary["output_sha256"] = hashlib.sha256(payload).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Read-only directory containing *.traj.json files")
    parser.add_argument("--output", type=Path, required=True, help="Local output JSONL (contains private task text; do not commit)")
    parser.add_argument("--summary", type=Path, help="Summary JSON; defaults to OUTPUT.summary.json")
    parser.add_argument("--tool-schema-source", type=Path, help="Optional actions_toolcall.py; reads literal BASH_TOOL without executing code")
    args = parser.parse_args()
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    if args.output.resolve() == summary_path.resolve():
        parser.error("Output and summary must be different files")
    if args.source.resolve() in args.output.resolve().parents or args.source.resolve() in summary_path.resolve().parents:
        parser.error("Outputs must be outside the read-only source directory")
    try:
        summary = build(args.source, args.output, summary_path, args.tool_schema_source)
    except (ValueError, OSError, SyntaxError) as error:
        parser.error(str(error))
    print(json.dumps({"requests_selected": summary["requests_selected"], "files_examined": summary["files_examined"], "observed_prompt_tokens": summary["observed_prompt_tokens"], "output": str(args.output.resolve()), "summary": str(summary_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
