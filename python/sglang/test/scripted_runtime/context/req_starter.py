from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, List, Optional

from sglang.test.scripted_runtime.context.http_post import (
    _http_post_and_await_recv_msg,
)
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


class ScriptedContextReqStarter:
    def __init__(self, ctx: ScriptedContext) -> None:
        self._ctx = ctx
        self._req_counter = 0

    def start_req(
        self,
        *,
        prompt_len: int = 0,
        max_new_tokens: int,
        rid: Optional[str],
        ignore_eos: bool,
        priority: Optional[int],
        dp_rank: Optional[int],
        prompt_token: int = 1,
        prompt_ids: Optional[List[int]] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        lora_path: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> ScriptedReqHandle:
        ctx = self._ctx

        if rid is None:
            rid = f"scripted-{self._req_counter}-{uuid.uuid4().hex}"
            self._req_counter += 1

        # ``prompt_ids`` carries exact token ids (trace replay); otherwise a
        # uniform prompt of ``prompt_len`` repeats ``prompt_token``.
        if prompt_ids is not None:
            input_ids = list(prompt_ids)
        else:
            input_ids = [prompt_token] * prompt_len

        sampling_params = {"max_new_tokens": max_new_tokens, "ignore_eos": ignore_eos}
        if stop_token_ids is not None:
            sampling_params["stop_token_ids"] = stop_token_ids
        if temperature is not None:
            sampling_params["temperature"] = temperature
        payload = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "rid": rid,
            "stream": True,
        }
        payload["return_logprob"] = return_logprob
        if logprob_start_len is not None:
            payload["logprob_start_len"] = logprob_start_len
        if top_logprobs_num is not None:
            payload["top_logprobs_num"] = top_logprobs_num
        if priority is not None:
            payload["priority"] = priority
        if dp_rank is not None:
            payload["routed_dp_rank"] = dp_rank
        if lora_path is not None:
            payload["lora_path"] = lora_path
        await_kwargs = (
            {"timeout_s": timeout_s} if timeout_s is not None else {}
        )
        _http_post_and_await_recv_msg(
            ctx,
            path="/generate",
            json=payload,
            predicate=lambda obj: getattr(obj, "rid", None) == rid,
            description=f"request with rid {rid!r}",
            **await_kwargs,
        )

        return ScriptedReqHandle(rid=rid, context=ctx)
