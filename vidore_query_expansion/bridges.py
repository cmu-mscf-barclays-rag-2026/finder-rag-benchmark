"""Use framework interfaces with the shared checkpointed API transport."""

import dspy
from dspy.lm15 import Message, Response, TextPart, Usage
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda


class QueryGenerationOnlyRetriever(BaseRetriever):
    """Candidate retrieval is performed by the benchmark's ranked search adapter."""

    def _get_relevant_documents(self, query, *, run_manager):
        raise RuntimeError("Use the shared ranked search adapter after generate_queries().")


def multiquery(api, question, instruction):
    def generate(values):
        response, _ = api.complete([
            {"role": "system", "content": instruction},
            {"role": "user", "content": values["question"]},
        ])
        return response["choices"][0]["message"]["content"].strip().splitlines()

    retriever = MultiQueryRetriever(
        retriever=QueryGenerationOnlyRetriever(), llm_chain=RunnableLambda(generate),
        include_original=True, verbose=False,
    )
    return retriever.generate_queries(question, CallbackManagerForRetrieverRun.get_noop_manager())


class LoggedEngine:
    supports_response_schema = False
    supports_reasoning = False
    supports_function_calling = False
    supported_params = {"temperature", "max_tokens"}

    def __init__(self, api):
        self.api = api

    def complete(self, request):
        def text(parts):
            if any(not isinstance(part, TextPart) for part in parts):
                raise ValueError("This experiment sends text-only query-generation inputs.")
            return "".join(part.text for part in parts)

        messages = []
        if request.system is not None:
            system = request.system if isinstance(request.system, str) else text(request.system)
            messages.append({"role": "system", "content": system})
        messages.extend({"role": message.role, "content": text(message.parts)}
                        for message in request.messages)
        result, saved = self.api.complete(
            messages, temperature=request.config.temperature or 0,
            max_tokens=request.config.max_tokens or 1024,
        )
        usage = result["usage"]
        return Response(
            id=result["id"], model=result["model"],
            message=Message.assistant(result["choices"][0]["message"]["content"]),
            finish_reason=result["choices"][0]["finish_reason"],
            usage=Usage(input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                        total_tokens=usage["total_tokens"]),
            provider_data={"estimated_usd": saved["estimated_usd"]},
        )


class DSPyLM(dspy.BaseLM):
    def __init__(self, api):
        super().__init__(
            model=api.model, temperature=0, max_tokens=1024,
            cache=False, num_retries=0, engine=LoggedEngine(api),
        )
