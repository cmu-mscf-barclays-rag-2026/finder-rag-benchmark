"""Framework adapters for the fixed Meeting 10 Legal comparison.

This module never handles an upstream credential. Both frameworks receive the
coordinator's local OpenAI-compatible budget proxy URL and an inert dummy key.
No API call occurs at import or construction time.

MultiQuery interface::

    multi = build_multiquery(proxy_url, search, depth=20)
    result = multi.invoke(question)

``search(query, depth)`` returns an ordered iterable of passage IDs, ``(id,
text)`` pairs, mappings with ``passage_id``/``id`` and ``text``/``page_content``,
or LangChain Documents carrying ``metadata['passage_id']``. The adapter records
the exact queries accepted by the native default line parser. RRF is deliberately
left to the coordinator; the native document union is returned separately.

DSPy interface::

    configure_dspy(proxy_url)
    control = unoptimized_program()
    optimized = optimize(train_rows, val_rows, metric, output_dir)
    response = predict(optimized, context, question)

Train/validation dictionaries need ``context``, ``question`` and reference
``answer``. Reference answers are labels, never inference inputs. Split isolation,
the fixed BM25 evidence budget and scoring are the coordinator's responsibility.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import threading
from typing import Any


MODEL = "gpt-4o-mini"
DUMMY_KEY = "local-budget-proxy"


def _versions(*names: str) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document_parts(item: Any) -> tuple[str, str]:
    if isinstance(item, str) or isinstance(item, int):
        return str(item), str(item)
    if isinstance(item, Mapping):
        identifier = item.get("passage_id", item.get("id"))
        if identifier is None:
            raise ValueError("BM25 record needs passage_id or id")
        return str(identifier), str(item.get("text", item.get("page_content", identifier)))
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), str(item[1])
    if hasattr(item, "page_content") and hasattr(item, "metadata"):
        identifier = item.metadata.get("passage_id", item.metadata.get("id"))
        if identifier is None:
            raise ValueError("BM25 Document needs metadata passage_id or id")
        return str(identifier), str(item.page_content)
    raise TypeError("Unsupported BM25 result; return IDs, (id,text), mappings or Documents")


class RecordedMultiQuery:
    """Serialize invocations while preserving native MQR query/union behavior."""

    def __init__(self, native: Any, adapter: Any, depth: int) -> None:
        self.native = native
        self.adapter = adapter
        self.depth = depth
        self._lock = threading.RLock()

    def invoke(self, question: str) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        with self._lock:
            self.adapter.clear_records()
            documents = self.native.invoke(question)
            records = self.adapter.snapshot_records()
            # include_original=True appends the original question after the
            # native parser's actual generated lines, even if it is duplicated.
            if not records or records[-1]["query"] != question:
                raise AssertionError("Native MultiQueryRetriever omitted original query")
            union = [str(doc.metadata["passage_id"]) for doc in documents]
            if len(union) != len(set(union)):
                raise AssertionError("Same passage returned with inconsistent document content")
            return {
                "question": question,
                "queries": [row["query"] for row in records],
                "generated_queries": [row["query"] for row in records[:-1]],
                "query_count": len(records),
                "generated_query_count": len(records) - 1,
                "per_query_rankings": records,
                "native_union_ids": union,
                "native_union_size": len(union),
                "per_query_depth": self.depth,
                "include_original": True,
                "default_prompt_and_parser": True,
                "versions": _versions("langchain-classic", "langchain-core", "langchain-openai"),
            }

    def invoke_multiquery(self, question: str) -> dict[str, Any]:
        return self.invoke(question)


def build_multiquery(
    base_url: str,
    bm25_search_callable: Callable[[str, int], Iterable[Any]],
    depth: int = 20,
) -> RecordedMultiQuery:
    """Build actual MultiQueryRetriever with its unchanged default prompt/parser.

    The supplied BM25 function owns tokenization, scoring, tie-breaking and the
    corpus. Empty/duplicate IDs in one ranking are errors, not silently repaired.
    Generated query lines, numbering and whitespace are passed through unchanged.
    """
    if depth < 1:
        raise ValueError("depth must be positive")
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_openai import ChatOpenAI
    from pydantic import Field

    class RecordingBM25(BaseRetriever):
        search_fn: Any = Field(exclude=True)
        depth: int
        records: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
        record_lock: Any = Field(default_factory=threading.RLock, exclude=True)

        def clear_records(self) -> None:
            with self.record_lock:
                self.records.clear()

        def snapshot_records(self) -> list[dict[str, Any]]:
            with self.record_lock:
                return [{"query": row["query"], "passage_ids": list(row["passage_ids"])}
                        for row in self.records]

        def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Any]:
            items = list(self.search_fn(query, self.depth))
            if len(items) > self.depth:
                raise AssertionError("BM25 search returned more than the requested depth")
            pairs = [_document_parts(item) for item in items]
            identifiers = [identifier for identifier, _ in pairs]
            if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
                raise AssertionError("BM25 ranking contains an empty or duplicate passage ID")
            with self.record_lock:
                self.records.append({"query": query, "passage_ids": identifiers})
            return [Document(page_content=text, metadata={"passage_id": identifier})
                    for identifier, text in pairs]

    adapter = RecordingBM25(search_fn=bm25_search_callable, depth=depth)
    llm = ChatOpenAI(
        model=MODEL,
        base_url=base_url.rstrip("/"),
        api_key=DUMMY_KEY,
        temperature=0,
        max_tokens=256,
        max_retries=0,
        timeout=120,
    )
    native = MultiQueryRetriever.from_llm(
        retriever=adapter,
        llm=llm,
        include_original=True,
    )
    return RecordedMultiQuery(native, adapter, depth)


def invoke_multiquery(retriever: RecordedMultiQuery, question: str) -> dict[str, Any]:
    """Functional alias for ``build_multiquery(...).invoke(question)``."""
    return retriever.invoke(question)


def configure_dspy(base_url: str) -> Any:
    """Configure DSPy against the local durable budget proxy, without a call.

    DSPy response caching is disabled because the proxy provides the authoritative
    cross-process cache and cost ledger. Framework parser recovery, if triggered,
    also passes through that proxy and is counted there. Transport retries are 0.
    """
    import dspy

    lm = dspy.LM(
        "openai/" + MODEL,
        model_type="chat",
        api_base=base_url.rstrip("/"),
        api_key=DUMMY_KEY,
        temperature=0,
        max_tokens=512,
        num_retries=0,
        cache=False,
        timeout=120,
    )
    dspy.configure(lm=lm)
    return lm


def unoptimized_program() -> Any:
    """Return the same CoT architecture used by the optimized reader control."""
    import dspy

    class LegalAnswer(dspy.Signature):
        """Answer the legal question accurately and concisely using the supplied context.

        Preserve qualifications and exceptions. If the context does not support
        an answer, state that the supplied evidence is insufficient. Do not
        invent case names, statutory provisions or factual details.
        """

        context: str = dspy.InputField(desc="Retrieved legal passages, in fixed ranking order.")
        question: str = dspy.InputField(desc="The legal question to answer.")
        answer: str = dspy.OutputField(desc="A concise answer supported by the supplied evidence.")

    return dspy.ChainOfThought(LegalAnswer, rationale_field_type=str)


def _examples(rows: Iterable[Any]) -> list[Any]:
    import dspy

    examples = []
    for row in rows:
        if isinstance(row, Mapping):
            values = dict(row)
        elif hasattr(row, "toDict"):
            values = row.toDict()
        else:
            raise TypeError("Optimizer data must contain mappings or dspy.Example objects")
        for key in ["context", "question", "answer"]:
            if key not in values or not isinstance(values[key], str):
                raise ValueError(f"Optimizer example needs string {key}")
        examples.append(dspy.Example(**values).with_inputs("context", "question"))
    if not examples:
        raise ValueError("Optimizer data must not be empty")
    return examples


def _signature_record(program: Any) -> list[dict[str, Any]]:
    return [
        {
            "predictor": name,
            "instructions": predictor.signature.instructions,
            "input_fields": list(predictor.signature.input_fields),
            "output_fields": list(predictor.signature.output_fields),
            "demonstration_count": len(predictor.demos),
        }
        for name, predictor in program.named_predictors()
    ]


def metadata() -> dict[str, Any]:
    """Return exact default MQR prompt and initial DSPy signature, without calls."""
    from langchain_classic.retrievers.multi_query import DEFAULT_QUERY_PROMPT

    return {
        "versions": _versions("langchain-classic", "langchain-core", "langchain-openai", "dspy"),
        "multiquery": {
            "model": MODEL,
            "temperature": 0,
            "max_output_tokens": 256,
            "max_retries": 0,
            "default_prompt_template": DEFAULT_QUERY_PROMPT.template,
            "default_line_parser": "strip outer whitespace; split on newline; remove empty lines only",
            "requested_rewrites": 3,
            "include_original": True,
            "native_output": "unique Document union; RRF performed separately by coordinator",
        },
        "dspy": {
            "model": MODEL,
            "api_family": "chat",
            "temperature": 0,
            "max_output_tokens": 512,
            "num_retries": 0,
            "dspy_response_cache": False,
            "adapter": "DSPy 3.5 default XMLAdapter; native JSON fallback if output parsing fails",
            "authoritative_cache": "coordinator local budget proxy",
            "initial_signature": _signature_record(unoptimized_program()),
            "optimizer": "MIPROv2",
            "auto": "light",
            "max_bootstrapped_demos": 0,
            "max_labeled_demos": 0,
            "seed": 42,
            "num_threads": 2,
            "max_errors": 1,
        },
    }


def framework_metadata() -> dict[str, Any]:
    """Coordinator-compatible metadata keys plus the detailed framework record."""
    result = metadata()
    return {
        "multiquery_default_prompt": result["multiquery"]["default_prompt_template"],
        "dspy_signature": result["dspy"]["initial_signature"],
        **result,
    }


def load_optimized(outputdir: str | Path) -> Any:
    """Load a complete, hash-checked zero-demo program onto the current proxy LM.

    Rebinding the LM is necessary because a restarted local proxy may have a new
    port. The compiled instructions and demonstrations are checked after loading.
    This function makes no model request and never invokes the optimizer.
    """
    import dspy

    output = Path(outputdir)
    status = json.loads((output / "optimization_status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("Cannot load an incomplete or failed optimization")
    for filename, digest in status["completed_artifact_sha256"].items():
        if _sha(output / filename) != digest:
            raise AssertionError(f"Optimized artifact changed: {filename}")
    program = unoptimized_program()
    program.load(str(output / "compiled_program.json"))
    program.set_lm(dspy.settings.lm)
    signature = _signature_record(program)
    expected = json.loads((output / "compiled_signature.json").read_text())
    if signature != expected or any(row["demonstration_count"] for row in signature):
        raise AssertionError("Loaded optimized signature differs or includes demonstrations")
    return program


def optimize(
    trainlist: Iterable[Any],
    vallist: Iterable[Any],
    metric: Callable[..., float],
    outputdir: str | Path,
) -> Any:
    """Run native instruction-only MIPROv2 light on development examples only.

    The caller freezes and validates splits before calling this function. Its
    metric receives the reference label; model inference receives only context
    and question. Native optimizer logs retain candidate/trial checkpoints.
    Exceptions are saved and re-raised; no fallback result is called optimized.
    """
    import dspy

    train, val = _examples(trainlist), _examples(vallist)
    train_questions = {ex.question for ex in train}
    if train_questions & {ex.question for ex in val}:
        raise ValueError("Training and optimizer-validation questions overlap")
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "optimizer_logs").mkdir(exist_ok=True)
    try:
        metric_source = inspect.getsource(metric)
    except (OSError, TypeError):
        metric_source = f"{getattr(metric, '__module__', '')}.{getattr(metric, '__qualname__', repr(metric))}"
    input_state = {
        "train": [example.toDict() for example in train],
        "validation": [example.toDict() for example in val],
        "metric_source": metric_source,
        "method_metadata": metadata(),
    }
    input_sha = hashlib.sha256(json.dumps(input_state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    input_path = output / "optimization_inputs.json"
    if input_path.exists():
        previous = json.loads(input_path.read_text())
        if previous["input_sha256"] != input_sha:
            raise ValueError("Optimizer train/validation data, metric or method settings changed")
    else:
        _write_json(input_path, {"input_sha256": input_sha, **input_state})
    status_path = output / "optimization_status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if existing.get("status") == "complete":
            if existing.get("input_sha256") != input_sha:
                raise ValueError("Complete optimizer run does not match current frozen inputs")
            return load_optimized(output)
    student = unoptimized_program()
    student.save(str(output / "unoptimized_program.json"))
    _write_json(output / "unoptimized_signature.json", _signature_record(student))
    status = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": "dspy.MIPROv2",
        "auto": "light",
        "seed": 42,
        "num_threads": 2,
        "max_bootstrapped_demos": 0,
        "max_labeled_demos": 0,
        "training_examples": len(train),
        "optimizer_validation_examples": len(val),
        "input_sha256": input_sha,
        "inference_inputs": ["context", "question"],
        "metric": "Caller-supplied frozen development metric; not a test-set score",
        "versions": _versions("dspy"),
    }
    _write_json(output / "optimization_status.json", status)
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto="light",
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_threads=2,
        max_errors=1,
        seed=42,
        log_dir=str(output / "optimizer_logs"),
        verbose=False,
    )
    try:
        compiled = optimizer.compile(student, trainset=train, valset=val)
        signature = _signature_record(compiled)
        if any(record["demonstration_count"] for record in signature):
            raise AssertionError("Instruction-only optimization unexpectedly retained demonstrations")
        compiled.save(str(output / "compiled_program.json"))
        _write_json(output / "compiled_signature.json", signature)
        status.update(
            status="complete",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            completed_artifact_sha256={name: _sha(output / name)
                for name in ["compiled_program.json", "compiled_signature.json", "optimization_inputs.json"]},
        )
        _write_json(output / "optimization_status.json", status)
        return compiled
    except Exception as error:
        status.update(
            status="failed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__,
            error=str(error),
        )
        _write_json(output / "optimization_status.json", status)
        raise


def predict(program: Any, context: str, question: str) -> dict[str, Any]:
    """Execute one reader call with no reference answer supplied to the model."""
    if not isinstance(context, str) or not isinstance(question, str):
        raise TypeError("context and question must be strings")
    result = program(context=context, question=question)
    answer = getattr(result, "answer", None)
    if not isinstance(answer, str):
        raise TypeError("DSPy did not produce a string answer")
    reasoning = getattr(result, "reasoning", None)
    return {
        "answer": answer,
        "reasoning": str(reasoning) if reasoning is not None else None,
        "parse_success": True,
        "empty_answer": not bool(answer.strip()),
    }
