from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import gc
import logging
import os
from queue import Full, Queue
from threading import Thread

from fastapi import HTTPException
from llama_cpp import Llama

import model_manager
from schemas import GenerateRequest


logger = logging.getLogger(__name__)


@dataclass
class GenerationJob:
    request: GenerateRequest
    future: Future


class GenerationEngine:
    def __init__(self, max_queue_size: int | None = None):
        self.max_queue_size = max_queue_size or int(os.getenv("MAX_QUEUE_SIZE", "100"))
        self._llm = None
        self.loaded_model_id: str | None = None
        self._queue: Queue[GenerationJob | None] = Queue(maxsize=self.max_queue_size)
        self._worker: Thread | None = None

    def startup(self) -> None:
        model_id = model_manager.get_default_model_id()
        try:
            self._load_model(model_id)
            logger.info("Default model loaded: %s", model_id)
        except Exception:
            self._llm = None
            self.loaded_model_id = None
            logger.exception("Default model failed to load during startup")

        self._worker = Thread(target=self._worker_loop, name="generation-worker", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=10)

    def model_status(self) -> str:
        return "loaded" if self._llm is not None else "not loaded"

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def worker_available(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def generate(self, request: GenerateRequest) -> dict:
        if not self.worker_available():
            raise HTTPException(status_code=503, detail="Generation worker unavailable")

        future: Future = Future()
        queue_depth_before = self._queue.qsize()

        try:
            self._queue.put_nowait(GenerationJob(request=request, future=future))
        except Full:
            raise HTTPException(
                status_code=429,
                detail=f"Request queue is full. Try again later (max queue size: {self.max_queue_size}).",
            )

        logger.info("Generation request queued (queue depth before enqueue: %s)", queue_depth_before)

        try:
            return future.result()
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation error: {str(exc)}")

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break

            try:
                result = self._generate_text_internal(job.request)
                job.future.set_result(result)
            except HTTPException as exc:
                job.future.set_exception(exc)
            except Exception as exc:
                logger.exception("Generation worker failed")
                job.future.set_exception(
                    HTTPException(status_code=500, detail=f"Generation error: {str(exc)}")
                )
            finally:
                self._queue.task_done()

    def _generate_text_internal(self, request: GenerateRequest) -> dict:
        if self._llm is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        requested_model_id = request.model or self.loaded_model_id

        if not model_manager.is_supported(requested_model_id):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: '{requested_model_id}'. See /models for supported options.",
            )

        if self.loaded_model_id != requested_model_id:
            self._unload_model()
            self._load_model(requested_model_id)
            logger.info("Model switched to: %s", requested_model_id)

        output = self._llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False,
        )

        return {
            "prompt": request.prompt,
            "generated_text": output["choices"][0]["text"],
            "tokens_used": output["usage"]["total_tokens"],
            "model_used": self.loaded_model_id,
        }

    def _load_model(self, model_id: str) -> None:
        model_path = model_manager.get_model_path(model_id)
        config = model_manager.get_model_config(model_id)
        self._llm = Llama(model_path=str(model_path), **config["runtime"])
        self.loaded_model_id = model_id

    def _unload_model(self) -> None:
        if self._llm is None:
            return
        old_llm = self._llm
        self._llm = None
        del old_llm
        gc.collect()