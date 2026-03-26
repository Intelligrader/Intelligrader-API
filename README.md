# Intelligrader API Service

FastAPI service for Intelligrader

## Quick Start
Test our API's integrity with the following command:
```bash
curl -I https://peral.one/health
```

## Our API is hosted at: [https://peral.one](https://peral.one)
## You can see the docs at: [https://peral.one/docs](https://peral.one/docs)

## Endpoints
- `POST /generate`: Generates one SAQ (short-answer question) only.
- `POST /grade-saq`: Grades a student response to an SAQ.

Example request body:
```json
{
	"topic": "Photosynthesis",
	"grade_level": "middle school",
	"max_tokens": 96,
	"temperature": 0.7,
	"top_p": 0.9
}
```

Example response body:
```json
{
	"saq_question": "Explain how sunlight helps plants produce glucose during photosynthesis.",
	"model_used": "intelligrader-saq-qwen",
	"tokens_used": 87
}
```

Grade SAQ request body:
```json
{
	"question": "Explain one major cause of World War I.",
	"student_answer": "One cause was nationalism, because countries wanted to prove they were stronger than rivals.",
	"rubric": "4 = clear cause and accurate explanation; 2-3 = partially correct; 0-1 = incorrect or too vague",
	"max_score": 4,
	"temperature": 0.2,
	"top_p": 0.9
}
```

Grade SAQ response body:
```json
{
	"score": 4,
	"max_score": 4,
	"feedback": "Accurately identifies nationalism and explains its role in escalating tensions.",
	"model_used": "intelligrader-saq-qwen",
	"tokens_used": 112
}
```

## Request Queueing
- Generation requests are processed by a single worker queue to prevent concurrent model access races.
- Queue depth is exposed by `/health` and `/models` in `queue_depth`.
- The queue size limit is configurable with `MAX_QUEUE_SIZE` (default: `100`).
- When the queue is full, `/generate` returns HTTP `429`.