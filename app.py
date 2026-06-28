from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import sqlite3
import json
import asyncio
from project import workflow

app = FastAPI(title="QanoonAI", description="Pakistan Law Assistant")

class QueryRequest(BaseModel):
    query: str
    thread_id: str
class ThreadInfo(BaseModel):
    thread_id: str
    first_message: Optional[str] = ""

DB_PATH = "CRAG.db"


def get_threads() -> list[ThreadInfo]:
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT thread_id 
            FROM checkpoints 
            ORDER BY thread_id
        """)
        rows = cursor.fetchall()
        conn.close()
        threads = []
        for row in rows:
            tid = row[0]
            threads.append(ThreadInfo(thread_id=tid, first_message=f"Thread {tid}"))
        return threads
    except Exception as e:
        print(f"DB error: {e}")
        return []


def get_thread_messages(thread_id: str) -> list[dict]:
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT checkpoint 
            FROM checkpoints 
            WHERE thread_id = ? 
            ORDER BY checkpoint_id DESC 
            LIMIT 1
        """, (thread_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return []
        checkpoint = json.loads(row[0])
        messages = checkpoint.get("channel_values", {}).get("messages", [])
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                role = "user" if msg.get("type") == "human" else "assistant"
                content = msg.get("content", "")
                result.append({"role": role, "content": content})
        return result
    except Exception as e:
        print(f"Message fetch error: {e}")
        return []


def extract_answer(result: dict) -> str:
    """Extract answer from workflow result handling both Pydantic and dict."""
    try:
        decision = result['planner_decision']['decision']
        print(f" Decision: {decision}")
        print(f" Generator type: {type(result['generator'])}")
        print(f" Generator value: {result['generator']}")

        if decision == 'chat':
            answer = result['messages'][-1].content
            print(f" Chat answer: {answer[:100]}")
            return answer
        else:
            gen = result['generator']
          
            if hasattr(gen, 'answer'):
                answer = gen.answer
          
            elif isinstance(gen, dict):
                answer = gen.get('answer', 'No answer found')

            else:
                answer = str(gen)
            print(f" RAG answer: {answer[:100]}")
            return answer

    except Exception as e:
        print(f" Extract answer error: {e}")
        return f"Error extracting answer: {str(e)}"

@app.get("/")
def root():
    return {"message": "QanoonAI is running!"}


@app.get("/threads", response_model=list[ThreadInfo])
def list_threads():
    return get_threads()


@app.get("/threads/{thread_id}/messages")
def get_messages(thread_id: str):
    messages = get_thread_messages(thread_id)
    return {"thread_id": thread_id, "messages": messages}


@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """Stream the response token by token."""

    async def event_generator():
        try:
            config = {"configurable": {"thread_id": request.thread_id}}

            print(f"\n{'='*50}")
            print(f" Query: {request.query}")
            print(f" Thread: {request.thread_id}")

            result = workflow.invoke({
                'query'                 : request.query,
                'raw_docs'              : [],
                'document_scores'       : [],
                'router'                : '',
                'planner_decision'      : {},
                'allowed_docs'          : [],
                'ambiguous_docs'        : [],
                'incorrect_docs'        : [],
                'strip_doc'             : [],
                'ambiguous_incorrect'   : [],
                'after_stripping_doc'   : [],
                'web_search_result'     : '',
                'generator'             : {},
                'rewritten_query'       : '',
                'chat_web_search_result': '',
                'chat_rewritten_query'  : ''
                
            }, config=config)
            answer = extract_answer(result)
            for word in answer.split():
                chunk = json.dumps({"token": word + " "})
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.05)

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            print(f" Stream error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
def chat(request: QueryRequest):
    """Non-streaming endpoint (fallback)."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}

        print(f"\n{'='*50}")
        print(f" Query: {request.query}")
        print(f" Thread: {request.thread_id}")

        result = workflow.invoke({
            'query'                 : request.query,
            'raw_docs'              : [],
            'document_scores'       : [],
            'router'                : '',
            'planner_decision'      : {},
            'allowed_docs'          : [],
            'ambiguous_docs'        : [],
            'incorrect_docs'        : [],
            'strip_doc'             : [],
            'ambiguous_incorrect'   : [],
            'after_stripping_doc'   : [],
            'web_search_result'     : '',
            'generator'             : {},
            'rewritten_query'       : '',
            'chat_web_search_result': '',
            'chat_rewritten_query'  : ''
            
        }, config=config)

        answer = extract_answer(result)
        return {"answer": answer, "thread_id": request.thread_id}

    except Exception as e:
        print(f" Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))