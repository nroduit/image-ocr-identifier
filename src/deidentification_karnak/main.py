import logging
import os

from fastapi import FastAPI
import uvicorn

from deidentification_karnak.routers.deidentify_image import router

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    force=True,
)

app = FastAPI(
    title="Deidentification API",
    description="API for deidentifying images using OCR",
    version="0.0.1",
)
app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "deidentification_karnak.main:app",
        host="0.0.0.0",
        port=8000,
        h11_max_incomplete_event_size=50 * 1024 * 1024,
        workers=int(os.environ.get("WORKERS", os.cpu_count() or 1)),
    )
