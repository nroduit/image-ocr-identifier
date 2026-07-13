import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from image_ocr_identifier.routers.deidentify_image import router as deidentify_router
from image_ocr_identifier.routers.reporting import router as reporting_router

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image OCR Identifier",
    description="API for deidentifying images using OCR",
    version="0.0.1",
)
app.include_router(deidentify_router)
app.include_router(reporting_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled error processing %s %s", request.method, request.url.path
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "image_ocr_identifier.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        h11_max_incomplete_event_size=50 * 1024 * 1024,
        workers=int(os.environ.get("WORKERS", os.cpu_count() or 1)),
    )
