from fastapi import FastAPI, HTTPException, Depends, Request, Query, Path
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import logging
from datetime import datetime
import ast

# Database Configuration
DATABASE_URL = "postgresql://postgres:PSc19188@localhost/mydb"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Pilot Project Log API",
    description="API for retrieving log entries with flexible date range filtering",
    version="1.3.0"
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Function to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to fetch filtered records with date range
def fetch_filtered_records(
    db: Session,
    point_prefix: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100  # Changed default limit from 20 to 100
):
    try:
        query = text("""
            SELECT 
                "LOG_TIME_d",
                "POINT_ID",
                "VALUE",
                "QUALITY"
            FROM "PILOT_PROJECT_LOG"
            WHERE SPLIT_PART("POINT_ID", '_', 1) = :point_prefix
            AND (:start_time IS NULL OR "LOG_TIME_d" >= :start_time)
            AND (:end_time IS NULL OR "LOG_TIME_d" <= :end_time)
            ORDER BY "LOG_TIME_d" DESC
            LIMIT :limit
        """)

        params = {
            "point_prefix": point_prefix,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit
        }

        result = db.execute(query, params)

        records = []
        for row in result:
            record = {
                "LOG_TIME_d": row[0].isoformat(),
                "POINT_ID": row[1],
                "VALUE": row[2],
                "QUALITY": row[3]
            }
            records.append(record)

        logger.info(f"Successfully retrieved {len(records)} records for point prefix '{point_prefix}'")
        return records

    except SQLAlchemyError as db_error:
        logger.error(f"Database error: {str(db_error)}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(db_error)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
# Route to display form for records
@app.get("/records", response_class=HTMLResponse)
async def records_form(request: Request):
    return templates.TemplateResponse("records_form.html", {"request": request})

# Route to fetch records with optional parameters
@app.get("/records/{point_prefix}", response_class=JSONResponse)
async def get_records(
    request: Request,
    point_prefix: str = Path(..., description="Point prefix to filter records"),
    start_time: Optional[str] = Query(None, description="Start time for filtering records"),
    end_time: Optional[str] = Query(None, description="End time for filtering records"),
    db: Session = Depends(get_db)
):
    # Convert string dates to datetime objects
    start_time_dt = datetime.fromisoformat(start_time) if start_time else None
    end_time_dt = datetime.fromisoformat(end_time) if end_time else None

    # Fetch records
    records = fetch_filtered_records(
        db,
        point_prefix,
        start_time_dt,
        end_time_dt
    )

    # Return JSON response
    return {
        "point_prefix": point_prefix,
        "start_time": start_time,
        "end_time": end_time,
        "records": records
 }

# Ensure the application runs correctly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)