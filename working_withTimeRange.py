from fastapi import FastAPI, HTTPException, Depends, Path, Query
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field, ConfigDict
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
    version="1.1.0"
)


# Pydantic Model for Log Entry
class LogEntry(BaseModel):
    LOG_TIME_d: datetime
    POINT_ID: str
    VALUE: Union[str, List[Optional[float]], None] = Field(
        default=None,
        description="Flexible value field that can be a string, list of floats, or None"
    )
    QUALITY: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True
    )


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Advanced value parsing function
def parse_value_safely(raw_value: str) -> Union[List[Optional[float]], str]:
    """
    Safely parse the value field with multiple fallback strategies
    """
    if not raw_value or raw_value.lower() == 'null':
        return []

    try:
        # Strategy 1: Direct AST literal evaluation
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return [
                float(v) if v not in ['NULL', 'null', None, ''] else None
                for v in parsed
            ]
        return raw_value
    except (ValueError, SyntaxError):
        try:
            # Strategy 2: Manual parsing with string manipulation
            cleaned = raw_value.strip('{}').replace('NULL', 'None')
            parsed_list = [
                float(v) if v.lower() != 'none' else None
                for v in cleaned.split(',')
            ]
            return parsed_list
        except Exception:
            # Fallback to original string
            return raw_value


# Function to fetch filtered records with date range
def fetch_filtered_records(
        db: Session,
        point_prefix: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 20
):
    try:
        # Base query with dynamic date filtering
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

        # Prepare query parameters
        params = {
            "point_prefix": point_prefix,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit
        }

        result = db.execute(query, params)

        records = []
        for row in result:
            try:
                parsed_value = parse_value_safely(row[2])

                record = {
                    "LOG_TIME_d": row[0],
                    "POINT_ID": row[1],
                    "VALUE": parsed_value,
                    "QUALITY": row[3]
                }
                records.append(record)
            except Exception as row_error:
                logger.warning(f"Error processing row: {row_error}")
                continue

        logger.info(
            f"Successfully retrieved {len(records)} records for point prefix '{point_prefix}' "
            f"between {start_time} and {end_time}"
        )
        return records

    except SQLAlchemyError as db_error:
        logger.error(f"Database error: {str(db_error)}")
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {str(db_error)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )


# Dynamic API Endpoint for fetching filtered records with date range
@app.get("/records/{point_prefix}", response_model=List[LogEntry])
async def get_records(
        point_prefix: str = Path(..., description="Prefix of the Point ID (e.g., P1, P2, P3)"),
        start_time: Optional[datetime] = Query(
            None,
            description="Start time for filtering records (ISO 8601 format)"
        ),
        end_time: Optional[datetime] = Query(
            None,
            description="End time for filtering records (ISO 8601 format)"
        ),
        limit: int = Query(
            20,
            ge=1,
            le=100,
            description="Maximum number of records to retrieve (1-100)"
        ),
        db: Session = Depends(get_db)
):
    """
    Retrieve log records with flexible filtering

    - **point_prefix**: Prefix to filter point IDs
    - **start_time**: Optional start time for date range filter
    - **end_time**: Optional end time for date range filter
    - **limit**: Maximum number of records to retrieve (default 20, max 100)

    Examples:
    - Fetch P2 records between specific times:
      `/records/P2?start_time=2024-11-16T00:01:00&end_time=2024-11-16T00:03:00`
    - Fetch latest 20 P2 records:
      `/records/P2`
    - Fetch records after a specific time:
      `/records/P2?start_time=2024-11-16T00:01:00`
    """
    records = fetch_filtered_records(
        db,
        point_prefix,
        start_time,
        end_time,
        limit
    )
    return records


# Configuration for running the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )