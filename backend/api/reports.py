"""Reports API endpoints"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Dict, Any, List
from datetime import datetime

from backend.models.requests import ReportRequest
from backend.models.responses import (
    ReportResponse,
    ReportListResponse,
    ErrorResponse
)
from backend.services.report_service import report_service


router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/generate", status_code=202)
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks
) -> ReportResponse:
    """
    Generate a report from simulation results
    
    Args:
        request: Report request with simulation_id, report_type, sections, title
        background_tasks: FastAPI background tasks
        
    Returns:
        ReportResponse: Report metadata with ID and file path
        
    Raises:
        HTTPException 404: If simulation results not found
        HTTPException 400: If report type is invalid
        HTTPException 500: If report generation fails
    """
    try:
        # Generate report (synchronous for now, could be made async)
        metadata = await report_service.generate_report(
            simulation_id=request.simulation_id,
            report_type=request.report_type,
            include_sections=request.include_sections,
            title=request.title
        )
        
        return ReportResponse(
            report_id=metadata['report_id'],
            simulation_id=metadata['simulation_id'],
            report_type=metadata['report_type'],
            title=metadata['title'],
            generated_at=metadata['generated_at'],
            file_path=metadata['file_path'],
            file_size_kb=metadata['file_size_kb'],
            sections=metadata['sections'],
            status="completed",
            message=f"Report generated successfully: {metadata['report_id']}"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ValidationError",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    except ImportError as e:
        # WeasyPrint not installed - HTML fallback was generated
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DependencyError",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ReportGenerationError",
                "message": f"Failed to generate report: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/list", response_model=ReportListResponse)
async def list_reports() -> ReportListResponse:
    """
    List all generated reports
    
    Returns:
        ReportListResponse: List of report metadata
        
    Raises:
        HTTPException 500: If error listing reports
    """
    try:
        reports = report_service.list_reports()
        
        return ReportListResponse(
            reports=reports,
            total=len(reports)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ListError",
                "message": f"Failed to list reports: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/{report_id}")
async def download_report(report_id: str) -> FileResponse:
    """
    Download a generated report file
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        FileResponse: Report file for download
        
    Raises:
        HTTPException 404: If report not found
        HTTPException 500: If error retrieving report
    """
    try:
        # Get report file path
        file_path = report_service.get_report_file_path(report_id)
        
        # Get report metadata for filename
        metadata = report_service.get_report(report_id)
        
        # Determine media type
        media_type = "text/html" if file_path.suffix == ".html" else "application/pdf"
        
        # Create download filename
        filename = f"{report_id}{file_path.suffix}"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RetrievalError",
                "message": f"Failed to retrieve report: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/{report_id}/metadata")
async def get_report_metadata(report_id: str) -> Dict[str, Any]:
    """
    Get metadata for a specific report
    
    Args:
        report_id: Unique report identifier
        
    Returns:
        dict: Report metadata
        
    Raises:
        HTTPException 404: If report not found
        HTTPException 500: If error retrieving metadata
    """
    try:
        metadata = report_service.get_report(report_id)
        return metadata
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RetrievalError",
                "message": f"Failed to retrieve report metadata: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )
