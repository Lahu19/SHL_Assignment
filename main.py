# Import required libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from query_functions import query_handling

# Initialize FastAPI app
app = FastAPI(title="Assessment Recommendation System")

# Define request model
class QueryRequest(BaseModel):
    """Request model for recommendations."""
    query: str
    num_results: int = 5

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Service is running"}

# Main recommendation endpoint
@app.post("/recommendations/")
async def get_recommendations(request: QueryRequest):
    """Get assessment recommendations based on query."""
    try:
        # Get recommendations using the query_handling function
        results_df = query_handling(request.query)
        
        if results_df.empty:
            return {"recommendations": [], "message": "No matching assessments found"}
        
        # Convert DataFrame to list of dictionaries
        recommendations = results_df.to_dict('records')
        
        # Limit results if needed
        recommendations = recommendations[:request.num_results]
        
        return {
            "recommendations": recommendations,
            "message": f"Found {len(recommendations)} matching assessments"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
