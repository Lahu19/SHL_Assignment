from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from query_functions import query_handling
import uvicorn
import os
import gc


app = FastAPI(
    title="SHL Assessment Recommendation System API",
    description="API for recommending SHL assessments based on job descriptions or queries",
    version="1.0.0"
)

class RecommendationRequest(BaseModel):
    query: str

class AssessmentResponse(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: RecommendationRequest):
    """Recommend assessments based on query"""
    try:
        # Get recommendations using existing query handling
        df = query_handling(request.query)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No assessments found matching the criteria")
        
        # Convert DataFrame to required response format
        recommendations = []
        for _, row in df.head(10).iterrows():  # Limit to 10 recommendations
            recommendation = AssessmentResponse(
                url=row.get('Relative URL', ''),
                adaptive_support=row.get('Adaptive/IRT', 'No'),
                description=row.get('Description', ''),
                duration=int(row.get('Duration in mins', 0)),
                remote_support=row.get('Remote Testing', 'No'),
                test_type=[row.get('Test Type', '')] if pd.notna(row.get('Test Type')) else []
            )
            recommendations.append(recommendation)
        
        # Clear memory
        del df
        gc.collect()
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No assessments found matching the criteria")
        
        return RecommendationResponse(recommended_assessments=recommendations)
    
    except Exception as e:
        # Clear memory in case of error
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)  # Use single worker to save memory 