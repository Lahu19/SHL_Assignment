# SHL Assessment Recommendation System

A comprehensive system for recommending SHL assessments based on job descriptions or natural language queries.

## Features

- Web-based UI for easy interaction
- REST API for programmatic access
- Natural language processing for query understanding
- Support for job description input
- Comprehensive evaluation metrics
- Web scraping for up-to-date SHL catalog data

## System Architecture

The system consists of several components:

1. **Web UI (Streamlit)**
   - User-friendly interface
   - Search functionality
   - Filtering options
   - Detailed assessment information

2. **REST API (FastAPI)**
   - Health check endpoint
   - Recommendation endpoint
   - JSON request/response format
   - Proper error handling

3. **Recommendation Engine**
   - Natural language processing
   - Semantic search capabilities
   - Support for job descriptions
   - Maximum 10 recommendations

4. **Data Management**
   - Web scraping for SHL catalog
   - Data cleaning and processing
   - Regular updates

5. **Evaluation System**
   - Mean Recall@3 metric
   - MAP@3 metric
   - Test dataset support
   - Performance tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lahu19/SHL_Assignment
cd SHL_Assignment
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web UI
```bash
streamlit run app.py
```

### API Server
```bash
python api.py
```

### Web Scraper
```bash
python scraper.py
```

### Evaluation
```bash
python evaluation.py
```

## API Documentation

### Health Check
- **Endpoint**: GET `/health`
- **Response**: `{"status": "healthy"}`

### Recommendation
- **Endpoint**: POST `/recommend`
- **Request Body**:
```json
{
  "query": "string"
}
```
- **Response**:
```json
{
  "recommended_assessments": [
    {
      "url": "string",
      "adaptive_support": "Yes/No",
      "description": "string",
      "duration": 0,
      "remote_support": "Yes/No",
      "test_type": ["string"]
    }
  ]
}
```

## Evaluation Metrics

The system is evaluated using two main metrics:

1. **Mean Recall@3**
   - Measures the proportion of relevant items found in the top 3 recommendations
   - Higher values indicate better performance

2. **MAP@3 (Mean Average Precision@3)**
   - Measures the precision of recommendations at different positions
   - Takes into account the order of recommendations
   - Higher values indicate better performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Feel free to contact me for any thing regarding this repo Happy Coding...