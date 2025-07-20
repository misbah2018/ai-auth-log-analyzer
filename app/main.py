from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Auth Log Analyzer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthLogAnalyzer:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models and encoders"""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                logger.warning("Models directory not found. Using dummy models.")
                return
            
            # Load models
            with open(os.path.join(models_dir, "is_failed_login_model.pkl"), "rb") as f:
                self.models["failed_login"] = pickle.load(f)
            
            with open(os.path.join(models_dir, "is_intrusion_model.pkl"), "rb") as f:
                self.models["intrusion"] = pickle.load(f)
            
            # Load encoders
            with open(os.path.join(models_dir, "encoder.pkl"), "rb") as f:
                self.encoders = pickle.load(f)
                
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a single auth log line"""
        # Updated pattern to match the actual log format
        pattern = r'(?P<timestamp>\w{3} \d+ \d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<service>\S+): (?P<message>.+)'
        match = re.match(pattern, line.strip())
        if not match:
            return None

        message = match.group("message")
        
        # Extract user
        user_match = re.search(r'(?:for|user) (?:invalid user )?(\w+)', message)
        user = user_match.group(1) if user_match else "unknown"
        
        # Extract IP address
        ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', message)
        ip_address = ip_match.group(1) if ip_match else "0.0.0.0"
        
        # Determine success/failure
        success = 'Accepted' in message
        
        return {
            "raw_line": line.strip(),
            "timestamp": match.group("timestamp"),
            "user": user,
            "ip_address": ip_address,
            "success": success,
            "event_type": "login_attempt"
        }
    
    def extract_features(self, entries: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features from parsed log entries"""
        df = pd.DataFrame(entries)
        df['success'] = df['success'].astype(int)
        
        # Use encoders if available
        if self.encoders:
            try:
                # Encode user and IP using the trained encoders
                df['user_enc'] = self.encoders['username'].transform(df['user'])
                df['ip_enc'] = self.encoders['ip'].transform(df['ip_address'])
                return df[['user_enc', 'ip_enc', 'success']]
            except Exception as e:
                logger.warning(f"Encoder failed, using raw features: {e}")
                # Fallback to raw features
                return df[['user', 'ip_address', 'success']]
        else:
            # No encoders available, use raw features
            return df[['user', 'ip_address', 'success']]
    
    def analyze_log_lines(self, log_lines: List[str]) -> Dict[str, Any]:
        """Analyze log lines for anomalies"""
        if not log_lines:
            return {"error": "No log lines provided"}

        try:
            # Parse log lines
            parsed_entries = []
            for line in log_lines:
                parsed = self.parse_log_line(line)
                if parsed:
                    parsed_entries.append(parsed)

            if not parsed_entries:
                return {"error": "No valid log entries found"}

            # Extract features
            features_df = self.extract_features(parsed_entries)
            
            results = []
            for i, entry in enumerate(parsed_entries):
                result = {
                    'line': entry['raw_line'],
                    'timestamp': entry.get('timestamp'),
                    'event_type': entry.get('event_type'),
                    'user': entry.get('user'),
                    'ip_address': entry.get('ip_address'),
                    'success': entry.get('success'),
                    'anomaly_score': 0.0,
                    'is_anomaly': False,
                    'anomaly_severity': 'low'
                }
                
                # Use models if available
                if self.models and not features_df.empty:
                    try:
                        # Use the first two columns for prediction (user and IP encodings)
                        features = features_df.iloc[i:i+1, :2].values
                        
                        # Predict failed login
                        failed_prob = self.models['failed_login'].predict_proba(features)[0, 1]
                        intrusion_prob = self.models['intrusion'].predict_proba(features)[0, 1]
                        
                        # Combine probabilities
                        anomaly_score = max(failed_prob, intrusion_prob)
                        result['anomaly_score'] = float(anomaly_score)
                        result['is_anomaly'] = anomaly_score >= 0.5
                        result['anomaly_severity'] = 'high' if anomaly_score >= 0.8 else 'medium' if anomaly_score >= 0.6 else 'low'
                        
                    except Exception as e:
                        logger.warning(f"Model prediction failed for entry {i}: {e}")
                
                results.append(result)

            # Calculate summary statistics
            anomaly_count = sum(r['is_anomaly'] for r in results)
            high_severity_count = sum(r['anomaly_severity'] == 'high' for r in results)

            return {
                'summary': {
                    'total_entries': len(results),
                    'anomalies_detected': anomaly_count,
                    'anomaly_percentage': (anomaly_count / len(results)) * 100 if results else 0,
                    'high_severity_anomalies': high_severity_count
                },
                'anomalies': [r for r in results if r['is_anomaly']],
                'all_entries': results
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

# Initialize analyzer
analyzer = AuthLogAnalyzer()

@app.post("/analyze")
async def analyze_log(file: UploadFile = File(...)):
    """Analyze uploaded auth log file"""
    try:
        # Read file content
        content = await file.read()
        log_lines = content.decode("utf-8").split('\n')
        
        # Remove empty lines
        log_lines = [line.strip() for line in log_lines if line.strip()]
        
        # Analyze the log
        result = analyzer.analyze_log_lines(log_lines)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {"error": f"Error processing file: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(analyzer.models) > 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 