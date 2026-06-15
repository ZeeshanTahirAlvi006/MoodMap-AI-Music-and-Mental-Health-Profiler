import requests
from PyQt5.QtCore import QThread, pyqtSignal

class MistralClient:
    def __init__(self, api_key: str, endpoint: str = 'https://api.mistral.ai/v1/chat/completions', model: str = 'mistral-large-latest'):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
    
    def generate_insights(self, respondent_data: dict, predicted_scores: dict, cohort_stats: dict, static_recommendations: list) -> dict:
        system_prompt = """You are a clinical psychologist specializing in music therapy and mental health. Analyze the user's music listening behavior, predicted mental health scores, and cohort statistics to provide structured clinical insights and recommendations."""
        
        user_prompt = f"""
        ###Respondent Data:
        Age: {respondent_data.get('age','N/A')}
        Gender:{respondent_data.get('gender','N/A')}
        Average Daily Listening Hours: {respondent_data.get('hours','N/A')} hours
        Predominant Genres:{','.join(respondent_data.get('genres',[]))}
        Isophilicity: {respondent_data.get('Isophilicity','N/A')}
        ###Predicted Scores
        Anxiety :{predicted_scores.get('Anxiety',0.0):.2f},
        Depression:{predicted_scores.get('Depression',0.0):.2f},
        Insomnia:{predicted_scores.get("Insomnia",0.0):.2f}
        OCD: {predicted_scores.get("OCD",0.0):.2f}
        BPM Preference: {predicted_scores.get("BPM_Preference",0.0):.2f}

        ### Cohort Statistics
        Mean Anxiety:{cohort_stats.get('mean_anxiety',0.0):.2f},
        Mean Depression:{cohort_stats.get('mean_depression',0.0):.2f},
        Mean Insomnia:{cohort_stats.get('mean_insomnia',0.0):.2f},
        Mean OCD:{cohort_stats.get('mean_ocd',0.0):.2f}

        ###Static Recommendations
        {static_recommendations}
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3 # low temperature for more clinical tone
            }
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            # parse response into json
            data = response.json()
            markdown_content = data['choices'][0]['message']['content']

            # split content into insights and recommendations
            split_marker = '###Recommended Playlists / Artists'
            if split_marker in markdown_content:
                parts = markdown_content.split(split_marker, 1)
                insights = parts[0].strip()
                recommendations = split_marker + parts[1]
            else:
                # treat everything as insight
                insights = markdown_content
                recommendations = 'No structured recommendations found'
            return {
                "insights": insights,
                "recommendations": recommendations
            }

        except Exception as e:
            raise Exception(f"Failed to generate AI Insights: {str(e)}")

class AIWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, client, respondent_data, predicted_scores, cohort_stats, static_recommendations) -> None:
        super().__init__()
        self.client = client
        self.respondent_data = respondent_data
        self.predicted_scores = predicted_scores
        self.cohort_stats = cohort_stats
        self.static_recommendations = static_recommendations

    def run(self):
        try:
            result = self.client.generate_insights(
                self.respondent_data,
                self.predicted_scores,
                self.cohort_stats,
                self.static_recommendations,
            )
            if result and result.get('recommendations') and result.get("insights"):
                self.finished.emit(result)
            else:
                self.error.emit("AI returned incomplete or invalid response")
        except Exception as e:
            self.error.emit(str(e))