import requests
from PyQt5.QtCore import QThread, pyqtSignal


class MistralClient:
    """Integrates with the Mistral API to generate clinical
    interpretations and custom playlist/music recommendations."""

    endpoint = 'https://api.mistral.ai/v1/chat/completions'

    def __init__(self, api_key: str, model: str = 'mistral-large-latest') -> None:
        """Stores the API key and model config."""
        self.api_key = api_key
        self.model = model

    def generate_insights(
        self,
        respondent_data: dict,
        predicted_scores: dict,
        cohort_stats: dict,
        static_recommendations: list,
    ) -> dict:
        """Formulates a structured system prompt and user prompt,
        sends a synchronous HTTP POST to Mistral's Chat Completions
        endpoint, and returns {'insights': str, 'recommendations': str}."""
        system_prompt = (
            "You are a clinical psychologist specializing in music therapy "
            "and mental health. Analyze the user's music listening behavior, "
            "predicted mental health scores, and cohort statistics to provide "
            "structured clinical insights and recommendations."
        )

        user_prompt = f"""
### Respondent Data:
Age: {respondent_data.get('Age', 'N/A')}
Favorite Genre: {respondent_data.get('Fav genre', 'N/A')}
Average Daily Listening Hours: {respondent_data.get('Hours per day', 'N/A')} hours
While Working: {respondent_data.get('While working', 'N/A')}
Instrumentalist: {respondent_data.get('Instrumentalist', 'N/A')}
Streaming Service: {respondent_data.get('Primary streaming service', 'N/A')}

### Predicted Scores:
Anxiety: {predicted_scores.get('Anxiety', 0.0):.2f}
Depression: {predicted_scores.get('Depression', 0.0):.2f}
Insomnia: {predicted_scores.get('Insomnia', 0.0):.2f}
OCD: {predicted_scores.get('OCD', 0.0):.2f}

### Cohort Statistics (Genre Average):
Mean Anxiety: {cohort_stats.get('Anxiety', 0.0):.2f}
Mean Depression: {cohort_stats.get('Depression', 0.0):.2f}
Mean Insomnia: {cohort_stats.get('Insomnia', 0.0):.2f}
Mean OCD: {cohort_stats.get('OCD', 0.0):.2f}
Cohort Size: {cohort_stats.get('cohort_size', 0)}

### Static Recommendations:
{chr(10).join(f'- {r}' for r in static_recommendations) if static_recommendations else 'None'}
"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
            }
            response = requests.post(
                self.endpoint, headers=headers, json=payload, timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            markdown_content = data['choices'][0]['message']['content']

            # Split content into insights and recommendations
            split_marker = '### Recommended Playlists / Artists'
            if split_marker in markdown_content:
                parts = markdown_content.split(split_marker, 1)
                insights = parts[0].strip()
                recommendations = split_marker + parts[1]
            else:
                insights = markdown_content
                recommendations = 'No structured recommendations found'

            return {
                "insights": insights,
                "recommendations": recommendations,
            }

        except Exception as e:
            raise Exception(f"Failed to generate AI Insights: {str(e)}")


class AIWorker(QThread):
    """Executes the API call asynchronously so the GUI thread
    does not block."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        client: MistralClient,
        respondent_data: dict,
        predicted_scores: dict,
        cohort_stats: dict,
        static_recommendations: list,
    ) -> None:
        """Stores references to variables and the API client."""
        super().__init__()
        self.client = client
        self.respondent_data = respondent_data
        self.predicted_scores = predicted_scores
        self.cohort_stats = cohort_stats
        self.static_recommendations = static_recommendations

    def run(self) -> None:
        """Executes self.client.generate_insights(...).
        On success: emits self.finished.emit(result).
        On exception: catches it and emits self.error.emit(str(e))."""
        try:
            result = self.client.generate_insights(
                self.respondent_data,
                self.predicted_scores,
                self.cohort_stats,
                self.static_recommendations,
            )
            if result and result.get('recommendations') and result.get('insights'):
                self.finished.emit(result)
            else:
                self.error.emit("AI returned incomplete or invalid response")
        except Exception as e:
            self.error.emit(str(e))