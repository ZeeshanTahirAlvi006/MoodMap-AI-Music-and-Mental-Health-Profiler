import csv
import os
from datetime import datetime

HISTORY_HEADERS=[
    'id','timestamp','age','genre','hours','while_working','instrumentalist','streaming','anxiety','depression',
    'insomnia','ocd','anxiety_pct','depression_pct','insomnia_pct','ocd_pct'
]
SETTINGS_HEADERS = ['key','value']

class HistoryManager:
    def __init__(self,history_path:str = 'moodmap_history.csv',settings_path:str = 'moodmap_settings.csv')-> None:
        self.history_path = history_path
        self.settings_path = settings_path
        self._ensure_file(self.history_path,HISTORY_HEADERS)
        self._ensure_file(self.settings_path,SETTINGS_HEADERS)
    def _ensure_file(self,path:str,headers:list)-> None:
        if not os.path.exists(path):
            with open(path,'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    def save_session(self,respondent,predicted_scores:dict,percentiles:dict)->int:
        session_id=self.get_session_count()+1
        row = [
            session_id,
            datetime.now().isoformat(),
            respondent.age,
            respondent.fav_genre,
            respondent.hours_per_day,
            respondent.while_working,
            respondent.instrumentalist,
            respondent.streaming_service,
            predicted_scores.get('Anxiety',0.0),
            predicted_scores.get('Depression',0.0),
            predicted_scores.get('Insomnia',0.0),
            predicted_scores.get('OCD',0.0),
            percentiles.get('Anxiety',0.0),
            percentiles.get('Depression',0.0),
            percentiles.get('Insomnia',0.0),
            percentiles.get('OCD',0.0)
        ]
        with open(self.history_path,'a',newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return session_id
    def get_all_session(self)->list[dict]:
        sessions =[]
        if not os.path.exists(self.history_path):
            return sessions
        with open(self.history_path,'r',newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sessions.append(row)
        sessions.reverse()
        return sessions
    def get_trend_data(self)->dict[str,list[float]]:
        trend ={
            'Anxiety':[],
            'Depression':[],
            'Insomnia':[],
            'OCD':[]
        }
        if not os.path.exists(self.history_path):
            return trend
        with open(self.history_path,'r',newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trend['Anxiety'].append(float(row.get('anxiety',0.0)))
                trend['Depression'].append(float(row.get('depression',0.0)))
                trend['Insomnia'].append(float(row.get('insomnia',0.0)))
                trend['OCD'].append(float(row.get('ocd',0.0)))
        return trend
    def clear_all(self)->None:
        with open(self.history_path,'w',newline='') as f:
            write = csv.writer(f)
            write.writerow(HISTORY_HEADERS)
    def get_setting(self,key:str)->str:
        if not os.path.exists(self.settings_path):
            return ''
        with open(self.settings_path,'r',newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['key'] == key:
                    return row['value']
        return ''
    def set_setting(self,key:str,value:str)-> None:
        
        rows = []
        found = False
        if os.path.exists(self.settings_path):
            with open(self.settings_path,'r',newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['key'] == key:
                        row['value'] = value
                        found = True
                    rows.append(row)
        if not found:
            rows.append({'key':key,'value':value})
        with open(self.settings_path,'w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=SETTINGS_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
    def delete_session(self,session_id:int)->None:
        if not os.path.exists(self.history_path):
            return   
        with open(self.history_path,'r',newline='') as f:
            reader = csv.DictReader(f) 
            rows = [row for row in reader if int(row['id'])!= session_id]
        with open(self.history_path,'w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=HISTORY_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
    def get_session_count(self)->int:
        if not os.path.exists(self.history_path):
            return 0
        with open(self.history_path,'r',newline='') as f:
            reader = csv.reader(f)
            next(reader,None) # skip header
            return sum(1 for _ in reader)

