import sys
import os
from PyQt5.QtWidgets import QApplication
from gui import MoodMapApp
from data_loader import DataLoader
from history import HistoryManager

def main():
    # Ensure correct working directory so relative paths resolve properly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    app = QApplication(sys.argv)
    app.setApplicationName("MoodMap")
    
    # Initialize DataLoader and HistoryManager
    csv_path = os.path.join("archive", "mxmh_survey_results.csv")
    data_loader = DataLoader(csv_path)
    history_manager = HistoryManager(
        history_path="moodmap_history.csv",
        settings_path="moodmap_settings.csv"
    )
    
    # Launch main QMainWindow
    window = MoodMapApp(data_loader, history_manager)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
