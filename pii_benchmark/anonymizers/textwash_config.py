import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "textwash", "data")

class Config:
    def __init__(self, language):
        self.path_to_months_file = os.path.join(_DATA_DIR, "months.txt")
        self.path_to_written_numbers_file = os.path.join(_DATA_DIR, "written_numbers.txt")

        assert language in ["nl", "en"], f"Invalid language {language} specified (only 'nl' and 'en' supported)"

        self.path_to_model = os.path.join(_DATA_DIR, language)
        self.model_type = "bert" if language == "nl" else "roberta"