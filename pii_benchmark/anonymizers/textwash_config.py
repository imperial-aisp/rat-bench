class Config:
    def __init__(self, language):
        self.path_to_months_file = "data_textwash/months.txt"
        self.path_to_written_numbers_file = "data_textwash/written_numbers.txt"

        assert language in ["nl", "en"], f"Invalid language {language} specified (only 'nl' and 'en' supported)"
        
        self.path_to_model = f"data_textwash/{language}"
        self.model_type = "bert" if language == "nl" else "roberta"