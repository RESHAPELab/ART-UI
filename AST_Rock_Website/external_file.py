"""
ART-CoreEngine - 2024 - Benjamin Carter 

This file holds the outbound API calls that the UI team can later use.
It is meant to be imported into any external application

The contents of this class shall not depend on anything external.
It must be self contained.

"""
import pandas as pd
import json
import os
import pickle
import sys
from joblib import Memory
from django.core.cache import cache  # Assuming default Django cache setup
from open_issue import (
    generate_system_message,
    get_gpt_response_one_issue,
    clean_text_rf,
    predict_open_issues,
)
from database_manager_file import DatabaseManager

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CoreEngine', 'src'))
sys.path.insert(0, src_dir)


from issue_class import Issue



class External_Model_Interface:
    def __init__(
        self,
        open_ai_key: str,
        db: DatabaseManager,
        model_file: str,
        domain_file: str,
        response_cache_key: str,
    ):
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        with open(domain_file, "r") as f:
            self.domains = json.load(f)

        self.db = db
        self.model_file_name = model_file
        self.__open_ai_key = open_ai_key
        self.response_cache_key = response_cache_key

    def predict_issue(self, issue: Issue):
        # Cache key incorporates the model to ensure updates to the model invalidate the cache
        cache_key = f"{self.response_cache_key}_{self.model['type']}_{self.model_file_name}_{issue.number}"
        # Check if we have a cached result first
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # If not cached, compute the result and cache it
        result = self.__predict_issue(issue.number, issue.title, issue.body, None)
        cache.set(cache_key, result, timeout=None)  # Adjust timeout as needed
        return result

    def __predict_issue(self, num, title, body, _ignore_me):
        issue = Issue(num=num, title=title, body=body)  # Ensure Issue is either a Django model or appropriately formatted

        if self.model["type"] == "gpt":
            return self.__gpt_predict(issue)
        elif self.model["type"] == "rf":
            return self.__rf_predict(issue)
        else:
            raise NotImplementedError("Model type not recognized")

    def __gpt_predict(self, issue: Issue):
        # Assuming llm_classifier is part of self.model and configured
        columns = self.db.get_df_column_names()
        empty = [list(range(len(columns)))]
        df = pd.DataFrame(data=empty, columns=columns)

        # Placeholder for actual GPT response function
        response = self.__simulate_gpt_response(issue)
        return response

    def __rf_predict(self, issue: Issue):
        # Assuming clf (classifier) and vx (vectorizer) are part of self.model
        df = pd.DataFrame(columns=["Issue #", "Title", "Body"], data=[[issue.number, issue.title, issue.body]])
        vectorized_text = self.__clean_text_rf(df)  # Assuming clean_text_rf is implemented properly

        # Placeholder for actual prediction function
        predictions = self.__simulate_rf_predictions(df)
        return predictions[:3]  # Assuming we want the top 3 predictions

    # Dummy functions to simulate responses
    def __simulate_gpt_response(self, issue):
        return "GPT response based on issue"

    def __simulate_rf_predictions(self, df):
        return ["Prediction1", "Prediction2", "Prediction3"]

    def __clean_text_rf(self, df):
        # Implement text cleaning for RF model prediction
        return df
