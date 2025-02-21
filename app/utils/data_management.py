# app/utils/data_management.py

import os
from datetime import datetime, timedelta

def get_last_update_time(file_path):
    """
    Get the last modified time of a file.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - datetime: Last modified datetime of the file.
    """
    if not os.path.exists(file_path):
        return None
    timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(timestamp)

def needs_update(last_update, update_interval_days=1):
    """
    Determine if data needs to be updated based on the last update time.

    Parameters:
    - last_update (datetime): Last update datetime.
    - update_interval_days (int): Number of days after which data should be updated.

    Returns:
    - bool: True if update is needed, False otherwise.
    """
    if not last_update:
        return True
    return (datetime.now() - last_update) > timedelta(days=update_interval_days)