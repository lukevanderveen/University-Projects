import sys
import os

from ml4floods.data.copernicusEMS import activations
from ml4floods.data import utils
from pathlib import Path

table_activations_ems = activations.table_floods_ems(event_start_date="2021-01-01")
table_activations_ems

emsr_code = "EMSR501"
zip_files_activation_url_list = activations.fetch_zip_file_urls(emsr_code)
zip_files_activation_url_list
