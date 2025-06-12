# %%
import os
import platform

_base_local_dataset_path_map = {
    "computer-name": r"path/to/processed/datasets/folder/",
}

_base_local_results_path_map = {
    "computer-name": r"path/to/results/folder/",
}

_base_server_dataset_path_map = {
    "user-name": r"path/to/processed/datasets/folder/",
}
_base_server_results_path_map = {
    "user-name": r"path/to/results/folder/",
}

relative_data_paths = {}

# %%


def _is_running_on_server():
    return False


def get_data_path(key):
    # determine whether we are running on server
    if _is_running_on_server():
        username = os.getlogin()

        if username not in _base_server_dataset_path_map:
            raise KeyError(
                f"No dataset location found for user '{username}' on server. "
                f"Please extend '_base_server_dataset_path_map' in 'data_paths.py'."
            )
        path = os.path.join(
            _base_server_dataset_path_map[username], relative_data_paths[key]
        )
    else:
        computer_name = platform.node()
        if computer_name not in _base_local_dataset_path_map:
            raise KeyError(
                f"No dataset location found on computer '{computer_name}'. "
                f"Please extend '_base_local_dataset_path_map' in 'data_paths.py'."
            )
        path = os.path.join(
            _base_local_dataset_path_map[computer_name], relative_data_paths[key]
        )

    return path


def get_available_datasets():
    # determine whether we are running on server
    if _is_running_on_server():
        username = os.getlogin()
        if username not in _base_server_dataset_path_map:
            raise KeyError(
                f"No dataset location found for user '{username}' on server. "
                f"Please extend '_base_server_dataset_path_map' in 'data_paths.py'."
            )
        path = _base_server_dataset_path_map[username]
    else:
        computer_name = platform.node()
        if computer_name not in _base_local_dataset_path_map:
            raise KeyError(
                f"No dataset location found on computer '{computer_name}'. "
                f"Please extend '_base_local_dataset_path_map' in 'data_paths.py'."
            )
        path = _base_local_dataset_path_map[computer_name]
    relative_data_paths = {
        dataset_name: dataset_name for dataset_name in os.listdir(path)
    }
    return relative_data_paths


def get_storage_path():
    # determine whether we are running on server
    if _is_running_on_server():
        username = os.getlogin()
        if username not in _base_server_results_path_map:
            raise KeyError(
                f"No results location found for user '{username}' on server. "
                f"Please extend '_base_server_results_path_map' in 'data_paths.py'."
            )
        path = _base_server_results_path_map[username]
    else:
        computer_name = platform.node()
        datasource_conf = os.path.join("datasources", f"{computer_name}.yaml")
        if computer_name not in _base_local_results_path_map:
            raise KeyError(
                f"No results location found on computer '{computer_name}'. "
                f"Please extend '_base_local_results_path_map' in 'data_paths.py'."
            )
        path = _base_local_results_path_map[computer_name]
    return path


relative_data_paths = get_available_datasets()
