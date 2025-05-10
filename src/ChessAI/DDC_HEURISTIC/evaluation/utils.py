from typing import List, Dict, Any
import yaml 
import os 

def load_yaml(file_path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents.
    
    Args:
        file_path (str): The path to the YAML file.
        
    Returns:
        dict: The contents of the YAML file.
        
    Example:
        >>> config = load_yaml('config.yaml')
        >>> print(config)
        ... {'key': 'value'}
    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not a valid YAML file.
    """
    try: 
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise