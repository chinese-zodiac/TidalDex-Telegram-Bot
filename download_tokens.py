import os
import json
import requests
from web3 import Web3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get token list URL from .env
TOKEN_LIST_URL = os.getenv('DEFAULT_TOKEN_LIST')
TOKEN_LIST_FILE = os.getenv("TOKENS_FILE", "tidaldex_tokens.json")

def download_token_list():
    """Download and save TidalDex token list if version has changed"""
    try:
        # Download the token list
        print(f"Downloading TidalDex token list...")
        response = requests.get(TOKEN_LIST_URL)
        response.raise_for_status()
        token_list = response.json()
        
        # Check current version
        current_version = None
        if os.path.exists(TOKEN_LIST_FILE):
            with open(TOKEN_LIST_FILE, 'r') as f:
                saved_data = json.load(f)
                current_version = saved_data.get('version')
        
        # Compare versions
        new_version = token_list.get('version')
        if current_version == new_version:
            print(f"Token list already at latest version: {new_version}")
            return False
            
        # Process tokens
        processed_tokens = {}
        for token in token_list['tokens']:
            # Only include tokens on BSC (chainId 56)
            if token['chainId'] == 56:
                address = Web3.to_checksum_address(token['address'])
                processed_tokens[address] = {
                    'name': token['name'],
                    'symbol': token['symbol'],
                    'decimals': token['decimals'],
                    'logoURI': token.get('logoURI', '')
                }
        
        # Save to file with version
        output_data = {
            'version': new_version,
            'tokens': processed_tokens
        }
        
        with open(TOKEN_LIST_FILE, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"Updated token list to version {new_version} with {len(processed_tokens)} tokens")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading token list: {e}")
        return False
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing token list: {e}")
        return False

if __name__ == "__main__":
    download_token_list() 