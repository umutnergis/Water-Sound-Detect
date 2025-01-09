import requests
import json

def send_sensor_data():
    # API endpoint
    url = "https://www.waguse.com/api.php"
    
    # Query parameters
    params = {
        "action": "ekle"
    }
    
    # Request headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Request body data
    data = {
        "username": "admin",
        "password": "admin",
        "sensor_id": 1,
        "veri_tipi": "su",
        "veri_degeri": "fazla akış",
        "isik_suresi": 3.5
    }

    try:
        # Make POST request
        response = requests.post(
            url=url,
            params=params,
            headers=headers,
            json=data  # requests will automatically JSON encode the dictionary
        )
        
        # Check if request was successful
        if response.status_code == 200:
            try:
                return response.json()  # Return parsed JSON response
            except json.JSONDecodeError:
                return response.text    # Return raw response if not JSON
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

# Use the function
if __name__ == "__main__":
    result = send_sensor_data()
    if result:
        print("Response:")
        print(json.dumps(result, indent=2))  # Pretty print the JSON response