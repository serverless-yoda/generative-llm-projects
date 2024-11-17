import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

def verify_azure_credentials(subscription_key, endpoint):
    """
    Verify Azure credentials by attempting to create a client and make a simple API call.
    """
    try:
        # Create client with credentials
        client = ComputerVisionClient(
            endpoint,
            CognitiveServicesCredentials(subscription_key)
        )
        
        # Print credential information (partially masked)
        print("\nCredentials being used:")
        print(f"Endpoint: {endpoint}")
        print(f"Subscription Key: {subscription_key[:4]}...{subscription_key[-4:]}")
        
        # Get the brands list (lightweight API call)
        print("\nTesting API connection...")
        brands = client.list_models()
        print("✓ Connection successful! Azure credentials are valid.")
        return True
        
    except Exception as e:
        print("\n❌ Error: Authentication failed")
        print("\nDetailed error message:")
        print(str(e))
        
        # Common error solutions
        print("\nPossible solutions:")
        print("1. Check if the subscription key is copied correctly (no extra spaces)")
        print("2. Verify the endpoint URL format (should end with azure.com)")
        print("3. Confirm the region in the endpoint matches your resource")
        print("4. Ensure your Azure resource is active and not suspended")
        return False

def main():
    print("Azure Credentials Verification Tool")
    print("==================================")
    
    # Get credentials from user
    print("\nPlease enter your Azure credentials:")
    subscription_key = input("Subscription Key: ").strip()
    endpoint = input("Endpoint URL: ").strip()
    
    # Verify endpoint format
    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}"
    if not endpoint.endswith("azure.com"):
        if "azure.com" not in endpoint:
            endpoint = f"{endpoint}.cognitiveservices.azure.com"
    
    # Verify credentials
    verify_azure_credentials(subscription_key, endpoint)

if __name__ == "__main__":
    main()