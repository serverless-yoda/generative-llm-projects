# Using Azure CLI
az login
az group create --name RoadsideAssistantRG --location eastus

# Create Azure OpenAI Service
az cognitiveservices account create \
  --name roadside-openai-service \
  --resource-group RoadsideAssistantRG \
  --location eastus \
  --kind OpenAI \
  --sku S0

# Create Azure Cognitive Search
az search service create \
  --name roadside-search-service \
  --resource-group RoadsideAssistantRG \
  --location eastus \
  --sku Standard


# Create Storage Account
az storage account create \
  --name roadsideassistantstorage \
  --resource-group RoadsideAssistantRG \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2

# Create Blob Container
az storage container create \
  --name roadside-docs \
  --account-name roadsideassistantstorage