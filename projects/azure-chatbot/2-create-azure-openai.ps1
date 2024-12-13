# Create Azure OpenAI Service
az cognitiveservices account create `
  --name roadside-openai-service `
  --resource-group RoadsideAssistantRG `
  --location eastus `
  --kind OpenAI `
  --sku S0
