# Create Azure Cognitive Search
az search service create `
  --name roadside-search-service `
  --resource-group RoadsideAssistantRG `
  --location eastus `
  --sku Standard
