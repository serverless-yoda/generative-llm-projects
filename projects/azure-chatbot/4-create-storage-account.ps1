# Create Storage Account
az storage account create `
  --name roadsideassistantstorage `
  --resource-group RoadsideAssistantRG `
  --location eastus `
  --sku Standard_LRS `
  --kind StorageV2