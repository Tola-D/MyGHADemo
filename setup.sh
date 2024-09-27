#!/bin/bash

# Set the necessary variables
suffix=$(date +%s)  # Generate a suffix using the current timestamp (optional, or define your own)
RESOURCE_GROUP="rg-aiamlpractres-l${suffix}"
RESOURCE_PROVIDER="Microsoft.MachineLearningServices"
REGIONS=("eastus" "eastus2" "westus" "centralus" "northeurope" "westeurope")
RANDOM_REGION=${REGIONS[$RANDOM % ${#REGIONS[@]}]}
WORKSPACE_NAME="mlw-aiamlpract-l${suffix}"
COMPUTE_INSTANCE="ci${suffix}"
COMPUTE_CLUSTER="aml-cluster"

# Log in to Azure (optional if you're already logged in)
# echo "Logging in to Azure..."
# az login

# Set the subscription (optional if you want to specify the subscription)
echo "Setting subscription..."
az account set --subscription $SUBSCRIPTION_ID

# Register the Azure Machine Learning resource provider in the subscription
echo "Register the Machine Learning resource provider:"
az provider register --namespace $RESOURCE_PROVIDER

# Create the resource group and workspace and set to default
echo "Create a resource group and set as default:"
az group create --name $RESOURCE_GROUP --location $RANDOM_REGION
az configure --defaults group=$RESOURCE_GROUP

echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name $WORKSPACE_NAME
az configure --defaults workspace=$WORKSPACE_NAME

# Create compute instance
echo "Creating a compute instance with name: $COMPUTE_INSTANCE"
az ml compute create --name ${COMPUTE_INSTANCE} --size STANDARD_DS11_V2 --type ComputeInstance

# Create compute cluster
echo "Creating a compute cluster with name: $COMPUTE_CLUSTER"
az ml compute create --name ${COMPUTE_CLUSTER} --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute