terraform {
  required_version = ">= 1.5.0"

  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.6.8"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.4.0"
    }
  }
}

provider "nebius" {
  # Reads NEBIUS_IAM_TOKEN from environment or ~/.nebius/credentials by default.
}
