output "nebius_instance_id" {
  description = "Nebius Compute GPU instance ID (consumed by scripts/remote/nebius_ctl.sh)."
  value       = nebius_compute_v1_instance.gpu.id
}

output "nebius_ssh_host" {
  description = "SSH target user@ip for the Nebius GPU instance."
  value       = local.ssh_host
}

output "nebius_s3_bucket" {
  description = "Nebius Object Storage bucket for Tenet v3 artifacts."
  value       = nebius_storage_v1_bucket.artifacts.name
}

output "mlflow_tracking_uri" {
  description = "Managed MLflow tracking URI."
  value       = local.mlflow_uri
}
