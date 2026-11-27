variable "parent_id" {
  type        = string
  description = "Nebius AI Cloud project / parent ID (e.g. project-e00...)."
}

variable "region" {
  type        = string
  default     = "eu-north1"
  description = "Nebius AI Cloud region."
}

variable "gpu_preset" {
  type        = string
  default     = "wave1"
  description = "GPU preset tier: 'wave1' (1x H200 141GB) or 'wave2' (8x H200 1128GB)."

  validation {
    condition     = contains(["wave1", "wave2"], var.gpu_preset)
    error_message = "gpu_preset must be either 'wave1' or 'wave2'."
  }
}

variable "boot_disk_size_gb" {
  type        = number
  default     = 200
  description = "Persistent boot SSD size in GiB (preserves OS and .venv across stops)."
}

variable "nvme_scratch_size_gb" {
  type        = number
  default     = 1023
  description = "NVMe scratch disk size in GiB for /mnt/nvme (multiple of 93 GiB for non-replicated SSD)."
}

variable "s3_bucket_name" {
  type        = string
  default     = "tenet-v3-artifacts"
  description = "Nebius Object Storage bucket name for Tenet v3 artifacts and MLflow."
}

variable "ssh_public_key_path" {
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
  description = "Path to SSH public key injected via cloud-init."
}

variable "ssh_user" {
  type        = string
  default     = "ubuntu"
  description = "SSH username on the Nebius GPU VM."
}

variable "instance_stopped" {
  type        = bool
  default     = false
  description = "Set true to stop the GPU VM ($0/hr GPU billing while keeping disks)."
}

variable "preemptible" {
  type        = bool
  default     = true
  description = "Launch GPU VM as preemptible spot instance (on_preemption=STOP, ~3x cheaper)."
}

variable "boot_image_family" {
  type        = string
  default     = "ubuntu24.04-cuda13.0"
  description = "Nebius compute image family for the GPU VM."
}

variable "mlflow_admin_username" {
  type        = string
  default     = "admin"
  description = "Admin username for Managed MLflow."
}

variable "mlflow_admin_password" {
  type        = string
  sensitive   = true
  default     = "change-me-mlflow-password"
  description = "Admin password for Managed MLflow."
}

variable "mlflow_cluster_size" {
  type        = string
  default     = "medium"
  description = "Managed MLflow cluster size (medium)."
}
