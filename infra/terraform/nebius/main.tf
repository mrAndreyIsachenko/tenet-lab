locals {
  gpu_presets = {
    wave1 = {
      platform = "gpu-h200-sxm"
      preset   = "1gpu-16vcpu-200gb"
    }
    wave2 = {
      platform = "gpu-h200-sxm"
      preset   = "4gpu-64vcpu-800gb"
    }
  }

  selected_gpu    = local.gpu_presets[var.gpu_preset]
  scratch_size_gb = max(93, floor(var.nvme_scratch_size_gb / 93) * 93)
  ssh_pub_key     = trimspace(file(pathexpand(var.ssh_public_key_path)))
  repo_root       = abspath("${path.module}/../../..")
  public_ip       = split("/", nebius_vpc_v1_allocation.public_ip.status.details.allocated_cidr)[0]
  ssh_host        = "${var.ssh_user}@${local.public_ip}"
  mlflow_uri = coalesce(
    try(nebius_msp_mlflow_v1alpha1_cluster.mlflow.status.tracking_endpoints.public, null),
    try(nebius_msp_mlflow_v1alpha1_cluster.mlflow.status.tracking_endpoint, null),
    ""
  )
}

resource "nebius_vpc_v1_network" "main" {
  parent_id = var.parent_id
  name      = "tenet-v3-net-${var.region}"
}

resource "nebius_vpc_v1_subnet" "main" {
  parent_id  = var.parent_id
  name       = "tenet-v3-subnet-${var.region}"
  network_id = nebius_vpc_v1_network.main.id

  ipv4_private_pools = {
    use_network_pools = true
  }
}

resource "nebius_vpc_v1_allocation" "public_ip" {
  parent_id = var.parent_id
  name      = "tenet-v3-gpu-ip"

  ipv4_public = {
    subnet_id = nebius_vpc_v1_subnet.main.id
  }
}

data "nebius_iam_v2_project" "current" {
  id = var.parent_id
}

data "nebius_iam_v1_group" "editors" {
  parent_id = data.nebius_iam_v2_project.current.parent_id
  name      = "editors"
}

resource "nebius_iam_v1_service_account" "tenet" {
  parent_id   = var.parent_id
  name        = "tenet-v3-sa"
  description = "Service account for Tenet v3 GPU VM, Object Storage, and Managed MLflow"
}

resource "nebius_iam_v1_group_membership" "tenet_editors" {
  parent_id = data.nebius_iam_v1_group.editors.id
  member_id = nebius_iam_v1_service_account.tenet.id
}

resource "nebius_storage_v1_bucket" "artifacts" {
  parent_id = var.parent_id
  name      = var.s3_bucket_name
}

resource "nebius_msp_mlflow_v1alpha1_cluster" "mlflow" {
  parent_id           = var.parent_id
  name                = "tenet-v3-mlflow"
  description         = "Managed MLflow tracking server for Small Model Lab (Tenet v1-v3)"
  network_id          = nebius_vpc_v1_network.main.id
  service_account_id  = nebius_iam_v1_service_account.tenet.id
  admin_username      = var.mlflow_admin_username
  public_access       = true
  size                = var.mlflow_cluster_size
  storage_bucket_name = nebius_storage_v1_bucket.artifacts.name

  sensitive = {
    version        = "1"
    admin_password = var.mlflow_admin_password
  }

  depends_on = [nebius_iam_v1_group_membership.tenet_editors]
}

resource "nebius_compute_v1_disk" "boot" {
  parent_id      = var.parent_id
  name           = "tenet-v3-boot-ssd"
  type           = "NETWORK_SSD"
  size_gibibytes = var.boot_disk_size_gb

  source_image_family = {
    image_family = var.boot_image_family
  }
}

resource "nebius_compute_v1_disk" "nvme_scratch" {
  parent_id      = var.parent_id
  name           = "tenet-v3-nvme-scratch"
  type           = "NETWORK_SSD_NON_REPLICATED"
  size_gibibytes = local.scratch_size_gb
}

resource "nebius_compute_v1_instance" "gpu" {
  parent_id          = var.parent_id
  name               = "tenet-v3-gpu-${var.gpu_preset}"
  stopped            = var.instance_stopped
  service_account_id = nebius_iam_v1_service_account.tenet.id

  resources = {
    platform = local.selected_gpu.platform
    preset   = local.selected_gpu.preset
  }

  boot_disk = {
    attach_mode = "READ_WRITE"
    existing_disk = {
      id = nebius_compute_v1_disk.boot.id
    }
  }

  secondary_disks = [
    {
      attach_mode = "READ_WRITE"
      device_id   = "nvme-scratch"
      existing_disk = {
        id = nebius_compute_v1_disk.nvme_scratch.id
      }
    }
  ]

  network_interfaces = [
    {
      name       = "eth0"
      subnet_id  = nebius_vpc_v1_subnet.main.id
      ip_address = {}
      public_ip_address = {
        allocation_id = nebius_vpc_v1_allocation.public_ip.id
        static        = true
      }
    }
  ]

  cloud_init_user_data = <<-EOF
    #cloud-config
    users:
      - name: ${var.ssh_user}
        sudo: ALL=(ALL) NOPASSWD:ALL
        shell: /bin/bash
        ssh_authorized_keys:
          - ${local.ssh_pub_key}
  EOF
}

resource "local_file" "nebius_env" {
  filename        = "${local.repo_root}/.env.nebius"
  file_permission = "0600"
  content = templatefile("${path.module}/templates/nebius_env.tftpl", {
    nebius_instance_id  = nebius_compute_v1_instance.gpu.id
    nebius_ssh_host     = local.ssh_host
    nebius_s3_bucket    = nebius_storage_v1_bucket.artifacts.name
    mlflow_tracking_uri = local.mlflow_uri
    nebius_region       = var.region
    pod_host            = local.ssh_host
  })
}

resource "local_file" "ansible_inventory" {
  filename        = "${local.repo_root}/infra/ansible/inventory/hosts.yml"
  file_permission = "0644"
  content = templatefile("${path.module}/templates/ansible_inventory.tftpl", {
    public_ip           = local.public_ip
    ssh_user            = var.ssh_user
    nebius_instance_id  = nebius_compute_v1_instance.gpu.id
    nebius_s3_bucket    = nebius_storage_v1_bucket.artifacts.name
    mlflow_tracking_uri = local.mlflow_uri
    nebius_region       = var.region
  })
}
