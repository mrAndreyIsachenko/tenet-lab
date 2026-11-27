"""Tests for Nebius AI Cloud Terraform & Ansible IaC integration."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TF_DIR = REPO_ROOT / "infra" / "terraform" / "nebius"
ANSIBLE_DIR = REPO_ROOT / "infra" / "ansible"
NEBIUS_CTL = REPO_ROOT / "scripts" / "remote" / "nebius_ctl.sh"
MAKEFILE = REPO_ROOT / "Makefile"


def test_terraform_files_and_resources_present() -> None:
    """Verify all required Terraform files, resources, variables, and outputs exist."""
    required_files = [
        TF_DIR / "versions.tf",
        TF_DIR / "variables.tf",
        TF_DIR / "main.tf",
        TF_DIR / "outputs.tf",
        TF_DIR / "terraform.tfvars.example",
        TF_DIR / "templates" / "nebius_env.tftpl",
        TF_DIR / "templates" / "ansible_inventory.tftpl",
    ]
    for path in required_files:
        assert path.is_file(), f"Missing Terraform file: {path}"

    versions_text = (TF_DIR / "versions.tf").read_text(encoding="utf-8")
    assert "nebius/nebius" in versions_text

    variables_text = (TF_DIR / "variables.tf").read_text(encoding="utf-8")
    for var_name in (
        "parent_id",
        "region",
        "gpu_preset",
        "boot_disk_size_gb",
        "nvme_scratch_size_gb",
        "s3_bucket_name",
        "ssh_public_key_path",
        "instance_stopped",
    ):
        assert f'variable "{var_name}"' in variables_text

    main_text = (TF_DIR / "main.tf").read_text(encoding="utf-8")
    for resource_type in (
        "nebius_vpc_v1_network",
        "nebius_vpc_v1_subnet",
        "nebius_vpc_v1_allocation",
        "nebius_iam_v1_service_account",
        "nebius_storage_v1_bucket",
        "nebius_msp_mlflow_v1alpha1_cluster",
        "nebius_compute_v1_disk",
        "nebius_compute_v1_instance",
    ):
        assert f'resource "{resource_type}"' in main_text
    assert (
        "stopped            = var.instance_stopped" in main_text
        or "stopped = var.instance_stopped" in main_text
    )

    outputs_text = (TF_DIR / "outputs.tf").read_text(encoding="utf-8")
    for out_name in (
        "nebius_instance_id",
        "nebius_ssh_host",
        "nebius_s3_bucket",
        "mlflow_tracking_uri",
    ):
        assert f'output "{out_name}"' in outputs_text


def test_nebius_env_template_matches_nebius_ctl() -> None:
    """Verify .env.nebius template exports every variable consumed by nebius_ctl.sh."""
    tpl_text = (TF_DIR / "templates" / "nebius_env.tftpl").read_text(encoding="utf-8")
    ctl_text = NEBIUS_CTL.read_text(encoding="utf-8")

    for env_key in (
        "NEBIUS_INSTANCE_ID",
        "NEBIUS_SSH_HOST",
        "NEBIUS_S3_BUCKET",
        "NEBIUS_RESULTS_DIR",
        "MLFLOW_TRACKING_URI",
    ):
        assert f"{env_key}=" in tpl_text
        assert env_key in ctl_text

    assert "UV_NO_SYNC=1" in tpl_text
    assert "SKIP_UV_SYNC=1" in tpl_text


def test_hf_token_resolution_and_idle_watchdog_guard() -> None:
    """Watchdog pgrep covers foundry; runtime role resolves/asserts HF token; watchdog restarts."""
    watchdog_text = (REPO_ROOT / "scripts" / "remote" / "idle_watchdog.sh").read_text(
        encoding="utf-8"
    )
    assert "run_tenet_v3_foundry" in watchdog_text
    assert 'pgrep -f "run_tenet_v3_foundry' in watchdog_text

    runtime_path = ANSIBLE_DIR / "roles" / "runtime" / "tasks" / "main.yml"
    runtime_data = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    names = [task.get("name") for task in runtime_data]
    resolve_name = "Resolve HF_TOKEN from controller environment or root .env"
    assert_name = "Assert HF_TOKEN is non-empty before writing VM .env"
    template_name = "Write runtime .env file for Tenet v3"
    for required in (resolve_name, assert_name, template_name):
        assert required in names, f"Missing runtime task: {required}"

    resolve_task = runtime_data[names.index(resolve_name)]
    resolve_src = yaml.safe_dump(resolve_task)
    assert "HF_TOKEN" in resolve_src
    assert ".env" in resolve_src
    assert resolve_task.get("no_log") is True

    assert_task = runtime_data[names.index(assert_name)]
    assert "tenet_hf_token | length >= 10" in yaml.safe_dump(assert_task)
    assert assert_task.get("no_log") is True

    template_task = runtime_data[names.index(template_name)]
    assert template_task["ansible.builtin.template"]["src"] == "env.j2"

    env_text = (ANSIBLE_DIR / "roles" / "runtime" / "templates" / "env.j2").read_text(
        encoding="utf-8"
    )
    assert "tenet_hf_token" in env_text

    guards_data = yaml.safe_load(
        (ANSIBLE_DIR / "roles" / "systemd_guards" / "tasks" / "main.yml").read_text(
            encoding="utf-8"
        )
    )
    watchdog_tasks = [
        task
        for task in guards_data
        if task.get("ansible.builtin.systemd", {}).get("name") == "tenet-idle-watchdog.service"
    ]
    assert watchdog_tasks, "expected a systemd task for tenet-idle-watchdog.service"
    assert watchdog_tasks[0]["ansible.builtin.systemd"]["state"] == "restarted"


def test_ansible_structure_and_roles() -> None:
    """Verify Ansible config, playbook, and all 3 roles/templates are valid YAML & complete."""
    assert (ANSIBLE_DIR / "ansible.cfg").is_file()
    playbook_path = ANSIBLE_DIR / "playbook.yml"
    assert playbook_path.is_file()

    playbook_data = yaml.safe_load(playbook_path.read_text(encoding="utf-8"))
    assert isinstance(playbook_data, list) and len(playbook_data) == 1
    assert playbook_data[0]["roles"] == ["nvme_scratch", "runtime", "systemd_guards"]

    for role in ("nvme_scratch", "runtime", "systemd_guards"):
        tasks_path = ANSIBLE_DIR / "roles" / role / "tasks" / "main.yml"
        assert tasks_path.is_file(), f"Missing role tasks: {tasks_path}"
        parsed = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))
        assert isinstance(parsed, list) and len(parsed) > 0

    runtime_tpl = ANSIBLE_DIR / "roles" / "runtime" / "templates"
    env_j2 = (runtime_tpl / "env.j2").read_text(encoding="utf-8")
    for key in (
        "UV_NO_SYNC=1",
        "SKIP_UV_SYNC=1",
        "HF_TOKEN=",
        "MLFLOW_TRACKING_URI=",
        "NEBIUS_S3_BUCKET=",
    ):
        assert key in env_j2

    guard_tpl = ANSIBLE_DIR / "roles" / "systemd_guards" / "templates"
    s3_timer = (guard_tpl / "tenet-s3-sync.timer.j2").read_text(encoding="utf-8")
    assert "OnUnitActiveSec=60s" in s3_timer

    s3_service = (guard_tpl / "tenet-s3-sync.service.j2").read_text(encoding="utf-8")
    assert "nebius_ctl.sh s3-sync-up" in s3_service

    watchdog_service = (guard_tpl / "tenet-idle-watchdog.service.j2").read_text(encoding="utf-8")
    assert "idle_watchdog.sh" in watchdog_service
    assert "NEBIUS_INSTANCE_ID=" in watchdog_service


def test_makefile_nebius_iac_targets() -> None:
    """Verify Makefile exposes nebius-up, nebius-down, and nebius-iac-validate."""
    mk_text = MAKEFILE.read_text(encoding="utf-8")
    for target in ("nebius-up:", "nebius-down:", "nebius-iac-validate:"):
        assert target in mk_text


def test_s3_sync_down_uses_local_results_dir_not_vm_path(tmp_path: Path) -> None:
    """s3-sync-down must not mkdir /mnt/nvme on macOS: it uses NEBIUS_LOCAL_RESULTS_DIR."""
    local_results = tmp_path / "results"
    env = {
        "PATH": subprocess.os.environ["PATH"],
        "HOME": str(tmp_path),
        "NEBIUS_RESULTS_DIR": "/mnt/nvme/tenet-lab/results",
        "NEBIUS_LOCAL_RESULTS_DIR": str(local_results),
        "AWS_ACCESS_KEY_ID": "",
    }
    res = subprocess.run(
        ["bash", str(NEBIUS_CTL), "s3-sync-down"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"nebius_ctl.sh s3-sync-down failed:\n{res.stderr}"
    assert local_results.is_dir(), "NEBIUS_LOCAL_RESULTS_DIR was not created"
    assert not Path("/mnt/nvme/tenet-lab/results").is_dir()


def test_makefile_passes_nebius_local_results_dir_for_s3_sync_down() -> None:
    """Makefile s3-sync-down invocations set NEBIUS_LOCAL_RESULTS_DIR to a local path."""
    mk_text = MAKEFILE.read_text(encoding="utf-8")
    assert "NEBIUS_LOCAL_RESULTS_DIR" in mk_text
    for target in ("nebius-wait-and-stop:", "nebius-s3-pull:"):
        assert target in mk_text
    assert 'NEBIUS_LOCAL_RESULTS_DIR="$(POD_LOCAL_DIR)/results"' in mk_text
    assert 'NEBIUS_LOCAL_RESULTS_DIR="$(CURDIR)/results"' in mk_text
    for line in mk_text.splitlines():
        if "nebius_ctl.sh s3-sync-down" in line:
            assert "NEBIUS_LOCAL_RESULTS_DIR=" in line


def test_github_actions_nebius_workflow() -> None:
    """Verify .github/workflows/nebius-infra.yml is valid YAML with workflow_dispatch."""
    wf_path = REPO_ROOT / ".github" / "workflows" / "nebius-infra.yml"
    assert wf_path.is_file()
    wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
    # PyYAML parses bare `on:` as boolean True unless quoted
    on_block = wf.get("on") if "on" in wf else wf.get(True)
    assert isinstance(on_block, dict) and "workflow_dispatch" in on_block
    inputs = on_block["workflow_dispatch"]["inputs"]
    assert "action" in inputs and "gpu_preset" in inputs


def test_cli_terraform_fmt_and_ansible_syntax() -> None:
    """Run terraform fmt -check and ansible-playbook --syntax-check when tools are installed."""
    if shutil.which("terraform"):
        res = subprocess.run(
            ["terraform", "fmt", "-check", "-recursive"],
            cwd=TF_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"terraform fmt failed:\n{res.stdout}\n{res.stderr}"

        if (TF_DIR / ".terraform").is_dir():
            val = subprocess.run(
                ["terraform", "validate"],
                cwd=TF_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            assert val.returncode == 0, f"terraform validate failed:\n{val.stdout}\n{val.stderr}"

    if shutil.which("ansible-playbook"):
        # Sandboxed environments may deny writes to ~/.ansible/tmp; point the
        # local temp dir at the system temp location instead.
        ans_env = dict(subprocess.os.environ)
        ans_env["ANSIBLE_LOCAL_TEMP"] = tempfile.mkdtemp(prefix="ansible-local-")
        ans = subprocess.run(
            ["ansible-playbook", "-i", "localhost,", "--syntax-check", "playbook.yml"],
            cwd=ANSIBLE_DIR,
            capture_output=True,
            text=True,
            check=False,
            env=ans_env,
        )
        assert ans.returncode == 0, f"ansible syntax-check failed:\n{ans.stdout}\n{ans.stderr}"
