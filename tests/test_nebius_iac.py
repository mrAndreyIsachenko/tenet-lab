"""Tests for Nebius AI Cloud Terraform & Ansible IaC integration."""

from __future__ import annotations

import json
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


def test_reconfigure_dry_run_plans_expected_fields() -> None:
    """DRY-RUN reconfigure prints the plan (preset, preemptibility, nvme, stop state)."""
    env = {
        "PATH": subprocess.os.environ["PATH"],
        "HOME": subprocess.os.environ["HOME"],
        "DRY_RUN": "1",
        "NEBIUS_INSTANCE_ID": "inst-dry-1",
        "NEBIUS_SSH_HOST": "",
    }
    res = subprocess.run(
        [
            "bash",
            str(NEBIUS_CTL),
            "reconfigure",
            "--preset",
            "wave2",
            "--preemptible",
            "--nvme-gib",
            "1860",
            "--running",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"reconfigure dry-run failed:\n{res.stderr}"
    assert (
        "DRY-RUN reconfigure: instance=inst-dry-1 platform=gpu-h200-sxm"
        " preset=8gpu-128vcpu-1600gb preemptible=1 nvme_gib=1860 stopped=false"
    ) in res.stdout


def test_reconfigure_rejects_unknown_preset(tmp_path: Path) -> None:
    """--preset must be wave1 or wave2; anything else fails before any API call."""
    env = {
        "PATH": subprocess.os.environ["PATH"],
        "HOME": str(tmp_path),
        "DRY_RUN": "1",
        "NEBIUS_INSTANCE_ID": "inst-x",
    }
    res = subprocess.run(
        ["bash", str(NEBIUS_CTL), "reconfigure", "--preset", "wave3"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "wave1 or wave2" in res.stderr


def _write_nebius_shim(bin_dir: Path) -> None:
    """Fake `nebius` CLI: logs every invocation, serves canned JSON, no network."""
    shim = bin_dir / "nebius"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "python3 -c 'import json, os, sys; "
        'open(os.environ["NEBIUS_SHIM_LOG"], "a").write(json.dumps(sys.argv[1:]) + "\\n")\' '
        '"$@"\n'
        'case "$1 $2 $3" in\n'
        '  "profile list") exit 0 ;;\n'
        '  "compute instance get") cat "$NEBIUS_SHIM_FIXTURES/instance.json" ;;\n'
        '  "compute disk get") cat "$NEBIUS_SHIM_FIXTURES/disk.json" ;;\n'
        '  "compute instance create") echo \'{"metadata": {"id": "inst-new-999"}}\' ;;\n'
        '  "compute instance get-by-name")'
        ' echo \'{"metadata": {"id": "inst-new-999"}}\' ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)


def _reconfigure_live_env(tmp_path: Path) -> dict:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    instance = {
        "metadata": {"id": "inst-old-123", "parent_id": "project-abc"},
        "spec": {
            "service_account_id": "sa-xyz",
            "resources": {"platform": "gpu-h200-sxm", "preset": "1gpu-16vcpu-200gb"},
            "boot_disk": {"attach_mode": "READ_WRITE", "existing_disk": {"id": "disk-boot-1"}},
            "secondary_disks": [
                {
                    "attach_mode": "READ_WRITE",
                    "device_id": "nvme-scratch",
                    "existing_disk": {"id": "disk-nvme-1"},
                }
            ],
            "network_interfaces": [
                {
                    "name": "eth0",
                    "subnet_id": "subnet-1",
                    "ip_address": {},
                    "public_ip_address": {"allocation_id": "alloc-1", "static": False},
                }
            ],
            "cloud_init_user_data": "#cloud-config\n",
        },
        "status": {"state": "STOPPED"},
    }
    disk = {"spec": {"size_gibibytes": 1023}}
    (fixtures / "instance.json").write_text(json.dumps(instance), encoding="utf-8")
    (fixtures / "disk.json").write_text(json.dumps(disk), encoding="utf-8")

    env_file = tmp_path / "env.nebius"
    env_file.write_text(
        "NEBIUS_INSTANCE_ID=inst-old-123\nNEBIUS_SSH_HOST=1.2.3.4\n",
        encoding="utf-8",
    )
    hosts_file = tmp_path / "hosts.yml"
    hosts_file.write_text(
        "all:\n"
        "  children:\n"
        "    nebius-gpu:\n"
        "      hosts:\n"
        "        gpu-vm:\n"
        "          ansible_host: 1.2.3.4\n"
        "          nebius_instance_id: inst-old-123\n",
        encoding="utf-8",
    )

    env = {
        "PATH": f"{tmp_path / 'bin'}:{subprocess.os.environ['PATH']}",
        "HOME": str(tmp_path),
        "NEBIUS_INSTANCE_ID": "inst-old-123",
        "NEBIUS_SHIM_LOG": str(tmp_path / "shim.log"),
        "NEBIUS_SHIM_FIXTURES": str(fixtures),
        "NEBIUS_ENV_FILE": str(env_file),
        "NEBIUS_ANSIBLE_HOSTS_FILE": str(hosts_file),
    }
    return env


def test_reconfigure_live_switches_to_wave2_preemptible(tmp_path: Path) -> None:
    """Live reconfigure: resize NVMe, recreate shell, update env + hosts files."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_nebius_shim(bin_dir)

    env = _reconfigure_live_env(tmp_path)
    log_path = Path(env["NEBIUS_SHIM_LOG"])
    env_file = Path(env["NEBIUS_ENV_FILE"])
    hosts_file = Path(env["NEBIUS_ANSIBLE_HOSTS_FILE"])

    res = subprocess.run(
        [
            "bash",
            str(NEBIUS_CTL),
            "reconfigure",
            "--preset",
            "wave2",
            "--preemptible",
            "--nvme-gib",
            "1860",
            "--running",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"live reconfigure failed:\n{res.stdout}\n{res.stderr}"

    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    flat = "\n".join(" ".join(c) for c in calls)

    assert "compute disk update --id disk-nvme-1 --size-gibibytes 1860" in flat
    assert "compute instance delete --id inst-old-123" in flat
    assert "compute instance create" in flat
    # Instance started STOPPED -> no stop call; want_stopped=false handled at create time.
    assert "compute instance stop" not in flat

    create_call = next(c for c in calls if c[1:3] == ["instance", "create"])
    payload = json.loads(next(a for a in create_call if a.lstrip().startswith("{")))
    spec = payload["spec"]
    assert spec["preemptible"] == {"on_preemption": "STOP"}
    assert spec["recovery_policy"] == "FAIL"
    assert spec["resources"] == {"platform": "gpu-h200-sxm", "preset": "8gpu-128vcpu-1600gb"}
    assert spec["boot_disk"]["existing_disk"]["id"] == "disk-boot-1"
    assert spec["secondary_disks"][0]["existing_disk"]["id"] == "disk-nvme-1"
    assert payload["metadata"]["parent_id"] == "project-abc"

    env_text = env_file.read_text(encoding="utf-8")
    assert "NEBIUS_INSTANCE_ID=inst-new-999" in env_text
    assert "NEBIUS_PROJECT_ID=project-abc" in env_text
    assert "NEBIUS_SSH_HOST=1.2.3.4" in env_text  # other lines preserved

    hosts_data = yaml.safe_load(hosts_file.read_text(encoding="utf-8"))
    gpu_vars = hosts_data["all"]["children"]["nebius-gpu"]["hosts"]["gpu-vm"]
    assert gpu_vars["nebius_instance_id"] == "inst-new-999"
    assert gpu_vars["nebius_project_id"] == "project-abc"
    assert gpu_vars["ansible_host"] == "1.2.3.4"


def test_reconfigure_live_skips_resize_and_recreation_when_unchanged(tmp_path: Path) -> None:
    """Same preset/preemptible + equal nvme target: no delete/create, start only."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_nebius_shim(bin_dir)

    env = _reconfigure_live_env(tmp_path)
    log_path = Path(env["NEBIUS_SHIM_LOG"])

    res = subprocess.run(
        [
            "bash",
            str(NEBIUS_CTL),
            "reconfigure",
            "--preset",
            "wave1",
            "--on-demand",
            "--nvme-gib",
            "1023",
            "--running",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"live reconfigure failed:\n{res.stdout}\n{res.stderr}"

    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    flat = "\n".join(" ".join(c) for c in calls)
    assert "instance delete" not in flat
    assert "instance create" not in flat
    assert "disk update" not in flat
    assert "compute instance start --id inst-old-123" in flat


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
