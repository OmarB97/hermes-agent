"""Regression tests for profile-scoped skills_tool path resolution."""

import importlib
import json
from pathlib import Path


def _write_skill(root: Path, category: str, name: str, description: str) -> Path:
    skill_dir = root / "skills" / category / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"Loaded from {description}.\n",
        encoding="utf-8",
    )
    return skill_dir


def _reload_skills_tool(import_home: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(import_home))
    import tools.skills_tool as skills_tool

    return importlib.reload(skills_tool)


def test_skill_view_uses_live_profile_home_after_module_import(tmp_path, monkeypatch):
    """skill_view should not stay pinned to HERMES_HOME from import time."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    _write_skill(default_home, "autonomous-ai-agents", "default-only", "default home")
    profile_skill_dir = _write_skill(
        profile_home,
        "software-development",
        "kanban-orchestrator-operations",
        "orchestrator profile",
    )

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    assert skills_tool.SKILLS_DIR == default_home / "skills"

    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    result = json.loads(
        skills_tool.skill_view("kanban-orchestrator-operations", preprocess=False)
    )

    assert result["success"] is True
    assert result["name"] == "kanban-orchestrator-operations"
    assert Path(result["skill_dir"]) == profile_skill_dir
    assert "orchestrator profile" in result["content"]


def test_skills_list_uses_live_profile_home_after_module_import(tmp_path, monkeypatch):
    """skills_list should list the active profile skills, not the import-time root."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    _write_skill(default_home, "autonomous-ai-agents", "default-only", "default home")
    _write_skill(
        profile_home,
        "software-development",
        "kanban-orchestrator-operations",
        "orchestrator profile",
    )

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    result = json.loads(skills_tool.skills_list())
    names = {skill["name"] for skill in result["skills"]}

    assert result["success"] is True
    assert "kanban-orchestrator-operations" in names
    assert "default-only" not in names


def test_explicit_skills_dir_monkeypatch_still_wins(tmp_path, monkeypatch):
    """Existing tests can still override tools.skills_tool.SKILLS_DIR directly."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    patched_root = tmp_path / "patched"
    patched_skill_dir = _write_skill(
        patched_root,
        "software-development",
        "patched-skill",
        "patched skills dir",
    )
    _write_skill(
        profile_home,
        "software-development",
        "profile-skill",
        "orchestrator profile",
    )

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", patched_root / "skills")

    result = json.loads(skills_tool.skill_view("patched-skill", preprocess=False))

    assert result["success"] is True
    assert Path(result["skill_dir"]) == patched_skill_dir


# ── slash-command discovery (agent/skill_commands.py) ────────────────────
#
# scan_skill_commands() is the /<skill-name> vocabulary the gateway's
# command.dispatch resolves against. It ran the same scan as skills_tool but
# off the frozen module-level SKILLS_DIR, so in a long-lived backend serving
# several profiles a session saw its OWN skills mid-turn (the agent resolves
# via _skills_dir()) while /<skill-name> saw the launch profile's.


def _reload_skill_commands(import_home: Path, monkeypatch):
    """Reload both modules so skills_tool's frozen SKILLS_DIR is import_home."""
    _reload_skills_tool(import_home, monkeypatch)
    import agent.skill_commands as skill_commands

    return importlib.reload(skill_commands)


def test_scan_skill_commands_uses_live_profile_home_after_module_import(
    tmp_path, monkeypatch
):
    """/<skill-name> discovery follows the active profile, not the import-time root."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    _write_skill(default_home, "autonomous-ai-agents", "launch-only-skill", "default home")
    _write_skill(
        profile_home, "software-development", "profile-only-skill", "orchestrator profile"
    )

    skill_commands = _reload_skill_commands(default_home, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    cmds = skill_commands.scan_skill_commands()

    assert "/profile-only-skill" in cmds
    assert "/launch-only-skill" not in cmds


def test_scan_skill_commands_resolves_the_profile_skill_dir(tmp_path, monkeypatch):
    """The discovered entry points at the profile's copy, not the launch home's."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    # Same skill name in both homes — only the path distinguishes them.
    _write_skill(default_home, "software-development", "shared-skill", "default home")
    profile_skill_dir = _write_skill(
        profile_home, "software-development", "shared-skill", "orchestrator profile"
    )

    skill_commands = _reload_skill_commands(default_home, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    cmds = skill_commands.scan_skill_commands()

    assert "/shared-skill" in cmds
    assert Path(cmds["/shared-skill"]["skill_dir"]) == profile_skill_dir


def test_scan_skill_commands_honors_an_explicit_skills_dir_monkeypatch(
    tmp_path, monkeypatch
):
    """The SKILLS_DIR escape hatch still wins here too, as it does in skills_tool."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    patched_root = tmp_path / "patched"
    _write_skill(patched_root, "software-development", "patched-only-skill", "patched dir")
    _write_skill(profile_home, "software-development", "profile-only-skill", "profile")

    skill_commands = _reload_skill_commands(default_home, monkeypatch)
    import tools.skills_tool as skills_tool

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", patched_root / "skills")

    cmds = skill_commands.scan_skill_commands()

    assert "/patched-only-skill" in cmds
    assert "/profile-only-skill" not in cmds
