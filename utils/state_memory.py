from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class StateMemory:
    """Lightweight state-memory module for grasp reasoning.

    It keeps track of:
    1. objects visible in the current scene,
    2. whether the target object is currently occluded,
    3. past grasp outcomes so future reasoning can adapt.
    """

    def __init__(self) -> None:
        self.scene_objects: List[str] = []
        self.target_occlusion: Dict[str, bool] = {}
        self.history: List[Dict[str, Any]] = []

    def update_scene(
        self,
        objects: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        occlusion: Optional[bool] = None,
    ) -> None:
        if objects is not None:
            self.scene_objects = [str(obj) for obj in objects]
        if target_name is not None and occlusion is not None:
            self.target_occlusion[target_name] = bool(occlusion)

    def record_grasp_result(
        self,
        target_name: str,
        region_hint: Optional[str],
        success: bool,
        reason: str,
        scene_id: Optional[str] = None,
    ) -> None:
        record = {
            "target_name": target_name,
            "region_hint": region_hint,
            "success": bool(success),
            "reason": reason,
            "scene_id": scene_id,
        }
        self.history.append(record)

    def get_reasoning_guidance(self, target_name: Optional[str]) -> Dict[str, Any]:
        if not target_name:
            return {
                "scene_objects": self.scene_objects,
                "occlusion_status": False,
                "next_action": "use_default_region",
                "region_hint": None,
            }

        occluded = self.target_occlusion.get(target_name, False)
        if occluded:
            next_action = "focus_on_visible_part"
            region_hint = "edge"
        else:
            next_action = "use_default_region"
            region_hint = None

        recent = [h for h in self.history if h.get("target_name") == target_name][-3:]
        if recent:
            last = recent[-1]
            if not last.get("success", False):
                next_action = "switch_region"
                region_hint = "alternate_visible_region"
            elif last.get("success", False):
                next_action = "reuse_preferred_region"
                region_hint = last.get("region_hint") or region_hint

        return {
            "scene_objects": self.scene_objects,
            "occlusion_status": occluded,
            "next_action": next_action,
            "region_hint": region_hint,
        }

    def build_memory_context(self, target_name: Optional[str], task_text: Optional[str] = None) -> str:
        guidance = self.get_reasoning_guidance(target_name)
        scene_objects = ", ".join(guidance["scene_objects"]) if guidance["scene_objects"] else "none"
        occlusion = "occluded" if guidance["occlusion_status"] else "not occluded"
        next_action = guidance["next_action"]
        region_hint = guidance["region_hint"] or "default"

        if task_text:
            return (
                "State memory context:\n"
                f"- Task: {task_text}\n"
                f"- Scene objects: {scene_objects}\n"
                f"- Target occlusion: {occlusion}\n"
                f"- Suggested next action: {next_action}\n"
                f"- Preferred region hint: {region_hint}"
            )

        return (
            "State memory context:\n"
            f"- Scene objects: {scene_objects}\n"
            f"- Target occlusion: {occlusion}\n"
            f"- Suggested next action: {next_action}\n"
            f"- Preferred region hint: {region_hint}"
        )

    def reset_episode(self) -> None:
        self.scene_objects = []
        self.target_occlusion = {}
        self.history = []


state_memory = StateMemory()
