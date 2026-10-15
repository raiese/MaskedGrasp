import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.state_memory import StateMemory


class StateMemoryTest(unittest.TestCase):
    def test_update_scene_and_get_guidance(self):
        memory = StateMemory()
        memory.update_scene(["cup", "box"], target_name="cup", occlusion=True)

        guidance = memory.get_reasoning_guidance("cup")

        self.assertEqual(guidance["scene_objects"], ["cup", "box"])
        self.assertTrue(guidance["occlusion_status"])
        self.assertEqual(guidance["next_action"], "focus_on_visible_part")

    def test_record_failure_switches_region(self):
        memory = StateMemory()
        memory.update_scene(["cup"], target_name="cup", occlusion=False)
        memory.record_grasp_result(
            target_name="cup",
            region_hint="handle",
            success=False,
            reason="slipped",
            scene_id="scene1",
        )

        guidance = memory.get_reasoning_guidance("cup")

        self.assertEqual(guidance["next_action"], "switch_region")
        self.assertEqual(guidance["region_hint"], "alternate_visible_region")


if __name__ == "__main__":
    unittest.main()
