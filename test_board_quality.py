import unittest

from board_check import load_engine_ns, simulate_lesson
from board_quality import diagram_family, improve_step_board


def _template_steps(question, n=4):
    return [
        improve_step_board(question, i, question, [])
        for i in range(n)
    ]


class BoardQualityTests(unittest.TestCase):
    def test_rocket_orbit_curve_rises_then_flattens(self):
        board = improve_step_board(
            "How do rockets actually reach orbit?", 0,
            "A rocket climbs, turns, and builds sideways orbital speed.", []
        )
        curve = next(op for op in board if op["op"] == "curve")
        ys = [p[1] for p in curve["points"]]
        self.assertLess(ys[-1], ys[0])
        self.assertLessEqual(max(abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 2, len(ys) - 1)), 3)
        self.assertTrue(all(ys[i + 1] <= ys[i] for i in range(len(ys) - 1)))

    def test_gradient_descent_graph_is_a_true_loss_valley(self):
        board = improve_step_board(
            "How does gradient descent work?", 0,
            "We follow the loss curve down to the minimum.", []
        )
        graph = next(op for op in board if op["op"] == "graph")
        pts = graph["series"][0]["points"]
        min_y_idx = min(range(len(pts)), key=lambda i: pts[i][1])
        self.assertEqual(pts[min_y_idx], [0, 0])
        self.assertGreater(pts[0][1], pts[min_y_idx][1])
        self.assertGreater(pts[-1][1], pts[min_y_idx][1])
        self.assertEqual(graph["y_range"], [0, 9])

    def test_gradient_markers_stay_in_data_coordinates_after_layout(self):
        ns = load_engine_ns()
        question = "How does gradient descent work?"
        violations, ops = simulate_lesson(ns, _template_steps(question))
        self.assertEqual(violations, [])
        graph = next(op for op in ops if op.get("op") == "graph")
        start = next(m for m in graph["markers"] if m.get("label") == "start")
        minimum = next(m for m in graph["markers"] if m.get("label") == "minimum")
        self.assertEqual(start["at"], [-2.5, 6.25])
        self.assertEqual(minimum["at"], [0, 0])

    def test_science_templates_keep_visual_teaching_cues(self):
        rocket_ops = [op for step in _template_steps("How do rockets actually reach orbit?")
                      for op in step]
        rocket_text = " ".join(str(op.get(k, "")) for op in rocket_ops
                               for k in ("text", "label", "title"))
        self.assertIn("thick air", rocket_text)
        self.assertIn("gravity pulls", rocket_text)
        self.assertIn("build sideways speed", rocket_text)

        gradient_ops = [op for step in _template_steps("How does gradient descent work?")
                        for op in step]
        gradient_text = " ".join(str(op.get(k, "")) for op in gradient_ops
                                 for k in ("text", "label", "title"))
        graph_markers = sum(len(op.get("markers") or []) for op in gradient_ops
                            if op.get("op") == "graph")
        self.assertGreaterEqual(graph_markers + sum(
            1 for op in gradient_ops if op.get("op") == "dot"), 4)
        self.assertIn("Update rule", gradient_text)
        self.assertIn("rate * slope", gradient_text)

    def test_pythagorean_template_uses_compact_visual_callouts(self):
        ns = load_engine_ns()
        question = "Teach me the Pythagorean theorem with a 3 4 5 triangle"
        violations, ops = simulate_lesson(ns, _template_steps(question))
        self.assertEqual(violations, [])
        parts = []
        for op in ops:
            parts.extend(str(op.get(k, "")) for k in ("text", "label", "title"))
            parts.extend(str(line) for line in op.get("lines", []) or [])
        text = " ".join(parts)
        self.assertGreaterEqual(len([op for op in ops if op.get("op") == "callout"]), 2)
        self.assertIn("hypotenuse", text)
        self.assertIn("a = 3", text)
        self.assertIn("9 + 16 = 25", text)
        self.assertIn("c = 5", text)
        tri = next(op for op in ops if op.get("op") == "polygon")
        xs = [p[0] for p in tri["points"]]
        self.assertLessEqual(max(xs) - min(xs), 40)

    def test_fable_mythos_access_template_matches_directive(self):
        self.assertEqual(
            diagram_family("Anthropic Fable 5 Mythos 5 government ban access"),
            "model_access_directive",
        )
        ns = load_engine_ns()
        question = "What happened to Anthropic Fable 5 and Mythos 5 access?"
        violations, ops = simulate_lesson(ns, _template_steps(question))
        self.assertEqual(violations, [])
        parts = []
        for op in ops:
            parts.extend(str(op.get(k, "")) for k in ("text", "label", "title"))
            parts.extend(str(line) for line in op.get("lines", []) or [])
        text = " ".join(parts)
        self.assertIn("US directive", text)
        self.assertIn("Access off", text)
        self.assertIn("Jun 11, 2026", text)
        self.assertIn("Anthropic disputes scope", text)

    def test_common_topics_are_classified(self):
        self.assertEqual(diagram_family("Why is the sky blue?"), "sky_blue")
        self.assertEqual(diagram_family("Teach me the Pythagorean theorem"), "pythagorean")
        self.assertEqual(diagram_family("How do neural networks learn?"), "neural_network")
        self.assertEqual(diagram_family("Explain supply and demand"), "supply_demand")
        self.assertEqual(diagram_family("How does Unqork work?"), "no_code_platform")
        self.assertEqual(diagram_family("What happened in space exploration this month?"), None)

    def test_sky_blue_template_is_clean_and_scientifically_specific(self):
        ns = load_engine_ns()
        question = "Why is the sky blue?"
        violations, ops = simulate_lesson(ns, _template_steps(question))
        self.assertEqual(violations, [])
        graph = next(op for op in ops if op.get("op") == "graph")
        points = graph["series"][0]["points"]
        self.assertGreater(points[0][1], points[-1][1])
        text = " ".join(
            [str(op.get(key, "")) for op in ops for key in ("text", "label", "title")]
            + [str(line) for op in ops for line in (op.get("lines") or [])]
        ).lower()
        self.assertIn("blue to you", text)
        self.assertIn("red travels on", text)
        self.assertIn("blue waves scatter most", text)

    def test_checkin_steps_stay_empty_when_model_drew_nothing(self):
        board = improve_step_board("Explain caching", 3, "Does that make sense so far?", [])
        self.assertEqual(board, [])

    def test_templates_survive_layout_without_overlap(self):
        ns = load_engine_ns()
        cases = [
            "How do rockets actually reach orbit?",
            "How does gradient descent work?",
            "How does Unqork work?",
            "Teach me the Pythagorean theorem",
            "How do neural networks learn?",
            "Explain supply and demand",
            "How does binary search work?",
            "Explain recursion with a call tree",
            "How does photosynthesis work?",
            "How do rainbows form?",
            "Explain the water cycle",
        ]
        for question in cases:
            with self.subTest(question=question):
                violations, _ = simulate_lesson(ns, _template_steps(question))
                self.assertEqual(violations, [])

    def test_generic_layout_compacts_shapes_and_drops_orphan_arrows(self):
        ns = load_engine_ns()
        steps = [[
            {"op": "title", "text": "Large Generic Board"},
            {"op": "box", "at": [12, 25], "w": 62, "h": 18,
             "label": "Status: Publicly Available", "color": "blue"},
            {"op": "arrow", "from": [65, 61], "to": [65, 72],
             "color": "red"},
            {"op": "text", "text": "Usage limits", "at": [83, 66],
             "size": "m", "color": "red"},
        ]]
        violations, ops = simulate_lesson(ns, steps)
        self.assertEqual(violations, [])
        box = next(op for op in ops if op.get("op") == "box")
        self.assertLessEqual(box["w"], 34)
        self.assertFalse(any(op.get("op") == "arrow" for op in ops))
        txt = next(op for op in ops if op.get("op") == "text")
        self.assertEqual(txt.get("size"), "m")

    def test_long_arrows_are_capped(self):
        ns = load_engine_ns()
        steps = [[
            {"op": "title", "text": "Compact Connectors"},
            {"op": "box", "at": [10, 30], "w": 18, "h": 8,
             "label": "A", "color": "blue"},
            {"op": "box", "at": [78, 30], "w": 18, "h": 8,
             "label": "B", "color": "green"},
            {"op": "arrow", "from": [28, 34], "to": [78, 34],
             "label": "then", "color": "ink"},
        ]]
        violations, ops = simulate_lesson(ns, steps)
        self.assertEqual(violations, [])
        arrow = next(op for op in ops if op.get("op") == "arrow")
        dist = ((arrow["to"][0] - arrow["from"][0]) ** 2 +
                (arrow["to"][1] - arrow["from"][1]) ** 2) ** 0.5
        self.assertLessEqual(dist, 28.1)

    def test_template_arrows_do_not_collapse_after_layout(self):
        ns = load_engine_ns()
        for question in [
            "How do rockets actually reach orbit?",
            "How does gradient descent work?",
            "How do neural networks learn?",
            "Explain recursion with a call tree",
        ]:
            with self.subTest(question=question):
                _, ops = simulate_lesson(ns, _template_steps(question))
                arrows = [op for op in ops if op.get("op") == "arrow"]
                self.assertTrue(arrows)
                for op in arrows:
                    fx, fy = op["from"]
                    tx, ty = op["to"]
                    self.assertGreater(((tx - fx) ** 2 + (ty - fy) ** 2) ** 0.5, 4)


if __name__ == "__main__":
    unittest.main()
