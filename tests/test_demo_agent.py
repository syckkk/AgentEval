import unittest

from agents.demo_agent import DemoAgent


class DemoAgentTest(unittest.TestCase):
    def setUp(self):
        self.agent = DemoAgent()

    def test_price_query(self):
        result = self.agent.run("OpenFit 2多少钱？")
        self.assertEqual(result.trajectory, ["search_product"])
        self.assertIn("1299元", result.answer)

    def test_inventory_query(self):
        result = self.agent.run("OpenFit 2有库存吗？")
        self.assertEqual(result.trajectory, ["search_product", "check_inventory"])
        self.assertIn("有货", result.answer)

    def test_total_query(self):
        result = self.agent.run("买3个OpenFit 2多少钱？")
        self.assertEqual(result.trajectory, ["search_product", "calculate_price"])
        self.assertIn("3897元", result.answer)

    def test_greeting(self):
        result = self.agent.run("你好")
        self.assertEqual(result.trajectory, [])
        self.assertIn("你好", result.answer)

    def test_refusal(self):
        result = self.agent.run("今天上海天气怎么样？")
        self.assertEqual(result.trajectory, [])
        self.assertIn("无法回答", result.answer)

    def test_clarification(self):
        result = self.agent.run("耳机多少钱？")
        self.assertEqual(result.trajectory, [])
        self.assertIn("哪款耳机", result.answer)

    def test_unknown_product(self):
        result = self.agent.run("OpenFit 3多少钱？")
        self.assertEqual(result.trajectory, ["search_product"])
        self.assertIn("未找到", result.answer)


if __name__ == "__main__":
    unittest.main()
