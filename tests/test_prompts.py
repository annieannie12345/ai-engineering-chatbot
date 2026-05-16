import unittest

from app.rag.prompts import AI_ENGINEERING_SYSTEM_PROMPT


class PromptTests(unittest.TestCase):
    def test_ai_tutor_rules_are_present(self):
        prompt = AI_ENGINEERING_SYSTEM_PROMPT.lower()

        self.assertIn("ai engineering tutor", prompt)
        self.assertIn("step by step", prompt)
        self.assertIn("engineering practice", prompt)
        self.assertIn("retrieved context", prompt)
        self.assertIn("brief refusal", prompt)


if __name__ == "__main__":
    unittest.main()
