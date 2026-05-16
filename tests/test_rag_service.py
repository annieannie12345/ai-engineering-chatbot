import unittest

from app.rag.service import AiEngineeringRagService


class RagServiceTests(unittest.TestCase):
    def test_refuses_obvious_gibberish_before_retrieval(self):
        self.assertTrue(AiEngineeringRagService._should_refuse_without_retrieval("abcdef"))

    def test_allows_known_ai_acronyms(self):
        self.assertFalse(AiEngineeringRagService._should_refuse_without_retrieval("RAG"))
        self.assertFalse(AiEngineeringRagService._should_refuse_without_retrieval("SVM"))

    def test_handles_small_talk_without_retrieval(self):
        self.assertTrue(AiEngineeringRagService._is_small_talk("hii"))
        self.assertTrue(AiEngineeringRagService._is_small_talk("how are you"))
        self.assertFalse(AiEngineeringRagService._should_refuse_without_retrieval("hii"))

    def test_refuses_personal_statements(self):
        self.assertTrue(AiEngineeringRagService._should_refuse_without_retrieval("my name is anisha"))

    def test_allows_domain_questions_and_expands_acronyms(self):
        self.assertFalse(AiEngineeringRagService._should_refuse_without_retrieval("what is rag"))
        expanded = AiEngineeringRagService._build_retrieval_query("rag")

        self.assertIn("Retrieval-Augmented Generation", expanded)
        self.assertIn("vector database", expanded)

    def test_history_is_only_used_for_follow_ups(self):
        self.assertFalse(AiEngineeringRagService._is_contextual_follow_up("rag"))
        self.assertTrue(AiEngineeringRagService._is_contextual_follow_up("tell me more about it"))


if __name__ == "__main__":
    unittest.main()
