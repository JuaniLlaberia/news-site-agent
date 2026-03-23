import logging
from typing import TypedDict, List, Literal, Dict, Any

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from src.agents.reporter.sub_agents.writer.models.output import WriterOutput
from src.agents.reporter.sub_agents.concluder.models.output import ConclusionOutput, IntroductionOutput, EvaluationOutput
from src.agents.reporter.sub_agents.concluder.utils.prompts import GENERATE_INTRODUCTION_PROMPT, GENERATE_CONCLUSION_PROMPT, EVALUATE_CONCLUDER_PROMPT
from src.utils.decorators.retry import retry_with_backoff

class State(TypedDict):
    content: List[WriterOutput]

    title: str
    introduction: str
    conclusion: str

    score: float
    sections_to_improve: Literal["conclusion", "introduction"]
    improvements: List[str]

class Concluder:
    """
    Concluder agent to generate title, introduction and conclusion of the report
    """
    def __init__(self, ollama_model: str,
                 ollama_base_url: str,
                 temperature: float = 0.05,
                 top_p: float = 0.2):
        """
        Initializes concluder instance

        Args:
            ollama_model (str): Name of the ollama model
            ollama_base_url (str): Base URL for ollama
            temperature (float): Controls the randomness of the output
            top_p (float): Lowering top_p narrows the field of possible tokens.
        """
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            format="json",
            num_predict=1024,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the concluder graph with conditional validation
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("generate_conclusion", self._generate_conclusion)
        graph.add_node("generate_introduction", self._generate_introduction)
        graph.add_node("validate_results", self._validate_results)

        # Add edges
        graph.set_entry_point("generate_conclusion")
        graph.add_edge("generate_conclusion", "generate_introduction")
        graph.add_edge("generate_introduction", "validate_results")
        graph.add_conditional_edges(
            "validate_results",
            self._validate_router,
            {
                "continue_conclusion": "generate_conclusion",
                "continue_introduction": "generate_introduction",
                "end": END
            })

        return graph.compile()

    @retry_with_backoff()
    def _generate_conclusion(self, state: State) -> Dict[str, Any]:
        """
        Generate conclusion from content data

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        try:
            logging.info("Generating report conclusion")
            structured_llm = self.llm.with_structured_output(ConclusionOutput)
            chain = GENERATE_CONCLUSION_PROMPT | structured_llm

            response = chain.invoke({
                        "content": state["content"],
                        "improvements": state["improvements"]
                    })

            if isinstance(response, ConclusionOutput):
                conclusion_data = {"conclusion": response.conclusion}
            else:
                response_data = response.model_dump()
                conclusion_data = {"conclusion": response_data.get("conclusion")}

            logging.info("Successfully generated reports conclusion")
            return {**conclusion_data}

        except Exception as e:
            logging.error(f"Failed to generate report's conclusion")
            return {"conclusion": ""}

    @retry_with_backoff()
    def _generate_introduction(self, state: State) -> Dict[str, Any]:
        """
        Generate introduction from content data and conclusion

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        try:
            logging.info("Generating title & introduction for report")
            structured_llm = self.llm.with_structured_output(IntroductionOutput)
            chain = GENERATE_INTRODUCTION_PROMPT | structured_llm

            response = chain.invoke({
                        "content": state["content"],
                        "conclusion": state["conclusion"],
                        "improvements": state["improvements"]
                    })

            if isinstance(response, IntroductionOutput):
                introduction_data = {
                    "title": response.title,
                    "introduction": response.introduction,
                }
            else:
                response_data = response.model_dump()
                introduction_data = {
                    "title": response_data.get("title"),
                    "introduction": response_data.get("introduction")
                }

            logging.info("Successfully generated title & introduction for report")
            return {**introduction_data}
        except Exception as e:
            logging.error(f"Failed to generate title & introduction for report: {e}")
            return {
                "title": "",
                "introduction": ""
            }

    @retry_with_backoff()
    def _validate_results(self, state: State) -> Dict[str, Any]:
        """
        Validates the generated introduction and conclusion

        Args:
            state (State): Graph state that the node receives
        Returns:
            dict[str, any]: Dictionary to update graph state
        """
        try:
            logging.info("Generating title, introduction and conclusion evaluation")
            structured_llm = self.llm.with_structured_output(EvaluationOutput)
            chain = EVALUATE_CONCLUDER_PROMPT | structured_llm

            response = chain.invoke({
                    "title": state["title"],
                    "introduction": state["introduction"],
                    "conclusion": state["conclusion"],
                    "content": state["content"]
                })

            if isinstance(response, EvaluationOutput):
                evaluation_data = {
                    "score": response.score,
                    "sections_to_improve": response.sections_to_improve,
                    "improvements": response.improvements
                    }
            else:
                response_data = response.model_dump()
                evaluation_data = {
                    "score": response_data.get("score"),
                    "sections_to_improve": response_data.get("sections_to_improve"),
                    "improvements": response_data.get("improvements")
                    }

            logging.info("Successfully generated title, introduction and conclusion evaluation")
            return {**evaluation_data}

        except Exception as e:
            logging.error(f"Failed to evaluate title, introduction & conclusion: {e}")
            return {
                "score": 0
            }

    def _validate_router(self, state: State) -> Literal["continue_conclusion", "continue_introduction", "end"]:
        """
        Validator method to check score and sections to improve, in order to decide the next route

        Args:
            state (State): Graph state that the node receives
        Returns:
            Literal["continue_conclusion", "continue_introduction", "end"]:
                - continue_conclusion: To re-iterate the conclusion
                - continue_introduction: To re-iterate the introduction and title
                - continue_conclusion: Finish/Validation passed
        """
        score = state["score"]
        sections_to_improve = state["sections_to_improve"]
        route = None

        if score >= 0.8:
            route = "end"
        else:
            if "conclusion" == sections_to_improve:
                route = "continue_conclusion"
            else:
                route = "continue_introduction"

        return route

    def run(self, content: List[WriterOutput]):
        """
        Run concluder agent
        """
        initial_state = State(
            content=content,
            title="",
            introduction="",
            conclusion="",
            score=0.0,
            improvements=[],
            sections_to_improve=[]
        )

        logging.info("Running concluder agent...")
        result = self.graph.invoke(initial_state)

        return result["title"], result["introduction"], result["conclusion"]