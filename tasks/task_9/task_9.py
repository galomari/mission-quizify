import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from langchain_core.runnables import RouterRunnable,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import VertexAI


class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.

        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
        """
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = [] # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Context: {context}
            """
    
    def init_llm(self):
        """
        Initializes and configures the Large Language Model (LLM) for generating quiz questions.

        This method should handle any setup required to interact with the LLM, including authentication,
        setting up any necessary parameters, or selecting a specific model.

        :return: An instance or configuration for the LLM.
        """
        self.llm = VertexAI(
            model_name = "gemini-pro",
            temperature = 0.8, # Increased for less deterministic questions 
            max_output_tokens = 500
        )

    def generate_question_with_vectorstore(self):
        """
        Generates a quiz question based on the topic provided using a vectorstore

        :return: A JSON object representing the generated quiz question.
        """
        if self.llm is None:
                self.llm = VertexAI(
                model="gemini-pro",
                temperature=0.5,
                max_output_tokens=400
            )

        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized on the class")

        ############# YOUR CODE HERE ############
        # Enable a Retriever using the as_retriever() method on the VectorStore object
        # HINT: Use the vectorstore as the retriever initialized on the class
        ############# YOUR CODE HERE ############
        retriever = self.vectorstore.as_retriever()
        
        
        ############# YOUR CODE HERE ############
        # Use the system template to create a PromptTemplate
        # HINT: Use the .from_template method on the PromptTemplate class and pass in the system template
        ############# YOUR CODE HERE ############
        
        prompt = PromptTemplate.from_template(self.system_template)
        
        # RunnableParallel allows Retriever to get relevant documents
        # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        
        ############# YOUR CODE HERE ############
        # Create a chain with the Retriever, PromptTemplate, and LLM
        # HINT: chain = RETRIEVER | PROMPT | LLM 
        ############# YOUR CODE HERE ############
        chain =(  {"context": retriever, "topic": RunnablePassthrough()} | prompt | self.llm | StrOutputParser())

        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response

    def generate_quiz(self) -> list:
        """
        Task: Generate a list of unique quiz questions based on the specified topic and number of questions.

        This method orchestrates the quiz generation process by utilizing the `generate_question_with_vectorstore` method to generate each question and the `validate_question` method to ensure its uniqueness before adding it to the quiz.

        Steps:
            1. Initialize an empty list to store the unique quiz questions.
            2. Loop through the desired number of questions (`num_questions`), generating each question via `generate_question_with_vectorstore`.
            3. For each generated question, validate its uniqueness using `validate_question`.
            4. If the question is unique, add it to the quiz; if not, attempt to generate a new question (consider implementing a retry limit).
            5. Return the compiled list of unique quiz questions.

        Returns:
        - A list of dictionaries, where each dictionary represents a unique quiz question generated based on the topic.

        Note: This method relies on `generate_question_with_vectorstore` for question generation and `validate_question` for ensuring question uniqueness. Ensure `question_bank` is properly initialized and managed.
        """
        self.question_bank = []  # Reset the question bank
        for _ in range(self.num_questions):
            question_str = self.generate_question_with_vectorstore()
            try:
                question_dict = json.loads(question_str)  # Convert from JSON string to dictionary
                if self.validate_question(question_dict):
                    print("Successfully generated unique question")
                    self.question_bank.append(question_dict)
                else:
                    print("Duplicate or invalid question detected.")
            except json.JSONDecodeError:
                print("Failed to decode question JSON.")
        print("Final question bank:", self.question_bank)
        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        """
        Task: Validate a quiz question for uniqueness within the generated quiz.

        This method checks if the provided question (as a dictionary) is unique based on its text content compared to previously generated questions stored in `question_bank`. The goal is to ensure that no duplicate questions are added to the quiz.

        Steps:
            1. Extract the question text from the provided dictionary.
            2. Iterate over the existing questions in `question_bank` and compare their texts to the current question's text.
            3. If a duplicate is found, return False to indicate the question is not unique.
            4. If no duplicates are found, return True, indicating the question is unique and can be added to the quiz.

        Parameters:
        - question: A dictionary representing the generated quiz question, expected to contain at least a "question" key.

        Returns:
        - A boolean value: True if the question is unique, False otherwise.

        Note: This method assumes `question` is a valid dictionary and `question_bank` has been properly initialized.
        """
        if question is None or "question" not in question:
            return False  # Invalid or missing "question" key in the dictionary
    
        new_question_text = question["question"]  # Extract the question text from the provided dictionary
        
        if question["question"] is not None:
            for existing_question in self.question_bank:
                existing_question_text = existing_question.get("question")
                if existing_question_text == new_question_text:
                    return False  # Duplicate found, the question is not unique
            
            return True  

class QuizManager:
    ##########################################################
    def __init__(self, questions: list):
        """
        Task: Initialize the QuizManager class with a list of quiz questions.

        Overview:
        This task involves setting up the `QuizManager` class by initializing it with a list of quiz question objects. Each quiz question object is a dictionary that includes the question text, multiple choice options, the correct answer, and an explanation. The initialization process should prepare the class for managing these quiz questions, including tracking the total number of questions.

        Instructions:
        1. Store the provided list of quiz question objects in an instance variable named `questions`.
        2. Calculate and store the total number of questions in the list in an instance variable named `total_questions`.

        Parameters:
        - questions: A list of dictionaries, where each dictionary represents a quiz question along with its choices, correct answer, and an explanation.

        Note: This initialization method is crucial for setting the foundation of the `QuizManager` class, enabling it to manage the quiz questions effectively. The class will rely on this setup to perform operations such as retrieving specific questions by index and navigating through the quiz.
        """
        ##### YOUR CODE HERE #####
        self.questions = questions
        self.total_questions = len(questions)
        
    ##########################################################

    def get_question_at_index(self, index: int):
        """
    Retrieves the quiz question object at the specified index. If the index is out of bounds,
    it restarts from the beginning index.

    :param index: The index of the question to retrieve.
    :return: The quiz question object at the specified index, with indexing wrapping around if out of bounds.
    """
        if self.total_questions == 0:
            # Safeguard against division by zero if the questions list is empty
            return None

        # Calculate a valid index using modulo to ensure it wraps around if out of bounds
        else:
            valid_index = index % self.total_questions
            return self.questions[valid_index]
    
    def next_question_index(self, direction=1):
        """
        Task: Adjust the current quiz question index based on the specified direction.

        Overview:
        Develop a method to navigate to the next or previous quiz question by adjusting the `question_index` in Streamlit's session state. This method should account for wrapping, meaning if advancing past the last question or moving before the first question, it should continue from the opposite end.

        Instructions:
        1. Retrieve the current question index from Streamlit's session state.
        2. Adjust the index based on the provided `direction` (1 for next, -1 for previous), using modulo arithmetic to wrap around the total number of questions.
        3. Update the `question_index` in Streamlit's session state with the new, valid index.
            # st.session_state["question_index"] = new_index

        Parameters:
        - direction: An integer indicating the direction to move in the quiz questions list (1 for next, -1 for previous).

        Note: Ensure that `st.session_state["question_index"]` is initialized before calling this method. This navigation method enhances the user experience by providing fluid access to quiz questions.
        """
        ##### YOUR CODE HERE #####
        if 'question_index' not in st.session_state:
                st.session_state['question_index'] = 0
        new_index = (st.session_state['question_index'] + direction) % self.total_questions
        st.session_state['question_index'] = new_index
       
    ##########################################################

# Test Generating the Quiz
if __name__ == "__main__":
    # Configuration for embedding client and document processor
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-quizifytask-1-421919",
        "location": "us-central1"
    }
    
    # Clear previous session states
    if 'question_index' not in st.session_state:
        st.session_state['question_index'] = 0

    screen = st.empty()
    question_bank = None
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
        embed_client = EmbeddingClient(**embed_config)
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                generator = QuizGenerator(topic_input, num_questions, chroma_creator)
                question_bank = generator.generate_quiz()

    if question_bank:
                quiz_manager = QuizManager(question_bank)
                current_question = quiz_manager.get_question_at_index(st.session_state.get('question_index', 0))
                if current_question:
                    # Adjust the following line to correctly unpack choices from a list of dictionaries
                    choices = [f"{choice['key']}) {choice['value']}" for choice in current_question['choices']]
                    question_text = current_question['question']
                    st.write(question_text)
                    selected_answer = st.radio("Choose the correct answer:", choices)
                    if st.button("Submit Answer"):
                        correct_answer_key = current_question['answer']
                        # Ensure the selected answer checks against the format used in choices
                        if selected_answer.startswith(correct_answer_key + ")"):  # Assumes the format 'A) Answer'
                            st.success("Correct!")
                        else:
                            st.error("Incorrect!")
                        quiz_manager.next_question_index()
                else:
                    st.error("Failed to load the question. Please try again.")
    else:
        st.error("No questions generated. Please check the generation process.")