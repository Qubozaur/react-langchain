from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import os
from typing import List, Dict
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class AdvancedDocumentSummarizer:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0, api_key=None):
        """
        Initialize the summarizer.
        
        Args:
            model_name: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Creativity level (0-1)
            api_key: OpenAI API key (if not set in environment)
        """
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_document(self, file_path: str):
        """
        Load document based on file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            from langchain_classic.schema import Document
            return [Document(page_content=content)]
        else:
            raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT")
        
        documents = loader.load()
        return documents
    
    def summarize_stuff(self, file_path: str, custom_prompt: str = None) -> str:
        """
        Simple summarization for shorter documents (stuff method).
        Best for documents under 4000 tokens.
        
        Args:
            file_path: Path to document
            custom_prompt: Optional custom prompt template
            
        Returns:
            Summary text
        """
        documents = self.load_document(file_path)
        
        if custom_prompt:
            prompt = PromptTemplate(template=custom_prompt, input_variables=["text"])
        else:
            prompt_template = """Write a concise and comprehensive summary of the following document.
                                Focus on the main ideas, key points, and important conclusions.
                                Document:
                                {text}
                                SUMMARY:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        chain = load_summarize_chain(
            self.llm,
            chain_type="stuff",
            prompt=prompt
        )
        
        summary = chain.run(documents)
        return summary
    
    def summarize_map_reduce(self, file_path: str) -> str:
        """
        Better for longer documents - summarizes chunks then combines them.
        Good balance between speed and quality.
        
        Args:
            file_path: Path to document
            
        Returns:
            Summary text
        """
        documents = self.load_document(file_path)
        docs = self.text_splitter.split_documents(documents)
        map_prompt_template = """Write a detailed summary of this section:
                                {text}
                                SECTION SUMMARY:"""
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        
        combine_prompt_template = """Combine the following section summaries into a comprehensive, coherent summary.
                                     Ensure all key points are captured and the summary flows naturally.
                                     Section Summaries:
                                     {text}
                                     FINAL COMPREHENSIVE SUMMARY:"""
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )
        
        summary = chain.run(docs)
        return summary
    
    def summarize_refine(self, file_path: str) -> str:
        """
        Iteratively refines summary - best quality but slower.
        Builds summary incrementally, refining with each chunk.
        
        Args:
            file_path: Path to document
            
        Returns:
            Summary text
        """
        documents = self.load_document(file_path)
        docs = self.text_splitter.split_documents(documents)
        
        refine_template = """Your task is to produce a final summary.
                             We have provided an existing summary up to a certain point: {existing_answer}
                             We have the opportunity to refine the existing summary (only if needed) with more context below.
                             {text}
                             Given the new context, refine the original summary. If the context isn't useful, return the original summary.
                             REFINED SUMMARY:"""
        
        refine_prompt = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )
        
        initial_template = """Write a summary of the following:
                              {text}
                              SUMMARY:"""
        
        initial_prompt = PromptTemplate(
            input_variables=["text"],
            template=initial_template,
        )
        
        chain = load_summarize_chain(
            self.llm,
            chain_type="refine",
            question_prompt=initial_prompt,
            refine_prompt=refine_prompt,
            verbose=False
        )
        
        summary = chain.run(docs)
        return summary
    
    def summarize_with_structure(self, file_path: str) -> Dict[str, str]:
        """
        Creates structured summary with multiple sections.
        
        Args:
            file_path: Path to document
            
        Returns:
            Dictionary with structured summary components
        """
        documents = self.load_document(file_path)
        docs = self.text_splitter.split_documents(documents)
        
        full_text = "\n\n".join([doc.page_content for doc in docs[:10]]) 
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Analyze the following document and provide a structured summary:
                        Document:
                        {text}
                        Please provide:
                        1. EXECUTIVE SUMMARY (2-3 sentences):
                        [Your executive summary here]
                        2. KEY POINTS (bullet points):
                        [Your key points here]
                        3. MAIN TOPICS COVERED:
                        [Your main topics here]
                        4. CONCLUSIONS AND RECOMMENDATIONS:
                        [Your conclusions here]
                        5. IMPORTANT DETAILS OR STATISTICS:
                        [Any important numbers, dates, or facts]
                        STRUCTURED SUMMARY:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        summary = chain.run(text=full_text)
        
        return {
            "structured_summary": summary,
            "document_name": os.path.basename(file_path),
            "timestamp": datetime.now().isoformat()
        }
    
    def extract_key_information(self, file_path: str, questions: List[str]) -> str:
        """
        Extract specific information based on questions.
        
        Args:
            file_path: Path to document
            questions: List of questions to answer
            
        Returns:
            Answers to the questions
        """
        documents = self.load_document(file_path)
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        if len(full_text) > 12000:
            full_text = full_text[:12000] + "...[truncated]"
        
        prompt = PromptTemplate(
            input_variables=["text", "questions"],
            template="""Based on the following document, answer these questions as thoroughly as possible.
                        If information for a question is not found in the document, state "Information not found in document."
                        Questions:
                        {questions}
                        Document:
                        {text}
                        ANSWERS:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        answers = chain.run(text=full_text, questions=questions_text)
        
        return answers
    
    def compare_documents(self, file_paths: List[str]) -> str:
        """
        Compare multiple documents and highlight similarities/differences.
        
        Args:
            file_paths: List of document paths to compare
            
        Returns:
            Comparison summary
        """
        if len(file_paths) < 2:
            raise ValueError("Need at least 2 documents to compare")
        
        summaries = []
        for i, path in enumerate(file_paths):
            doc = self.load_document(path)
            content = "\n".join([d.page_content for d in doc])[:3000]  # Limit size
            summaries.append(f"Document {i+1} ({os.path.basename(path)}):\n{content}")
        
        combined_text = "\n\n---\n\n".join(summaries)
        
        prompt = PromptTemplate(
            input_variables=["documents"],
            template="""Compare the following documents and provide:
                        1. Common themes and topics across all documents
                        2. Key differences between documents
                        3. Unique points in each document
                        4. Overall comparison summary
                        Documents to compare:
                        {documents}
                        COMPARISON ANALYSIS:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        comparison = chain.run(documents=combined_text)
        
        return comparison
    
    def summarize_by_topic(self, file_path: str, topics: List[str]) -> Dict[str, str]:
        """
        Summarize document focusing on specific topics.
        
        Args:
            file_path: Path to document
            topics: List of topics to focus on
            
        Returns:
            Dictionary with summaries for each topic
        """
        documents = self.load_document(file_path)
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        if len(full_text) > 12000:
            full_text = full_text[:12000] + "...[truncated]"
        
        results = {}
        
        for topic in topics:
            prompt = PromptTemplate(
                input_variables=["text", "topic"],
                template="""From the following document, extract and summarize all information related to: {topic}
                            If the topic is not discussed in the document, state "This topic is not covered in the document."
                            Document:
                            {text}
                            SUMMARY FOR TOPIC "{topic}":"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            topic_summary = chain.run(text=full_text, topic=topic)
            results[topic] = topic_summary
        
        return results
    
    def save_summary(self, summary: str, output_path: str, format: str = "txt"):
        """
        Save summary to file.
        
        Args:
            summary: Summary text to save
            output_path: Output file path
            format: Output format (txt, json, md)
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        if format == "json":
            data = {
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "model": self.llm.model_name
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == "md":
            md_content = f"""# Document Summary
                             **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                             **Model:** {self.llm.model_name}
                            ---
                            {summary}
                            """
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        else:  
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
        
        print(f"Summary saved to: {output_path}")


def main():
    summarizer = AdvancedDocumentSummarizer(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    

if __name__ == "__main__":
    main()